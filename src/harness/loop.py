from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .pricing import cost_usd
from .prompts import build_system_prompt
from .providers.base import Provider
from .tools import ToolRegistry
from .tools.base import ToolContext
from .trace import Tracer
from .types import Message, Usage


@dataclass
class Limits:
    max_turns: int = 40
    max_cost_usd: float = 2.0       # enforced only if the model has a price in pricing.py
    max_wall_s: float = 900.0
    max_output_tokens: int = 8192   # per model call


@dataclass
class RunResult:
    status: str  # completed | truncated | max_turns | max_cost | timeout | error
    final_text: str
    turns: int
    usage: Usage
    cost_usd: float | None
    error: str | None = None
    messages: list[Message] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        root: Path,
        limits: Limits | None = None,
        tracer: Tracer | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        self.provider = provider
        self.registry = registry
        self.ctx = ToolContext(root)
        self.limits = limits or Limits()
        self.tracer = tracer or Tracer(None)
        self.on_event = on_event
        self.system = build_system_prompt(self.ctx.root)

    def _emit(self, event: str, **data) -> None:
        self.tracer.emit(event, **data)
        if self.on_event:
            self.on_event(event, data)

    def run(self, task: str) -> RunResult:
        lim = self.limits
        messages: list[Message] = [Message("user", text=task)]
        usage = Usage()
        cost: float | None = None
        start = time.monotonic()
        status, final, error, turns = "max_turns", "", None, 0

        self._emit(
            "run_start", task=task, provider=self.provider.name, model=self.provider.model,
            limits=asdict(lim), system=self.system,
        )

        while turns < lim.max_turns:
            if time.monotonic() - start > lim.max_wall_s:
                status = "timeout"
                break
            if cost is not None and cost > lim.max_cost_usd:
                status = "max_cost"
                break

            try:
                resp = self.provider.complete(
                    system=self.system, messages=messages,
                    tools=self.registry.tools, max_tokens=lim.max_output_tokens,
                )
            except Exception as e:  # noqa: BLE001 - network/API failure must end the run cleanly
                status, error = "error", f"{type(e).__name__}: {e}"
                break

            turns += 1
            usage.add(resp.usage)
            cost = cost_usd(self.provider.model, usage)
            messages.append(Message("assistant", text=resp.text, tool_calls=resp.tool_calls))
            self._emit(
                "model_response", turn=turns, text=resp.text,
                tool_calls=[asdict(c) for c in resp.tool_calls], usage=asdict(resp.usage),
                stop_reason=resp.stop_reason, latency_s=resp.latency_s, cost_so_far=cost,
            )

            if resp.stop_reason == "max_tokens":
                # The reply was cut off; any tool call in it may be incomplete. Stop honestly.
                final, status = resp.text, "truncated"
                break

            if not resp.tool_calls:
                final, status = resp.text, "completed"
                break

            results = []
            for call in resp.tool_calls:  # EVERY call must be answered
                t0 = time.perf_counter()
                result = self.registry.execute(call, self.ctx)
                self._emit(
                    "tool_result", turn=turns, call_id=call.id, name=call.name, input=call.input,
                    content=result.content, is_error=result.is_error,
                    duration_s=time.perf_counter() - t0,
                )
                results.append(result)
            messages.append(Message("user", tool_results=results))

        self._emit(
            "run_end", status=status, turns=turns, usage=asdict(usage), cost_usd=cost,
            error=error, wall_s=time.monotonic() - start,
        )
        return RunResult(status, final, turns, usage, cost, error, messages)