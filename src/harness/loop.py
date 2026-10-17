from __future__ import annotations

import json
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
from .types import Message, ModelResponse, ToolCall, ToolResult, Usage

# Error text that means "the model produced a bad generation", not "your request is wrong".
RECOVERABLE_MARKERS = ("tool_use_failed", "failed_generation", "Failed to call a function")
REPEAT_FAILURE_WARN_AT = 3


def _is_recoverable(e: Exception) -> bool:
    text = f"{type(e).__name__}: {e}"
    return any(marker in text for marker in RECOVERABLE_MARKERS)


@dataclass
class Limits:
    max_turns: int = 40
    max_cost_usd: float = 2.0        # enforced only if the model has a price in pricing.py
    max_wall_s: float = 900.0
    max_output_tokens: int = 8192    # per model call
    max_provider_retries: int = 2    # retries per turn for flaky-generation errors


@dataclass
class RunResult:
    status: str  # completed | truncated | max_turns | max_cost | timeout | error
    final_text: str
    turns: int
    usage: Usage
    cost_usd: float | None
    error: str | None = None
    wall_s: float = 0.0
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

    def _call_model(
        self, messages: list[Message], turn: int
    ) -> tuple[ModelResponse | None, str | None]:
        """Returns (response, None) on success or (None, error_text) on failure."""
        attempt = 0
        while True:
            try:
                resp = self.provider.complete(
                    system=self.system,
                    messages=messages,
                    tools=self.registry.tools,
                    max_tokens=self.limits.max_output_tokens,
                )
                return resp, None
            except Exception as e:  # noqa: BLE001 - any API failure must end the run cleanly
                err = f"{type(e).__name__}: {e}"
                if _is_recoverable(e) and attempt < self.limits.max_provider_retries:
                    attempt += 1
                    self._emit("provider_retry", turn=turn, attempt=attempt, error=err)
                    continue
                return None, err

    @staticmethod
    def _guard_repeats(call: ToolCall, result: ToolResult, streaks: dict[str, int]) -> None:
        """After N failures of the exact same call, tell the model to stop repeating it."""
        sig = f"{call.name}:{json.dumps(call.input, sort_keys=True, default=str)}"
        if not result.is_error:
            streaks.pop(sig, None)  # it worked, so forget the failures
            return
        streaks[sig] = streaks.get(sig, 0) + 1
        if streaks[sig] >= REPEAT_FAILURE_WARN_AT:
            result.content += (
                f"\n[harness note: this identical call has now failed {streaks[sig]} times. "
                "Do not repeat it; try a different approach.]"
            )

    def run(self, task: str) -> RunResult:
        lim = self.limits
        messages: list[Message] = [Message("user", text=task)]
        usage = Usage()
        cost: float | None = None
        streaks: dict[str, int] = {}
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

            resp, err = self._call_model(messages, turns + 1)
            if resp is None:
                status, error = "error", err
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
                # Cut off mid-reply; any tool call in it may be incomplete. Stop honestly.
                final, status = resp.text, "truncated"
                break

            if not resp.tool_calls:
                final, status = resp.text, "completed"
                break

            results = []
            for call in resp.tool_calls:  # EVERY call must be answered
                t0 = time.perf_counter()
                result = self.registry.execute(call, self.ctx)
                self._guard_repeats(call, result, streaks)
                self._emit(
                    "tool_result", turn=turns, call_id=call.id, name=call.name, input=call.input,
                    content=result.content, is_error=result.is_error,
                    duration_s=time.perf_counter() - t0,
                )
                results.append(result)
            messages.append(Message("user", tool_results=results))

        wall = time.monotonic() - start
        self._emit(
            "run_end", status=status, turns=turns, usage=asdict(usage), cost_usd=cost,
            error=error, wall_s=wall,
        )
        return RunResult(
            status=status, final_text=final, turns=turns, usage=usage, cost_usd=cost,
            error=error, wall_s=wall, messages=messages,
        )