from __future__ import annotations

import json
import time
from typing import Any

from ..tools.base import Tool
from ..types import Message, ModelResponse, ToolCall, Usage

_FINISH_REASONS = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}


class ChatCompletionsProvider:
    """Shared logic for APIs that speak the OpenAI 'chat completions' format.

    Subclasses (OpenAI, Groq) only supply a name and a client.
    """

    name = "chat-completions"

    def __init__(self, model: str, client: Any):
        self.model = model
        self.client = client

    @staticmethod
    def _to_wire(system: str, messages: list[Message]) -> list[dict]:
        wire: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "user":
                # One separate "tool" message per result, in the same order as the calls.
                for r in m.tool_results:
                    content = f"ERROR: {r.content}" if r.is_error else r.content
                    wire.append({"role": "tool", "tool_call_id": r.call_id, "content": content})
                if m.text:
                    wire.append({"role": "user", "content": m.text})
            else:
                msg: dict = {"role": "assistant", "content": m.text or None}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.input)},
                        }
                        for c in m.tool_calls
                    ]
                wire.append(msg)
        return wire

    @staticmethod
    def _parse_tool_call(tc: Any) -> ToolCall:
        raw = tc.function.arguments or "{}"
        try:
            args = json.loads(raw)
            error = None if isinstance(args, dict) else "arguments must be a JSON object"
            if error:
                args = {}
        except json.JSONDecodeError as e:
            args, error = {}, f"{e.msg} (line {e.lineno}, column {e.colno})"
        return ToolCall(tc.id, tc.function.name, args, parse_error=error)

    def complete(
        self, *, system: str, messages: list[Message], tools: list[Tool], max_tokens: int
    ) -> ModelResponse:
        kwargs: dict = {
            "model": self.model,
            "messages": self._to_wire(system, messages),
            "max_completion_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0

        choice = resp.choices[0]
        calls = [self._parse_tool_call(tc) for tc in (choice.message.tool_calls or [])]

        u = resp.usage
        details = getattr(u, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        usage = Usage(
            input_tokens=u.prompt_tokens - cached,  # normalise: OpenAI counts cached inside prompt_tokens
            output_tokens=u.completion_tokens,
            cache_read_tokens=cached,
        )
        stop = _FINISH_REASONS.get(choice.finish_reason, "other")
        return ModelResponse(choice.message.content or "", calls, usage, stop, latency)