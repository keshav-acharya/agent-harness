from __future__ import annotations

import time

import anthropic

from ..tools.base import Tool
from ..types import Message, ModelResponse, ToolCall, Usage

_STOP_REASONS = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens"}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, client: anthropic.Anthropic | None = None):
        self.model = model
        # The SDK retries 429 / 5xx / connection errors with exponential backoff.
        self.client = client if client is not None else anthropic.Anthropic(max_retries=5)

    @staticmethod
    def _to_wire(messages: list[Message]) -> list[dict]:
        wire: list[dict] = []
        for m in messages:
            blocks: list[dict] = []
            if m.role == "user":
                for r in m.tool_results:  # tool_result blocks must come FIRST
                    blocks.append({
                        "type": "tool_result",
                        "tool_use_id": r.call_id,
                        "content": r.content,
                        "is_error": r.is_error,
                    })
                if m.text:
                    blocks.append({"type": "text", "text": m.text})
            else:
                if m.text:
                    blocks.append({"type": "text", "text": m.text})
                for c in m.tool_calls:
                    blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.input})
            if not blocks:  # the API rejects empty messages and empty text blocks
                blocks.append({"type": "text", "text": "(empty)"})
            wire.append({"role": m.role, "content": blocks})
        return wire

    def complete(
        self, *, system: str, messages: list[Message], tools: list[Tool], max_tokens: int
    ) -> ModelResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": self._to_wire(messages),
        }
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]

        t0 = time.perf_counter()
        resp = self.client.messages.create(**kwargs)
        latency = time.perf_counter() - t0

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
            # other block types (e.g. thinking) are ignored for now

        u = resp.usage
        usage = Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        stop = _STOP_REASONS.get(resp.stop_reason, "other")
        return ModelResponse("".join(text_parts), calls, usage, stop, latency)