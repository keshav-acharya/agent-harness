from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]
    parse_error: str | None = None  # set if the model's arguments weren't valid JSON


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """Provider-neutral message. Each adapter converts to/from its wire format."""
    role: Literal["user", "assistant"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)      # assistant only
    tool_results: list[ToolResult] = field(default_factory=list)  # user only


@dataclass
class Usage:
    """input_tokens means UNCACHED input for every provider (normalised in the adapters)."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "other"]
    latency_s: float = 0.0