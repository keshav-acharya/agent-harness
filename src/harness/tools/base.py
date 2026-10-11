from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ToolError(Exception):
    """Expected failure. The message is shown to the model so it can recover."""


@dataclass
class ToolContext:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    def resolve(self, p: str) -> Path:
        path = (self.root / p).resolve()
        if not path.is_relative_to(self.root):
            raise ToolError(f"Path '{p}' is outside the working directory.")
        return path


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    run: Callable[[dict[str, Any], ToolContext], str]
    read_only: bool = False


def truncate(text: str, max_chars: int = 20_000) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    omitted = len(text) - max_chars
    return f"{text[:head]}\n... [{omitted} characters truncated] ...\n{text[-tail:]}"