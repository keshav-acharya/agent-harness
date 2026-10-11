from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..types import Message, ModelResponse

if TYPE_CHECKING:  # only for type checkers; avoids needing tools/base.py before Day 2
    from ..tools.base import Tool


class Provider(Protocol):
    """Anything with these attributes and this method counts as a provider."""

    name: str
    model: str

    def complete(
        self, *, system: str, messages: list[Message], tools: list[Tool], max_tokens: int
    ) -> ModelResponse: ...