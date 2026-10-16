from __future__ import annotations

from groq import Groq

from .chat_completions import ChatCompletionsProvider


class GroqProvider(ChatCompletionsProvider):
    name = "groq"

    def __init__(self, model: str, client: Groq | None = None):
        super().__init__(model, client if client is not None else Groq(max_retries=5))