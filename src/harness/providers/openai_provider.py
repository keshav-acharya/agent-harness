from __future__ import annotations

import openai

from .chat_completions import ChatCompletionsProvider


class OpenAIProvider(ChatCompletionsProvider):
    name = "openai"

    def __init__(self, model: str, client: openai.OpenAI | None = None):
        super().__init__(model, client if client is not None else openai.OpenAI(max_retries=5))