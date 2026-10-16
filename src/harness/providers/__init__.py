from __future__ import annotations

import os

from .base import Provider

PROVIDER_NAMES = ("anthropic", "openai", "groq")


def make_provider(name: str, model: str | None = None) -> Provider:
    """Build a provider by name. The model falls back to <NAME>_MODEL from the environment."""
    if name not in PROVIDER_NAMES:
        raise ValueError(f"Unknown provider '{name}'. Choose from: {', '.join(PROVIDER_NAMES)}")
    model = model or os.environ.get(f"{name.upper()}_MODEL")
    if not model:
        raise ValueError(f"No model given. Pass one or set {name.upper()}_MODEL in .env")

    # Imported lazily so that using one vendor never requires importing the others.
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(model)
    if name == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(model)
    from .groq_provider import GroqProvider
    return GroqProvider(model)