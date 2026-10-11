from __future__ import annotations

from dataclasses import dataclass

from .types import Usage


@dataclass(frozen=True)
class Price:
    """USD per 1,000,000 tokens."""
    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


# Fill these in from each provider's pricing page. Unknown models return None (cost unknown).
PRICES: dict[str, Price] = {
    # "claude-sonnet-5-5": Price(input=..., output=..., cache_read=..., cache_write=...),
    # "your-groq-model":   Price(input=..., output=...),
    # "your-openai-model": Price(input=..., output=..., cache_read=...),
}


def cost_usd(model: str, u: Usage) -> float | None:
    p = PRICES.get(model)
    if p is None:
        return None
    return (
        u.input_tokens * p.input
        + u.output_tokens * p.output
        + u.cache_read_tokens * p.cache_read
        + u.cache_write_tokens * p.cache_write
    ) / 1_000_000