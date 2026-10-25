"""Per-model token prices used to compute review cost.

Source: Anthropic first-party API prices from the bundled Claude API reference, cached
2026-10-06 (USD per million tokens). Prices change: re-check https://platform.claude.com/docs
before quoting cost numbers. Unknown models raise instead of silently costing $0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float | None = None


PRICES: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(input=4.00, output=20.00, cache_read=0.20),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00, cache_read=0.20),
    # Haiku 5.5 price applies to prompts up to 100K tokens ($0.50/$2.50 beyond); our prompts stay below that.
    "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50),
    # Possible refusal-fallback targets (D: fallbacks="default").
    "claude-opus-5": ModelPrice(input=5.00, output=25.00),
    "claude-opus-4-8": ModelPrice(input=5.00, output=25.00),
}


class UnknownModelPriceError(KeyError):
    """No price is configured for a model, so its cost cannot be reported honestly."""


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0) -> float:
    """Return the USD cost of one API attempt."""
    price = PRICES.get(model)
    if price is None:
        raise UnknownModelPriceError(f"no price configured for model {model!r}; add it to prreview.llm.pricing")
    cache_rate = price.cache_read if price.cache_read is not None else price.input
    total = input_tokens * price.input + output_tokens * price.output + cache_read_tokens * cache_rate
    return round(total / 1_000_000, 6)
