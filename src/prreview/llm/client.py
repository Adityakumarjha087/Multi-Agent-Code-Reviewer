"""Claude API access for the agents.

API shape verified against anthropic==1.12.1 source and the bundled Claude API reference:
- `client.beta.messages.create(..., betas=[...], fallbacks="default")` - server-side refusal
  fallback; the scalar "default" form needs the `server-side-fallback-2026-07-01` beta.
- `output_config={"effort": ..., "format": {"type": "json_schema", "schema": ...}}` - structured
  output. `anthropic.transform_schema` turns a Pydantic model into an accepted schema; it moves
  unsupported constraints (min/max) into descriptions, so results are re-validated with Pydantic.
- Forced `tool_choice` ("any"/"tool") returns 400 on claude-opus-5-5; only "auto"/"none" are used.
- Thinking is adaptive and cannot be disabled on claude-opus-5-5; depth is set via `effort`.
- The SDK retries 408/409/429/5xx and connection errors itself (`max_retries`).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import anthropic
from anthropic.types.beta import BetaMessage
from pydantic import BaseModel

from prreview.config import LLMSettings
from prreview.llm.pricing import cost_usd

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class LLMCall:
    """Everything needed for one Messages API request."""

    model: str
    system: str
    messages: list[dict[str, Any]]
    output_model: type[BaseModel]
    tools: list[dict[str, Any]]
    tool_choice: Literal["auto", "none"] = "auto"


@dataclass(frozen=True)
class LLMResult:
    """The raw response plus measured cost and latency."""

    message: BetaMessage
    cost_usd: float
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int


class LLMClient(Protocol):
    """What the agent loop needs from an LLM. Tests provide a scripted fake."""

    async def create(self, call: LLMCall) -> LLMResult: ...


class MissingCredentialsError(RuntimeError):
    """No Anthropic credentials. Checked up front: without them the SDK raises a plain TypeError
    at request time (observed with anthropic 1.12.1), which would abort a review halfway through."""


class AnthropicLLM:
    """Production client backed by the Anthropic SDK."""

    def __init__(self, settings: LLMSettings, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._settings = settings
        self._client = client or _build_client(settings)

    async def create(self, call: LLMCall) -> LLMResult:
        """Send one request and return the response with its cost."""
        started = time.monotonic()
        message = await self._client.beta.messages.create(**self._request_kwargs(call))
        latency_ms = int((time.monotonic() - started) * 1000)
        return measure(message, call.model, latency_ms)

    def _request_kwargs(self, call: LLMCall) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": call.model,
            "max_tokens": self._settings.max_tokens,
            "system": call.system,
            "messages": call.messages,
            "output_config": {
                "effort": self._settings.effort,
                "format": {"type": "json_schema", "schema": anthropic.transform_schema(call.output_model)},
            },
        }
        if call.tools:
            kwargs["tools"] = call.tools
            kwargs["tool_choice"] = {"type": call.tool_choice}
        if self._settings.use_refusal_fallbacks:
            kwargs["fallbacks"] = "default"
            kwargs["betas"] = [FALLBACK_BETA]
        return kwargs


def _build_client(settings: LLMSettings) -> anthropic.AsyncAnthropic:
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    if key is None and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        raise MissingCredentialsError("ANTHROPIC_API_KEY is not set (add it to .env or the environment)")
    return anthropic.AsyncAnthropic(api_key=key, timeout=settings.request_timeout_s, max_retries=settings.sdk_max_retries)


def measure(message: BetaMessage, requested_model: str, latency_ms: int) -> LLMResult:
    """Compute tokens and cost. With fallbacks, `usage.iterations` is the per-attempt source of truth.

    Declined attempts are counted too, so with a refusal the reported cost may overstate the bill.
    """
    usage = message.usage
    entries = [i for i in (usage.iterations or []) if i.type in ("message", "fallback_message")]
    if not entries:
        entries = [usage]
    totals = {"input": 0, "output": 0, "cache": 0}
    cost = 0.0
    for entry in entries:
        model = getattr(entry, "model", None) or message.model or requested_model
        cache_read = entry.cache_read_input_tokens or 0
        cost += cost_usd(model, entry.input_tokens, entry.output_tokens, cache_read)
        totals["input"] += entry.input_tokens
        totals["output"] += entry.output_tokens
        totals["cache"] += cache_read
    return LLMResult(message, round(cost, 6), latency_ms, totals["input"], totals["output"], totals["cache"])
