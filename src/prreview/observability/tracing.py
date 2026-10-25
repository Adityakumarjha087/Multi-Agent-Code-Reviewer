"""Tracing of every agent step, tool call, token count, and cost.

Langfuse 4.17 API used (verified from the installed source):
- `Langfuse(public_key=, secret_key=, base_url=)`
- `client.start_as_current_observation(name=, as_type="span"|"agent"|"tool"|"generation", input=,
  metadata=, model=)` -> context manager yielding an observation with `.update(output=, usage_details=,
  cost_details=, level=, status_message=)`
- `langfuse.propagate_attributes(trace_name=, session_id=, tags=)` for trace-level attributes
- `client.flush()`

Without LANGFUSE_PUBLIC_KEY/SECRET_KEY the `NoopTracer` is used, so nothing is sent anywhere.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal, Protocol

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

SpanType = Literal["span", "agent", "tool", "generation"]


class Span(Protocol):
    """A live observation."""

    def update(self, **fields: Any) -> None: ...


class Tracer(Protocol):
    """What the rest of the code depends on, so Langfuse stays optional."""

    def span(self, name: str, as_type: SpanType = "span", **fields: Any) -> Any: ...

    def review(self, review_id: str, name: str, tags: list[str]) -> Any: ...

    def flush(self) -> None: ...


class LangfuseSettings(BaseSettings):
    """Langfuse credentials. Cloud by default (decision Q6)."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LANGFUSE_", extra="ignore")

    public_key: str | None = None
    secret_key: SecretStr | None = None
    base_url: str = "https://cloud.langfuse.com"


class _NoopSpan:
    def update(self, **fields: Any) -> None:
        return None


class NoopTracer:
    """Used when Langfuse is not configured. Records nothing."""

    @contextmanager
    def span(self, name: str, as_type: SpanType = "span", **fields: Any) -> Iterator[_NoopSpan]:
        yield _NoopSpan()

    @contextmanager
    def review(self, review_id: str, name: str, tags: list[str]) -> Iterator[_NoopSpan]:
        yield _NoopSpan()

    def flush(self) -> None:
        return None


class LangfuseTracer:
    """Sends observations to Langfuse."""

    def __init__(self, settings: LangfuseSettings, span_exporter: Any = None) -> None:
        """`span_exporter` (an OpenTelemetry exporter) is for tests: spans are captured instead of sent."""
        from langfuse import Langfuse  # imported lazily: only needed when configured

        self._client = Langfuse(
            public_key=settings.public_key,
            secret_key=settings.secret_key.get_secret_value() if settings.secret_key else None,
            base_url=settings.base_url,
            span_exporter=span_exporter,
        )

    @contextmanager
    def span(self, name: str, as_type: SpanType = "span", **fields: Any) -> Iterator[Any]:
        with self._client.start_as_current_observation(name=name, as_type=as_type, **fields) as observation:
            yield observation

    @contextmanager
    def review(self, review_id: str, name: str, tags: list[str]) -> Iterator[Any]:
        from langfuse import propagate_attributes

        with propagate_attributes(trace_name=name, session_id=review_id, tags=tags):
            with self._client.start_as_current_observation(name=name, as_type="agent") as observation:
                yield observation

    def flush(self) -> None:
        self._client.flush()


def build_tracer(settings: LangfuseSettings | None = None) -> Tracer:
    """Langfuse if both keys are set, otherwise a no-op tracer."""
    settings = settings or LangfuseSettings()
    if settings.public_key and settings.secret_key:
        return LangfuseTracer(settings)
    logger.info("Langfuse keys not set: tracing disabled")
    return NoopTracer()
