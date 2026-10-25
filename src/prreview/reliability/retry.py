"""Retry policy shared by every outbound call (GitHub API, later the LLM API).

Only transient failures are retried. Auth errors, 404s and validation errors fail
fast because retrying them only adds latency and cost.
"""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class RetryableError(Exception):
    """Raised by clients to mark a failure as transient (e.g. HTTP 503)."""


def is_retryable(exc: BaseException) -> bool:
    """Transport errors (connect/read timeouts, resets) and explicitly-marked errors are transient."""
    return isinstance(exc, (httpx.TransportError, RetryableError))


def retrying(max_attempts: int, initial_wait_s: float = 0.5, max_wait_s: float = 8.0) -> AsyncRetrying:
    """Build an async retry loop: exponential backoff with jitter, re-raising the last error.

    Wait before retry n is `initial * 2**(n-1) + uniform(0, initial)`, capped at `max_wait_s`.
    (tenacity's default jitter is a fixed 0-1 s, so it is scaled here to the base wait.)
    """
    return AsyncRetrying(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(multiplier=initial_wait_s, jitter=initial_wait_s, max=max_wait_s),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
