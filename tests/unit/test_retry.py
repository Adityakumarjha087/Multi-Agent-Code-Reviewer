"""Tests for the retry policy."""

import httpx
import pytest

from prreview.reliability.retry import RetryableError, is_retryable, retrying


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.ConnectError("refused"), True),
        (httpx.ReadTimeout("slow"), True),
        (RetryableError("503"), True),
        (ValueError("bad schema"), False),
        (PermissionError("401"), False),
    ],
)
def test_classifies_transient_errors(exc: Exception, expected: bool) -> None:
    assert is_retryable(exc) is expected


async def test_stops_after_max_attempts_and_reraises_original() -> None:
    calls = 0
    with pytest.raises(RetryableError, match="still down"):
        async for attempt in retrying(max_attempts=3, initial_wait_s=0.001):
            with attempt:
                calls += 1
                raise RetryableError("still down")
    assert calls == 3


async def test_non_retryable_error_is_raised_immediately() -> None:
    calls = 0
    with pytest.raises(ValueError):
        async for attempt in retrying(max_attempts=5, initial_wait_s=0.001):
            with attempt:
                calls += 1
                raise ValueError("schema mismatch")
    assert calls == 1
