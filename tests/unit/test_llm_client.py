"""Tests for request construction and cost measurement (no network)."""

import pytest
from anthropic.types.beta import BetaMessage

from prreview.config import LLMSettings
from prreview.llm.client import FALLBACK_BETA, AnthropicLLM, LLMCall, measure
from prreview.llm.pricing import UnknownModelPriceError, cost_usd
from prreview.schemas.review import FindingsReport


def _kwargs(**settings: object) -> dict:
    llm = AnthropicLLM(LLMSettings(_env_file=None, **settings), client=object())  # client unused by _request_kwargs
    call = LLMCall("claude-opus-5-5", "sys", [{"role": "user", "content": "hi"}], FindingsReport,
                   [{"name": "t", "description": "", "input_schema": {"type": "object"}}], "none")
    return llm._request_kwargs(call)


def test_request_uses_structured_output_effort_and_fallbacks() -> None:
    kwargs = _kwargs()
    assert kwargs["output_config"]["effort"] == "high"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert "findings" in kwargs["output_config"]["format"]["schema"]["properties"]
    assert kwargs["fallbacks"] == "default" and kwargs["betas"] == [FALLBACK_BETA]
    assert kwargs["tool_choice"] == {"type": "none"}
    assert "thinking" not in kwargs and "temperature" not in kwargs


def test_fallbacks_can_be_disabled() -> None:
    kwargs = _kwargs(use_refusal_fallbacks=False)
    assert "fallbacks" not in kwargs and "betas" not in kwargs


def _message(usage: dict, model: str = "claude-opus-5-5") -> BetaMessage:
    return BetaMessage.model_validate({
        "id": "m", "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": "end_turn", "stop_sequence": None, "usage": usage,
    })


def test_cost_from_top_level_usage() -> None:
    result = measure(_message({"input_tokens": 1_000_000, "output_tokens": 100_000, "cache_read_input_tokens": 0}), "claude-opus-5-5", 1)
    assert result.cost_usd == pytest.approx(4.0 + 2.0)


def test_cost_sums_fallback_iterations_at_each_models_price() -> None:
    usage = {
        "input_tokens": 1_000_000, "output_tokens": 0, "cache_read_input_tokens": 0,
        "iterations": [
            {"type": "message", "model": "claude-opus-5-5", "input_tokens": 1_000_000, "output_tokens": 0,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 1_000_000, "output_tokens": 0,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        ],
    }
    assert measure(_message(usage, model="claude-opus-4-8"), "claude-opus-5-5", 1).cost_usd == pytest.approx(4.0 + 5.0)


def test_unknown_model_price_is_an_error_not_zero() -> None:
    with pytest.raises(UnknownModelPriceError):
        cost_usd("claude-made-up", 10, 10)


def test_missing_credentials_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    from prreview.llm.client import MissingCredentialsError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(MissingCredentialsError):
        AnthropicLLM(LLMSettings(_env_file=None))


def test_api_key_from_settings_is_passed_to_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-123")
    llm = AnthropicLLM(LLMSettings(_env_file=None))
    assert llm._client.api_key == "sk-ant-test-123"


def test_single_error_unwraps_nested_groups_but_keeps_multiple() -> None:
    from prreview.review.factory import single_error

    inner = ValueError("real cause")
    nested = ExceptionGroup("outer", [ExceptionGroup("mid", [ExceptionGroup("inner", [inner])])])
    assert single_error(nested) is inner
    two = ExceptionGroup("g", [ValueError("a"), KeyError("b")])
    assert single_error(two) is two
