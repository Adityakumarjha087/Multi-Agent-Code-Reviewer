"""Test doubles. NOT used by production code.

`ScriptedLLM` replays pre-written Claude responses (real `BetaMessage` objects from the SDK) so
the agent loop, graph, and benchmark harness can be tested without an API key or API spend.
Passing tests here prove our control flow, not model quality.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

from anthropic.types.beta import BetaMessage

from prreview.llm.client import LLMCall, LLMResult, measure

_ids = itertools.count(1)


def text_message(payload: dict[str, Any] | str, model: str = "claude-opus-5-5", stop_reason: str = "end_turn",
                 input_tokens: int = 1000, output_tokens: int = 200) -> BetaMessage:
    """A final assistant reply carrying JSON text (as structured output produces)."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return _message([{"type": "text", "text": text}], model, stop_reason, input_tokens, output_tokens)


def tool_use_message(calls: list[tuple[str, dict[str, Any]]], model: str = "claude-opus-5-5") -> BetaMessage:
    """An assistant turn that requests one or more tool calls."""
    blocks = [{"type": "tool_use", "id": f"toolu_{next(_ids)}", "name": n, "input": a} for n, a in calls]
    return _message(blocks, model, "tool_use", 800, 50)


def refusal_message(category: str = "cyber") -> BetaMessage:
    """A declined request (HTTP 200, stop_reason=refusal)."""
    details = {"type": "refusal", "category": category, "explanation": None}
    return _message([], "claude-opus-5-5", "refusal", 500, 0, stop_details=details)


def _message(content: list[dict[str, Any]], model: str, stop_reason: str, input_tokens: int, output_tokens: int,
             **extra: Any) -> BetaMessage:
    return BetaMessage.model_validate({**extra,
        "id": f"msg_{next(_ids)}", "type": "message", "role": "assistant", "model": model,
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    })


class ScriptedLLM:
    """Returns scripted responses in order, routed per system-prompt keyword when given a dict."""

    def __init__(self, script: list[BetaMessage] | dict[str, list[BetaMessage]]) -> None:
        self._script = script
        self.calls: list[LLMCall] = []

    async def create(self, call: LLMCall) -> LLMResult:
        self.calls.append(call)
        queue = self._queue_for(call)
        if not queue:
            raise AssertionError(f"ScriptedLLM ran out of responses (system prompt starts {call.system[:60]!r})")
        return measure(queue.pop(0), call.model, latency_ms=5)

    def _queue_for(self, call: LLMCall) -> list[BetaMessage]:
        if isinstance(self._script, list):
            return self._script
        for keyword, queue in self._script.items():
            if keyword in call.system:
                return queue
        raise AssertionError(f"no scripted queue matches system prompt {call.system[:80]!r}")


class MinimalLLM:
    """Answers every request with the smallest valid output for its schema (no tool calls).

    Used to exercise live wiring (real GitHub, MCP subprocesses, Docker) without an API key.
    """

    _MINIMAL = {
        "PlannerOutput": {"priorities": [], "summary": "(test double: no model was called)"},
        "FindingsReport": {"findings": [], "notes": ""},
        "MergeGroups": {"duplicate_groups": [], "overall_summary": ""},
    }

    def __init__(self) -> None:
        self.calls: list[LLMCall] = []

    async def create(self, call: LLMCall) -> LLMResult:
        self.calls.append(call)
        return measure(text_message(self._MINIMAL[call.output_model.__name__]), call.model, latency_ms=1)


class OracleLLM:
    """Reports exactly the labelled issues of benchmark cases (a perfect reviewer).

    Exists only to check that the benchmark harness and scoring give 100% for perfect output.
    It says nothing about real model quality.
    """

    def __init__(self, cases: list) -> None:
        self._cases = cases
        self.calls: list[LLMCall] = []

    async def create(self, call: LLMCall) -> LLMResult:
        from prreview.github.diff import new_side_lines

        self.calls.append(call)
        name = call.output_model.__name__
        if name != "FindingsReport":
            return measure(text_message(MinimalLLM._MINIMAL[name]), call.model, latency_ms=1)
        prompt = call.messages[0]["content"]
        case = next(c for c in self._cases if f"/{c.id} " in prompt or f"bench/{c.id}" in prompt)
        reviewer = "security" if "application security" in call.system else "bug"
        findings = []
        for label in case.labels:
            if "prompt_injection_attempt" in label.categories or label.file not in prompt:
                continue
            is_security = bool({"injection", "hardcoded_secret", "insecure_config", "unsafe_dependency"} & set(label.categories))
            if is_security != (reviewer == "security"):
                continue
            patch = next(f for f in case.changed_files() if f.path == label.file).patch
            findings.append({
                "category": label.categories[0], "severity": "high", "confidence": 0.9, "file": label.file,
                "line_start": label.lines[0], "line_end": label.lines[0], "title": label.note[:60],
                "explanation": label.note, "evidence": new_side_lines(patch)[label.lines[0]].strip(), "suggested_fix": "",
            })
        return measure(text_message({"findings": findings, "notes": ""}), call.model, latency_ms=1)
