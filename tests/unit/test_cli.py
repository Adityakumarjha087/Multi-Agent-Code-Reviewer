"""The command line prints a readable review and reports bad input without a traceback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from prreview import cli
from prreview.llm.client import MissingCredentialsError


def outcome(drafts: list, notes: list, summary: str = "") -> SimpleNamespace:
    state = SimpleNamespace(
        review_id="r1", drafts=drafts, steps=[1, 2, 3], total_cost_usd=0.0123, notes=notes, summary=summary,
    )
    return SimpleNamespace(state=state, latency_s=4.5)


def test_render_lists_header_summary_comments_and_notes() -> None:
    draft = SimpleNamespace(severity="high", file="a.py", line=9, body="Possible None dereference.")
    note = SimpleNamespace(agent="security", kind="timeout", message="scanner timed out")
    text = cli.render(outcome([draft], [note], summary="One real issue."))
    assert text.splitlines()[0] == "Review r1: 1 draft comment(s), 3 steps, $0.0123, 4.5s"
    assert "Summary: One real issue." in text
    assert "[HIGH] a.py:9\nPossible None dereference." in text
    assert "Notes:\n- security/timeout: scanner timed out" in text


def test_render_omits_empty_sections() -> None:
    text = cli.render(outcome([], []))
    assert "Summary:" not in text and "Notes:" not in text


def test_bad_url_exits_with_a_readable_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["review", "https://example.com/nope"]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error: not a GitHub pull request URL")
    assert "Traceback" not in captured.err


def test_missing_credentials_are_reported_not_raised(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    async def no_key(ref, enable_sandbox):
        raise MissingCredentialsError("the API key is not set")

    monkeypatch.setattr("prreview.review.factory.review_pr", no_key)
    assert cli.main(["review", "https://github.com/octo/demo/pull/7"]) == 1
    assert "error: the API key is not set" in capsys.readouterr().err
