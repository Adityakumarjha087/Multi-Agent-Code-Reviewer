"""The Streamlit page shows helpful text instead of blank areas (stub repository, no database)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from prreview.persistence.repo import Repository

APP = Path(__file__).resolve().parents[2] / "ui" / "app.py"


class StubRepo:
    """Just the read methods the page calls."""

    def __init__(self, reviews: list, comments: list, events: list) -> None:
        self._reviews, self._comments, self._events = reviews, comments, events

    def recent_reviews(self) -> list:
        return self._reviews

    def review(self, review_id: str):
        return next((r for r in self._reviews if r.id == review_id), None)

    def comments(self, review_id: str) -> list:
        return self._comments

    def events(self, review_id: str) -> list:
        return self._events

    def enqueue(self, ref) -> str:  # only reached with a URL that parsed
        raise AssertionError("a rejected URL must not be queued")


def completed(review_id: str = "r1") -> SimpleNamespace:
    return SimpleNamespace(
        id=review_id, owner="octo", repo="demo", number=7, status="completed", summary="", steps=3,
        cost_usd=0.01, latency_s=1.5, error="", state={"notes": []},
    )


def run_app(monkeypatch: pytest.MonkeyPatch, stub: StubRepo, review: str | None = None) -> AppTest:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setattr(Repository, "from_url", classmethod(lambda cls, url: stub))
    st.cache_resource.clear()
    app = AppTest.from_file(APP, default_timeout=20)
    if review:
        app.query_params["review"] = review
    return app.run()


def test_empty_sidebar_explains_how_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    app = run_app(monkeypatch, StubRepo([], [], []))
    assert not app.exception
    assert any("No reviews yet" in c.value for c in app.sidebar.caption)


def test_completed_review_without_comments_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    app = run_app(monkeypatch, StubRepo([completed()], [], []), review="r1")
    assert not app.exception
    assert any("No draft comments" in i.value for i in app.info)
    captions = [c.value for c in app.caption]
    assert any("No notes" in c for c in captions)
    assert any("No agent events" in c for c in captions)


def comment(comment_id: str, status: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=comment_id, severity="high", file="a.py", start_line=None, line=3, status=status, body="x",
        posted_url="", post_error="",
    )


def test_results_link_to_the_pull_request_and_show_decision_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    comments = [comment("c1", "approved"), comment("c2", "pending"), comment("c3", "rejected")]
    app = run_app(monkeypatch, StubRepo([completed()], comments, []), review="r1")
    assert not app.exception
    assert any("https://github.com/octo/demo/pull/7" in m.value for m in app.markdown)
    assert any(p.proto.text == "2 of 3 comments decided" for p in app.get("progress"))


def test_sidebar_marks_status_and_highlights_the_open_review(monkeypatch: pytest.MonkeyPatch) -> None:
    other = completed("r2")
    other.status, other.number = "failed", 8
    app = run_app(monkeypatch, StubRepo([completed(), other], [], []), review="r1")
    assert not app.exception
    buttons = {b.key: b for b in app.sidebar.button}
    assert buttons["open-r1"].label.startswith("✅") and buttons["open-r2"].label.startswith("❌")
    assert buttons["open-r1"].proto.type == "primary"
    assert buttons["open-r2"].proto.type == "secondary"


def test_blank_url_gets_a_specific_message(monkeypatch: pytest.MonkeyPatch) -> None:
    app = run_app(monkeypatch, StubRepo([], [], []))
    next(b for b in app.button if b.label == "Start review").click().run()
    assert any("Enter a pull request URL" in e.value for e in app.error)


def test_malformed_url_shows_the_expected_form(monkeypatch: pytest.MonkeyPatch) -> None:
    app = run_app(monkeypatch, StubRepo([], [], []))
    app.text_input[0].input("https://example.com/nope")
    next(b for b in app.button if b.label == "Start review").click().run()
    assert any("owner/repo/pull/123" in e.value and "example.com/nope" in e.value for e in app.error)


def test_posted_comment_links_to_github(monkeypatch: pytest.MonkeyPatch) -> None:
    posted = comment("c1", "posted")
    posted.posted_url = "https://github.com/octo/demo/pull/7#discussion_r1"
    app = run_app(monkeypatch, StubRepo([completed()], [posted], []), review="r1")
    assert not app.exception
    assert any("](https://github.com/octo/demo/pull/7#discussion_r1)" in m.value for m in app.markdown)
