"""Drive the real Streamlit page with AppTest against real Postgres. Run with: pytest -m docker"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from prreview.config import database_settings
from prreview.persistence.repo import Repository
from tests.integration.test_persistence import REF, outcome

pytestmark = pytest.mark.docker


@pytest.fixture
def app(clean_db: str, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("DATABASE_URL", clean_db)
    database_settings.cache_clear()
    st.cache_resource.clear()
    # AppTest resolves relative paths against this test file, so pass an absolute path.
    yield AppTest.from_file(Path(__file__).resolve().parents[2] / "ui" / "app.py", default_timeout=20)
    database_settings.cache_clear()


def submit(app: AppTest) -> None:
    """Click the form's submit button (selected by label: the sidebar has buttons too)."""
    next(b for b in app.button if b.label == "Start review").click().run()


def completed_review(repo: Repository) -> str:
    review_id = repo.enqueue(REF)
    repo.claim_next()
    repo.complete(review_id, outcome(review_id, drafts=2))
    repo.add_event(review_id, "planner", "finished", "bug(1 files), security(1 files)")
    return review_id


def test_results_page_and_approval(app: AppTest, database_url: str) -> None:
    repo = Repository.from_url(database_url)
    review_id = completed_review(repo)
    app.query_params["review"] = review_id
    app.run()
    assert not app.exception
    assert [m.value for m in app.metric][:2] == ["2", "0"]
    assert [c.value for c in app.code] == ["issue 0", "issue 1"]  # literal comment bodies
    first = repo.comments(review_id)[0]
    app.button(key=f"approved-{first.id}").click().run()
    assert repo.comments(review_id)[0].status == "approved"
    assert repo.comments(review_id)[1].status == "pending"


def test_submitting_a_url_queues_a_review(app: AppTest, database_url: str) -> None:
    app.run()
    app.text_input[0].input("https://github.com/octo/demo/pull/42")
    submit(app)
    review_id = app.query_params["review"]
    row = Repository.from_url(database_url).review(review_id)
    assert (row.owner, row.repo, row.number, row.status) == ("octo", "demo", 42, "queued")


def test_invalid_url_shows_an_error(app: AppTest) -> None:
    app.run()
    app.text_input[0].input("https://example.com/not-a-pr")
    submit(app)
    assert "not a GitHub pull request URL" in app.error[0].value
