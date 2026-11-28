"""Worker job handling with a stub repository (no Postgres, no LLM, no network)."""

from __future__ import annotations

from types import SimpleNamespace

from prreview.review.runner import ReviewOutcome
from prreview.schemas.github import PRRef
from prreview.schemas.review import DraftComment, ReviewState
from prreview.worker import process_one
from tests.builders import pr_context


class StubRepo:
    """Serves one queued review and records every state change the worker makes."""

    def __init__(self, queued: bool = True) -> None:
        self._row = SimpleNamespace(id="r1", owner="octo", repo="demo", number=7) if queued else None
        self.events: list[tuple[str, str, str]] = []
        self.completed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def claim_next(self):
        row, self._row = self._row, None
        return row

    def add_event(self, review_id: str, agent: str, kind: str, message: str) -> None:
        self.events.append((agent, kind, message))

    def complete(self, review_id: str, outcome: ReviewOutcome) -> None:
        self.completed.append(review_id)

    def fail(self, review_id: str, error: str) -> None:
        self.failed.append((review_id, error))


def outcome(drafts: int) -> ReviewOutcome:
    state = ReviewState(review_id="r1", context=pr_context([]), drafts=[
        DraftComment(finding_ids=[f"f{i}"], file="a.py", line=10 + i, severity="high", body=f"issue {i}")
        for i in range(drafts)
    ])
    return ReviewOutcome(state=state, latency_s=0.5)


async def test_empty_queue_returns_false_without_running_a_review() -> None:
    called = []

    async def review(*args, **kwargs):
        called.append(True)

    assert await process_one(StubRepo(queued=False), review) is False
    assert called == []


async def test_successful_review_is_completed_with_start_and_finish_events() -> None:
    repo, seen = StubRepo(), {}

    async def review(ref: PRRef, review_id: str, events) -> ReviewOutcome:
        seen["ref"] = ref
        events.emit("planner", "started", "2 files")
        return outcome(drafts=2)

    assert await process_one(repo, review) is True
    assert seen["ref"] == PRRef(owner="octo", repo="demo", number=7)
    assert repo.completed == ["r1"] and repo.failed == []
    assert [e[:2] for e in repo.events] == [("system", "started"), ("planner", "started"), ("system", "finished")]
    assert repo.events[0][2] == "reviewing octo/demo#7"
    assert repo.events[-1][2].startswith("2 draft comments")


async def test_failed_review_stores_the_error_and_reports_the_job_as_handled() -> None:
    repo = StubRepo()

    async def review(ref: PRRef, review_id: str, events) -> ReviewOutcome:
        raise RuntimeError("GitHub API 404: Not Found")

    assert await process_one(repo, review) is True  # the worker keeps polling
    assert repo.completed == []
    [(review_id, error)] = repo.failed
    assert review_id == "r1" and error.startswith("RuntimeError: GitHub API 404: Not Found")
    assert "Traceback" in error
    assert repo.events[-1] == ("system", "failed", "RuntimeError: GitHub API 404: Not Found")
