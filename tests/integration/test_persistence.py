"""Repository, worker, and publisher against a real PostgreSQL 17 container.

The schema is created by running the real Alembic migration. Run with: pytest -m docker
"""

from __future__ import annotations

import httpx
import pytest
from mcp import Client

from prreview.config import GitHubSettings
from prreview.github.client import GitHubClient
from prreview.mcp_servers import github_write
from prreview.persistence.repo import InvalidTransitionError, Repository
from prreview.publisher import publish_approved
from prreview.review.runner import ReviewOutcome
from prreview.schemas.github import PRRef
from prreview.schemas.review import DraftComment, ReviewState
from prreview.worker import process_one
from tests.builders import pr_context

pytestmark = pytest.mark.docker
REF = PRRef(owner="octo", repo="demo", number=7)


@pytest.fixture
def repo(clean_db: str) -> Repository:
    return Repository.from_url(clean_db)


def outcome(review_id: str, drafts: int = 2) -> ReviewOutcome:
    state = ReviewState(review_id=review_id, context=pr_context([]), summary="ok", drafts=[
        DraftComment(finding_ids=[f"f{i}"], file="a.py", line=10 + i, severity="high", body=f"issue {i}")
        for i in range(drafts)
    ])
    return ReviewOutcome(state=state, latency_s=1.5)


def test_concurrent_claims_get_different_jobs(repo: Repository) -> None:
    first, second = repo.enqueue(REF), repo.enqueue(REF)
    claimed = {repo.claim_next().id, repo.claim_next().id}
    assert claimed == {first, second}  # exactly our two jobs, each claimed once
    assert repo.claim_next() is None


async def test_worker_stores_results_and_events(repo: Repository) -> None:
    review_id = repo.enqueue(REF)

    async def fake_review(ref: PRRef, review_id: str, events) -> ReviewOutcome:
        events.emit("planner", "started", "2 files")
        return outcome(review_id)

    assert await process_one(repo, fake_review) is True
    row = repo.review(review_id)
    assert row.status == "completed" and float(row.latency_s) == 1.5 and row.state["summary"] == "ok"
    assert [c.status for c in repo.comments(review_id)] == ["pending", "pending"]
    assert [e.agent for e in repo.events(review_id)] == ["system", "planner", "system"]
    assert await process_one(repo, fake_review) is False  # queue empty


async def test_worker_records_failures_and_keeps_going(repo: Repository) -> None:
    review_id = repo.enqueue(REF)

    async def broken_review(ref: PRRef, review_id: str, events) -> ReviewOutcome:
        raise RuntimeError("GitHub API 404: Not Found")

    assert await process_one(repo, broken_review) is True
    row = repo.review(review_id)
    assert row.status == "failed" and "RuntimeError: GitHub API 404" in row.error
    assert repo.events(review_id)[-1].kind == "failed"


def test_comment_lifecycle_rules(repo: Repository) -> None:
    review_id = repo.enqueue(REF)
    repo.claim_next()
    repo.complete(review_id, outcome(review_id))
    first, second = repo.comments(review_id)
    repo.decide(first.id, "approved")
    repo.decide(second.id, "rejected")
    with pytest.raises(InvalidTransitionError):
        repo.decide(first.id, "posted")  # humans cannot mark things posted
    assert repo.comment_for_posting(first.id) is None  # approved but not yet claimed
    claimed = repo.claim_approved()
    assert claimed.id == first.id and claimed.commit_id == "b" * 40
    assert repo.comment_for_posting(first.id) is not None
    with pytest.raises(InvalidTransitionError):
        repo.decide(first.id, "pending")  # cannot un-approve while posting
    assert repo.claim_approved() is None  # rejected comment is never claimed


async def test_publisher_posts_only_approved_comments(repo: Repository) -> None:
    review_id = repo.enqueue(REF)
    repo.claim_next()
    repo.complete(review_id, outcome(review_id, drafts=3))
    approved, failing, pending = repo.comments(review_id)
    repo.decide(approved.id, "approved")
    repo.decide(failing.id, "approved")
    posted_paths: list[int] = []

    def github(request: httpx.Request) -> httpx.Response:
        line = int(request.read().split(b'"line":')[1].split(b",")[0])
        posted_paths.append(line)
        if line == failing.line:
            return httpx.Response(422, json={"message": "line must be part of the diff"})
        return httpx.Response(201, json={"id": 555, "html_url": "https://github.com/c/555", "path": "a.py", "line": line})

    settings = GitHubSettings(_env_file=None, github_retry_initial_wait_s=0.01)
    client = GitHubClient(settings, "t0k", transport=httpx.MockTransport(github))
    async with Client(github_write.build_server(client, repo.comment_for_posting)) as write_server:
        attempted = await publish_approved(repo, write_server)

    assert attempted == 2 and sorted(posted_paths) == sorted([approved.line, failing.line])
    statuses = {c.id: c for c in repo.comments(review_id)}
    assert statuses[approved.id].status == "posted" and statuses[approved.id].posted_comment_id == 555
    assert statuses[failing.id].status == "post_failed" and "422" in statuses[failing.id].post_error
    assert statuses[pending.id].status == "pending"
