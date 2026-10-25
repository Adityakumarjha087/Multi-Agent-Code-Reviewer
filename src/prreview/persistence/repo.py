"""Database operations used by the UI, worker, publisher, and write MCP server."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import Engine, create_engine, select, update
from sqlalchemy.orm import Session, sessionmaker

from prreview.persistence.models import CommentRow, EventRow, ReviewRow, utcnow
from prreview.review.runner import ReviewOutcome
from prreview.schemas.github import PRRef

# Who may move a comment where. `posting` is committed BEFORE calling GitHub: if the publisher dies
# after GitHub accepted the comment but before we record it, the row stays `posting` and is never
# retried automatically, so a crash cannot double-post.
HUMAN_TRANSITIONS = {
    "pending": {"approved", "rejected"},
    "approved": {"pending", "rejected"},
    "rejected": {"pending", "approved"},
    "post_failed": {"approved", "rejected"},
}
PUBLISHER_TRANSITIONS = {"approved": {"posting"}, "posting": {"posted", "post_failed"}}


class InvalidTransitionError(ValueError):
    """A status change that the lifecycle does not allow."""


@dataclass(frozen=True)
class PostableComment:
    """Everything needed to post one approved comment."""

    id: str
    owner: str
    repo: str
    number: int
    commit_id: str
    file: str
    line: int
    start_line: int | None
    body: str


class Repository:
    """Thin data-access layer. Each method is its own transaction."""

    def __init__(self, engine: Engine) -> None:
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    @classmethod
    def from_url(cls, url: str) -> Repository:
        """Create a repository with its own engine."""
        return cls(create_engine(url, pool_pre_ping=True))

    def enqueue(self, ref: PRRef) -> str:
        """Add a queued review and return its id."""
        review_id = uuid.uuid4().hex[:12]
        with self._sessions.begin() as session:
            session.add(ReviewRow(id=review_id, owner=ref.owner, repo=ref.repo, number=ref.number, status="queued"))
        return review_id

    def claim_next(self) -> ReviewRow | None:
        """Atomically take the oldest queued review (safe with several workers: SKIP LOCKED)."""
        with self._sessions.begin() as session:
            row = session.scalars(
                select(ReviewRow).where(ReviewRow.status == "queued").order_by(ReviewRow.created_at)
                .limit(1).with_for_update(skip_locked=True)
            ).first()
            if row is None:
                return None
            row.status, row.started_at = "running", utcnow()
            return row

    def complete(self, review_id: str, outcome: ReviewOutcome) -> None:
        """Store results and draft comments."""
        state = outcome.state
        with self._sessions.begin() as session:
            row = session.get(ReviewRow, review_id)
            row.status, row.finished_at = "completed", utcnow()
            row.head_sha = state.context.pr.head_sha
            row.summary, row.cost_usd, row.steps = state.summary, state.total_cost_usd, len(state.steps)
            row.latency_s, row.state = outcome.latency_s, state.model_dump(mode="json")
            for position, draft in enumerate(state.drafts):
                session.add(CommentRow(
                    id=draft.id, review_id=review_id, position=position, file=draft.file, line=draft.line,
                    start_line=draft.start_line, severity=draft.severity, body=draft.body,
                    finding_ids=draft.finding_ids, status="pending",
                ))

    def fail(self, review_id: str, error: str) -> None:
        """Mark a review as failed with its error message."""
        with self._sessions.begin() as session:
            row = session.get(ReviewRow, review_id)
            row.status, row.finished_at, row.error = "failed", utcnow(), error[:4000]

    def add_event(self, review_id: str, agent: str, kind: str, message: str) -> None:
        """Append one progress event."""
        with self._sessions.begin() as session:
            session.add(EventRow(review_id=review_id, agent=agent, kind=kind, message=message[:2000]))

    def events(self, review_id: str, after_id: int = 0) -> list[EventRow]:
        """Events newer than `after_id`, oldest first."""
        with self._sessions() as session:
            query = select(EventRow).where(EventRow.review_id == review_id, EventRow.id > after_id).order_by(EventRow.id)
            return list(session.scalars(query))

    def review(self, review_id: str) -> ReviewRow | None:
        """One review row."""
        with self._sessions() as session:
            return session.get(ReviewRow, review_id)

    def recent_reviews(self, limit: int = 20) -> list[ReviewRow]:
        """Newest reviews first."""
        with self._sessions() as session:
            return list(session.scalars(select(ReviewRow).order_by(ReviewRow.created_at.desc()).limit(limit)))

    def comments(self, review_id: str) -> list[CommentRow]:
        """Draft comments in rank order."""
        with self._sessions() as session:
            query = select(CommentRow).where(CommentRow.review_id == review_id).order_by(CommentRow.position)
            return list(session.scalars(query))

    def decide(self, comment_id: str, status: str) -> None:
        """Human approval decision."""
        self._transition(comment_id, status, HUMAN_TRANSITIONS, decided=True)

    def claim_approved(self) -> PostableComment | None:
        """Take one approved comment and mark it `posting` before any GitHub call."""
        with self._sessions.begin() as session:
            row = session.scalars(
                select(CommentRow).where(CommentRow.status == "approved").order_by(CommentRow.decided_at)
                .limit(1).with_for_update(skip_locked=True)
            ).first()
            if row is None:
                return None
            row.status = "posting"
            review = session.get(ReviewRow, row.review_id)
            return PostableComment(row.id, review.owner, review.repo, review.number, review.head_sha,
                                   row.file, row.line, row.start_line, row.body)

    def comment_for_posting(self, comment_id: str) -> PostableComment | None:
        """Used by the write server to re-check, itself, that a comment is approved for posting."""
        with self._sessions() as session:
            row = session.get(CommentRow, comment_id)
            if row is None or row.status != "posting":
                return None
            review = session.get(ReviewRow, row.review_id)
            return PostableComment(row.id, review.owner, review.repo, review.number, review.head_sha,
                                   row.file, row.line, row.start_line, row.body)

    def record_posted(self, comment_id: str, github_id: int, url: str) -> None:
        """GitHub accepted the comment."""
        self._transition(comment_id, "posted", PUBLISHER_TRANSITIONS, posted_comment_id=github_id, posted_url=url)

    def record_post_failed(self, comment_id: str, error: str) -> None:
        """GitHub rejected the comment (definitively)."""
        self._transition(comment_id, "post_failed", PUBLISHER_TRANSITIONS, post_error=error[:2000])

    def _transition(self, comment_id: str, status: str, allowed: dict[str, set[str]], decided: bool = False, **fields: object) -> None:
        with self._sessions.begin() as session:
            row = session.get(CommentRow, comment_id, with_for_update=True)
            if row is None:
                raise InvalidTransitionError(f"no comment {comment_id!r}")
            if status not in allowed.get(row.status, set()):
                raise InvalidTransitionError(f"comment {comment_id}: {row.status} -> {status} is not allowed")
            values = {"status": status, **fields}
            if decided:
                values["decided_at"] = utcnow()
            session.execute(update(CommentRow).where(CommentRow.id == comment_id).values(**values))
