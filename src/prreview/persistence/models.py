"""ORM tables. The schema is created by Alembic migrations (`alembic upgrade head`), not create_all."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Timezone-aware current time."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all tables."""


Timestamp = DateTime(timezone=True)


class ReviewRow(Base):
    """One review job and, once finished, its results."""

    __tablename__ = "reviews"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    owner: Mapped[str] = mapped_column(String(100))
    repo: Mapped[str] = mapped_column(String(100))
    number: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), index=True)  # queued|running|completed|failed
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(Timestamp, default=None)
    finished_at: Mapped[datetime | None] = mapped_column(Timestamp, default=None)
    head_sha: Mapped[str | None] = mapped_column(String(40), default=None)
    summary: Mapped[str] = mapped_column(Text, default="")
    cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), default=0)
    steps: Mapped[int] = mapped_column(Integer, default=0)
    latency_s: Mapped[float] = mapped_column(Numeric(10, 2), default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    state: Mapped[dict | None] = mapped_column(JSONB, default=None)


class EventRow(Base):
    """Progress events shown live in the UI."""

    __tablename__ = "review_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    review_id: Mapped[str] = mapped_column(ForeignKey("reviews.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)
    agent: Mapped[str] = mapped_column(String(16))
    kind: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)


class CommentRow(Base):
    """A draft review comment and its approval/posting lifecycle."""

    __tablename__ = "draft_comments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    review_id: Mapped[str] = mapped_column(ForeignKey("reviews.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer)  # rank order within the review
    file: Mapped[str] = mapped_column(Text)
    line: Mapped[int] = mapped_column(Integer)
    start_line: Mapped[int | None] = mapped_column(Integer, default=None)
    severity: Mapped[str] = mapped_column(String(16))
    body: Mapped[str] = mapped_column(Text)
    finding_ids: Mapped[list] = mapped_column(JSONB, default=list)
    status: Mapped[str] = mapped_column(String(16), index=True)  # pending|approved|rejected|posting|posted|post_failed
    decided_at: Mapped[datetime | None] = mapped_column(Timestamp, default=None)
    posted_comment_id: Mapped[int | None] = mapped_column(BigInteger, default=None)
    posted_url: Mapped[str | None] = mapped_column(Text, default=None)
    post_error: Mapped[str] = mapped_column(Text, default="")
