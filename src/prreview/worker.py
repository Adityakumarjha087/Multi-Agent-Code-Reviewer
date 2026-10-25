"""Review worker (D9): claims queued reviews from Postgres and runs them. The UI never runs the graph.

`python -m prreview.worker` - safe to run several copies (claims use FOR UPDATE SKIP LOCKED).
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from collections.abc import Awaitable, Callable

from prreview.config import database_settings
from prreview.persistence.repo import Repository
from prreview.review.factory import review_pr
from prreview.review.runner import ReviewOutcome
from prreview.schemas.github import PRRef

logger = logging.getLogger(__name__)
ReviewFn = Callable[..., Awaitable[ReviewOutcome]]


class DbEvents:
    """Writes progress events for the UI to poll."""

    def __init__(self, repo: Repository, review_id: str) -> None:
        self._repo, self._review_id = repo, review_id

    def emit(self, agent: str, kind: str, message: str) -> None:
        self._repo.add_event(self._review_id, agent, kind, message)


async def process_one(repo: Repository, review_fn: ReviewFn = review_pr) -> bool:
    """Run the oldest queued review. Returns False when the queue is empty."""
    row = repo.claim_next()
    if row is None:
        return False
    events = DbEvents(repo, row.id)
    events.emit("system", "started", f"reviewing {row.owner}/{row.repo}#{row.number}")
    try:
        outcome = await review_fn(PRRef(owner=row.owner, repo=row.repo, number=row.number), review_id=row.id, events=events)
    except Exception as exc:  # noqa: BLE001 - the one top-level boundary; see docstring of _record_failure
        _record_failure(repo, row.id, exc)
        return True
    repo.complete(row.id, outcome)
    events.emit("system", "finished", f"{len(outcome.state.drafts)} draft comments, ${outcome.state.total_cost_usd:.4f}")
    return True


def _record_failure(repo: Repository, review_id: str, exc: Exception) -> None:
    """A failed review must not kill the worker, and must not disappear: the full traceback is stored
    on the review (shown in the UI) and logged. Expected failures (bad URL, rate limit, missing key)
    and bugs both land here, distinguishable by the exception type in the message."""
    logger.exception("review %s failed", review_id)
    detail = "".join(traceback.format_exception_only(exc)).strip()
    repo.fail(review_id, f"{detail}\n\n{traceback.format_exc()}")
    repo.add_event(review_id, "system", "failed", detail)


async def run_forever(repo: Repository, poll_interval_s: float) -> None:
    """Process jobs until interrupted."""
    while True:
        if not await process_one(repo):
            await asyncio.sleep(poll_interval_s)


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO)
    settings = database_settings()
    asyncio.run(run_forever(Repository.from_url(settings.database_url), settings.worker_poll_interval_s))


if __name__ == "__main__":
    main()
