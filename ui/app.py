"""Streamlit UI: queue a review, watch progress, approve or reject each draft comment.

The UI never runs the review graph and never posts to GitHub (D9). It writes a job row for the
worker, polls Postgres for progress, and records approval decisions that the publisher acts on.
Run: `streamlit run ui/app.py`
"""

from __future__ import annotations

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from prreview.config import database_settings
from prreview.persistence.models import CommentRow, ReviewRow
from prreview.persistence.repo import InvalidTransitionError, Repository
from prreview.review.runner import PRRef, parse_pr_url

ACTIVE = ("queued", "running")
SEVERITY_COLOR = {"critical": "red", "high": "orange", "medium": "yellow", "low": "blue", "info": "gray"}
REVIEW_ICON = {"queued": "⏳", "running": "🔄", "completed": "✅", "failed": "❌"}
STATUS_COLOR = {"pending": "gray", "approved": "green", "rejected": "red", "posting": "orange", "posted": "violet", "post_failed": "red"}


@st.cache_resource
def repository() -> Repository:
    """One connection pool per Streamlit server process."""
    return Repository.from_url(database_settings().database_url)


def main() -> None:
    """Page entry point."""
    st.set_page_config(page_title="PR review", page_icon=":mag:", layout="wide")
    st.title("Multi-agent PR review")
    try:
        repo = repository()
        _sidebar(repo)
    except SQLAlchemyError as exc:
        st.error(
            "Cannot reach the database, so reviews cannot be listed or queued. "
            "Check that Postgres is running and that DATABASE_URL is correct, then reload this page."
        )
        st.caption(f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else 'no detail'}")
        st.stop()
    _new_review_form(repo)
    review_id = st.query_params.get("review")
    if review_id:
        _review_panel(repo, review_id)


def _sidebar(repo: Repository) -> None:
    st.sidebar.header("Recent reviews")
    rows = repo.recent_reviews()
    if not rows:
        st.sidebar.caption("No reviews yet. Paste a pull request URL to start one.")
    current = st.query_params.get("review")
    for row in rows:
        label = f"{REVIEW_ICON.get(row.status, '')} {row.owner}/{row.repo}#{row.number} · {row.status}".strip()
        kind = "primary" if row.id == current else "secondary"
        if st.sidebar.button(label, key=f"open-{row.id}", type=kind, width="stretch"):
            st.query_params["review"] = row.id
            st.rerun()


def _new_review_form(repo: Repository) -> None:
    with st.form("new-review", clear_on_submit=True):
        url = st.text_input("Pull request URL", placeholder="https://github.com/owner/repo/pull/123")
        if st.form_submit_button("Start review"):
            if not url.strip():
                st.error("Enter a pull request URL, for example https://github.com/owner/repo/pull/123.")
                return
            try:
                review_id = repo.enqueue(parse_pr_url(url))
            except ValueError:
                st.error(
                    f"{url.strip()!r} is not a GitHub pull request URL. "
                    "Use the form https://github.com/owner/repo/pull/123."
                )
                return
            st.query_params["review"] = review_id
            st.rerun()


def _review_panel(repo: Repository, review_id: str) -> None:
    row = repo.review(review_id)
    if row is None:
        st.warning(f"No review with id {review_id}.")
        return
    st.subheader(f"{row.owner}/{row.repo}#{row.number}")
    st.markdown(f"[Open on GitHub](https://github.com/{row.owner}/{row.repo}/pull/{row.number})")
    st.caption(_timeline(row))
    if row.status in ACTIVE:
        _live_progress(repo, review_id)
        return
    if row.status == "failed":
        st.error("Review failed")
        st.code(row.error, language="text")
        if st.button("Retry review", key=f"retry-{row.id}", help="Queue a new review of this pull request"):
            st.query_params["review"] = repo.enqueue(PRRef(owner=row.owner, repo=row.repo, number=row.number))
            st.rerun()
        _events(repo, review_id, expanded=False)
        return
    _results(repo, row)


def _timeline(row: ReviewRow) -> str:
    """One line such as 'Queued 2026-10-09 14:02 · started 14:02 · finished 14:05', skipping unknown times."""
    parts = [f"Queued {row.created_at:%Y-%m-%d %H:%M}"]
    if row.started_at:
        parts.append(f"started {row.started_at:%H:%M}")
    if row.finished_at:
        parts.append(f"{'failed' if row.status == 'failed' else 'finished'} {row.finished_at:%H:%M}")
    return " · ".join(parts)


@st.fragment(run_every=2)
def _live_progress(repo: Repository, review_id: str) -> None:
    """Re-runs every 2 s while the worker is busy; reloads the whole page once the review finishes."""
    row = repo.review(review_id)
    if row is not None and row.status not in ACTIVE:
        st.rerun()
    st.info("Queued, waiting for a worker…" if row.status == "queued" else "Review running…")
    events = repo.events(review_id)
    if events:
        latest = events[-1]
        st.caption(f"Latest: {latest.agent} {latest.kind} at {latest.created_at:%H:%M:%S}")
    _events(repo, review_id, expanded=True)


def _events(repo: Repository, review_id: str, expanded: bool) -> None:
    events = repo.events(review_id)
    with st.expander(f"Agent progress ({len(events)} events)", expanded=expanded):
        if not events:
            st.caption("No agent events recorded yet.")
        for event in events:
            st.markdown(f"`{event.created_at:%H:%M:%S}` **{event.agent}** {event.kind}: {event.message}")


def _results(repo: Repository, row: ReviewRow) -> None:
    comments = repo.comments(row.id)
    cols = st.columns(4)
    cols[0].metric("Draft comments", len(comments))
    cols[1].metric("Steps", row.steps, help="Agent tool-loop steps taken across all reviewers")
    cols[2].metric("Cost", f"${float(row.cost_usd):.4f}", help="Model spend for this review")
    cols[3].metric("Latency", f"{float(row.latency_s):.1f} s", help="Wall-clock time from start to finish")
    if row.summary:
        st.markdown(f"**Summary:** {row.summary}")
    notes = (row.state or {}).get("notes", [])
    with st.expander(f"Notes ({len(notes)}): skipped work, tool failures, discarded findings"):
        if not notes:
            st.caption("No notes: nothing was skipped, no tool failed, no finding was discarded.")
        for note in notes:
            st.markdown(f"- **{note['agent']}/{note['kind']}**: {note['message']}")
    _events(repo, row.id, expanded=False)
    st.divider()
    if comments:
        decided = sum(c.status != "pending" for c in comments)
        st.progress(decided / len(comments), text=f"{decided} of {len(comments)} comments decided")
        st.caption(f"{_severity_breakdown(comments)} | {_status_breakdown(comments)}")
    _bulk_decisions(repo, comments)
    st.caption("Approve the comments to post. Approved comments are posted by the publisher service, not by this page.")
    if not comments:
        st.info("No draft comments: the reviewers found nothing worth posting on this pull request.")
    for comment in _filter_comments(comments):
        _comment_card(repo, comment)


def _filter_comments(comments: list[CommentRow]) -> list[CommentRow]:
    """Let the approver narrow a long list to one status (offered only when statuses are mixed)."""
    present = [s for s in STATUS_COLOR if any(c.status == s for c in comments)]
    if len(comments) < 3 or len(present) < 2:
        return comments
    options = ["all", *present]
    chosen = st.segmented_control(
        "Show comments", options, default="all", key="comment-filter",
        format_func=lambda s: s if s == "all" else f"{s.replace('_', ' ')} ({sum(c.status == s for c in comments)})",
    )
    if chosen in (None, "all"):
        return comments
    return [c for c in comments if c.status == chosen]


def _bulk_decisions(repo: Repository, comments: list[CommentRow]) -> None:
    """Approve or reject every still-pending comment in one click (shown only when two or more wait)."""
    pending = [c for c in comments if c.status == "pending"]
    if len(pending) < 2:
        return
    approve, reject, _ = st.columns([2, 2, 5])
    for column, label, status in ((approve, "Approve all pending", "approved"), (reject, "Reject all pending", "rejected")):
        if column.button(f"{label} ({len(pending)})", key=f"bulk-{status}", help="Only comments that have not been decided yet"):
            for comment in pending:
                try:
                    repo.decide(comment.id, status)
                except InvalidTransitionError:  # claimed or decided elsewhere meanwhile: leave it as it is
                    continue
            st.rerun()


def _severity_breakdown(comments: list[CommentRow]) -> str:
    """One line such as '1 critical · 3 medium', most severe first, omitting severities with no comments."""
    counts = {severity: sum(c.severity == severity for c in comments) for severity in SEVERITY_COLOR}
    return " · ".join(f"{n} {severity}" for severity, n in counts.items() if n)


def _status_breakdown(comments: list[CommentRow]) -> str:
    """One line such as '2 pending · 1 approved', in a fixed status order, omitting empty statuses."""
    counts = {status: sum(c.status == status for c in comments) for status in STATUS_COLOR}
    return " · ".join(f"{n} {status.replace('_', ' ')}" for status, n in counts.items() if n)


def _comment_card(repo: Repository, comment: CommentRow) -> None:
    with st.container(border=True):
        location = f"{comment.file}:{comment.start_line}-{comment.line}" if comment.start_line else f"{comment.file}:{comment.line}"
        severity, where, status = st.columns([1, 6, 2], vertical_alignment="center")
        severity.badge(comment.severity.upper(), color=SEVERITY_COLOR.get(comment.severity, "gray"))
        where.markdown(f"`{location}`")
        status.badge(comment.status, color=STATUS_COLOR.get(comment.status, "gray"))
        # Shown as literal text: this is exactly what will be posted, and nothing in it is rendered
        # in the approver's browser (no remote images or links from PR-derived content).
        st.code(comment.body, language="markdown", wrap_lines=True)
        if comment.posted_url:
            st.markdown(f"Posted: [view the comment on GitHub]({comment.posted_url})")
        if comment.post_error:
            st.error(comment.post_error)
        _decision_buttons(repo, comment)


def _decision_buttons(repo: Repository, comment: CommentRow) -> None:
    if comment.status not in ("pending", "approved", "rejected", "post_failed"):
        return
    approve, reject, _ = st.columns([1, 1, 6])
    for column, label, status in ((approve, "Approve", "approved"), (reject, "Reject", "rejected")):
        if column.button(label, key=f"{status}-{comment.id}", disabled=comment.status == status):
            try:
                repo.decide(comment.id, status)
            except InvalidTransitionError as exc:  # e.g. the publisher claimed it meanwhile
                st.error(str(exc))
                return
            st.rerun()


main()
