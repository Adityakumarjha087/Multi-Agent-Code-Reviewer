"""Streamlit UI: queue a review, watch progress, approve or reject each draft comment.

The UI never runs the review graph and never posts to GitHub (D9). It writes a job row for the
worker, polls Postgres for progress, and records approval decisions that the publisher acts on.
Run: `streamlit run ui/app.py`
"""

from __future__ import annotations

import streamlit as st

from prreview.config import database_settings
from prreview.persistence.models import CommentRow, ReviewRow
from prreview.persistence.repo import InvalidTransitionError, Repository
from prreview.review.runner import parse_pr_url

ACTIVE = ("queued", "running")
SEVERITY_COLOR = {"critical": "red", "high": "orange", "medium": "yellow", "low": "blue", "info": "gray"}
STATUS_COLOR = {"pending": "gray", "approved": "green", "rejected": "red", "posting": "orange", "posted": "violet", "post_failed": "red"}


@st.cache_resource
def repository() -> Repository:
    """One connection pool per Streamlit server process."""
    return Repository.from_url(database_settings().database_url)


def main() -> None:
    """Page entry point."""
    st.set_page_config(page_title="PR review", page_icon=":mag:", layout="wide")
    st.title("Multi-agent PR review")
    repo = repository()
    _sidebar(repo)
    _new_review_form(repo)
    review_id = st.query_params.get("review")
    if review_id:
        _review_panel(repo, review_id)


def _sidebar(repo: Repository) -> None:
    st.sidebar.header("Recent reviews")
    rows = repo.recent_reviews()
    if not rows:
        st.sidebar.caption("No reviews yet. Paste a pull request URL to start one.")
    for row in rows:
        label = f"{row.owner}/{row.repo}#{row.number} · {row.status}"
        if st.sidebar.button(label, key=f"open-{row.id}", width="stretch"):
            st.query_params["review"] = row.id
            st.rerun()


def _new_review_form(repo: Repository) -> None:
    with st.form("new-review", clear_on_submit=True):
        url = st.text_input("Pull request URL", placeholder="https://github.com/owner/repo/pull/123")
        if st.form_submit_button("Start review"):
            try:
                review_id = repo.enqueue(parse_pr_url(url))
            except ValueError as exc:
                st.error(str(exc))
                return
            st.query_params["review"] = review_id
            st.rerun()


def _review_panel(repo: Repository, review_id: str) -> None:
    row = repo.review(review_id)
    if row is None:
        st.warning(f"No review with id {review_id}.")
        return
    st.subheader(f"{row.owner}/{row.repo}#{row.number}")
    if row.status in ACTIVE:
        _live_progress(repo, review_id)
        return
    if row.status == "failed":
        st.error("Review failed")
        st.code(row.error, language="text")
        _events(repo, review_id, expanded=False)
        return
    _results(repo, row)


@st.fragment(run_every=2)
def _live_progress(repo: Repository, review_id: str) -> None:
    """Re-runs every 2 s while the worker is busy; reloads the whole page once the review finishes."""
    row = repo.review(review_id)
    if row is not None and row.status not in ACTIVE:
        st.rerun()
    st.info("Queued, waiting for a worker…" if row.status == "queued" else "Review running…")
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
    cols[1].metric("Steps", row.steps)
    cols[2].metric("Cost", f"${float(row.cost_usd):.4f}")
    cols[3].metric("Latency", f"{float(row.latency_s):.1f} s")
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
    st.caption("Approve the comments to post. Approved comments are posted by the publisher service, not by this page.")
    if not comments:
        st.info("No draft comments: the reviewers found nothing worth posting on this pull request.")
    for comment in comments:
        _comment_card(repo, comment)


def _comment_card(repo: Repository, comment: CommentRow) -> None:
    with st.container(border=True):
        location = f"{comment.file}:{comment.start_line}-{comment.line}" if comment.start_line else f"{comment.file}:{comment.line}"
        st.badge(comment.severity.upper(), color=SEVERITY_COLOR.get(comment.severity, "gray"))
        st.markdown(f"`{location}`")
        st.badge(comment.status, color=STATUS_COLOR.get(comment.status, "gray"))
        # Shown as literal text: this is exactly what will be posted, and nothing in it is rendered
        # in the approver's browser (no remote images or links from PR-derived content).
        st.code(comment.body, language="markdown", wrap_lines=True)
        if comment.posted_url:
            st.markdown(f"Posted: {comment.posted_url}")
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
