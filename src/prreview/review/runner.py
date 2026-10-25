"""Run one complete review: fetch PR context over MCP, then execute the graph."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from mcp import Client
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel

from prreview.agents.deps import ReviewDeps
from prreview.graph.build import build_graph
from prreview.schemas.github import ChangedFilesPage, PRRef, PullRequest
from prreview.schemas.review import PRContext, ReviewState
from prreview.schemas.search import FileListing

_PR_URL = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/pull/(\d+)/?$")
TEST_FILE_GLOBS = ("*test_*.py", "*_test.py", "pytest.ini")


class ReviewFailedError(RuntimeError):
    """The review could not run at all (e.g. the PR could not be fetched)."""


@dataclass(frozen=True)
class ReviewOutcome:
    """Final state plus wall-clock latency."""

    state: ReviewState
    latency_s: float


def parse_pr_url(url: str) -> PRRef:
    """Parse `https://github.com/<owner>/<repo>/pull/<n>`."""
    match = _PR_URL.match(url.strip())
    if not match:
        raise ValueError(f"not a GitHub pull request URL: {url!r}")
    owner, repo, number = match.groups()
    return PRRef(owner=owner, repo=repo, number=int(number))


async def fetch_context(ref: PRRef, deps: ReviewDeps) -> PRContext:
    """Deterministic data gathering before any agent runs. Failure here fails the review."""
    args = {"owner": ref.owner, "repo": ref.repo, "number": ref.number}
    pr = await call_structured(deps.mcp.github_read, "get_pull_request", args, PullRequest, deps.limits.mcp_call_timeout_s)
    files = await call_structured(deps.mcp.github_read, "list_changed_files", {**args, "max_files": 300},
                                  ChangedFilesPage, deps.limits.mcp_call_timeout_s)
    has_tests = await _has_test_config(pr, deps)
    return PRContext(pr=pr, files=files.files, files_truncated=files.truncated, has_test_config=has_tests)


async def run_review(ref: PRRef, review_id: str, deps: ReviewDeps) -> ReviewOutcome:
    """Fetch context and run the graph inside one trace."""
    started = time.monotonic()
    with deps.tracer.review(review_id, name=f"review {ref.owner}/{ref.repo}#{ref.number}", tags=["prreview"]) as span:
        context = await fetch_context(ref, deps)
        graph = build_graph(deps)
        initial = ReviewState(review_id=review_id, context=context)
        result = await graph.ainvoke(initial, config={"recursion_limit": deps.limits.graph_recursion_limit})
        state = ReviewState.model_validate(result)
        span.update(output={"drafts": len(state.drafts), "cost_usd": state.total_cost_usd, "steps": len(state.steps)})
    deps.tracer.flush()
    return ReviewOutcome(state=state, latency_s=round(time.monotonic() - started, 2))


async def call_structured(client: Client, tool: str, args: dict[str, Any], model: type[BaseModel], timeout_s: float) -> Any:
    """Call an MCP tool and validate its structured result, raising ReviewFailedError on any failure."""
    try:
        result = await client.call_tool(tool, args, read_timeout_seconds=timeout_s)
    except MCPError as exc:
        raise ReviewFailedError(f"{tool} failed: {exc.error.message}") from exc
    if result.is_error:
        message = " ".join(getattr(block, "text", "") for block in result.content)
        raise ReviewFailedError(f"{tool} failed: {message}")
    return model.model_validate(result.structured_content)


async def _has_test_config(pr: PullRequest, deps: ReviewDeps) -> bool:
    args = {"owner": pr.ref.owner, "repo": pr.ref.repo, "sha": pr.head_sha, "max_results": 1}
    for glob in TEST_FILE_GLOBS:
        listing = await call_structured(deps.mcp.search, "list_files", {**args, "glob": glob}, FileListing,
                                        deps.limits.mcp_call_timeout_s)
        if listing.paths:
            return True
    return False
