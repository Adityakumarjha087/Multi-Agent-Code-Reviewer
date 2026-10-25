"""Wire real dependencies (Claude, MCP subprocesses, OSV, Langfuse) for one review."""

from __future__ import annotations

import uuid

import httpx

from prreview.agents.deps import ReviewDeps
from prreview.agents.loop import Budget
from prreview.config import agent_limits, github_settings, llm_settings, workspace_settings
from prreview.llm.client import AnthropicLLM, LLMClient
from prreview.observability.events import EventSink, NullEvents
from prreview.observability.tracing import Tracer, build_tracer
from prreview.review.connections import ConnectionSettings, connect
from prreview.review.runner import ReviewOutcome, run_review
from prreview.safety.untrusted import new_nonce
from prreview.schemas.github import PRRef


async def review_pr(
    ref: PRRef,
    review_id: str | None = None,
    enable_sandbox: bool = True,
    llm: LLMClient | None = None,
    tracer: Tracer | None = None,
    events: EventSink | None = None,
) -> ReviewOutcome:
    """Run one review end to end with production dependencies (an `llm` may be injected)."""
    llm_conf, limits = llm_settings(), agent_limits()
    llm = llm or AnthropicLLM(llm_conf)  # raises MissingCredentialsError before any work starts
    connection = ConnectionSettings() if enable_sandbox else ConnectionSettings(enable_sandbox=False)
    try:
        async with connect(github_settings(), workspace_settings(), connection) as clients, httpx.AsyncClient(timeout=20) as http:
            deps = ReviewDeps(
                llm=llm, llm_settings=llm_conf, limits=limits, budget=Budget(limits.review_max_cost_usd),
                nonce=new_nonce(), mcp=clients, http=http, tracer=tracer or build_tracer(),
                events=events or NullEvents(),
            )
            return await run_review(ref, review_id or uuid.uuid4().hex[:12], deps)
    except BaseExceptionGroup as group:
        raise single_error(group) from None


def single_error(group: BaseExceptionGroup) -> BaseException:
    """Unwrap anyio task-group nesting when exactly one real error caused it.

    Each MCP client runs an anyio task group, so an error raised inside `connect()` arrives wrapped
    in several ExceptionGroups and callers could not `except ReviewFailedError`. With more than
    one underlying error the group is returned unchanged so nothing is hidden.
    """
    leaves: list[BaseException] = []
    stack: list[BaseException] = [group]
    while stack:
        current = stack.pop()
        if isinstance(current, BaseExceptionGroup):
            stack.extend(current.exceptions)
        else:
            leaves.append(current)
    return leaves[0] if len(leaves) == 1 else group
