"""Full review against live GitHub through real stdio MCP subprocesses, with a test-double LLM.

Proves the production wiring (connect(), env filtering, context fetch, graph) without an API key.
It does NOT measure review quality; that needs the real model (benchmark).
"""

from __future__ import annotations

import pytest

from prreview.review.factory import review_pr
from prreview.review.runner import parse_pr_url
from prreview.observability.tracing import NoopTracer
from tests.fakes import MinimalLLM

pytestmark = pytest.mark.network


async def test_live_review_wiring_without_sandbox(github_quota: None) -> None:
    llm = MinimalLLM()
    outcome = await review_pr(parse_pr_url("https://github.com/psf/requests/pull/6963"), "live1",
                              enable_sandbox=False, llm=llm, tracer=NoopTracer())
    state = outcome.state
    paths = {f.path for f in state.context.files}
    assert "src/requests/utils.py" in paths
    assert state.context.has_test_config is True  # tests/test_utils.py exists at head
    assert {t.kind for t in state.plan.tasks} >= {"bug", "security", "tests"}
    assert any(n.agent == "tests" and "sandbox disabled" in n.message for n in state.notes)
    assert {c.output_model.__name__ for c in llm.calls} == {"PlannerOutput", "FindingsReport"}
    assert state.drafts == []  # the double reports nothing, and requests' PR has no scanner hits
