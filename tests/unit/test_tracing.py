"""Instrumentation checked against the real Langfuse SDK, with spans captured in memory (no network)."""

from __future__ import annotations

import httpx
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from prreview.config import GitHubSettings, WorkspaceSettings
from prreview.observability.tracing import LangfuseSettings, LangfuseTracer
from prreview.review.runner import parse_pr_url, run_review
from tests.fakes import ScriptedLLM, tool_use_message, text_message
from tests.unit.test_graph import BUG_REPORT, deps_for, mcp_clients


async def test_review_emits_agent_generation_and_tool_spans(
    github_settings: GitHubSettings, workspace_settings: WorkspaceSettings
) -> None:
    exporter = InMemorySpanExporter()
    tracer = LangfuseTracer(LangfuseSettings(_env_file=None, public_key="pk-test", secret_key="sk-test"), exporter)
    script = {
        "You triage": [text_message({"priorities": [], "summary": "s"})],
        "senior software engineer": [tool_use_message([("search_code", {"query": "len(xs)"})]), text_message(BUG_REPORT)],
        "application security engineer": [text_message({"findings": [], "notes": ""})],
        "You merge findings": [text_message({"duplicate_groups": [], "overall_summary": "x"})],
    }
    osv = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    async with mcp_clients(github_settings, workspace_settings) as clients:
        deps = deps_for(ScriptedLLM(script), clients, osv)
        deps.tracer = tracer
        await run_review(parse_pr_url("https://github.com/octo/demo/pull/7"), "trace1", deps)

    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert {"planner", "bug", "security", "aggregator", "tool.search_code", "review octo/demo#7"} <= set(names)
    generations = [s for s in spans if s.name.endswith(".llm")]
    assert len(generations) == 5
    planner_generation = next(s for s in generations if s.name == "planner.llm").attributes
    assert planner_generation["langfuse.observation.type"] == "generation"
    assert planner_generation["langfuse.observation.model.name"] == "claude-opus-5-5"
    assert planner_generation["langfuse.observation.usage_details"] == '{"input": 1000, "output": 200, "cache_read": 0}'
    assert planner_generation["langfuse.observation.cost_details"] == '{"total": 0.008}'  # 1000*$4/M + 200*$20/M
    tool_span = next(s for s in spans if s.name == "tool.search_code").attributes
    assert tool_span["langfuse.observation.type"] == "tool"
    root = next(s for s in spans if s.name == "review octo/demo#7")
    assert all(s.context.trace_id == root.context.trace_id for s in spans)  # one trace per review
