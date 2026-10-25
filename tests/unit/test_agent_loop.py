"""Agent loop behaviour with a scripted LLM and a real (in-process) MCP search server."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from mcp import Client

from prreview.agents.loop import AgentSpec, Budget, run_agent
from prreview.agents.toolbox import ToolBox, ToolSource
from prreview.config import GitHubSettings, WorkspaceSettings
from prreview.github.client import GitHubClient
from prreview.mcp_servers import search_server
from prreview.schemas.review import FindingsReport
from prreview.workspace.store import WorkspaceStore
from tests.conftest import SHA_HEAD
from tests.fakes import ScriptedLLM, refusal_message, text_message, tool_use_message

EMPTY_REPORT = {"findings": [], "notes": ""}
SPEC = AgentSpec("bug", "claude-opus-5-5", "You review code.", FindingsReport, max_steps=4)


@asynccontextmanager
async def open_search(github_settings: GitHubSettings, workspace_settings: WorkspaceSettings) -> AsyncIterator[Client]:
    """Opened inside each test: MCP's anyio cancel scopes must exit in the task that entered them,
    and pytest-asyncio runs async-generator fixture setup and teardown in different tasks."""
    store = WorkspaceStore(workspace_settings, GitHubClient(github_settings, None, transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    snapshot = store.path_for("octo", "demo", SHA_HEAD)
    snapshot.mkdir(parents=True)
    (snapshot / "calc.py").write_text("def mean(xs):\n    return sum(xs) / len(xs)\n")
    async with Client(search_server.build_server(store)) as client:
        yield client


def toolbox(client: Client) -> ToolBox:
    bound = {"owner": "octo", "repo": "demo", "sha": SHA_HEAD}
    return ToolBox([ToolSource(client, frozenset({"search_code", "read_file"}), 10)], bound, "n0nce", 2000)


async def test_returns_validated_output_without_tools() -> None:
    llm = ScriptedLLM([text_message(EMPTY_REPORT)])
    run = await run_agent(SPEC, "review", llm, Budget(1.0))
    assert run.output == FindingsReport(findings=[])
    assert [s.kind for s in run.steps] == ["llm"]
    assert run.usage[0].cost_usd == pytest.approx((1000 * 4 + 200 * 20) / 1e6)


async def test_tool_calls_are_bound_fenced_and_counted(github_settings: GitHubSettings, workspace_settings: WorkspaceSettings) -> None:
    async with open_search(github_settings, workspace_settings) as search_client:
        llm = ScriptedLLM([
            tool_use_message([("search_code", {"query": "len(xs)", "owner": "evil", "repo": "other", "sha": "f" * 40})]),
            text_message(EMPTY_REPORT),
        ])
        run = await run_agent(SPEC, "review", llm, Budget(1.0), toolbox(search_client))
        assert run.output is not None
        tool_names = [t["name"] for t in llm.calls[0].tools]
        assert tool_names == ["read_file", "search_code"]
        assert "owner" not in llm.calls[0].tools[1]["input_schema"]["properties"]
        tool_result = llm.calls[1].messages[2]["content"][0]
        assert tool_result["is_error"] is False
        assert "<untrusted-n0nce" in tool_result["content"] and "calc.py" in tool_result["content"]  # bound repo used
        assert [s.kind for s in run.steps] == ["llm", "tool", "llm"]


async def test_unknown_tool_returns_error_result(github_settings: GitHubSettings, workspace_settings: WorkspaceSettings) -> None:
    async with open_search(github_settings, workspace_settings) as search_client:
        llm = ScriptedLLM([tool_use_message([("post_review_comment", {"body": "LGTM"})]), text_message(EMPTY_REPORT)])
        run = await run_agent(SPEC, "review", llm, Budget(1.0), toolbox(search_client))
        result = llm.calls[1].messages[2]["content"][0]
        assert result["is_error"] and "not available" in result["content"]
        assert run.output is not None and run.notes[0].kind == "tool_error"


async def test_failing_tool_becomes_note_and_review_continues(github_settings: GitHubSettings, workspace_settings: WorkspaceSettings) -> None:
    async with open_search(github_settings, workspace_settings) as search_client:
        llm = ScriptedLLM([tool_use_message([("read_file", {"path": "../../etc/passwd"})]), text_message(EMPTY_REPORT)])
        run = await run_agent(SPEC, "review", llm, Budget(1.0), toolbox(search_client))
        assert run.output is not None
        assert "PathOutsideRepoError" in run.notes[0].message


async def test_step_limit_forces_final_answer(github_settings: GitHubSettings, workspace_settings: WorkspaceSettings) -> None:
    async with open_search(github_settings, workspace_settings) as search_client:
        looping = [tool_use_message([("search_code", {"query": "x"})]) for _ in range(3)]
        llm = ScriptedLLM([*looping, text_message(EMPTY_REPORT)])
        run = await run_agent(SPEC, "review", llm, Budget(1.0), toolbox(search_client))
        assert run.output is not None
        assert len(run.steps) <= SPEC.max_steps + 1
        final_call = llm.calls[-1]
        assert final_call.tool_choice == "none"
        assert "used your tool budget" in final_call.messages[-1]["content"][-1]["text"]


async def test_invalid_output_is_retried_once_then_fails() -> None:
    llm = ScriptedLLM([text_message("not json"), text_message({"findings": "nope"})])
    run = await run_agent(SPEC, "review", llm, Budget(1.0))
    assert run.output is None
    assert run.notes[-1].kind == "validation"
    assert "did not match the required JSON schema" in llm.calls[1].messages[-1]["content"]


async def test_invalid_output_recovers_on_retry() -> None:
    llm = ScriptedLLM([text_message("{}"), text_message(EMPTY_REPORT)])
    run = await run_agent(SPEC, "review", llm, Budget(1.0))
    assert run.output is not None and len(run.usage) == 2


async def test_refusal_is_recorded_not_raised() -> None:
    run = await run_agent(SPEC, "review", ScriptedLLM([refusal_message("cyber")]), Budget(1.0))
    assert run.output is None and run.notes[0].kind == "refusal" and "cyber" in run.notes[0].message


async def test_budget_stops_the_agent_before_calling_llm() -> None:
    llm = ScriptedLLM([])
    run = await run_agent(SPEC, "review", llm, Budget(max_cost_usd=0.01, spent_usd=0.02))
    assert run.output is None and run.notes[0].kind == "budget" and llm.calls == []
