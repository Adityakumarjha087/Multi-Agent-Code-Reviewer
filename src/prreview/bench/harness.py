"""Run benchmark cases through the real review pipeline.

What is real: the MCP servers (github_read, search, sandbox), the graph, the agents, the Docker
sandbox, and the LLM. What is replaced: GitHub's HTTP API is answered from the case fixture (an
`httpx.MockTransport`), and repository snapshots are written from the fixture instead of downloaded.
That keeps runs reproducible and needs no GitHub token or real PRs.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

import docker
import httpx
from mcp import Client

from prreview.agents.deps import McpClients, ReviewDeps
from prreview.agents.loop import Budget
from prreview.bench.dataset import Case
from prreview.bench.metrics import CaseRun
from prreview.config import AgentLimits, GitHubSettings, LLMSettings, SandboxSettings, WorkspaceSettings
from prreview.github.client import GitHubClient
from prreview.llm.client import LLMClient
from prreview.mcp_servers import github_read, sandbox_server, search_server
from prreview.observability.tracing import Tracer
from prreview.review.factory import single_error
from prreview.review.runner import run_review
from prreview.safety.untrusted import new_nonce
from prreview.sandbox.docker_sandbox import DockerSandbox
from prreview.schemas.github import PRRef
from prreview.workspace.store import WorkspaceStore

logger = logging.getLogger(__name__)
OWNER = "bench"


def fixture_github(case: Case) -> httpx.MockTransport:
    """Answer the two GitHub endpoints the pipeline uses from the case fixture."""
    files = case.changed_files()
    pr = {
        "title": case.title, "body": case.body, "user": {"login": "bench-author"}, "state": "open", "draft": False,
        "base": {"ref": "main", "sha": case.sha("base")}, "head": {"ref": "feature", "sha": case.sha("head")},
        "changed_files": len(files), "additions": sum(f.additions for f in files),
        "deletions": sum(f.deletions for f in files), "html_url": f"https://example.invalid/{case.id}/pull/1",
    }
    listing = [
        {"filename": f.path, "status": f.status, "additions": f.additions, "deletions": f.deletions, "patch": f.patch}
        for f in files
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        base = f"/repos/{OWNER}/{case.id}/pulls/1"
        if request.url.path == base:
            return httpx.Response(200, json=pr)
        if request.url.path == f"{base}/files":
            return httpx.Response(200, json=listing if request.url.params.get("page") == "1" else [])
        return httpx.Response(404, json={"message": f"benchmark fixture has no {request.url.path}"})

    return httpx.MockTransport(handler)


async def run_case(
    case: Case, trial: int, llm: LLMClient, work_root: Path, *, llm_settings: LLMSettings, limits: AgentLimits,
    tracer: Tracer, enable_sandbox: bool, http: httpx.AsyncClient,
) -> CaseRun:
    """Review one case once. Any failure is recorded on the run, never dropped."""
    run = CaseRun(case_id=case.id, kind=case.kind, trial=trial)
    try:
        outcome = await _review(case, trial, llm, work_root, llm_settings, limits, tracer, enable_sandbox, http)
    except Exception as exc:  # noqa: BLE001 - one broken case must not abort the whole benchmark; it is scored as an error
        cause = single_error(exc) if isinstance(exc, BaseExceptionGroup) else exc
        run.error = "".join(traceback.format_exception_only(cause)).strip()
        logger.error("case %s trial %d failed: %s", case.id, trial, run.error)
        return run
    state = outcome.state
    final_ids = {fid for draft in state.drafts for fid in draft.finding_ids}
    run.findings = [f for f in state.findings if f.id in final_ids]
    run.notes = [n.message for n in state.notes]
    run.summary = state.summary
    run.steps, run.cost_usd, run.latency_s = len(state.steps), state.total_cost_usd, outcome.latency_s
    run.grounding_rejections = sum(1 for n in state.notes if n.kind == "grounding")
    return run


async def _review(
    case: Case, trial: int, llm: LLMClient, work_root: Path, llm_settings: LLMSettings, limits: AgentLimits,
    tracer: Tracer, enable_sandbox: bool, http: httpx.AsyncClient,
) -> Any:
    github = GitHubClient(GitHubSettings(_env_file=None), None, transport=fixture_github(case))
    store = WorkspaceStore(WorkspaceSettings(_env_file=None, root=work_root / f"trial{trial}"), github)
    for side in ("base", "head"):
        snapshot = store.path_for(OWNER, case.id, case.sha(side))
        if not snapshot.exists():
            case.write_snapshot(side, snapshot)
            snapshot.mkdir(parents=True, exist_ok=True)
    sandbox = sandbox_server.build_server(store, DockerSandbox(SandboxSettings(), docker.from_env())) if enable_sandbox else None
    async with Client(github_read.build_server(github)) as read, Client(search_server.build_server(store)) as search:
        if sandbox is None:
            return await _run(case, llm, McpClients(read, search, None), llm_settings, limits, tracer, http)
        async with Client(sandbox) as sandbox_client:
            return await _run(case, llm, McpClients(read, search, sandbox_client), llm_settings, limits, tracer, http)


async def _run(case: Case, llm: LLMClient, clients: McpClients, llm_settings: LLMSettings, limits: AgentLimits,
               tracer: Tracer, http: httpx.AsyncClient) -> Any:
    deps = ReviewDeps(llm=llm, llm_settings=llm_settings, limits=limits, budget=Budget(limits.review_max_cost_usd),
                      nonce=new_nonce(), mcp=clients, http=http, tracer=tracer)
    return await run_review(PRRef(owner=OWNER, repo=case.id, number=1), f"bench-{case.id}", deps)
