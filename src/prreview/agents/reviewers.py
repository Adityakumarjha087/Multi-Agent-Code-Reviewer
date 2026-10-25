"""Bug and security reviewers: LLM + read-only repo tools, outputs grounded before acceptance."""

from __future__ import annotations

import httpx

from prreview.agents.context import render_files, render_pr_header
from prreview.agents.deps import ReviewDeps
from prreview.agents.loop import AgentRun, AgentSpec, run_agent
from prreview.agents.prompts import BUG, SECURITY, system_prompt
from prreview.safety.grounding import ground_all
from prreview.safety.untrusted import fence
from prreview.scanners import Candidate, scan_dependencies, scan_secrets
from prreview.schemas.review import AgentName, AgentNote, Finding, FindingsReport, PRContext, ReviewTask


async def review_bugs(ctx: PRContext, task: ReviewTask, deps: ReviewDeps) -> tuple[list[Finding], AgentRun[FindingsReport]]:
    """Run the bug reviewer on one batch of files."""
    spec = AgentSpec("bug", deps.llm_settings.bug_model, system_prompt(BUG, deps.nonce), FindingsReport,
                     deps.limits.agent_max_steps)
    prompt = _task_prompt(ctx, task, deps)
    run = await run_agent(spec, prompt, deps.llm, deps.budget, deps.search_toolbox(ctx.pr), tracer=deps.tracer)
    return _accept(ctx, run, "bug"), run


async def review_security(ctx: PRContext, task: ReviewTask, deps: ReviewDeps) -> tuple[list[Finding], AgentRun[FindingsReport]]:
    """Run deterministic scanners, then the security reviewer with the weaker hits as candidates."""
    files = [f for f in ctx.files if f.path in set(task.files)]
    scanner_findings, candidates = scan_secrets(files)
    notes: list[AgentNote] = []
    try:
        scanner_findings += await scan_dependencies(files, deps.http)
    except httpx.HTTPError as exc:
        notes.append(AgentNote(agent="security", kind="tool_error", message=f"OSV dependency check failed: {exc}"))
    spec = AgentSpec("security", deps.llm_settings.security_model, system_prompt(SECURITY, deps.nonce),
                     FindingsReport, deps.limits.agent_max_steps)
    prompt = _task_prompt(ctx, task, deps) + _candidates_section(candidates, deps.nonce)
    run = await run_agent(spec, prompt, deps.llm, deps.budget, deps.search_toolbox(ctx.pr), tracer=deps.tracer)
    run.notes = notes + run.notes
    return scanner_findings + _accept(ctx, run, "security"), run


def _accept(ctx: PRContext, run: AgentRun[FindingsReport], agent: AgentName) -> list[Finding]:
    """Ground the LLM's drafts; every rejection becomes a note (also feeds a benchmark metric)."""
    if run.output is None:
        return []
    accepted, rejected = ground_all(run.output.findings, {f.path: f for f in ctx.files}, agent)
    run.notes += [AgentNote(agent=agent, kind="grounding", message=f"discarded {r}") for r in rejected]
    if run.output.notes:
        run.notes.append(AgentNote(agent=agent, kind="info", message=run.output.notes[:500]))
    return accepted


def _task_prompt(ctx: PRContext, task: ReviewTask, deps: ReviewDeps) -> str:
    files = [f for f in ctx.files if f.path in set(task.files)]
    diffs = render_files(files, deps.nonce, deps.limits.max_patch_chars_per_file, task.focus)
    return (
        f"{render_pr_header(ctx.pr, deps.nonce)}\n\n"
        f"Review these {len(files)} changed files. Tools read the repository at the PR head commit.\n\n{diffs}"
    )


def _candidates_section(candidates: list[Candidate], nonce: str) -> str:
    if not candidates:
        return "\n\nScanner candidates: none."
    listing = "\n".join(f"- {c.file}:{c.line} [{c.kind}] {c.text}" for c in candidates)
    return f"\n\nScanner candidates to confirm or dismiss:\n{fence(listing, 'scanner_candidates', nonce)}"
