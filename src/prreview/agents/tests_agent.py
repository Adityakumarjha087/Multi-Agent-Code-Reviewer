"""Test agent (D6): run the suite at base and head in the sandbox, attribute only NEW failures.

The two sandbox runs are plain code, not LLM tool calls: which commits get tested must not be
something PR text can influence. The LLM is used only to link each new failure to a changed line.
"""

from __future__ import annotations

from mcp.shared.exceptions import MCPError
from pydantic import ValidationError

from prreview.agents.context import render_files, render_pr_header
from prreview.agents.deps import ReviewDeps
from prreview.agents.loop import AgentRun, AgentSpec, run_agent
from prreview.agents.prompts import TESTS, system_prompt
from prreview.safety.grounding import ground_all
from prreview.safety.untrusted import fence
from prreview.schemas.review import AgentNote, Finding, FindingsReport, PRContext, ReviewTask, TestComparison
from prreview.schemas.sandbox import TestRunResult

_UNUSABLE = ("install_failed", "timeout", "error")


class SandboxUnavailableError(RuntimeError):
    """The sandbox tool failed or returned something we cannot use."""


async def review_tests(
    ctx: PRContext, task: ReviewTask, deps: ReviewDeps
) -> tuple[list[Finding], TestComparison | None, list[AgentNote], AgentRun[FindingsReport] | None]:
    """Returns (findings, comparison, notes, llm run or None if no LLM was needed)."""
    if deps.mcp.sandbox is None:
        return [], None, [_note("skipped", "sandbox disabled in this deployment")], None
    try:
        base = await _run_tests(ctx.pr.base_sha, ctx, deps)
        head = await _run_tests(ctx.pr.head_sha, ctx, deps)
    except SandboxUnavailableError as exc:
        return [], None, [_note("tool_error", str(exc))], None
    comparison = compare(base, head)
    notes = _status_notes(base, head)
    if not comparison.newly_failing:
        return [], comparison, notes, None
    run = await run_agent(_spec(deps), _prompt(ctx, task, comparison, deps), deps.llm, deps.budget, deps.search_toolbox(ctx.pr), tracer=deps.tracer)
    findings: list[Finding] = []
    if run.output:
        findings, rejected = ground_all(run.output.findings, {f.path: f for f in ctx.files}, "tests")
        notes += [_note("grounding", f"discarded {r}") for r in rejected]
    unexplained = len(comparison.newly_failing) - len(findings)
    if unexplained > 0:
        names = ", ".join(c.test_id for c in comparison.newly_failing[:10])
        notes.append(_note("info", f"{len(comparison.newly_failing)} test(s) newly fail on the PR head ({names}); "
                                   f"{unexplained} could not be linked to a changed line"))
    return findings, comparison, notes, run


def compare(base: TestRunResult, head: TestRunResult) -> TestComparison:
    """A failure counts against the PR only if the same test did not already fail on base."""
    base_failing = {c.test_id for c in base.failing}
    newly = [c for c in head.failing if c.test_id not in base_failing]
    still = sorted(c.test_id for c in head.failing if c.test_id in base_failing)
    return TestComparison(base_status=base.status, head_status=head.status, newly_failing=newly, still_failing=still)


async def _run_tests(sha: str, ctx: PRContext, deps: ReviewDeps) -> TestRunResult:
    args = {"owner": ctx.pr.ref.owner, "repo": ctx.pr.ref.repo, "sha": sha}
    try:
        result = await deps.mcp.sandbox.call_tool("run_tests", args, read_timeout_seconds=deps.limits.sandbox_call_timeout_s)
    except MCPError as exc:
        raise SandboxUnavailableError(f"run_tests at {sha[:12]} failed: {exc.error.message}") from exc
    if result.is_error:
        text = " ".join(getattr(b, "text", "") for b in result.content)
        raise SandboxUnavailableError(f"run_tests at {sha[:12]} returned an error: {text[:300]}")
    try:
        return TestRunResult.model_validate(result.structured_content)
    except ValidationError as exc:
        raise SandboxUnavailableError(f"run_tests at {sha[:12]} returned an unexpected shape: {exc}") from exc


def _status_notes(base: TestRunResult, head: TestRunResult) -> list[AgentNote]:
    notes = []
    for label, run in (("base", base), ("head", head)):
        if run.status in _UNUSABLE:
            notes.append(_note("tool_error", f"tests on {label} ended with status '{run.status}'; results are incomplete"))
    return notes


def _spec(deps: ReviewDeps) -> AgentSpec[FindingsReport]:
    return AgentSpec("tests", deps.llm_settings.test_model, system_prompt(TESTS, deps.nonce), FindingsReport,
                     deps.limits.agent_max_steps)


def _prompt(ctx: PRContext, task: ReviewTask, comparison: TestComparison, deps: ReviewDeps) -> str:
    failures = "\n\n".join(f"{c.test_id}\n{c.message}" for c in comparison.newly_failing[:20])
    files = [f for f in ctx.files if f.path in set(task.files)]
    return (
        f"{render_pr_header(ctx.pr, deps.nonce)}\n\nTests failing on the PR head but passing on base:\n"
        f"{fence(failures, 'test_failures', deps.nonce)}\n\n"
        f"Changed files:\n{render_files(files, deps.nonce, deps.limits.max_patch_chars_per_file)}"
    )


def _note(kind: str, message: str) -> AgentNote:
    return AgentNote(agent="tests", kind=kind, message=message)
