"""Planner (D3): rules decide WHICH tasks run; the LLM only prioritises files and summarises.

Security review always runs and the test task depends only on repository files, so text in
the PR cannot remove a mandatory review.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from prreview.agents.context import render_file_list, render_pr_header
from prreview.agents.deps import ReviewDeps
from prreview.agents.loop import AgentRun, AgentSpec, run_agent
from prreview.agents.prompts import PLANNER, system_prompt
from prreview.schemas.github import ChangedFile
from prreview.schemas.review import PlannerOutput, PRContext, ReviewPlan, ReviewTask

CODE_SUFFIXES = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".kt", ".rb", ".rs", ".c", ".h",
    ".cc", ".cpp", ".hpp", ".cs", ".php", ".swift", ".scala", ".sh", ".sql",
})
BUG_BATCH_SIZE = 8


async def plan_review(ctx: PRContext, deps: ReviewDeps) -> tuple[ReviewPlan, AgentRun[PlannerOutput]]:
    """Build the task list. Works (without prioritisation) even if the planner LLM fails."""
    reviewable, skipped = _select_files(ctx, deps.limits.max_files_reviewed)
    run = await run_agent(_spec(deps), _prompt(ctx, deps), deps.llm, deps.budget, tracer=deps.tracer)
    paths = {f.path for f in reviewable}
    focus = {p.path: p.focus for p in run.output.priorities if p.path in paths} if run.output else {}
    rank = {path: i for i, path in enumerate(focus)}  # planner order; unranked files keep GitHub order after
    ordered = sorted(reviewable, key=lambda f: rank.get(f.path, len(rank)))
    tasks = _bug_tasks(ordered, focus) + [ReviewTask(kind="security", files=[f.path for f in ordered], focus=focus)]
    if ctx.has_test_config:
        tasks.append(ReviewTask(kind="tests", files=[f.path for f in ordered]))
    else:
        skipped.append("tests: no pytest configuration or tests/ directory found at the PR head")
    summary = run.output.summary if run.output else ""
    return ReviewPlan(tasks=tasks, skipped=skipped, pr_summary=summary), run


def is_code(path: str) -> bool:
    """True for source files the bug reviewer should read."""
    return PurePosixPath(path).suffix.lower() in CODE_SUFFIXES


def _select_files(ctx: PRContext, limit: int) -> tuple[list[ChangedFile], list[str]]:
    skipped: list[str] = []
    with_patch = [f for f in ctx.files if f.patch and f.status != "removed"]
    skipped += [f"{f.path}: no textual diff (binary, removed, or too large)" for f in ctx.files if f not in with_patch]
    if len(with_patch) > limit:
        skipped += [f"{f.path}: over the {limit}-file review limit" for f in with_patch[limit:]]
    if ctx.files_truncated:
        skipped.append("GitHub file list was truncated; later files were not fetched")
    return with_patch[:limit], skipped


def _bug_tasks(files: list[ChangedFile], focus: dict[str, str]) -> list[ReviewTask]:
    code = [f.path for f in files if is_code(f.path)]
    batches = [code[i : i + BUG_BATCH_SIZE] for i in range(0, len(code), BUG_BATCH_SIZE)]
    return [ReviewTask(kind="bug", files=batch, focus={p: focus[p] for p in batch if p in focus}) for batch in batches]


def _spec(deps: ReviewDeps) -> AgentSpec[PlannerOutput]:
    return AgentSpec("planner", deps.llm_settings.planner_model, system_prompt(PLANNER, deps.nonce),
                     PlannerOutput, max_steps=2)


def _prompt(ctx: PRContext, deps: ReviewDeps) -> str:
    return f"{render_pr_header(ctx.pr, deps.nonce)}\n\nChanged files:\n{render_file_list(ctx.files, deps.nonce)}"
