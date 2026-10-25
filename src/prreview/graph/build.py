"""The review graph (D1): planner -> parallel specialists (Send fan-out) -> aggregator.

LangGraph 1.2.14 behaviour verified with a probe script before use:
- Pydantic state works; `Annotated[list, operator.add]` merges writes from parallel branches.
- `Send(node, payload)` fans out with a per-branch payload of a different type than the state.
- `ainvoke(..., config={"recursion_limit": n})` raises `GraphRecursionError` past n supersteps.
- `ainvoke` returns a plain dict, so callers re-validate it into `ReviewState`.

Nodes are closures over `ReviewDeps` (LLM client, MCP clients, budget, tracer) so the state holds
only serialisable review data.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from prreview.agents.aggregator import aggregate, description_injection_notes, draft_comments, injection_findings
from prreview.agents.deps import ReviewDeps
from prreview.agents.loop import AgentRun
from prreview.agents.planner import plan_review
from prreview.agents.reviewers import review_bugs, review_security
from prreview.agents.tests_agent import review_tests
from prreview.schemas.review import AgentNote, PRContext, ReviewState, ReviewTask

NODE_FOR_TASK = {"bug": "bug_reviewer", "security": "security_reviewer", "tests": "test_runner"}


class TaskInput(BaseModel):
    """Payload sent to one specialist branch."""

    context: PRContext
    task: ReviewTask


def build_graph(deps: ReviewDeps) -> Any:
    """Compile the review graph for one set of dependencies."""

    async def planner(state: ReviewState) -> dict[str, Any]:
        deps.events.emit("planner", "started", f"{len(state.context.files)} changed files")
        plan, run = await plan_review(state.context, deps)
        deps.events.emit("planner", "finished", ", ".join(f"{t.kind}({len(t.files)} files)" for t in plan.tasks))
        notes = run.notes + [AgentNote(agent="planner", kind="skipped", message=s) for s in plan.skipped]
        return {"plan": plan, "summary": plan.pr_summary, **_run_fields(run, notes)}

    def fan_out(state: ReviewState) -> list[Send]:
        return [Send(NODE_FOR_TASK[t.kind], TaskInput(context=state.context, task=t)) for t in state.plan.tasks]

    async def bug_reviewer(payload: TaskInput) -> dict[str, Any]:
        deps.events.emit("bug", "started", f"{len(payload.task.files)} files")
        findings, run = await review_bugs(payload.context, payload.task, deps)
        _emit_done(deps, "bug", findings, run)
        return {"findings": findings, **_run_fields(run)}

    async def security_reviewer(payload: TaskInput) -> dict[str, Any]:
        deps.events.emit("security", "started", f"{len(payload.task.files)} files")
        findings, run = await review_security(payload.context, payload.task, deps)
        _emit_done(deps, "security", findings, run)
        return {"findings": findings, **_run_fields(run)}

    async def test_runner(payload: TaskInput) -> dict[str, Any]:
        deps.events.emit("tests", "started", "running tests at base and head in the sandbox")
        findings, comparison, notes, run = await review_tests(payload.context, payload.task, deps)
        deps.events.emit("tests", "finished", f"{len(findings)} findings; " + "; ".join(n.message for n in notes)[:300])
        fields = _run_fields(run, notes) if run else {"notes": notes}
        return {"findings": findings, "test_comparison": comparison, **fields}

    async def aggregator(state: ReviewState) -> dict[str, Any]:
        flagged = injection_findings(state.context.files)
        findings = state.findings + flagged
        deps.events.emit("aggregator", "started", f"{len(findings)} findings to merge")
        final, summary, run = await aggregate(findings, deps)
        deps.events.emit("aggregator", "finished", f"{len(final)} draft comments")
        notes = description_injection_notes(state.context)
        update: dict[str, Any] = {"drafts": draft_comments(final), "summary": summary or state.summary, "findings": flagged}
        if run:
            return {**update, **_run_fields(run, run.notes + notes)}
        return {**update, "notes": notes}

    graph = StateGraph(ReviewState)
    graph.add_node("planner", planner)
    graph.add_node("bug_reviewer", bug_reviewer)
    graph.add_node("security_reviewer", security_reviewer)
    graph.add_node("test_runner", test_runner)
    graph.add_node("aggregator", aggregator)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", fan_out, list(NODE_FOR_TASK.values()))
    for node in NODE_FOR_TASK.values():
        graph.add_edge(node, "aggregator")
    graph.add_edge("aggregator", END)
    return graph.compile()


def _emit_done(deps: ReviewDeps, agent: str, findings: list[Any], run: AgentRun[Any]) -> None:
    problems = [n.kind for n in run.notes if n.kind not in ("info", "grounding")]
    suffix = f" (issues: {', '.join(problems)})" if problems else ""
    deps.events.emit(agent, "finished", f"{len(findings)} findings, {len(run.steps)} steps{suffix}")


def _run_fields(run: AgentRun[Any], notes: list[AgentNote] | None = None) -> dict[str, Any]:
    return {"notes": notes if notes is not None else run.notes, "usage": run.usage, "steps": run.steps}
