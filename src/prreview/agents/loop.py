"""The bounded tool-use loop every LLM agent runs (D1, D4, D11).

Why our own loop instead of the SDK's beta tool runner: the steps-per-review metric, the
per-agent step cap, the shared cost budget, and turning every failure into an `AgentNote`
all need to happen between calls, and they are easier to audit in ~100 explicit lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from prreview.agents.toolbox import ToolBox
from prreview.llm.client import LLMCall, LLMClient, LLMResult
from prreview.observability.tracing import NoopTracer, Tracer
from prreview.schemas.review import AgentName, AgentNote, StepRecord, UsageRecord

T = TypeVar("T", bound=BaseModel)

FINAL_TURN_INSTRUCTION = (
    "You have used your tool budget. Do not call tools. Reply now with the final JSON report "
    "based on what you have seen."
)


@dataclass
class Budget:
    """Spend limit shared by every agent in one review (D11)."""

    max_cost_usd: float
    spent_usd: float = 0.0

    @property
    def exhausted(self) -> bool:
        """True once spending has reached the limit."""
        return self.spent_usd >= self.max_cost_usd


@dataclass(frozen=True)
class AgentSpec(Generic[T]):
    """Static description of one agent."""

    name: AgentName
    model: str
    system: str
    output_model: type[T]
    max_steps: int


@dataclass
class AgentRun(Generic[T]):
    """Result of one agent run. `output` is None when the agent could not finish."""

    output: T | None = None
    notes: list[AgentNote] = field(default_factory=list)
    usage: list[UsageRecord] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)


async def run_agent(
    spec: AgentSpec[T], prompt: str, llm: LLMClient, budget: Budget, toolbox: ToolBox | None = None,
    tracer: Tracer | None = None,
) -> AgentRun[T]:
    """Run the loop until the agent returns a valid report, fails, or hits a limit."""
    tracer = tracer or NoopTracer()
    with tracer.span(spec.name, as_type="agent", metadata={"model": spec.model, "max_steps": spec.max_steps}) as span:
        run = await _loop(spec, prompt, llm, budget, toolbox, tracer)
        span.update(
            output=run.output.model_dump() if run.output else None,
            metadata={"steps": len(run.steps), "notes": [n.message for n in run.notes]},
            level="WARNING" if run.output is None else "DEFAULT",
        )
    return run


async def _loop(
    spec: AgentSpec[T], prompt: str, llm: LLMClient, budget: Budget, toolbox: ToolBox | None, tracer: Tracer
) -> AgentRun[T]:
    run: AgentRun[T] = AgentRun()
    tools = await toolbox.definitions() if toolbox else []
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    retried_validation = False
    while True:
        if budget.exhausted:
            return _fail(run, spec, "budget", f"review cost budget of ${budget.max_cost_usd} exhausted")
        final_turn = bool(tools) and len(run.steps) >= spec.max_steps - 1
        if final_turn and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list):
            messages[-1]["content"].append({"type": "text", "text": FINAL_TURN_INSTRUCTION})
        result = await _call_llm(spec, llm, messages, tools, final_turn, run, budget, tracer)
        if result is None:
            return run
        message = result.message
        if message.stop_reason == "refusal":
            category = message.stop_details.category if message.stop_details else None
            return _fail(run, spec, "refusal", f"model declined the request (category={category})")
        if message.stop_reason == "max_tokens":
            return _fail(run, spec, "validation", "response hit max_tokens before finishing")
        messages.append({"role": "assistant", "content": message.content})
        if message.stop_reason in ("tool_use", "pause_turn"):
            if message.stop_reason == "tool_use":
                messages.append({"role": "user", "content": await _run_tools(spec, message, toolbox, run, tracer)})
            continue
        text = "".join(block.text for block in message.content if block.type == "text")
        try:
            run.output = spec.output_model.model_validate_json(text)
            return run
        except ValidationError as exc:
            if retried_validation:
                return _fail(run, spec, "validation", f"output failed schema validation twice: {exc.error_count()} errors")
            retried_validation = True
            messages.append({"role": "user", "content": f"Your reply did not match the required JSON schema:\n{exc}\nReply again with JSON only."})


async def _call_llm(
    spec: AgentSpec[T], llm: LLMClient, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    final_turn: bool, run: AgentRun[T], budget: Budget, tracer: Tracer,
) -> LLMResult | None:
    # Snapshot: the loop keeps appending to `messages`, and anything that records this call (tests,
    # tracing) must see exactly what was sent.
    call = LLMCall(spec.model, spec.system, list(messages), spec.output_model, tools, "none" if final_turn else "auto")
    with tracer.span(f"{spec.name}.llm", as_type="generation", model=spec.model,
                     input=messages[-1]["content"], metadata={"turn": len(run.steps), "final_turn": final_turn}) as span:
        try:
            result = await llm.create(call)
        except anthropic.APITimeoutError as exc:
            span.update(level="ERROR", status_message=str(exc))
            _fail(run, spec, "timeout", f"LLM request timed out: {exc}")
            return None
        except anthropic.APIError as exc:
            span.update(level="ERROR", status_message=str(exc))
            _fail(run, spec, "tool_error", f"LLM request failed: {type(exc).__name__}: {exc}")
            return None
        span.update(
            output=[block.model_dump() for block in result.message.content],
            usage_details={"input": result.input_tokens, "output": result.output_tokens, "cache_read": result.cache_read_tokens},
            cost_details={"total": result.cost_usd},
            metadata={"stop_reason": result.message.stop_reason, "served_by": result.message.model, "latency_ms": result.latency_ms},
        )
    budget.spent_usd += result.cost_usd
    run.steps.append(StepRecord(agent=spec.name, kind="llm", name=spec.model))
    run.usage.append(UsageRecord(
        agent=spec.name, model=result.message.model or spec.model, input_tokens=result.input_tokens,
        output_tokens=result.output_tokens, cache_read_tokens=result.cache_read_tokens,
        cost_usd=result.cost_usd, latency_ms=result.latency_ms,
    ))
    return result


async def _run_tools(
    spec: AgentSpec[T], message: Any, toolbox: ToolBox | None, run: AgentRun[T], tracer: Tracer
) -> list[dict[str, Any]]:
    """Execute every tool_use block; all results go back in ONE user message (parallel tool use)."""
    results: list[dict[str, Any]] = []
    for block in (b for b in message.content if b.type == "tool_use"):
        if toolbox is None or len(run.steps) >= spec.max_steps - 1:
            content, is_error = "Step limit reached: no more tool calls. Answer with what you have.", True
        else:
            with tracer.span(f"tool.{block.name}", as_type="tool", input=dict(block.input)) as span:
                outcome = await toolbox.call(block.name, dict(block.input))
                span.update(output=outcome.content[:4000], level="ERROR" if outcome.is_error else "DEFAULT")
            run.steps.append(StepRecord(agent=spec.name, kind="tool", name=block.name))
            content, is_error = outcome.content, outcome.is_error
            if is_error:
                run.notes.append(AgentNote(agent=spec.name, kind="tool_error", message=f"{block.name}: {content[:300]}"))
        results.append({"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error})
    return results


def _fail(run: AgentRun[T], spec: AgentSpec[T], kind: str, message: str) -> AgentRun[T]:
    run.notes.append(AgentNote(agent=spec.name, kind=kind, message=message))
    return run
