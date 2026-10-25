"""Typed review state shared by all agents (D2: agents exchange findings, never conversations).

`*Draft` models are what an LLM is asked to produce. They deliberately have no `approve`,
`verdict`, or `merge` field (D7): there is nothing a prompt injection could flip to approve a PR.
"""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from prreview.schemas.github import ChangedFile, PullRequest
from prreview.schemas.sandbox import TestCaseResult

AgentName = Literal["planner", "bug", "security", "tests", "aggregator", "system"]
Severity = Literal["critical", "high", "medium", "low", "info"]
Category = Literal[
    "logic_error",
    "edge_case",
    "error_handling",
    "concurrency",
    "performance",
    "hardcoded_secret",
    "injection",
    "unsafe_dependency",
    "insecure_config",
    "prompt_injection_attempt",
    "test_failure",
    "other",
]
SEVERITY_RANK: dict[str, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


class FindingDraft(BaseModel):
    """One issue as proposed by an LLM agent, before grounding validation."""

    category: Category
    severity: Severity
    confidence: float = Field(description="0.0 to 1.0", ge=0.0, le=1.0)
    file: str = Field(description="Repository-relative path of a file changed in this PR")
    line_start: int = Field(description="First line in the NEW version of the file", ge=1)
    line_end: int = Field(description="Last line in the NEW version of the file (>= line_start)", ge=1)
    title: str = Field(description="Short summary, max ~80 characters")
    explanation: str = Field(description="Why this is a problem and when it triggers")
    evidence: str = Field(description="A short snippet copied VERBATIM from the changed lines")
    suggested_fix: str = Field(default="", description="Concrete fix, may be empty")


class FindingsReport(BaseModel):
    """What the bug, security, and test agents return."""

    findings: list[FindingDraft]
    notes: str = Field(default="", description="Anything the reviewer could not check, or caveats")


class Finding(FindingDraft):
    """A finding that passed grounding validation."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent: AgentName


class AgentNote(BaseModel):
    """Something that went wrong or was skipped. Notes are shown in the final review (D11)."""

    agent: AgentName
    kind: Literal["tool_error", "timeout", "budget", "validation", "refusal", "skipped", "grounding", "info"]
    message: str


class UsageRecord(BaseModel):
    """Token usage and cost of one LLM call."""

    agent: AgentName
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cost_usd: float
    latency_ms: int


class StepRecord(BaseModel):
    """One step (LLM call or tool call), used for the steps-per-review metric."""

    agent: AgentName
    kind: Literal["llm", "tool"]
    name: str


class FilePriority(BaseModel):
    """Planner output for one file."""

    path: str
    focus: str = Field(description="What reviewers should look at in this file, max ~200 chars")


class PlannerOutput(BaseModel):
    """What the planner LLM returns: prioritisation only; mandatory tasks are added by code (D3)."""

    priorities: list[FilePriority]
    summary: str = Field(description="One-paragraph neutral summary of what the PR changes")


class ReviewTask(BaseModel):
    """One unit of work for a specialist agent."""

    kind: Literal["bug", "security", "tests"]
    files: list[str]
    focus: dict[str, str] = Field(default_factory=dict)


class ReviewPlan(BaseModel):
    """The tasks the graph will fan out to, plus anything skipped and why."""

    tasks: list[ReviewTask]
    skipped: list[str] = Field(default_factory=list)
    pr_summary: str = ""


class TestComparison(BaseModel):
    """Base vs head test results (D6). Only `newly_failing` is attributed to the PR."""

    base_status: str
    head_status: str
    newly_failing: list[TestCaseResult]
    still_failing: list[str]


class MergeGroups(BaseModel):
    """Aggregator LLM output: which findings describe the same underlying issue."""

    duplicate_groups: list[list[str]] = Field(
        description="Each inner list holds ids of findings that describe the SAME issue; omit singletons"
    )
    overall_summary: str = Field(description="2-4 sentence neutral summary of the review")


CommentStatus = Literal["pending", "approved", "rejected", "posted", "post_failed"]


class DraftComment(BaseModel):
    """A review comment awaiting human approval."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    finding_ids: list[str]
    file: str
    line: int
    start_line: int | None = None
    severity: Severity
    body: str
    status: CommentStatus = "pending"


class PRContext(BaseModel):
    """Everything fetched about the PR before any agent runs."""

    pr: PullRequest
    files: list[ChangedFile]
    files_truncated: bool
    has_test_config: bool


class ReviewState(BaseModel):
    """LangGraph state. Lists written by parallel agents use an append reducer."""

    review_id: str
    context: PRContext
    plan: ReviewPlan | None = None
    findings: Annotated[list[Finding], operator.add] = Field(default_factory=list)
    notes: Annotated[list[AgentNote], operator.add] = Field(default_factory=list)
    usage: Annotated[list[UsageRecord], operator.add] = Field(default_factory=list)
    steps: Annotated[list[StepRecord], operator.add] = Field(default_factory=list)
    test_comparison: TestComparison | None = None
    drafts: list[DraftComment] = Field(default_factory=list)
    summary: str = ""

    @property
    def total_cost_usd(self) -> float:
        """Sum of all LLM call costs so far."""
        return round(sum(u.cost_usd for u in self.usage), 6)
