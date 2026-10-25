"""Results of running code inside the Docker sandbox."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CommandResult(BaseModel):
    """Outcome of one command executed in the sandbox."""

    exit_code: int | None
    timed_out: bool
    killed_by_signal_9: bool = Field(
        description="Exit code 137 without our timeout: almost always the memory limit (OOM)."
    )
    duration_s: float
    output: str = Field(description="Combined stdout+stderr, untrusted, possibly truncated.")
    output_truncated: bool


class TestCaseResult(BaseModel):
    """One test case parsed from pytest's JUnit XML report."""

    test_id: str
    outcome: Literal["passed", "failed", "error", "skipped"]
    duration_s: float
    message: str = ""


class DependencyInstall(BaseModel):
    """What the install stage tried to install and whether it worked."""

    requested: list[str]
    skipped_lines: list[str]
    succeeded: bool
    log_tail: str


TestRunStatus = Literal["passed", "failed", "error", "timeout", "install_failed", "no_tests"]


class TestRunResult(BaseModel):
    """Outcome of running the repository's test suite at one commit."""

    status: TestRunStatus
    install: DependencyInstall
    command: CommandResult | None
    cases: list[TestCaseResult]
    notes: list[str] = Field(default_factory=list)

    @property
    def failing(self) -> list[TestCaseResult]:
        """Test cases that failed or errored."""
        return [c for c in self.cases if c.outcome in ("failed", "error")]
