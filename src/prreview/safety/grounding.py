"""Grounding check for LLM findings (D4).

A finding survives only if it points at lines this PR changed (or shows as context) and
its `evidence` actually appears there. This removes hallucinated files/line numbers, and
GitHub only accepts review comments on lines that are part of the diff anyway.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from prreview.github.diff import new_side_lines
from prreview.schemas.github import ChangedFile
from prreview.schemas.review import AgentName, Finding, FindingDraft

MAX_RANGE_LINES = 50
EVIDENCE_WINDOW = 3
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class GroundingResult:
    """Either a validated finding or the reason it was rejected."""

    finding: Finding | None
    reason: str = ""


def ground(draft: FindingDraft, files: dict[str, ChangedFile], agent: AgentName) -> GroundingResult:
    """Validate one draft against the PR's changed files."""
    changed = files.get(draft.file)
    if changed is None or not changed.patch:
        return GroundingResult(None, f"{draft.file!r} is not a changed file with a diff")
    if draft.line_end < draft.line_start or draft.line_end - draft.line_start > MAX_RANGE_LINES:
        return GroundingResult(None, f"invalid line range {draft.line_start}-{draft.line_end}")
    commentable = changed.commentable_lines
    if draft.line_start not in commentable or draft.line_end not in commentable:
        return GroundingResult(None, f"lines {draft.line_start}-{draft.line_end} of {draft.file} are not in the diff")
    if not _evidence_matches(draft, new_side_lines(changed.patch)):
        return GroundingResult(None, f"evidence not found near {draft.file}:{draft.line_start}")
    return GroundingResult(Finding(**draft.model_dump(), agent=agent))


def ground_all(
    drafts: list[FindingDraft], files: dict[str, ChangedFile], agent: AgentName
) -> tuple[list[Finding], list[str]]:
    """Validate many drafts. Returns (accepted findings, rejection reasons)."""
    accepted: list[Finding] = []
    rejected: list[str] = []
    for draft in drafts:
        result = ground(draft, files, agent)
        if result.finding:
            accepted.append(result.finding)
        else:
            rejected.append(f"{draft.title[:60]!r}: {result.reason}")
    return accepted, rejected


def _evidence_matches(draft: FindingDraft, lines: dict[int, str]) -> bool:
    evidence = _normalise(draft.evidence)
    if len(evidence) < 3:
        return False
    window = range(draft.line_start - EVIDENCE_WINDOW, draft.line_end + EVIDENCE_WINDOW + 1)
    nearby = _normalise(" ".join(lines[n] for n in window if n in lines))
    return evidence in nearby


def _normalise(text: str) -> str:
    return _WS.sub(" ", text).strip()
