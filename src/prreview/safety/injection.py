"""Heuristic detection of prompt-injection attempts in PR content (D7, layer 5).

This is a tripwire, not a defence on its own: patterns are easy to evade. Its job is to
turn obvious attempts into visible findings, so an attack becomes evidence against the PR
instead of being silently obeyed or silently ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from prreview.github.diff import new_side_lines
from prreview.schemas.github import ChangedFile

_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "override_instructions": r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your|system)\b.{0,20}\b(instructions?|prompts?|rules|guidelines)\b",
        "approve_request": r"\b(approve|lgtm|merge)\b.{0,30}\b(this|the)\s+(pr|pull request|change|diff)\b",
        "suppress_findings": r"\b(do not|don't|never)\s+(report|flag|mention|comment on)\b",
        "addressed_to_ai": r"\b(note|message|instructions?)\s+(to|for)\s+(the\s+)?(ai|llm|assistant|review bot|claude|gpt)\b",
        "role_tag": r"</?\s*(system|assistant|user|instructions?)\s*>",
        "new_identity": r"\byou are now\b|\bas an ai (model|assistant|reviewer)\b",
        "mark_safe": r"\b(mark|treat|classify)\b.{0,30}\b(as\s+)?(safe|secure|approved|no issues)\b",
        "bracket_role_tag": r"\[\s*(system|assistant|instructions?)\s*\]",
        "addressed_to_reviewer_bot": r"\b(automated|ai|llm)\s+(code\s+)?(reviewers?|review bots?|assistants?)\b",
        "pre_approved": r"\bpre-?approved\b",
    }.items()
}


@dataclass(frozen=True)
class InjectionSignal:
    """One line that matched an injection pattern."""

    file: str
    line: int
    pattern: str
    text: str


def scan_text(text: str) -> list[str]:
    """Return the names of patterns that match `text`."""
    return [name for name, pattern in _PATTERNS.items() if pattern.search(text)]


def scan_added_lines(files: list[ChangedFile]) -> list[InjectionSignal]:
    """Scan only lines the PR adds; existing code is not the PR author's doing."""
    signals: list[InjectionSignal] = []
    for changed in files:
        if not changed.patch:
            continue
        added = {n for hunk in changed.hunks for n in hunk.added_lines}
        for line_no, text in new_side_lines(changed.patch).items():
            if line_no not in added:
                continue
            for name in scan_text(text):
                signals.append(InjectionSignal(changed.path, line_no, name, text.strip()[:200]))
    return signals
