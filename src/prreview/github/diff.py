"""Parse the per-file `patch` strings GitHub returns into hunks with line numbers.

GitHub's `patch` field is a unified diff body without the `---`/`+++` headers,
starting directly at the first `@@` line.
"""

from __future__ import annotations

import difflib
import re

from prreview.schemas.github import Hunk

HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class DiffParseError(ValueError):
    """The patch text is not a well-formed unified diff."""


def parse_patch(patch: str) -> list[Hunk]:
    """Split a unified-diff patch into hunks, recording new-file added and old-file removed lines."""
    hunks: list[Hunk] = []
    current: _HunkBuilder | None = None
    for raw_line in patch.splitlines():
        header = HUNK_HEADER.match(raw_line)
        if header:
            if current:
                hunks.append(current.build())
            current = _HunkBuilder.from_header(header)
        elif current is None:
            raise DiffParseError(f"content before first hunk header: {raw_line[:80]!r}")
        else:
            current.consume(raw_line)
    if current:
        hunks.append(current.build())
    return hunks


def new_side_lines(patch: str) -> dict[int, str]:
    """Map new-file line numbers to their text for every added or context line in the patch."""
    lines: dict[int, str] = {}
    new_line = 0
    for raw_line in patch.splitlines():
        header = HUNK_HEADER.match(raw_line)
        if header:
            new_line = int(header.group(3))
        elif raw_line.startswith("-") or raw_line.startswith("\\"):
            continue
        elif new_line:
            lines[new_line] = raw_line[1:] if raw_line[:1] in ("+", " ") else raw_line
            new_line += 1
    return lines


class _HunkBuilder:
    """Walks one hunk's body while tracking old/new line counters."""

    def __init__(self, old_start: int, old_lines: int, new_start: int, new_lines: int) -> None:
        self.old_start, self.old_lines = old_start, old_lines
        self.new_start, self.new_lines = new_start, new_lines
        self._old_line, self._new_line = old_start, new_start
        self.added: list[int] = []
        self.removed: list[int] = []

    @classmethod
    def from_header(cls, match: re.Match[str]) -> _HunkBuilder:
        old_start, old_len, new_start, new_len = match.groups()
        # A missing length means 1 (unified diff format).
        return cls(int(old_start), int(old_len or 1), int(new_start), int(new_len or 1))

    def consume(self, line: str) -> None:
        if line.startswith("+"):
            self.added.append(self._new_line)
            self._new_line += 1
        elif line.startswith("-"):
            self.removed.append(self._old_line)
            self._old_line += 1
        elif line.startswith("\\"):
            return  # "\ No newline at end of file"
        else:
            # Context line. GitHub sometimes strips the leading space from empty context lines.
            self._old_line += 1
            self._new_line += 1

    def build(self) -> Hunk:
        return Hunk(
            old_start=self.old_start,
            old_lines=self.old_lines,
            new_start=self.new_start,
            new_lines=self.new_lines,
            added_lines=self.added,
            removed_lines=self.removed,
        )


def unified_patch(old_text: str, new_text: str, context: int = 3) -> str:
    """Produce a patch in GitHub's `patch` format (unified diff body without ---/+++ headers)."""
    diff = difflib.unified_diff(old_text.splitlines(), new_text.splitlines(), lineterm="", n=context)
    return "\n".join(line for line in diff if not line.startswith(("---", "+++")))
