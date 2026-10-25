"""Typed views of GitHub pull request data.

Every string that originates from the PR author (title, body, patch, file contents)
is untrusted. These models only describe shape; quarantining that text before it
reaches an LLM is the job of `prreview.safety` (Phase 4).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def validate_repo_name(value: str) -> str:
    """Reject owner/repo names that could alter a URL path (e.g. containing '/' or '..')."""
    if not _NAME_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"invalid GitHub owner/repo name: {value!r}")
    return value


def validate_sha(value: str) -> str:
    """Require a full 40-char lowercase commit SHA (immutable, safe as a cache key)."""
    if not _SHA_RE.fullmatch(value):
        raise ValueError(f"expected a full 40-character commit SHA, got {value!r}")
    return value


class PRRef(BaseModel):
    """Identifies one pull request."""

    owner: str
    repo: str
    number: int = Field(ge=1)

    @field_validator("owner", "repo")
    @classmethod
    def _check_name(cls, value: str) -> str:
        return validate_repo_name(value)


class PullRequest(BaseModel):
    """Pull request metadata. `title` and `body` are untrusted author input."""

    ref: PRRef
    title: str
    body: str
    author: str
    state: str
    draft: bool
    base_ref: str
    base_sha: str
    head_ref: str
    head_sha: str
    changed_files: int
    additions: int
    deletions: int
    html_url: str


class Hunk(BaseModel):
    """One `@@` section of a unified diff, with new-file line numbers of added lines."""

    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    added_lines: list[int]
    removed_lines: list[int]


FileStatus = Literal["added", "removed", "modified", "renamed", "copied", "changed", "unchanged"]


class ChangedFile(BaseModel):
    """A file touched by the PR. `patch` is None when GitHub omits it (binary or too large)."""

    path: str
    previous_path: str | None = None
    status: FileStatus
    additions: int
    deletions: int
    patch: str | None = None
    hunks: list[Hunk] = Field(default_factory=list)

    @property
    def commentable_lines(self) -> set[int]:
        """New-file lines GitHub accepts a RIGHT-side review comment on (added or context lines)."""
        lines: set[int] = set()
        for hunk in self.hunks:
            lines.update(range(hunk.new_start, hunk.new_start + hunk.new_lines))
        return lines


class ChangedFilesPage(BaseModel):
    """Result of listing changed files, with an explicit truncation flag."""

    files: list[ChangedFile]
    truncated: bool
    files_fetched: int


class FileContent(BaseModel):
    """Text content of one file at one commit."""

    path: str
    ref: str
    size: int
    content: str


class PostedComment(BaseModel):
    """A review comment GitHub accepted."""

    id: int
    html_url: str
    path: str
    line: int | None
