"""Benchmark cases: one YAML file per PR with base/head file contents and human-reviewable labels.

A file whose `base` equals its `head` is part of the repository snapshot (e.g. existing tests)
but not of the PR diff. `base: ""` means the PR adds the file.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from prreview.github.diff import parse_patch, unified_patch
from prreview.schemas.github import ChangedFile
from prreview.schemas.review import Category

CaseKind = Literal["bug", "security", "clean", "injection"]


class CaseFile(BaseModel):
    """One file at base and head."""

    path: str
    base: str = ""
    head: str = ""


class Label(BaseModel):
    """One known issue. A finding matches if file and categories match and lines overlap (± tolerance)."""

    file: str
    lines: tuple[int, int]
    categories: list[Category]
    note: str


class Case(BaseModel):
    """One labelled pull request."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,60}$")
    kind: CaseKind
    title: str
    body: str = ""
    files: list[CaseFile]
    labels: list[Label] = Field(default_factory=list)
    forbidden_in_comments: list[str] = Field(default_factory=list, description="injection cases: text that must never appear in drafts")

    @model_validator(mode="after")
    def _check_kind(self) -> Case:
        if self.kind == "clean" and self.labels:
            raise ValueError(f"{self.id}: clean cases must have no labels")
        if self.kind != "clean" and not self.labels:
            raise ValueError(f"{self.id}: {self.kind} cases need at least one label")
        if self.kind == "injection" and not self.forbidden_in_comments:
            raise ValueError(f"{self.id}: injection cases must list forbidden_in_comments")
        return self

    def sha(self, side: Literal["base", "head"]) -> str:
        """Deterministic fake commit SHA for a snapshot."""
        return hashlib.sha1(f"{self.id}:{side}".encode()).hexdigest()

    def changed_files(self) -> list[ChangedFile]:
        """GitHub-style changed-file entries, computed from base/head contents."""
        changed: list[ChangedFile] = []
        for f in self.files:
            if f.base == f.head:
                continue
            patch = unified_patch(f.base, f.head)
            hunks = parse_patch(patch)
            status = "added" if not f.base else "removed" if not f.head else "modified"
            changed.append(ChangedFile(
                path=f.path, status=status, patch=patch, hunks=hunks,
                additions=sum(len(h.added_lines) for h in hunks), deletions=sum(len(h.removed_lines) for h in hunks),
            ))
        return changed

    def write_snapshot(self, side: Literal["base", "head"], root: Path) -> None:
        """Materialise the repository at base or head."""
        for f in self.files:
            text = f.base if side == "base" else f.head
            if text:
                target = root / f.path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)


# Fake credentials for secret-detection cases, assembled here so no secret-shaped literal sits in the
# repository (GitHub push protection / secret scanning would flag it). These are AWS's documented
# example values, not real keys.
FAKE_SECRETS = {
    "aws_access_key_id": "AKIA" + "IOSFODNN7" + "EXAMPLE",
    "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/" + "bPxRfiCYEXAMPLEKEY",
}


def render_placeholders(text: str) -> str:
    """Replace `{{fake:<name>}}` with the matching fake credential; unknown names are an error."""
    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in FAKE_SECRETS:
            raise ValueError(f"unknown placeholder {{{{fake:{name}}}}}")
        return FAKE_SECRETS[name]

    return re.sub(r"\{\{fake:([a-z_]+)\}\}", substitute, text)


def load_cases(directory: Path, pattern: str = "*.yaml") -> list[Case]:
    """Load and validate every case file, sorted by id."""
    cases = [Case.model_validate(yaml.safe_load(render_placeholders(p.read_text()))) for p in sorted(directory.glob(pattern))]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    return cases


def label_problems(case: Case) -> list[str]:
    """Labels that point outside the diff can never be matched (and could not be commented on)."""
    files = {f.path: f for f in case.changed_files()}
    problems = []
    for label in case.labels:
        changed = files.get(label.file)
        if changed is None:
            problems.append(f"{case.id}: {label.file} is not changed by the PR")
            continue
        start, end = label.lines
        missing = [n for n in range(start, end + 1) if n not in changed.commentable_lines]
        if start > end or missing:
            problems.append(f"{case.id}: {label.file}:{start}-{end} not fully inside the diff (missing {missing})")
    return problems
