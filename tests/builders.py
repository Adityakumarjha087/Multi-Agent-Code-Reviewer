"""Builders for PR objects used across tests."""

from __future__ import annotations

from prreview.github.diff import parse_patch, unified_patch
from prreview.schemas.github import ChangedFile, PRRef, PullRequest
from prreview.schemas.review import PRContext
from tests.conftest import SHA_BASE, SHA_HEAD


def changed_file(path: str, old: str, new: str) -> ChangedFile:
    """A modified file whose patch is computed from before/after text."""
    patch = unified_patch(old, new)
    hunks = parse_patch(patch)
    return ChangedFile(
        path=path, status="added" if not old else "modified",
        additions=sum(len(h.added_lines) for h in hunks), deletions=sum(len(h.removed_lines) for h in hunks),
        patch=patch, hunks=hunks,
    )


def pull_request(title: str = "Add feature", body: str = "") -> PullRequest:
    """PR metadata for octo/demo#7."""
    return PullRequest(
        ref=PRRef(owner="octo", repo="demo", number=7), title=title, body=body, author="alice", state="open",
        draft=False, base_ref="main", base_sha=SHA_BASE, head_ref="feature", head_sha=SHA_HEAD,
        changed_files=1, additions=1, deletions=0, html_url="https://github.com/octo/demo/pull/7",
    )


def pr_context(files: list[ChangedFile], has_test_config: bool = False, body: str = "") -> PRContext:
    """A PRContext around the given files."""
    return PRContext(pr=pull_request(body=body), files=files, files_truncated=False, has_test_config=has_test_config)


CALC_OLD = "def divide(a, b):\n    if b == 0:\n        return None\n    return a / b\n"
CALC_NEW = "def divide(a, b):\n    return a / b\n\n\ndef mean(xs):\n    return sum(xs) / len(xs)\n"
