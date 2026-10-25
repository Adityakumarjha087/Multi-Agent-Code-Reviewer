"""Shared test fixtures and builders."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from prreview.config import GitHubSettings, SandboxSettings, WorkspaceSettings

SHA_BASE = "a" * 40
SHA_HEAD = "b" * 40


def make_tarball(files: dict[str, bytes], top: str = "octo-demo-abc1234", gz: bool = True) -> bytes:
    """Build a GitHub-style tarball: every entry lives under one top-level folder."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz" if gz else "w") as tar:
        top_info = tarfile.TarInfo(top)
        top_info.type = tarfile.DIRTYPE
        top_info.mode = 0o755
        tar.addfile(top_info)
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture
def github_settings() -> GitHubSettings:
    """Settings that ignore the developer's `.env` and retry quickly."""
    return GitHubSettings(_env_file=None, github_retry_initial_wait_s=0.01, github_max_attempts=3)


@pytest.fixture
def workspace_settings(tmp_path: Path) -> WorkspaceSettings:
    """Workspace cache rooted in pytest's temporary directory."""
    return WorkspaceSettings(_env_file=None, root=tmp_path / "workspaces")


@pytest.fixture
def sandbox_settings() -> SandboxSettings:
    """Default sandbox limits, ignoring `.env`."""
    return SandboxSettings(_env_file=None)
