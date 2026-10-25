"""Local cache of repository snapshots, keyed by immutable commit SHA."""

from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

import anyio

from prreview.config import WorkspaceSettings
from prreview.github.client import GitHubClient
from prreview.schemas.github import validate_repo_name, validate_sha
from prreview.workspace.archive import extract_github_tarball

_MB = 1024 * 1024


class WorkspaceStore:
    """Downloads `owner/repo@sha` once and serves the extracted directory afterwards.

    Completeness is signalled by an atomic rename from a temporary directory, so a
    crash mid-extraction never leaves a half-populated snapshot that looks valid.
    """

    def __init__(self, settings: WorkspaceSettings, github: GitHubClient) -> None:
        self._settings = settings
        self._github = github
        self._locks: dict[Path, asyncio.Lock] = {}

    def path_for(self, owner: str, repo: str, sha: str) -> Path:
        """Return where a snapshot lives (it may not exist yet)."""
        return self._settings.root / validate_repo_name(owner) / validate_repo_name(repo) / validate_sha(sha)

    async def get(self, owner: str, repo: str, sha: str) -> Path:
        """Return the snapshot directory, downloading and extracting it if needed."""
        target = self.path_for(owner, repo, sha)
        lock = self._locks.setdefault(target, asyncio.Lock())
        async with lock:
            if not target.is_dir():
                await self._populate(owner, repo, sha, target)
        return target

    async def _populate(self, owner: str, repo: str, sha: str, target: Path) -> None:
        data = await self._github.download_tarball(owner, repo, sha, self._settings.max_download_mb * _MB)
        staging = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
        try:
            await anyio.to_thread.run_sync(
                extract_github_tarball,
                data,
                staging,
                self._settings.max_files,
                self._settings.max_extracted_mb * _MB,
            )
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
