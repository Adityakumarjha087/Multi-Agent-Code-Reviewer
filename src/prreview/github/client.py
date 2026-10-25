"""Thin async client for the GitHub REST endpoints this project needs.

Endpoints (checked against docs.github.com on 2026-10-09):
  GET  /repos/{owner}/{repo}/pulls/{n}
  GET  /repos/{owner}/{repo}/pulls/{n}/files       (per_page<=100, at most 3000 files)
  GET  /repos/{owner}/{repo}/contents/{path}?ref=  (base64 JSON, files up to 1 MB)
  GET  /repos/{owner}/{repo}/tarball/{ref}         (302 redirect to codeload)
  POST /repos/{owner}/{repo}/pulls/{n}/comments    (201 Created)
"""

from __future__ import annotations

import base64
import binascii
from typing import Any, Literal
from urllib.parse import quote

import httpx
from tenacity import AsyncRetrying

from prreview.config import GitHubSettings
from prreview.github.diff import parse_patch
from prreview.reliability.retry import RETRYABLE_STATUS_CODES, RetryableError, retrying
from prreview.schemas.github import (
    ChangedFile,
    ChangedFilesPage,
    FileContent,
    PostedComment,
    PRRef,
    PullRequest,
    validate_repo_name,
    validate_sha,
)

_PER_PAGE = 100
_GITHUB_FILES_HARD_CAP = 3000


class GitHubAPIError(Exception):
    """A non-2xx response from GitHub."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"GitHub API {status_code}: {message}")
        self.status_code = status_code


class GitHubRetryableError(GitHubAPIError, RetryableError):
    """A GitHub error worth retrying (5xx, 429 without an exhausted rate limit)."""


class GitHubRateLimitError(GitHubAPIError):
    """The rate limit is exhausted. Not retried: the reset can be up to an hour away."""


class PayloadTooLargeError(Exception):
    """A download exceeded its configured size cap."""


class GitHubClient:
    """Async GitHub REST client. Pass `transport` in tests to avoid real network calls."""

    def __init__(
        self,
        settings: GitHubSettings,
        token: str | None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": settings.github_api_version,
            "User-Agent": "prreview",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._settings = settings
        self._http = httpx.AsyncClient(
            base_url=settings.github_api_url,
            headers=headers,
            timeout=settings.github_timeout_s,
            transport=transport,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._http.aclose()

    async def get_pull_request(self, ref: PRRef) -> PullRequest:
        """Fetch PR metadata, including the base and head commit SHAs."""
        data = await self._request_json("GET", _pr_path(ref))
        return PullRequest(
            ref=ref,
            title=data["title"],
            body=data.get("body") or "",
            author=data["user"]["login"],
            state=data["state"],
            draft=bool(data.get("draft", False)),
            base_ref=data["base"]["ref"],
            base_sha=data["base"]["sha"],
            head_ref=data["head"]["ref"],
            head_sha=data["head"]["sha"],
            changed_files=data["changed_files"],
            additions=data["additions"],
            deletions=data["deletions"],
            html_url=data["html_url"],
        )

    async def list_changed_files(self, ref: PRRef, max_files: int | None = None) -> ChangedFilesPage:
        """List changed files with parsed hunks, stopping at `max_files` (flagged as truncated)."""
        limit = min(max_files or self._settings.github_max_files, _GITHUB_FILES_HARD_CAP)
        files: list[ChangedFile] = []
        page = 1
        more_pages = True
        while more_pages and len(files) < limit:
            batch = await self._request_json(
                "GET", f"{_pr_path(ref)}/files", params={"per_page": _PER_PAGE, "page": page}
            )
            files.extend(_to_changed_file(item) for item in batch)
            more_pages = len(batch) == _PER_PAGE
            page += 1
        # Conservative: if we stopped at the limit right after a full page we cannot know whether
        # more files exist without another request, so we report truncation.
        truncated = len(files) > limit or more_pages
        return ChangedFilesPage(files=files[:limit], truncated=truncated, files_fetched=len(files))

    async def get_file_content(self, owner: str, repo: str, path: str, sha: str) -> FileContent:
        """Fetch one UTF-8 text file at a commit via the contents API (files up to 1 MB)."""
        url = f"{_repo_path(owner, repo)}/contents/{_safe_repo_path(path)}"
        data = await self._request_json("GET", url, params={"ref": validate_sha(sha)})
        if isinstance(data, list) or data.get("type") != "file":
            raise GitHubAPIError(400, f"{path!r} is not a regular file")
        if data.get("encoding") != "base64":
            raise PayloadTooLargeError(f"{path!r} is too large for the contents API (size={data.get('size')})")
        return FileContent(path=path, ref=sha, size=data["size"], content=_decode_text(data["content"], path))

    async def download_tarball(self, owner: str, repo: str, sha: str, max_bytes: int) -> bytes:
        """Download the repository tarball at `sha`, aborting once it exceeds `max_bytes`."""
        url = f"{_repo_path(owner, repo)}/tarball/{validate_sha(sha)}"
        async for attempt in self._retrying(self._settings.github_max_attempts):
            with attempt:
                return await self._stream_capped(url, max_bytes)
        raise AssertionError("unreachable: retrying() re-raises the last error")

    async def create_review_comment(
        self,
        ref: PRRef,
        commit_id: str,
        path: str,
        line: int,
        body: str,
        side: Literal["LEFT", "RIGHT"] = "RIGHT",
        start_line: int | None = None,
    ) -> PostedComment:
        """Post one review comment on a diff line. Not retried: a retry after a lost response could double-post."""
        payload: dict[str, Any] = {
            "body": body,
            "commit_id": validate_sha(commit_id),
            "path": path,
            "line": line,
            "side": side,
        }
        if start_line is not None:
            payload |= {"start_line": start_line, "start_side": side}
        data = await self._request_json("POST", f"{_pr_path(ref)}/comments", json=payload, retry=False)
        return PostedComment(id=data["id"], html_url=data["html_url"], path=data["path"], line=data.get("line"))

    async def _request_json(self, method: str, url: str, retry: bool = True, **kwargs: Any) -> Any:
        attempts = self._settings.github_max_attempts if retry else 1
        async for attempt in self._retrying(attempts):
            with attempt:
                response = await self._http.request(method, url, **kwargs)
                _raise_for_status(response)
                return response.json()
        raise AssertionError("unreachable: retrying() re-raises the last error")

    def _retrying(self, attempts: int) -> AsyncRetrying:
        return retrying(attempts, initial_wait_s=self._settings.github_retry_initial_wait_s)

    async def _stream_capped(self, url: str, max_bytes: int) -> bytes:
        chunks: list[bytes] = []
        received = 0
        async with self._http.stream("GET", url, follow_redirects=True) as response:
            if response.is_error:
                await response.aread()
                _raise_for_status(response)
            async for chunk in response.aiter_bytes():
                received += len(chunk)
                if received > max_bytes:
                    raise PayloadTooLargeError(f"download exceeded {max_bytes} bytes")
                chunks.append(chunk)
        return b"".join(chunks)


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    message = _error_message(response)
    status = response.status_code
    if status in (403, 429) and response.headers.get("x-ratelimit-remaining") == "0":
        reset = response.headers.get("x-ratelimit-reset", "unknown")
        raise GitHubRateLimitError(status, f"rate limit exhausted (resets at epoch {reset}): {message}")
    if status in RETRYABLE_STATUS_CODES:
        raise GitHubRetryableError(status, message)
    raise GitHubAPIError(status, message)


def _error_message(response: httpx.Response) -> str:
    try:
        return str(response.json().get("message", response.text[:200]))
    except ValueError:
        return response.text[:200]


def _to_changed_file(item: dict[str, Any]) -> ChangedFile:
    patch = item.get("patch")
    return ChangedFile(
        path=item["filename"],
        previous_path=item.get("previous_filename"),
        status=item["status"],
        additions=item["additions"],
        deletions=item["deletions"],
        patch=patch,
        hunks=parse_patch(patch) if patch else [],
    )


def _decode_text(content_b64: str, path: str) -> str:
    try:
        return base64.b64decode(content_b64).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise GitHubAPIError(415, f"{path!r} is not UTF-8 text") from exc


def _repo_path(owner: str, repo: str) -> str:
    return f"/repos/{validate_repo_name(owner)}/{validate_repo_name(repo)}"


def _pr_path(ref: PRRef) -> str:
    return f"{_repo_path(ref.owner, ref.repo)}/pulls/{ref.number}"


def _safe_repo_path(path: str) -> str:
    """URL-encode a repo-relative path, rejecting absolute paths and `..` segments."""
    parts = path.split("/")
    if not path or path.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"invalid repository path: {path!r}")
    return quote(path, safe="/")
