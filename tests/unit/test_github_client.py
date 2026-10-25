"""Tests for the GitHub client using httpx.MockTransport (no real network)."""

from __future__ import annotations

import base64
from collections.abc import Callable

import httpx
import pytest

from prreview.config import GitHubSettings
from prreview.github.client import (
    GitHubAPIError,
    GitHubClient,
    GitHubRateLimitError,
    GitHubRetryableError,
    PayloadTooLargeError,
)
from prreview.schemas.github import PRRef
from tests.conftest import SHA_BASE, SHA_HEAD

REF = PRRef(owner="octo", repo="demo", number=7)

PR_JSON = {
    "title": "Fix parser",
    "body": None,
    "user": {"login": "alice"},
    "state": "open",
    "draft": False,
    "base": {"ref": "main", "sha": SHA_BASE},
    "head": {"ref": "fix", "sha": SHA_HEAD},
    "changed_files": 1,
    "additions": 2,
    "deletions": 1,
    "html_url": "https://github.com/octo/demo/pull/7",
}


class Recorder:
    """A MockTransport handler that records requests and replays scripted responses."""

    def __init__(self, *responses: httpx.Response | Callable[[httpx.Request], httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        return response(request) if callable(response) else response


def client_for(settings: GitHubSettings, recorder: Recorder, token: str | None = "t0k") -> GitHubClient:
    return GitHubClient(settings, token, transport=httpx.MockTransport(recorder))


async def test_get_pull_request_maps_fields_and_sends_headers(github_settings: GitHubSettings) -> None:
    recorder = Recorder(httpx.Response(200, json=PR_JSON))
    pr = await client_for(github_settings, recorder).get_pull_request(REF)
    assert (pr.head_sha, pr.base_sha, pr.body, pr.author) == (SHA_HEAD, SHA_BASE, "", "alice")
    sent = recorder.requests[0]
    assert sent.url.path == "/repos/octo/demo/pulls/7"
    assert sent.headers["Authorization"] == "Bearer t0k"
    assert sent.headers["X-GitHub-Api-Version"] == github_settings.github_api_version


async def test_list_changed_files_paginates_and_parses_hunks(github_settings: GitHubSettings) -> None:
    item = {"filename": "a.py", "status": "modified", "additions": 1, "deletions": 0, "patch": "@@ -1 +1,2 @@\n x\n+y"}
    recorder = Recorder(httpx.Response(200, json=[item] * 100), httpx.Response(200, json=[item] * 3))
    page = await client_for(github_settings, recorder).list_changed_files(REF, max_files=500)
    assert len(page.files) == 103 and not page.truncated
    assert page.files[0].hunks[0].added_lines == [2]
    assert [r.url.params["page"] for r in recorder.requests] == ["1", "2"]


async def test_list_changed_files_flags_truncation(github_settings: GitHubSettings) -> None:
    item = {"filename": "a.py", "status": "added", "additions": 1, "deletions": 0}
    recorder = Recorder(httpx.Response(200, json=[item] * 100))
    page = await client_for(github_settings, recorder).list_changed_files(REF, max_files=50)
    assert len(page.files) == 50 and page.truncated
    assert page.files[0].patch is None and page.files[0].hunks == []


async def test_retries_transient_errors_then_succeeds(github_settings: GitHubSettings) -> None:
    recorder = Recorder(httpx.Response(503), httpx.Response(502), httpx.Response(200, json=PR_JSON))
    pr = await client_for(github_settings, recorder).get_pull_request(REF)
    assert pr.title == "Fix parser" and len(recorder.requests) == 3


async def test_gives_up_after_max_attempts(github_settings: GitHubSettings) -> None:
    recorder = Recorder(*[httpx.Response(503) for _ in range(3)])
    with pytest.raises(GitHubRetryableError):
        await client_for(github_settings, recorder).get_pull_request(REF)
    assert len(recorder.requests) == github_settings.github_max_attempts


async def test_retries_transport_errors(github_settings: GitHubSettings) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    recorder = Recorder(boom, httpx.Response(200, json=PR_JSON))
    await client_for(github_settings, recorder).get_pull_request(REF)
    assert len(recorder.requests) == 2


async def test_does_not_retry_not_found(github_settings: GitHubSettings) -> None:
    recorder = Recorder(httpx.Response(404, json={"message": "Not Found"}))
    with pytest.raises(GitHubAPIError, match="404: Not Found"):
        await client_for(github_settings, recorder).get_pull_request(REF)
    assert len(recorder.requests) == 1


async def test_exhausted_rate_limit_fails_fast(github_settings: GitHubSettings) -> None:
    headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1760000000"}
    recorder = Recorder(httpx.Response(403, headers=headers, json={"message": "API rate limit exceeded"}))
    with pytest.raises(GitHubRateLimitError, match="1760000000"):
        await client_for(github_settings, recorder).get_pull_request(REF)
    assert len(recorder.requests) == 1


async def test_posting_a_comment_is_never_retried(github_settings: GitHubSettings) -> None:
    recorder = Recorder(httpx.Response(502), httpx.Response(201, json={}))
    with pytest.raises(GitHubRetryableError):
        await client_for(github_settings, recorder).create_review_comment(REF, SHA_HEAD, "a.py", 3, "nit")
    assert len(recorder.requests) == 1


async def test_posts_comment_payload(github_settings: GitHubSettings) -> None:
    created = {"id": 99, "html_url": "https://github.com/x", "path": "a.py", "line": 5}
    recorder = Recorder(httpx.Response(201, json=created))
    posted = await client_for(github_settings, recorder).create_review_comment(REF, SHA_HEAD, "a.py", 5, "bug", start_line=3)
    assert posted.id == 99
    assert recorder.requests[0].method == "POST"
    body = recorder.requests[0].read()
    assert b'"start_line":3' in body and b'"start_side":"RIGHT"' in body and b'"line":5' in body


async def test_file_content_decodes_base64(github_settings: GitHubSettings) -> None:
    payload = {"type": "file", "encoding": "base64", "size": 3, "content": base64.b64encode(b"hi\n").decode()}
    recorder = Recorder(httpx.Response(200, json=payload))
    content = await client_for(github_settings, recorder).get_file_content("octo", "demo", "dir/a b.py", SHA_HEAD)
    assert content.content == "hi\n"
    assert recorder.requests[0].url.raw_path.startswith(b"/repos/octo/demo/contents/dir/a%20b.py")


async def test_file_content_over_one_megabyte_is_reported(github_settings: GitHubSettings) -> None:
    recorder = Recorder(httpx.Response(200, json={"type": "file", "encoding": "none", "size": 2_000_000, "content": ""}))
    with pytest.raises(PayloadTooLargeError):
        await client_for(github_settings, recorder).get_file_content("octo", "demo", "big.txt", SHA_HEAD)


@pytest.mark.parametrize("path", ["../secrets", "/etc/passwd", "a//b", "a/./b"])
async def test_rejects_unsafe_paths_without_calling_github(github_settings: GitHubSettings, path: str) -> None:
    recorder = Recorder()
    with pytest.raises(ValueError):
        await client_for(github_settings, recorder).get_file_content("octo", "demo", path, SHA_HEAD)
    assert recorder.requests == []


async def test_rejects_short_sha(github_settings: GitHubSettings) -> None:
    with pytest.raises(ValueError, match="40-character"):
        await client_for(github_settings, Recorder()).get_file_content("octo", "demo", "a.py", "main")


async def test_tarball_download_is_capped(github_settings: GitHubSettings) -> None:
    recorder = Recorder(httpx.Response(200, content=b"x" * 5000))
    with pytest.raises(PayloadTooLargeError):
        await client_for(github_settings, recorder).download_tarball("octo", "demo", SHA_HEAD, max_bytes=1000)


async def test_tarball_follows_redirect(github_settings: GitHubSettings) -> None:
    recorder = Recorder(
        httpx.Response(302, headers={"location": "https://codeload.github.com/octo/demo/tar.gz/x"}),
        httpx.Response(200, content=b"tar-bytes"),
    )
    data = await client_for(github_settings, recorder).download_tarball("octo", "demo", SHA_HEAD, max_bytes=1000)
    assert data == b"tar-bytes"
    assert recorder.requests[1].url.host == "codeload.github.com"
