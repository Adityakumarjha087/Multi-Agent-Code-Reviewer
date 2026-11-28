"""Publisher bookkeeping with a stub repository and a stub write server (no Postgres, no network)."""

from __future__ import annotations

from types import SimpleNamespace

from mcp.shared.exceptions import MCPError

from prreview.publisher import publish_approved


class StubRepo:
    """Hands out the queued comment ids once each and records what the publisher reports back."""

    def __init__(self, *ids: str) -> None:
        self._queue = [SimpleNamespace(id=i) for i in ids]
        self.posted: dict[str, tuple[int, str]] = {}
        self.failed: dict[str, str] = {}

    def claim_approved(self):
        return self._queue.pop(0) if self._queue else None

    def record_posted(self, comment_id: str, github_id: int, url: str) -> None:
        self.posted[comment_id] = (github_id, url)

    def record_post_failed(self, comment_id: str, error: str) -> None:
        self.failed[comment_id] = error


class StubServer:
    """Answers `call_tool` with the next scripted result, or raises it if it is an exception."""

    def __init__(self, *results: object) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict, read_timeout_seconds: float):
        self.calls.append((name, arguments))
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def ok(github_id: int, url: str):
    return SimpleNamespace(is_error=False, content=[], structured_content={"id": github_id, "html_url": url})


def error(*texts: str):
    return SimpleNamespace(is_error=True, content=[SimpleNamespace(text=t) for t in texts], structured_content=None)


async def test_nothing_approved_means_no_calls() -> None:
    repo, server = StubRepo(), StubServer()
    assert await publish_approved(repo, server) == 0
    assert server.calls == []


async def test_successful_post_records_github_id_and_url() -> None:
    repo, server = StubRepo("c1"), StubServer(ok(555, "https://github.com/o/r/pull/1#c555"))
    assert await publish_approved(repo, server) == 1
    assert server.calls == [("post_approved_comment", {"comment_id": "c1"})]
    assert repo.posted == {"c1": (555, "https://github.com/o/r/pull/1#c555")}
    assert repo.failed == {}


async def test_tool_error_is_recorded_and_the_next_comment_still_posts() -> None:
    repo = StubRepo("bad", "good")
    server = StubServer(error("GitHub API 422:", "line not in diff"), ok(7, "https://example.test/c7"))
    assert await publish_approved(repo, server) == 2
    assert repo.failed == {"bad": "GitHub API 422: line not in diff"}
    assert list(repo.posted) == ["good"]


async def test_transport_error_says_the_outcome_is_unknown() -> None:
    repo, server = StubRepo("c1"), StubServer(MCPError(-32000, "connection closed"))
    assert await publish_approved(repo, server) == 1
    message = repo.failed["c1"]
    assert "outcome unknown" in message and "connection closed" in message
    assert "check the PR before re-approving" in message
    assert repo.posted == {}
