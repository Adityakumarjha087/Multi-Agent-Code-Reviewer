"""MCP servers exercised through the real MCP client, in-process. GitHub is faked with MockTransport."""

from __future__ import annotations

from pathlib import Path

import httpx
from mcp import Client

from prreview.config import GitHubSettings, WorkspaceSettings
from prreview.github.client import GitHubClient
from prreview.mcp_servers import github_read, github_write, search_server
from prreview.workspace.store import WorkspaceStore
from tests.conftest import SHA_HEAD, make_tarball
from tests.unit.test_github_client import PR_JSON, Recorder


def fake_github(settings: GitHubSettings, *responses: httpx.Response) -> tuple[GitHubClient, Recorder]:
    recorder = Recorder(*responses)
    return GitHubClient(settings, "t0k", transport=httpx.MockTransport(recorder)), recorder


async def test_read_server_lists_only_read_tools(github_settings: GitHubSettings) -> None:
    github, _ = fake_github(github_settings)
    async with Client(github_read.build_server(github)) as client:
        tools = (await client.list_tools()).tools
    assert sorted(t.name for t in tools) == ["get_file_content", "get_pull_request", "list_changed_files"]
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)


async def test_read_server_returns_structured_pull_request(github_settings: GitHubSettings) -> None:
    github, _ = fake_github(github_settings, httpx.Response(200, json=PR_JSON))
    async with Client(github_read.build_server(github)) as client:
        result = await client.call_tool("get_pull_request", {"owner": "octo", "repo": "demo", "number": 7})
    assert not result.is_error
    assert result.structured_content["head_sha"] == SHA_HEAD


async def test_github_failure_becomes_tool_error_with_reason(github_settings: GitHubSettings) -> None:
    github, _ = fake_github(github_settings, httpx.Response(404, json={"message": "Not Found"}))
    async with Client(github_read.build_server(github)) as client:
        result = await client.call_tool("get_pull_request", {"owner": "octo", "repo": "demo", "number": 7})
    assert result.is_error
    assert "GitHubAPIError: GitHub API 404: Not Found" in result.content[0].text


async def test_invalid_owner_is_rejected_before_any_request(github_settings: GitHubSettings) -> None:
    github, recorder = fake_github(github_settings)
    async with Client(github_read.build_server(github)) as client:
        result = await client.call_tool("get_pull_request", {"owner": "../evil", "repo": "demo", "number": 7})
    assert result.is_error and recorder.requests == []


async def test_write_server_refuses_unapproved_comment(github_settings: GitHubSettings) -> None:
    github, recorder = fake_github(github_settings)
    async with Client(github_write.build_server(github, lambda comment_id: None)) as client:
        result = await client.call_tool("post_approved_comment", {"comment_id": "abc"})
    assert result.is_error and "NotApprovedError" in result.content[0].text
    assert recorder.requests == []


async def test_write_server_posts_the_stored_comment(github_settings: GitHubSettings) -> None:
    from prreview.persistence.repo import PostableComment

    stored = PostableComment("c1", "octo", "demo", 7, SHA_HEAD, "a.py", 5, 3, "**[HIGH] bug**")
    created = {"id": 99, "html_url": "https://github.com/x", "path": "a.py", "line": 5}
    github, recorder = fake_github(github_settings, httpx.Response(201, json=created))
    async with Client(github_write.build_server(github, {"c1": stored}.get)) as client:
        tools = [t.name for t in (await client.list_tools()).tools]
        result = await client.call_tool("post_approved_comment", {"comment_id": "c1"})
    assert tools == ["post_approved_comment"]  # no free-form posting tool exists
    assert result.structured_content["id"] == 99
    body = recorder.requests[0].read()
    assert b'"body":"**[HIGH] bug**"' in body and b'"start_line":3' in body


async def test_search_server_downloads_once_then_searches(
    github_settings: GitHubSettings, workspace_settings: WorkspaceSettings
) -> None:
    tarball = make_tarball({"app.py": b"API_KEY = 'abc'\n", "lib/util.py": b"def f():\n    return 1\n"})
    github, recorder = fake_github(github_settings, httpx.Response(200, content=tarball))
    server = search_server.build_server(WorkspaceStore(workspace_settings, github))
    repo = {"owner": "octo", "repo": "demo", "sha": SHA_HEAD}
    async with Client(server) as client:
        found = await client.call_tool("search_code", {**repo, "query": "API_KEY"})
        sliced = await client.call_tool("read_file", {**repo, "path": "lib/util.py", "start_line": 2})
        escaped = await client.call_tool("read_file", {**repo, "path": "../../etc/passwd"})
    assert found.structured_content["matches"] == [{"path": "app.py", "line": 1, "text": "API_KEY = 'abc'"}]
    assert sliced.structured_content["content"] == "    return 1"
    assert escaped.is_error and "PathOutsideRepoError" in escaped.content[0].text
    assert len(recorder.requests) == 1  # snapshot cached after the first call


async def test_workspace_store_leaves_no_partial_snapshot_on_failure(
    github_settings: GitHubSettings, workspace_settings: WorkspaceSettings
) -> None:
    github, _ = fake_github(github_settings, httpx.Response(200, content=b"corrupt"))
    store = WorkspaceStore(workspace_settings, github)
    async with Client(search_server.build_server(store)) as client:
        result = await client.call_tool("list_files", {"owner": "octo", "repo": "demo", "sha": SHA_HEAD})
    assert result.is_error and "ArchiveError" in result.content[0].text
    parent: Path = store.path_for("octo", "demo", SHA_HEAD).parent
    assert not parent.exists() or list(parent.iterdir()) == []
