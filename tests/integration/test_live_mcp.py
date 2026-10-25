"""End-to-end through the MCP protocol against live GitHub (public repo, ~4 API calls) and real Docker.

Run with: pytest -m "network or docker" tests/integration/test_live_mcp.py
Uses GITHUB_READ_TOKEN from .env if set; otherwise the unauthenticated limit (60 requests/hour) applies.
"""

from __future__ import annotations

from pathlib import Path

import docker
import pytest
from mcp import Client

from prreview.config import GitHubSettings, SandboxSettings, WorkspaceSettings
from prreview.github.client import GitHubClient
from prreview.mcp_servers import github_read, sandbox_server, search_server
from prreview.sandbox.docker_sandbox import DockerSandbox
from prreview.workspace.store import WorkspaceStore

# psf/requests PR #6963: a small merged PR touching src/requests/utils.py and tests/test_utils.py.
LIVE_PR = {"owner": "psf", "repo": "requests", "number": 6963}


def live_github() -> GitHubClient:
    settings = GitHubSettings()
    token = settings.github_read_token.get_secret_value() if settings.github_read_token else None
    return GitHubClient(settings, token)


@pytest.mark.network
async def test_read_and_search_a_real_pull_request(workspace_settings: WorkspaceSettings, github_quota: None) -> None:
    github = live_github()
    async with Client(github_read.build_server(github)) as client:
        pr = (await client.call_tool("get_pull_request", LIVE_PR)).structured_content
        files = (await client.call_tool("list_changed_files", LIVE_PR)).structured_content
    assert len(pr["head_sha"]) == 40
    changed = {f["path"]: f for f in files["files"]}
    assert "src/requests/utils.py" in changed
    assert changed["src/requests/utils.py"]["hunks"], "patch should parse into hunks"

    repo = {"owner": "psf", "repo": "requests", "sha": pr["head_sha"]}
    async with Client(search_server.build_server(WorkspaceStore(workspace_settings, github))) as client:
        found = (await client.call_tool("search_code", {**repo, "query": "def get_netrc_auth("})).structured_content
    assert [m["path"] for m in found["matches"]] == ["src/requests/utils.py"]
    await github.aclose()


@pytest.mark.docker
async def test_sandbox_server_runs_tests_over_mcp(
    github_settings: GitHubSettings, workspace_settings: WorkspaceSettings, tmp_path: Path
) -> None:
    sha = "c" * 40
    store = WorkspaceStore(workspace_settings, GitHubClient(github_settings, None))
    snapshot = store.path_for("octo", "demo", sha)  # pre-populated, so no download happens
    (snapshot / "tests").mkdir(parents=True)
    (snapshot / "tests/test_ok.py").write_text("def test_ok():\n    assert True\n")
    server = sandbox_server.build_server(store, DockerSandbox(SandboxSettings(_env_file=None), docker.from_env()))
    args = {"owner": "octo", "repo": "demo", "sha": sha}
    async with Client(server) as client:
        result = await client.call_tool("run_tests", args)
        rejected = await client.call_tool("run_tests", {**args, "test_paths": ["--collect-only"]})
    assert not result.is_error, result.content
    assert result.structured_content["status"] == "passed"
    assert result.structured_content["cases"][0]["test_id"] == "tests.test_ok::test_ok"
    assert rejected.is_error and "InvalidTestPathError" in rejected.content[0].text
