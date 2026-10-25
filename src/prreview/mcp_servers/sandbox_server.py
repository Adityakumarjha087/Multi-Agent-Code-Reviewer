"""MCP server for sandboxed code execution. The only process with access to the Docker socket.

It exposes fixed operations. Callers choose *what* runs inside the container, never
*how* the container is configured (network, mounts, limits come from settings).
"""

from __future__ import annotations

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from prreview.config import github_settings, sandbox_settings, workspace_settings
from prreview.github.client import GitHubClient
from prreview.mcp_servers.common import serve, tool_errors
from prreview.sandbox.docker_sandbox import DockerSandbox
from prreview.schemas.sandbox import CommandResult, DependencyInstall, TestRunResult
from prreview.workspace.store import WorkspaceStore

EXECUTES_CODE = ToolAnnotations(read_only_hint=True, open_world_hint=False, idempotent_hint=True)


class CommandRunResult(BaseModel):
    """Result of `run_command`; `install` is None when dependencies were not requested."""

    install: DependencyInstall | None
    command: CommandResult


def build_server(store: WorkspaceStore, sandbox: DockerSandbox, max_concurrent_runs: int = 2) -> MCPServer:
    """Create the server. Concurrent sandbox runs are capped to protect the host."""
    server = MCPServer("prreview-sandbox", instructions="Runs repository code in an offline, resource-limited container.")
    limiter = anyio.CapacityLimiter(max_concurrent_runs)

    @server.tool(annotations=EXECUTES_CODE)
    async def run_tests(owner: str, repo: str, sha: str, test_paths: list[str] | None = None) -> TestRunResult:
        """Install the repo's declared Python deps (wheels only) and run pytest offline at `sha`."""
        with tool_errors():
            repo_dir = await store.get(owner, repo, sha)
            return await anyio.to_thread.run_sync(sandbox.run_tests, repo_dir, test_paths, limiter=limiter)

    @server.tool(annotations=EXECUTES_CODE)
    async def run_command(owner: str, repo: str, sha: str, argv: list[str], install_deps: bool = False) -> CommandRunResult:
        """Run `argv` offline in a fresh copy of the repo at `sha` (e.g. to reproduce a bug)."""
        with tool_errors():
            repo_dir = await store.get(owner, repo, sha)
            install, command = await anyio.to_thread.run_sync(
                sandbox.run_command, repo_dir, argv, install_deps, limiter=limiter
            )
            return CommandRunResult(install=install, command=command)

    return server


def main() -> None:
    """Entry point. Removes containers/volumes orphaned by a previous crash before serving."""
    gh_settings = github_settings()
    token = gh_settings.github_read_token.get_secret_value() if gh_settings.github_read_token else None
    store = WorkspaceStore(workspace_settings(), GitHubClient(gh_settings, token))
    sandbox = DockerSandbox(sandbox_settings())
    sandbox.remove_orphans()
    serve(build_server(store, sandbox), "prreview sandbox MCP server")


if __name__ == "__main__":
    main()
