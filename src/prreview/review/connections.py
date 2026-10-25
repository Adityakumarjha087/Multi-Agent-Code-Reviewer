"""Connect to the MCP servers a review needs, with least privilege.

Read and search servers run as stdio subprocesses. The MCP stdio client only passes the child a
small safe set of environment variables (HOME, PATH, ...; verified in mcp 2.3.0), and we add
exactly what each server needs. Each child's working directory is the workspace cache, which has
no `.env`, so it cannot pick up the API key or the GitHub write token.

The sandbox server is reached over Streamable HTTP when SANDBOX_MCP_URL is set (Docker Compose:
only that container mounts the Docker socket), or launched over stdio for local development.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from mcp import Client, StdioServerParameters
from pydantic_settings import BaseSettings, SettingsConfigDict

from prreview.agents.deps import McpClients
from prreview.config import GitHubSettings, WorkspaceSettings

_FORWARDED_PREFIXES = ("SANDBOX_", "WORKSPACE_")


class ConnectionSettings(BaseSettings):
    """How to reach the sandbox server."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    sandbox_mcp_url: str | None = None
    enable_sandbox: bool = True


def _server_params(module: str, github: GitHubSettings, workspace_root: Path, extra_env: dict[str, str]) -> StdioServerParameters:
    env = {k: v for k, v in os.environ.items() if k.startswith(_FORWARDED_PREFIXES)}
    env.update(extra_env)
    env["WORKSPACE_ROOT"] = str(workspace_root)
    if github.github_read_token:
        env["GITHUB_READ_TOKEN"] = github.github_read_token.get_secret_value()
    return StdioServerParameters(
        command=sys.executable, args=["-m", module, "--transport", "stdio"], env=env, cwd=str(workspace_root)
    )


@asynccontextmanager
async def connect(
    github: GitHubSettings, workspace: WorkspaceSettings, settings: ConnectionSettings | None = None
) -> AsyncIterator[McpClients]:
    """Open all MCP clients for one review and close them afterwards."""
    settings = settings or ConnectionSettings()
    root = workspace.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    async with AsyncExitStack() as stack:
        read = await stack.enter_async_context(Client(_server_params("prreview.mcp_servers.github_read", github, root, {})))
        search = await stack.enter_async_context(Client(_server_params("prreview.mcp_servers.search_server", github, root, {})))
        sandbox = None
        if settings.enable_sandbox:
            target = settings.sandbox_mcp_url or _server_params("prreview.mcp_servers.sandbox_server", github, root, {})
            sandbox = await stack.enter_async_context(Client(target))
        yield McpClients(github_read=read, search=search, sandbox=sandbox)
