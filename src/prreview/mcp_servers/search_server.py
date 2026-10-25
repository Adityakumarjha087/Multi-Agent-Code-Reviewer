"""MCP server for read-only search over a repository snapshot at a commit."""

from __future__ import annotations

import anyio
from mcp.server.mcpserver import MCPServer

from prreview.config import github_settings, workspace_settings
from prreview.github.client import GitHubClient
from prreview.mcp_servers.common import READ_ONLY, serve, tool_errors
from prreview.schemas.search import FileListing, FileSlice, SearchResult
from prreview.search import code_search
from prreview.workspace.store import WorkspaceStore


def build_server(store: WorkspaceStore) -> MCPServer:
    """Create the server around a workspace store."""
    server = MCPServer("prreview-search", instructions="Search and read files of a repository at a commit SHA.")

    @server.tool(annotations=READ_ONLY)
    async def list_files(owner: str, repo: str, sha: str, glob: str = "*", max_results: int = 500) -> FileListing:
        """List file paths matching an fnmatch glob (e.g. `*.py`, `tests/*`)."""
        with tool_errors():
            root = await store.get(owner, repo, sha)
            return await anyio.to_thread.run_sync(code_search.list_files, root, glob, max_results)

    @server.tool(annotations=READ_ONLY)
    async def search_code(
        owner: str, repo: str, sha: str, query: str, glob: str = "*", case_sensitive: bool = True, max_results: int = 100
    ) -> SearchResult:
        """Find lines containing `query` as a literal string (no regex)."""
        with tool_errors():
            root = await store.get(owner, repo, sha)
            return await anyio.to_thread.run_sync(
                code_search.search_text, root, query, glob, case_sensitive, max_results
            )

    @server.tool(annotations=READ_ONLY)
    async def read_file(owner: str, repo: str, sha: str, path: str, start_line: int = 1, end_line: int | None = None) -> FileSlice:
        """Read lines `start_line..end_line` (1-indexed, inclusive) of a text file."""
        with tool_errors():
            root = await store.get(owner, repo, sha)
            return await anyio.to_thread.run_sync(code_search.read_lines, root, path, start_line, end_line)

    return server


def main() -> None:
    """Entry point: `python -m prreview.mcp_servers.search_server`."""
    settings = github_settings()
    token = settings.github_read_token.get_secret_value() if settings.github_read_token else None
    serve(build_server(WorkspaceStore(workspace_settings(), GitHubClient(settings, token))), "prreview search MCP server")


if __name__ == "__main__":
    main()
