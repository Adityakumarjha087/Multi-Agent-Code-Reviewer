"""MCP server that posts review comments. Holds the only write token.

Its single tool takes a comment *id*, not comment text. The server loads the comment from the
database and posts it only if a human approved it and the publisher claimed it (`posting`).
So even a compromised caller cannot post arbitrary text or post an unapproved comment
(defence in depth on top of "no LLM agent is connected to this server").
"""

from __future__ import annotations

from collections.abc import Callable

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from prreview.config import database_settings, github_settings
from prreview.github.client import GitHubClient
from prreview.mcp_servers.common import serve, tool_errors
from prreview.persistence.repo import PostableComment, Repository
from prreview.schemas.github import PostedComment, PRRef

MAX_COMMENT_CHARS = 65_000  # GitHub rejects bodies over 65536 characters


class NotApprovedError(ValueError):
    """The comment does not exist or is not in the `posting` state."""


def build_server(github: GitHubClient, lookup: Callable[[str], PostableComment | None]) -> MCPServer:
    """Create the server. `lookup` returns a comment only if it is approved and claimed for posting."""
    server = MCPServer("prreview-github-write", instructions="Posts human-approved PR review comments by id.")

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False))
    async def post_approved_comment(comment_id: str) -> PostedComment:
        """Post the approved draft comment with this id to its pull request."""
        with tool_errors():
            comment = lookup(comment_id)
            if comment is None:
                raise NotApprovedError(f"comment {comment_id!r} is not approved and claimed for posting")
            if not comment.body.strip() or len(comment.body) > MAX_COMMENT_CHARS:
                raise ValueError(f"comment body must be 1..{MAX_COMMENT_CHARS} characters")
            ref = PRRef(owner=comment.owner, repo=comment.repo, number=comment.number)
            return await github.create_review_comment(
                ref, comment.commit_id, comment.file, comment.line, comment.body, "RIGHT", comment.start_line
            )

    return server


def main() -> None:
    """Entry point. Refuses to start without a write token."""
    settings = github_settings()
    if settings.github_write_token is None:
        raise SystemExit("GITHUB_WRITE_TOKEN is not set; the write server has nothing to post with.")
    client = GitHubClient(settings, settings.github_write_token.get_secret_value())
    repo = Repository.from_url(database_settings().database_url)
    serve(build_server(client, repo.comment_for_posting), "prreview GitHub write MCP server")


if __name__ == "__main__":
    main()
