"""Publisher (D9): posts human-approved comments through the github_write MCP server. No LLM here.

`python -m prreview.publisher`. Only this process (and the write server it launches) ever sees
GITHUB_WRITE_TOKEN.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError

from prreview.config import database_settings, github_settings
from prreview.persistence.repo import Repository

logger = logging.getLogger(__name__)
POST_TIMEOUT_S = 60.0


async def publish_approved(repo: Repository, write_server: Client) -> int:
    """Post every approved comment once. Returns how many were attempted."""
    attempted = 0
    while (comment := repo.claim_approved()) is not None:
        attempted += 1
        try:
            result = await write_server.call_tool("post_approved_comment", {"comment_id": comment.id},
                                                  read_timeout_seconds=POST_TIMEOUT_S)
        except MCPError as exc:
            # We do not know whether GitHub created the comment, so we must not retry automatically.
            repo.record_post_failed(comment.id, f"outcome unknown ({exc.error.message}); check the PR before re-approving")
            continue
        if result.is_error:
            repo.record_post_failed(comment.id, " ".join(getattr(b, "text", "") for b in result.content))
        else:
            repo.record_posted(comment.id, result.structured_content["id"], result.structured_content["html_url"])
    return attempted


def write_server_params() -> StdioServerParameters:
    """Launch the write server with only the variables it needs."""
    settings = github_settings()
    if settings.github_write_token is None:
        raise SystemExit("GITHUB_WRITE_TOKEN is not set; nothing can be posted.")
    env = {
        "GITHUB_WRITE_TOKEN": settings.github_write_token.get_secret_value(),
        "DATABASE_URL": database_settings().database_url,
        **({"PATH": os.environ["PATH"]} if "PATH" in os.environ else {}),
    }
    return StdioServerParameters(command=sys.executable, args=["-m", "prreview.mcp_servers.github_write"], env=env)


async def run_forever(poll_interval_s: float) -> None:
    """Poll for approved comments until interrupted."""
    repo = Repository.from_url(database_settings().database_url)
    async with Client(write_server_params()) as write_server:
        while True:
            if await publish_approved(repo, write_server) == 0:
                await asyncio.sleep(poll_interval_s)


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_forever(database_settings().worker_poll_interval_s))


if __name__ == "__main__":
    main()
