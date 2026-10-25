"""Shared helpers for the MCP servers: error translation and the command-line entry point."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from prreview.github.client import GitHubAPIError, PayloadTooLargeError
from prreview.workspace.archive import ArchiveError

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True)

# Failures a caller can reasonably act on. They become `is_error` tool results with a clear
# message so the agent can record a note and continue. Anything else is a bug and surfaces
# as MCP's generic "Error executing tool" with the traceback in the server log.
EXPECTED_ERRORS: tuple[type[Exception], ...] = (
    GitHubAPIError,
    PayloadTooLargeError,
    ArchiveError,
    httpx.TransportError,
    ValueError,  # includes pydantic.ValidationError and our input validators
    FileNotFoundError,
)


@contextmanager
def tool_errors() -> Iterator[None]:
    """Re-raise expected failures as `ToolError` so the client receives the real reason."""
    try:
        yield
    except EXPECTED_ERRORS as exc:
        raise ToolError(f"{type(exc).__name__}: {exc}") from exc


def serve(server: MCPServer, description: str) -> None:
    """Parse `--transport/--host/--port` and run the server until it is stopped."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allowed-host", action="append", default=[],
                        help="Host header to accept over HTTP, e.g. 'sandbox:*' (repeatable)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.transport == "stdio":
        server.run("stdio")
        return
    # mcp 2.3.0 only enables DNS-rebinding protection automatically when bound to localhost; when
    # bound to 0.0.0.0 (Docker Compose) we enable it explicitly with the expected service hostnames.
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=args.allowed_host or [f"{args.host}:*"],
    )
    server.run("streamable-http", host=args.host, port=args.port, transport_security=security)
