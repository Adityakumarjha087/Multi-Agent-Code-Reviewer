"""Runtime dependencies shared by the agents of one review."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
from mcp import Client

from prreview.agents.loop import Budget
from prreview.agents.toolbox import ToolBox, ToolSource
from prreview.config import AgentLimits, LLMSettings
from prreview.llm.client import LLMClient
from prreview.observability.events import EventSink, NullEvents
from prreview.observability.tracing import NoopTracer, Tracer
from prreview.schemas.github import PullRequest

# Tool allowlists per agent (D7, layer 3). No agent can reach github_write.
SEARCH_TOOLS = frozenset({"list_files", "search_code", "read_file"})


@dataclass
class McpClients:
    """Connected MCP clients. `sandbox` is None when sandboxed test runs are disabled."""

    github_read: Client
    search: Client
    sandbox: Client | None


@dataclass
class ReviewDeps:
    """Everything an agent node needs besides the review state."""

    llm: LLMClient
    llm_settings: LLMSettings
    limits: AgentLimits
    budget: Budget
    nonce: str
    mcp: McpClients
    http: httpx.AsyncClient
    tracer: Tracer = field(default_factory=NoopTracer)
    events: EventSink = field(default_factory=NullEvents)

    def search_toolbox(self, pr: PullRequest) -> ToolBox:
        """Read-only repository tools bound to the PR head commit."""
        bound = {"owner": pr.ref.owner, "repo": pr.ref.repo, "sha": pr.head_sha}
        source = ToolSource(self.mcp.search, SEARCH_TOOLS, self.limits.mcp_call_timeout_s)
        return ToolBox([source], bound, self.nonce, self.limits.tool_output_max_chars)
