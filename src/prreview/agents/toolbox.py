"""Expose a curated subset of MCP tools to one agent (D7, layer 3: per-agent allowlists).

Two safety properties live here:
- Allowlist: an agent only sees, and can only call, the tools named for it.
- Argument binding: arguments such as owner/repo/sha are removed from the schema the LLM
  sees and forced to this PR's values on every call, so injected text cannot redirect an
  agent to another repository or commit.
Every tool result is fenced as untrusted data before it goes back to the model.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from mcp import Client
from mcp.shared.exceptions import MCPError

from prreview.safety.untrusted import fence


@dataclass
class ToolSource:
    """One connected MCP server and the tools an agent may use from it."""

    client: Client
    allowed: frozenset[str]
    timeout_s: float


@dataclass(frozen=True)
class ToolCallOutcome:
    """What goes back to the model as a `tool_result`."""

    content: str
    is_error: bool


@dataclass
class ToolBox:
    """Allowlisted, argument-bound view over several MCP servers."""

    sources: list[ToolSource]
    bound_args: dict[str, Any]
    nonce: str
    max_chars: int
    _routes: dict[str, ToolSource] = field(default_factory=dict, init=False)
    _params: dict[str, set[str]] = field(default_factory=dict, init=False)

    async def definitions(self) -> list[dict[str, Any]]:
        """Return Messages API tool definitions for every allowed tool, with bound args hidden."""
        defs: list[dict[str, Any]] = []
        for source in self.sources:
            listed = {t.name: t for t in (await source.client.list_tools()).tools}
            missing = source.allowed - listed.keys()
            if missing:
                raise LookupError(f"allowlisted tools not offered by server: {sorted(missing)}")
            for name in sorted(source.allowed):
                if name in self._routes:
                    raise ValueError(f"tool name {name!r} offered by two servers")
                self._routes[name] = source
                tool = listed[name]
                self._params[name] = set(tool.input_schema.get("properties", {}))
                defs.append({
                    "name": name,
                    "description": tool.description or "",
                    "input_schema": self._hide_bound(tool.input_schema),
                })
        return defs

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolCallOutcome:
        """Call a tool. Failures become error results so the agent can continue (D11)."""
        source = self._routes.get(name)
        if source is None:
            return ToolCallOutcome(f"Tool {name!r} is not available to you.", is_error=True)
        args = {**arguments, **{k: v for k, v in self.bound_args.items() if k in self._params[name]}}
        try:
            result = await source.client.call_tool(name, args, read_timeout_seconds=source.timeout_s)
        except MCPError as exc:
            return ToolCallOutcome(f"Tool {name} failed: {exc.error.message}", is_error=True)
        text = _result_text(result.structured_content, result.content)
        return ToolCallOutcome(self._wrap(name, text), is_error=result.is_error)

    def _hide_bound(self, schema: dict[str, Any]) -> dict[str, Any]:
        visible = copy.deepcopy(schema)
        properties = visible.get("properties", {})
        for key in self.bound_args:
            properties.pop(key, None)
        visible["required"] = [r for r in visible.get("required", []) if r not in self.bound_args]
        return visible

    def _wrap(self, name: str, text: str) -> str:
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + f"\n[truncated: output exceeded {self.max_chars} characters]"
        return fence(text, f"tool:{name}", self.nonce)


def _result_text(structured: Any, content: list[Any]) -> str:
    if structured is not None:
        return json.dumps(structured, indent=1, default=str)
    return "\n".join(getattr(block, "text", "") for block in content)
