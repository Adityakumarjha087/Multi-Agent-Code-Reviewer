"""Progress events for the UI ("planner started", "security: 2 findings"). Separate from tracing:
events are a small user-facing feed stored in Postgres; traces are full debugging detail in Langfuse."""

from __future__ import annotations

from typing import Protocol


class EventSink(Protocol):
    """Receives progress events for one review."""

    def emit(self, agent: str, kind: str, message: str) -> None: ...


class NullEvents:
    """Discards events (CLI, tests, benchmark)."""

    def emit(self, agent: str, kind: str, message: str) -> None:
        return None


class ListEvents:
    """Keeps events in memory (tests)."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str, str]] = []

    def emit(self, agent: str, kind: str, message: str) -> None:
        self.items.append((agent, kind, message))
