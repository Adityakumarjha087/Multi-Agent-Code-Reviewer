"""Results of searching and reading a repository snapshot."""

from __future__ import annotations

from pydantic import BaseModel


class SearchMatch(BaseModel):
    """One line containing the search string."""

    path: str
    line: int
    text: str


class SearchResult(BaseModel):
    """All matches for a query, with an explicit truncation flag."""

    query: str
    matches: list[SearchMatch]
    truncated: bool
    files_scanned: int


class FileListing(BaseModel):
    """Repository-relative paths matching a glob."""

    paths: list[str]
    truncated: bool


class FileSlice(BaseModel):
    """A range of lines from one file (1-indexed, inclusive)."""

    path: str
    start_line: int
    end_line: int
    total_lines: int
    content: str
