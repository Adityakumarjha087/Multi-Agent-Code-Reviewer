"""Read-only search and file access inside one repository snapshot.

Search is fixed-string only. Regex is deliberately not offered: patterns may be
influenced by untrusted PR text, and Python's `re` has no timeout, so a crafted
pattern could hang the server (ReDoS).
"""

from __future__ import annotations

from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path

from prreview.schemas.search import FileListing, FileSlice, SearchMatch, SearchResult

MAX_FILE_BYTES = 1_000_000
MAX_LINE_CHARS = 300
_SKIP_DIRS = frozenset({".git", "node_modules", ".venv", "venv", "__pycache__", ".tox", "dist", "build"})


class PathOutsideRepoError(ValueError):
    """A requested path resolves outside the snapshot (via `..` or a symlink)."""


def resolve_inside(root: Path, rel_path: str) -> Path:
    """Resolve `rel_path` under `root`, refusing anything that escapes it."""
    root = root.resolve()
    candidate = (root / rel_path).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise PathOutsideRepoError(f"path escapes the repository: {rel_path!r}")
    return candidate


def iter_files(root: Path, glob: str = "*") -> Iterator[Path]:
    """Yield regular, non-symlink files under `root` whose relative path matches `glob`."""
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if path.is_file() and not path.is_symlink() and fnmatch(rel.as_posix(), glob):
            yield path


def list_files(root: Path, glob: str = "*", max_results: int = 500) -> FileListing:
    """List repository-relative paths matching `glob` (fnmatch syntax; `*` also matches `/`)."""
    paths: list[str] = []
    for path in iter_files(root, glob):
        if len(paths) == max_results:
            return FileListing(paths=paths, truncated=True)
        paths.append(path.relative_to(root).as_posix())
    return FileListing(paths=paths, truncated=False)


def search_text(
    root: Path, query: str, glob: str = "*", case_sensitive: bool = True, max_results: int = 100
) -> SearchResult:
    """Find lines containing `query` as a literal substring."""
    if not query:
        raise ValueError("query must not be empty")
    needle = query if case_sensitive else query.lower()
    matches: list[SearchMatch] = []
    scanned = 0
    for path in iter_files(root, glob):
        text = _read_text(path)
        if text is None:
            continue
        scanned += 1
        for line_no, line in enumerate(text.splitlines(), start=1):
            haystack = line if case_sensitive else line.lower()
            if needle in haystack:
                if len(matches) == max_results:
                    return SearchResult(query=query, matches=matches, truncated=True, files_scanned=scanned)
                rel = path.relative_to(root).as_posix()
                matches.append(SearchMatch(path=rel, line=line_no, text=line[:MAX_LINE_CHARS]))
    return SearchResult(query=query, matches=matches, truncated=False, files_scanned=scanned)


def read_lines(root: Path, rel_path: str, start_line: int = 1, end_line: int | None = None) -> FileSlice:
    """Return lines `start_line..end_line` (1-indexed, inclusive) of a text file."""
    path = resolve_inside(root, rel_path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file in repository: {rel_path!r}")
    text = _read_text(path)
    if text is None:
        raise ValueError(f"{rel_path!r} is binary or larger than {MAX_FILE_BYTES} bytes")
    lines = text.splitlines()
    start = max(start_line, 1)
    end = min(end_line or len(lines), len(lines))
    return FileSlice(
        path=rel_path,
        start_line=start,
        end_line=end,
        total_lines=len(lines),
        content="\n".join(lines[start - 1 : end]),
    )


def _read_text(path: Path) -> str | None:
    """Return file text, or None for binary/oversized files."""
    if path.stat().st_size > MAX_FILE_BYTES:
        return None
    data = path.read_bytes()
    if b"\x00" in data:
        return None
    return data.decode("utf-8", errors="replace")
