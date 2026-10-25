"""Safe extraction of untrusted repository tarballs.

Defends against path traversal and link escapes (Python's `filter="data"`), device
files, and decompression bombs (file-count and total-size caps checked before
anything is written).
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Iterator
from pathlib import Path


class ArchiveError(Exception):
    """The archive is malformed, unsafe, or over a size limit."""


def extract_github_tarball(data: bytes, dest: Path, max_files: int, max_bytes: int) -> int:
    """Extract a GitHub tarball into `dest`, dropping its `owner-repo-sha/` top-level folder.

    Returns the number of regular files extracted.
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            members = _checked_members(tar.getmembers(), max_files, max_bytes)
            dest.mkdir(parents=True, exist_ok=True)
            tar.extractall(dest, members=members, filter="data")
    except tarfile.TarError as exc:
        raise ArchiveError(f"cannot extract tarball: {exc}") from exc
    return sum(1 for m in members if m.isfile())


def _checked_members(members: list[tarfile.TarInfo], max_files: int, max_bytes: int) -> list[tarfile.TarInfo]:
    """Strip the top-level folder and enforce the caps. Device files and FIFOs are dropped."""
    kept = [m for m in _strip_top_level(members) if m.isfile() or m.isdir() or m.issym() or m.islnk()]
    file_count = sum(1 for m in kept if m.isfile())
    total_bytes = sum(m.size for m in kept if m.isfile())
    if file_count > max_files:
        raise ArchiveError(f"archive has {file_count} files, limit is {max_files}")
    if total_bytes > max_bytes:
        raise ArchiveError(f"archive expands to {total_bytes} bytes, limit is {max_bytes}")
    return kept


def _strip_top_level(members: list[tarfile.TarInfo]) -> Iterator[tarfile.TarInfo]:
    for member in members:
        _, sep, rest = member.name.partition("/")
        if not sep or not rest:
            continue  # the top-level folder itself, or the pax global header
        member.name = rest
        if member.islnk():
            # Hard-link targets are archive paths, so they carry the same prefix.
            member.linkname = member.linkname.partition("/")[2]
        yield member
