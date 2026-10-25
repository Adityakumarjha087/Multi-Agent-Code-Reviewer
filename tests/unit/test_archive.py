"""Tests for safe extraction of untrusted tarballs."""

import io
import tarfile
from pathlib import Path

import pytest

from prreview.workspace.archive import ArchiveError, extract_github_tarball
from tests.conftest import make_tarball

TOP = "octo-demo-abc1234"


def _tar_with(member: tarfile.TarInfo, data: bytes = b"") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.addfile(member, io.BytesIO(data) if data else None)
    return buffer.getvalue()


def test_strips_top_level_folder(tmp_path: Path) -> None:
    data = make_tarball({"pkg/mod.py": b"x = 1\n", "README.md": b"hi"})
    count = extract_github_tarball(data, tmp_path / "out", max_files=10, max_bytes=1000)
    assert count == 2
    assert (tmp_path / "out/pkg/mod.py").read_text() == "x = 1\n"
    assert not (tmp_path / "out" / TOP).exists()


def test_rejects_path_traversal(tmp_path: Path) -> None:
    member = tarfile.TarInfo(f"{TOP}/../../escaped.txt")
    member.size = 3
    with pytest.raises(ArchiveError):
        extract_github_tarball(_tar_with(member, b"bad"), tmp_path / "out", 10, 1000)
    assert not (tmp_path / "escaped.txt").exists()


def test_rejects_symlink_pointing_outside(tmp_path: Path) -> None:
    member = tarfile.TarInfo(f"{TOP}/link")
    member.type = tarfile.SYMTYPE
    member.linkname = "/etc/passwd"
    with pytest.raises(ArchiveError):
        extract_github_tarball(_tar_with(member), tmp_path / "out", 10, 1000)


def test_drops_device_files(tmp_path: Path) -> None:
    member = tarfile.TarInfo(f"{TOP}/dev")
    member.type = tarfile.CHRTYPE
    assert extract_github_tarball(_tar_with(member), tmp_path / "out", 10, 1000) == 0
    assert not (tmp_path / "out/dev").exists()


def test_enforces_file_count_limit(tmp_path: Path) -> None:
    data = make_tarball({f"f{i}.txt": b"x" for i in range(5)})
    with pytest.raises(ArchiveError, match="5 files"):
        extract_github_tarball(data, tmp_path / "out", max_files=4, max_bytes=1000)


def test_enforces_size_limit_before_writing(tmp_path: Path) -> None:
    data = make_tarball({"big.bin": b"0" * 2000})
    with pytest.raises(ArchiveError, match="expands to 2000"):
        extract_github_tarball(data, tmp_path / "out", max_files=10, max_bytes=1000)
    assert not (tmp_path / "out").exists()


def test_rejects_non_gzip_data(tmp_path: Path) -> None:
    with pytest.raises(ArchiveError):
        extract_github_tarball(b"not a tarball", tmp_path / "out", 10, 1000)
