"""Tests for repository search and file reading."""

import os
from pathlib import Path

import pytest

from prreview.search.code_search import PathOutsideRepoError, list_files, read_lines, search_text


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg/app.py").write_text("import os\nPASSWORD = 'hunter2'\nprint(os.getcwd())\n")
    (root / "pkg/util.py").write_text("def helper():\n    return 'password'\n")
    (root / "image.bin").write_bytes(b"\x00\x01password")
    (root / ".git").mkdir()
    (root / ".git/config").write_text("password in git dir\n")
    return root


def test_literal_search_with_line_numbers(repo: Path) -> None:
    result = search_text(repo, "PASSWORD")
    assert [(m.path, m.line) for m in result.matches] == [("pkg/app.py", 2)]


def test_case_insensitive_skips_binary_and_git_dir(repo: Path) -> None:
    result = search_text(repo, "password", case_sensitive=False)
    assert {m.path for m in result.matches} == {"pkg/app.py", "pkg/util.py"}


def test_regex_metacharacters_are_literal(repo: Path) -> None:
    assert search_text(repo, "os.getcwd()").matches[0].line == 3
    assert search_text(repo, "o.*cwd").matches == []


def test_truncates_at_max_results(repo: Path) -> None:
    result = search_text(repo, "o", max_results=2)
    assert len(result.matches) == 2 and result.truncated


def test_glob_filters_files(repo: Path) -> None:
    assert list_files(repo, "*util*").paths == ["pkg/util.py"]


def test_read_lines_returns_inclusive_range(repo: Path) -> None:
    sliced = read_lines(repo, "pkg/app.py", 2, 3)
    assert sliced.content == "PASSWORD = 'hunter2'\nprint(os.getcwd())"
    assert sliced.total_lines == 3


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "pkg/../../outside.txt"])
def test_read_refuses_paths_outside_repo(repo: Path, path: str) -> None:
    (repo.parent / "outside.txt").write_text("secret")
    with pytest.raises(PathOutsideRepoError):
        read_lines(repo, path)


def test_read_refuses_symlink_escaping_repo(repo: Path) -> None:
    (repo.parent / "outside.txt").write_text("secret")
    os.symlink(repo.parent / "outside.txt", repo / "sneaky")
    with pytest.raises(PathOutsideRepoError):
        read_lines(repo, "sneaky")


def test_symlink_loop_does_not_hang(repo: Path) -> None:
    os.symlink(repo, repo / "pkg" / "loop")
    assert "pkg/app.py" in list_files(repo).paths
