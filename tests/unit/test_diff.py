"""Tests for unified-diff parsing."""

import pytest

from prreview.github.diff import DiffParseError, parse_patch
from prreview.schemas.github import ChangedFile

PATCH = """@@ -10,4 +10,5 @@ def f():
 context a
-removed b
+added b
+added c
 context d
@@ -40 +41,2 @@
-old
+new
+newer
\\ No newline at end of file"""


def test_parses_line_numbers_for_each_hunk() -> None:
    first, second = parse_patch(PATCH)
    assert (first.new_start, first.new_lines) == (10, 5)
    assert first.added_lines == [11, 12]
    assert first.removed_lines == [11]
    assert second.old_lines == 1  # omitted length means 1
    assert second.added_lines == [41, 42]
    assert second.removed_lines == [40]


def test_commentable_lines_cover_new_side_of_hunks() -> None:
    file = ChangedFile(path="a.py", status="modified", additions=4, deletions=2, patch=PATCH, hunks=parse_patch(PATCH))
    assert file.commentable_lines == {10, 11, 12, 13, 14, 41, 42}


def test_rejects_text_before_first_hunk() -> None:
    with pytest.raises(DiffParseError):
        parse_patch("garbage\n@@ -1 +1 @@\n-a\n+b")


def test_empty_patch_has_no_hunks() -> None:
    assert parse_patch("") == []
