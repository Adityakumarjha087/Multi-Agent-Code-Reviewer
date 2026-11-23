"""`parse_pr_url` accepts what people paste from the browser and rejects everything else."""

from __future__ import annotations

import pytest

from prreview.review.runner import parse_pr_url


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/octo/demo/pull/7",
        "  https://github.com/octo/demo/pull/7/  ",
        "https://github.com/octo/demo/pull/7/files",
        "https://github.com/octo/demo/pull/7/commits?w=1",
        "https://github.com/octo/demo/pull/7#issuecomment-123",
        "https://www.github.com/octo/demo/pull/7",
    ],
)
def test_accepts_common_browser_urls(url: str) -> None:
    ref = parse_pr_url(url)
    assert (ref.owner, ref.repo, ref.number) == ("octo", "demo", 7)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://github.com/octo/demo/pull/",
        "https://github.com/octo/demo/pull/abc",
        "https://github.com/octo/demo/issues/7",
        "https://github.com/octo/demo/pull/7/other",
        "http://github.com/octo/demo/pull/7",
        "https://evil.example/github.com/octo/demo/pull/7",
        "https://github.com.evil.example/octo/demo/pull/7",
    ],
)
def test_rejects_anything_else(url: str) -> None:
    with pytest.raises(ValueError):
        parse_pr_url(url)
