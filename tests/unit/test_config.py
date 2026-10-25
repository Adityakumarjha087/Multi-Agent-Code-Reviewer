"""Tests for settings parsing."""

import pytest

from prreview.config import GitHubSettings


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_tokens_are_treated_as_missing(monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
    monkeypatch.setenv("GITHUB_WRITE_TOKEN", blank)
    monkeypatch.setenv("GITHUB_READ_TOKEN", blank)
    settings = GitHubSettings(_env_file=None)
    assert settings.github_write_token is None and settings.github_read_token is None


def test_real_token_is_kept_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_WRITE_TOKEN", "ghp_example")
    settings = GitHubSettings(_env_file=None)
    assert settings.github_write_token.get_secret_value() == "ghp_example"
    assert "ghp_example" not in repr(settings)
