"""Fixtures shared by integration tests."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import docker
import pytest
from alembic import command
from alembic.config import Config


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    client = docker.from_env()
    container = client.containers.run(
        "postgres:17", detach=True, environment={"POSTGRES_PASSWORD": "test", "POSTGRES_DB": "prreview"},
        ports={"5432/tcp": ("127.0.0.1", None)}, labels={"prreview.test": "1"},
    )
    try:
        for _ in range(60):
            # Check over TCP: the image first runs a temporary socket-only init server, then restarts.
            if container.exec_run(["pg_isready", "-h", "127.0.0.1", "-U", "postgres", "-d", "prreview"]).exit_code == 0:
                break
            time.sleep(0.5)
        container.reload()
        port = container.ports["5432/tcp"][0]["HostPort"]
        url = f"postgresql+psycopg://postgres:test@127.0.0.1:{port}/prreview"
        previous = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = url
        try:
            command.upgrade(Config("alembic.ini"), "head")
        finally:
            os.environ.pop("DATABASE_URL") if previous is None else os.environ.__setitem__("DATABASE_URL", previous)
        yield url
    finally:
        container.remove(force=True)




@pytest.fixture
def clean_db(database_url: str) -> str:
    """Empty every table before a test: the container is shared by the whole session."""
    from sqlalchemy import create_engine, text

    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE review_events, draft_comments, reviews"))
    engine.dispose()
    return database_url


@pytest.fixture
def github_quota() -> None:
    """Skip (visibly, with the reason) when the GitHub API quota is too low to run a live test.

    Unauthenticated callers get 60 requests/hour; set GITHUB_READ_TOKEN for 5000/hour.
    The /rate_limit endpoint itself does not consume quota.
    """
    import httpx

    from prreview.config import GitHubSettings

    settings = GitHubSettings()
    headers = {"Authorization": f"Bearer {settings.github_read_token.get_secret_value()}"} if settings.github_read_token else {}
    core = httpx.get("https://api.github.com/rate_limit", headers=headers, timeout=10).json()["resources"]["core"]
    if core["remaining"] < 10:
        pytest.skip(f"GitHub API quota too low ({core['remaining']}/{core['limit']} left, resets at epoch {core['reset']}); "
                    "set GITHUB_READ_TOKEN to run live tests")
