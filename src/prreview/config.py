"""Runtime configuration loaded from environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# python:3.11-slim as pulled on 2026-10-09. Pinned by digest so sandbox runs are reproducible.
DEFAULT_SANDBOX_IMAGE = (
    "python:3.11-slim@sha256:0dd364ba7e10242f07755449e3a3d0e35f9efd987952737b90def6709ab0c5ce"
)


class GitHubSettings(BaseSettings):
    """GitHub API access. Read and write tokens are separate so each MCP server gets least privilege."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    github_api_url: str = "https://api.github.com"
    github_api_version: str = "2026-03-10"
    github_read_token: SecretStr | None = None
    github_write_token: SecretStr | None = None
    github_timeout_s: float = 30.0
    github_max_attempts: int = Field(default=4, ge=1)
    github_retry_initial_wait_s: float = Field(default=0.5, ge=0)
    github_max_files: int = Field(default=300, ge=1, le=3000)

    @field_validator("github_read_token", "github_write_token", mode="before")
    @classmethod
    def _blank_token_is_none(cls, value: object) -> object:
        """`GITHUB_WRITE_TOKEN=` in .env must mean "no token", not an empty token that fails later with 401."""
        return None if isinstance(value, str) and not value.strip() else value


class SandboxSettings(BaseSettings):
    """Resource limits for code executed from untrusted repositories."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SANDBOX_", extra="ignore")

    image: str = DEFAULT_SANDBOX_IMAGE
    cpus: float = Field(default=1.0, gt=0)
    memory_mb: int = Field(default=1024, ge=64)
    pids_limit: int = Field(default=256, ge=16)
    workspace_tmpfs_mb: int = Field(default=512, ge=16)
    run_timeout_s: int = Field(default=300, ge=1)
    install_timeout_s: int = Field(default=300, ge=1)
    max_output_bytes: int = Field(default=200_000, ge=1_000)
    max_deps_mb: int = Field(default=1024, ge=10)
    max_requirements: int = Field(default=200, ge=1)
    default_pytest_spec: str = "pytest==9.1.1"


class WorkspaceSettings(BaseSettings):
    """Where downloaded repository snapshots are cached, and how large they may be."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="WORKSPACE_", extra="ignore")

    root: Path = Path(".workspaces")
    max_download_mb: int = Field(default=200, ge=1)
    max_extracted_mb: int = Field(default=500, ge=1)
    max_files: int = Field(default=20_000, ge=1)


@lru_cache(maxsize=1)
def github_settings() -> GitHubSettings:
    """Return process-wide GitHub settings."""
    return GitHubSettings()


@lru_cache(maxsize=1)
def sandbox_settings() -> SandboxSettings:
    """Return process-wide sandbox settings."""
    return SandboxSettings()


@lru_cache(maxsize=1)
def workspace_settings() -> WorkspaceSettings:
    """Return process-wide workspace settings."""
    return WorkspaceSettings()


class LLMSettings(BaseSettings):
    """Claude API settings. Each agent's model is configurable so cost/quality can be tuned per role."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LLM_", extra="ignore")

    planner_model: str = "claude-opus-5-5"
    bug_model: str = "claude-opus-5-5"
    security_model: str = "claude-opus-5-5"
    test_model: str = "claude-opus-5-5"
    aggregator_model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    max_tokens: int = Field(default=16_000, ge=256)
    request_timeout_s: float = Field(default=300.0, gt=0)
    sdk_max_retries: int = Field(default=3, ge=0)
    use_refusal_fallbacks: bool = True
    # Read from .env via settings: pydantic-settings does not export .env into os.environ, and the
    # Anthropic SDK only looks at os.environ, so the key must be passed to the client explicitly.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value


class AgentLimits(BaseSettings):
    """Step and spend limits (D11). A step is one LLM call or one tool call."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LIMIT_", extra="ignore")

    agent_max_steps: int = Field(default=12, ge=2)
    graph_recursion_limit: int = Field(default=40, ge=5)
    review_max_cost_usd: float = Field(default=2.0, gt=0)
    tool_output_max_chars: int = Field(default=20_000, ge=500)
    max_files_reviewed: int = Field(default=40, ge=1)
    max_patch_chars_per_file: int = Field(default=12_000, ge=500)
    mcp_call_timeout_s: float = Field(default=60.0, gt=0)
    sandbox_call_timeout_s: float = Field(default=1_500.0, gt=0)


@lru_cache(maxsize=1)
def llm_settings() -> LLMSettings:
    """Return process-wide LLM settings."""
    return LLMSettings()


@lru_cache(maxsize=1)
def agent_limits() -> AgentLimits:
    """Return process-wide agent limits."""
    return AgentLimits()


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection (SQLAlchemy URL with the psycopg 3 driver)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://prreview:prreview@127.0.0.1:5432/prreview"
    worker_poll_interval_s: float = Field(default=2.0, gt=0)


@lru_cache(maxsize=1)
def database_settings() -> DatabaseSettings:
    """Return process-wide database settings."""
    return DatabaseSettings()
