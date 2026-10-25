"""Collect the pip requirements a repository declares, treating its files as untrusted.

Only plain PEP 508 requirement specifiers are passed to pip. Option lines
(`--index-url`, `--extra-index-url`, `-e`, `-r`, `-c`), URLs and local paths are
skipped and reported, because they could point pip at an attacker's package index
or make it build from source.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

REQUIREMENT_FILES = (
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "requirements_dev.txt",
    "requirements_test.txt",
    "test-requirements.txt",
    "dev-requirements.txt",
)
TEST_EXTRAS = ("test", "tests", "testing", "dev")


@dataclass
class CollectedRequirements:
    """Requirement specifiers safe to hand to pip, plus everything that was skipped and why."""

    specs: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def collect_requirements(repo_root: Path, max_requirements: int, pytest_spec: str) -> CollectedRequirements:
    """Gather requirements from requirements files and pyproject.toml, and make sure pytest is included."""
    result = CollectedRequirements()
    for name in REQUIREMENT_FILES:
        path = repo_root / name
        if path.is_file() and not path.is_symlink():
            _add_lines(result, path.read_text(errors="replace").splitlines(), source=name)
    _add_pyproject(result, repo_root / "pyproject.toml")
    if not any(Requirement(s).name.lower() == "pytest" for s in result.specs):
        result.specs.append(pytest_spec)
    if len(result.specs) > max_requirements:
        result.skipped.append(f"dropped {len(result.specs) - max_requirements} specs over the limit of {max_requirements}")
        result.specs = result.specs[:max_requirements]
    return result


def parse_requirement_line(line: str) -> str | None:
    """Return a normalised PEP 508 specifier, or None if the line must not reach pip."""
    text = line.split(" #", 1)[0].strip()
    if not text or text.startswith("#") or text.startswith("-"):
        return None
    try:
        requirement = Requirement(text)
    except InvalidRequirement:
        return None
    if requirement.url:
        return None  # `name @ https://...` would bypass the package index
    return str(requirement)


def _add_lines(result: CollectedRequirements, lines: list[str], source: str) -> None:
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        spec = parse_requirement_line(stripped)
        if spec is None:
            result.skipped.append(f"{source}: {stripped[:120]}")
        elif spec not in result.specs:
            result.specs.append(spec)


def _add_pyproject(result: CollectedRequirements, path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        return
    try:
        project = tomllib.loads(path.read_text(errors="replace")).get("project", {})
    except tomllib.TOMLDecodeError as exc:
        result.skipped.append(f"pyproject.toml: unparseable ({exc})")
        return
    _add_lines(result, _string_list(project.get("dependencies")), source="pyproject.toml")
    extras = project.get("optional-dependencies", {})
    if isinstance(extras, dict):
        for extra in TEST_EXTRAS:
            _add_lines(result, _string_list(extras.get(extra)), source=f"pyproject.toml[{extra}]")


def _string_list(value: object) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []
