"""Tests for sandbox input handling that does not need Docker: requirements, JUnit, test paths."""

from pathlib import Path

import pytest

from prreview.sandbox.deps import collect_requirements, parse_requirement_line
from prreview.sandbox.docker_sandbox import InvalidTestPathError, validate_test_path
from prreview.sandbox.junit import JUnitParseError, parse_junit

PYTEST = "pytest==9.1.1"


@pytest.mark.parametrize(
    "line",
    [
        "--index-url https://evil.example/simple",
        "--extra-index-url=https://evil.example",
        "-e .",
        "-r other.txt",
        "evilpkg @ https://evil.example/evilpkg.whl",
        "./local/path",
        "# comment",
        "",
    ],
)
def test_unsafe_or_empty_lines_are_not_passed_to_pip(line: str) -> None:
    assert parse_requirement_line(line) is None


def test_plain_specifiers_are_kept_and_normalised() -> None:
    assert parse_requirement_line("requests>=2.0  # http") == "requests>=2.0"
    assert parse_requirement_line('tomli; python_version < "3.11"') == 'tomli; python_version < "3.11"'


def test_collects_from_files_and_pyproject_and_adds_pytest(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("six==1.17.0\n--index-url https://evil.example\n")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\ndependencies=["attrs"]\n[project.optional-dependencies]\ntest=["hypothesis"]\ndocs=["sphinx"]\n'
    )
    result = collect_requirements(tmp_path, max_requirements=50, pytest_spec=PYTEST)
    assert result.specs == ["six==1.17.0", "attrs", "hypothesis", PYTEST]
    assert result.skipped == ["requirements.txt: --index-url https://evil.example"]


def test_existing_pytest_pin_is_respected(tmp_path: Path) -> None:
    (tmp_path / "requirements-dev.txt").write_text("pytest==8.0.0\n")
    assert collect_requirements(tmp_path, 50, PYTEST).specs == ["pytest==8.0.0"]


def test_requirement_count_is_capped(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("\n".join(f"pkg{i}" for i in range(10)))
    result = collect_requirements(tmp_path, max_requirements=3, pytest_spec=PYTEST)
    assert len(result.specs) == 3
    assert "over the limit" in result.skipped[-1]


def test_unparseable_pyproject_is_reported_not_raised(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project\n")
    assert "unparseable" in collect_requirements(tmp_path, 50, PYTEST).skipped[0]


JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="4">
<testcase classname="tests.test_a" name="test_ok" time="0.01"/>
<testcase classname="tests.test_a" name="test_bad" time="0.02"><failure message="assert 1 == 2">trace</failure></testcase>
<testcase classname="tests.test_b" name="test_err" time="0"><error message="fixture boom"/></testcase>
<testcase classname="tests.test_b" name="test_skip" time="0"><skipped message="not on linux"/></testcase>
</testsuite></testsuites>"""


def test_parses_junit_outcomes() -> None:
    cases = parse_junit(JUNIT)
    assert [(c.test_id, c.outcome) for c in cases] == [
        ("tests.test_a::test_ok", "passed"),
        ("tests.test_a::test_bad", "failed"),
        ("tests.test_b::test_err", "error"),
        ("tests.test_b::test_skip", "skipped"),
    ]
    assert cases[1].message == "assert 1 == 2"


def test_junit_entity_expansion_is_blocked() -> None:
    bomb = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;">]><testsuites>&b;</testsuites>'
    with pytest.raises(JUnitParseError):
        parse_junit(bomb)


@pytest.mark.parametrize("path", ["tests/test_a.py", "tests/test_a.py::test_ok", "tests"])
def test_valid_test_paths(path: str) -> None:
    assert validate_test_path(path) == path


@pytest.mark.parametrize("path", ["", "--co", "-p evil", "/etc/passwd", "../x.py", "tests/../../x.py::t"])
def test_test_paths_that_could_be_options_or_escape_are_rejected(path: str) -> None:
    with pytest.raises(InvalidTestPathError):
        validate_test_path(path)
