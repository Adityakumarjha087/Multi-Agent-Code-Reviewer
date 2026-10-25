"""Parse pytest's JUnit XML report. The XML is produced by untrusted code, so it is parsed
with defusedxml (blocks entity-expansion bombs and external entities)."""

from __future__ import annotations

from xml.etree.ElementTree import Element

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from prreview.schemas.sandbox import TestCaseResult

MAX_MESSAGE_CHARS = 500


class JUnitParseError(ValueError):
    """The report is missing, malformed, or uses forbidden XML features."""


def parse_junit(xml_text: str) -> list[TestCaseResult]:
    """Turn a JUnit XML document into one result per `<testcase>`."""
    try:
        root = SafeET.fromstring(xml_text)
    except (SafeET.ParseError, DefusedXmlException) as exc:
        raise JUnitParseError(f"invalid JUnit XML: {exc}") from exc
    return [_to_case(case) for case in root.iter("testcase")]


def _to_case(case: Element) -> TestCaseResult:
    classname = case.get("classname", "")
    name = case.get("name", "")
    outcome, detail = _outcome(case)
    return TestCaseResult(
        test_id=f"{classname}::{name}" if classname else name,
        outcome=outcome,
        duration_s=_to_float(case.get("time")),
        message=detail[:MAX_MESSAGE_CHARS],
    )


def _outcome(case: Element) -> tuple[str, str]:
    for tag in ("failure", "error", "skipped"):
        child = case.find(tag)
        if child is not None:
            outcome = "failed" if tag == "failure" else tag
            return outcome, child.get("message") or (child.text or "")
    return "passed", ""


def _to_float(value: str | None) -> float:
    try:
        return float(value) if value else 0.0
    except ValueError:
        return 0.0
