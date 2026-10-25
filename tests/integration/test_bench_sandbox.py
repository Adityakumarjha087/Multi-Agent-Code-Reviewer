"""Benchmark harness with the real Docker sandbox: base vs head test comparison on case b13.

Needs Docker and PyPI access (the install stage downloads pytest). Run with: pytest -m docker
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from prreview.bench.dataset import load_cases
from prreview.bench.harness import _review
from prreview.config import AgentLimits, LLMSettings
from prreview.observability.tracing import NoopTracer
from tests.fakes import OracleLLM

pytestmark = pytest.mark.docker


async def test_sandbox_attributes_only_the_new_failure(tmp_path: Path) -> None:
    cases = load_cases(Path("benchmark/cases"), "b1[34]-*.yaml")
    case = next(c for c in cases if c.id == "b13-discount-breaks-tests")
    async with httpx.AsyncClient() as http:
        outcome = await _review(case, 1, OracleLLM(cases), tmp_path, LLMSettings(_env_file=None),
                                AgentLimits(_env_file=None), NoopTracer(), True, http)
    comparison = outcome.state.test_comparison
    assert comparison is not None, [n.message for n in outcome.state.notes]
    assert (comparison.base_status, comparison.head_status) == ("passed", "failed")
    assert [c.test_id for c in comparison.newly_failing] == ["tests.test_pricing::test_ten_percent"]
    assert comparison.still_failing == []
    assert any(f.agent == "tests" for f in outcome.state.findings)
