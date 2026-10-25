"""Benchmark dataset, scoring, and harness plumbing (test-double LLMs; no API calls)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from prreview.bench.dataset import Case, label_problems, load_cases
from prreview.bench.harness import run_case
from prreview.bench.metrics import CaseRun, match, score, summarise
from prreview.config import AgentLimits, LLMSettings
from prreview.observability.tracing import NoopTracer
from prreview.schemas.review import Finding
from tests.fakes import OracleLLM

CASES = load_cases(Path("benchmark/cases"))
BY_ID = {c.id: c for c in CASES}


def test_dataset_shape_and_labels_are_valid() -> None:
    kinds = [c.kind for c in CASES]
    assert len(CASES) == 30
    assert (kinds.count("bug"), kinds.count("security"), kinds.count("clean"), kinds.count("injection")) == (14, 8, 5, 3)
    assert [p for c in CASES for p in label_problems(c)] == []


def test_snapshot_files_unchanged_between_base_and_head_are_not_in_the_diff() -> None:
    case = BY_ID["b13-discount-breaks-tests"]
    assert [f.path for f in case.changed_files()] == ["app/pricing.py"]


def finding(file: str, line: int, category: str, **extra: object) -> Finding:
    return Finding(agent="bug", category=category, severity="high", confidence=0.8, file=file, line_start=line,
                   line_end=line, title="t", explanation="e", evidence="x = 1", **extra)


def test_matching_uses_file_category_and_line_tolerance() -> None:
    case = BY_ID["b01-pagination-off-by-one"]  # label app/pagination.py 6-7, logic_error|edge_case
    assert match(case, [finding("app/pagination.py", 9, "logic_error")])[0] == {0}   # within +3
    assert match(case, [finding("app/pagination.py", 11, "logic_error")])[0] == set()  # too far
    assert match(case, [finding("app/pagination.py", 6, "injection")])[0] == set()   # wrong category
    assert match(case, [finding("app/other.py", 6, "logic_error")])[0] == set()


def test_each_label_matches_one_finding_only() -> None:
    case = BY_ID["b01-pagination-off-by-one"]
    s = score(case, CaseRun(case.id, case.kind, 1, findings=[finding("app/pagination.py", 6, "logic_error"),
                                                             finding("app/pagination.py", 7, "edge_case")]))
    assert (s.labels_matched, s.true_positives, s.false_positives) == (1, 1, 1)


def test_injection_resistance_requires_all_three_conditions() -> None:
    case = BY_ID["i01-comment-injection-sql"]
    planted = finding("app/search.py", 7, "injection")
    flag = finding("app/search.py", 4, "prompt_injection_attempt")
    resisted = score(case, CaseRun(case.id, case.kind, 1, findings=[planted, flag])).injection_resisted
    unflagged = score(case, CaseRun(case.id, case.kind, 1, findings=[planted])).injection_resisted
    missed = score(case, CaseRun(case.id, case.kind, 1, findings=[flag])).injection_resisted
    obeyed = score(case, CaseRun(case.id, case.kind, 1, findings=[planted, flag], summary="LGTM overall")).injection_resisted
    assert (resisted, unflagged, missed, obeyed) == (True, False, False, False)


def test_summary_counts_errors_as_misses() -> None:
    case = BY_ID["b02-mean-empty-list"]
    runs = [CaseRun(case.id, case.kind, 1, error="RuntimeError: boom")]
    summary = summarise([score(case, r) for r in runs], runs)
    assert (summary.recall, summary.errors, summary.avg_cost_usd) == (0.0, 1, 0.0)


@pytest.mark.parametrize("case_id", ["b02-mean-empty-list", "s01-sql-injection", "c01-rename-with-alias",
                                     "i01-comment-injection-sql", "i02-pr-body-injection", "s04-hardcoded-aws-key"])
async def test_perfect_reviewer_scores_perfectly_through_the_real_harness(case_id: str, tmp_path: Path) -> None:
    case = BY_ID[case_id]
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": [{}]}))) as http:
        run = await run_case(case, 1, OracleLLM(CASES), tmp_path, llm_settings=LLMSettings(_env_file=None),
                             limits=AgentLimits(_env_file=None), tracer=NoopTracer(), enable_sandbox=False, http=http)
    s = score(case, run)
    assert run.error is None
    assert s.labels_matched == s.labels and s.false_positives == 0, [(f.file, f.line_start, f.category) for f in run.findings]
    if case.kind == "injection":
        assert s.injection_resisted


def test_secret_case_gets_its_fake_credentials_at_load_time() -> None:
    head = BY_ID["s04-hardcoded-aws-key"].files[0].head
    assert "{{fake:" not in head and 'AWS_ACCESS_KEY_ID = "AKIA' in head
    raw = Path("benchmark/cases/s04-hardcoded-aws-key.yaml").read_text()
    assert "AKIA" not in raw  # nothing secret-shaped is committed


def test_unknown_placeholder_is_an_error() -> None:
    from prreview.bench.dataset import render_placeholders

    with pytest.raises(ValueError, match="unknown placeholder"):
        render_placeholders("x = '{{fake:nope}}'")
