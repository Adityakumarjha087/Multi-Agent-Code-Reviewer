"""`python -m prreview.bench validate | run` - the single benchmark command.

`run` calls the real Claude API (cost!), so it refuses to start without `--yes`.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from prreview.bench.dataset import Case, label_problems, load_cases
from prreview.bench.harness import run_case
from prreview.bench.metrics import CaseRun, score, summarise
from prreview.bench.report import headline_table, write_results
from prreview.config import agent_limits, llm_settings
from prreview.github.diff import new_side_lines
from prreview.llm.client import AnthropicLLM, LLMClient, MissingCredentialsError
from prreview.observability.tracing import build_tracer

CASES_DIR = Path("benchmark/cases")
RESULTS_DIR = Path("benchmark/results")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(prog="python -m prreview.bench")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="check labels and print the labelled lines for manual review")
    run = sub.add_parser("run", help="run the benchmark against the real model and write a results table")
    run.add_argument("--trials", type=int, default=1, help="repeat every case N times (LLMs are non-deterministic)")
    run.add_argument("--cases", default="*", help="glob over case ids, e.g. 'i0*'")
    run.add_argument("--no-sandbox", action="store_true", help="skip Docker test runs")
    run.add_argument("--yes", action="store_true", help="confirm that this makes paid API calls")
    args = parser.parse_args(argv)
    cases = load_cases(CASES_DIR, f"{args.cases}.yaml" if args.command == "run" else "*.yaml")
    if args.command == "validate":
        return _validate(cases)
    return _run(cases, args)


def _validate(cases: list[Case]) -> int:
    problems = [p for c in cases for p in label_problems(c)]
    for case in cases:
        files = {f.path: f for f in case.changed_files()}
        for label in case.labels:
            lines = new_side_lines(files[label.file].patch) if label.file in files else {}
            text = " | ".join(lines.get(n, "?").strip() for n in range(label.lines[0], label.lines[1] + 1))
            print(f"{case.id:38} {label.file}:{label.lines[0]}-{label.lines[1]}  {text[:100]}")
    kinds = {k: sum(1 for c in cases if c.kind == k) for k in ("bug", "security", "clean", "injection")}
    print(f"\n{len(cases)} cases {kinds}, {sum(len(c.labels) for c in cases)} labels")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    return 1 if problems else 0


def _run(cases: list[Case], args: argparse.Namespace) -> int:
    conf = llm_settings()
    models = sorted({conf.planner_model, conf.bug_model, conf.security_model, conf.test_model, conf.aggregator_model})
    plan = f"{len(cases)} cases x {args.trials} trial(s) with {', '.join(models)} (effort={conf.effort})"
    if not args.yes:
        print(f"This would run {plan} and make paid Claude API calls. Re-run with --yes to proceed.")
        return 2
    try:
        llm = AnthropicLLM(conf)
    except MissingCredentialsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Running {plan} ...")
    runs = asyncio.run(_run_all(cases, args.trials, llm, not args.no_sandbox))
    by_id = {c.id: c for c in cases}
    scores = [score(by_id[r.case_id], r) for r in runs]
    summary = summarise(scores, runs)
    command = "python -m prreview.bench " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    config = {
        "models": models, "effort": conf.effort, "trials": args.trials, "sandbox": not args.no_sandbox,
        "dataset_sha256": _dataset_hash(cases), "limits": agent_limits().model_dump(),
    }
    out = RESULTS_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = write_results(out, command, config, summary, scores, runs)
    print(f"\n{headline_table(summary)}\n\nFull results: {path}")
    return 0


async def _run_all(cases: list[Case], trials: int, llm: LLMClient, sandbox: bool) -> list[CaseRun]:
    runs: list[CaseRun] = []
    tracer, limits, conf = build_tracer(), agent_limits(), llm_settings()
    work = Path(".workspaces/bench")
    async with httpx.AsyncClient(timeout=20) as http:
        for trial in range(1, trials + 1):
            for case in cases:
                run = await run_case(case, trial, llm, work / case.id, llm_settings=conf, limits=limits,
                                     tracer=tracer, enable_sandbox=sandbox, http=http)
                status = run.error or f"{len(run.findings)} findings, ${run.cost_usd:.4f}"
                print(f"  [{trial}] {case.id}: {status}")
                runs.append(run)
    tracer.flush()
    return runs


def _dataset_hash(cases: list[Case]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(case.model_dump_json().encode())
    return digest.hexdigest()[:16]


if __name__ == "__main__":
    sys.exit(main())
