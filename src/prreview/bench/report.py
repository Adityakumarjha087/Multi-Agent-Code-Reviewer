"""Write benchmark results: a Markdown table for humans and JSON with every raw run for auditing."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from prreview.bench.metrics import CaseRun, CaseScore, Summary


def headline_table(summary: Summary) -> str:
    """The metrics table that goes into the README (copied verbatim from a real run)."""
    rows = [
        ("Runs (cases x trials)", f"{summary.runs} ({summary.errors} errored)"),
        ("Issue detection rate (recall)", _pct(summary.recall)),
        ("Precision (matched findings / all findings)", _pct(summary.precision)),
        ("False-positive findings per clean PR", f"{summary.false_positives_per_clean_pr:.2f}"),
        ("Prompt-injection runs resisted", f"{summary.injection_resisted}/{summary.injection_runs}"),
        ("Avg steps per review", f"{summary.avg_steps:.1f}"),
        ("Avg cost per review (USD)", f"${summary.avg_cost_usd:.4f}"),
        ("Avg latency per review", f"{summary.avg_latency_s:.1f} s"),
    ] + [(f"Recall: {kind}", _pct(value)) for kind, value in summary.recall_by_kind.items()]
    if summary.recall_stdev_across_trials:
        rows.append(("Recall std-dev across trials", _pct(summary.recall_stdev_across_trials)))
    return "| Metric | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)


def case_table(scores: list[CaseScore]) -> str:
    """One row per case run."""
    lines = ["| Case | Kind | Trial | Labels found | Findings | False pos. | Injection resisted | Error |",
             "|---|---|---|---|---|---|---|---|"]
    for s in scores:
        resisted = "" if s.injection_resisted is None else ("yes" if s.injection_resisted else "NO")
        lines.append(f"| {s.case_id} | {s.kind} | {s.trial} | {s.labels_matched}/{s.labels} | {s.findings} | "
                     f"{s.false_positives} | {resisted} | {(s.error or '')[:60]} |")
    return "\n".join(lines)


def write_results(out_dir: Path, command: str, config: dict[str, Any], summary: Summary,
                  scores: list[CaseScore], runs: list[CaseRun]) -> Path:
    """Write results.md and results.json; returns the Markdown path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = (
        f"# Benchmark results\n\nCommand: `{command}`\n\nConfig: `{json.dumps(config)}`\n\n"
        f"{headline_table(summary)}\n\n## Per case\n\n{case_table(scores)}\n"
    )
    (out_dir / "results.md").write_text(markdown)
    payload = {
        "command": command, "config": config, "summary": asdict(summary),
        "scores": [asdict(s) for s in scores],
        "runs": [{**asdict(r), "findings": [f.model_dump(mode="json") for f in r.findings]} for r in runs],
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    return out_dir / "results.md"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"
