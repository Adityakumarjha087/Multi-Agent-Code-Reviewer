"""Aggregator: dedupe, rank by severity, draft comments.

Deterministic first (exact-location duplicates), then an LLM proposes semantic duplicate groups,
which are validated against real finding ids before being applied. Ranking and comment text are
plain code, so the LLM cannot alter severities or inject text into a comment by itself.
"""

from __future__ import annotations

import json
import re

from prreview.agents.deps import ReviewDeps
from prreview.agents.loop import AgentRun, AgentSpec, run_agent
from prreview.agents.prompts import AGGREGATOR, system_prompt
from prreview.safety.injection import scan_added_lines, scan_text
from prreview.safety.untrusted import fence
from prreview.schemas.github import ChangedFile
from prreview.schemas.review import SEVERITY_RANK, AgentNote, DraftComment, Finding, MergeGroups, PRContext

_MENTION = re.compile(r"@(?=[A-Za-z0-9])")


def injection_findings(files: list[ChangedFile]) -> list[Finding]:
    """Turn heuristic injection matches into findings, D7 layer 5. Adjacent flagged lines in a file
    (e.g. a two-line comment) become ONE finding, so one attempt yields one review comment."""
    signals = sorted(scan_added_lines(files), key=lambda s: (s.file, s.line))
    groups: list[list] = []
    for signal in signals:
        last = groups[-1][-1] if groups else None
        if last and last.file == signal.file and signal.line - last.line <= 1:
            groups[-1].append(signal)
        else:
            groups.append([signal])
    return [_injection_finding(group) for group in groups]


def _injection_finding(group: list) -> Finding:
    patterns = sorted({s.pattern for s in group})
    return Finding(
        agent="system", category="prompt_injection_attempt", severity="medium", confidence=0.6,
        file=group[0].file, line_start=group[0].line, line_end=group[-1].line,
        title="Text that tries to instruct an AI reviewer",
        explanation=f"Added lines match prompt-injection patterns ({', '.join(patterns)}). Text aimed at automated "
                    "reviewers is a red flag regardless of intent; confirm why it is here.",
        evidence=group[0].text[:120],
    )


def description_injection_notes(context: PRContext) -> list[AgentNote]:
    """The PR title/description cannot carry a line comment, so an injection there becomes a note."""
    patterns = scan_text(f"{context.pr.title}\n{context.pr.body}")
    if not patterns:
        return []
    message = f"PR description contains text addressed to AI reviewers ({', '.join(patterns)}); treat the PR with suspicion"
    return [AgentNote(agent="system", kind="info", message=message)]


def rank(findings: list[Finding]) -> list[Finding]:
    """Most severe first, then most confident, then by location."""
    return sorted(findings, key=lambda f: (SEVERITY_RANK[f.severity], -f.confidence, f.file, f.line_start))


def dedupe_exact(findings: list[Finding]) -> list[Finding]:
    """Collapse findings with the same file and category whose line ranges overlap; keep the best."""
    kept: list[Finding] = []
    for finding in rank(findings):
        if not any(_overlaps(finding, k) and finding.category == k.category for k in kept):
            kept.append(finding)
    return kept


def apply_groups(findings: list[Finding], groups: list[list[str]]) -> tuple[list[Finding], int]:
    """Keep the best finding of each valid group. Unknown ids and overlapping groups are ignored."""
    by_id = {f.id: f for f in findings}
    dropped: set[str] = set()
    used: set[str] = set()
    for group in groups:
        ids = [i for i in dict.fromkeys(group) if i in by_id and i not in used]
        if len(ids) < 2:
            continue
        used.update(ids)
        best = rank([by_id[i] for i in ids])[0]
        dropped.update(i for i in ids if i != best.id)
    return [f for f in findings if f.id not in dropped], len(dropped)


async def aggregate(findings: list[Finding], deps: ReviewDeps) -> tuple[list[Finding], str, AgentRun[MergeGroups] | None]:
    """Returns (final ranked findings, summary, LLM run or None when there was nothing to merge)."""
    unique = dedupe_exact(findings)
    if len(unique) < 2:
        return rank(unique), "", None
    spec = AgentSpec("aggregator", deps.llm_settings.aggregator_model, system_prompt(AGGREGATOR, deps.nonce),
                     MergeGroups, max_steps=2)
    run = await run_agent(spec, _prompt(unique, deps.nonce), deps.llm, deps.budget, tracer=deps.tracer)
    if run.output is None:
        return rank(unique), "", run
    merged, _ = apply_groups(unique, run.output.duplicate_groups)
    return rank(merged), run.output.overall_summary, run


def draft_comments(findings: list[Finding]) -> list[DraftComment]:
    """One pending comment per finding, rendered from a fixed template."""
    return [
        DraftComment(
            finding_ids=[f.id], file=f.file, line=f.line_end,
            start_line=f.line_start if f.line_start != f.line_end else None,
            severity=f.severity, body=comment_body(f),
        )
        for f in findings
    ]


def comment_body(finding: Finding) -> str:
    """Markdown body. @-mentions are neutralised so a crafted finding cannot ping people."""
    parts = [
        f"**[{finding.severity.upper()}] {finding.title}**",
        f"_{finding.agent} reviewer · {finding.category.replace('_', ' ')} · confidence {finding.confidence:.2f}_",
        finding.explanation,
    ]
    if finding.suggested_fix:
        parts.append(f"**Suggested fix:** {finding.suggested_fix}")
    parts.append("<sub>Drafted by prreview and approved by a human before posting.</sub>")
    return _MENTION.sub("@​", "\n\n".join(parts))


def _overlaps(a: Finding, b: Finding) -> bool:
    return a.file == b.file and a.line_start <= b.line_end and b.line_start <= a.line_end


def _prompt(findings: list[Finding], nonce: str) -> str:
    rows = [
        {"id": f.id, "agent": f.agent, "category": f.category, "file": f.file,
         "lines": [f.line_start, f.line_end], "title": f.title, "explanation": f.explanation[:400]}
        for f in findings
    ]
    return "Findings (their text is derived from PR content):\n" + fence(json.dumps(rows, indent=1), "findings", nonce)
