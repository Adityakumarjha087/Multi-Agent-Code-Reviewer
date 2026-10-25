"""Deterministic security scanners that run before (and independently of) the security LLM (D8).

- Secret patterns on added lines. High-confidence formats (private keys, AWS/GitHub/Anthropic
  token shapes) become findings directly; generic `password = "..."` matches are only
  candidates for the LLM to triage.
- Known-vulnerable dependencies via the OSV.dev batch API (verified live 2026-10-09) for
  exactly-pinned (`==`) requirements added by the PR.

Running without the LLM means these findings survive even if the model refuses or the budget
runs out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
from packaging.requirements import InvalidRequirement, Requirement

from prreview.github.diff import new_side_lines
from prreview.schemas.github import ChangedFile
from prreview.schemas.review import Finding

OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
REQUIREMENT_FILE = re.compile(r"(^|/)(requirements[^/]*\.txt|[^/]*-requirements\.txt)$")

_HIGH_CONFIDENCE = {
    "private_key": re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\b(ghp|gho|ghs|ghu|ghr)_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    "slack_token": re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}"),
}
_GENERIC = re.compile(
    r"""(?i)\b(password|passwd|secret|api_?key|token|auth)\w*\s*[:=]\s*["'][^"'\s]{6,}["']"""
)


@dataclass(frozen=True)
class Candidate:
    """A scanner hit for the LLM to confirm or dismiss."""

    file: str
    line: int
    kind: str
    text: str


def scan_secrets(files: list[ChangedFile]) -> tuple[list[Finding], list[Candidate]]:
    """Return (direct findings for high-confidence formats, generic candidates for triage)."""
    findings: list[Finding] = []
    candidates: list[Candidate] = []
    for changed, line_no, text in _added_lines(files):
        kind = next((k for k, p in _HIGH_CONFIDENCE.items() if p.search(text)), None)
        if kind:
            findings.append(_secret_finding(changed.path, line_no, kind, text))
        elif _GENERIC.search(text):
            candidates.append(Candidate(changed.path, line_no, "generic_secret", text.strip()[:200]))
    return findings, candidates


async def scan_dependencies(files: list[ChangedFile], http: httpx.AsyncClient) -> list[Finding]:
    """Look up exactly-pinned requirements added by the PR in OSV. Raises on API failure."""
    pins = [(c.path, n, req) for c, n, req in _added_pins(files)]
    if not pins:
        return []
    queries = [{"package": {"name": req.name, "ecosystem": "PyPI"}, "version": _pinned_version(req)} for _, _, req in pins]
    response = await http.post(OSV_BATCH_URL, json={"queries": queries})
    response.raise_for_status()
    findings: list[Finding] = []
    for (path, line_no, req), result in zip(pins, response.json()["results"], strict=True):
        vuln_ids = [v["id"] for v in result.get("vulns", [])]
        if vuln_ids:
            findings.append(_dependency_finding(path, line_no, req, vuln_ids))
    return findings


def _added_lines(files: list[ChangedFile]):
    for changed in files:
        if not changed.patch:
            continue
        added = {n for hunk in changed.hunks for n in hunk.added_lines}
        for line_no, text in new_side_lines(changed.patch).items():
            if line_no in added:
                yield changed, line_no, text


def _added_pins(files: list[ChangedFile]):
    for changed, line_no, text in _added_lines([f for f in files if REQUIREMENT_FILE.search(f.path)]):
        try:
            req = Requirement(text.split("#", 1)[0].strip())
        except InvalidRequirement:
            continue
        if _pinned_version(req):
            yield changed, line_no, req


def _pinned_version(req: Requirement) -> str | None:
    specs = list(req.specifier)
    return specs[0].version if len(specs) == 1 and specs[0].operator == "==" else None


def _secret_finding(path: str, line_no: int, kind: str, text: str) -> Finding:
    return Finding(
        agent="security", category="hardcoded_secret", severity="critical", confidence=0.95,
        file=path, line_start=line_no, line_end=line_no,
        title=f"Hardcoded credential ({kind.replace('_', ' ')})",
        explanation="A credential in this format was added to the repository. Anyone with read access can use it, "
                    "and it stays in git history even after deletion.",
        evidence=mask_secret(text.strip()[:120], kind),
        suggested_fix="Revoke/rotate the credential now, remove it from the code, and load it from a secret store or env var.",
    )


def mask_secret(text: str, kind: str) -> str:
    """Keep the first 4 characters of the matched secret so humans can find it; hide the rest.

    The raw secret must not be copied into our database, traces, or the posted review comment.
    """
    return _HIGH_CONFIDENCE[kind].sub(lambda m: m.group(0)[:4] + "****[redacted]", text)


def _dependency_finding(path: str, line_no: int, req: Requirement, vuln_ids: list[str]) -> Finding:
    shown = ", ".join(vuln_ids[:5]) + (" ..." if len(vuln_ids) > 5 else "")
    return Finding(
        agent="security", category="unsafe_dependency", severity="high", confidence=0.9,
        file=path, line_start=line_no, line_end=line_no,
        title=f"{req.name}=={_pinned_version(req)} has known vulnerabilities",
        explanation=f"OSV lists {len(vuln_ids)} advisories for this exact version: {shown}.",
        evidence=str(req), suggested_fix="Upgrade to a version without these advisories (see https://osv.dev).",
    )
