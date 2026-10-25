"""Command-line entry point: `python -m prreview <command>`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from prreview.llm.client import MissingCredentialsError
from prreview.review.runner import ReviewFailedError, ReviewOutcome, parse_pr_url


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a subcommand."""
    parser = argparse.ArgumentParser(prog="prreview")
    commands = parser.add_subparsers(dest="command", required=True)
    review = commands.add_parser("review", help="review one pull request and print draft comments")
    review.add_argument("url", help="https://github.com/<owner>/<repo>/pull/<n>")
    review.add_argument("--no-sandbox", action="store_true", help="skip running tests in Docker")
    review.add_argument("--out", type=Path, help="also write the full review state as JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.command == "review":
        return _review(args.url, not args.no_sandbox, args.out)
    return 2


def _review(url: str, sandbox: bool, out: Path | None) -> int:
    from prreview.review.factory import review_pr

    try:
        outcome = asyncio.run(review_pr(parse_pr_url(url), enable_sandbox=sandbox))
    except (MissingCredentialsError, ReviewFailedError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(render(outcome))
    if out:
        out.write_text(json.dumps(outcome.state.model_dump(mode="json"), indent=2))
    return 0


def render(outcome: ReviewOutcome) -> str:
    """Human-readable summary of a review."""
    state = outcome.state
    lines = [
        f"Review {state.review_id}: {len(state.drafts)} draft comment(s), {len(state.steps)} steps, "
        f"${state.total_cost_usd:.4f}, {outcome.latency_s}s",
    ]
    if state.summary:
        lines.append(f"Summary: {state.summary}")
    for draft in state.drafts:
        lines.append(f"\n[{draft.severity.upper()}] {draft.file}:{draft.line}\n{draft.body}")
    if state.notes:
        lines.append("\nNotes:")
        lines += [f"- {n.agent}/{n.kind}: {n.message}" for n in state.notes]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
