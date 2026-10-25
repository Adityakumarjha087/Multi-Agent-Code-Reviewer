"""Render PR data into prompt text. Everything PR-controlled goes through `fence()`."""

from __future__ import annotations

from prreview.github.diff import HUNK_HEADER
from prreview.safety.untrusted import fence
from prreview.schemas.github import ChangedFile, PullRequest


def numbered_patch(patch: str, max_chars: int) -> str:
    """Show a patch with explicit NEW-file line numbers so the model can cite lines exactly.

    Added lines: `  42 + code`; context: `  43   code`; removed lines carry no new number: `     - code`.
    """
    out: list[str] = []
    new_line = 0
    for raw in patch.splitlines():
        header = HUNK_HEADER.match(raw)
        if header:
            new_line = int(header.group(3))
            out.append(raw)
        elif raw.startswith("-"):
            out.append(f"      - {raw[1:]}")
        elif raw.startswith("\\"):
            continue
        else:
            marker = "+" if raw.startswith("+") else " "
            out.append(f"{new_line:>5} {marker} {raw[1:] if raw[:1] in ('+', ' ') else raw}")
            new_line += 1
    text = "\n".join(out)
    if len(text) > max_chars:
        return text[:max_chars] + f"\n[diff truncated at {max_chars} characters; use read_file for the rest]"
    return text


def render_pr_header(pr: PullRequest, nonce: str) -> str:
    """PR metadata. Title and body are author-controlled, so they are fenced."""
    meta = (
        f"Repository: {pr.ref.owner}/{pr.ref.repo}  PR #{pr.ref.number}\n"
        f"Base {pr.base_ref} @ {pr.base_sha[:12]} <- head {pr.head_ref} @ {pr.head_sha[:12]}\n"
        f"Changed files: {pr.changed_files} (+{pr.additions}/-{pr.deletions})"
    )
    authored = f"Title: {pr.title}\n\n{pr.body or '(no description)'}"
    return f"{meta}\n\nPR title and description (author-written):\n{fence(authored, 'pr_description', nonce)}"


def render_files(files: list[ChangedFile], nonce: str, max_patch_chars: int, focus: dict[str, str] | None = None) -> str:
    """Numbered diffs for the given files, each fenced separately."""
    sections: list[str] = []
    for changed in files:
        title = f"### {changed.path} ({changed.status}, +{changed.additions}/-{changed.deletions})"
        if focus and changed.path in focus:
            title += f"\nPlanner focus (derived from PR content): {focus[changed.path]}"
        body = numbered_patch(changed.patch, max_patch_chars) if changed.patch else "(no textual diff: binary or too large)"
        # The path and the planner's focus are PR-derived too, so the whole section is fenced.
        sections.append(fence(f"{title}\n{body}", "diff", nonce))
    return "\n\n".join(sections)


def render_file_list(files: list[ChangedFile], nonce: str) -> str:
    """One line per changed file, for the planner. File names are author-controlled, so fenced."""
    listing = "\n".join(f"- {f.path} ({f.status}, +{f.additions}/-{f.deletions})" for f in files)
    return fence(listing, "changed_files", nonce)
