#!/usr/bin/env python3
"""agent_merge_policy.py -- which changes the driver agent (Hermes) may merge into main by itself.

Owner ruling 2026-10-01 ("Routine merges"): Hermes merges its own routine work -- results data, docs, worklogs,
tests and findings marked PROVISIONAL -- once the gates pass and two MiMo Pro reviewers approve. Claude keeps the
simulator code, default flips, preregistrations / bars / scorers / seed lists, the gates themselves, and confirming
any PASS. `tools/agent_merge.sh` runs this check before it merges; anything not listed as routine below is reserved.

    python3 tools/agent_merge_policy.py research/<branch> [<sha>]   # exit 0 = routine, 1 = reserved (listed), 2 = usage
    python3 tools/agent_merge_policy.py --staged                     # pre-commit backstop for the agent's commits on main
(with <sha>, that exact commit is checked -- agent_merge.sh pins the commit it then merges)

Routine (everything else, and every deletion or rename, is reserved):
  research/findings/raw/**              NEW results files only (an edit would rewrite evidence); _provenance/ below
  **/*WORKLOG*.md                        worklogs: add or modify
  docs/drafts/**/*.{md,txt,csv,tsv,json} draft docs and their data: add or modify, except HANDOFF.md and *PREREG*
  research/findings/raw/_provenance/*.jsonl  the run ledger: APPEND-ONLY growth of an existing ledger (2026-10-01)
  docs/.vikunja_sync                     the board-sync stamp: APPEND-ONLY (after a vikunja-skill board sync)
  tests/test_*.py                        NEW test files only (editing an existing test could weaken it)
  research/findings/*.md                 NEW findings only, and only if the file says PROVISIONAL; never *PREREG*
  research/runners/_*.py                 NEW draft runner copies only (underscore-prefixed); never an edit
research/coordination/ is not merged at all: the runbook has Hermes edit those files on main, and gates read several.
"""
from __future__ import annotations

import fnmatch
import subprocess
import sys
from typing import Callable, Iterable


def _is_prereg(path: str) -> bool:
    return "PREREG" in path.rsplit("/", 1)[-1].upper()


def reason_reserved(status: str, path: str, content: Callable[[str], str], mode: str = "100644",
                    staged: bool = False, old: Callable[[str], str] | None = None) -> str | None:
    """None if the change is routine, else why it is reserved. `status` is git's status letter (A, M, D, R...),
    `mode` the destination file mode. `staged` = a direct commit on main (the pre-commit backstop), where the
    coordination files the runbook has the agent edit on main are allowed. `old` reads the path before the change;
    without it an append-only ledger change cannot be proven and stays reserved."""
    name = path.rsplit("/", 1)[-1]
    if status not in ("A", "M"):
        return f"{status}: only additions and modifications are routine (no delete, rename, copy or type change)"
    if mode not in ("100644", "100755"):
        return f"mode {mode}: only plain files are routine (no symlink or submodule)"
    if path == "docs/.vikunja_sync":   # the board-sync stamp the vikunja-freshness gate reads (owner 2026-10-01)
        if status == "M" and old is not None and content(path).startswith(old(path)):
            return None
        return "the board-sync stamp may only grow by appended lines"
    if staged and path.startswith("research/coordination/"):
        return None
    if _is_prereg(path):
        return "a preregistration is Claude's"
    if path.startswith("research/findings/raw/"):
        if path.startswith("research/findings/raw/_provenance/"):
            if status == "M" and name.endswith(".jsonl") and path.count("/") == 4 and old is not None \
                    and content(path).startswith(old(path)):
                return None
            return "the provenance ledger may only grow by appended lines (no new ledger, no rewrite)"
        return None if status == "A" else "editing an existing results file would rewrite evidence"
    if path.startswith("research/coordination/"):
        return "coordination files are edited on main (several are gate inputs), never merged"
    if fnmatch.fnmatch(name, "*WORKLOG*.md"):
        return None
    if path.startswith("docs/drafts/") and name.rsplit(".", 1)[-1] in ("md", "txt", "csv", "tsv", "json"):
        return "a HANDOFF is Claude's" if name == "HANDOFF.md" else None
    if path.startswith("tests/") and path.count("/") == 1 and fnmatch.fnmatch(name, "test_*.py"):
        return None if status == "A" else "editing an existing test is reserved"
    if path.startswith("research/findings/") and path.count("/") == 2 and name.endswith(".md"):
        if status != "A":
            return "editing an existing finding is reserved"
        return None if "PROVISIONAL" in content(path) else "a new finding must say PROVISIONAL"
    if path.startswith("research/runners/") and path.count("/") == 2 and fnmatch.fnmatch(name, "_*.py"):
        return None if status == "A" else "editing a runner is reserved"
    return "outside the routine classes"


def reserved_changes(changes: Iterable[tuple], content: Callable[[str], str], staged: bool = False,
                     old: Callable[[str], str] | None = None) -> list[tuple[str, str]]:
    """`changes` holds (status, path) or (status, path, dst_mode) tuples."""
    out = []
    for status, path, *mode in changes:
        why = reason_reserved(status, path, content, mode[0] if mode else "100644", staged, old)
        if why:
            out.append((path, why))
    return out


def parse_raw_z(text: str) -> list[tuple[str, str, str]]:
    """Parse `git diff --raw -z --no-renames` into (status, path, dst_mode); -z keeps odd paths unquoted."""
    parts, out, i = text.split("\0"), [], 0
    while i + 1 < len(parts) and parts[i].startswith(":"):
        meta = parts[i][1:].split()
        out.append((meta[4][0], parts[i + 1], meta[1]))
        i += 2
    return out


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def branch_changes(rev: str) -> list[tuple[str, str, str]]:
    """The changes `rev` introduces since its merge base with main (three-dot). A PRE-SCREEN only: with several merge
    bases or renames on main this can differ from the merge result, which `--staged` checks before the commit."""
    return parse_raw_z(_git("diff", "--raw", "-z", "--no-renames", f"refs/heads/main...{rev}"))


def staged_changes() -> list[tuple[str, str, str]]:
    """What the commit being made adds to HEAD: the real merge result, or a direct commit."""
    return parse_raw_z(_git("diff", "--cached", "--raw", "-z", "--no-renames", "HEAD"))


def main(argv: list[str]) -> int:
    if argv[1:] == ["--staged"]:   # the pre-commit backstop: the agent's commits on main, merges included
        bad = reserved_changes(staged_changes(), lambda p: _git("show", f":{p}"), staged=True,
                               old=lambda p: _git("show", f"HEAD:{p}"))
        if bad:
            print(f"⛔ agent commit on main touches {len(bad)} reserved path(s); that change is Claude's:")
            for path, why in bad[:40]:
                print(f"   {path}: {why}")
            return 1
        return 0
    if len(argv) not in (2, 3) or not argv[1].startswith("research/"):
        print("usage: agent_merge_policy.py research/<branch> [<sha>]", file=sys.stderr)
        return 2
    branch = argv[1]
    rev = argv[2] if len(argv) == 3 else f"refs/heads/{branch}"
    changes = branch_changes(rev)
    if not changes:
        print(f"agent-merge policy: {branch} has no changes against main")
        return 1
    base = _git("merge-base", "refs/heads/main", rev).strip()
    bad = reserved_changes(changes, lambda p: _git("show", f"{rev}:{p}"), old=lambda p: _git("show", f"{base}:{p}"))
    if bad:
        print(f"⛔ agent-merge policy: {branch} touches {len(bad)} reserved path(s); this merge is Claude's:")
        for path, why in bad[:40]:
            print(f"   {path}: {why}")
        if len(bad) > 40:
            print(f"   ... and {len(bad) - 40} more")
        return 1
    print(f"agent-merge policy: OK, {len(changes)} routine change(s) on {branch}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
