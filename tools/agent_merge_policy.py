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
  research/FAILURE_LOG.md                INSERT-ONLY: complete `| YYYY-MM-DD | failure | gate |` rows below the table
                                         header, every old line kept in order (Claude ruling 2026-10-04;
                                         gates/coverage still checks each row's gate)
  tests/test_*.py                        NEW test files only (editing an existing test could weaken it)
  research/findings/*.md                 NEW findings only, and only if the file says PROVISIONAL; never *PREREG*
  research/runners/_*.py                 NEW draft runner copies only (underscore-prefixed); never an edit
  GAP_CLOSURE_MISSION.md                 INSERT-ONLY board lines that claim no PASS/GO -- ONLY once the owner sets
                                         OWNER_APPROVED_BOARD_INSERTS = True below (shipped False, 2026-10-04)
research/coordination/ is not merged at all: the runbook has Hermes edit those files on main, and gates read several.
"""
from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from typing import Callable, Iterable


def _is_prereg(path: str) -> bool:
    return "PREREG" in path.rsplit("/", 1)[-1].upper()


# One complete FAILURE_LOG row: a date cell, a non-empty failure cell, a non-empty gate cell, nothing after.
_FAILURE_ROW = re.compile(r"^\| \d{4}-\d{2}-\d{2} \| \S.*\S \| \S.*\|$")


# OWNER DECISION (Claude session 6, 2026-10-04): when True, a merge or a direct commit on main may INSERT lines into
# GAP_CLOSURE_MISSION.md (every old line kept in order, no edit or deletion), as long as no inserted line claims a PASS
# or a GO (a NO-GO / UNDEFINED / FAIL / INFO line is fine). This is what made T5 (selection-seam-need-reversal)
# non-routine: its HANDOFF had the driver add one NO-GO board line. Shipped False: it widens what the driver may merge
# beyond bookkeeping, because that file steers every session.
OWNER_APPROVED_BOARD_INSERTS = True    # owner approved 2026-10-04
_CLAIMS_PASS = re.compile(r"(?<!NO-)(?<!NO )\bGO\b|\bPASS(?:ES|ED)?\b")


def _inserted_lines(new: str, old: str) -> tuple[list[str] | None, str | None]:
    """The lines `new` inserts into `old` with every old line kept in order, or (None, why not)."""
    import difflib
    o, n = old.splitlines(), new.splitlines()
    inserted = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, o, n, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag != "insert":
            return None, "existing lines changed (only inserted lines are routine)"
        inserted += n[j1:j2]
    return inserted, None


def _board_inserts(new: str, old: str) -> str | None:
    if not OWNER_APPROVED_BOARD_INSERTS:
        return "board lines are Claude's until the owner sets OWNER_APPROVED_BOARD_INSERTS"
    inserted, why = _inserted_lines(new, old)
    if why:
        return why
    if not any(line.strip() for line in inserted):
        return "no lines inserted"
    for line in inserted:
        if _CLAIMS_PASS.search(line):
            return f"an inserted board line claims a PASS or GO (Claude confirms those): {line[:80]!r}"
    return None


def _appended_failure_rows(new: str, old: str) -> str | None:
    """None if `new` is `old` with whole FAILURE_LOG rows inserted below the table header, else why not. Insertion
    anywhere below the header is allowed because a three-way merge places an older branch's appended rows where its
    copy of the file ended, which is mid-file once main has grown. Every old line must survive unchanged and in
    order; a row edit, a deletion, a heading, a blank line or a half row is refused."""
    import difflib
    if old and not old.endswith("\n"):
        return "the old file does not end in a newline, so an append would join its last row"
    if not new.endswith("\n"):
        return "the new file must end in a newline"
    o, n = old.splitlines(), new.splitlines()
    header = next((i for i, line in enumerate(o) if line.startswith("|---")), None)
    if header is None:
        return "the old file has no table header to append under"
    inserted = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, o, n, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag != "insert":
            return "existing lines changed (the failure log only takes inserted rows)"
        if i1 <= header:
            return "rows may only be inserted below the table header"
        inserted += n[j1:j2]
    if not inserted:
        return "no rows appended"
    for line in inserted:
        if not _FAILURE_ROW.match(line) or line.count(" | ") < 2:
            return f"not a complete `| date | failure | gate |` row: {line[:80]!r}"
    return None


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
    if path == "GAP_CLOSURE_MISSION.md":
        if status != "M" or old is None:
            return "the board may only take inserted lines (growth must be provable)"
        return _board_inserts(content(path), old(path))
    if path == "research/FAILURE_LOG.md":   # append-only rows (Claude ruling 2026-10-04, runbook agent-failure-log)
        if status != "M" or old is None:
            return "the failure log may only grow by appended rows (no new log; growth must be provable)"
        why = _appended_failure_rows(content(path), old(path))
        return None if why is None else f"failure log: {why}"
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
