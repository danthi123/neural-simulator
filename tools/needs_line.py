#!/usr/bin/env python3
"""needs_line.py -- compose (and optionally append) one NEEDS-CLAUDE line for the driver agent (Claude, 2026-10-04).

Two failures this prevents, both seen in session 6: NEEDS lines far over the runbook's 300-character format (8 of 25
open lines), and a NEEDS line re-raising a question a section-6 RULING had already answered (T12, closed 10-01, was
re-filed 10-04). So the helper (1) builds the exact format, (2) refuses a line over 300 characters (put the detail
in the packet's WORKLOG and cite it), and (3) prints every RULING in section 6 that mentions the key and refuses to
write until you pass --rulings-read (read them first: the answer may already be there).

    python3 tools/needs_line.py --key T12 --kind STOP --text "tone-bg finding missing on main; blocks: T12"
    python3 tools/needs_line.py --key T12 --kind STOP --text "..." --rulings-read --write
Exit 0 = printed (and appended with --write); 1 = refused (reason printed); 2 = usage.

Fourth refusal (Claude review 2026-10-06; uncommitted files in Hermes's worktrees blocked Claude's merges twice): an
ASK or RESULT line that names a branch for Claude to merge or confirm is refused while that branch's worktree
(`git worktree list`) has uncommitted tracked changes or untracked, non-ignored files; they are printed. Commit or
`git restore` them (or write a patch outside the repo, RULING no-reserved-leftovers), then re-run. A token counts as a
branch only if `refs/heads/<token>` exists, so file paths never trigger it; `--branch <name>` names one explicitly.
Escape hatch, recorded in the line itself so Claude sees it: `--dirty-ok "<reason>"`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNBOOK = os.path.join(ROOT, "research", "coordination", "LOCAL_LLM_RUNBOOK.md")
MAX_LEN = 300
KINDS = ("STOP", "ASK", "RESULT", "INFO")
DIRTY_KINDS = ("ASK", "RESULT")      # the kinds that hand Claude a branch to merge or confirm
BRANCH_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9._-]+)+")


def _git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return r.returncode, r.stdout


def branches_named(text, repo, extra=()):
    """Local branches the line names: every path-like token with `refs/heads/<token>` present, plus `extra`."""
    out = []
    for tok in list(BRANCH_TOKEN.findall(text)) + list(extra):
        tok = tok.rstrip(".")
        if tok not in out and _git(repo, "show-ref", "--verify", "--quiet", "refs/heads/" + tok)[0] == 0:
            out.append(tok)
    return out


def worktree_of(branch, repo):
    """Path of the worktree that has `branch` checked out, or None."""
    rc, txt = _git(repo, "worktree", "list", "--porcelain")
    path = None
    for line in txt.splitlines():
        if line.startswith("worktree "):
            path = line[len("worktree "):]
        elif line == "branch refs/heads/" + branch:
            return path
    return None


def dirty_files(branch, repo):
    """`git status --porcelain` lines (tracked changes and untracked non-ignored files) of `branch`'s worktree; []
    when it is clean or has no worktree."""
    wt = worktree_of(branch, repo)
    if not wt or not os.path.isdir(wt):
        return []
    rc, txt = _git(wt, "status", "--porcelain", "--untracked-files=normal")
    return txt.splitlines() if rc == 0 else []


def compose(key: str, kind: str, text: str, n: str = "", when: _dt.datetime | None = None) -> str:
    when = when or _dt.datetime.now()
    tag = f"{kind} {n}".strip()
    return f"- {when:%Y-%m-%d %H:%M} [{key}] {tag}: {' '.join(text.split())}"


def rulings_for(key: str, runbook_text: str) -> list[str]:
    """Every section-6 ruling (a `- <name>: ...` item, with its continuation lines) that mentions `key`."""
    try:
        sec = runbook_text.split("\n## 6.", 1)[1].split("\n## NEEDS CLAUDE", 1)[0]
    except IndexError:
        return []
    items, cur = [], []
    for line in sec.splitlines():
        if re.match(r"^- \S", line):
            if cur:
                items.append("\n".join(cur))
            cur = [line]
        elif cur and line.startswith("  "):
            cur.append(line)
    if cur:
        items.append("\n".join(cur))
    pat = re.compile(r"(?<![\w-])" + re.escape(key) + r"(?![\w-])", re.IGNORECASE)
    return [it for it in items if pat.search(it)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--key", required=True, help="task or packet key, e.g. T12 or s1-c4-void")
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--n", default="", help="the HANDOFF's NEEDS-CLAUDE item, e.g. '#2'")
    ap.add_argument("--text", required=True, help="what you saw; tried: what; blocks: what")
    ap.add_argument("--rulings-read", action="store_true", help="you have read the rulings this prints")
    ap.add_argument("--write", action="store_true", help="append the line to the NEEDS CLAUDE section")
    ap.add_argument("--runbook", default=RUNBOOK)
    ap.add_argument("--branch", action="append", default=[], help="a branch this ASK/RESULT hands Claude (repeatable)")
    ap.add_argument("--dirty-ok", default=None, metavar="REASON",
                    help="file anyway though a named branch's worktree is dirty; REASON is written into the line")
    ap.add_argument("--repo", default=ROOT, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9§._/ -]{1,60}", a.key):
        print("bad --key", file=sys.stderr)
        return 2
    text_in = a.text
    dirty = {}
    if a.kind in DIRTY_KINDS:
        for br in branches_named(a.text, a.repo, a.branch):
            files = dirty_files(br, a.repo)
            if files:
                dirty[br] = files
    if dirty and a.dirty_ok is not None:
        if not a.dirty_ok.strip():
            print("bad --dirty-ok: give the reason", file=sys.stderr)
            return 2
        text_in = "%s [dirty-ok: %s]" % (a.text, a.dirty_ok.strip())
    line = compose(a.key, a.kind, text_in, a.n)
    text = open(a.runbook, encoding="utf-8").read()
    found = rulings_for(a.key, text)
    print(line)
    print(f"({len(line)} characters, limit {MAX_LEN})")
    if found:
        print(f"\n{len(found)} section-6 RULING(s) mention [{a.key}]; read them, the answer may already be there:")
        for it in found:
            print("  " + it.replace("\n", "\n  "))
    if dirty and a.dirty_ok is None:
        for br, files in dirty.items():
            print("\n%s: worktree %s has %d uncommitted/untracked path(s):" % (br, worktree_of(br, a.repo), len(files)))
            for f in files[:20]:
                print("  " + f)
            if len(files) > 20:
                print("  ... %d more" % (len(files) - 20))
        print("\n⛔ REFUSED: a branch this line hands to Claude has uncommitted files (they blocked Claude's merges "
              "twice). Commit them, or `git restore` / `git clean` the ones you do not need (a Claude-reserved file: "
              "write a patch outside the repo, RULING no-reserved-leftovers), then re-run. Escape: --dirty-ok "
              "\"<reason>\".", file=sys.stderr)
        return 1
    if len(line) > MAX_LEN:
        print(f"\n⛔ REFUSED: {len(line)} > {MAX_LEN} characters. Put the detail in the packet WORKLOG and cite it.",
              file=sys.stderr)
        return 1
    if not a.write:
        return 0
    if found and not a.rulings_read:
        print("\n⛔ REFUSED: rulings exist for this key; read them, then re-run with --rulings-read", file=sys.stderr)
        return 1
    if "\n## NEEDS CLAUDE" not in text:
        print("⛔ REFUSED: no '## NEEDS CLAUDE' section in the runbook", file=sys.stderr)
        return 1
    with open(a.runbook, "a", encoding="utf-8") as f:
        f.write(("" if text.endswith("\n") else "\n") + line + "\n")
    print("appended to NEEDS CLAUDE (commit it: `runbook: NEEDS [<key>] ...`)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
