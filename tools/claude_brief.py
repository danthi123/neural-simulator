#!/usr/bin/env python3
"""claude_brief.py -- one command for the START of a Claude check-in. Read-only, plain text, fast.

Why (Claude's 10-06 review of Hermes): each open NEEDS CLAUDE line cost a branch/diff/data discovery round before the
actual judgment could start. This prints that discovery up front, per open line of the runbook's `## NEEDS CLAUDE`:

    python tools/claude_brief.py [--runbook <path>] [--now 'YYYY-MM-DD HH:MM'] [--max-files 15] [--key <KEY>]

Per line: key, age in hours, type, the line text; the branch it names (a token with `refs/heads/<token>`), else the
packet's branch inferred from the key (research/<key>, research/draft-<key>, claude-review/<key>, claude/<key>), else
the branch that contains a commit sha the line cites (marked "inferred"). Per branch: tip, ahead/behind main,
merge-base, diffstat vs main (top N files), worktree dirty status (the files, if dirty), whether it fast-forwards from
its base (a claude-review/<x> or claude/<x> branch from research/<x>; a research/* branch from main), a merge-tree
conflict probe against main, the cited file paths that exist or are MISSING (on the branch tip, else on main), and the
section-6 RULINGs that already mention the key. It changes nothing the work depends on: no checkout, no fetch, no worktree, no ref, no file write (the conflict
probe, `git merge-tree --write-tree`, may leave unreachable tree objects that git gc removes).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import needs_line as NL  # noqa: E402

RUNBOOK = NL.RUNBOOK
LINE_RE = re.compile(r"^- (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}) \[([^\]]+)\] (\w+)(?: ([#(][^:]*?))?: (.*)$")
PATH_RE = re.compile(r"(?<![\w./-])((?:research|docs|tools|tests|webapp|sim|bridges|\.claude)/[\w./@+-]*[\w])")
SHA_RE = re.compile(r"(?<![0-9A-Za-z])([0-9a-f]{7,40})(?![0-9A-Za-z])")
MAIN = "main"


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def parse_needs(text):
    """[{when, key, kind, n, text, raw}] for each open line of the `## NEEDS CLAUDE` section."""
    try:
        sec = text.split("\n## NEEDS CLAUDE", 1)[1]
    except IndexError:
        return []
    sec = re.split(r"\n## ", sec, 1)[0]
    out = []
    for raw in sec.splitlines():
        m = LINE_RE.match(raw.strip())
        if not m:
            continue
        d, t, key, kind, n, body = m.groups()
        out.append({"when": _dt.datetime.strptime("%s %s" % (d, t), "%Y-%m-%d %H:%M"), "key": key, "kind": kind,
                    "n": (n or "").strip(), "text": body, "raw": raw.strip()})
    return out


def local_branches(repo):
    rc, txt = git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads")
    return set(txt.splitlines()) if rc == 0 else set()


def find_branch(line, branches, repo):
    """(branch, how): named in the text, inferred from the key, or inferred from a cited commit sha."""
    for tok in NL.BRANCH_TOKEN.findall(line["text"]):
        tok = tok.rstrip(".")
        if tok in branches:
            return tok, "named"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", line["key"]).strip("-")
    for cand in ("research/%s" % slug, "research/draft-%s" % slug, "claude-review/%s" % slug, "claude/%s" % slug):
        if cand in branches:
            return cand, "inferred from key"
    for sha in SHA_RE.findall(line["text"]):
        rc, _ = git(repo, "cat-file", "-e", sha + "^{commit}")
        if rc != 0 or is_ancestor(repo, sha, MAIN):
            continue                      # not a commit here, or already on main: it says nothing about a branch
        rc, txt = git(repo, "branch", "--contains", sha, "--format=%(refname:short)")
        hits = [b for b in txt.splitlines() if b and b != MAIN]
        if rc == 0 and hits:
            return sorted(hits, key=lambda b: (not b.startswith("research/"), b))[0], "inferred from sha %s" % sha[:9]
    return None, ""


def base_of(branch, branches):
    """The ref a branch is meant to fast-forward from."""
    for pre in ("claude-review/", "claude/"):
        if branch.startswith(pre):
            r = "research/" + branch[len(pre):]
            if r in branches:
                return r
    return MAIN


def is_ancestor(repo, a, b):
    return git(repo, "merge-base", "--is-ancestor", a, b)[0] == 0


def top_files(repo, branch, n):
    rc, txt = git(repo, "diff", "--numstat", "%s...%s" % (MAIN, branch))
    rows = []
    for ln in txt.splitlines():
        p = ln.split("\t")
        if len(p) == 3:
            add, rem = (0 if x == "-" else int(x) for x in p[:2])
            rows.append((add + rem, add, rem, p[2]))
    rows.sort(key=lambda r: -r[0])
    return len(rows), sum(r[1] for r in rows), sum(r[2] for r in rows), rows[:n]


def branch_report(repo, branch, branches, max_files):
    lines = []
    rc, tip = git(repo, "log", "-1", "--format=%h %s", branch)
    lines.append("    tip: %s" % tip[:150])
    rc, lr = git(repo, "rev-list", "--left-right", "--count", "%s...%s" % (MAIN, branch))
    behind, ahead = (lr.split() + ["?", "?"])[:2]
    rc, mb = git(repo, "merge-base", MAIN, branch)
    lines.append("    ahead %s / behind %s main; merge-base %s" % (ahead, behind, mb[:9] or "none"))
    nfiles, add, rem, top = top_files(repo, branch, max_files)
    lines.append("    diffstat vs main: %d file(s), +%d -%d" % (nfiles, add, rem))
    for _c, a, r, p in top:
        lines.append("      +%-6d -%-6d %s" % (a, r, p))
    if nfiles > len(top):
        lines.append("      ... %d more file(s)" % (nfiles - len(top)))
    wt = NL.worktree_of(branch, repo)
    if wt is None:
        lines.append("    worktree: none checked out")
    else:
        dirty = NL.dirty_files(branch, repo)
        if dirty:
            lines.append("    worktree: %s is DIRTY (%d path(s)): blocks a merge until committed or restored" % (wt, len(dirty)))
            lines += ["      " + d for d in dirty[:12]]
            if len(dirty) > 12:
                lines.append("      ... %d more" % (len(dirty) - 12))
        else:
            lines.append("    worktree: %s clean" % wt)
    base = base_of(branch, branches)
    ff = is_ancestor(repo, base, branch)
    lines.append("    fast-forward from base %s: %s" % (base, "yes" if ff else "NO (base is not an ancestor of the tip: "
                 "merge or rebase needed)"))
    if base != MAIN:
        lines.append("    base %s is %s main" % (base, "contained in" if is_ancestor(repo, base, MAIN) else "ahead of"))
    rc, out = git(repo, "merge-tree", "--write-tree", "--name-only", "--no-messages", MAIN, branch)
    if rc == 0:
        lines.append("    merge into main: clean")
    elif rc == 1:
        conflicted = [x for x in out.splitlines()[1:] if x.strip()]
        lines.append("    merge into main: CONFLICTS in %d file(s): %s" % (len(conflicted), ", ".join(conflicted[:8])))
    else:
        lines.append("    merge into main: not probed (git merge-tree unavailable)")
    return lines


def cited_paths(line, repo, branch, branches=()):
    """[(path, state)] for every repo path the line cites: on the branch tip, on main, or MISSING."""
    seen, out = set(), []
    for p in PATH_RE.findall(line["text"]):
        p = p.rstrip(".,;:)")
        if p in seen or p in branches:      # a branch name (research/<x>) is not a file path
            continue
        seen.add(p)
        if branch and git(repo, "cat-file", "-e", "%s:%s" % (branch, p))[0] == 0:
            out.append((p, "on %s" % branch))
        elif git(repo, "cat-file", "-e", "%s:%s" % (MAIN, p))[0] == 0:
            out.append((p, "on main"))
        elif os.path.exists(os.path.join(repo, p)):
            out.append((p, "in the main checkout only (untracked)"))
        else:
            out.append((p, "MISSING"))
    return out


def brief(runbook_text, repo, now, max_files=15, only_key=None):
    lines_in = parse_needs(runbook_text)
    if only_key:
        lines_in = [x for x in lines_in if x["key"] == only_key]
    branches = local_branches(repo)
    out = ["claude_brief: %d open NEEDS CLAUDE line(s) as of %s; main at %s" % (
        len(lines_in), now.strftime("%Y-%m-%d %H:%M"), git(repo, "rev-parse", "--short", MAIN)[1])]
    for i, ln in enumerate(lines_in, 1):
        age = (now - ln["when"]).total_seconds() / 3600.0
        out.append("")
        out.append("[%d] %s  age %.1f h  %s %s  (%s)" % (i, ln["key"], age, ln["kind"], ln["n"], ln["when"].strftime("%m-%d %H:%M")))
        out.append("    " + ln["text"][:300])
        br, how = find_branch(ln, branches, repo)
        if br:
            out.append("    branch: %s (%s)" % (br, how))
            out += branch_report(repo, br, branches, max_files)
        else:
            out.append("    branch: none named or inferable (no refs/heads token, no research/<key>, no cited sha on a branch)")
        for p, state in cited_paths(ln, repo, br, branches):
            out.append("    cites %s: %s" % (p, state))
        rl = NL.rulings_for(ln["key"], runbook_text)
        if rl:
            out.append("    section-6 rulings naming this key: %s" % ", ".join(
                (re.match(r"^- ([^:]+):", r).group(1) if re.match(r"^- ([^:]+):", r) else r[:30]) for r in rl))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runbook", default=RUNBOOK)
    ap.add_argument("--repo", default=ROOT)
    ap.add_argument("--now", default=None, help="'YYYY-MM-DD HH:MM' (default: the local clock)")
    ap.add_argument("--max-files", type=int, default=15)
    ap.add_argument("--key", default=None, help="only this NEEDS key")
    a = ap.parse_args(argv)
    try:
        text = open(a.runbook, encoding="utf-8").read()
    except OSError as e:
        print("claude_brief: cannot read %s: %s" % (a.runbook, e), file=sys.stderr)
        return 2
    now = _dt.datetime.strptime(a.now, "%Y-%m-%d %H:%M") if a.now else _dt.datetime.now()
    print(brief(text, a.repo, now, a.max_files, a.key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
