#!/usr/bin/env python3
"""Which of a merge commit's ADDED paths it merely CARRIES from main -- the merge scope of pre-commit GATE 2 (claims).

    git diff --cached --name-only --diff-filter=A | grep ... | python tools/merge_carried.py   # prints what to check

WHY (2026-10-28). GATE 2 runs tools/claim_check.py on every research/findings/*.md the commit ADDS, and git computes
"added" against the FIRST parent only. In a main-into-branch merge (tools/agent_sync_branch.sh does one for every Hermes
branch sync: `git merge --no-ff --no-commit main` + `git commit`) that list holds every finding main gained since the
branch forked, so ONE stale citation on main blocked every branch sync. The case that showed it: main's
research/findings/2026-10-27-emo-bounded-flip-PREREGISTRATION.md cited its scorer's `--out` placeholder, a file no read
was ever saved under. The citation came in with Amendment 1, a MODIFY, which GATE 2 does not check by design (it checks
findings as WRITTEN), so main itself never saw it; every branch merging main did, and the branch could not fix it.

THE RULE. During a merge (MERGE_HEAD present) a path is CARRIED, and not re-checked, when its staged content is
byte-identical to its content in a parent of the merge that main already contains (an ancestor of refs/heads/main or of
any refs/remotes/<remote>/main). That is main's content as main holds it, judged when it landed there; this merge did not
write it. Everything else is checked exactly as before:
  * a merge that writes NEW content into a finding (a conflict resolution, an evil merge, a new file) -- its content
    matches no parent, so it is checked;
  * a branch's own new findings landed INTO main (tools/land_claude_branch.sh, tools/agent_merge.sh): the branch parent
    is not on main yet, so they are checked as before. That landing is where the gates run for cloud-sandbox work
    (land_claude_branch.sh step 3); the literal rule "identical to ANY parent" would have exempted all of it;
  * a merge of two branches neither of which main contains: checked as before (against the first parent);
  * a non-merge commit: the hook does not call this at all.
NOT COVERED: a clean `git merge` that commits by itself. git runs pre-merge-commit BEFORE it writes MERGE_HEAD
(builtin/merge.c prepare_to_commit; measured 2026-10-28 with git 2.56: MERGE_HEAD absent, ORIG_HEAD = HEAD), so the
other parent is not visible to a hook and GATE 2 judges that merge against HEAD as before. Merge with `--no-commit` and
conclude with `git commit` (every repo merge tool already does) and the rule above applies.
FAILS CLOSED: if git cannot be read (MERGE_HEAD unreadable, a git error or timeout), nothing is carried, every path is
printed, and the reason goes to stderr. The hook also keeps its own list when this script exits non-zero.
Tests: tests/test_gate2_merge_aware.py (real merges through the hook's own GATE 2 block in scratch repos).
"""
from __future__ import annotations

import os
import subprocess
import sys

TRUNK_REFS = ("refs/heads/main", "refs/remotes/*/main")
_TIMEOUT = 60
_CHUNK = 400          # paths per git call: a long-diverged sync merge can add hundreds of findings


class GitReadError(Exception):
    pass


def _git(args, root):
    try:
        r = subprocess.run(["git", "--literal-pathspecs"] + args, cwd=root, capture_output=True, timeout=_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as e:
        raise GitReadError("`git %s` could not run: %s" % (" ".join(args[:2]), e))
    if r.returncode != 0:
        raise GitReadError("`git %s` exited %d: %s" % (" ".join(args[:2]), r.returncode,
                                                       r.stderr.decode("utf-8", "replace").strip()[:160]))
    return r.stdout


def merge_parents(root):
    """[HEAD, *MERGE_HEAD] while a merge is being concluded; [] otherwise. Raises GitReadError."""
    mh = _git(["rev-parse", "--path-format=absolute", "--git-path", "MERGE_HEAD"], root).decode().strip()
    if not mh or not os.path.lexists(mh):
        return []
    try:
        with open(mh, encoding="utf-8") as fh:
            heads = [ln.split()[0] for ln in fh if ln.strip()]
    except (OSError, UnicodeDecodeError) as e:
        raise GitReadError("MERGE_HEAD exists but cannot be read: %s" % e)
    if not heads:
        raise GitReadError("MERGE_HEAD exists but names no commit")
    head = _git(["rev-parse", "--verify", "HEAD^{commit}"], root).decode().strip()
    return [head] + heads


def on_trunk(root, commit):
    """True when some trunk ref (TRUNK_REFS) contains `commit`."""
    return bool(_git(["for-each-ref", "--contains", commit, "--format=%(refname)"] + list(TRUNK_REFS), root).strip())


def _blobs(root, args, paths, meta_fields):
    """{path: blob sha} from `git <args> -- <paths>` -z records "<meta>\\t<path>"; `meta_fields` picks the sha and
    keeps the record (ls-files -s: stage 0 only)."""
    res = {}
    for i in range(0, len(paths), _CHUNK):
        for rec in _git(args + ["--"] + paths[i:i + _CHUNK], root).split(b"\0"):
            if not rec:
                continue
            meta, path = rec.split(b"\t", 1)
            sha = meta_fields(meta.split())
            if sha:
                res[path.decode("utf-8", "surrogateescape")] = sha
    return res


def staged_blobs(root, paths):
    return _blobs(root, ["ls-files", "-s", "-z"], paths, lambda m: m[1].decode() if m[2] == b"0" else None)


def tree_blobs(root, commit, paths):
    return _blobs(root, ["ls-tree", "-z", "--full-tree", commit], paths,
                  lambda m: m[2].decode() if m[1] == b"blob" else None)


def split(paths, root=None):
    """(to_check, carried, note). `to_check` keeps the input order; `note` is "" or one line for stderr."""
    root = os.path.abspath(root or os.getcwd())
    paths = [p for p in paths if p]
    if not paths:
        return [], [], ""
    try:
        parents = merge_parents(root)
        if not parents:
            return paths, [], ""
        trunk = [p for p in parents if on_trunk(root, p)]
        if not trunk:
            return paths, [], ""
        staged = staged_blobs(root, paths)
        carried = set()
        for parent in trunk:
            theirs = tree_blobs(root, parent, paths)
            carried.update(p for p in paths if p in staged and theirs.get(p) == staged[p])
    except GitReadError as e:
        return paths, [], ("GATE 2: could not read the merge (%s); failing CLOSED, every added finding is checked"
                           % e)
    keep = [p for p in paths if p not in carried]
    if not carried:
        return keep, [], ""
    shown = ", ".join(sorted(carried)[:3]) + (" and %d more" % (len(carried) - 3) if len(carried) > 3 else "")
    return keep, sorted(carried), ("GATE 2 (merge): %d added finding(s) identical to a merge parent main already "
                                   "contains are main's content, not this merge's; not re-checked: %s"
                                   % (len(carried), shown))


def main():
    paths = sys.stdin.read().split("\n")
    keep, _carried, note = split(paths)
    if note:
        print(note, file=sys.stderr)
    if keep:
        print("\n".join(keep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
