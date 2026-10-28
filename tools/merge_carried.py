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

THE RULE. During a merge (MERGE_HEAD present) a finding is CARRIED, and not re-checked, when BOTH hold for one parent of
the merge that main already contains (an ancestor of refs/heads/main or of any refs/remotes/<remote>/main):
  1. the finding's staged blob is byte-identical to its blob in that parent, AND
  2. no artifact the finding cites differs between the index and that parent: its citations are read with claim_check's
     own PATH_RE (the paths claim_check loads), globs matched with fnmatch, against
     `git diff --cached --name-only --no-renames <parent>` (an artifact added, deleted or rewritten all count).
Then claim_check would judge the finding on exactly what it judged on main: main's text over main's artifacts, which
this merge did not write. Condition 2 is not optional (skeptic review 2026-10-28): without it a merge whose branch had
`git rm`ed, or re-run a runner over, an artifact main's new finding cites (or a merge that deletes or rewrites it while
it is concluded) landed that finding with its support gone, and nothing checked it.
Checked as before, because the rule does not hold:
  * a merge that writes NEW content into a finding (a conflict resolution, an evil merge, a new file) -- its content
    matches no trunk parent;
  * a finding main carries whose cited artifact this merge result changes (condition 2); stderr names the artifact;
  * a branch's own new findings landed INTO main (tools/land_claude_branch.sh, tools/agent_merge.sh): the branch parent
    is not on main yet, so they are checked as before. That landing is where the gates run for cloud-sandbox work
    (land_claude_branch.sh step 3); the literal rule "identical to ANY parent" would have exempted all of it;
  * a merge of two branches neither of which main contains: checked as before (against the first parent);
  * a non-merge commit: the hook does not call this at all.
A citation that resolves outside the repository is not compared: no merge of this repository can change it.
NOT COVERED: a clean `git merge` that commits by itself. git runs pre-merge-commit BEFORE it writes MERGE_HEAD
(builtin/merge.c prepare_to_commit; measured 2026-10-28 with git 2.56: MERGE_HEAD absent, ORIG_HEAD = HEAD), so the
other parent is not visible to a hook and GATE 2 judges that merge against HEAD as before. Merge with `--no-commit` and
conclude with `git commit` (every repo merge tool already does) and the rule above applies.
FAILS CLOSED: if git cannot be read (MERGE_HEAD unreadable, a git error or timeout) or claim_check's PATH_RE cannot be
loaded, nothing is carried, every path is printed, and the reason goes to stderr. The hook also keeps its own list when
this script exits non-zero.
Tests: tests/test_gate2_merge_aware.py (real merges through the hook's own GATE 2 block in scratch repos).
"""
from __future__ import annotations

import fnmatch
import importlib.util
import os
import re
import subprocess
import sys

TRUNK_REFS = ("refs/heads/main", "refs/remotes/*/main")
_TIMEOUT = 60
_CHUNK = 400          # paths per git call: a long-diverged sync merge can add hundreds of findings
_GLOB_CHARS = "*?["
_GLOB_RX = {}       # citation glob -> compiled fnmatch regex


class GitReadError(Exception):
    pass


def _git(args, root, stdin=None):
    try:
        r = subprocess.run(["git", "--literal-pathspecs"] + args, cwd=root, input=stdin, capture_output=True,
                           timeout=_TIMEOUT)
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


def blob_texts(root, shas):
    """{sha: text} for blob shas, through one `git cat-file --batch`. Raises GitReadError on anything but a blob."""
    uniq = sorted(set(shas))
    if not uniq:
        return {}
    out = _git(["cat-file", "--batch"], root, stdin=("\n".join(uniq) + "\n").encode())
    res, i = {}, 0
    for sha in uniq:
        nl = out.find(b"\n", i)
        head = out[i:nl].split() if nl >= 0 else []
        if len(head) != 3 or head[0].decode() != sha or head[1] != b"blob":
            raise GitReadError("`git cat-file --batch` did not return blob %s: %r" % (sha, out[i:i + 80]))
        size = int(head[2])
        res[sha] = out[nl + 1:nl + 1 + size].decode("utf-8", "replace")
        i = nl + 1 + size + 1
    return res


def changed_paths(root, commit):
    """Every path whose index entry differs from `commit` (added, deleted, modified, type-changed; renames split)."""
    out = _git(["diff", "--cached", "--name-only", "--no-renames", "--no-relative", "-z", commit], root)
    return {p.decode("utf-8", "surrogateescape") for p in out.split(b"\0") if p}


def claim_path_re(root):
    """claim_check's PATH_RE: from the claim_check GATE 2 runs (<root>/tools/claim_check.py), else the one beside this
    file. Raises GitReadError (fail closed) when neither loads."""
    errors = []
    for cand in (os.path.join(root, "tools", "claim_check.py"),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "claim_check.py")):
        if not os.path.isfile(cand):
            continue
        try:
            spec = importlib.util.spec_from_file_location("_gate2_claim_check", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if isinstance(mod.PATH_RE, re.Pattern):
                return mod.PATH_RE
            errors.append("%s: PATH_RE is not a compiled pattern" % cand)
        except Exception as e:                            # any failure to load means: carry nothing
            errors.append("%s: %s" % (cand, type(e).__name__))
    raise GitReadError("claim_check PATH_RE could not be loaded (%s)" % ("; ".join(errors) or "no claim_check.py"))


def citations(text, path_re, root):
    """The artifact paths a finding cites, as claim_check reads them (PATH_RE, resolved against the repo root), made
    repo-relative. A citation that resolves outside the repo is dropped: no merge of this repo can change it."""
    out = set()
    for c in path_re.findall(text):
        rel = os.path.relpath(os.path.normpath(os.path.join(root, c)), root).replace(os.sep, "/")
        if rel != ".." and not rel.startswith("../"):
            out.add(rel)
    return out


def first_changed_citation(cites, changed):
    """The first citation (sorted) that names a changed path, a glob matched with fnmatch (its `*` also crosses `/`, so
    it matches at least what claim_check's glob loads); None when none does."""
    for c in sorted(cites):
        if any(ch in c for ch in _GLOB_CHARS):
            rx = _GLOB_RX.get(c) or _GLOB_RX.setdefault(c, re.compile(fnmatch.translate(c)))
            hit = min((p for p in changed if rx.match(p)), default=None)
            if hit is not None:
                return "%s (matches %s)" % (c, hit)
        elif c in changed:
            return c
    return None


def split(paths, root=None):
    """(to_check, carried, note). `to_check` keeps the input order; `note` is "" or lines for stderr."""
    root = os.path.abspath(root or os.getcwd())
    paths = [p for p in paths if p]
    if not paths:
        return [], [], ""
    support_changed = {}          # path -> the cited artifact this merge changed (vs. a trunk parent with its blob)
    try:
        parents = merge_parents(root)
        if not parents:
            return paths, [], ""
        trunk = [p for p in parents if on_trunk(root, p)]
        if not trunk:
            return paths, [], ""
        staged = staged_blobs(root, paths)
        carried = set()
        path_re = None
        for parent in trunk:
            theirs = tree_blobs(root, parent, paths)
            same = [p for p in paths if p in staged and theirs.get(p) == staged[p] and p not in carried]
            if not same:
                continue
            if path_re is None:
                path_re = claim_path_re(root)
            texts = blob_texts(root, [staged[p] for p in same])
            changed = changed_paths(root, parent)
            for p in same:
                hit = first_changed_citation(citations(texts[staged[p]], path_re, root), changed)
                if hit is None:
                    carried.add(p)
                else:
                    support_changed.setdefault(p, hit)
    except GitReadError as e:
        return paths, [], ("GATE 2: could not read the merge (%s); failing CLOSED, every added finding is checked"
                           % e)
    keep = [p for p in paths if p not in carried]
    notes = []
    if carried:
        shown = ", ".join(sorted(carried)[:3]) + (" and %d more" % (len(carried) - 3) if len(carried) > 3 else "")
        notes.append("GATE 2 (merge): %d added finding(s) identical to a merge parent main already contains, every "
                     "artifact they cite unchanged from that parent, are main's content, not this merge's; not "
                     "re-checked: %s" % (len(carried), shown))
    for p in sorted(support_changed):
        if p not in carried:
            notes.append("GATE 2 (merge): %s is main's text, but this merge changes an artifact it cites (%s); "
                         "checked" % (p, support_changed[p]))
    return keep, sorted(carried), "\n".join(notes)


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
