"""CLASS PR — a pre-registration committed TOGETHER WITH the run artifacts it governs.

EVIDENCE (2026-09-23). Three of five build lanes in one day came back from adversarial review with the same defect:
the PREREG file, the runner and the decisive seed-42 artifact all landed in ONE commit (curiosity 7ba619532,
perception 7a3bcccaa, and the D6 v3 prereg's earlier rounds). The artifact's provenance showed it was produced from
UNCOMMITTED prereg text, so git could not show the thresholds were written before the result was seen -- and in the
perception case the prereg's bands were demonstrably written after s42 was observed. docs/BUILD_LANE_CHECKLIST.md
already said "Pre-register in its own commit BEFORE any run it governs"; the rule was read and violated anyway.

THE GATE. At commit time: if the staged set ADDS a pre-registration document (a research/findings/ or docs/plans/
markdown whose filename contains PREREG, case-insensitive) AND stages any research/findings/raw/** data artifact
(anything except *.prov.json sidecars), BLOCK. Commit the prereg first, then run, then commit the artifacts.

ESCAPE (declared, visible in the document): a line `prereg-same-commit: <reason, >= 15 chars>` in the prereg -- e.g.
the artifacts are integrity smokes that no gate in the prereg reads, or a seed explicitly declared "observed before
registration, carries no pre-registered weight".

WHICH COMMIT IS JUDGED (2026-09-25, found while fixing CLASS PRA). "Added" means added relative to the parents
the NEW commit will have, which is not always HEAD. The git command running the hook is read the way CLASS PRA reads
it (`prereg_amendment_order._detect_invocation`: the nearest ancestor `git` process working in this checkout, via
/proc):
  * `git commit --amend` REPLACES HEAD, so it is judged against HEAD's parents (every one; the empty tree for a root
    commit). Judged against HEAD, a prereg committed alone and its data folded in by `--amend` landed as ONE commit
    adding both, rc 0 (measured in a scratch repo). `paths` (added relative to HEAD) can then be empty, so the early
    return is skipped.
  * `git merge` / `git pull` with no MERGE_HEAD is a CLEAN AUTO-MERGE and is not checked. git runs pre-merge-commit
    BEFORE it writes MERGE_HEAD, so the 2026-09-24 exemption below never fired there: a clean `git merge --no-ff` of
    a lane that committed its prereg, then its data, was blocked ("Not committing merge", measured). git refuses to
    start a merge over staged changes, so no person wrote that tree, and each side's commits were judged when made.
  * MERGE_HEAD present (a conflicted merge finished by `git commit` or `git merge --continue`; MERGE_HEAD is read
    BEFORE the command, because `--continue` runs the hook as `merge`): judged against HEAD, with a prereg whose
    staged blob equals MERGE_HEAD's exempt (`_merged_in_unchanged`, 2026-09-24).
  * anything else, and whenever the command cannot be read: HEAD, as before.

WHAT THIS GATE CANNOT CATCH.
  * A prereg committed first, then EDITED after the run -- a MODIFIED prereg is CLASS PRA's scope
    (`gates/prereg_amendment_order`), which catches a new amendment or an edited amendment body landing with data.
  * A run executed before the prereg commit but whose artifacts are committed later, separately: git order looks
    right while the thresholds may still have been fitted after seeing the data. Provenance timestamps vs the prereg
    commit time would catch it; not implemented here.
  * A pre-registration whose filename does not contain PREREG.
  * No /proc (not Linux), or the gate run by hand: the command cannot be read, so `--amend` goes unseen and a clean
    auto-merge that brings in a prereg with its data blocks (fail-closed; finish it with `git commit`).
"""
from __future__ import annotations

import os
import re
import subprocess

NAME = "prereg_before_run"
CLASS_ID = "PR"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PREREG = re.compile(r"^(research/findings|docs/plans)/[^/]*prereg[^/]*\.md$", re.I)
_RAW = re.compile(r"^research/findings/raw/.+")
_ESCAPE = re.compile(r"^\s*[-*]?\s*prereg-same-commit:\s*(.{15,})$", re.I | re.M)


def _problems(added, staged, read):
    """added/staged: repo-relative paths; read(path) -> text. Pure, for the selftest."""
    preregs = [p for p in added if _PREREG.match(p)]
    raws = [p for p in staged if _RAW.match(p) and not p.endswith(".prov.json")]
    if not preregs or not raws:
        return []
    out = []
    for p in preregs:
        try:
            text = read(p)
        except OSError:
            text = ""
        if _ESCAPE.search(text):
            continue
        out.append("%s is ADDED in the same commit as %d run artifact(s) (e.g. %s). Commit the pre-registration "
                   "FIRST, then run, then commit the artifacts -- or declare `prereg-same-commit: <reason>` in it."
                   % (p, len(raws), raws[0]))
    return out


def _git_out(root, env, *a):
    r = subprocess.run(["git", *a], cwd=root, env=env, capture_output=True, text=True, timeout=15)
    return r.stdout.strip() if r.returncode == 0 else None


def _merged_in_unchanged(path, root=None, env=None):
    """True during a merge when `path`'s staged blob equals the incoming parent's (MERGE_HEAD) blob: the prereg is not
    new here, it arrives with its own history from the other branch (2026-09-24: merging main into a lane branch
    brought main's preregs + artifacts in as 'added' together and false-blocked the merge)."""
    root = root or _ROOT
    if not _git_out(root, env, "rev-parse", "-q", "--verify", "MERGE_HEAD"):
        return False
    theirs = _git_out(root, env, "rev-parse", "-q", "--verify", "MERGE_HEAD:" + path)
    staged = _git_out(root, env, "rev-parse", "-q", "--verify", ":" + path)
    return bool(theirs) and theirs == staged


def _bases(root, env, kind):
    """What the new commit's changes are measured against (see the docstring): ["HEAD"], HEAD's parents under
    `--amend` (the empty tree for a root commit), or None for a clean auto-merge (not checked)."""
    if kind == "merge" and not _git_out(root, env, "rev-parse", "-q", "--verify", "MERGE_HEAD"):
        return None
    if kind == "amend":
        parents = (_git_out(root, env, "rev-parse", "HEAD^@") or "").split()
        return parents or [_git_out(root, env, "hash-object", "-t", "tree", os.devnull)]
    if not _git_out(root, env, "rev-parse", "-q", "--verify", "HEAD^{commit}"):
        return [_git_out(root, env, "hash-object", "-t", "tree", os.devnull)]        # the first commit of a repo
    return ["HEAD"]


def check(paths, root=None, invocation=None):
    """`invocation` ('commit' | 'amend' | 'merge') overrides the /proc detection, for the selftest's scratch repo."""
    from tools.gates import prereg_amendment_order as _pra
    root = os.path.abspath(root or _ROOT)
    try:
        kind = invocation if invocation is not None else _pra._detect_invocation(root)     # /proc reads only
        if kind != "amend" and (paths is None or len(paths) == 0):
            return []                   # commit-scoped: nothing to say about the whole repo
        env = _pra._git_env(root)
        bases = _bases(root, env, kind)
        if not bases:
            return []
        diffs = []
        for base in bases:
            r = subprocess.run(["git", "diff", "--cached", "--name-status", base, "--"], cwd=root,
                               env=env, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                return []
            d = {}
            for ln in r.stdout.splitlines():
                parts = ln.split("\t")
                if len(parts) >= 2:
                    d[parts[-1]] = parts[0]
            diffs.append(d)
    except Exception:
        return []
    staged = sorted(set(diffs[0]).intersection(*diffs[1:]))         # an amended merge: new relative to EVERY parent
    added = [p for p in staged if all(d[p].startswith("A") for d in diffs)]
    added = [p for p in added if not _merged_in_unchanged(p, root, env)]
    return _problems(added, staged, lambda p: open(os.path.join(root, p), errors="ignore").read())


def selftest():
    """FAILING DIRECTION FIRST: a prereg added with a raw artifact MUST be caught."""
    bad = []
    txt = {"research/findings/2026-01-01-x-PREREG.md": "# prereg\nthresholds...\n",
           "docs/plans/2026-01-01-y-prereg.md": "# p\nprereg-same-commit: artifacts are integrity smokes only, no gate reads them\n"}
    rd = lambda p: txt[p]
    if not _problems(["research/findings/2026-01-01-x-PREREG.md"],
                     ["research/findings/2026-01-01-x-PREREG.md", "research/findings/raw/x/s42.json"], rd):
        bad.append("did NOT catch a prereg added together with a raw run artifact")
    if _problems(["research/findings/2026-01-01-x-PREREG.md"],
                 ["research/findings/2026-01-01-x-PREREG.md", "research/findings/raw/x/s42.json.prov.json"], rd):
        bad.append("FALSE POSITIVE: a provenance sidecar alone is not a run artifact")
    if _problems(["research/findings/2026-01-01-x-PREREG.md"], ["research/findings/2026-01-01-x-PREREG.md"], rd):
        bad.append("FALSE POSITIVE: a prereg committed on its own")
    if _problems([], ["research/findings/2026-01-01-x-PREREG.md", "research/findings/raw/x/s42.json"], rd):
        bad.append("FALSE POSITIVE: a MODIFIED (not added) prereg with artifacts (amendments are reviewed, not gated)")
    if _problems(["docs/plans/2026-01-01-y-prereg.md"],
                 ["docs/plans/2026-01-01-y-prereg.md", "research/findings/raw/y/s42.json"], rd):
        bad.append("FALSE POSITIVE: the declared prereg-same-commit escape was not honoured")
    try:
        _selftest_repo(bad)
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        bad.append("selftest scratch repo could not be built: %s" % e)
    return bad


def _selftest_repo(bad):
    """check()-level, in scratch repos: which commit is judged. Every check() names its `invocation`, so the answer
    does not depend on which real git command is running the hook around the selftest."""
    import tempfile
    from tools.gates import prereg_amendment_order as _pra
    env0 = _pra._stripped_env()
    with tempfile.TemporaryDirectory() as td:
        def g(*args, cwd=td):
            r = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"] + list(args),
                               cwd=cwd, env=env0, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                raise RuntimeError("selftest git %s: %s" % (args[:2], r.stderr.strip()[:120]))
            return r.stdout

        def write(rel, text, root=td):
            full = os.path.join(root, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(text)

        def init(root):
            os.makedirs(root, exist_ok=True)
            g("init", "-q", "-b", "main", cwd=root)
            g("config", "user.email", "gate-selftest@example.invalid", cwd=root)
            g("config", "user.name", "gate selftest", cwd=root)

        pre, raw = "research/findings/2026-01-01-st-PREREG.md", "research/findings/raw/st/s7.json"
        init(td)
        write("README", "x\n")
        g("add", "-A")
        g("commit", "-q", "-m", "base")

        # (1) `commit --amend` REPLACES HEAD: data folded into the commit that added the prereg alone is ONE commit
        write(pre, "# prereg\nG1 >= 0.5\n")
        g("add", "-A")
        g("commit", "-q", "-m", "prereg alone")
        write(raw, "{}")
        g("add", "-A")
        if check([raw], root=td, invocation="commit"):
            bad.append("selftest premise broken: data after a committed prereg should pass as a NEW commit")
        if not check([raw], root=td, invocation="amend"):
            bad.append("did NOT catch `commit --amend` folding run data into the commit that ADDS the prereg")
        if not check([], root=td, invocation="amend"):
            bad.append("`commit --amend` returned early on empty `paths` (added relative to HEAD, not the new parent)")
        g("commit", "-q", "-m", "the data, as its own commit")
        write("research/findings/raw/st/s8.json", "{}")
        g("add", "-A")
        if check(["research/findings/raw/st/s8.json"], root=td, invocation="amend"):
            bad.append("FALSE POSITIVE: `--amend` of a data commit whose prereg was committed before it")
        g("commit", "-q", "-m", "more data")

        # (2) merges: a lane that committed its prereg, then its data
        pre2, raw2 = "research/findings/2026-01-02-lane-PREREG.md", "research/findings/raw/lane/s7.json"
        g("checkout", "-q", "-b", "lane")
        write(pre2, "# lane prereg\n")
        g("add", "-A")
        g("commit", "-q", "-m", "lane prereg first")
        write(raw2, "{}")
        g("add", "-A")
        g("commit", "-q", "-m", "then its data")
        g("checkout", "-q", "main")
        write("unrelated.txt", "x\n")
        g("add", "-A")
        g("commit", "-q", "-m", "main moves on")
        g("merge", "-q", "--no-ff", "--no-commit", "lane")
        both = [pre2, raw2]
        mh = os.path.join(td, ".git", "MERGE_HEAD")
        os.rename(mh, mh + ".st")                     # pre-merge-commit of a CLEAN auto-merge: no MERGE_HEAD yet
        if not check(both, root=td, invocation="commit"):
            bad.append("selftest premise broken: the merged tree judged against HEAD alone should block")
        if check(both, root=td, invocation="merge"):
            bad.append("FALSE POSITIVE: a clean auto-merge (pre-merge-commit runs before MERGE_HEAD exists) was "
                       "judged against HEAD alone")
        os.rename(mh + ".st", mh)
        if check(both, root=td, invocation="merge"):
            bad.append("FALSE POSITIVE: `git merge --continue` of a prereg merged in unchanged")
        # a prereg WRITTEN in the merge, with data, finished by `git merge --continue` (command `merge`, MERGE_HEAD
        # present): consulting the command before MERGE_HEAD would read it as a clean auto-merge and fail OPEN
        write("research/findings/2026-01-03-merge-PREREG.md", "# written in the merge\n")
        write("research/findings/raw/merge/s7.json", "{}")
        g("add", "-A")
        if not check(["research/findings/2026-01-03-merge-PREREG.md", "research/findings/raw/merge/s7.json"],
                     root=td, invocation="merge"):
            bad.append("check() FAILED OPEN on `git merge --continue` of a merge that ADDS a prereg and its data")
        g("merge", "--abort")
        # `--amend` of a MERGE commit: new relative to EVERY parent, as for the merge itself
        g("merge", "-q", "--no-ff", "--no-edit", "lane")
        if check([], root=td, invocation="amend"):
            bad.append("FALSE POSITIVE: `--amend` of a merge of an ordered lane was judged against its first parent")

        # (3) `--amend` of a ROOT commit is judged against the empty tree
        r2 = os.path.join(td, "r2")
        init(r2)
        write(pre, "# prereg\n", root=r2)
        g("add", "-A", cwd=r2)
        g("commit", "-q", "-m", "root: prereg alone", cwd=r2)
        write(raw, "{}", root=r2)
        g("add", "-A", cwd=r2)
        if not check([raw], root=r2, invocation="amend"):
            bad.append("did NOT catch `--amend` of a ROOT commit folding data into the commit that adds the prereg")
        # (4) the FIRST commit of a repo (no HEAD yet) is judged against the empty tree
        r3 = os.path.join(td, "r3")
        init(r3)
        write(pre, "# prereg\n", root=r3)
        write(raw, "{}", root=r3)
        g("add", "-A", cwd=r3)
        if not check([pre, raw], root=r3, invocation="commit"):
            bad.append("did NOT catch a prereg and its data in the FIRST commit of a repo (no HEAD to diff against)")
