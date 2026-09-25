"""tests for tools/gates/prereg_amendment_order.py (CLASS PRA).

Imports the REAL gate module the pre-commit registry calls. Three layers, each covering what the one before cannot:

  1. selftest() -- the registry's only trust signal -- and a MUTATION check that it actually fails when the wiring
     the 2026-09-25 review found broken is broken again (a selftest that survives these mutants is decoration).
  2. REAL COMMITS through a REAL pre-commit hook in a scratch repo: plain `git add`, an overwritten artifact with
     nothing added, `git commit -a`, `git commit -- <paths>`, a linked worktree, merges. The hook calls check()
     exactly as the registry does, with the registry's --diff-filter=A list as `paths`.
  3. The review's named commits, replayed from this repository's own history (skipped when absent, e.g. shallow CI).
"""
from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest

import tools.gates.prereg_amendment_order as pra_gate

PREREG = "research/findings/2026-01-01-x-PREREGISTRATION.md"
BASE = "# prereg\n\nthresholds: G1 >= 0.5\n\n## Amendment log\n\n(none at filing)\n"
BOLD_AMEND = "\n**AMENDMENT 1: 2026-01-02, after the seed-7 smoke, before round 2.** G1 is now >= 0.6.\n"


# ---------------------------------------------------------------------------------------------------------
# 1. selftest + its mutation check
# ---------------------------------------------------------------------------------------------------------
def test_registry_selftest_passes():
    assert pra_gate.selftest() == []


def test_registry_discovers_this_gate_with_the_expected_contract():
    from tools.gates import discover
    hits = [t for t in discover() if t[0] == pra_gate.NAME]
    assert len(hits) == 1
    name, mod, err = hits[0]
    assert err is None, err
    assert mod.CLASS_ID == "PRA" and mod.BLOCKING is True


MUTANTS = {
    "check() returns early on empty paths (the registry's --diff-filter=A list)": (
        "    root = os.path.abspath(root or _ROOT)\n    blobs = {}",
        "    if not paths:\n        return []\n    root = os.path.abspath(root or _ROOT)\n    blobs = {}"),
    "modified-prereg status filter no longer matches M": (
        'changes[0][p][0] not in ("M", "R")', 'changes[0][p][0] not in ("A",)'),
    "raw filter only counts ADDED artifacts": (
        'all(c[p][0] != "D" for c in changes)', 'all(c[p][0] == "A" for c in changes)'),
    "part 1 unwired from check()": (
        "        return _problems(prereg_changes, raw_written)", "        return []"),
    "bold / log-form amendment entries not detected": (
        "bm = _BOLD_RE.match(ln)", "bm = None"),
    "GIT_INDEX_FILE dropped": (
        '            env["GIT_INDEX_FILE"] = idx_abs', "            pass"),
    "merge handling removed (HEAD treated as the only parent)": (
        "changes = [_staged_changes(root, env, p) for p in parents]",
        "changes = [_staged_changes(root, env, p) for p in parents[:1]]"),
    "record-subsection exemption removed": (
        'if is_record(idx) or ent["i"] not in added:', 'if ent["i"] not in added:'),
    "provenance log / sidecars counted as run data": (
        "return bool(_RAW_RE.match(path)) and not _RAW_NOT_DATA_RE.search(path)", "return bool(_RAW_RE.match(path))"),
    "edits to an existing amendment's body (N3) ignored": (
        "if bodies and all(new.body(idx) != b for b in bodies):", "if False:"),
    "amendment-same-commit escape ignored": (
        "if not act or _escaped(new_text, parent_texts):", "if not act:"),
    "fails OPEN when git cannot read the index": (
        '        return ["CLASS PRA could not read', '        return []\n        return ["CLASS PRA could not read'),
}


@pytest.mark.parametrize("label", sorted(MUTANTS))
def test_selftest_kills_mutants(label):
    old, new = MUTANTS[label]
    src = open(pra_gate.__file__, encoding="utf-8").read()
    assert src.count(old) == 1, "mutation anchor for %r no longer matches the gate source exactly once" % label
    mod = types.ModuleType("pra_mutant")
    mod.__file__ = pra_gate.__file__
    exec(compile(src.replace(old, new), pra_gate.__file__, "exec"), mod.__dict__)
    assert mod.selftest(), "selftest() still PASSES with the mutant %r -- the registry would trust a broken gate" % label


# ---------------------------------------------------------------------------------------------------------
# 2. real commits through a real pre-commit hook
# ---------------------------------------------------------------------------------------------------------
_REPO = pra_gate._ROOT
_HOOK = """#!/bin/sh
exec "%s" - <<'PY'
import os, subprocess, sys
sys.path.insert(0, %r)
import tools.gates.prereg_amendment_order as m
added = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=A"],
                       capture_output=True, text=True).stdout.split()
problems = m.check(added, root=os.getcwd())
print("\\n".join(problems))
sys.exit(1 if problems else 0)
PY
"""


def _env():
    return pra_gate._stripped_env()


def _git(root, *args, check=True):
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=root, env=_env(), capture_output=True,
                       text=True, timeout=60)
    if check:
        assert r.returncode == 0, "git %s failed: %s %s" % (" ".join(args), r.stdout, r.stderr)
    return r


def _write(root, rel, text, mode="w"):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode, encoding="utf-8") as fh:
        fh.write(text)


@pytest.fixture()
def repo(tmp_path):
    root = str(tmp_path / "main")
    os.makedirs(root)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    hooks = str(tmp_path / "hooks")
    os.makedirs(hooks)
    with open(os.path.join(hooks, "pre-commit"), "w") as fh:
        fh.write(_HOOK % (sys.executable, _REPO))
    os.chmod(os.path.join(hooks, "pre-commit"), 0o755)
    _git(root, "config", "core.hooksPath", hooks)
    _write(root, PREREG, BASE)
    _write(root, "research/findings/raw/x/s7.json", '{"v": 1}')
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _blocked(r):
    return r.returncode != 0 and "CLASS PRA" in (r.stdout + r.stderr)


def test_plain_add_amendment_with_a_new_artifact_is_blocked(repo):
    _write(repo, PREREG, BOLD_AMEND, "a")
    _write(repo, "research/findings/raw/x/s42.json", "{}")
    _git(repo, "add", "-A")
    assert _blocked(_git(repo, "commit", "-m", "amend+data", check=False))


def test_overwritten_artifact_with_nothing_added_is_blocked(repo):
    """review issue 1: the registry's `paths` is EMPTY here (nothing has status A)."""
    _write(repo, PREREG, BOLD_AMEND, "a")
    _write(repo, "research/findings/raw/x/s7.json", '{"v": 2}')
    _git(repo, "add", "-u")
    assert _git(repo, "diff", "--cached", "--name-only", "--diff-filter=A").stdout.strip() == ""
    assert _blocked(_git(repo, "commit", "-m", "amend+overwrite", check=False))


def test_commit_dash_a_is_blocked(repo):
    """review issue 2: `commit -a` commits a temporary index named by GIT_INDEX_FILE."""
    _write(repo, PREREG, BOLD_AMEND, "a")
    _write(repo, "research/findings/raw/x/s7.json", '{"v": 2}')
    assert _blocked(_git(repo, "commit", "-a", "-m", "amend -a", check=False))


def test_commit_with_pathspec_is_blocked(repo):
    """review issue 2: `commit -- <paths>` commits a next-index-<pid>.lock temporary index."""
    _write(repo, PREREG, BOLD_AMEND, "a")
    _write(repo, "research/findings/raw/x/s7.json", '{"v": 2}')
    r = _git(repo, "commit", "-m", "amend pathspec", "--", PREREG, "research/findings/raw/x/s7.json", check=False)
    assert _blocked(r)


def test_commit_dash_a_in_a_linked_worktree_is_blocked(repo, tmp_path):
    wt = str(tmp_path / "wt")
    _git(repo, "worktree", "add", "-q", "-b", "lane", wt)
    _write(wt, PREREG, BOLD_AMEND, "a")
    _write(wt, "research/findings/raw/x/s7.json", '{"v": 2}')
    assert _blocked(_git(wt, "commit", "-a", "-m", "amend -a in a worktree", check=False))


@pytest.mark.parametrize("amend, raw_rel", [
    (BOLD_AMEND, None),                                                            # amendment alone
    ("\nsome prose outside the log.\n", "research/findings/raw/x/s42.json"),        # data, prose outside any amendment
    (BOLD_AMEND + "- amendment-same-commit: these files are the seed-7 smoke this amendment records\n",
     "research/findings/raw/x/s42.json"),                                          # declared escape
    (BOLD_AMEND, "research/findings/raw/_provenance/runs.jsonl"),                  # provenance log is not data
    (BOLD_AMEND, "research/findings/raw/x/s42.json.prov.json"),                    # sidecar is not data
])
def test_passing_cases_commit(repo, amend, raw_rel):
    if amend.startswith("\nsome prose"):
        text = open(os.path.join(repo, PREREG)).read().replace("thresholds: G1 >= 0.5\n",
                                                               "thresholds: G1 >= 0.5\n" + amend)
        _write(repo, PREREG, text)
    else:
        _write(repo, PREREG, amend, "a")
    if raw_rel:
        _write(repo, raw_rel, "{}\n", "a")
    _git(repo, "add", "-A")
    r = _git(repo, "commit", "-m", "should pass", check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_record_subsection_of_a_committed_amendment_with_its_data_passes(repo):
    """review issue 3 (414e1ba4f's shape): the amendment was committed first; its smoke record lands with the data."""
    _write(repo, PREREG, "\n## AMENDMENT 1 (2026-01-02, before round 2)\n\nG1 >= 0.6\n", "a")
    _git(repo, "commit", "-q", "-am", "amendment first")
    _write(repo, PREREG, "\n### AMENDMENT 1, smoke record (appended after the declared smoke ran)\n\nG1 read 0.7.\n", "a")
    _write(repo, "research/findings/raw/x/s42.json", "{}")
    _git(repo, "add", "-A")
    r = _git(repo, "commit", "-m", "record + data", check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_editing_a_committed_amendment_body_with_data_is_blocked(repo):
    """835fc252e's shape: a DRAFT item of an existing amendment completed in the same commit as its data."""
    _write(repo, PREREG, "\n## AMENDMENT 1 (2026-01-02)\n\nItem 3: budget -- DRAFT, TBD once N=8/32 land.\n", "a")
    _git(repo, "commit", "-q", "-am", "amendment with a draft item")
    text = open(os.path.join(repo, PREREG)).read().replace("DRAFT, TBD once N=8/32 land.", "COMPLETE: budget 1 h, N=32.")
    _write(repo, PREREG, text)
    _write(repo, "research/findings/raw/x/n32.json", "{}")
    _git(repo, "add", "-A")
    assert _blocked(_git(repo, "commit", "-m", "complete the draft with the data", check=False))


def _lane_with_ordered_history(repo):
    _git(repo, "checkout", "-q", "-b", "lane")
    _write(repo, PREREG, BOLD_AMEND, "a")
    _git(repo, "commit", "-q", "-am", "amendment first")
    _write(repo, "research/findings/raw/x/s42.json", "{}")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "then data")
    _git(repo, "checkout", "-q", "main")
    _write(repo, "unrelated.txt", "x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main moves")


def test_merge_of_correctly_ordered_history_passes(repo):
    """review issue 5: a merge brings the amendment AND the data in as changes relative to HEAD."""
    _lane_with_ordered_history(repo)
    r = _git(repo, "merge", "--no-ff", "-m", "merge lane", "lane", check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_evil_merge_writing_a_new_amendment_and_data_is_blocked(repo):
    _lane_with_ordered_history(repo)
    _git(repo, "merge", "--no-ff", "--no-commit", "lane")
    _write(repo, PREREG, "\n## AMENDMENT 2 (written in the merge itself)\n\nG1 >= 0.7\n", "a")
    _write(repo, "research/findings/raw/x/s43.json", "{}")
    _git(repo, "add", "-A")
    assert _blocked(_git(repo, "commit", "-m", "evil merge", check=False))


def test_unreadable_repository_fails_closed(tmp_path):
    broken = str(tmp_path / "broken")
    os.makedirs(broken)
    with open(os.path.join(broken, ".git"), "w") as fh:
        fh.write("gitdir: %s\n" % (tmp_path / "missing"))
    problems = pra_gate.check([], root=broken)
    assert len(problems) == 1 and "failing CLOSED" in problems[0]


# ---------------------------------------------------------------------------------------------------------
# 3. the review's named commits, from this repository's own history
# ---------------------------------------------------------------------------------------------------------
REAL = [
    ("835fc252e", True),    # slotbinder: AMENDMENT 2's DRAFT item completed with the sizing data (the motivating case)
    ("0ee39e293", True),    # **AMENDMENT 1: ...** bold log entry + smoke data
    ("356c9f040", True),    # **AMENDMENT 2: ...** bold log entry + probe data
    ("f2b9e69b7", True),    # **A2, ...** log entry + OVERWRITTEN (status M) artifacts, nothing added
    ("5fbc5cc112", True),   # ## ADDENDUM A4 + data
    ("414e1ba4f", False),   # ### AMENDMENT 6, smoke record -- a record of an already-committed amendment
    ("5b5ea1b74", False),   # new amendment, but the only raw path is _provenance/runs.jsonl
    ("72b744c4c", False),   # likewise: only _provenance/runs.jsonl
]


def _real(*args):
    return subprocess.run(["git", *args], cwd=_REPO, env=_env(), capture_output=True, timeout=60)


def _replay(sha):
    """The gate's pure decision on a real non-merge commit, against its first parent."""
    out = _real("diff-tree", "-r", "-z", "--no-commit-id", "--name-status", "-M", sha + "^", sha).stdout
    toks, i, ch = out.decode("utf-8", "surrogateescape").split("\0"), 0, {}
    while i < len(toks) and toks[i]:
        if toks[i][:1] in "RC":
            ch[toks[i + 2]] = (toks[i][:1], toks[i + 1])
            i += 3
        else:
            ch[toks[i + 1]] = (toks[i][:1], toks[i + 1])
            i += 2
    raw = [p for p, (st, _) in ch.items() if pra_gate._RAW_RE.match(p) and st != "D"]
    preregs = []
    for p, (st, old) in ch.items():
        if pra_gate._PREREG_RE.match(p) and st in "MR":
            new = _real("cat-file", "blob", "%s:%s" % (sha, p)).stdout.decode("utf-8", "replace")
            par = _real("cat-file", "blob", "%s^:%s" % (sha, old)).stdout.decode("utf-8", "replace")
            preregs.append((p, new, [par]))
    return pra_gate._problems(preregs, raw)


@pytest.mark.parametrize("sha, should_block", REAL)
def test_review_named_commits(sha, should_block):
    if _real("cat-file", "-e", sha + "^{commit}").returncode != 0 or _real("cat-file", "-e", sha + "^").returncode != 0:
        pytest.skip("commit %s not in this clone" % sha)
    assert bool(_replay(sha)) is should_block
