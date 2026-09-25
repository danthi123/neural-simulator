"""tests for tools/gates/prereg_before_run.py (CLASS PR): WHICH commit is judged (2026-09-25).

Found while fixing CLASS PRA (review r4 of research/gate-prereg-amendment-order): PR judged every commit against
HEAD, so (1) a prereg committed alone and its data folded in by `git commit --amend` landed as ONE commit adding both,
rc 0; and (2) its 2026-09-24 merge exemption (`_merged_in_unchanged`) needs MERGE_HEAD, which git has not written yet
when pre-merge-commit runs, so a clean `git merge` of a lane that committed its prereg and then its data was blocked.
Both were measured through real hooks before the fix. Three layers:

  1. selftest() -- the registry's only trust signal -- and a MUTATION check that it fails when each fix is undone.
  2. REAL commits through REAL hooks in a scratch repo that holds a copy of the gate (so the gate's _ROOT is that
     repo), with this repo's real tools/githooks/pre-merge-commit, so a clean `git merge` runs the gate the way it
     does here.
  3. The real-hook scenarios against MUTATED copies of the gate: each fix is what makes its real-hook test pass.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types

import pytest

import tools.gates.prereg_amendment_order as pra_gate
import tools.gates.prereg_before_run as pr_gate

_REPO = pr_gate._ROOT
_GATE_SRC = pr_gate.__file__
PREREG = "research/findings/2026-01-01-x-PREREGISTRATION.md"
RAW = "research/findings/raw/x/s42.json"


# ---------------------------------------------------------------------------------------------------------
# 1. selftest + its mutation check
# ---------------------------------------------------------------------------------------------------------
def test_registry_selftest_passes():
    assert pr_gate.selftest() == []


def test_registry_discovers_this_gate_with_the_expected_contract():
    from tools.gates import discover
    hits = [t for t in discover() if t[0] == pr_gate.NAME]
    assert len(hits) == 1
    _name, mod, err = hits[0]
    assert err is None, err
    assert mod.CLASS_ID == "PR" and mod.BLOCKING is True


MUTANTS = {
    "`commit --amend` judged against HEAD (the pre-fix behaviour)": (
        '    if kind == "amend":\n        parents =', '    if False:\n        parents ='),
    "early return on empty `paths` under --amend (added relative to HEAD, not the new parent)": (
        'if kind != "amend" and (paths is None or len(paths) == 0):', "if paths is None or len(paths) == 0:"),
    "clean auto-merge judged against HEAD alone (the pre-fix behaviour)": (
        '    if kind == "merge" and not _git_out(', '    if False and not _git_out('),
    "`git merge --continue`: the command consulted BEFORE MERGE_HEAD (fails OPEN)": (
        'if kind == "merge" and not _git_out(root, env, "rev-parse", "-q", "--verify", "MERGE_HEAD"):',
        'if kind == "merge":'),
    "`--amend` of a ROOT commit: the empty tree dropped": (
        "        return parents or [_git_out(root, env, \"hash-object\", \"-t\", \"tree\", os.devnull)]",
        "        return parents"),
    "an amended MERGE judged against its first parent only": (
        "    staged = sorted(set(diffs[0]).intersection(*diffs[1:]))",
        "    diffs = diffs[:1]\n    staged = sorted(set(diffs[0]))"),
    "merged-in-unchanged exemption removed": (
        "    added = [p for p in added if not _merged_in_unchanged(p, root, env)]\n", ""),
    "the first commit of a repo (no HEAD) diffed against a missing HEAD (fails OPEN)": (
        '    if not _git_out(root, env, "rev-parse", "-q", "--verify", "HEAD^{commit}"):\n', "    if False:\n"),
}


def _mutated_source(old, new):
    src = open(_GATE_SRC, encoding="utf-8").read()
    assert src.count(old) == 1, "mutation anchor no longer matches the gate source exactly once: %r" % old[:80]
    return src.replace(old, new)


@pytest.mark.parametrize("label", sorted(MUTANTS))
def test_selftest_kills_mutants(label):
    mod = types.ModuleType("pr_mutant")
    mod.__file__ = _GATE_SRC
    exec(compile(_mutated_source(*MUTANTS[label]), _GATE_SRC, "exec"), mod.__dict__)
    assert mod.selftest(), "selftest() still PASSES with the mutant %r -- the registry would trust a broken gate" % label


# ---------------------------------------------------------------------------------------------------------
# 2. real commits through real hooks
# ---------------------------------------------------------------------------------------------------------
_HOOK = """#!/bin/sh
exec "%s" - <<'PY'
import os, subprocess, sys
sys.path.insert(0, os.getcwd())
import tools.gates.prereg_before_run as m
import tools.gates.prereg_amendment_order as pra
assert os.path.realpath(m._ROOT) == os.path.realpath(os.getcwd()), m._ROOT
added = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=A"],
                       capture_output=True, text=True).stdout.split()
problems = m.check(added)
with open(%r, "a") as fh:
    fh.write("%%s %%s\\n" %% (pra._detect_invocation(os.getcwd()), "BLOCK" if problems else "pass"))
print("\\n".join(problems))
sys.exit(1 if problems else 0)
PY
"""


def _env():
    return pra_gate._stripped_env()


def _git(root, *args, check=True, env=None):
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=root, env=env or _env(),
                       capture_output=True, text=True, timeout=60)
    if check:
        assert r.returncode == 0, "git %s failed: %s %s" % (" ".join(args), r.stdout, r.stderr)
    return r


def _write(root, rel, text, mode="w"):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode, encoding="utf-8") as fh:
        fh.write(text)


def _make_repo(tmp_path, gate_source=None):
    """A scratch repo holding a copy of the gate under its own tools/gates/ (untracked), so the gate's _ROOT is the
    scratch repo; the fixture's pre-commit plus this repo's REAL pre-merge-commit (which execs pre-commit)."""
    root = os.path.realpath(str(tmp_path / "main"))
    os.makedirs(root)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    gates, hooks = os.path.join(root, "tools", "gates"), os.path.join(root, "tools", "githooks")
    os.makedirs(gates)
    os.makedirs(hooks)
    for init in ("tools/__init__.py", "tools/gates/__init__.py"):
        open(os.path.join(root, init), "w").close()
    with open(os.path.join(gates, "prereg_before_run.py"), "w", encoding="utf-8") as fh:
        fh.write(gate_source if gate_source is not None else open(_GATE_SRC, encoding="utf-8").read())
    shutil.copy(pra_gate.__file__, os.path.join(gates, "prereg_amendment_order.py"))
    log = str(tmp_path / "hook-runs.log")
    with open(os.path.join(hooks, "pre-commit"), "w") as fh:
        fh.write(_HOOK % (sys.executable, log))
    shutil.copy(os.path.join(_REPO, "tools", "githooks", "pre-merge-commit"), os.path.join(hooks, "pre-merge-commit"))
    for h in ("pre-commit", "pre-merge-commit"):
        os.chmod(os.path.join(hooks, h), 0o755)
    with open(os.path.join(root, ".git", "info", "exclude"), "a") as fh:
        fh.write("/tools/\n")
    _git(root, "config", "core.hooksPath", hooks)
    _write(root, "README", "x\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _hook_log(root):
    p = os.path.join(os.path.dirname(root), "hook-runs.log")
    return open(p).read().split("\n") if os.path.exists(p) else []


@pytest.fixture()
def repo(tmp_path):
    return _make_repo(tmp_path)


def _blocked(r):
    return r.returncode != 0 and "is ADDED in the same commit as" in (r.stdout + r.stderr)


def test_prereg_added_with_its_data_is_blocked(repo):
    _write(repo, PREREG, "# prereg\nG1 >= 0.5\n")
    _write(repo, RAW, "{}")
    _git(repo, "add", "-A")
    assert _blocked(_git(repo, "commit", "-m", "prereg + data", check=False))
    assert _hook_log(repo)[-2] == "commit BLOCK"


def test_prereg_alone_then_data_as_its_own_commit_passes(repo):
    _write(repo, PREREG, "# prereg\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "prereg alone")
    _write(repo, RAW, "{}")
    _git(repo, "add", "-A")
    r = _git(repo, "commit", "-m", "then data", check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _hook_log(repo)[-2] == "commit pass"


def _amend_folds_data_into_the_prereg_commit(repo, *flags):
    _write(repo, PREREG, "# prereg\nG1 >= 0.5\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "prereg alone")
    _write(repo, RAW, "{}")
    _git(repo, "add", "-A")
    return _git(repo, "commit", *(flags or ("--amend", "--no-edit")), check=False)


@pytest.mark.parametrize("flag", ["--amend", "--amen"])
def test_amend_folding_data_into_the_prereg_commit_is_blocked(repo, flag):
    """measured before the fix: rc 0, and HEAD then ADDED both the prereg and the data."""
    assert _blocked(_amend_folds_data_into_the_prereg_commit(repo, flag, "--no-edit"))
    assert _hook_log(repo)[-2] == "amend BLOCK"


def test_amend_overwriting_an_existing_artifact_is_blocked(repo):
    """nothing is ADDED relative to HEAD, so the registry's `paths` is empty -- the early return must not fire."""
    _write(repo, RAW, '{"v": 1}')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "an older artifact")
    _write(repo, PREREG, "# prereg\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "prereg alone")
    _write(repo, RAW, '{"v": 2}')
    _git(repo, "add", "-A")
    assert _git(repo, "diff", "--cached", "--name-only", "--diff-filter=A").stdout.strip() == ""
    assert _blocked(_git(repo, "commit", "--amend", "--no-edit", check=False))


def test_amend_of_a_data_commit_after_its_prereg_passes(repo):
    _write(repo, PREREG, "# prereg\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "prereg alone")
    _write(repo, RAW, "{}")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "then data")
    _write(repo, "research/findings/raw/x/s43.json", "{}")
    _git(repo, "add", "-A")
    r = _git(repo, "commit", "--amend", "--no-edit", check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _hook_log(repo)[-2] == "amend pass"


def test_amend_rewording_a_prereg_only_commit_passes(repo):
    _write(repo, PREREG, "# prereg\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "prereg alone")
    r = _git(repo, "commit", "--amend", "-m", "prereg alone, reworded", check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_abbreviated_message_option_value_is_not_an_amend(repo):
    """`--mess --amend` makes a NEW commit whose message is `--amend`: data after a committed prereg, ordered."""
    r = _amend_folds_data_into_the_prereg_commit(repo, "--mess", "--amend")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _hook_log(repo)[-2] == "commit pass"


def _lane_with_ordered_history(repo):
    _git(repo, "checkout", "-q", "-b", "lane")
    _write(repo, PREREG, "# prereg\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "prereg first")
    _write(repo, RAW, "{}")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "then data")
    _git(repo, "checkout", "-q", "main")
    _write(repo, "unrelated.txt", "x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main moves")


def _clean_merge(repo):
    _lane_with_ordered_history(repo)
    r = _git(repo, "merge", "--no-ff", "-m", "merge lane", "lane", check=False)
    ok = r.returncode == 0 and len(_git(repo, "rev-list", "--parents", "-n1", "HEAD").stdout.split()) == 3
    return ok, r


def test_clean_auto_merge_of_an_ordered_lane_passes(repo):
    """measured before the fix: blocked, 'Not committing merge' -- the MERGE_HEAD exemption cannot fire yet."""
    ok, r = _clean_merge(repo)
    assert ok, r.stdout + r.stderr
    assert _hook_log(repo)[-2] == "merge pass", _hook_log(repo)


def test_pull_of_an_ordered_lane_passes(repo):
    _lane_with_ordered_history(repo)
    r = _git(repo, "pull", "--no-rebase", "--no-ff", "--no-edit", ".", "lane", check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _hook_log(repo)[-2] == "merge pass"


def test_amend_of_a_merge_commit_is_judged_against_every_parent(repo):
    ok, r = _clean_merge(repo)
    assert ok, r.stdout + r.stderr
    r = _git(repo, "commit", "--amend", "--no-edit", check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _hook_log(repo)[-2] == "amend pass"


def _conflicted_merge(repo):
    _lane_with_ordered_history(repo)
    _git(repo, "checkout", "-q", "lane")
    _write(repo, "unrelated.txt", "lane side\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "lane writes unrelated.txt too")
    _git(repo, "checkout", "-q", "main")
    r = _git(repo, "merge", "--no-ff", "lane", check=False)
    assert r.returncode != 0 and "CONFLICT" in r.stdout, r.stdout + r.stderr
    _write(repo, "unrelated.txt", "resolved\n")


def _merge_continue(repo):
    return _git(repo, "merge", "--continue", check=False, env=dict(_env(), GIT_EDITOR="true"))


def test_merge_continue_of_a_prereg_merged_in_unchanged_passes(repo):
    _conflicted_merge(repo)
    _git(repo, "add", "-A")
    r = _merge_continue(repo)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _hook_log(repo)[-2] == "merge pass"


def _evil_merge_continue(repo):
    _conflicted_merge(repo)
    _write(repo, "research/findings/2026-01-03-written-in-the-merge-PREREG.md", "# new prereg\n")
    _write(repo, "research/findings/raw/merge/s7.json", "{}")
    _git(repo, "add", "-A")
    return _merge_continue(repo)


def test_merge_continue_adding_a_prereg_and_its_data_is_blocked(repo):
    assert _blocked(_evil_merge_continue(repo))
    assert _hook_log(repo)[-2] == "merge BLOCK"


# ---------------------------------------------------------------------------------------------------------
# 3. the real-hook scenarios against MUTATED gates: each fix is what makes its real-hook test pass
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("label, scenario", [
    ("`commit --amend` judged against HEAD (the pre-fix behaviour)", "amend"),
    ("clean auto-merge judged against HEAD alone (the pre-fix behaviour)", "merge"),
    ("`git merge --continue`: the command consulted BEFORE MERGE_HEAD (fails OPEN)", "merge-continue"),
])
def test_real_hook_scenarios_fail_under_mutant(tmp_path, label, scenario):
    repo = _make_repo(tmp_path, _mutated_source(*MUTANTS[label]))
    if scenario == "amend":
        r = _amend_folds_data_into_the_prereg_commit(repo)
        assert r.returncode == 0, "the amend still blocks with the mutant %r -- its test proves nothing" % label
    elif scenario == "merge":
        ok, _r = _clean_merge(repo)
        assert not ok, "the clean merge still passes with the mutant %r -- its test proves nothing" % label
    else:
        r = _evil_merge_continue(repo)
        assert r.returncode == 0, "the evil `merge --continue` still blocks with the mutant %r" % label
