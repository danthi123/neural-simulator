"""tests for tools/gates/prereg_amendment_order.py (CLASS PRA).

Imports the REAL gate module the pre-commit registry calls. Four layers, each covering what the one before cannot:

  1. selftest() -- the registry's only trust signal -- and a MUTATION check that it actually fails when the wiring
     the 2026-09-25 reviews found broken is broken again (a selftest that survives these mutants is decoration).
  2. REAL COMMITS through REAL hooks in a scratch repo: the fixture installs a pre-commit that calls check() exactly
     as the registry does, AND this repo's own tools/githooks/pre-merge-commit (which execs pre-commit), so a clean
     `git merge` really runs the gate the way it does here -- before git has written MERGE_HEAD. (The first fix
     round's fixture installed pre-commit only, so no hook ran during a merge and its merge test was vacuous.)
     Every hook run is logged, with the git command the gate detected, to a file inside the test's tmp dir.
  3. The same hook scenarios against MUTATED copies of the gate, so the /proc detection, the clean-auto-merge rule
     and the amend rule are each shown to be what makes its real-hook test pass.
  4. The review's named commits, replayed from this repository's own history through the gate's own _evaluate
     (skipped when absent, e.g. shallow CI).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types

import pytest

import tools.gates.prereg_amendment_order as pra_gate

PREREG = "research/findings/2026-01-01-x-PREREGISTRATION.md"
BASE = "# prereg\n\nthresholds: G1 >= 0.5\n\n## Amendment log\n\n(none at filing)\n"
BOLD_AMEND = "\n**AMENDMENT 1: 2026-01-02, after the seed-7 smoke, before round 2.** G1 is now >= 0.6.\n"
_REPO = pra_gate._ROOT
_GATE_SRC = pra_gate.__file__


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
        "    root = os.path.abspath(root or _ROOT)\n",
        "    if not paths:\n        return []\n    root = os.path.abspath(root or _ROOT)\n"),
    "modified-prereg status filter no longer matches M": (
        'pre[0][p][0] in ("M", "R")', 'pre[0][p][0] in ("A",)'),
    "raw filter only counts ADDED artifacts": (
        'all(c[p][0] != "D" for c in raw)', 'all(c[p][0] == "A" for c in raw)'),
    "part 1 unwired from check()": (
        "    return _problems(prereg_changes, raw_written)", "    return []"),
    "bold / log-form amendment entries not detected": (
        "bm = _BOLD_RE.match(ln)", "bm = None"),
    "GIT_INDEX_FILE dropped": (
        '            env["GIT_INDEX_FILE"] = idx_abs', "            pass"),
    "merge handling removed (HEAD treated as the only parent)": (
        "        return _evaluate(root, env, parents)", "        return _evaluate(root, env, parents[:1])"),
    "record-subsection exemption removed": (
        'if is_record(idx) or ent["i"] not in added:', 'if ent["i"] not in added:'),
    "record exemption applied to ANY label saying `record` (no existing entry required)": (
        'and ent["key"] in plain_keys and not ent["key"].startswith("TEXT:")',
        'and not ent["key"].startswith("TEXT:")'),
    "provenance log / sidecars counted as run data": (
        "return bool(_RAW_RE.match(path)) and not _RAW_NOT_DATA_RE.search(path)", "return bool(_RAW_RE.match(path))"),
    "edits to an existing amendment's body (N3) ignored": (
        "if bodies and all(new.body(idx) != b for b in bodies):", "if False:"),
    "new amendment-log prose (N4) ignored": (
        "    if prose:\n        out.append((\"N4\"", "    if False:\n        out.append((\"N4\""),
    "amendment-same-commit escape ignored": (
        "if not act or _escaped(new_text, parent_texts):", "if not act:"),
    "escape matched on ANY line, not only the commit's own added lines": (
        "for i in (added or ()))", "for i in range(len(new_lines)))"),
    "qualified / erratum amendment labels not detected": (
        "        word, rest = (q.group(2).lower(), q.group(3)) if q else (None, \"\")",
        "        word, rest = (None, \"\")"),
    "glued bold entries accepted without an ID and a date": (
        "                    if glued and not (kind == \"entry\" and not key.startswith(\"TEXT:\")\n"
        "                                      and _DATE_RE.search(bm.group(2)[:60])):",
        "                    if glued and kind != \"entry\":"),
    "fails OPEN when git cannot read the index": (
        '        return ["CLASS PRA could not read', '        return []\n        return ["CLASS PRA could not read'),
    "fails OPEN on an unreadable MERGE_HEAD": (
        '            raise _GitReadError("MERGE_HEAD exists but cannot be read: %s" % e)',
        "            merge_heads = []"),
    "clean auto-merge judged against HEAD alone": (
        "    if kind == \"merge\":\n        return None", "    if kind == \"merge\":\n        return [head]"),
    "commit --amend judged against HEAD, not HEAD's parents": (
        '        return _git(["rev-parse", "HEAD^@"], root, env).stdout.decode().split()', "        return [head]"),
    "abbreviated --amend (`--am`, `--amen`) not recognised": (
        '            if opt == "amend":', '            if name == "amend":'),
    "an option's separate value read as a flag (`--message --amend`)": (
        "            elif opt and _COMMIT_LONG[opt] and not eq:\n                i += 1",
        "            elif False:\n                i += 1"),
    "an ABBREVIATED option's separate value read as a flag (`--mess --amend`, review r4 NIT)": (
        "            elif opt and _COMMIT_LONG[opt] and not eq:\n                i += 1",
        "            elif opt == name and _COMMIT_LONG[opt] and not eq:\n                i += 1"),
    "`-U <n>` (git 2.55) not known to take a value": (
        '_SHORT_WITH_VALUE = "mFcCtU"', '_SHORT_WITH_VALUE = "mFcCt"'),
    "commit --amend of a MERGE judged against its first parent only (HEAD^, not HEAD^@)": (
        '        return _git(["rev-parse", "HEAD^@"], root, env).stdout.decode().split()',
        '        return _git(["rev-parse", "HEAD^"], root, env).stdout.decode().split()'),
    "`git merge --continue`: the command consulted BEFORE MERGE_HEAD is read (review r4: fails OPEN)": (
        "    # MERGE_HEAD BEFORE `kind`:",
        "    if kind == \"merge\":\n        return None\n    # MERGE_HEAD BEFORE `kind`:"),
    "a qualified label naming an ID read as an entry (`## Rerun under AMENDMENT-1`, review r4)": (
        "        if q and not _DATE_RE.match(rest) and _ID_RE.match(rest):\n            word, rest = None, \"\"\n", ""),
    "the qualifier stop-list (`this`, `the`, `per`, ...) emptied": (
        "_QUAL_STOP = frozenset(\n", "_QUAL_STOP = frozenset() and frozenset(\n"),
    "prereg rename detection dropped": (
        "_changes(root, env, p, target, _PREREG_SPECS, True)", "_changes(root, env, p, target, _PREREG_SPECS, False)"),
    "/proc walker borrows a git process working in another directory": (
        '                if os.path.realpath("/proc/%d/cwd" % pid) != want:\n                    return None',
        "                if False:\n                    return None"),
    "/proc walker does not climb past the first ancestor": (
        '                pid = int(fh.read().rsplit(")", 1)[1].split()[1])', "                return None"),
}


def _mutated_source(old, new):
    src = open(_GATE_SRC, encoding="utf-8").read()
    assert src.count(old) == 1, "mutation anchor no longer matches the gate source exactly once: %r" % old[:80]
    return src.replace(old, new)


@pytest.mark.parametrize("label", sorted(MUTANTS))
def test_selftest_kills_mutants(label):
    mod = types.ModuleType("pra_mutant")
    mod.__file__ = _GATE_SRC
    exec(compile(_mutated_source(*MUTANTS[label]), _GATE_SRC, "exec"), mod.__dict__)
    assert mod.selftest(), "selftest() still PASSES with the mutant %r -- the registry would trust a broken gate" % label


# ---------------------------------------------------------------------------------------------------------
# 2. real commits through real hooks
# ---------------------------------------------------------------------------------------------------------
_HOOK = """#!/bin/sh
exec "%s" - <<'PY'
import os, subprocess, sys
sys.path.insert(0, %r)
import tools.gates.prereg_amendment_order as m
assert os.path.realpath(m.__file__) == os.path.realpath(%r), m.__file__
added = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=A"],
                       capture_output=True, text=True).stdout.split()
problems = m.check(added, root=os.getcwd())
with open(%r, "a") as fh:
    fh.write("%%s %%s\\n" %% (m._detect_invocation(os.getcwd()), "BLOCK" if problems else "pass"))
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


def _make_repo(tmp_path, gate_root=_REPO, gate_file=_GATE_SRC):
    """A scratch repo whose hooks dir is <root>/tools/githooks (untracked): the fixture's pre-commit, and this
    repo's REAL pre-merge-commit, which checks core.hooksPath and execs $ROOT/tools/githooks/pre-commit."""
    root = os.path.realpath(str(tmp_path / "main"))
    os.makedirs(root)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    hooks = os.path.join(root, "tools", "githooks")
    os.makedirs(hooks)
    log = str(tmp_path / "hook-runs.log")
    with open(os.path.join(hooks, "pre-commit"), "w") as fh:
        fh.write(_HOOK % (sys.executable, gate_root, gate_file, log))
    shutil.copy(os.path.join(_REPO, "tools", "githooks", "pre-merge-commit"), os.path.join(hooks, "pre-merge-commit"))
    for h in ("pre-commit", "pre-merge-commit"):
        os.chmod(os.path.join(hooks, h), 0o755)
    with open(os.path.join(root, ".git", "info", "exclude"), "a") as fh:
        fh.write("/tools/\n")
    _git(root, "config", "core.hooksPath", hooks)
    _write(root, PREREG, BASE)
    _write(root, "research/findings/raw/x/s7.json", '{"v": 1}')
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
    return r.returncode != 0 and "CLASS PRA" in (r.stdout + r.stderr)


def test_plain_add_amendment_with_a_new_artifact_is_blocked(repo):
    _write(repo, PREREG, BOLD_AMEND, "a")
    _write(repo, "research/findings/raw/x/s42.json", "{}")
    _git(repo, "add", "-A")
    assert _blocked(_git(repo, "commit", "-m", "amend+data", check=False))
    assert _hook_log(repo)[-2] == "commit BLOCK"


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


def test_commit_from_a_subdirectory_is_detected_as_a_commit(repo):
    """the /proc detection matches the git process by its cwd; git chdirs to the work-tree top before a hook."""
    _write(repo, PREREG, BOLD_AMEND, "a")
    _write(repo, "research/findings/raw/x/s42.json", "{}")
    _git(repo, "add", "-A")
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-m", "from a subdir"],
                       cwd=os.path.join(repo, "research", "findings"), env=_env(), capture_output=True, text=True)
    assert _blocked(r)
    assert _hook_log(repo)[-2] == "commit BLOCK"


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


# --- merges: pre-merge-commit runs BEFORE git writes MERGE_HEAD (review 2, MEDIUM-HIGH) -----------------------
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


def _clean_merge_commits(repo):
    _lane_with_ordered_history(repo)
    r = _git(repo, "merge", "--no-ff", "-m", "merge lane", "lane", check=False)
    return r.returncode == 0 and len(_git(repo, "rev-list", "--parents", "-n1", "HEAD").stdout.split()) == 3, r


def test_clean_auto_merge_of_correctly_ordered_history_passes(repo):
    """The hook REALLY runs during this merge (pre-merge-commit -> pre-commit), with no MERGE_HEAD yet; judged against
    HEAD alone it would block, as it did for 19 of 534 real main merges."""
    ok, r = _clean_merge_commits(repo)
    assert ok, r.stdout + r.stderr
    assert _hook_log(repo)[-2] == "merge pass", "the gate did not run during the merge, or misread it: %r" % _hook_log(repo)


def test_pull_merge_of_correctly_ordered_history_passes(repo):
    """`git pull` spawns `git merge FETCH_HEAD`; GIT_REFLOG_ACTION then reads `pull ...`, not `merge ...`."""
    _lane_with_ordered_history(repo)
    r = _git(repo, "pull", "--no-rebase", "--no-ff", "--no-edit", ".", "lane", check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _hook_log(repo)[-2] == "merge pass"


def test_no_commit_merge_finished_by_commit_passes(repo):
    """MERGE_HEAD present: judged against every parent (the path a conflicted merge takes)."""
    _lane_with_ordered_history(repo)
    _git(repo, "merge", "--no-ff", "--no-commit", "lane")
    r = _git(repo, "commit", "--no-edit", check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _hook_log(repo)[-2] == "commit pass"


def test_evil_merge_writing_a_new_amendment_and_data_is_blocked(repo):
    _lane_with_ordered_history(repo)
    _git(repo, "merge", "--no-ff", "--no-commit", "lane")
    _write(repo, PREREG, "\n## AMENDMENT 2 (written in the merge itself)\n\nG1 >= 0.7\n", "a")
    _write(repo, "research/findings/raw/x/s43.json", "{}")
    _git(repo, "add", "-A")
    assert _blocked(_git(repo, "commit", "-m", "evil merge", check=False))


# --- `git merge --continue` (review r4, LOW-MEDIUM): the hook's git command is `merge`, MERGE_HEAD present ------
def _conflicted_merge(repo):
    """lane: amendment, then data, then its own unrelated.txt; main: a different unrelated.txt -> add/add conflict."""
    _lane_with_ordered_history(repo)
    _git(repo, "checkout", "-q", "lane")
    _write(repo, "unrelated.txt", "lane side\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "lane writes unrelated.txt too")
    _git(repo, "checkout", "-q", "main")
    r = _git(repo, "merge", "--no-ff", "lane", check=False)
    assert r.returncode != 0 and "CONFLICT" in r.stdout, r.stdout + r.stderr
    assert os.path.exists(os.path.join(repo, ".git", "MERGE_HEAD"))
    _write(repo, "unrelated.txt", "resolved\n")


def _merge_continue(repo):
    """`git merge --continue` runs cmd_commit IN-PROCESS: the pre-commit hook's git process reads `git merge`."""
    return subprocess.run(["git", "-c", "commit.gpgsign=false", "merge", "--continue"], cwd=repo,
                          env=dict(_env(), GIT_EDITOR="true"), capture_output=True, text=True, timeout=60)


def _evil_merge_continue(repo):
    _conflicted_merge(repo)
    _write(repo, PREREG, "\n## AMENDMENT 2 (written while resolving the conflict)\n\nG1 >= 0.7\n", "a")
    _write(repo, "research/findings/raw/x/s43.json", "{}")
    _git(repo, "add", "-A")
    return _merge_continue(repo)


def test_merge_continue_of_a_resolved_conflict_passes(repo):
    _conflicted_merge(repo)
    _git(repo, "add", "-A")
    r = _merge_continue(repo)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(_git(repo, "rev-list", "--parents", "-n1", "HEAD").stdout.split()) == 3
    assert _hook_log(repo)[-2] == "merge pass", _hook_log(repo)


def test_merge_continue_writing_a_new_amendment_and_data_is_blocked(repo):
    r = _evil_merge_continue(repo)
    assert _blocked(r), r.stdout + r.stderr
    assert _hook_log(repo)[-2] == "merge BLOCK", "the hook did not see `git merge` as the command: %r" % _hook_log(repo)
    assert os.path.exists(os.path.join(repo, ".git", "MERGE_HEAD"))          # nothing was committed


def test_amend_of_a_merge_commit_is_judged_against_every_parent(repo):
    ok, r = _clean_merge_commits(repo)
    assert ok, r.stdout + r.stderr
    r = _git(repo, "commit", "--amend", "--no-edit", check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _hook_log(repo)[-2] == "amend pass"


# --- review r4: results headings that only POINT at an amendment ------------------------------------------------
@pytest.mark.parametrize("results", [
    "\n## Rerun under AMENDMENT-1 (v2, seed 7): NO-GO\n\nG1 read 0.4.\n",
    "\n## Results\n\n### Results under amendment 1\n\nG1 read 0.4.\n",
    "\n## Results\n\n**Verdict after amendment 1:** NO-GO, G1 read 0.4.\n",
])
def test_results_naming_a_committed_amendment_appended_with_data_pass(repo, results):
    _write(repo, PREREG, "\n## AMENDMENT 1 (2026-01-02, before round 2)\n\nG1 >= 0.6\n", "a")
    _git(repo, "commit", "-q", "-am", "amendment first")
    _write(repo, PREREG, results, "a")
    _write(repo, "research/findings/raw/x/s42.json", "{}")
    _git(repo, "add", "-A")
    r = _git(repo, "commit", "-m", "results + data", check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_abbreviated_message_option_takes_the_next_word_as_its_value(repo):
    """review r4 NIT: `git commit --mess --amend` makes a NEW commit whose message is `--amend`, so data after an
    amendment committed alone is ordered and must pass (read as --amend it was judged against HEAD^ and blocked)."""
    _write(repo, PREREG, BOLD_AMEND, "a")
    _git(repo, "commit", "-q", "-am", "amendment alone")
    _write(repo, "research/findings/raw/x/s42.json", "{}")
    _git(repo, "add", "-A")
    r = _git(repo, "commit", "--mess", "--amend", check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _git(repo, "log", "-1", "--format=%s").stdout.strip() == "--amend"
    assert _hook_log(repo)[-2] == "commit pass"


# --- commit --amend REPLACES HEAD (review 2, MEDIUM) ------------------------------------------------------------
def _amend_folds_data_into_the_amendment_commit(repo, flag="--amend"):
    _write(repo, PREREG, BOLD_AMEND, "a")
    _git(repo, "commit", "-q", "-am", "amendment alone")
    _write(repo, "research/findings/raw/x/s42.json", "{}")
    _git(repo, "add", "-A")
    return _git(repo, "commit", flag, "--no-edit", check=False)


@pytest.mark.parametrize("flag", ["--amend", "--amen"])
def test_amend_folding_data_into_an_amendment_commit_is_blocked(repo, flag):
    """the review's reproduction: amendment committed alone, data added by `commit --amend` -> ONE commit, both."""
    assert _blocked(_amend_folds_data_into_the_amendment_commit(repo, flag))
    assert _hook_log(repo)[-2] == "amend BLOCK"


def test_amend_of_a_data_commit_after_an_ordered_amendment_passes(repo):
    _write(repo, PREREG, BOLD_AMEND, "a")
    _git(repo, "commit", "-q", "-am", "amendment first")
    _write(repo, "research/findings/raw/x/s42.json", "{}")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "then data")
    _write(repo, "research/findings/raw/x/s43.json", "{}")
    _git(repo, "add", "-A")
    r = _git(repo, "commit", "--amend", "--no-edit", check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _hook_log(repo)[-2] == "amend pass"


def test_amend_rewording_an_amendment_only_commit_passes(repo):
    _write(repo, PREREG, BOLD_AMEND, "a")
    _git(repo, "commit", "-q", "-am", "amendment alone")
    r = _git(repo, "commit", "--amend", "-m", "amendment alone, reworded", check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_every_diff_is_pathspec_limited_and_only_prereg_diffs_detect_renames(repo, monkeypatch):
    """review 2 (LOW, cost): whole-tree `-M` cost 5.8 s per 1000-commit-divergent parent, and past the timeout the
    gate fails CLOSED. Every diff the gate runs must name a pathspec; `-M` only over the prereg pathspecs."""
    calls = []
    real_git = pra_gate._git

    def spy(args, root, env, ok_codes=(0,)):
        calls.append(list(args))
        return real_git(args, root, env, ok_codes)

    monkeypatch.setattr(pra_gate, "_git", spy)
    _write(repo, PREREG, BOLD_AMEND, "a")
    _write(repo, "research/findings/raw/x/s42.json", "{}")
    _git(repo, "add", "-A")
    assert pra_gate.check([], root=repo, invocation="commit")          # it read far enough to block
    diffs = [a for a in calls if a[0] == "diff"]
    assert len(diffs) == 2, diffs                                        # the prereg diff, then the raw/ diff
    for a in diffs:
        specs = a[a.index("--") + 1:]
        assert specs, "a diff over the whole tree: %r" % a
        if "-M" in a:
            assert set(specs) <= set(pra_gate._PREREG_SPECS), a
        else:
            assert "--no-renames" in a, a


def test_unreadable_repository_fails_closed(tmp_path):
    broken = str(tmp_path / "broken")
    os.makedirs(broken)
    with open(os.path.join(broken, ".git"), "w") as fh:
        fh.write("gitdir: %s\n" % (tmp_path / "missing"))
    problems = pra_gate.check([], root=broken)
    assert len(problems) == 1 and "failing CLOSED" in problems[0]


# ---------------------------------------------------------------------------------------------------------
# 3. the real-hook scenarios against MUTATED gates: each fix is what makes its test pass
# ---------------------------------------------------------------------------------------------------------
HOOK_MUTANTS = {
    "clean auto-merge judged against HEAD alone": (
        "    if kind == \"merge\":\n        return None", "    if kind == \"merge\":\n        return [head]", "merge"),
    "commit --amend judged against HEAD": (
        '        return _git(["rev-parse", "HEAD^@"], root, env).stdout.decode().split()', "        return [head]",
        "amend"),
    "the invoking git command never detected": (
        "def _detect_invocation(root):\n    return _invocation_kind(_invoking_git_argv(root))",
        "def _detect_invocation(root):\n    return None", "both"),
    "the command consulted BEFORE MERGE_HEAD (review r4: `git merge --continue` fails open)": (
        "    # MERGE_HEAD BEFORE `kind`:",
        "    if kind == \"merge\":\n        return None\n    # MERGE_HEAD BEFORE `kind`:", "merge-continue"),
    "an abbreviated option's value read as a flag (`--mess --amend`)": (
        "            elif opt and _COMMIT_LONG[opt] and not eq:\n                i += 1",
        "            elif opt == name and _COMMIT_LONG[opt] and not eq:\n                i += 1", "mess-amend"),
}


@pytest.mark.parametrize("label", sorted(HOOK_MUTANTS))
def test_real_hook_scenarios_fail_under_mutant(tmp_path, label):
    old, new, scenario = HOOK_MUTANTS[label]
    gate_root = str(tmp_path / "mutant_gate")
    os.makedirs(os.path.join(gate_root, "tools", "gates"))
    for init in ("tools/__init__.py", "tools/gates/__init__.py"):
        open(os.path.join(gate_root, init), "w").close()
    gate_file = os.path.join(gate_root, "tools", "gates", "prereg_amendment_order.py")
    with open(gate_file, "w", encoding="utf-8") as fh:
        fh.write(_mutated_source(old, new))
    if scenario in ("merge", "both"):
        repo = _make_repo(tmp_path / "m", gate_root, gate_file)
        ok, _r = _clean_merge_commits(repo)
        assert not ok, "the clean merge still passes with the mutant %r -- its test proves nothing" % label
    if scenario in ("amend", "both"):
        repo = _make_repo(tmp_path / "a", gate_root, gate_file)
        r = _amend_folds_data_into_the_amendment_commit(repo)
        assert r.returncode == 0, "the amend still blocks with the mutant %r -- its test proves nothing" % label
    if scenario == "merge-continue":
        repo = _make_repo(tmp_path / "c", gate_root, gate_file)
        r = _evil_merge_continue(repo)
        assert r.returncode == 0, "the evil `merge --continue` still blocks with the mutant %r" % label
        assert _hook_log(repo)[-2] == "merge pass"                          # the gate ran, and failed OPEN
    if scenario == "mess-amend":
        repo = _make_repo(tmp_path / "s", gate_root, gate_file)
        _write(repo, PREREG, BOLD_AMEND, "a")
        _git(repo, "commit", "-q", "-am", "amendment alone")
        _write(repo, "research/findings/raw/x/s42.json", "{}")
        _git(repo, "add", "-A")
        assert _blocked(_git(repo, "commit", "--mess", "--amend", check=False)), \
            "`--mess --amend` still passes with the mutant %r -- its test proves nothing" % label


# ---------------------------------------------------------------------------------------------------------
# 4. the review's named commits, from this repository's own history, through the gate's own _evaluate
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
    parents = _real("rev-parse", sha + "^@").stdout.decode().split()
    return pra_gate._evaluate(_REPO, _env(), parents, target=sha)


@pytest.mark.parametrize("sha, should_block", REAL)
def test_review_named_commits(sha, should_block):
    if _real("cat-file", "-e", sha + "^{commit}").returncode != 0 or _real("cat-file", "-e", sha + "^").returncode != 0:
        pytest.skip("commit %s not in this clone" % sha)
    assert bool(_replay(sha)) is should_block


@pytest.mark.parametrize("sha", ["df12ec1cc", "f793b6945"])
def test_real_merges_pass_against_every_parent_but_block_against_head_alone(sha):
    """two of the 19 real main merges the pre-fix hook would have false-blocked during a clean `git merge`."""
    if _real("cat-file", "-e", sha + "^2").returncode != 0:
        pytest.skip("merge %s not in this clone" % sha)
    parents = _real("rev-parse", sha + "^@").stdout.decode().split()
    assert pra_gate._evaluate(_REPO, _env(), parents, target=sha) == []
    assert pra_gate._evaluate(_REPO, _env(), parents[:1], target=sha)
