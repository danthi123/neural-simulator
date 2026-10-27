"""Tests for tools/pool_queue.sh `add`'s GATE-SEED GUARD (2026-09-29,
docs/plans/2026-09-29-local-model-transition-PLAN.md section 4).

The local model ran the six GATE seeds (42/43/44/100/101/102) on 2026-09-28 with no committed
preregistration (LOCAL_LLM_RUNBOOK.md's "2026-09-28 VIOLATION" entry). A GATE-seed result is meant to be a
verdict, and a verdict without a prereg is exactly the tuning-toward-the-answer failure mode the dev/gate-seed
split exists to prevent. This guard refuses `pool_queue.sh add` for any job whose `--seed`/`--seeds` value
includes a GATE seed, unless the `--checked` reason names a prereg (`prereg <path>`) that exists (via
`git cat-file -e <sha>:<path>`) at the job's pinned revision (`cd ~/derisk-pool/revisions/<sha>`).

Dev seeds (7, 1-15) are never gated. Uses commands with NO `-m research.runners.X` pattern so `add`'s
MOD-based local-import / remote-node checks never engage (same technique as
test_pool_queue_first_word_gate.py / test_pool_queue_node_validation.py), keeping these tests fast and
independent of any pool node. Every invocation overrides POOL_QUEUE_PATH to a tmp_path file -- the real
research/queue/pool.queue is never touched.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL_QUEUE = ROOT / "tools" / "pool_queue.sh"

HEAD_SHA = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
).stdout.strip()
EXISTING_PATH = "research/findings/2026-09-29-host-share-baselines-PREREGISTRATION.md"   # a real committed prereg at HEAD (name must contain "prereg")
MISSING_PATH = "docs/no-such-prereg-xyz-zzz.md"


def _preflight_clean(tmp_path, prereg, shas):
    """A CLEAN scorer pre-flight row per (prereg, sha): the 2026-10-16 pre-flight guard (tests/test_gate_preflight.py)
    runs after the canary guard; these tests keep testing their own guard."""
    p = tmp_path / "preflight.tsv"
    if not p.exists():
        p.write_text("".join("2026-10-16 00:00\t%s\t%s\tCLEAN\tok\t\tfixture\n" % (prereg, s[:9]) for s in shas))
    return str(p)


def _run(cmd, tmp_path):
    env = dict(os.environ)
    env["GATE_PREFLIGHT_PATH"] = _preflight_clean(tmp_path, EXISTING_PATH, [HEAD_SHA])
    env["POOL_QUEUE_PATH"] = str(tmp_path / "pool.queue")
    env["POOL_SSH_CONFIG"] = str(tmp_path / "no-such-pool-ssh-config")
    env["POOL_EXTRA_NODES_FILE"] = str(tmp_path / "no-such-extra-nodes")
    # the pool-canary guard (2026-09-30, tests/test_pool_canary_guard.py) runs after this guard: point it at a
    # tmp record file so these tests never read the real research/queue/pool_canaries.tsv
    env["POOL_CANARIES_PATH"] = str(tmp_path / "pool_canaries.tsv")
    return subprocess.run(["bash", str(POOL_QUEUE), *cmd], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=30)


CANARY_MOD = "tools.fake_canary_runner"


def _canary_pass(tmp_path):
    """A PASS pool-canary record for (CANARY_MOD, HEAD): the pool-canary guard's own precondition, so the
    acceptance tests below keep testing THIS guard (prereg) and not that one."""
    (tmp_path / "pool_canaries.tsv").write_text(
        "2026-09-30T00:00:00-04:00\t%s\t%s\tPASS\tpool41\tgate\ttest\tfixture record for the prereg-guard tests\n"
        % (CANARY_MOD, HEAD_SHA))


def _queue_is_empty(tmp_path):
    q = tmp_path / "pool.queue"
    return not q.exists() or q.read_text() == ""


def _pinned(sha):
    return f"cd ~/derisk-pool/revisions/{sha}"


# ---------------------------------------------------------------- gate seeds refused without a prereg token

def test_refuses_gate_seed_with_no_prereg_token_at_all(tmp_path):
    res = _run(["add", f"{_pinned(HEAD_SHA)} && true --seed 42", "--checked", "nothing on record for this"],
                tmp_path)
    assert res.returncode == 2
    assert "REFUSED" in res.stderr
    assert "42" in res.stderr
    assert "prereg" in res.stderr.lower()
    assert _queue_is_empty(tmp_path)


def test_refuses_gate_seed_with_prereg_but_no_pinned_revision(tmp_path):
    # A prereg is named, but the job is not pinned to any revision, so it cannot be checked against one.
    res = _run(["add", "true --seed 100", "--checked", f"prereg {EXISTING_PATH}"], tmp_path)
    assert res.returncode == 2
    assert "REFUSED" in res.stderr
    assert "pinned" in res.stderr.lower()
    assert _queue_is_empty(tmp_path)


def test_refuses_gate_seed_when_prereg_path_does_not_exist_at_pinned_revision(tmp_path):
    res = _run(
        ["add", f"{_pinned(HEAD_SHA)} && true --seed 44", "--checked", f"prereg {MISSING_PATH}"],
        tmp_path,
    )
    assert res.returncode == 2
    assert "REFUSED" in res.stderr
    assert "does not exist" in res.stderr
    assert MISSING_PATH in res.stderr
    assert _queue_is_empty(tmp_path)


def test_refuses_when_any_one_of_several_seeds_is_a_gate_seed(tmp_path):
    res = _run(["add", f"{_pinned(HEAD_SHA)} && true --seeds 7 42", "--checked", "no prereg"], tmp_path)
    assert res.returncode == 2
    assert "REFUSED" in res.stderr
    assert "42" in res.stderr
    assert _queue_is_empty(tmp_path)


def test_refuses_comma_separated_gate_seeds_without_prereg(tmp_path):
    res = _run(
        ["add", f"{_pinned(HEAD_SHA)} && true --seeds 42,43,44,100,101,102", "--checked", "no prereg"],
        tmp_path,
    )
    assert res.returncode == 2
    assert "REFUSED" in res.stderr
    assert _queue_is_empty(tmp_path)


# ------------------------------------------- the cited path must be NAMED a prereg (2026-10-27 loophole closed)
# On 2026-10-22 cs-race f1g gate-seed lines (seeds 41-48 incl. 42/43/44) were queued with
# `prereg research/coordination/LOCAL_LLM_RUNBOOK.md`: it is committed at the pin, so the cat-file test passed. A
# runbook / packet / ruling is not a preregistration; the basename must contain "prereg" (case-insensitive).
RUNBOOK_PATH = "research/coordination/LOCAL_LLM_RUNBOOK.md"


def test_refuses_gate_seed_citing_a_committed_runbook_as_its_prereg(tmp_path):
    assert subprocess.run(["git", "cat-file", "-e", f"HEAD:{RUNBOOK_PATH}"], cwd=ROOT).returncode == 0
    _canary_pass(tmp_path)
    res = _run(
        ["add", f"{_pinned(HEAD_SHA)} && true -m {CANARY_MOD} --seeds 41 42 43 44",
         "--checked", f"prereg {RUNBOOK_PATH} cs-race f1g gate seeds"],
        tmp_path,
    )
    assert res.returncode == 2
    assert "REFUSED" in res.stderr
    assert "42" in res.stderr and "43" in res.stderr and "44" in res.stderr
    assert "not a preregistration" in res.stderr
    assert "prereg" in res.stderr and RUNBOOK_PATH in res.stderr
    assert _queue_is_empty(tmp_path)


def test_refuses_other_committed_non_prereg_paths_on_a_gate_seed(tmp_path):
    # CLAUDE.md and a ruling-style file exist at HEAD but are not preregistrations either
    _canary_pass(tmp_path)
    for bad in ("CLAUDE.md", "docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md"):
        res = _run(["add", f"{_pinned(HEAD_SHA)} && true -m {CANARY_MOD} --seed 102", "--checked", f"prereg {bad}"],
                   tmp_path)
        assert res.returncode == 2, bad
        assert "not a preregistration" in res.stderr, bad
        assert _queue_is_empty(tmp_path), bad


def test_name_check_is_on_the_basename_not_the_directory(tmp_path):
    # a directory called prereg/ does not make a runbook a prereg
    _canary_pass(tmp_path)
    res = _run(["add", f"{_pinned(HEAD_SHA)} && true -m {CANARY_MOD} --seed 43", "--checked",
                "prereg research/prereg/LOCAL_LLM_RUNBOOK.md"], tmp_path)
    assert res.returncode == 2
    assert "not a preregistration" in res.stderr
    assert _queue_is_empty(tmp_path)


def test_dev_seed_may_still_cite_a_non_prereg_path(tmp_path):
    res = _run(["add", f"{_pinned(HEAD_SHA)} && true --seed 7", "--checked", f"prereg {RUNBOOK_PATH} dev smoke"],
               tmp_path)
    assert res.returncode == 0, res.stderr
    assert "queued" in res.stdout


def test_accepts_a_lowercase_prereg_named_path(tmp_path):
    # the name convention is case-insensitive: *-prereg.md (lowercase) passes the name check and the pinned-rev check
    lower = "research/findings/2026-10-12-validation-battery-prereg.md"
    assert subprocess.run(["git", "cat-file", "-e", f"HEAD:{lower}"], cwd=ROOT).returncode == 0
    _canary_pass(tmp_path)
    env_pf = tmp_path / "preflight.tsv"
    env_pf.write_text("2026-10-16 00:00\t%s\t%s\tCLEAN\tok\t\tfixture\n" % (lower, HEAD_SHA[:9]))
    env = dict(os.environ)
    env.update(GATE_PREFLIGHT_PATH=str(env_pf), POOL_QUEUE_PATH=str(tmp_path / "pool.queue"),
               POOL_SSH_CONFIG=str(tmp_path / "no-such-pool-ssh-config"),
               POOL_EXTRA_NODES_FILE=str(tmp_path / "no-such-extra-nodes"),
               POOL_CANARIES_PATH=str(tmp_path / "pool_canaries.tsv"))
    res = subprocess.run(
        ["bash", str(POOL_QUEUE), "add", f"{_pinned(HEAD_SHA)} && true -m {CANARY_MOD} --seed 101",
         "--checked", f"prereg {lower} lowercase convention"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert "queued" in res.stdout
    assert (tmp_path / "pool.queue").read_text().strip() != ""


# ------------------------------------------------------------------------ gate seeds accepted with a prereg

def test_accepts_gate_seed_with_a_prereg_that_exists_at_the_pinned_revision(tmp_path):
    _canary_pass(tmp_path)
    res = _run(
        ["add", f"{_pinned(HEAD_SHA)} && true -m {CANARY_MOD} --seed 100", "--checked", f"prereg {EXISTING_PATH}"],
        tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert "queued" in res.stdout
    assert (tmp_path / "pool.queue").read_text().strip() != ""


def test_accepts_comma_separated_gate_seeds_with_a_valid_prereg(tmp_path):
    _canary_pass(tmp_path)
    res = _run(
        ["add", f"{_pinned(HEAD_SHA)} && true -m {CANARY_MOD} --seeds 42,43,44,100,101,102",
         "--checked", f"prereg {EXISTING_PATH}"],
        tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert "queued" in res.stdout


# ----------------------------------------------------------------------------------- dev seeds are unaffected

def test_accepts_a_single_dev_seed_with_no_prereg_and_no_pinned_revision(tmp_path):
    res = _run(["add", "true --seed 7", "--checked", "dev smoke, no prereg needed"], tmp_path)
    assert res.returncode == 0, res.stderr
    assert "queued" in res.stdout


def test_accepts_several_dev_seeds_with_no_prereg(tmp_path):
    res = _run(["add", f"{_pinned(HEAD_SHA)} && true --seeds 1 2 3", "--checked", "dev sweep"], tmp_path)
    assert res.returncode == 0, res.stderr
    assert "queued" in res.stdout


def test_seed_frac_flag_is_not_mistaken_for_a_seed_value(tmp_path):
    # --seed-frac 0.5 must never be parsed as a --seed value (it isn't even an integer).
    res = _run(["add", "true --seed-frac 0.5", "--checked", "dev, no gate seeds here"], tmp_path)
    assert res.returncode == 0, res.stderr
    assert "queued" in res.stdout

# ------------------------------------------------- a measurement of an OLDER pin (RULING lbf-gate-pin-1020, 2026-10-20)
# The LBF gate-seed prereg (flip: no) registers a measurement of S main f2885df70, a revision older than the prereg
# itself (first added at bc9ac887d), so the prereg can never be in the pinned tree. It is accepted on the narrow
# historical-pin rule; a pin it does not name, or a prereg without `flip: no`, is still refused.
LBF_PREREG = "research/findings/2026-10-20-lbf-at-head-gate-seed-measurement-PREREGISTRATION.md"
LBF_PIN = "f2885df70cfb99a8715c376bca12ac3c2134ef46"
WRC_PREREG = "research/findings/2026-10-16-word-role-flip-PREREGISTRATION.md"   # no `flip: no` line


def _run_pinned(cmd, tmp_path, prereg, sha):
    (tmp_path / "pool_canaries.tsv").write_text(
        "2026-09-30T00:00:00-04:00\t%s\t%s\tPASS\tpool41\tgate\ttest\tfixture record\n" % (CANARY_MOD, sha))
    (tmp_path / "preflight.tsv").write_text("2026-10-20 00:00\t%s\t%s\tCLEAN\tok\t\tfixture\n" % (prereg, sha[:9]))
    env = dict(os.environ)
    env.update(GATE_PREFLIGHT_PATH=str(tmp_path / "preflight.tsv"), POOL_QUEUE_PATH=str(tmp_path / "pool.queue"),
               POOL_SSH_CONFIG=str(tmp_path / "no-such-pool-ssh-config"),
               POOL_EXTRA_NODES_FILE=str(tmp_path / "no-such-extra-nodes"),
               POOL_CANARIES_PATH=str(tmp_path / "pool_canaries.tsv"))
    return subprocess.run(["bash", str(POOL_QUEUE), *cmd], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)


def _rev(spec):
    return subprocess.run(["git", "rev-parse", spec], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def test_accepts_a_flip_no_measurement_of_the_older_pin_it_names(tmp_path):
    res = _run_pinned(["add", f"{_pinned(LBF_PIN[:9])} && true -m {CANARY_MOD} --seed 42", "--checked",
                       f"prereg {LBF_PREREG} gate seed 42"], tmp_path, LBF_PREREG, LBF_PIN)
    assert res.returncode == 0, res.stderr
    assert "older pin" in res.stderr and "bc9ac887d" in res.stderr
    assert (tmp_path / "pool.queue").read_text().strip() != ""


def test_refuses_an_older_pin_the_prereg_does_not_name(tmp_path):
    parent = _rev(LBF_PIN + "^")
    res = _run_pinned(["add", f"{_pinned(parent)} && true -m {CANARY_MOD} --seed 42", "--checked",
                       f"prereg {LBF_PREREG}"], tmp_path, LBF_PREREG, parent)
    assert res.returncode == 2
    assert "does not exist at pinned revision" in res.stderr and "does not name the full pinned sha" in res.stderr
    assert _queue_is_empty(tmp_path)


def test_refuses_an_older_pin_for_a_prereg_without_flip_no(tmp_path):
    born = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "HEAD", "--", WRC_PREREG], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.split()[-1]
    parent = _rev(born + "^")
    res = _run_pinned(["add", f"{_pinned(parent)} && true -m {CANARY_MOD} --seed 42", "--checked",
                       f"prereg {WRC_PREREG}"], tmp_path, WRC_PREREG, parent)
    assert res.returncode == 2
    assert "does not declare 'flip: no'" in res.stderr
    assert _queue_is_empty(tmp_path)
