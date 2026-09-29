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
EXISTING_PATH = "CLAUDE.md"           # committed at HEAD in every checkout of this repo
MISSING_PATH = "docs/no-such-prereg-xyz-zzz.md"


def _run(cmd, tmp_path):
    env = dict(os.environ)
    env["POOL_QUEUE_PATH"] = str(tmp_path / "pool.queue")
    env["POOL_SSH_CONFIG"] = str(tmp_path / "no-such-pool-ssh-config")
    env["POOL_EXTRA_NODES_FILE"] = str(tmp_path / "no-such-extra-nodes")
    return subprocess.run(["bash", str(POOL_QUEUE), *cmd], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=30)


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


# ------------------------------------------------------------------------ gate seeds accepted with a prereg

def test_accepts_gate_seed_with_a_prereg_that_exists_at_the_pinned_revision(tmp_path):
    res = _run(
        ["add", f"{_pinned(HEAD_SHA)} && true --seed 100", "--checked", f"prereg {EXISTING_PATH}"],
        tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert "queued" in res.stdout
    assert (tmp_path / "pool.queue").read_text().strip() != ""


def test_accepts_comma_separated_gate_seeds_with_a_valid_prereg(tmp_path):
    res = _run(
        ["add", f"{_pinned(HEAD_SHA)} && true --seeds 42,43,44,100,101,102",
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
