"""Tests for the POOL-CANARY GUARD (2026-09-30): tools/pool_queue.sh `add` refuses a GATE-seed line
(42/43/44/100/101/102, detected exactly as the gate-seed guard detects them) until a canary of the same runner
module at the same revision has PASSED on a pool node, as recorded by tools/pool_canary.sh in
research/queue/pool_canaries.tsv (the latest record for the pair decides).

WHY: on 2026-09-29 about 470 gate-seed lines were queued for eight runner families after workstation dev-seed
checks only. Run for real on pool41/pool42, every line of three families died in the worker's argparse
(host_share, stage1_exit_gate and stage1_c1_category spawned '--worker --seed N' without --allow-gate-seed) and
a fourth died at import (no httpx in the pool venvs).

Guard tests use commands whose module is not under research.runners, so `add`'s argparse / remote-node probes
never engage (the technique test_pool_queue_gate_seed_guard.py uses). `run` tests drive tools/pool_canary.sh
through a stub `ssh` that runs the remote command in a fake $HOME holding a fake provisioned revision, so the
launch, the wait, the result scan and the record are all the real code. Nothing touches the real queue or the
real canary records: every call overrides POOL_QUEUE_PATH and POOL_CANARIES_PATH.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL_QUEUE = ROOT / "tools" / "pool_queue.sh"
CANARY = ROOT / "tools" / "pool_canary.sh"


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


HEAD_SHA = _git("rev-parse", "HEAD")
PARENT_SHA = _git("rev-parse", "HEAD~1")
PREREG = "CLAUDE.md"                      # committed at HEAD in every checkout
MOD = "tools.fake_canary_runner"


def _env(tmp_path, **extra):
    env = dict(os.environ)
    env["POOL_QUEUE_PATH"] = str(tmp_path / "pool.queue")
    env["POOL_CANARIES_PATH"] = str(tmp_path / "pool_canaries.tsv")
    env["POOL_SSH_CONFIG"] = str(tmp_path / "no-such-pool-ssh-config")
    env["POOL_EXTRA_NODES_FILE"] = str(tmp_path / "no-such-extra-nodes")
    env.update(extra)
    return env


def _add(tmp_path, cmd, checked=f"prereg {PREREG} (canary guard test) mem_gb=2 max_age_h=24"):
    return subprocess.run(["bash", str(POOL_QUEUE), "add", cmd, "--checked", checked], cwd=ROOT,
                          env=_env(tmp_path), capture_output=True, text=True, timeout=60)


def _canary(tmp_path, *args, **extra):
    return subprocess.run(["bash", str(CANARY), *args], cwd=ROOT, env=_env(tmp_path, **extra),
                          capture_output=True, text=True, timeout=120)


def _gate_line(sha=HEAD_SHA, mod=MOD, seed="42"):
    return f"cd ~/derisk-pool/revisions/{sha} && true -m {mod} --seed {seed}"


def _queue_is_empty(tmp_path):
    q = tmp_path / "pool.queue"
    return not q.exists() or q.read_text() == ""


def _record(tmp_path, verdict, sha=HEAD_SHA, mod=MOD, kind="gate"):
    r = _canary(tmp_path, "record", mod, sha, verdict, "pool41", kind,
                "sandbox selftest --job, v2 rc=0, 4 turns, 0 errored turns")
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------------------------------------------ the guard itself

def test_refuses_a_gate_seed_line_with_no_canary_record(tmp_path):
    res = _add(tmp_path, _gate_line())
    assert res.returncode == 2, res.stdout + res.stderr
    assert "REFUSED" in res.stderr and "canary" in res.stderr
    assert "pool_canary.sh run" in res.stderr
    assert _queue_is_empty(tmp_path)


def test_accepts_a_gate_seed_line_once_its_runner_and_revision_passed_a_canary(tmp_path):
    _record(tmp_path, "PASS")
    res = _add(tmp_path, _gate_line())
    assert res.returncode == 0, res.stderr
    assert "queued" in res.stdout


def test_a_pass_at_another_revision_does_not_count(tmp_path):
    _record(tmp_path, "PASS", sha=PARENT_SHA)
    res = _add(tmp_path, _gate_line(sha=HEAD_SHA))
    assert res.returncode == 2
    assert "canary" in res.stderr
    assert _queue_is_empty(tmp_path)


def test_a_pass_for_another_runner_does_not_count(tmp_path):
    _record(tmp_path, "PASS", mod="tools.some_other_runner")
    res = _add(tmp_path, _gate_line())
    assert res.returncode == 2
    assert _queue_is_empty(tmp_path)


def test_a_later_fail_supersedes_an_earlier_pass(tmp_path):
    _record(tmp_path, "PASS")
    _record(tmp_path, "FAIL")
    res = _add(tmp_path, _gate_line())
    assert res.returncode == 2
    assert "FAIL" in res.stderr
    assert _queue_is_empty(tmp_path)


def test_a_dev_seed_canary_does_not_open_a_pair_to_gate_seed_lines(tmp_path):
    # THE 2026-09-30 DEFECT: the worker's gate-seed guard fired only for gate seeds, so a dev-seed canary passed.
    _record(tmp_path, "PASS", kind="dev")
    res = _add(tmp_path, _gate_line())
    assert res.returncode == 2
    assert "GATE-seed pool canary" in res.stderr
    assert _queue_is_empty(tmp_path)


def test_a_later_gate_fail_is_not_cleared_by_a_dev_pass(tmp_path):
    _record(tmp_path, "PASS")
    _record(tmp_path, "FAIL")
    _record(tmp_path, "PASS", kind="dev")
    res = _add(tmp_path, _gate_line())
    assert res.returncode == 2
    assert "gate-seed pool canary" in res.stderr and "FAIL" in res.stderr


def test_a_later_dev_fail_blocks_until_a_newer_pass(tmp_path):
    _record(tmp_path, "PASS")
    _record(tmp_path, "FAIL", kind="dev")
    assert _add(tmp_path, _gate_line()).returncode == 2
    _record(tmp_path, "PASS", kind="dev")
    assert _add(tmp_path, _gate_line()).returncode == 0


def test_every_gate_seed_is_guarded(tmp_path):
    for seed in ("42", "43", "44", "100", "101", "102"):
        res = _add(tmp_path, _gate_line(seed=seed))
        assert res.returncode == 2, seed
    res = _add(tmp_path, f"cd ~/derisk-pool/revisions/{HEAD_SHA} && true -m {MOD} --seeds 7,101")
    assert res.returncode == 2
    assert _queue_is_empty(tmp_path)


def test_a_gate_seed_line_with_no_module_cannot_be_matched_and_is_refused(tmp_path):
    _record(tmp_path, "PASS")
    res = _add(tmp_path, f"cd ~/derisk-pool/revisions/{HEAD_SHA} && true --seed 42")
    assert res.returncode == 2
    assert "module" in res.stderr
    assert _queue_is_empty(tmp_path)


def test_a_short_sha_in_the_line_matches_a_full_sha_record(tmp_path):
    _record(tmp_path, "PASS", sha=HEAD_SHA)
    res = _add(tmp_path, _gate_line(sha=HEAD_SHA[:9]))
    assert res.returncode == 0, res.stderr


def test_dev_seed_lines_need_no_canary(tmp_path):
    res = _add(tmp_path, _gate_line(seed="7"), checked="dev smoke, seed 7")
    assert res.returncode == 0, res.stderr
    res = _add(tmp_path, f"cd ~/derisk-pool/revisions/{HEAD_SHA} && true -m {MOD} --seeds 1 2 3",
               checked="dev sweep")
    assert res.returncode == 0, res.stderr


def test_the_prereg_guard_still_runs_first(tmp_path):
    _record(tmp_path, "PASS")
    res = _add(tmp_path, _gate_line(), checked="no prereg named here")
    assert res.returncode == 2
    assert "prereg" in res.stderr.lower()


# ----------------------------------------------------------------------------------- record / has / key

def test_record_refuses_a_bad_verdict_a_control_character_or_no_evidence(tmp_path):
    ev = "sandbox selftest --job, v2 rc=0, 4 turns"
    assert _canary(tmp_path, "record", MOD, HEAD_SHA, "OK", "pool41", "gate", ev).returncode == 2
    assert _canary(tmp_path, "record", MOD, HEAD_SHA, "PASS", "pool41", "gate", "rc=0\tfine turns ok").returncode == 2
    assert _canary(tmp_path, "record", MOD, HEAD_SHA, "PASS", "pool41", "gate", "ok").returncode == 2
    assert _canary(tmp_path, "record", MOD, "not-a-sha", "PASS", "pool41", "gate", ev).returncode == 2
    assert _canary(tmp_path, "record", MOD, HEAD_SHA, "PASS", "pool41", "maybe", ev).returncode == 2
    assert _canary(tmp_path, "record", MOD, HEAD_SHA, "PASS", "pool41", ev).returncode == 2   # kind missing
    assert not (tmp_path / "pool_canaries.tsv").exists()


def test_has_and_key(tmp_path):
    assert _canary(tmp_path, "has", MOD, HEAD_SHA).returncode == 1
    _record(tmp_path, "PASS")
    assert _canary(tmp_path, "has", MOD, HEAD_SHA).returncode == 0
    queued = f"1790000000\t{_gate_line(sha=HEAD_SHA[:9])}  #checked:prereg {PREREG}"
    key = _canary(tmp_path, "key", queued)
    assert key.returncode == 0
    assert key.stdout.strip() == f"{MOD}\t{HEAD_SHA}"
    real = ("cd ~/derisk-pool/revisions/" + HEAD_SHA + " && bash tools/mem_ok.sh 7 1 && bash tools/memcap.sh 9 -- "
            "env SIM_BACKEND=numpy .venv/bin/python -u -m research.runners.host_share --shard --seed 42")
    assert _canary(tmp_path, "key", real).stdout.strip() == f"research.runners.host_share\t{HEAD_SHA}"


# ------------------------------------------------------------------------- `run` through a stub ssh + fake node

STUB_SSH = r"""#!/usr/bin/env bash
while [ $# -gt 0 ]; do
  case "$1" in
    -F|-o|-i|-p|-l) shift 2 ;;
    -*) shift ;;
    *) break ;;
  esac
done
shift   # the host
cd "$HOME" || exit 255
exec bash -c "$*"
"""

FAKE_RUNNER = r"""#!/usr/bin/env bash
out=""; mode=ok
while [ $# -gt 0 ]; do
  case "$1" in --out) out="$2"; shift ;; --mode) mode="$2"; shift ;; esac
  shift
done
case "$mode" in
  ok)     printf '{"turns": [{"answer": "x", "_error": null}], "error": null}' > "$out" ;;
  err)    printf '{"turns": [{"answer": "", "_error": "KeyError: boom"}]}' > "$out" ;;
  crash)  echo "Traceback (most recent call last):" >&2; exit 1 ;;
  memref) echo "[mem_ok] REFUSE: avail=1G need=7G" >&2; exit 1 ;;
  noout)  exit 0 ;;
  slow)   sleep 5;  printf '{}' > "$out" ;;
esac
"""


def _fake_node(tmp_path):
    home = tmp_path / "home"
    rev = home / "derisk-pool" / "revisions" / HEAD_SHA
    rev.mkdir(parents=True)
    (home / "derisk-pool" / "sim").mkdir(parents=True)
    (rev / ".provisioned_ok").write_text("")
    (rev / "fake_runner.sh").write_text(FAKE_RUNNER)
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "ssh").write_text(STUB_SSH)
    (stub / "ssh").chmod((stub / "ssh").stat().st_mode | stat.S_IEXEC)
    env = dict(HOME=str(home), PATH=f"{stub}:{os.environ['PATH']}", POOL_CANARY_POLL_S="0.2",
               POOL_CANARY_WAIT_S="30")
    return rev, env


def _run_line(mode, out="research/findings/raw/_fake/gate/s42/x.json", seed="42"):
    return (f"cd ~/derisk-pool/revisions/{HEAD_SHA} && mkdir -p research/findings/raw/_fake/gate/s42 && "
            f"bash fake_runner.sh -m {MOD} --seed {seed} --mode {mode} --out {out}")


def _records(tmp_path):
    p = tmp_path / "pool_canaries.tsv"
    if not p.exists():
        return []
    return [l.split("\t") for l in p.read_text().splitlines() if l and not l.startswith("#")]


def test_run_records_pass_and_keeps_output_out_of_the_gate_path(tmp_path):
    rev, env = _fake_node(tmp_path)
    res = _canary(tmp_path, "run", "pool41", _run_line("ok"), **env)
    assert res.returncode == 0, res.stdout + res.stderr
    recs = _records(tmp_path)
    assert len(recs) == 1
    assert recs[0][1:6] == [MOD, HEAD_SHA, "PASS", "pool41", "gate"]
    assert "rc=0" in recs[0][7] and "json=1" in recs[0][7]
    assert not (rev / "research/findings/raw/_fake/gate/s42/x.json").exists()
    assert list((rev / "research/findings/raw/_pool_canary").rglob("x.json"))
    # ...and the guard now lets the gate-seed line through
    assert _add(tmp_path, _gate_line()).returncode == 0


def test_run_fails_a_zero_exit_that_recorded_an_error(tmp_path):
    rev, env = _fake_node(tmp_path)
    res = _canary(tmp_path, "run", "pool41", _run_line("err"), **env)
    assert res.returncode == 1
    recs = _records(tmp_path)
    assert recs[-1][3] == "FAIL" and "errors=1" in recs[-1][7]
    assert "boom" in res.stdout
    assert _add(tmp_path, _gate_line()).returncode == 2


def test_run_fails_a_nonzero_exit(tmp_path):
    rev, env = _fake_node(tmp_path)
    res = _canary(tmp_path, "run", "pool41", _run_line("crash"), **env)
    assert res.returncode == 1
    assert _records(tmp_path)[-1][3] == "FAIL"
    assert "rc=1" in _records(tmp_path)[-1][7]


def test_run_fails_a_run_that_wrote_no_result(tmp_path):
    rev, env = _fake_node(tmp_path)
    res = _canary(tmp_path, "run", "pool41", _run_line("noout"), **env)
    assert res.returncode == 1
    assert _records(tmp_path)[-1][3] == "FAIL" and "json=0" in _records(tmp_path)[-1][7]


def test_a_mem_ok_refusal_is_inconclusive_and_not_recorded(tmp_path):
    rev, env = _fake_node(tmp_path)
    res = _canary(tmp_path, "run", "pool41", _run_line("memref"), **env)
    assert res.returncode == 3
    assert "INCONCLUSIVE" in res.stdout
    assert _records(tmp_path) == []


def test_a_timeout_is_inconclusive_and_not_recorded(tmp_path):
    rev, env = _fake_node(tmp_path)
    env["POOL_CANARY_WAIT_S"] = "1"
    res = _canary(tmp_path, "run", "pool41", _run_line("slow"), **env)
    assert res.returncode == 3
    assert _records(tmp_path) == []


def test_a_dev_seed_run_is_recorded_as_dev_and_does_not_open_gate_lines(tmp_path):
    rev, env = _fake_node(tmp_path)
    res = _canary(tmp_path, "run", "pool41", _run_line("ok", seed="7"), **env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert _records(tmp_path)[-1][3:6] == ["PASS", "pool41", "dev"]
    assert _add(tmp_path, _gate_line()).returncode == 2


def test_run_exact_writes_the_real_output_path(tmp_path):
    rev, env = _fake_node(tmp_path)
    res = _canary(tmp_path, "run", "pool41", _run_line("ok"), "--exact", **env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (rev / "research/findings/raw/_fake/gate/s42/x.json").exists()


def test_run_refuses_an_unprovisioned_revision(tmp_path):
    rev, env = _fake_node(tmp_path)
    (rev / ".provisioned_ok").unlink()
    res = _canary(tmp_path, "run", "pool41", _run_line("ok"), **env)
    assert res.returncode == 3
    assert _records(tmp_path) == []
