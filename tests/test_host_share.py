"""Host-share instrument (research/runners/host_share.py + host_share_schema.py): aggregation logic and the
preregistered structure, with no brain build. The live arms are exercised by the dev run and the six-seed gate."""
import os
import subprocess
import sys

from research.runners import host_share as HS
from research.runners import host_share_schema as SCHEMA


def test_runner_selftest_passes():
    ok, checks = HS.selftest(verbose=False)
    assert ok, {k: v for k, v in checks.items() if not v}


def test_schema_selftest_passes():
    ok, checks = SCHEMA.selftest()
    assert ok, {k: v for k, v in checks.items() if not v}


def test_host_causal_event_is_never_neural():
    # the audit's HIGH issue: F=1 with S=0 (host flip reaches the reply, neural rule agrees) must not count as neural
    assert SCHEMA.event_bucket(1, 0) == "host_causal_replaceable"
    p = SCHEMA.panel([dict(seam="k-select", live=True, F=1, S=0, seed=s) for s in (1, 2, 3, 4)], ["k-select"])
    assert p["neural_decision_fraction"] == 0.0 and p["host_decision_share"] == 1.0


def test_every_group_has_sham_and_each_seams_modes():
    for group, g in HS.GROUPS.items():
        arms = HS.arms_for(group)
        assert arms[:2] == ["intact_a", "intact_b"]
        for seam in g["seams"]:
            for mode in HS.SEAM_SPEC[seam]["modes"]:
                assert "%s:%s" % (seam, mode) in arms


def test_every_group_has_target_turns_with_cues():
    for group in HS.GROUPS:
        specs = HS.turn_specs(group)
        targets = [sp for sp in specs.values() if sp["target"]]
        assert targets and all(sp["cue"] for sp in targets)


def test_all_six_seams_are_live_somewhere_and_measured_split_is_frozen():
    assert {s for g in HS.GROUPS.values() for s in g["seams"]} == set(HS.SEAMS)
    assert set(HS.MEASURED_SEAMS) == {"k-select", "v-choice", "bg-select"}


def test_ltm_tier_is_pinned_and_expected_source_declared():
    assert HS.BASE_ENV.get("BRAIN_LTM_SHIP_DEFAULT") == "1" and HS.EXPECTED_SOURCE == "tiny-demo +LTM"


def test_gate_seed_is_refused_without_the_flag():
    p = subprocess.run([sys.executable, "-m", "research.runners.host_share", "--seed-all", "--seed", "42",
                        "--out-dir", "/nonexistent"], capture_output=True, text=True)
    assert p.returncode != 0 and "GATE seed" in p.stderr


def test_pool_job_lines_are_pinned_and_sized():
    lines = HS.pool_job_lines("0123456789abcdef")
    assert len(lines) == len(HS.GATE_SEEDS) * sum(len(HS.arms_for(g)) for g in HS.GROUPS)
    for cmd, why in lines:
        assert cmd.startswith("cd ~/derisk-pool/revisions/0123456789abcdef && ")
        assert "--allow-gate-seed" in cmd and "mem_gb=7" in why


def test_classifier_boundaries():
    st = lambda **kw: dict(dict(live_seeds=[1, 2, 3, 4], HS=None, NS=None, MS=None, SC=None), **kw)  # noqa: E731
    assert SCHEMA.classify_seam(st(HS=0.8, SC=0.2)) == "HOST-OWNED"
    assert SCHEMA.classify_seam(st(HS=0.8, SC=0.1)) == "HOST-OWNED (replaceable)"
    assert SCHEMA.classify_seam(st(HS=0.1, NS=0.9)) == "NEURAL-OWNED"
    assert SCHEMA.classify_seam(st(HS=0.1, NS=0.1, MS=0.9)) == "MASKED"
    assert SCHEMA.classify_seam(st(HS=0.5, NS=0.3, MS=0.2)) == "MIXED"
    assert SCHEMA.classify_seam(dict(st(HS=1.0), live_seeds=[1, 2, 3])) == "UNDEFINED"


# The worker, stubbed: the REAL host_share CLI (its argparse and gate-seed checks) in a REAL child process; only
# run_worker is replaced, so no brain is built. It writes the part file run_shard merges.
_STUB_WORKER = r'''
import json, os, sys
from research.runners import host_share as HS
def run_worker(seed, group, arm, out_path, backend="numpy", session_index=None, only_labels=None):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump({"instrument": "host_share", "seed": seed, "group": group, "arm": arm, "session_index": session_index,
               "pythonhashseed_ok": True, "responses": {"t%s" % session_index: {"answer": "stub"}}, "events": {},
               "snapshots": {}, "turn_seconds": {}, "wall_seconds": 0.0}, open(out_path, "w"))
    return 0
HS.run_worker = run_worker
sys.exit(HS.main())
'''


def _stub_spawns(monkeypatch):
    """Route run_shard's worker spawns through _STUB_WORKER; returns the list of spawned argv (as run_shard built them)."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(HS.__file__))))
    real_run = subprocess.run
    spawned = []

    def fake_run(cmd, env=None, **kw):
        assert cmd[1:4] == ["-u", "-m", "research.runners.host_share"], cmd
        spawned.append(list(cmd))
        p = real_run([cmd[0], "-c", _STUB_WORKER] + list(cmd[4:]), env=env, cwd=repo, capture_output=True, text=True)
        spawned[-1].append("#rc=%d stderr=%s" % (p.returncode, p.stderr.strip()[-300:]))
        return p

    monkeypatch.setattr(HS.subprocess, "run", fake_run)
    return spawned


def test_gate_seed_shard_workers_get_the_gate_flag(tmp_path, monkeypatch):
    # POOL DEFECT 2026-09-30 (pre-data): at 59edf3255 run_shard spawned `--worker --seed 42` WITHOUT --allow-gate-seed,
    # so every preregistered gate shard died in the worker's argparse on the pool (rc=1, no data). The pool line is
    # `--shard --seed 42 --group G --arm A --allow-gate-seed --out ...`: run exactly that path.
    spawned = _stub_spawns(monkeypatch)
    out = tmp_path / "s42" / "recall__intact_a.json"
    monkeypatch.setattr(sys, "argv", ["host_share", "--shard", "--seed", "42", "--group", "recall", "--arm",
                                      "intact_a", "--allow-gate-seed", "--out", str(out)])
    rc = HS.main()
    assert spawned, "run_shard spawned no worker"
    assert rc == 0 and out.exists(), [c[-1] for c in spawned]
    assert all("--allow-gate-seed" in cmd for cmd in spawned), spawned


def test_gate_seed_seed_all_workers_get_the_gate_flag(tmp_path, monkeypatch):
    # the same defect on the --seed-all path (it also goes through run_shard)
    spawned = _stub_spawns(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["host_share", "--seed-all", "--seed", "43", "--group", "compete",
                                      "--allow-gate-seed", "--out-dir", str(tmp_path)])
    rc = HS.main()
    arms = HS.arms_for("compete")
    assert rc == 0 and len(spawned) == len(arms), [c[-1] for c in spawned]
    assert all(os.path.exists(HS.shard_path(str(tmp_path), 43, "compete", a)) for a in arms)
    assert all("--allow-gate-seed" in cmd for cmd in spawned), spawned


def test_dev_seed_shard_workers_do_not_need_the_gate_flag(tmp_path, monkeypatch):
    spawned = _stub_spawns(monkeypatch)
    out = tmp_path / "s7" / "recall__intact_a.json"
    monkeypatch.setattr(sys, "argv", ["host_share", "--shard", "--seed", "7", "--group", "recall", "--arm",
                                      "intact_a", "--out", str(out)])
    assert HS.main() == 0 and out.exists(), [c[-1] for c in spawned]
    assert not any("--allow-gate-seed" in cmd for cmd in spawned), spawned
