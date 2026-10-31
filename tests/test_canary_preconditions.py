"""Tests for tools/canary_preconditions.py and its two call sites (2026-10-31).

WHY: the BG live-seam step-1 sub-step grid (research/findings/raw/_bg_live_seam/step1/507b9c789/, 64 cells, ~80
CPU-hours) was queued without `--commit`; its registered scorer read every set UNDEFINED (commit_engaged 0.0,
commit_op_ok 0.0, commit_seed_ok 0.0), yet the grid's pool canary PASSED, because tools/pool_canary.sh judged only rc,
the JSON count and error fields (research/FAILURE_LOG.md, 2026-10-31; RULING bg-t3-1031). Now (1) `pool_queue.sh add`
refuses any line, dev seeds included, that omits a flag its runner's registry entry requires, (2) `pool_canary.sh run`
refuses to launch such a line, and (3) a canary PASS also needs the registered scorer to read the canary's own output
with every declared precondition holding.

The `run` tests reuse the stub-ssh fake node of tests/test_pool_canary_guard.py with a test registry
(CANARY_PRECONDITIONS_REGISTRY) whose fake scorer prints a `pooled` line.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

import canary_preconditions as cp  # noqa: E402
import test_pool_canary_guard as g  # noqa: E402

TOOL = ROOT / "tools" / "canary_preconditions.py"
REV = "507b9c7890e680ae5d607f9dff4d49cad33d3944"
BG = ("cd ~/derisk-pool/revisions/%s && bash tools/mem_ok.sh 8 2 && bash tools/memcap.sh 12 -- env PYTHONHASHSEED=0 "
      "SIM_BACKEND=numpy BRAIN_BG_TONIC_DRIVE=1 .venv/bin/python -u -m research.runners.host_share_bg_seam "
      "--run-all research/findings/raw/_bg_live_seam/step1/507b9c789/stn100_gpe50 --seed 75" % REV)


def _tool(*args, **env):
    e = dict(os.environ)
    e.update(env)
    return subprocess.run([sys.executable, str(TOOL), *args], cwd=ROOT, env=e, capture_output=True, text=True,
                          timeout=60)


# --------------------------------------------------------------------------------------- the real BG registry entry

def test_the_incident_line_without_commit_is_refused_by_check_line():
    r = _tool("check-line", BG)
    assert r.returncode == 1 and "--commit" in r.stdout


def test_the_same_line_with_commit_passes_check_line_and_a_raw_queue_line_parses():
    assert _tool("check-line", BG + " --commit").returncode == 0
    assert _tool("check-line", "1793454287\t" + BG + " --commit  #checked:prereg x --commit-free").returncode == 0
    # the reason text is not the command: a --commit only in the #checked reason does not satisfy the rule
    assert _tool("check-line", "1793454287\t" + BG + "  #checked:prereg x --commit").returncode == 1


def test_other_bg_output_dirs_and_other_runners_are_not_constrained():
    assert _tool("check-line", BG.replace("_bg_live_seam/step1/", "_bg_live_seam/tonic_dev/")).returncode == 0
    assert _tool("check-line", BG.replace("host_share_bg_seam", "host_share")).returncode == 0


def test_resolve_reads_the_line_and_only_requires_commit_fields_with_commit():
    spec = cp.resolve(BG + " --commit")
    assert spec["argv"] == [".venv/bin/python", "-m", "research.runners.host_share_bg_seam", "--score",
                            "research/findings/raw/_bg_live_seam/step1/507b9c789/stn100_gpe50", "--seeds", "75",
                            "--commit"]
    paths = [r["path"] for r in spec["require"]]
    assert paths == ["pooled.membrane_ok", "pooled.commit_engaged", "pooled.commit_event_count_ok",
                     "pooled.commit_op_ok", "pooled.commit_seed_ok"]
    spec = cp.resolve(BG.replace("_bg_live_seam/step1/", "_bg_live_seam/tonic_dev/"))
    assert [r["path"] for r in spec["require"]] == ["pooled.membrane_ok"]
    assert "--commit" not in spec["argv"]
    assert cp.resolve(BG.replace("host_share_bg_seam", "host_share")) is None


def test_judge_fails_the_incident_reading_and_passes_a_clean_one():
    spec = cp.resolve(BG + " --commit")
    void = {"pooled": {"membrane_ok": 1.0, "commit_engaged": 0.0, "commit_event_count_ok": 1.0,
                       "commit_op_ok": 0.0, "commit_seed_ok": 0.0}}
    res = cp.judge(spec, "noise\n" + json.dumps(void) + "\n")
    assert not res["ok"] and "commit_engaged=0.0" in cp.summary(res)
    clean = {"pooled": dict.fromkeys(void["pooled"], 1.0)}
    assert cp.judge(spec, json.dumps(void) + "\n" + json.dumps(clean))["ok"]      # the LAST pooled line decides


def test_judge_fails_a_missing_field_a_none_and_no_line():
    spec = cp.resolve(BG + " --commit")
    assert not cp.judge(spec, json.dumps({"pooled": {"membrane_ok": 1.0}}))["ok"]
    assert not cp.judge(spec, json.dumps({"pooled": dict.fromkeys(
        ["membrane_ok", "commit_engaged", "commit_event_count_ok", "commit_op_ok", "commit_seed_ok"], None)}))["ok"]
    res = cp.judge(spec, "Traceback (most recent call last):\n")
    assert not res["ok"] and "no JSON line" in res["error"]


def test_node_run_fails_a_crashing_scorer_and_a_missing_one(tmp_path):
    spec = {"argv": [sys.executable, "-c", "import sys; print('{\"pooled\": {\"x\": 1.0}}'); sys.exit(3)"],
            "env": {}, "json_key": "pooled", "require": [{"path": "pooled.x", "op": "==", "value": 1.0, "label": "x"}]}
    res = cp.node_run(spec)
    assert not res["ok"] and "exited 3" in res["error"]
    spec["argv"] = [str(tmp_path / "no-such-scorer")]
    assert not cp.node_run(spec)["ok"]
    b64 = base64.b64encode(json.dumps(spec).encode()).decode()
    r = subprocess.run([sys.executable, str(TOOL), "node-run", b64], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and json.loads(r.stdout)["summary"].startswith("pre=FAIL")


# -------------------------------------------------------------------------- pool_queue.sh add: every line, dev too

def test_pool_queue_add_refuses_the_incident_line_on_a_dev_seed(tmp_path):
    res = g._add(tmp_path, BG, checked="prereg research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md"
                 " step1 rerun mem_gb=8 max_age_h=24")
    assert res.returncode == 2, res.stdout + res.stderr
    assert "REFUSED" in res.stderr and "--commit" in res.stderr
    assert g._queue_is_empty(tmp_path)


# ----------------------------------------------------- pool_canary.sh run: a fake runner + a registered fake scorer

FAKE_SCORER = r"""#!/usr/bin/env bash
case "$1" in
  good)  echo "scoring"; echo '{"pooled": {"engaged": 1.0, "membrane_ok": 1.0}}' ;;
  bad)   echo '{"pooled": {"engaged": 0.0, "membrane_ok": 1.0}}' ;;
  crash) echo "Traceback (most recent call last):" >&2; exit 1 ;;
esac
"""


def _registry(tmp_path):
    reg = {g.MOD: {
        "requires_flags": [{"path_regex": "_fake/gate/", "flag": "--mode", "why": "test rule"}],
        "checker": {"argv": ["bash", "fake_scorer.sh", "{arg:--pre}"], "json_key": "pooled", "require": [
            {"path": "pooled.membrane_ok", "op": "==", "value": 1.0},
            {"path": "pooled.engaged", "op": "==", "value": 1.0, "when": "flag:--pre"}]}}}
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(reg))
    return str(p)


def _node(tmp_path):
    rev, env = g._fake_node(tmp_path)
    (rev / "fake_scorer.sh").write_text(FAKE_SCORER)
    env["CANARY_PRECONDITIONS_REGISTRY"] = _registry(tmp_path)
    return rev, env


def test_run_passes_when_the_registered_preconditions_hold(tmp_path):
    rev, env = _node(tmp_path)
    res = g._canary(tmp_path, "run", "pool41", g._run_line("ok") + " --pre good", **env)
    assert res.returncode == 0, res.stdout + res.stderr
    rec = g._records(tmp_path)[-1]
    assert rec[3] == "PASS" and "pre=ok(membrane_ok=1.0 engaged=1.0)" in rec[7]


def test_run_fails_a_clean_exit_whose_scorer_reads_a_precondition_false(tmp_path):
    # the 2026-10-31 shape: rc 0, a JSON result, no error field -- and an instrument that did not engage
    rev, env = _node(tmp_path)
    res = g._canary(tmp_path, "run", "pool41", g._run_line("ok") + " --pre bad", **env)
    assert res.returncode == 1, res.stdout + res.stderr
    rec = g._records(tmp_path)[-1]
    assert rec[3] == "FAIL" and "rc=0" in rec[7] and "pre=FAIL(engaged=0.0 want == 1.0)" in rec[7]
    assert g._add(tmp_path, g._gate_line()).returncode == 2       # and the pair stays closed to gate-seed lines


def test_run_fails_when_the_scorer_crashes(tmp_path):
    rev, env = _node(tmp_path)
    res = g._canary(tmp_path, "run", "pool41", g._run_line("ok") + " --pre crash", **env)
    assert res.returncode == 1
    assert "pre=FAIL(" in g._records(tmp_path)[-1][7]


def test_run_fails_when_the_checker_cannot_resolve_its_arguments(tmp_path):
    # a line without the argument the scorer template needs (--pre is consulted by {arg:--pre}) is a FAIL, not a skip
    rev, env = _node(tmp_path)
    res = g._canary(tmp_path, "run", "pool41", g._run_line("ok"), **env)
    assert res.returncode == 1
    assert "pre=FAIL(" in g._records(tmp_path)[-1][7]


def test_run_refuses_to_launch_a_line_missing_a_required_flag(tmp_path):
    rev, env = _node(tmp_path)
    line = (f"cd ~/derisk-pool/revisions/{g.HEAD_SHA} && mkdir -p research/findings/raw/_fake/gate/s42 && "
            f"bash fake_runner.sh -m {g.MOD} --seed 42 --out research/findings/raw/_fake/gate/s42/x.json")
    res = g._canary(tmp_path, "run", "pool41", line, **env)
    assert res.returncode == 2 and "--mode" in res.stderr
    assert g._records(tmp_path) == []
    assert not list(rev.rglob("x.json"))


def test_a_runner_with_no_registered_checker_keeps_the_old_verdict(tmp_path):
    rev, env = g._fake_node(tmp_path)          # the real registry: no entry for the fake module
    res = g._canary(tmp_path, "run", "pool41", g._run_line("ok"), **env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "pre=none" in g._records(tmp_path)[-1][7]
