"""Direct unit tests for tools/aws_stop_safety_lib.sh's CLI test seam (2026-09-25 review, fix round 3, LOW:
"sync_node_before_stop's own key/ip guard is unreachable through the main script -- have_ssh=1 already implies
both are set, so mutation-deleting the guard passed 28/28"). The seam (`--sync`/`--pause`/`--resume`) exercises
each helper directly, with no real AWS/ssh call needed, independent of either caller script's own resolution
logic -- and the guard is now genuinely reachable in production too, from tools/aws_budget.sh, whose own
node/ip/key resolution can legitimately come up empty (see tests/test_aws_budget_guard_workflow.py's own
`test_budget_enforce_stops_even_with_no_verified_key_for_the_instance`).
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "tools" / "aws_stop_safety_lib.sh"


def _run(args, extra_env=None, tmp_path=None):
    env = dict(os.environ)
    if tmp_path is not None:
        env.setdefault("AWS_SYNC_LOG", str(tmp_path / "sync.log"))
        env.setdefault("POOL_SSH_CONFIG", str(tmp_path / "no-such-pool-ssh-config"))
        env.setdefault("POOL_EXTRA_NODES_FILE", str(tmp_path / "no-such-extra-nodes"))
    if extra_env:
        env.update(extra_env)
    return subprocess.run(["bash", str(LIB), *args], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=15)


# --------------------------------------------------------------------------------------------- --sync (guard)

def test_sync_refuses_with_empty_ip(tmp_path):
    res = _run(["--sync", "node1", "", "/some/key.pem"], tmp_path=tmp_path)
    assert res.returncode == 1
    log = (tmp_path / "sync.log").read_text()
    assert "no verified ssh key/ip on hand" in log


def test_sync_refuses_with_empty_key(tmp_path):
    res = _run(["--sync", "node1", "1.2.3.4", ""], tmp_path=tmp_path)
    assert res.returncode == 1
    log = (tmp_path / "sync.log").read_text()
    assert "no verified ssh key/ip on hand" in log


def test_sync_refuses_when_the_recorded_key_file_does_not_exist(tmp_path):
    # MUTATION CHECK (this is the exact case the review's own repro used): a recorded key PATH that does not
    # exist on disk must never be treated as "have ssh access", independent of whatever resolved it.
    missing_key = tmp_path / "does-not-exist.pem"
    res = _run(["--sync", "node1", "1.2.3.4", str(missing_key)], tmp_path=tmp_path)
    assert res.returncode == 1
    log = (tmp_path / "sync.log").read_text()
    assert "no verified ssh key/ip on hand" in log


def test_sync_usage_error_on_wrong_arg_count(tmp_path):
    res = _run(["--sync", "node1", "1.2.3.4"], tmp_path=tmp_path)
    assert res.returncode == 2
    assert "usage" in res.stderr


# ------------------------------------------------------------------------------------- --pause / --resume

def test_pause_is_a_noop_and_prints_0_when_the_node_is_not_registered(tmp_path):
    extra = tmp_path / "extra_nodes"
    extra.write_text("other-node\n")
    res = _run(["--pause", "gpu"], extra_env={"POOL_EXTRA_NODES_FILE": str(extra)}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "0"
    assert extra.read_text() == "other-node\n"


def test_pause_removes_a_registered_node_and_prints_1(tmp_path):
    extra = tmp_path / "extra_nodes"
    extra.write_text("gpu\nother-node\n")
    res = _run(["--pause", "gpu"], extra_env={"POOL_EXTRA_NODES_FILE": str(extra)}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "1"
    assert extra.read_text().splitlines() == ["other-node"]


def test_pause_removes_the_sole_registered_node_leaving_an_empty_file(tmp_path):
    # REGRESSION (caught during this fix round's own verification): `grep -v` exits 1 -- "no lines selected" --
    # when the removed node was the ONLY line, which is a successful empty result, not an error. A naive
    # `grep -v ... && mv ...` chain wrongly read that as failure and left the node registered.
    extra = tmp_path / "extra_nodes"
    extra.write_text("gpu\n")
    res = _run(["--pause", "gpu"], extra_env={"POOL_EXTRA_NODES_FILE": str(extra)}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "1"
    assert extra.read_text() == ""


def test_resume_restores_a_node_only_when_was_registered_is_1(tmp_path):
    extra = tmp_path / "extra_nodes"
    extra.write_text("other-node\n")
    res = _run(["--resume", "gpu", "1"], extra_env={"POOL_EXTRA_NODES_FILE": str(extra)}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert set(extra.read_text().split()) == {"gpu", "other-node"}


def test_resume_is_a_noop_when_was_registered_is_0(tmp_path):
    extra = tmp_path / "extra_nodes"
    extra.write_text("other-node\n")
    res = _run(["--resume", "gpu", "0"], extra_env={"POOL_EXTRA_NODES_FILE": str(extra)}, tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
    assert extra.read_text() == "other-node\n"


def test_pause_then_resume_round_trips_to_the_original_content(tmp_path):
    extra = tmp_path / "extra_nodes"
    extra.write_text("gpu\n")
    env = {"POOL_EXTRA_NODES_FILE": str(extra)}
    r1 = _run(["--pause", "gpu"], extra_env=env, tmp_path=tmp_path)
    assert r1.stdout.strip() == "1"
    assert extra.read_text() == ""
    r2 = _run(["--resume", "gpu", r1.stdout.strip()], extra_env=env, tmp_path=tmp_path)
    assert r2.returncode == 0, r2.stderr
    assert extra.read_text().splitlines() == ["gpu"]
