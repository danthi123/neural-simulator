"""Tests for tools/pool_queue.sh `add`'s pool_node= validation (2026-09-25 review, of
research/b2b-torn-cells-redo, LOW: "pool_queue.sh add does not check a pool_node=<name> against known nodes, so a
typo would leave a line stuck until it goes stale").

pop_job's node constraint (tools/pool_autodispatch.sh) `continue`s past any candidate whose declared pool_node=
does not match the current node -- silently, by design, so a MISTYPED node name never surfaces as an error; the
line simply matches no real node and sits in the queue until POOL_JOB_MAX_AGE retires it. This gate refuses an
unknown pool_node= at enqueue time instead, against the same candidate list pop_job/probe_nodes already iterate.

Uses commands with NO `-m research.runners.X` pattern (see test_pool_queue_first_word_gate.py's own note) so
`add`'s MOD-based local-import / remote-node checks never engage, keeping these tests fast and independent of any
real pool node.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL_QUEUE = ROOT / "tools" / "pool_queue.sh"


def _run(cmd, extra_env=None, tmp_path=None):
    env = dict(os.environ)
    if tmp_path is not None:
        env.setdefault("POOL_QUEUE_PATH", str(tmp_path / "pool.queue"))
        env.setdefault("POOL_SSH_CONFIG", str(tmp_path / "no-such-pool-ssh-config"))
        env.setdefault("POOL_EXTRA_NODES_FILE", str(tmp_path / "no-such-extra-nodes"))
    env.setdefault("POOL_NODES", "pool40 pool41 pool42")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(["bash", str(POOL_QUEUE), *cmd], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)


def test_add_refuses_an_unknown_pool_node(tmp_path):
    res = _run(
        ["add", "cd ~/derisk-pool/revisions/abc1234 && true  #pool_node=pool14", "--checked", "x"],
        tmp_path=tmp_path,
    )
    assert res.returncode == 2
    assert "REFUSED" in res.stderr
    assert "pool_node=pool14" in res.stderr
    queue = tmp_path / "pool.queue"
    assert not queue.exists() or queue.read_text() == "", "a line naming an unknown node must never reach the queue"


def test_add_accepts_a_known_default_pool_node(tmp_path):
    res = _run(
        ["add", "cd ~/derisk-pool/revisions/abc1234 && true  #pool_node=pool41", "--checked", "x"],
        tmp_path=tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert "queued" in res.stdout
    assert "pool_node=pool41" in (tmp_path / "pool.queue").read_text()


def test_add_accepts_a_pool_node_from_the_extra_nodes_file(tmp_path):
    extra = tmp_path / "extra-nodes"
    extra.write_text("pool1\npool2\n", encoding="utf-8")
    res = _run(
        ["add", "cd ~/derisk-pool/revisions/abc1234 && true  #pool_node=pool1", "--checked", "x"],
        extra_env={"POOL_EXTRA_NODES_FILE": str(extra)},
        tmp_path=tmp_path,
    )
    assert res.returncode == 0, res.stderr


def test_add_is_unaffected_for_a_line_with_no_pool_node_token(tmp_path):
    """Regression: opt-in behaviour must be preserved exactly, the same guarantee pop_job's own constraint gives."""
    res = _run(["add", "cd ~/derisk-pool/revisions/abc1234 && true", "--checked", "x"], tmp_path=tmp_path)
    assert res.returncode == 0, res.stderr
