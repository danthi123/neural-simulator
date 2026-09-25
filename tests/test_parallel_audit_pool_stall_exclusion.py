"""Pins that tools/parallel_audit.py's SATURATED/UNDER-PARALLELIZED decision EXCLUDES pool lanes that
tools/pool_stall_check.py flags DUP-OF-LANDED or OVERDUE (2026-09-25: the heartbeat read SATURATED while 7 of
17 D6 processes were duplicates re-deriving already-landed results, running 7-26h).

Two layers, so a regression is caught whichever half breaks:
  (1) pool_stall_summary() forwards check_all()'s flagged count/lines correctly and never raises past a failure
      (the heartbeat is exit-0-always).
  (2) main() actually SUBTRACTS that count from lanes_pool before printing/deciding -- pinned by running the
      real main() with every external dependency stubbed (no real ssh/nvidia-smi/vikunja/filesystem-state-write),
      so a future edit that stops calling pool_stall_summary(), or drops the subtraction, fails this test even
      though layer (1) alone would still pass.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import parallel_audit as pa  # noqa: E402


# --------------------------------------------------------------------------------------- layer 1: pool_stall_summary

def test_pool_stall_summary_forwards_flagged_rows_and_summary_line(monkeypatch):
    canned = {
        "flagged": [{
            "node": "pool40", "job_id": "j1", "pids": ["111"], "elapsed_s": 3600, "module": "d6_capacity_curve",
            "out_path": "research/findings/raw/x.json", "pinned_sha": "a" * 40, "dup_of_landed": True,
            "overdue": "UNKNOWN", "kill_cmd": 'ssh -n pool40 "kill -TERM 111"',
        }],
        "summary_line": "POOL STALL CHECK: 1 DUP-OF-LANDED (of 3 running across 3 node(s), 1 unknown-history, 0 unreachable)",
    }
    assert pa.pool_stall_check is not None, "pool_stall_check must import cleanly for this gate to mean anything"
    monkeypatch.setattr(pa.pool_stall_check, "check_all", lambda **kw: canned)
    n, lines = pa.pool_stall_summary(["pool40"])
    assert n == 1
    assert any("DUP-OF-LANDED" in l for l in lines)
    assert any("kill -TERM 111" in l for l in lines)


def test_pool_stall_summary_never_raises_when_check_all_blows_up(monkeypatch):
    def boom(**kw):
        raise RuntimeError("ssh exploded")
    monkeypatch.setattr(pa.pool_stall_check, "check_all", boom)
    n, lines = pa.pool_stall_summary(["pool40"])   # must not raise -- the heartbeat is exit-0-always
    assert n == 0
    assert any("failed to run" in l for l in lines)


def test_pool_stall_summary_disabled_cleanly_when_module_absent(monkeypatch):
    monkeypatch.setattr(pa, "pool_stall_check", None)
    n, lines = pa.pool_stall_summary(["pool40"])
    assert (n, lines) == (0, [])


def test_pool_stall_summary_quiet_when_nothing_flagged(monkeypatch):
    monkeypatch.setattr(pa.pool_stall_check, "check_all",
                         lambda **kw: {"flagged": [], "summary_line": "POOL STALL CHECK: clean (of 3 running ...)"})
    n, lines = pa.pool_stall_summary(["pool40"])
    assert (n, lines) == (0, [])


# --------------------------------------------------------------------------------------- layer 2: main() wiring

def _stub_main_dependencies(monkeypatch, *, lanes_pool, flagged_pool, agents=5):
    """Stub every external dependency main() touches EXCEPT the pool-stall wiring under test, so main() can run
    for real without a network call, a real ssh, or a write to the SHARED research/coordination state file
    (parallel_state.persist resolves through git-common-dir -- writing it for real from a worktree test would
    touch the primary checkout's live heartbeat state, exactly the 'do not touch live queue/running files'
    rule this task is scoped under)."""
    monkeypatch.setattr(pa, "local_idle", lambda: (20, 2.0, 0, 0))
    monkeypatch.setattr(pa, "gpu_state", lambda: -1)
    monkeypatch.setattr(pa, "pool_idle", lambda: (0, lanes_pool, 3))
    monkeypatch.setattr(pa, "pool_stall_summary",
                         lambda nodes, timeout=12: (flagged_pool, (["⚠ fake stall line"] if flagged_pool else [])))
    # queue_unrunnable_summary (2026-09-25 addition) makes its own ssh calls via pool_stall_check.check_queue --
    # stub it too, same reason as pool_stall_summary above (no real ssh from a test). `nodes=None` default
    # (fix round, 2026-09-25 review HIGH-1): main() must call this with NO explicit node list -- see
    # test_main_queue_check_called_with_no_explicit_node_list below, which pins that wiring directly.
    monkeypatch.setattr(pa, "queue_unrunnable_summary", lambda nodes=None, timeout=12: (0, []))
    monkeypatch.setattr(pa, "open_tasks", lambda: (1, [(1, "some task")]))
    monkeypatch.setattr(pa, "active_agents", lambda: agents)
    monkeypatch.setattr(pa, "gpu_queue_busy", lambda: False)
    monkeypatch.setattr(pa.parallel_state, "persist", lambda *a, **k: {})
    monkeypatch.setattr(pa, "WAIVER_SOURCES", ())
    monkeypatch.setattr(pa.os.path, "exists", lambda p: False)   # no GAME_MODE file


def test_main_subtracts_flagged_pool_lanes_before_printing_and_deciding(monkeypatch, capsys):
    _stub_main_dependencies(monkeypatch, lanes_pool=5, flagged_pool=2, agents=5)
    rc = pa.main()
    out = capsys.readouterr().out
    assert rc == 0
    # raw pool lanes=5, 2 flagged -> effective pool=3; local=0 + pool=3 + agents=5 = 8
    assert "lanes=8 (local 0 + pool 3 + agents 5)" in out
    assert "⚠ fake stall line" in out


def test_main_never_subtracts_below_zero(monkeypatch, capsys):
    _stub_main_dependencies(monkeypatch, lanes_pool=1, flagged_pool=5, agents=5)
    rc = pa.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "lanes=5 (local 0 + pool 0 + agents 5)" in out   # floored at 0, never negative


def test_main_unaffected_when_nothing_flagged(monkeypatch, capsys):
    _stub_main_dependencies(monkeypatch, lanes_pool=5, flagged_pool=0, agents=5)
    rc = pa.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "lanes=10 (local 0 + pool 5 + agents 5)" in out
    assert "⚠ fake stall line" not in out


def test_main_saturated_message_excludes_flagged_lanes_from_its_own_evidence(monkeypatch, capsys):
    # agents (5) already clears the floor and no dedicated lane reads idle, so `under` is False on its own
    # signals (agent count / idle_pool -- unrelated to lanes_pool) regardless of the pool-stall wiring. What
    # this pins is that the SATURATED line's OWN justification ("N compute lanes cover the frontier") -- the
    # exact sentence that read misleadingly true during the 2026-09-25 incident -- excludes flagged lanes.
    monkeypatch.setenv("PARALLEL_AGENT_FLOOR", "3")
    _stub_main_dependencies(monkeypatch, lanes_pool=5, flagged_pool=2, agents=5)
    rc = pa.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "SATURATED (5 agents + 3 compute lanes cover the frontier)" in out


# --------------------------------------------------------------------- fix round (2026-09-25 opus review)

def test_main_calls_queue_unrunnable_summary_with_no_explicit_node_list(monkeypatch, capsys):
    # HIGH-1 fix: main() used to call queue_unrunnable_summary(POOL), hardcoding this module's own mini-PC-only
    # node list -- pool_stall_check.check_queue() only falls back to get_pool_nodes() (which ALSO reads
    # research/queue/.pool_extra_nodes, the AWS lane) when its own `nodes` argument is None. A revision missing
    # only on an extra node (the exact 2026-09-25 incident) was invisible to the check under the old wiring.
    # This is the direct wiring test the review asked for (mutation M8 on the nodes argument).
    _stub_main_dependencies(monkeypatch, lanes_pool=5, flagged_pool=0, agents=5)
    captured = {}

    def spy(nodes=None, timeout=12):
        captured["nodes"] = nodes
        return (0, [])
    monkeypatch.setattr(pa, "queue_unrunnable_summary", spy)
    rc = pa.main()
    assert rc == 0
    assert "nodes" in captured
    assert captured["nodes"] is None


def test_main_appends_queue_stuck_count_to_the_verdict_line(monkeypatch, capsys):
    # LOW-4 fix: the live heartbeat greps only lines matching SATURATED|UNDER-PARALLELIZED, so a queue-stuck
    # warning printed on ITS OWN line is silently dropped. The count must ride on the verdict line itself.
    _stub_main_dependencies(monkeypatch, lanes_pool=5, flagged_pool=0, agents=5)
    monkeypatch.setattr(pa, "queue_unrunnable_summary", lambda nodes=None, timeout=12: (3, ["⚠ fake queue line"]))
    rc = pa.main()
    out = capsys.readouterr().out
    assert rc == 0
    saturated_lines = [l for l in out.splitlines() if l.startswith("✓ SATURATED")]
    assert len(saturated_lines) == 1
    assert "queue: 3 UNRUNNABLE" in saturated_lines[0]


def test_main_under_parallelized_line_also_carries_queue_suffix(monkeypatch, capsys):
    # Companion: the suffix must appear on the ⛔ UNDER-PARALLELIZED verdict line too, not just the SATURATED one
    # -- whichever verdict fires is the one the heartbeat's grep will keep.
    _stub_main_dependencies(monkeypatch, lanes_pool=5, flagged_pool=0, agents=0)   # agents=0 -> under floor
    monkeypatch.setattr(pa, "queue_unrunnable_summary", lambda nodes=None, timeout=12: (2, ["⚠ fake queue line"]))
    rc = pa.main()
    out = capsys.readouterr().out
    assert rc == 0
    stall_lines = [l for l in out.splitlines() if l.startswith("⛔ UNDER-PARALLELIZED")]
    assert len(stall_lines) == 1
    assert "queue: 2 UNRUNNABLE" in stall_lines[0]


def test_main_verdict_line_has_no_queue_suffix_when_nothing_stuck(monkeypatch, capsys):
    # No false-positive suffix when the queue is clean.
    _stub_main_dependencies(monkeypatch, lanes_pool=5, flagged_pool=0, agents=5)
    rc = pa.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "queue:" not in out
