"""Pins that tools/parallel_audit.py surfaces tools/pool_stall_check.py's QUEUED-line UNRUNNABLE / memory-budget
check every cycle (2026-09-25 addition): six revision-pinned research/queue/pool.queue lines sat 7.5h because
the pinned revision was never provisioned where it could fit, and pool_autodispatch.sh's own per-cycle log line
("revision ... not provisioned on pool2") was the ONLY place this was ever said -- not a heartbeat line, not a
report. Unlike the DUP-OF-LANDED/OVERDUE check (tests/test_parallel_audit_pool_stall_exclusion.py), a queued-not-
yet-dispatched line was never counted in lanes_pool to begin with, so there is nothing to subtract here -- this
is pure surfacing, printed the same way stall_lines already is.

Two layers, so a regression is caught whichever half breaks:
  (1) queue_unrunnable_summary() forwards check_queue()'s unrunnable/memory_budget_stalled rows correctly and
      never raises past a failure (the heartbeat is exit-0-always). It returns (count, lines) -- see the
      2026-09-25 fix-round note below.
  (2) main() actually PRINTS those lines -- pinned by running the real main() with every external dependency
      stubbed (no real ssh/nvidia-smi/vikunja/filesystem-state-write), so a future edit that stops calling
      queue_unrunnable_summary() fails this test even though layer (1) alone would still pass.

2026-09-25 (fix round, opus review). Two changes here, matching parallel_audit.py's own docstring:
  (a) queue_unrunnable_summary() now returns (count, lines) instead of a bare list (LOW-4 fix: the count rides
      on the SATURATED/UNDER-PARALLELIZED verdict line itself, since the live heartbeat greps only lines
      matching that pattern and a queue warning on its own line was silently dropped).
  (b) main() calls it with NO explicit node list (HIGH-1 fix: the old `queue_unrunnable_summary(POOL)` call
      hardcoded this module's mini-PC-only node list, so a revision missing only on an AWS
      `.pool_extra_nodes` node -- the exact 2026-09-25 incident -- was invisible to the check).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import parallel_audit as pa  # noqa: E402


# ----------------------------------------------------------------------------------- layer 1: queue_unrunnable_summary

def test_queue_unrunnable_summary_forwards_unrunnable_and_membudget_rows(monkeypatch):
    canned = {
        "unrunnable": [{
            "age_s": 27000, "module": "settle_a2", "pinned_sha": "a" * 40, "mem_gb": 8,
            "capable_nodes": ["pool1", "pool2"], "node_status": {"pool1": False, "pool2": False},
            "fix_cmd": "bash tools/pool_provision.sh --revision %s --isolated pool1 pool2" % ("a" * 40),
        }],
        "memory_budget_stalled": [{
            "age_s": 7200, "module": "huge_job", "mem_gb": 40, "max_known_ceiling_gb": 13,
        }],
        "summary_line": "POOL QUEUE CHECK: 1 UNRUNNABLE, 1 over every known node's memory ceiling (of 2 queued line(s), 0 node(s) mem-unreachable)",
    }
    assert pa.pool_stall_check is not None, "pool_stall_check must import cleanly for this gate to mean anything"
    monkeypatch.setattr(pa.pool_stall_check, "check_queue", lambda **kw: canned)
    n, lines = pa.queue_unrunnable_summary(["pool1", "pool2"])
    assert n == 2
    assert any("UNRUNNABLE" in l for l in lines)
    assert any("settle_a2" in l for l in lines)
    assert any("huge_job" in l for l in lines)
    assert any("fix: bash tools/pool_provision.sh" in l for l in lines)


def test_queue_unrunnable_summary_never_raises_when_check_queue_blows_up(monkeypatch):
    def boom(**kw):
        raise RuntimeError("ssh exploded")
    monkeypatch.setattr(pa.pool_stall_check, "check_queue", boom)
    n, lines = pa.queue_unrunnable_summary(["pool1"])   # must not raise -- the heartbeat is exit-0-always
    assert n == 0
    assert any("failed to run" in l for l in lines)


def test_queue_unrunnable_summary_disabled_cleanly_when_module_absent(monkeypatch):
    monkeypatch.setattr(pa, "pool_stall_check", None)
    assert pa.queue_unrunnable_summary(["pool1"]) == (0, [])


def test_queue_unrunnable_summary_quiet_when_nothing_flagged(monkeypatch):
    monkeypatch.setattr(pa.pool_stall_check, "check_queue", lambda **kw: {
        "unrunnable": [], "memory_budget_stalled": [],
        "summary_line": "POOL QUEUE CHECK: clean (of 3 queued line(s), 0 node(s) mem-unreachable)",
    })
    assert pa.queue_unrunnable_summary(["pool1"]) == (0, [])


def test_queue_unrunnable_summary_default_nodes_is_none_so_get_pool_nodes_governs(monkeypatch):
    # HIGH-1 fix: calling with NO node list at all must forward nodes=None to check_queue() (which then falls
    # back to pool_stall_check.get_pool_nodes(), the function that actually reads .pool_extra_nodes) -- not this
    # module's own hardcoded default.
    captured = {}

    def fake_check_queue(nodes=None, timeout=12):
        captured["nodes"] = nodes
        return {"unrunnable": [], "memory_budget_stalled": [], "summary_line": "POOL QUEUE CHECK: clean"}
    monkeypatch.setattr(pa.pool_stall_check, "check_queue", fake_check_queue)
    pa.queue_unrunnable_summary()
    assert captured["nodes"] is None


# ------------------------------------------------------------------------------------------- layer 2: main() wiring

def _stub_main_dependencies(monkeypatch, *, queue_lines=(), n_queue_stuck=None):
    """Same isolation contract as test_parallel_audit_pool_stall_exclusion.py's helper: no real ssh, no real
    write to the SHARED research/coordination state file. `nodes=None` default on the stub (fix round, 2026-09-25
    review HIGH-1) matches how main() actually calls this now -- with no explicit node list."""
    n = len(queue_lines) if n_queue_stuck is None else n_queue_stuck
    monkeypatch.setattr(pa, "local_idle", lambda: (20, 2.0, 0, 0))
    monkeypatch.setattr(pa, "gpu_state", lambda: -1)
    monkeypatch.setattr(pa, "pool_idle", lambda: (0, 5, 3))
    monkeypatch.setattr(pa, "pool_stall_summary", lambda nodes, timeout=12: (0, []))
    monkeypatch.setattr(pa, "queue_unrunnable_summary", lambda nodes=None, timeout=12: (n, list(queue_lines)))
    monkeypatch.setattr(pa, "open_tasks", lambda: (1, [(1, "some task")]))
    monkeypatch.setattr(pa, "active_agents", lambda: 5)
    monkeypatch.setattr(pa, "gpu_queue_busy", lambda: False)
    monkeypatch.setattr(pa.parallel_state, "persist", lambda *a, **k: {})
    monkeypatch.setattr(pa, "WAIVER_SOURCES", ())
    monkeypatch.setattr(pa.os.path, "exists", lambda p: False)   # no GAME_MODE file


def test_main_prints_queue_unrunnable_lines(monkeypatch, capsys):
    _stub_main_dependencies(monkeypatch, queue_lines=["⚠ fake queue-stuck line"])
    rc = pa.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "⚠ fake queue-stuck line" in out


def test_main_silent_on_queue_lines_when_nothing_flagged(monkeypatch, capsys):
    _stub_main_dependencies(monkeypatch, queue_lines=[])
    rc = pa.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "queue-stuck" not in out


def test_main_never_subtracts_queue_lines_from_lane_count(monkeypatch, capsys):
    # A queued (not-yet-dispatched) line was never counted in lanes_pool to begin with -- flagging it must not
    # change the printed lane total, unlike the running-job DUP-OF-LANDED/OVERDUE subtraction.
    _stub_main_dependencies(monkeypatch, queue_lines=["⚠ fake queue-stuck line"])
    rc = pa.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "lanes=10 (local 0 + pool 5 + agents 5)" in out
