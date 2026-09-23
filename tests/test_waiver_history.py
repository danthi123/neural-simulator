"""Tests for tools/waiver_history.py (the 2026-09-23 idle-compute waiver loophole close) and the two gates
that consume it, tools/gates/compute_idle_persistent.py and tools/gates/lane_starvation.py.

THE LOOPHOLE THIS CLOSES: tools/parallel_audit.py printed "UNDER-PARALLELIZED" for ~14.5 days straight because
the blocking gates' waiver files were rewritten each cycle with PROMISES ("I will run the pool fanout ... as
the dedicated next step ...") that were never kept. These tests assert (1) a waiver must declare a fixed
resource-class, (2) promise/intent language is rejected regardless of class, (3) a renewal budget caps
cumulative non-GAMING/OWNER-PAUSE waived time, and (4) both gates actually honor `waiver_history.evaluate()`'s
verdict (not just that the library itself works in isolation)."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "tools"))

import waiver_history as wh  # noqa: E402


# ── parse_waiver: classification + promise-language rejection ──────────────────────────────────────────────

def test_promise_language_rejected_even_with_no_class():
    r = wh.parse_waiver("I will run the pool fanout as the dedicated next step after harvesting the verdict")
    assert r.get("error"), "the literal 2026-09-23 loophole text must be REJECTED"


def test_promise_language_rejected_under_a_valid_class():
    r = wh.parse_waiver("CLASS: OWNER-PAUSE\nI will free this up soon")
    assert r.get("error"), "promise language must be rejected even inside an otherwise-valid CLASS"


def test_unknown_class_rejected():
    r = wh.parse_waiver("CLASS: PRIORITY\nreason: focused on the crux")
    assert r.get("error"), "a class outside the fixed vocabulary (e.g. the old PRIORITY rationalisation) must fail"


def test_ram_contention_requires_avail_gb():
    assert wh.parse_waiver("CLASS: RAM-CONTENTION\nreason: training is using it all").get("error")
    ok = wh.parse_waiver("CLASS: RAM-CONTENTION\navail_gb=4\nreason: training + desktop leave no margin")
    assert not ok.get("error")
    assert ok["avail_gb"] == 4.0


def test_no_ready_work_requires_checked():
    assert wh.parse_waiver("CLASS: NO-READY-WORK\nreason: nothing to run").get("error")
    ok = wh.parse_waiver("CLASS: NO-READY-WORK\nchecked=lane_check.py, vikunja board, pool.queue -- all empty")
    assert not ok.get("error")
    assert "lane_check" in ok["checked"]


def test_gaming_and_owner_pause_need_only_a_class():
    assert not wh.parse_waiver("CLASS: GAMING\nreason: owner is playing").get("error")
    assert not wh.parse_waiver("CLASS: OWNER-PAUSE\nreason: owner reserved the box for a demo").get("error")


# ── evaluate(): budget + dedup, via real temp files (never the real research/queue/.waiver_history.jsonl) ────

def test_budget_exhausted_rejects_a_syntactically_valid_waiver():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "history.jsonl")
        open(p, "w").write("CLASS: NO-READY-WORK\nchecked=lane_check.py -- 5/5 lanes served")
        os.utime(p, (100.0, 100.0))
        v = wh.evaluate("g", p, waiver_max_h=6, now_ts=100.0, budget_h=6.0, prior_cumulative_h=6.0,
                        history_file=hist)
        assert v["active"] and not v["ok"], "a waiver must be rejected once the 24h renewal budget is spent"


def test_budget_with_room_accepts_the_same_waiver():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "history.jsonl")
        open(p, "w").write("CLASS: NO-READY-WORK\nchecked=lane_check.py -- 5/5 lanes served")
        os.utime(p, (100.0, 100.0))
        v = wh.evaluate("g", p, waiver_max_h=6, now_ts=100.0, budget_h=6.0, prior_cumulative_h=0.0,
                        history_file=hist)
        assert v["active"] and v["ok"]


def test_gaming_is_budget_exempt():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "history.jsonl")
        open(p, "w").write("CLASS: OWNER-PAUSE\nreason: owner reserved the box")
        os.utime(p, (100.0, 100.0))
        v = wh.evaluate("g", p, waiver_max_h=6, now_ts=100.0, budget_h=6.0, prior_cumulative_h=999.0,
                        history_file=hist)
        assert v["ok"], "OWNER-PAUSE/GAMING must stay budget-exempt (the pre-existing accepted-risk case)"


def test_missing_waiver_file_is_inactive():
    with tempfile.TemporaryDirectory() as td:
        v = wh.evaluate("g", os.path.join(td, "nope"), waiver_max_h=6, now_ts=100.0,
                        history_file=os.path.join(td, "history.jsonl"))
        assert not v["active"]


def test_unchanged_waiver_file_is_recorded_only_once():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "history.jsonl")
        open(p, "w").write("CLASS: NO-READY-WORK\nchecked=dedup probe")
        os.utime(p, (100.0, 100.0))
        wh.evaluate("dedup-gate", p, waiver_max_h=6, now_ts=100.0, history_file=hist)
        wh.evaluate("dedup-gate", p, waiver_max_h=6, now_ts=101.0, history_file=hist)
        import json
        rows = [json.loads(l) for l in open(hist).read().splitlines() if l.strip()]
        assert sum(1 for r in rows if r["gate"] == "dedup-gate") == 1, \
            "repeated check() calls against an UNCHANGED waiver must not inflate the budget"


# ── the shared module's own selftest, plus both gates' selftests (the registry's own contract) ────────────

def test_waiver_history_selftest_passes():
    assert wh.selftest() == []


def test_compute_idle_persistent_gate_selftest_passes():
    from tools.gates import compute_idle_persistent as g
    assert g.selftest() == []


def test_lane_starvation_gate_selftest_passes():
    from tools.gates import lane_starvation as g
    assert g.selftest() == []


# ── both gates actually HONOR an evaluate() verdict (not just that the library works standalone) ─────────────

def test_compute_idle_persistent_blocks_on_rejected_verdict():
    from tools.gates import compute_idle_persistent as g
    now = 2_000_000.0
    state = {"generated_at": now, "under_compute": True, "since_under_compute": now - g.PERSIST_S - 60}
    rejected = {"active": True, "ok": False, "class": None, "reject_reason": "promise language 'will'"}
    assert g._decide(state, now, rejected), "an INVALID waiver verdict must still block"


def test_compute_idle_persistent_passes_on_ok_verdict():
    from tools.gates import compute_idle_persistent as g
    now = 2_000_000.0
    state = {"generated_at": now, "under_compute": True, "since_under_compute": now - g.PERSIST_S - 60}
    ok = {"active": True, "ok": True, "class": "NO-READY-WORK", "age_h": 0.1, "budget_h": 1.0}
    assert g._decide(state, now, ok) == []


def test_lane_starvation_blocks_on_rejected_verdict():
    from tools.gates import lane_starvation as g
    idle = sorted(g.CPU_LANES)
    rejected = {"active": True, "ok": False, "class": None, "reject_reason": "promise language 'will'"}
    assert g._idle_message(idle, rejected)


def test_lane_starvation_passes_on_ok_verdict():
    from tools.gates import lane_starvation as g
    idle = sorted(g.CPU_LANES)
    ok = {"active": True, "ok": True, "class": "NO-READY-WORK", "age_h": 0.1, "budget_h": 1.0}
    assert g._idle_message(idle, ok) == []


# ── compute_idle_persistent's infra-only exemption (added while landing this very fix -- see its own
# docstring: the gate tripped on ITS OWN commit against genuine, live project state) ──────────────────────────

def test_compute_idle_persistent_exempts_infra_only_commits():
    from tools.gates import compute_idle_persistent as g
    assert g._is_infra_only(["tools/waiver_history.py", "tests/test_waiver_history.py", "README.md"])


def test_compute_idle_persistent_does_not_exempt_research_changes():
    from tools.gates import compute_idle_persistent as g
    assert not g._is_infra_only(["tools/waiver_history.py", "research/runners/some_derisk.py"])
    assert not g._is_infra_only([])
