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


def test_gaming_is_budget_exempt_WITH_evidence():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "history.jsonl")
        open(p, "w").write("CLASS: OWNER-PAUSE\nreason: owner reserved the box")
        os.utime(p, (100.0, 100.0))
        v = wh.evaluate("g", p, waiver_max_h=6, now_ts=100.0, budget_h=6.0, prior_cumulative_h=999.0,
                        history_file=hist, evidence_check=lambda: True)
        assert v["ok"] and v["budget_h"] is None, \
            "OWNER-PAUSE/GAMING must stay budget-exempt when owner-reservation evidence IS present"


# ── 2026-09-23 REVIEW FIX 2: GAMING/OWNER-PAUSE self-declared with NO evidence is NOT exempt ──────────────────

def test_gaming_without_evidence_is_NOT_exempt():
    """THE LOOPHOLE: the merged code granted GAMING/OWNER-PAUSE an unlimited-renewal budget exemption on the
    strength of the waiver's own CLASS line alone -- any agent could self-declare it. Fixed: `evaluate()` now
    requires `evidence_check()` (default: a GAME_MODE/GPU_PAUSE sentinel or a real game process) to return True
    before granting the exemption; otherwise the waiver is treated like any other non-exempt class."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "history.jsonl")
        open(p, "w").write("CLASS: OWNER-PAUSE\nreason: owner reserved the box")
        os.utime(p, (100.0, 100.0))
        v = wh.evaluate("g", p, waiver_max_h=6, now_ts=100.0, budget_h=6.0, prior_cumulative_h=999.0,
                        history_file=hist, evidence_check=lambda: False)
        assert not v["ok"], ("LOOPHOLE STILL OPEN: a self-declared OWNER-PAUSE with no owner-reservation "
                              "evidence was still granted the exemption despite an exhausted budget")


def test_gaming_without_evidence_still_accepted_with_budget_room():
    """Unevidenced is not the same as invalid -- with budget room it is accepted, just charged like any other
    class (budget_h is a real cumulative number, not None as the exempt path returns)."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "history.jsonl")
        open(p, "w").write("CLASS: GAMING\nreason: owner is playing")
        os.utime(p, (100.0, 100.0))
        v = wh.evaluate("g", p, waiver_max_h=6, now_ts=100.0, budget_h=6.0, prior_cumulative_h=0.0,
                        history_file=hist, evidence_check=lambda: False)
        assert v["ok"] and v["budget_h"] == 6.0


def test_default_evidence_check_recognizes_game_mode_sentinel():
    """The default `_owner_reservation_evidence` (used when a caller passes no `evidence_check`) must actually
    look for the sentinel `tools/game.sh on` / `tools/gpu_queue.sh pause` writes, not just accept any text."""
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "research", "queue"))
        assert not wh._owner_reservation_evidence(root=td, ps_text="")
        open(os.path.join(td, "research", "queue", "GAME_MODE"), "w").close()
        assert wh._owner_reservation_evidence(root=td, ps_text="")


def test_default_evidence_check_recognizes_a_running_game_process():
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "research", "queue"))
        assert wh._owner_reservation_evidence(root=td, ps_text="/usr/bin/steam -silent -no-cef-sandbox")
        assert not wh._owner_reservation_evidence(root=td, ps_text="python -m research.runners.foo")


# ── 2026-09-23 REVIEW FIX 1: off-by-one -- a fresh waiver must not count its own row against its own budget ──

def test_fresh_waiver_does_not_count_itself_against_its_own_budget():
    """THE BUG: `evaluate()` used to call `record()` (which appends this waiver's own row to the REAL history
    file) BEFORE calling `cumulative_waived_hours()`, so a brand-new waiver's own just-written row was already
    included in the sum it was being checked against. With an EMPTY history and `waiver_max_h == budget_h`,
    this made the very first waiver ever written reject itself."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "history.jsonl")
        assert not os.path.exists(hist), "history must start genuinely empty for this test to mean anything"
        open(p, "w").write("CLASS: NO-READY-WORK\nchecked=first-ever waiver, empty history")
        os.utime(p, (100.0, 100.0))
        # prior_cumulative_h OMITTED on purpose: must be computed LIVE from the (empty) history file, not
        # supplied by the test, so the bug (recording before summing) would actually reproduce here.
        v = wh.evaluate("g", p, waiver_max_h=6, now_ts=100.0, budget_h=6.0, history_file=hist)
        assert v["ok"], ("OFF-BY-ONE STILL PRESENT: the first-ever waiver counted its own just-recorded row "
                          "against its own budget and rejected itself (budget_h=%r)" % v.get("budget_h"))
        assert v["budget_h"] == 6.0


# ── 2026-09-23 REVIEW FIX (rationalisation vocabulary under EVERY class, not just free prose) ─────────────────

def test_rationalisation_vocabulary_rejected_under_a_valid_class_with_required_evidence():
    r = wh.parse_waiver("CLASS: NO-READY-WORK\nchecked=focused on the crux, nothing else worth it")
    assert r.get("error"), ("a CLASS: NO-READY-WORK waiver with a syntactically-valid checked= field must "
                             "still be rejected when it smuggles a priority/focus excuse")


def test_rationalisation_vocabulary_rejected_under_ram_contention():
    r = wh.parse_waiver("CLASS: RAM-CONTENTION\navail_gb=4\nreason: deprioritized behind the crux")
    assert r.get("error")


def test_clean_waiver_without_rationalisation_vocabulary_still_accepted():
    r = wh.parse_waiver("CLASS: NO-READY-WORK\nchecked=lane_check.py -- 5/5 lanes served, pool.queue empty")
    assert not r.get("error"), "the rationalisation check must not false-positive on a genuinely clean waiver"


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


# ── 2026-09-23 REVIEW FIX 3: _is_infra_only's .md exemption was UNCONDITIONAL -- narrowed to match its own
# docstring ("no research/runners, no research/findings, no sim/") ─────────────────────────────────────────────

def test_compute_idle_persistent_does_not_exempt_a_findings_md():
    from tools.gates import compute_idle_persistent as g
    assert not g._is_infra_only(["research/findings/2026-09-23-some-result.md"]), \
        "a research/findings/*.md-only staged set is a research artifact, not compute-neutral infra"


def test_compute_idle_persistent_does_not_exempt_the_board_or_roadmap():
    from tools.gates import compute_idle_persistent as g
    assert not g._is_infra_only(["GAP_CLOSURE_MISSION.md"])
    assert not g._is_infra_only(["ROADMAP.md"])


def test_compute_idle_persistent_still_exempts_generic_docs():
    from tools.gates import compute_idle_persistent as g
    assert g._is_infra_only(["tools/waiver_history.py", "docs/FAILURE_GATE_MATRIX.md", "CLAUDE.md"]), \
        "narrowing the .md exemption must not regress the exemption for genuinely generic documentation"


# ── 2026-09-23 REVIEW FIX 4: parallel_audit.py's waiver-surfacing loop must key on the GATES' OWN NAME, not an
# ad hoc print label -- a mismatch double-counts the shared renewal budget (see WAIVER_SOURCES's own comment) ──

def test_parallel_audit_waiver_gate_names_match_the_real_gates():
    import importlib.util
    from tools.gates import compute_idle_persistent, lane_starvation
    spec = importlib.util.spec_from_file_location("parallel_audit", os.path.join(ROOT, "tools",
                                                                                  "parallel_audit.py"))
    pa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pa)
    names = {gate_name for (_label, gate_name, _path) in pa.WAIVER_SOURCES}
    assert compute_idle_persistent.NAME in names, \
        "parallel_audit.py's waiver surfacing must key on compute_idle_persistent.NAME, not an ad hoc label"
    assert lane_starvation.NAME in names, \
        "parallel_audit.py's waiver surfacing must key on lane_starvation.NAME, not an ad hoc label"


def test_parallel_audit_waiver_surfacing_shares_the_gates_own_dedup_key():
    """Reproduces the actual double-count shape: evaluating the SAME live waiver file under the gate's own
    NAME (as the commit-time gate does) and under a mismatched ad hoc label (the pre-fix behavior) must NOT
    both be counted -- once fixed, both call sites use the identical gate_name and dedup as ONE row."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "history.jsonl")
        open(p, "w").write("CLASS: NO-READY-WORK\nchecked=lane_check.py -- 5/5 lanes served")
        os.utime(p, (100.0, 100.0))
        # simulate the FIXED heartbeat call (correct gate name) followed by the commit-time gate's own call
        # (same gate name) against the SAME unchanged file -- must dedup to exactly one history row.
        wh.evaluate("compute-idle-persistent", p, waiver_max_h=6, now_ts=100.0, history_file=hist)
        wh.evaluate("compute-idle-persistent", p, waiver_max_h=6, now_ts=101.0, history_file=hist)
        import json
        rows = [json.loads(l) for l in open(hist).read().splitlines() if l.strip()]
        assert sum(1 for r in rows if r["gate"] == "compute-idle-persistent") == 1
        # the PRE-FIX shape: a mismatched label creates a SECOND, separately-keyed row for the identical file
        # -- demonstrating why the mismatch double-counts (this asserts the mechanism, not that the fixed code
        # still does this).
        wh.evaluate("compute", p, waiver_max_h=6, now_ts=102.0, history_file=hist)
        rows2 = [json.loads(l) for l in open(hist).read().splitlines() if l.strip()]
        assert sum(1 for r in rows2 if r["gate"] == "compute") == 1
        assert sum(1 for r in rows2 if r["gate"] == "compute-idle-persistent") == 1
        # two rows total for ONE live waiver file is exactly the double-count the NAME mismatch caused.
        assert len(rows2) == 2
