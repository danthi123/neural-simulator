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
        # fix round 3: budget_h is the CHARGED live time (0h at the instant of writing), not a declared cap --
        # what matters here is that it is a number (charged), not None (exempt).
        assert v["ok"] and v["budget_h"] == 0.0


def test_default_evidence_check_recognizes_game_mode_sentinel():
    """The default `_owner_reservation_evidence` (used when a caller passes no `evidence_check`) must actually
    look for the sentinel `tools/game.sh on` / `tools/gpu_queue.sh pause` writes, not just accept any text."""
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "research", "queue"))
        assert not wh._owner_reservation_evidence(root=td)
        open(os.path.join(td, "research", "queue", "GAME_MODE"), "w").close()
        assert wh._owner_reservation_evidence(root=td)


def test_a_running_game_process_is_no_longer_evidence():
    """Fix round 3 REMOVED the fix-2 process heuristic (ProtonVPN's always-on daemon matched it): the evidence
    check takes no process input at all -- see test_R3_protonvpn_and_steam_running_with_no_sentinel_is_NOT_exempt."""
    import inspect
    assert list(inspect.signature(wh._owner_reservation_evidence).parameters) == ["root"]
    assert not hasattr(wh, "_GAME_PROCESS_RE")


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
        assert v["budget_h"] == 0.0     # fix round 3: charged live time, 0h at the instant it was written


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


# ── 2026-09-23 REVIEW FIX (additional, same bullet as fix 4): the budget must be GLOBAL across worktrees, not
# silently per-worktree -- HISTORY_FILE / WAIVER_FILE must resolve through git-common-dir, like
# gates/lane_starvation's own `_shared_queue_root()` already does for `.lane_waiver` ────────────────────────────

def test_shared_root_resolves_via_git_common_dir_not_this_files_own_worktree():
    """When run from a git WORKTREE, `shared_root()` must still resolve to the canonical checkout (the one
    whose `.git` is the common dir) rather than to this worktree's own root -- otherwise two worktrees each
    read/write a SEPARATE `.waiver_history.jsonl` and each believes it owns the whole 6h/24h budget."""
    import subprocess
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    if not common:
        return  # not inside a git checkout in this environment; nothing to assert
    expected_root = os.path.dirname(common)
    assert wh.shared_root() == expected_root
    assert wh.HISTORY_FILE == os.path.join(expected_root, "research", "queue", ".waiver_history.jsonl")


def test_compute_idle_persistent_waiver_file_uses_the_shared_root_too():
    from tools.gates import compute_idle_persistent as g
    assert g.WAIVER_FILE == os.path.join(wh.shared_root(), "research", "queue", ".parallel_compute_waiver")


def test_shared_root_respects_sim_queue_root_override(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("SIM_QUEUE_ROOT", td)
        assert wh.shared_root() == os.path.abspath(td)


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
    # (tools/waiver_history.py itself is enforcement machinery and NOT exempt since fix round 3 -- see R3 tests)
    assert g._is_infra_only(["tools/lab.py", "tests/test_waiver_history.py", "README.md"])


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
    assert g._is_infra_only(["tools/lab.py", "docs/FAILURE_GATE_MATRIX.md", "CLAUDE.md"]), \
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
    names = {src[1] for src in pa.WAIVER_SOURCES}
    assert compute_idle_persistent.NAME in names, \
        "parallel_audit.py's waiver surfacing must key on compute_idle_persistent.NAME, not an ad hoc label"
    assert lane_starvation.NAME in names, \
        "parallel_audit.py's waiver surfacing must key on lane_starvation.NAME, not an ad hoc label"


def test_parallel_audit_waiver_surfacing_shares_the_gates_own_dedup_key():
    """The fix-2 double-count shape: the SAME live waiver read under the gate's own NAME and under a mismatched
    ad hoc label. Since fix round 3 the history is keyed by the waiver FILE's episode, not the reader's label,
    so a label mismatch can neither add a history row (within the throttle) nor add budget."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "history.jsonl")
        open(p, "w").write("CLASS: NO-READY-WORK\nchecked=lane_check.py -- 5/5 lanes served")
        os.utime(p, (100.0, 100.0))
        wh.evaluate("compute-idle-persistent", p, waiver_max_h=6, now_ts=100.0, history_file=hist)
        wh.evaluate("compute-idle-persistent", p, waiver_max_h=6, now_ts=101.0, history_file=hist)
        wh.evaluate("compute", p, waiver_max_h=6, now_ts=102.0, history_file=hist)     # the pre-fix label
        import json
        rows = [json.loads(l) for l in open(hist).read().splitlines() if l.strip()]
        assert len(rows) == 1, "one unchanged waiver read 3x within a minute must be ONE history row"
        assert wh.cumulative_waived_hours(now_ts=102.0, history_file=hist) == 0.0


# ══ FIX ROUND 3 (2026-09-23): adversarial tests for the four loopholes the fix-2 re-review found open ═══════════
# Each of these FAILS on 539f6f798 (the fix-2 head) and passes after the round-3 fix -- see the commit message
# for the recorded before/after run.

_NRW = "CLASS: NO-READY-WORK\nchecked=lane_check.py -- 5/5 lanes served, pool.queue empty"


def _write(p, text, mtime):
    with open(p, "w") as fh:
        fh.write(text)
    os.utime(p, (mtime, mtime))


# ── (1) EPISODE accounting: one unchanged waiver file is ONE episode, charged its real elapsed time ────────────

def test_R3_second_read_of_the_same_unchanged_waiver_is_not_rejected():
    """The exact re-review repro: read 0 ok, read 1 (+60s) was REJECTED 'EXHAUSTED (6.0h already used)' because
    the first read's own history row was summed against the second read."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "h.jsonl")
        _write(p, _NRW, 1000.0)
        v0 = wh.evaluate("compute-idle-persistent", p, 6, now_ts=1000.0, budget_h=6.0, history_file=hist)
        v1 = wh.evaluate("compute-idle-persistent", p, 6, now_ts=1060.0, budget_h=6.0, history_file=hist)
        v2 = wh.evaluate("compute-idle-persistent", p, 6, now_ts=1120.0, budget_h=6.0, history_file=hist)
        assert v0["ok"] and v1["ok"] and v2["ok"], (v0, v1, v2)


def test_R3_same_waiver_read_10_times_is_counted_once():
    """10 reads of ONE unchanged waiver over 45 min consume 45 min of budget -- the episode's real elapsed
    duration -- not 10x (or even 1x) its declared 6h cap."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "h.jsonl")
        t0 = 50_000.0
        _write(p, _NRW, t0)
        for k in range(10):
            v = wh.evaluate("lane-starvation", p, 6, now_ts=t0 + k * 300.0, budget_h=6.0, history_file=hist)
            assert v["ok"], "read %d of the SAME unchanged waiver was rejected: %r" % (k, v)
        used = wh.cumulative_waived_hours(now_ts=t0 + 9 * 300.0, history_file=hist)
        assert abs(used - 0.75) < 1e-6, "10 reads over 45 min must charge 0.75h (one episode), got %r" % used


def test_R3_a_genuine_waiver_stays_valid_for_its_whole_lifetime_under_heartbeat_reads():
    """The production shape: the heartbeat reads the live waiver every 15 min AND the commit gate reads it under
    the same NAME. A genuine 6h waiver with an empty history must be honoured on EVERY read until it expires."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "h.jsonl")
        t0 = 80_000.0
        _write(p, _NRW, t0)
        t = t0
        while t < t0 + 5.9 * 3600:
            for _caller in ("heartbeat", "commit-gate"):          # both read under the gate's own NAME
                v = wh.evaluate("compute-idle-persistent", p, 6, now_ts=t, budget_h=6.0, history_file=hist)
                assert v["ok"], "genuine waiver rejected %.2fh into its life: %r" % ((t - t0) / 3600, v)
            t += 900.0
        assert not wh.evaluate("compute-idle-persistent", p, 6, now_ts=t0 + 6.1 * 3600, budget_h=6.0,
                               history_file=hist)["active"], "an expired waiver must read inactive"


def test_R3_rewriting_the_waiver_every_few_minutes_cannot_escape_the_budget():
    """The renewal loophole the budget exists for: rewriting (a new mtime => a new episode) every 10 min must
    charge the CONTINUOUS time the file was live, so after 6h of back-to-back renewals the next is REJECTED."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "h.jsonl")
        t0 = 200_000.0
        rejected_at = None
        for k in range(60):                                  # 60 renewals x 10 min = 10h of attempted waiving
            t = t0 + k * 600.0
            _write(p, _NRW + "\nrenewal=%d" % k, t)
            v = wh.evaluate("compute-idle-persistent", p, 6, now_ts=t, budget_h=6.0, history_file=hist)
            if not v["ok"]:
                rejected_at = (t - t0) / 3600.0
                break
        assert rejected_at is not None, "back-to-back renewals were NEVER rejected -- the budget is escapable"
        assert 5.9 <= rejected_at <= 6.01, "renewals rejected at %.2fh, expected at the 6h budget" % rejected_at


def test_R3_a_deleted_waiver_is_charged_only_until_it_was_observed_gone():
    """A genuine waiver removed after ~1h must not be charged its full 6h cap: the first read that finds the
    file ABSENT closes the episode, so a later genuine waiver still has ~4.75h of budget."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "h.jsonl")
        t0 = 300_000.0
        _write(p, _NRW, t0)
        wh.evaluate("lane-starvation", p, 6, now_ts=t0, budget_h=6.0, history_file=hist)
        wh.evaluate("lane-starvation", p, 6, now_ts=t0 + 3600.0, budget_h=6.0, history_file=hist)
        os.remove(p)
        assert not wh.evaluate("lane-starvation", p, 6, now_ts=t0 + 4500.0, history_file=hist)["active"]
        _write(p, _NRW + "\nsecond episode", t0 + 10 * 3600.0)
        v = wh.evaluate("lane-starvation", p, 6, now_ts=t0 + 10 * 3600.0, budget_h=6.0, history_file=hist)
        assert v["ok"], v
        assert abs(v["budget_h"] - 1.25) < 1e-6, "expected 1.25h used (first episode only), got %r" % v["budget_h"]


def test_R3_future_dated_mtime_is_rejected_not_immortal():
    """`touch -d '+1 year'` gave a NEGATIVE age, so `age_h > max_age_h` never tripped: a waiver that never
    expires, charged a negative duration. A future mtime is now an invalid waiver."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "h.jsonl")
        _write(p, _NRW, 1_000_000.0 + 365 * 86400)
        v = wh.evaluate("compute-idle-persistent", p, 6, now_ts=1_000_000.0, budget_h=6.0, history_file=hist)
        assert v["active"] and not v["ok"], "a future-dated waiver must be REJECTED, got %r" % v


# ── (2) GAMING/OWNER-PAUSE exemption is SENTINEL-ONLY, read from the SHARED checkout ───────────────────────────

_PROTON_PS = ("/usr/bin/python3 -m proton.vpn.daemon\n/usr/lib/steam/steam -silent\n"
              "/home/u/.steam/ubuntu12_64/steamwebhelper --type=renderer\n")


def test_R3_protonvpn_and_steam_running_with_no_sentinel_is_NOT_exempt(monkeypatch):
    """THE LIVE LOOPHOLE: `python3 -m proton.vpn.daemon` runs permanently on this machine, and the Steam client
    idles in the background -- the fix-2 process heuristic matched both and granted a self-declared OWNER-PAUSE
    unlimited renewal with no owner action at all. The process list must not be evidence."""
    import subprocess as _sp

    class _Fake:
        stdout = _PROTON_PS
    monkeypatch.setattr(_sp, "run", lambda *a, **k: _Fake())
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "research", "queue"))
        monkeypatch.setenv("SIM_QUEUE_ROOT", td)
        assert not wh._owner_reservation_evidence(), \
            "ProtonVPN/Steam in the process list with NO sentinel was accepted as owner-reservation evidence"


def test_R3_owner_pause_with_no_sentinel_is_charged_and_rejected_when_budget_spent(monkeypatch):
    """End to end through the DEFAULT evidence check (no injected evidence_check), on the REAL process list of
    whatever host runs this: no GAME_MODE/GPU_PAUSE under the shared root => OWNER-PAUSE is an ordinary budgeted
    waiver, whatever processes (ProtonVPN, Steam) happen to be running."""
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "research", "queue"))
        monkeypatch.setenv("SIM_QUEUE_ROOT", td)
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "h.jsonl")
        _write(p, "CLASS: OWNER-PAUSE\nreason: owner reserved the box", 1000.0)
        v_room = wh.evaluate("g", p, 6, now_ts=1000.0, budget_h=6.0, prior_cumulative_h=0.0, history_file=hist)
        assert v_room["ok"] and v_room["budget_h"] is not None, "unevidenced OWNER-PAUSE must be CHARGED"
        v_spent = wh.evaluate("g", p, 6, now_ts=1000.0, budget_h=6.0, prior_cumulative_h=6.0, history_file=hist)
        assert not v_spent["ok"], "unevidenced OWNER-PAUSE was not rejected with the budget spent"


def test_R3_owner_pause_WITH_the_game_mode_sentinel_is_exempt(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "research", "queue"))
        open(os.path.join(td, "research", "queue", "GAME_MODE"), "w").close()
        monkeypatch.setenv("SIM_QUEUE_ROOT", td)
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "h.jsonl")
        _write(p, "CLASS: GAMING\nreason: owner is playing", 1000.0)
        v = wh.evaluate("g", p, 6, now_ts=1000.0, budget_h=6.0, prior_cumulative_h=999.0, history_file=hist)
        assert v["ok"] and v["budget_h"] is None


def test_R3_gpu_pause_sentinel_is_also_evidence(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "research", "queue"))
        open(os.path.join(td, "research", "queue", "GPU_PAUSE"), "w").close()
        monkeypatch.setenv("SIM_QUEUE_ROOT", td)
        assert wh._owner_reservation_evidence()


def test_R3_sentinel_is_read_from_the_SHARED_root_not_the_evaluating_worktree(monkeypatch):
    """tools/game.sh / tools/gpu_queue.sh write the sentinel under the MAIN checkout's research/queue. A
    sentinel sitting only in the evaluating worktree is not the owner's (and was never seen by the owner's
    tools either)."""
    with tempfile.TemporaryDirectory() as shared, tempfile.TemporaryDirectory() as worktree:
        os.makedirs(os.path.join(shared, "research", "queue"))
        os.makedirs(os.path.join(worktree, "research", "queue"))
        open(os.path.join(worktree, "research", "queue", "GAME_MODE"), "w").close()
        monkeypatch.setenv("SIM_QUEUE_ROOT", shared)
        monkeypatch.setattr(wh, "_ROOT", worktree)
        assert not wh._owner_reservation_evidence(), "a worktree-local GAME_MODE was accepted as evidence"
        open(os.path.join(shared, "research", "queue", "GAME_MODE"), "w").close()
        assert wh._owner_reservation_evidence(), "the SHARED-root GAME_MODE was not recognised"


# ── (3) _is_infra_only: an explicit allow-list, deny by default ─────────────────────────────────────────────

def test_R3_infra_only_does_not_exempt_research_status_docs_or_the_gates_themselves():
    from tools.gates import compute_idle_persistent as g
    for p in ("docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md", "research/biology/btsp.md", "ROADMAP.md",
              "GAP_CLOSURE_MISSION.md", "research/FAILURE_LOG.md", "HANDOFF.md", "docs/CURRENT-STATE.md",
              "tools/gates/compute_idle_persistent.py", "tools/gates/lane_starvation.py",
              "tools/waiver_history.py", "tools/parallel_state.py", "tools/parallel_audit.py",
              "tools/githooks/pre-commit"):
        assert not g._is_infra_only([p]), "%s was exempted as compute-neutral infra" % p
        assert not g._is_infra_only(["tools/lab.py", p]), "%s was exempted inside a tools/ commit" % p


def test_R3_infra_only_still_exempts_genuinely_generic_infra():
    from tools.gates import compute_idle_persistent as g
    assert g._is_infra_only(["tools/lab.py", "tests/test_lab.py", "README.md", "CLAUDE.md",
                             "docs/FAILURE_GATE_MATRIX.md", ".gitignore"])


# ── (4) one waiver is one budget line, whichever gate NAME / label reads it ─────────────────────────────────

def test_R3_one_waiver_read_under_two_different_gate_labels_is_charged_once():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".waiver"); hist = os.path.join(td, "h.jsonl")
        t0 = 400_000.0
        _write(p, _NRW, t0)
        wh.evaluate("compute-idle-persistent", p, 6, now_ts=t0, history_file=hist)
        wh.evaluate("compute", p, 6, now_ts=t0 + 1800.0, history_file=hist)      # the pre-fix ad hoc label
        wh.evaluate("compute-idle-persistent", p, 6, now_ts=t0 + 3600.0, history_file=hist)
        used = wh.cumulative_waived_hours(now_ts=t0 + 3600.0, history_file=hist)
        assert abs(used - 1.0) < 1e-6, "one waiver live 1h under two labels must charge 1h, got %r" % used


def test_R3_parallel_audit_waiver_sources_are_the_gates_own_name_file_and_cap():
    import importlib.util
    from tools.gates import compute_idle_persistent as cip, lane_starvation as ls
    spec = importlib.util.spec_from_file_location("parallel_audit_r3", os.path.join(ROOT, "tools",
                                                                                    "parallel_audit.py"))
    pa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pa)
    by_name = {src[1]: (src[2], src[3]) for src in pa.WAIVER_SOURCES}
    assert by_name[cip.NAME] == (cip.WAIVER_FILE, cip.WAIVER_MAX_H)
    assert by_name[ls.NAME] == (ls._waiver_file(), ls.LANE_WAIVER_MAX_H)


# ── (5) the acknowledged residual: parallel_state's persistence record must be read from the SHARED root too ──

def test_R3_parallel_state_file_resolves_through_the_shared_root():
    import parallel_state
    assert parallel_state.STATE_FILE == os.path.join(wh.shared_root(), "research", "coordination",
                                                     "parallel_audit_state.json")
