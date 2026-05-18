from research.runners.engram_bootstrap_gate import (
    scale_confidence, _SCALE_LADDER, _SCALE_TOL)


def _rung(B, gate, td, engram_only):
    # Minimal per-rung record shape the aggregator consumes.
    return {"B": B, "verdict": {"GATE": gate}, "td_mean": td,
            "engram_only_mean": engram_only}


def test_all_pass_monotone_is_scale_confident():
    rungs = [_rung(4, "PASS", 0.85, 0.20),
             _rung(8, "PASS", 0.90, 0.22),
             _rung(16, "PASS", 0.92, 0.25)]
    r = scale_confidence(rungs)
    assert r["scale_confident"] is True
    assert r["classification"] == "SCALE-CONFIDENT-PASS"


def test_works_small_but_plateaus_is_not_confident():
    # PASS at every rung but generative margin collapses by B=16.
    rungs = [_rung(4, "PASS", 0.95, 0.20),
             _rung(8, "PASS", 0.88, 0.20),
             _rung(16, "PASS", 0.80, 0.78)]  # (c) fails: margin < tol
    r = scale_confidence(rungs)
    assert r["scale_confident"] is False
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_degradation_beyond_tol_breaks_monotone():
    rungs = [_rung(4, "PASS", 0.95, 0.10),
             _rung(8, "PASS", 0.95, 0.10),
             _rung(16, "PASS", 0.85, 0.10)]  # 0.85 < 0.95 - 0.05 => (b) fails
    r = scale_confidence(rungs)
    assert r["scale_confident"] is False
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_any_void_rung_is_void():
    rungs = [_rung(4, "PASS", 0.9, 0.1),
             _rung(8, "VOID", 0.0, 0.0),
             _rung(16, "PASS", 0.9, 0.1)]
    r = scale_confidence(rungs)
    assert r["scale_confident"] is False
    assert r["classification"] == "VOID"


def test_any_fail_rung_is_fail():
    rungs = [_rung(4, "PASS", 0.9, 0.1),
             _rung(8, "FAIL", 0.5, 0.5),
             _rung(16, "PASS", 0.9, 0.1)]
    r = scale_confidence(rungs)
    assert r["scale_confident"] is False
    assert r["classification"] == "FAIL"


def test_smallest_rung_must_pass_else_void_or_fail_propagates():
    # Smallest rung VOID dominates (instrument unsound at base scale).
    rungs = [_rung(4, "VOID", 0.0, 0.0),
             _rung(8, "PASS", 0.9, 0.1),
             _rung(16, "PASS", 0.9, 0.1)]
    assert scale_confidence(rungs)["classification"] == "VOID"


def test_frozen_constants_pinned():
    assert _SCALE_LADDER == (4, 8, 16)
    assert _SCALE_TOL == 0.05


def test_non_numeric_or_missing_rung_metric_is_void_not_raise():
    bad = [{"B": 4, "verdict": {"GATE": "PASS"}, "td_mean": "oops",
            "engram_only_mean": 0.1},
           _rung(8, "PASS", 0.9, 0.1), _rung(16, "PASS", 0.9, 0.1)]
    r = scale_confidence(bad)
    assert r["scale_confident"] is False
    assert r["classification"] == "VOID"  # fail-closed, never raise
