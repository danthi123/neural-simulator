from research.runners.q2r_core import (
    q2r_scale_confidence, _Q2R_LADDER, _Q2R_SCALE_TOL, _Q2R_TOP_MIN,
    _Q2R_MIN_SEEDS)


def _rg(K, gate, nv):
    return {"K": K, "verdict": {"GATE": gate},
            "constrained_nonvac_rate_mean": nv}


def _good(nvs):  # all PASS rungs with given non-vacuity sequence
    return [_rg(K, "PASS", nv) for K, nv in zip(_Q2R_LADDER, nvs)]


def test_frozen_constants_pinned():
    assert _Q2R_LADDER == (12, 24, 48, 96)
    assert _Q2R_SCALE_TOL == 0.10
    assert _Q2R_TOP_MIN == 0.50
    assert _Q2R_MIN_SEEDS == 3


def test_all_pass_nondecreasing_top_clears_is_scale_confident():
    r = q2r_scale_confidence(_good([0.55, 0.60, 0.66, 0.72]))
    assert r["scale_confident"] is True
    assert r["classification"] == "SCALE-CONFIDENT-PASS"


def test_monotone_within_tol_ok():
    r = q2r_scale_confidence(_good([0.60, 0.55, 0.58, 0.62]))
    assert r["classification"] == "SCALE-CONFIDENT-PASS"


def test_trend_drop_beyond_tol_is_works_small():
    r = q2r_scale_confidence(_good([0.70, 0.55, 0.52, 0.55]))
    assert r["scale_confident"] is False
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_top_below_floor_is_works_small():
    r = q2r_scale_confidence(_good([0.52, 0.55, 0.58, 0.49]))
    assert r["scale_confident"] is False
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_any_void_rung_is_void_precedence_over_fail():
    rungs = _good([0.6, 0.6, 0.6, 0.6])
    rungs[1]["verdict"]["GATE"] = "VOID"
    rungs[2]["verdict"]["GATE"] = "FAIL"
    assert q2r_scale_confidence(rungs)["classification"] == "VOID"


def test_any_fail_rung_is_fail():
    rungs = _good([0.6, 0.6, 0.6, 0.6])
    rungs[2]["verdict"]["GATE"] = "FAIL"
    assert q2r_scale_confidence(rungs)["classification"] == "FAIL"


def test_ladder_mismatch_is_void():
    bad = [_rg(12, "PASS", 0.6), _rg(24, "PASS", 0.6), _rg(48, "PASS", 0.6)]
    assert q2r_scale_confidence(bad)["classification"] == "VOID"


def test_ladder_padding_duplicate_K_is_void():
    bad = [_rg(12, "PASS", 0.6), _rg(24, "PASS", 0.6),
           _rg(48, "PASS", 0.6), _rg(48, "PASS", 0.6)]
    assert q2r_scale_confidence(bad)["classification"] == "VOID"


def test_unknown_gate_is_void():
    rungs = _good([0.6, 0.6, 0.6, 0.6])
    rungs[0]["verdict"]["GATE"] = "MAYBE"
    assert q2r_scale_confidence(rungs)["classification"] == "VOID"


def test_non_numeric_nonvac_is_void_not_raise():
    rungs = _good([0.6, 0.6, 0.6, 0.6])
    rungs[3]["constrained_nonvac_rate_mean"] = "oops"
    assert q2r_scale_confidence(rungs)["classification"] == "VOID"


def test_missing_verdict_is_void_not_raise():
    rungs = _good([0.6, 0.6, 0.6, 0.6])
    del rungs[1]["verdict"]
    assert q2r_scale_confidence(rungs)["classification"] == "VOID"


def test_unorderable_is_void_not_raise():
    assert q2r_scale_confidence([{"K": object(),
        "verdict": {"GATE": "PASS"},
        "constrained_nonvac_rate_mean": 0.6}])["classification"] == "VOID"
