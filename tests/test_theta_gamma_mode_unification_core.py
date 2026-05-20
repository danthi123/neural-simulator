from research.runners.theta_gamma_mode_unification_core import (
    theta_gamma_mode_unification_verdict,
    _TG_FULL_MIN, _TG_UNIFORM_CTRL_MAX, _TG_DIRECT_RETAIN_MIN,
    _TG_ABSTAIN_CORRECT_MIN, _TG_SCALE_TOL, _TG_LADDER, _TG_MIN_SEEDS,
)


def _rung(N, full=0.88, uniform=0.05, direct=0.90, ab=0.97, n_seeds=3):
    return {"N": N, "n_seeds": n_seeds, "full_acc": full,
            "uniform_ctrl_acc": uniform,
            "direct_retain_acc": direct,
            "abstain_correct": ab}


def test_frozen_constant_pins():
    assert _TG_FULL_MIN == 0.80
    assert _TG_UNIFORM_CTRL_MAX == 0.10
    assert _TG_DIRECT_RETAIN_MIN == 0.80
    assert _TG_ABSTAIN_CORRECT_MIN == 0.90
    assert _TG_SCALE_TOL == 0.10
    assert _TG_LADDER == (2, 3, 5)
    assert _TG_MIN_SEEDS == 3


def test_clean_pass():
    rungs = [_rung(2), _rung(3, full=0.86), _rung(5, full=0.84)]
    assert theta_gamma_mode_unification_verdict(rungs)["gate"] == "PASS"


def test_uniform_ctrl_not_collapsing_is_fail():
    # capability NOT attributable to theta-gamma mode-unification -> FAIL
    rungs = [_rung(2, uniform=0.30)]
    v = theta_gamma_mode_unification_verdict(rungs)
    assert v["gate"] == "FAIL" and v["gate"] != "VOID"


def test_direct_retain_below_floor_is_fail():
    # direct retrieval degrades -> FAIL (no degradation allowed)
    assert theta_gamma_mode_unification_verdict([_rung(2, direct=0.50)])["gate"] == "FAIL"


def test_abstain_below_bar_is_fail():
    assert theta_gamma_mode_unification_verdict([_rung(2, ab=0.50)])["gate"] == "FAIL"


def test_full_below_bar_is_fail():
    assert theta_gamma_mode_unification_verdict([_rung(2, full=0.10)])["gate"] == "FAIL"


def test_small_load_only_is_works_small():
    rungs = [_rung(2, full=0.88), _rung(3, full=0.60), _rung(5, full=0.40)]
    assert theta_gamma_mode_unification_verdict(rungs)["gate"] == "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE"


def test_below_min_seeds_is_void():
    v = theta_gamma_mode_unification_verdict([_rung(2, n_seeds=2)])
    assert v["gate"] == "VOID" and v["gate"] != "FAIL"


def test_ladder_mismatch_is_void():
    # 4 is not in _TG_LADDER == (2,3,5)
    assert theta_gamma_mode_unification_verdict([_rung(4)])["gate"] == "VOID"


def test_nonfinite_is_void():
    assert theta_gamma_mode_unification_verdict([_rung(2, full=float("nan"))])["gate"] == "VOID"
    assert theta_gamma_mode_unification_verdict([_rung(2, uniform=float("inf"))])["gate"] == "VOID"
    assert theta_gamma_mode_unification_verdict([_rung(2, direct=float("nan"))])["gate"] == "VOID"


def test_missing_key_is_void():
    bad = {"N": 2, "n_seeds": 3, "full_acc": 0.9}
    assert theta_gamma_mode_unification_verdict([bad])["gate"] == "VOID"


def test_empty_nonlist_none_is_void():
    assert theta_gamma_mode_unification_verdict([])["gate"] == "VOID"
    assert theta_gamma_mode_unification_verdict("nope")["gate"] == "VOID"
    assert theta_gamma_mode_unification_verdict(None)["gate"] == "VOID"


def test_bool_not_numeric_is_void():
    assert theta_gamma_mode_unification_verdict([_rung(2, full=True)])["gate"] == "VOID"


def test_duplicate_N_is_void():
    assert theta_gamma_mode_unification_verdict([_rung(2), _rung(2)])["gate"] == "VOID"


def test_precomputed_verdict_ignored():
    r = _rung(2); r["verdict"] = "PASS"; r["full_acc"] = 0.10
    assert theta_gamma_mode_unification_verdict([r])["gate"] == "FAIL"


def test_degenerate_always_abstain_is_fail():
    rungs = [_rung(2, full=0.0, uniform=0.0, direct=0.0, ab=1.0)]
    assert theta_gamma_mode_unification_verdict(rungs)["gate"] == "FAIL"


def test_degenerate_always_answer_is_fail():
    assert theta_gamma_mode_unification_verdict([_rung(2, ab=0.0)])["gate"] == "FAIL"


def test_void_and_fail_distinct_with_metadata():
    void = theta_gamma_mode_unification_verdict([])
    fail = theta_gamma_mode_unification_verdict([_rung(2, full=0.10)])
    assert void["gate"] == "VOID" and fail["gate"] == "FAIL"
    assert void["gate"] != fail["gate"]
    for d in (void, fail):
        assert "reason" in d and "frozen_bars" in d
    fb = theta_gamma_mode_unification_verdict([_rung(2), _rung(3), _rung(5)])["frozen_bars"]
    assert fb["full_min"] == 0.80 and fb["uniform_ctrl_max"] == 0.10
    assert fb["direct_retain_min"] == 0.80 and fb["abstain_correct_min"] == 0.90
    assert fb["scale_tol"] == 0.10
    assert tuple(fb["ladder"]) == (2, 3, 5) and fb["min_seeds"] == 3
