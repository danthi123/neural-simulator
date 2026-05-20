from research.runners.pirazzini_three_layer_core import (
    pirazzini_three_layer_verdict,
    _PZ_FULL_MIN, _PZ_CONVERGENT_CEILING_MAX, _PZ_ABSTAIN_MIN,
    _PZ_SCALE_TOL, _PZ_LADDER, _PZ_MIN_SEEDS,
)


def _rung(N, full=0.88, theta_disabled=0.05, ab=0.97, n_seeds=3):
    return {"N": N, "n_seeds": n_seeds, "full_acc": full,
            "theta_disabled_acc": theta_disabled,
            "abstain_correct_theta_disabled": ab}


def test_frozen_constant_pins():
    assert _PZ_FULL_MIN == 0.80
    assert _PZ_CONVERGENT_CEILING_MAX == 0.10
    assert _PZ_ABSTAIN_MIN == 0.90
    assert _PZ_SCALE_TOL == 0.10
    assert _PZ_LADDER == (2, 3, 5)
    assert _PZ_MIN_SEEDS == 3


def test_clean_pass():
    rungs = [_rung(2), _rung(3, full=0.86), _rung(5, full=0.84)]
    assert pirazzini_three_layer_verdict(rungs)["gate"] == "PASS"


def test_theta_disabled_not_collapsing_to_convergent_ceiling_is_fail():
    # capability NOT attributable to the theta generator -> FAIL
    rungs = [_rung(2, theta_disabled=0.30)]
    v = pirazzini_three_layer_verdict(rungs)
    assert v["gate"] == "FAIL" and v["gate"] != "VOID"


def test_abstain_below_bar_is_fail():
    assert pirazzini_three_layer_verdict([_rung(2, ab=0.50)])["gate"] == "FAIL"


def test_full_below_bar_is_fail():
    assert pirazzini_three_layer_verdict([_rung(2, full=0.10)])["gate"] == "FAIL"


def test_small_load_only_is_works_small():
    rungs = [_rung(2, full=0.88), _rung(3, full=0.60), _rung(5, full=0.40)]
    assert pirazzini_three_layer_verdict(rungs)["gate"] == "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE"


def test_below_min_seeds_is_void():
    v = pirazzini_three_layer_verdict([_rung(2, n_seeds=2)])
    assert v["gate"] == "VOID" and v["gate"] != "FAIL"


def test_ladder_mismatch_is_void():
    # 4 is not in _PZ_LADDER == (2,3,5)
    assert pirazzini_three_layer_verdict([_rung(4)])["gate"] == "VOID"


def test_nonfinite_is_void():
    assert pirazzini_three_layer_verdict([_rung(2, full=float("nan"))])["gate"] == "VOID"
    assert pirazzini_three_layer_verdict([_rung(2, theta_disabled=float("inf"))])["gate"] == "VOID"


def test_missing_key_is_void():
    bad = {"N": 2, "n_seeds": 3, "full_acc": 0.9}
    assert pirazzini_three_layer_verdict([bad])["gate"] == "VOID"


def test_empty_nonlist_none_is_void():
    assert pirazzini_three_layer_verdict([])["gate"] == "VOID"
    assert pirazzini_three_layer_verdict("nope")["gate"] == "VOID"
    assert pirazzini_three_layer_verdict(None)["gate"] == "VOID"


def test_bool_not_numeric_is_void():
    assert pirazzini_three_layer_verdict([_rung(2, full=True)])["gate"] == "VOID"


def test_duplicate_N_is_void():
    assert pirazzini_three_layer_verdict([_rung(2), _rung(2)])["gate"] == "VOID"


def test_precomputed_verdict_ignored():
    r = _rung(2); r["verdict"] = "PASS"; r["full_acc"] = 0.10
    assert pirazzini_three_layer_verdict([r])["gate"] == "FAIL"


def test_degenerate_always_abstain_is_fail():
    rungs = [_rung(2, full=0.0, theta_disabled=0.0, ab=1.0)]
    assert pirazzini_three_layer_verdict(rungs)["gate"] == "FAIL"


def test_degenerate_always_answer_is_fail():
    assert pirazzini_three_layer_verdict([_rung(2, ab=0.0)])["gate"] == "FAIL"


def test_void_and_fail_distinct_with_metadata():
    void = pirazzini_three_layer_verdict([])
    fail = pirazzini_three_layer_verdict([_rung(2, full=0.10)])
    assert void["gate"] == "VOID" and fail["gate"] == "FAIL"
    assert void["gate"] != fail["gate"]
    for d in (void, fail):
        assert "reason" in d and "frozen_bars" in d
    fb = pirazzini_three_layer_verdict([_rung(2), _rung(3), _rung(5)])["frozen_bars"]
    assert fb["full_min"] == 0.80 and fb["convergent_ceiling_max"] == 0.10
    assert fb["abstain_min"] == 0.90 and fb["scale_tol"] == 0.10
    assert tuple(fb["ladder"]) == (2, 3, 5) and fb["min_seeds"] == 3
