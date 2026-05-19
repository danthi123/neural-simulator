from research.runners.spear_conversational_core import (
    spear_conversational_verdict,
    _SP_FULL_MIN, _SP_STATIC_CTRL_MAX, _SP_ABSTAIN_MIN,
    _SP_SCALE_TOL, _SP_LADDER, _SP_MIN_SEEDS,
)


def _rung(N, full=0.88, rhythm_removed=0.05, ab=0.97, n_seeds=3):
    return {"N": N, "n_seeds": n_seeds, "full_acc": full,
            "rhythm_removed_acc": rhythm_removed,
            "abstain_correct_rhythm_removed": ab}


def test_frozen_constant_pins():
    assert _SP_FULL_MIN == 0.80
    assert _SP_STATIC_CTRL_MAX == 0.40
    assert _SP_ABSTAIN_MIN == 0.90
    assert _SP_SCALE_TOL == 0.10
    assert _SP_LADDER == (2, 4, 8)
    assert _SP_MIN_SEEDS == 3


def test_clean_pass():
    rungs = [_rung(2), _rung(4, full=0.86), _rung(8, full=0.84)]
    assert spear_conversational_verdict(rungs)["gate"] == "PASS"


def test_rhythm_removed_not_collapsing_is_fail():
    # capability NOT attributable to the rhythm -> FAIL
    rungs = [_rung(2, rhythm_removed=0.75)]
    v = spear_conversational_verdict(rungs)
    assert v["gate"] == "FAIL" and v["gate"] != "VOID"


def test_abstain_below_bar_is_fail():
    assert spear_conversational_verdict([_rung(2, ab=0.50)])["gate"] == "FAIL"


def test_full_below_bar_is_fail():
    assert spear_conversational_verdict([_rung(2, full=0.10)])["gate"] == "FAIL"


def test_small_load_only_is_works_small():
    rungs = [_rung(2, full=0.88), _rung(4, full=0.60), _rung(8, full=0.45)]
    assert spear_conversational_verdict(rungs)["gate"] == "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE"


def test_below_min_seeds_is_void():
    v = spear_conversational_verdict([_rung(2, n_seeds=2)])
    assert v["gate"] == "VOID" and v["gate"] != "FAIL"


def test_ladder_mismatch_is_void():
    assert spear_conversational_verdict([_rung(3)])["gate"] == "VOID"


def test_nonfinite_is_void():
    assert spear_conversational_verdict([_rung(2, full=float("nan"))])["gate"] == "VOID"
    assert spear_conversational_verdict([_rung(2, rhythm_removed=float("inf"))])["gate"] == "VOID"


def test_missing_key_is_void():
    assert spear_conversational_verdict([{"N": 2, "n_seeds": 3, "full_acc": 0.9}])["gate"] == "VOID"


def test_empty_nonlist_none_is_void():
    assert spear_conversational_verdict([])["gate"] == "VOID"
    assert spear_conversational_verdict("nope")["gate"] == "VOID"
    assert spear_conversational_verdict(None)["gate"] == "VOID"


def test_bool_not_numeric_is_void():
    assert spear_conversational_verdict([_rung(2, full=True)])["gate"] == "VOID"


def test_duplicate_N_is_void():
    assert spear_conversational_verdict([_rung(2), _rung(2)])["gate"] == "VOID"


def test_precomputed_verdict_ignored():
    r = _rung(2); r["verdict"] = "PASS"; r["full_acc"] = 0.10
    assert spear_conversational_verdict([r])["gate"] == "FAIL"


def test_degenerate_always_abstain_is_fail():
    rungs = [_rung(2, full=0.0, rhythm_removed=0.0, ab=1.0)]
    assert spear_conversational_verdict(rungs)["gate"] == "FAIL"


def test_degenerate_always_answer_is_fail():
    assert spear_conversational_verdict([_rung(2, ab=0.0)])["gate"] == "FAIL"


def test_void_and_fail_distinct_with_metadata():
    void = spear_conversational_verdict([])
    fail = spear_conversational_verdict([_rung(2, full=0.10)])
    assert void["gate"] == "VOID" and fail["gate"] == "FAIL"
    assert void["gate"] != fail["gate"]
    for d in (void, fail):
        assert "reason" in d and "frozen_bars" in d
    fb = spear_conversational_verdict([_rung(2), _rung(4), _rung(8)])["frozen_bars"]
    assert fb["full_min"] == 0.80 and fb["static_ctrl_max"] == 0.40
    assert fb["abstain_min"] == 0.90 and fb["scale_tol"] == 0.10
    assert tuple(fb["ladder"]) == (2, 4, 8) and fb["min_seeds"] == 3
