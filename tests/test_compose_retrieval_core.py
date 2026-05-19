from research.runners.compose_retrieval_core import (
    compose_retrieval_verdict,
    _CR_FULL_MIN, _CR_ABLATION_MAX, _CR_ABSTAIN_MIN,
    _CR_SCALE_TOL, _CR_LADDER, _CR_MIN_SEEDS,
)


def _rung(N, full=0.88, recent_only=0.20, remote_only=0.18,
          ab_recent=0.97, ab_remote=0.96, n_seeds=3):
    return {"N": N, "n_seeds": n_seeds, "full_acc": full,
            "recent_only_acc": recent_only, "remote_only_acc": remote_only,
            "abstain_correct_recent_only": ab_recent,
            "abstain_correct_remote_only": ab_remote}


def test_frozen_constant_pins():
    assert _CR_FULL_MIN == 0.80
    assert _CR_ABLATION_MAX == 0.40
    assert _CR_ABSTAIN_MIN == 0.90
    assert _CR_SCALE_TOL == 0.10
    assert _CR_LADDER == (2, 4, 8)
    assert _CR_MIN_SEEDS == 3


def test_clean_pass():
    rungs = [_rung(2), _rung(4, full=0.86), _rung(8, full=0.84)]
    assert compose_retrieval_verdict(rungs)["gate"] == "PASS"


def test_recent_only_not_collapsing_is_fail():
    rungs = [_rung(2, recent_only=0.75)]
    v = compose_retrieval_verdict(rungs)
    assert v["gate"] == "FAIL" and v["gate"] != "VOID"


def test_remote_only_not_collapsing_is_fail():
    rungs = [_rung(2, remote_only=0.70)]
    assert compose_retrieval_verdict(rungs)["gate"] == "FAIL"


def test_abstain_recent_below_bar_is_fail():
    rungs = [_rung(2, ab_recent=0.50)]
    assert compose_retrieval_verdict(rungs)["gate"] == "FAIL"


def test_abstain_remote_below_bar_is_fail():
    rungs = [_rung(2, ab_remote=0.50)]
    assert compose_retrieval_verdict(rungs)["gate"] == "FAIL"


def test_small_load_only_is_works_small():
    rungs = [_rung(2, full=0.88), _rung(4, full=0.60), _rung(8, full=0.45)]
    assert compose_retrieval_verdict(rungs)["gate"] == "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE"


def test_below_min_seeds_is_void():
    rungs = [_rung(2, n_seeds=2)]
    v = compose_retrieval_verdict(rungs)
    assert v["gate"] == "VOID" and v["gate"] != "FAIL"


def test_ladder_mismatch_is_void():
    assert compose_retrieval_verdict([_rung(3)])["gate"] == "VOID"


def test_nonfinite_is_void():
    assert compose_retrieval_verdict([_rung(2, full=float("nan"))])["gate"] == "VOID"
    assert compose_retrieval_verdict([_rung(2, remote_only=float("inf"))])["gate"] == "VOID"


def test_missing_key_is_void():
    bad = {"N": 2, "n_seeds": 3, "full_acc": 0.9}
    assert compose_retrieval_verdict([bad])["gate"] == "VOID"


def test_empty_and_nonlist_is_void():
    assert compose_retrieval_verdict([])["gate"] == "VOID"
    assert compose_retrieval_verdict("nope")["gate"] == "VOID"
    assert compose_retrieval_verdict(None)["gate"] == "VOID"


def test_bool_is_not_accepted_as_numeric_void():
    assert compose_retrieval_verdict([_rung(2, full=True)])["gate"] == "VOID"


def test_precomputed_verdict_field_is_ignored():
    r = _rung(2)
    r["verdict"] = "PASS"
    r["full_acc"] = 0.10
    assert compose_retrieval_verdict([r])["gate"] == "FAIL"


def test_duplicate_rung_N_is_void():
    assert compose_retrieval_verdict([_rung(2), _rung(2)])["gate"] == "VOID"


def test_degenerate_always_abstain_is_fail():
    rungs = [_rung(2, full=0.0, recent_only=0.0, remote_only=0.0,
                   ab_recent=1.0, ab_remote=1.0)]
    assert compose_retrieval_verdict(rungs)["gate"] == "FAIL"


def test_degenerate_always_answer_is_fail():
    rungs = [_rung(2, ab_recent=0.0, ab_remote=0.0)]
    assert compose_retrieval_verdict(rungs)["gate"] == "FAIL"


def test_void_and_fail_are_distinct_strings():
    void = compose_retrieval_verdict([])
    fail = compose_retrieval_verdict([_rung(2, full=0.10)])
    assert void["gate"] == "VOID" and fail["gate"] == "FAIL"
    assert void["gate"] != fail["gate"]
    assert "frozen_bars" in void and "frozen_bars" in fail
    assert "reason" in void and "reason" in fail


def test_pass_is_recomputed_not_trusted_and_has_frozen_bars():
    v = compose_retrieval_verdict([_rung(2), _rung(4), _rung(8)])
    assert v["gate"] == "PASS"
    fb = v["frozen_bars"]
    assert fb["full_min"] == 0.80 and fb["ablation_max"] == 0.40
    assert fb["abstain_min"] == 0.90 and fb["scale_tol"] == 0.10
    assert tuple(fb["ladder"]) == (2, 4, 8) and fb["min_seeds"] == 3
