import math
import pytest

from research.runners.integrated_loop_core import (
    integrated_loop_verdict,
    _IL_V1_MIN, _IL_SCI_MIN, _IL_LESION_MAX, _IL_SCALE_TOL,
    _IL_LADDER, _IL_MIN_SEEDS,
    _SHARED, _HELPER_WM, _HELPER_EP, _HELPER_BOTH)


def _good_rung(N, full=0.9, n_seeds=5):
    """A rung where the full loop succeeds on BOTH readouts and EVERY
    lesion collapses exactly the readout(s) it is responsible for."""
    lesions = {}
    for name in _SHARED:
        lesions[name] = {"wm": 0.2, "ep": 0.2}          # both collapse
    for name in _HELPER_WM:
        lesions[name] = {"wm": 0.2, "ep": 0.9}          # wm collapses
    for name in _HELPER_EP:
        lesions[name] = {"wm": 0.9, "ep": 0.2}          # ep collapses
    for name in _HELPER_BOTH:
        lesions[name] = {"wm": 0.2, "ep": 0.2}          # both collapse
    return {"N": N, "n_seeds": n_seeds,
            "v1": {"wm": 0.95, "ep": 0.95},
            "full": {"wm": full, "ep": full},
            "lesions": lesions}


def test_01_scale_confident_pass():
    rungs = [_good_rung(2, 0.86), _good_rung(4, 0.88), _good_rung(8, 0.90)]
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "PASS"
    assert r["classification"] == "SCALE-CONFIDENT-PASS"


def test_02_works_small_trend_break():
    rungs = [_good_rung(2, 0.95), _good_rung(4, 0.95), _good_rung(8, 0.81)]
    rungs[2]["full"] = {"wm": 0.81, "ep": 0.66}
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "FAIL"
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_03_works_small_top_below_floor():
    rungs = [_good_rung(2, 0.86), _good_rung(4, 0.84), _good_rung(8, 0.82)]
    rungs[2]["full"] = {"wm": 0.78, "ep": 0.78}
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "FAIL"
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_04_shared_lesion_does_not_collapse_wm_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[1]["lesions"][_SHARED[0]] = {"wm": 0.85, "ep": 0.2}
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"
    assert r["instrument_valid"] is False


def test_05_shared_lesion_collapses_wm_but_not_ep_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[0]["lesions"][_SHARED[1]] = {"wm": 0.2, "ep": 0.88}
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"
    assert r["instrument_valid"] is False


def test_06_helper_wm_lesion_does_not_collapse_wm_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[2]["lesions"][_HELPER_WM[0]] = {"wm": 0.9, "ep": 0.9}
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"


def test_07_helper_ep_lesion_does_not_collapse_ep_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[0]["lesions"][_HELPER_EP[0]] = {"wm": 0.9, "ep": 0.9}
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"


def test_08_helper_both_lesion_collapses_only_one_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[1]["lesions"][_HELPER_BOTH[0]] = {"wm": 0.2, "ep": 0.9}
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"


def test_09_v1_unmet_is_void_not_fail():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[0]["v1"] = {"wm": 0.55, "ep": 0.95}
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"
    assert r["instrument_valid"] is False


def test_10_sound_discriminating_but_science_below_bar_is_fail():
    rungs = [_good_rung(2, 0.70), _good_rung(4, 0.70), _good_rung(8, 0.70)]
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "FAIL"
    assert r["instrument_valid"] is True
    assert r["classification"] == "FAIL"


def test_11_ladder_mismatch_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(4)]
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"


def test_12_non_numeric_and_nan_is_void_not_raise():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[1]["full"] = {"wm": "0.9", "ep": 0.9}
    assert integrated_loop_verdict(rungs)["GATE"] == "VOID"
    rungs2 = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs2[2]["full"] = {"wm": float("nan"), "ep": 0.9}
    assert integrated_loop_verdict(rungs2)["GATE"] == "VOID"


def test_13_too_few_seeds_is_void():
    rungs = [_good_rung(2, n_seeds=2), _good_rung(4), _good_rung(8)]
    r = integrated_loop_verdict(rungs)
    assert r["GATE"] == "VOID"


def test_14_malformed_top_level_is_void_not_raise():
    assert integrated_loop_verdict(None)["GATE"] == "VOID"
    assert integrated_loop_verdict([])["GATE"] == "VOID"
    assert integrated_loop_verdict("garbage")["GATE"] == "VOID"
    assert integrated_loop_verdict([{"no": "N"}])["GATE"] == "VOID"


def test_15_threshold_tamper_pins():
    assert _IL_LADDER == (2, 4, 8)
    assert _IL_V1_MIN == 0.90
    assert _IL_SCI_MIN == 0.80
    assert _IL_LESION_MAX == 0.40
    assert _IL_SCALE_TOL == 0.10
    assert _IL_MIN_SEEDS == 3


def test_16_lesion_set_pins():
    assert _SHARED == ("no_binding", "no_shared_clock", "no_hippo_store")
    assert _HELPER_WM == ("no_bg_gate",)
    assert _HELPER_EP == ("no_sequencing", "no_cls_replay")
    assert _HELPER_BOTH == ("no_neuromod_timing",)
