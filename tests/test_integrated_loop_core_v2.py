"""Adversarial matrix for the NEW corrected-necessity verdict module
(`research.runners.integrated_loop_core_v2`).

Mirrors `tests/test_integrated_loop_core.py`'s 16-case structure,
adapted to `integrated_loop_verdict_v2` and the `_ILV2_*` symbols,
under the single biologically-cited partition correction
(`no_cls_replay` moves from the episodic-helper set to the
working-memory-helper set). PLUS two mandatory drift pins:

- test 17: every `_ILV2_*` numeric bar is byte-equal to the
  corresponding original `_IL_*` bar (fails loudly on any drift).
- test 18: the v2 partition differs from the original partition by
  EXACTLY the single documented change (`no_cls_replay`: EP->WM) and
  in NO other way.
"""
import math
import pytest

from research.runners.integrated_loop_core_v2 import (
    integrated_loop_verdict_v2,
    _ILV2_V1_MIN, _ILV2_SCI_MIN, _ILV2_LESION_MAX, _ILV2_SCALE_TOL,
    _ILV2_LADDER, _ILV2_MIN_SEEDS,
    _ILV2_SHARED, _ILV2_HELPER_WM, _ILV2_HELPER_EP, _ILV2_HELPER_BOTH,
    _ILV2_ALL_LESIONS)

from research.runners.integrated_loop_core import (
    _IL_V1_MIN, _IL_SCI_MIN, _IL_LESION_MAX, _IL_SCALE_TOL,
    _IL_LADDER, _IL_MIN_SEEDS,
    _SHARED, _HELPER_WM, _HELPER_EP, _HELPER_BOTH)


def _good_rung(N, full=0.9, n_seeds=5):
    """A rung where the full loop succeeds on BOTH readouts and EVERY
    lesion collapses exactly the readout(s) it is responsible for under
    the CORRECTED partition: SHARED + HELPER_BOTH collapse both;
    HELPER_WM (now including no_cls_replay) collapses wm; HELPER_EP
    (now only no_sequencing) collapses ep."""
    lesions = {}
    for name in _ILV2_SHARED:
        lesions[name] = {"wm": 0.2, "ep": 0.2}          # both collapse
    for name in _ILV2_HELPER_WM:
        lesions[name] = {"wm": 0.2, "ep": 0.9}          # wm collapses
    for name in _ILV2_HELPER_EP:
        lesions[name] = {"wm": 0.9, "ep": 0.2}          # ep collapses
    for name in _ILV2_HELPER_BOTH:
        lesions[name] = {"wm": 0.2, "ep": 0.2}          # both collapse
    return {"N": N, "n_seeds": n_seeds,
            "v1": {"wm": 0.95, "ep": 0.95},
            "full": {"wm": full, "ep": full},
            "lesions": lesions}


def test_01_scale_confident_pass():
    rungs = [_good_rung(2, 0.86), _good_rung(4, 0.88), _good_rung(8, 0.90)]
    r = integrated_loop_verdict_v2(rungs)
    assert r["GATE"] == "PASS"
    assert r["classification"] == "SCALE-CONFIDENT-PASS"


def test_02_works_small_trend_break():
    rungs = [_good_rung(2, 0.95), _good_rung(4, 0.95), _good_rung(8, 0.81)]
    rungs[2]["full"] = {"wm": 0.81, "ep": 0.66}
    r = integrated_loop_verdict_v2(rungs)
    assert r["GATE"] == "FAIL"
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_03_works_small_top_below_floor():
    rungs = [_good_rung(2, 0.86), _good_rung(4, 0.84), _good_rung(8, 0.82)]
    rungs[2]["full"] = {"wm": 0.78, "ep": 0.78}
    r = integrated_loop_verdict_v2(rungs)
    assert r["GATE"] == "FAIL"
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_04_shared_lesion_does_not_collapse_wm_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[1]["lesions"][_ILV2_SHARED[0]] = {"wm": 0.85, "ep": 0.2}
    r = integrated_loop_verdict_v2(rungs)
    assert r["GATE"] == "VOID"
    assert r["instrument_valid"] is False


def test_05_shared_lesion_collapses_wm_but_not_ep_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[0]["lesions"][_ILV2_SHARED[1]] = {"wm": 0.2, "ep": 0.88}
    r = integrated_loop_verdict_v2(rungs)
    assert r["GATE"] == "VOID"
    assert r["instrument_valid"] is False


def test_06_helper_wm_no_bg_gate_does_not_collapse_wm_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[2]["lesions"]["no_bg_gate"] = {"wm": 0.9, "ep": 0.9}
    r = integrated_loop_verdict_v2(rungs)
    assert r["GATE"] == "VOID"


def test_07_helper_wm_no_cls_replay_does_not_collapse_wm_is_void():
    # THE CORRECTED-MEMBERSHIP CASE: no_cls_replay with wm high (not
    # collapsed) -> VOID. Under the original module this same numeric
    # record was scored against the episodic readout; this pins the
    # corrected behavior.
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[1]["lesions"]["no_cls_replay"] = {"wm": 0.9, "ep": 0.2}
    r = integrated_loop_verdict_v2(rungs)
    assert r["GATE"] == "VOID"
    assert r["instrument_valid"] is False


def test_08_helper_ep_no_sequencing_does_not_collapse_ep_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[0]["lesions"]["no_sequencing"] = {"wm": 0.9, "ep": 0.9}
    r = integrated_loop_verdict_v2(rungs)
    assert r["GATE"] == "VOID"


def test_09_no_cls_replay_collapsing_only_ep_not_wm_is_void():
    # EXPLICIT GUARD / the precise behavioral fingerprint of the single
    # correction: no_cls_replay collapses ONLY the episodic readout
    # (ep <= 0.40) and leaves working-memory intact (wm high) -> VOID,
    # because in the corrected partition no_cls_replay is a
    # WORKING-MEMORY helper and MUST collapse wm. This record would have
    # been a (spurious) "satisfied" under the original partition.
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[2]["lesions"]["no_cls_replay"] = {"wm": 0.9, "ep": 0.2}
    r = integrated_loop_verdict_v2(rungs)
    assert r["GATE"] == "VOID"
    assert r["instrument_valid"] is False


def test_10_helper_both_collapses_only_one_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[1]["lesions"][_ILV2_HELPER_BOTH[0]] = {"wm": 0.2, "ep": 0.9}
    r = integrated_loop_verdict_v2(rungs)
    assert r["GATE"] == "VOID"


def test_11_v1_unmet_is_void_not_fail():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[0]["v1"] = {"wm": 0.55, "ep": 0.95}
    r = integrated_loop_verdict_v2(rungs)
    assert r["GATE"] == "VOID"
    assert r["instrument_valid"] is False


def test_12_sound_discriminating_but_science_below_bar_is_fail():
    rungs = [_good_rung(2, 0.70), _good_rung(4, 0.70), _good_rung(8, 0.70)]
    r = integrated_loop_verdict_v2(rungs)
    assert r["GATE"] == "FAIL"
    assert r["instrument_valid"] is True
    assert r["classification"] == "FAIL"


def test_13_ladder_mismatch_is_void():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(4)]
    r = integrated_loop_verdict_v2(rungs)
    assert r["GATE"] == "VOID"


def test_14_non_numeric_and_nan_is_void_not_raise():
    rungs = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs[1]["full"] = {"wm": "0.9", "ep": 0.9}
    assert integrated_loop_verdict_v2(rungs)["GATE"] == "VOID"
    rungs2 = [_good_rung(2), _good_rung(4), _good_rung(8)]
    rungs2[2]["full"] = {"wm": float("nan"), "ep": 0.9}
    assert integrated_loop_verdict_v2(rungs2)["GATE"] == "VOID"


def test_15_too_few_seeds_is_void():
    rungs = [_good_rung(2, n_seeds=2), _good_rung(4), _good_rung(8)]
    r = integrated_loop_verdict_v2(rungs)
    assert r["GATE"] == "VOID"


def test_16_malformed_top_level_is_void_not_raise():
    assert integrated_loop_verdict_v2(None)["GATE"] == "VOID"
    assert integrated_loop_verdict_v2([])["GATE"] == "VOID"
    assert integrated_loop_verdict_v2("garbage")["GATE"] == "VOID"
    assert integrated_loop_verdict_v2([{"no": "N"}])["GATE"] == "VOID"


def test_17_threshold_tamper_pins_VERBATIM_EQUAL_TO_ORIGINAL():
    # Every v2 bar must be byte-equal to its original counterpart AND
    # to the literal pre-registered value. Fails loudly on any drift.
    assert _ILV2_LADDER == _IL_LADDER == (2, 4, 8)
    assert _ILV2_V1_MIN == _IL_V1_MIN == 0.90
    assert _ILV2_SCI_MIN == _IL_SCI_MIN == 0.80
    assert _ILV2_LESION_MAX == _IL_LESION_MAX == 0.40
    assert _ILV2_SCALE_TOL == _IL_SCALE_TOL == 0.10
    assert _ILV2_MIN_SEEDS == _IL_MIN_SEEDS == 3


def test_18_partition_has_exactly_one_documented_change_vs_original():
    # (a) SHARED unchanged.
    assert _ILV2_SHARED == _SHARED
    # (b) HELPER_BOTH unchanged.
    assert _ILV2_HELPER_BOTH == _HELPER_BOTH
    # (c) the SET of all lesion names is identical. The original
    #     frozen module's four partition tuples define the canonical
    #     lesion name set (reconstructed exactly as the plan's test_18
    #     specifies); v2 must reproduce that set exactly with only one
    #     membership moved. The cardinality is whatever the frozen
    #     original defines (it is NOT independently asserted here -- the
    #     frozen original is authoritative and is never contradicted).
    orig_all = set(_SHARED) | set(_HELPER_WM) | set(_HELPER_EP) | set(
        _HELPER_BOTH)
    assert set(_ILV2_ALL_LESIONS) == orig_all
    # (d) EXACTLY the name "no_cls_replay" changed helper set: it is in
    #     _ILV2_HELPER_WM and NOT in _ILV2_HELPER_EP, while in the
    #     original it is in _HELPER_EP and NOT in _HELPER_WM.
    assert "no_cls_replay" in _ILV2_HELPER_WM
    assert "no_cls_replay" not in _ILV2_HELPER_EP
    assert "no_cls_replay" in _HELPER_EP
    assert "no_cls_replay" not in _HELPER_WM
    # (e) the symmetric difference on each affected helper set is
    #     exactly {"no_cls_replay"}.
    assert (set(_HELPER_WM) ^ set(_ILV2_HELPER_WM)) == {"no_cls_replay"}
    assert (set(_HELPER_EP) ^ set(_ILV2_HELPER_EP)) == {"no_cls_replay"}
    # (f) the documented corrected memberships, exactly.
    assert set(_ILV2_HELPER_WM) == set(_HELPER_WM) | {"no_cls_replay"}
    assert set(_ILV2_HELPER_EP) == set(_HELPER_EP) - {"no_cls_replay"}
    # the union of all four v2 tuples == the union of all four
    # original tuples (same lesion names; only one membership moved).
    v2_all = set(_ILV2_SHARED) | set(_ILV2_HELPER_WM) | set(
        _ILV2_HELPER_EP) | set(_ILV2_HELPER_BOTH)
    assert v2_all == orig_all
