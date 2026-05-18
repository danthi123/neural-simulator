"""Pure CPU adversarial tests for the THREE-STATE (VOID/PASS/FAIL)
fair-scale verdict. Instrument-validity FIRST: oracle-broken or
non-discriminating => VOID (not FAIL, not PASS). Bars immutable;
strict-bool + non-numeric fail-closed to VOID."""
import pytest
from research.runners import dendritic_fair_core as c


def test_frozen_bars_exact():
    assert c._DFAIR_ORACLE_MIN == 0.95
    assert c._DFAIR_WRONGSIGN_MAX == 0.30
    assert c._DFAIR_CORRECT_MIN == 0.90
    assert c._DFAIR_GLOBALSCALAR_MAX == 0.30
    assert c._DFAIR_PERMUTED_MAX == 0.30
    assert c._DFAIR_ALIGN_MIN == 0.30
    assert c._DFAIR_MIN_SEEDS == 3


def _good(**kw):
    base = dict(oracle_heldout=0.97, correct_heldout=0.93,
                wrongsign_heldout=0.15, globalscalar_heldout=0.14,
                permuted_heldout=0.13, end_align_cos=0.55,
                biologically_local=True, has_controls=True)
    base.update(kw)
    return c.dfair_verdict(**base)


def test_pass_when_all_met():
    assert _good()["GATE"] == "PASS"


def test_oracle_below_min_is_VOID_not_fail():
    v = _good(oracle_heldout=0.80)
    assert v["GATE"] == "VOID" and v["instrument_valid"] is False


def test_wrongsign_rescued_is_VOID_not_fail():
    v = _good(wrongsign_heldout=0.85)
    assert v["GATE"] == "VOID" and v["instrument_valid"] is False


def test_not_biologically_local_is_VOID():
    assert _good(biologically_local=False)["GATE"] == "VOID"
    assert _good(biologically_local="yes")["GATE"] == "VOID"  # strict


def test_no_controls_is_VOID():
    assert _good(has_controls=False)["GATE"] == "VOID"
    assert _good(has_controls="true")["GATE"] == "VOID"        # strict


def test_non_finite_is_VOID():
    assert _good(correct_heldout=float("nan"))["GATE"] == "VOID"
    assert _good(oracle_heldout=float("inf"))["GATE"] == "VOID"


def test_string_numeric_is_VOID_not_raises():
    assert _good(correct_heldout="0.93")["GATE"] == "VOID"
    assert _good(oracle_heldout=None)["GATE"] == "VOID"


def test_instrument_valid_but_not_learned_is_FAIL():
    v = _good(correct_heldout=0.50)
    assert v["GATE"] == "FAIL" and v["instrument_valid"] is True


def test_instrument_valid_but_globalscalar_high_is_FAIL():
    assert _good(globalscalar_heldout=0.80)["GATE"] == "FAIL"


def test_instrument_valid_but_permuted_high_is_FAIL():
    assert _good(permuted_heldout=0.80)["GATE"] == "FAIL"


def test_instrument_valid_but_no_alignment_is_FAIL():
    assert _good(end_align_cos=0.05)["GATE"] == "FAIL"


def test_results_cannot_move_fixed_bars():
    c.dfair_verdict(9.9, 9.9, 9.9, 9.9, 9.9, 9.9, True, True)
    assert c._DFAIR_ORACLE_MIN == 0.95 and c._DFAIR_CORRECT_MIN == 0.90
    assert c._DFAIR_WRONGSIGN_MAX == 0.30


def test_aggregate_lt3_seeds_is_FAIL():
    one = [_good()]
    assert c.dfair_aggregate_multiseed(one)["GATE"] == "FAIL"


def test_aggregate_any_void_is_VOID():
    seeds = [_good(), _good(), _good(oracle_heldout=0.5)]
    assert c.dfair_aggregate_multiseed(seeds)["GATE"] == "VOID"


def test_aggregate_all_valid_all_pass_is_PASS():
    assert c.dfair_aggregate_multiseed(
        [_good(), _good(), _good()])["GATE"] == "PASS"


def test_aggregate_valid_but_a_fail_is_FAIL():
    seeds = [_good(), _good(), _good(correct_heldout=0.5)]
    assert c.dfair_aggregate_multiseed(seeds)["GATE"] == "FAIL"


def test_aggregate_malformed_seed_is_VOID_not_raises():
    # non-dict / missing-GATE entries must fail-closed to VOID,
    # never raise (robustness hardening).
    assert c.dfair_aggregate_multiseed([None, None, None]
                                       )["GATE"] == "VOID"
    assert c.dfair_aggregate_multiseed([3.0, 3.0, 3.0]
                                       )["GATE"] == "VOID"
    assert c.dfair_aggregate_multiseed(
        [{"GATE": "PASS"}, {"GATE": "PASS"}, {"no_gate": 1}]
    )["GATE"] == "VOID"


def test_aggregate_void_precedes_insufficient_seeds():
    # an under-replicated run that ALSO contains a broken-instrument
    # (VOID) seed must report VOID (instrument-invalid precedes the
    # science verdict), not FAIL.
    v = c.dfair_aggregate_multiseed([_good(), _good(oracle_heldout=0.5)])
    assert v["GATE"] == "VOID"
    # a clean under-replicated run with NO void seed is still FAIL:
    assert c.dfair_aggregate_multiseed([_good(), _good()]
                                       )["GATE"] == "FAIL"
