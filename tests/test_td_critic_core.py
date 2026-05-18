import math
import pytest
from research.runners.td_critic_core import (
    tdc_verdict, _TDC_V1_VALUE_RMSE_MAX, _TDC_TRANSFER_MIN,
    _TDC_US_DECAY_MAX, _TDC_MIN_SEEDS)


def _sound_seed():  # a single seed's sound+passing payload
    return dict(vrmse=0.004, transfer=0.997, us_decay=0.001,
                controls={"no_bootstrap": (180.0, 0.20, 400.0),
                          "permuted": (0.2, 0.07, 0.96),
                          "wrongsign": (float("nan"),) * 3})


def test_frozen_bars_exact():
    assert _TDC_V1_VALUE_RMSE_MAX == 0.05
    assert _TDC_TRANSFER_MIN == 0.90
    assert _TDC_US_DECAY_MAX == 0.15
    assert _TDC_MIN_SEEDS == 3


def test_pass_when_sound_and_science():
    v = tdc_verdict({42: _sound_seed(), 43: _sound_seed(),
                     44: _sound_seed()})
    assert v["GATE"] == "PASS" and v["instrument_valid"] is True


def test_v1_unmet_is_VOID_not_fail():
    s = _sound_seed(); s["vrmse"] = 0.5      # critic did NOT learn V*
    v = tdc_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "VOID" and v["instrument_valid"] is False


def test_control_passing_signature_is_VOID_not_pass():
    s = _sound_seed()
    s["controls"]["permuted"] = (0.01, 0.97, 0.001)  # control "passes"
    v = tdc_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "VOID"               # non-discriminating


def test_diverged_control_is_correctly_failed_not_void():
    s = _sound_seed()
    s["controls"]["wrongsign"] = (float("inf"), float("nan"), 1e9)
    v = tdc_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "PASS"               # diverged == correctly failed


def test_science_fail_when_sound_but_no_transfer():
    s = _sound_seed(); s["transfer"] = 0.40  # below frozen 0.90
    v = tdc_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "FAIL" and v["instrument_valid"] is True


def test_non_numeric_is_VOID_not_raise():
    s = _sound_seed(); s["transfer"] = "0.99"   # string must NOT pass
    v = tdc_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "VOID"


def test_fewer_than_min_seeds_is_VOID():
    v = tdc_verdict({42: _sound_seed()})
    assert v["GATE"] == "VOID"


def test_results_cannot_move_frozen_bars():
    before = (_TDC_V1_VALUE_RMSE_MAX, _TDC_TRANSFER_MIN)
    tdc_verdict({42: _sound_seed(), 43: _sound_seed(), 44: _sound_seed()})
    import research.runners.td_critic_core as c
    assert (c._TDC_V1_VALUE_RMSE_MAX, c._TDC_TRANSFER_MIN) == before


def test_unorderable_seed_keys_is_VOID_not_raise():
    v = tdc_verdict({42: _sound_seed(), "x": _sound_seed(),
                     43: _sound_seed()})
    assert v["GATE"] == "VOID" and v["instrument_valid"] is False


def test_non_sized_control_is_VOID_not_raise():
    s = _sound_seed()
    s["controls"]["no_bootstrap"] = 5   # non-sized -> treated missing
    v = tdc_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "VOID"
