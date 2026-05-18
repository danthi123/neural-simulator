from research.runners.compose_bind_core import (
    ctb_verdict, _CTB_V1_ACC_MIN, _CTB_SCIENCE_ACC_MIN,
    _CTB_CONTROL_ACC_MAX, _CTB_MIN_SEEDS)


def _sound():
    return dict(nogap_td=1.0, td=1.0,
                controls={"hebbian_no_trace": 0.083,
                          "permuted": 0.083, "wrongsign": 0.0})


def test_frozen_bars_exact():
    assert _CTB_V1_ACC_MIN == 0.90
    assert _CTB_SCIENCE_ACC_MIN == 0.90
    assert _CTB_CONTROL_ACC_MAX == 0.35
    assert _CTB_MIN_SEEDS == 3


def test_pass_when_sound_and_science():
    v = ctb_verdict({42: _sound(), 43: _sound(), 44: _sound()})
    assert v["GATE"] == "PASS" and v["instrument_valid"] is True


def test_v1_unmet_is_VOID_not_fail():
    s = _sound(); s["nogap_td"] = 0.4
    v = ctb_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "VOID" and v["instrument_valid"] is False


def test_control_learned_is_VOID_not_pass():
    s = _sound(); s["controls"]["hebbian_no_trace"] = 0.95
    v = ctb_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "VOID"


def test_diverged_control_is_correctly_failed_not_void():
    s = _sound(); s["controls"]["wrongsign"] = float("nan")
    v = ctb_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "PASS"


def test_science_fail_when_sound_but_td_low():
    s = _sound(); s["td"] = 0.4
    v = ctb_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "FAIL" and v["instrument_valid"] is True


def test_non_numeric_is_VOID_not_raise():
    s = _sound(); s["td"] = "0.99"
    v = ctb_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "VOID"


def test_missing_control_is_VOID():
    s = _sound(); del s["controls"]["permuted"]
    v = ctb_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "VOID"


def test_fewer_than_min_seeds_is_VOID():
    assert ctb_verdict({42: _sound()})["GATE"] == "VOID"


def test_unorderable_keys_is_VOID_not_raise():
    v = ctb_verdict({42: _sound(), "x": _sound(), 43: _sound()})
    assert v["GATE"] == "VOID"


def test_results_cannot_move_frozen_bars():
    before = (_CTB_V1_ACC_MIN, _CTB_SCIENCE_ACC_MIN)
    ctb_verdict({42: _sound(), 43: _sound(), 44: _sound()})
    import research.runners.compose_bind_core as c
    assert (c._CTB_V1_ACC_MIN, c._CTB_SCIENCE_ACC_MIN) == before
