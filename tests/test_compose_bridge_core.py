from research.runners.compose_bridge_core import (
    cbr_verdict, _CBR_V1_ACC_MIN, _CBR_SCI_ACC_MIN,
    _CBR_CTRL_ACC_MAX, _CBR_MIN_SEEDS)


def _sound():
    return dict(nogap_td=0.95, td=0.92,
                controls={"hebbian_no_trace": 0.10,
                          "permuted": 0.10, "wrongsign": 0.0})


def test_frozen_bars_exact():
    assert _CBR_V1_ACC_MIN == 0.80
    assert _CBR_SCI_ACC_MIN == 0.80
    assert _CBR_CTRL_ACC_MAX == 0.35
    assert _CBR_MIN_SEEDS == 3


def test_pass_when_sound_and_science():
    v = cbr_verdict({42: _sound(), 43: _sound(), 44: _sound()})
    assert v["GATE"] == "PASS" and v["instrument_valid"] is True


def test_v1_unmet_is_VOID_not_fail():
    s = _sound(); s["nogap_td"] = 0.5
    assert cbr_verdict({42: s, 43: s, 44: s})["GATE"] == "VOID"


def test_control_learned_is_VOID_not_pass():
    s = _sound(); s["controls"]["hebbian_no_trace"] = 0.85
    assert cbr_verdict({42: s, 43: s, 44: s})["GATE"] == "VOID"


def test_diverged_control_is_correctly_failed_not_void():
    s = _sound(); s["controls"]["wrongsign"] = float("nan")
    assert cbr_verdict({42: s, 43: s, 44: s})["GATE"] == "PASS"


def test_non_numeric_junk_control_is_VOID_not_fabricated_pass():
    s = _sound(); s["controls"]["permuted"] = "0.10"
    assert cbr_verdict({42: s, 43: s, 44: s})["GATE"] == "VOID"


def test_science_fail_when_sound_but_td_low():
    s = _sound(); s["td"] = 0.5
    v = cbr_verdict({42: s, 43: s, 44: s})
    assert v["GATE"] == "FAIL" and v["instrument_valid"] is True


def test_non_numeric_science_is_VOID_not_raise():
    s = _sound(); s["td"] = "0.99"
    assert cbr_verdict({42: s, 43: s, 44: s})["GATE"] == "VOID"


def test_missing_control_is_VOID():
    s = _sound(); del s["controls"]["permuted"]
    assert cbr_verdict({42: s, 43: s, 44: s})["GATE"] == "VOID"


def test_fewer_than_min_seeds_is_VOID():
    assert cbr_verdict({42: _sound()})["GATE"] == "VOID"


def test_unorderable_keys_is_VOID_not_raise():
    v = cbr_verdict({42: _sound(), "x": _sound(), 43: _sound()})
    assert v["GATE"] == "VOID"


def test_results_cannot_move_frozen_bars():
    before = (_CBR_V1_ACC_MIN, _CBR_SCI_ACC_MIN)
    cbr_verdict({42: _sound(), 43: _sound(), 44: _sound()})
    import research.runners.compose_bridge_core as c
    assert (c._CBR_V1_ACC_MIN, c._CBR_SCI_ACC_MIN) == before
