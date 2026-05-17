from research.runners.generator_g_core import (
    ungrounded_entity_rate, gg_verdict, gg_aggregate_multiseed,
    _GG_UNGROUNDED_ENTITY_MAX, _GG_MIN_GROUNDED_ANSWER_RATE,
    _GG_MIN_SEEDS, FUNCTION_WORDS,
)


def test_ungrounded_entity_rate_catches_renamed_entity():
    r = ungrounded_entity_rate("bob is a big dog",
                               "max is a big dog", FUNCTION_WORDS)
    assert r > 0.0
    assert ungrounded_entity_rate("max is a big dog",
                                  "max is a big dog",
                                  FUNCTION_WORDS) == 0.0


def test_verdict_passes_only_when_all_three_bars_met():
    v = gg_verdict(abstain_on_ungrounded_rate=1.0,
                   bare_moat_abstain_rate=1.0,
                   grounded_answer_rate=0.9,
                   mean_ungrounded_entity_rate=0.05,
                   has_ungrounded_control=True)
    assert v["GATE"] == "PASS"


def test_no_confab_regression_is_fail():
    v = gg_verdict(0.80, 1.0, 0.9, 0.0, True)
    assert v["GATE"] == "FAIL" and v["no_confab_preserved"] is False


def test_trivial_always_abstain_is_fail():
    v = gg_verdict(1.0, 1.0, 0.0, 0.0, True)
    assert v["GATE"] == "FAIL"


def test_unfaithful_generation_is_fail():
    v = gg_verdict(1.0, 1.0, 0.9, 0.55, True)
    assert v["GATE"] == "FAIL" and v["grounded_faithful"] is False


def test_missing_ungrounded_control_is_fail_closed():
    v = gg_verdict(1.0, 1.0, 0.9, 0.0, has_ungrounded_control=False)
    assert v["GATE"] == "FAIL"


def test_results_cannot_move_fixed_bars():
    assert (_GG_UNGROUNDED_ENTITY_MAX, _GG_MIN_GROUNDED_ANSWER_RATE,
            _GG_MIN_SEEDS) == (0.20, 0.5, 3)
    b = gg_verdict(1.0, 1.0, 1.0, 0.0, True)
    assert b["bars"] == {"ungrounded_entity_max": 0.20,
                         "min_grounded_answer_rate": 0.5}


def test_multiseed_requires_3_all_pass_each_has_both_probes():
    P = {"GATE": "PASS", "n_grounded": 4, "n_ungrounded": 4}
    F = {"GATE": "FAIL", "n_grounded": 4, "n_ungrounded": 4}
    Z = {"GATE": "PASS", "n_grounded": 0, "n_ungrounded": 4}
    assert gg_aggregate_multiseed([P, P, P])["GATE"] == "PASS"
    assert gg_aggregate_multiseed([P, F, P])["GATE"] == "FAIL"
    assert gg_aggregate_multiseed([P, P])["GATE"] == "FAIL"
    assert gg_aggregate_multiseed([P, P, Z])["GATE"] == "FAIL"
