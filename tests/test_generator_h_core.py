"""Pure CPU adversarial tests for the FIXED-bar Generator-H verdict.
Mirrors the generator_g_core adversarial discipline + the NEW
non-degeneracy bars (coverage + max-repeat). Bars are immutable to
results; the genuinely-open question (loop / under-coverage) FAILs."""
import pytest
from research.runners import generator_h_core as c


def test_frozen_bars_exact():
    assert c._GH_UNGROUNDED_ENTITY_MAX == 0.20
    assert c._GH_MIN_COVERAGE == 1.0
    assert c._GH_MAX_REPEAT == 0.50
    assert c._GH_MIN_GROUNDED_ANSWER_RATE == 0.5
    assert c._GH_MIN_SEEDS == 3


def test_ungrounded_entity_rate_matches_generator_g_definition():
    assert c.ungrounded_entity_rate("max is a big dog",
                                    "max is a big dog") == 0.0
    r = c.ungrounded_entity_rate("max is a big bob",
                                 "max is a big dog")
    assert abs(r - (1.0 / 3.0)) < 1e-9


def test_is_answered_anti_vacuous():
    assert c.is_answered("max big dog") is True
    assert c.is_answered("is a the and") is False
    assert c.is_answered("   . ,  ") is False
    assert c.is_answered("") is False


def test_coverage_all_present_is_one():
    assert c.coverage("the big max dog runs", "max dog") == 1.0


def test_coverage_missing_content_word_below_one():
    assert c.coverage("max is here", "max dog") == 0.5


def test_max_repeat_ngram_fraction_detects_loop():
    looped = "and fast and fast and fast and fast"
    assert c.max_repeat_ngram_fraction(looped) > 0.50
    clean = "max is a big friendly dog today"
    assert c.max_repeat_ngram_fraction(clean) <= 0.50


def _good(**kw):
    base = dict(abstain_on_ungrounded_rate=1.0,
                bare_moat_abstain_rate=1.0,
                grounded_answer_rate=1.0,
                mean_ungrounded_entity_rate=0.02,
                mean_coverage=1.0, mean_max_repeat=0.10,
                has_ungrounded_control=True)
    base.update(kw)
    return c.gh_verdict(**base)


def test_verdict_pass_when_all_bars_met():
    assert _good()["GATE"] == "PASS"


def test_always_abstain_fails():
    assert _good(grounded_answer_rate=0.0)["GATE"] == "FAIL"


def test_missing_control_fails_closed():
    assert _good(has_ungrounded_control=False)["GATE"] == "FAIL"


def test_vacuous_zero_bare_moat_fails_closed():
    assert _good(bare_moat_abstain_rate=0.0,
                 abstain_on_ungrounded_rate=0.0)["GATE"] == "FAIL"


def test_confabulation_below_bare_moat_fails():
    assert _good(abstain_on_ungrounded_rate=0.5,
                 bare_moat_abstain_rate=1.0)["GATE"] == "FAIL"


def test_unfaithful_fails():
    assert _good(mean_ungrounded_entity_rate=0.50)["GATE"] == "FAIL"


def test_under_coverage_fails():
    assert _good(mean_coverage=0.5)["GATE"] == "FAIL"


def test_loop_collapse_fails_even_if_faithful_and_covered():
    assert _good(mean_max_repeat=0.90)["GATE"] == "FAIL"


def test_aggregate_requires_three_seeds():
    one = [c.gh_verdict(1.0, 1.0, 1.0, 0.02, 1.0, 0.1, True)]
    one[0]["n_grounded"] = 1
    one[0]["n_ungrounded"] = 1
    assert c.gh_aggregate_multiseed(one)["GATE"] == "FAIL"


def test_aggregate_pass_three_good_seeds():
    seeds = []
    for _ in range(3):
        v = c.gh_verdict(1.0, 1.0, 1.0, 0.02, 1.0, 0.1, True)
        v["n_grounded"] = 1
        v["n_ungrounded"] = 1
        seeds.append(v)
    assert c.gh_aggregate_multiseed(seeds)["GATE"] == "PASS"


def test_results_cannot_move_fixed_bars():
    c.gh_verdict(0.0, 0.0, 0.0, 9.9, 0.0, 9.9, False)
    assert c._GH_UNGROUNDED_ENTITY_MAX == 0.20
    assert c._GH_MIN_COVERAGE == 1.0 and c._GH_MAX_REPEAT == 0.50


def test_non_finite_rate_args_fail_closed():
    """inf / -inf in ANY of the six rate args must FAIL closed --
    never a spurious PASS (e.g. -inf <= 0.20 -> faithful True;
    inf >= 1.0 -> covered True)."""
    for arg in ("abstain_on_ungrounded_rate", "bare_moat_abstain_rate",
                "grounded_answer_rate", "mean_ungrounded_entity_rate",
                "mean_coverage", "mean_max_repeat"):
        assert _good(**{arg: float("inf")})["GATE"] == "FAIL", arg
    assert _good(mean_ungrounded_entity_rate=float("-inf"))["GATE"] \
        == "FAIL"
    assert _good(mean_coverage=float("nan"))["GATE"] == "FAIL"


def test_max_repeat_exactly_at_bar_fails():
    """A fully-collapsed 3-token hard loop scores max-repeat == 0.50
    exactly. The strict comparator must FAIL it (0.50 < 0.50 is
    False). _GH_MAX_REPEAT literal stays 0.50 byte-unchanged."""
    assert _good(mean_max_repeat=0.50)["GATE"] == "FAIL"
