"""A2 abstention-congruence production gate (2026-09-25 amendment to research/findings/2026-09-24-affect-marker-
settle-flip-criteria-AMENDMENT-PREREG.md; biology: research/biology/affective-marker-abstention-congruence-gate.md).

Pure data-level checks of `webapp.affect_drives_chat.congruence_gate` — no brain build, no HTTP call. Proves:
  * default OFF (`BRAIN_AFFECT_MARKER_CONGRUENCE` unset) is an exact passthrough (byte-identical lead, no trace) —
    the additive/byte-identical-off contract every other coupling in this module carries.
  * ON withholds the marker on an abstention conflict and on a Gate-B valence-sign mismatch, and leaves a
    congruent marker untouched.
  * a missing/None Gate-B read never fabricates a valence conflict (mirrors the research runner's A2.2 rule).
  * the function never raises.
"""
import os

import pytest

from webapp import affect_drives_chat as adc

ENV = adc.CONGRUENCE_ENV


@pytest.fixture(autouse=True)
def _clean_env():
    prior = os.environ.pop(ENV, None)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = prior


def test_default_off_is_exact_passthrough_regardless_of_conflict():
    # flag unset -> byte-identical: the lead is returned UNCHANGED and no trace is attached, even though this
    # turn is BOTH an abstention conflict AND a valence mismatch (the strongest case the gate could ever fire on).
    lead = "Wonderful! "
    out_lead, trace = adc.congruence_gate(lead, abstained=True, gateb_affect_info={"valence_sign": "-"})
    assert out_lead == lead
    assert trace is None
    assert adc.congruence_gate_enabled() is False


@pytest.mark.parametrize("off_value", ["0", "false", "off", "no", ""])
def test_explicit_off_values_are_passthrough(off_value):
    os.environ[ENV] = off_value
    lead = "Gladly! "
    out_lead, trace = adc.congruence_gate(lead, abstained=True, gateb_affect_info=None)
    assert out_lead == lead
    assert trace is None


def test_empty_lead_is_always_passthrough_even_when_on():
    os.environ[ENV] = "1"
    out_lead, trace = adc.congruence_gate("", abstained=True, gateb_affect_info={"valence_sign": "-"})
    assert out_lead == ""
    assert trace is None


def test_on_abstention_conflict_withholds_the_marker():
    os.environ[ENV] = "1"
    out_lead, trace = adc.congruence_gate("Gladly! ", abstained=True, gateb_affect_info=None)
    assert out_lead == ""
    assert trace["suppressed"] is True
    assert trace["abstention_conflict"] is True
    assert trace["valence_conflict"] is False
    assert trace["reason"] == "abstention"


def test_on_valence_conflict_withholds_the_marker():
    os.environ[ENV] = "1"
    # a POSITIVE register ("Wonderful") while Gate-B's independent read is NEGATIVE -> conflict.
    out_lead, trace = adc.congruence_gate("Wonderful! ", abstained=False, gateb_affect_info={"valence_sign": "-"})
    assert out_lead == ""
    assert trace["suppressed"] is True
    assert trace["abstention_conflict"] is False
    assert trace["valence_conflict"] is True
    assert trace["reason"] == "valence_mismatch"


def test_on_congruent_marker_passes_through_unchanged():
    os.environ[ENV] = "1"
    out_lead, trace = adc.congruence_gate("Gladly! ", abstained=False, gateb_affect_info={"valence_sign": "+"})
    assert out_lead == "Gladly! "
    assert trace["suppressed"] is False
    assert trace["incongruent"] is False


@pytest.mark.parametrize("gateb", [None, {}, {"valence_sign": None}, {"valence_sign": "0"}])
def test_on_missing_or_neutral_gateb_read_never_fabricates_a_conflict(gateb):
    # A2.2 (carried over from the research runner): never manufacture a disagreement when there is nothing to
    # disagree WITH (Gate-B off, or its own read is the neutral "0").
    os.environ[ENV] = "1"
    out_lead, trace = adc.congruence_gate("Gladly! ", abstained=False, gateb_affect_info=gateb)
    assert out_lead == "Gladly! "
    assert trace["valence_conflict"] is False
    assert trace["suppressed"] is False


def test_never_raises_on_a_malformed_gateb_read():
    os.environ[ENV] = "1"
    # gateb_affect_info that is not dict-shaped in the way .get() expects -- the gate must degrade, not crash.
    out_lead, trace = adc.congruence_gate("Gladly! ", abstained=False, gateb_affect_info="not-a-dict")
    assert out_lead in ("Gladly! ", "")
    assert trace is not None


def test_register_word_strips_emphasis_punctuation():
    assert adc._register_word("Wonderful! ") == "Wonderful"
    assert adc._register_word("Honestly — ") == "Honestly"
    assert adc._register_word("") == ""


def test_register_sign_matches_the_production_lead_word_table():
    for word in ("Wonderful", "Gladly", "Sure"):
        assert adc._register_sign(word) == 1
    for word in ("Hm", "Honestly", "Frankly"):
        assert adc._register_sign(word) == -1
    assert adc._register_sign("") == 0
