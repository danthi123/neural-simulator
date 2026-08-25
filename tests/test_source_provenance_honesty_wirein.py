"""CI guards for the #129 source-provenance honesty production wire-in (board #129, Vikunja #137).

Board #129 is the de-risked (6-seed GO) opponent-comparator provenance monitor
(`research/runners/_laneC_source_provenance_opponent_derisk.py` /
`research/findings/2026-08-25-laneC-source-provenance-opponent-perceived-vs-generated-6seed-GO.md`). These
tests cover the PRODUCTION wire-in on top of it (`research/runners/source_provenance_honesty.py` +
`BrainConversationalAgent.known_fact_record` / `.reasoned_fact_record`): additive, default-off, moat-first,
and LOAD-BEARING (the judged label -- not a caller-supplied claim -- drives the reply text; lesioning the
monitor collapses the distinction).
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.runners.rf_phasor_composer import RFPhasorComposer

VOCAB = [f"w{i:03d}" for i in range(24)]


def _composer(seed=42, D=128):
    comp = RFPhasorComposer(seed=seed, D=D, vocab=list(VOCAB), trace=True)
    comp.store("w000", "w001", "w002")   # a directly-taught fact (PERCEIVED under the wire-in's mapping)
    comp.store("w003", "w004", "w005")   # chain hop 1
    comp.store("w005", "w006", "w007")   # chain hop 2 -> reason_chain('w003', ['w004','w006']) == 'w007'
    return comp


def _agent(*, enable_source_provenance_honesty=False, composer=None, seed=42, **config):
    comp = composer if composer is not None else _composer(seed=seed)
    return BrainConversationalAgent(
        seed=seed,
        concepts={w: None for w in VOCAB},
        composer=comp,
        enable_neural_render=False,
        defer_parser=True,
        enable_source_provenance_honesty=enable_source_provenance_honesty,
        source_provenance_honesty_config=config or None,
    )


def test_default_off_is_byte_identical_to_the_pre_existing_answer_text():
    ag_off = _agent(enable_source_provenance_honesty=False)
    rec = ag_off.known_fact_record(("w000", "w001"))
    assert ag_off.enable_source_provenance_honesty is False
    assert ag_off._source_provenance_monitor is None       # never built -> no substrate, no work done
    assert rec["answer_text"] == "w000 w001 w002."
    assert rec.get("provenance") is None

    rec_chain = ag_off.reasoned_fact_record("w003", ["w004", "w006"])
    assert rec_chain["answer_text"] == "w003 w004 w006 w007."
    assert rec_chain["answer"] == "w007"
    assert rec_chain.get("provenance") is None
    # reasoned_fact_record is a NEW method -- it does not exist to break, but its OWN default-off text must be
    # the plain composed sentence (no honesty flag) when the faculty is off.
    assert "I believe" not in rec_chain["answer_text"]


def test_hard_moat_stays_before_provenance_and_untouched():
    ag = _agent(enable_source_provenance_honesty=True)
    rec = ag.known_fact_record(("w010", "w011"))       # never stored -> hard abstain
    assert rec["hard_abstain"] is True
    assert rec["answer_text"] == "I don't know about that."
    assert rec["provenance"] is None
    assert ag._source_provenance_monitor is None            # a hard abstain never builds the monitor

    rec_chain = ag.reasoned_fact_record("w010", ["w011"])   # dead-end hop -> hard abstain
    assert rec_chain["hard_abstain"] is True
    assert rec_chain["answer_text"] == "I don't know about that."
    assert rec_chain["provenance"] is None


def test_perceived_and_generated_get_demonstrably_different_framing_driven_by_the_live_judge():
    ag = _agent(enable_source_provenance_honesty=True)

    rec_perceived = ag.known_fact_record(("w000", "w001"))
    rec_generated = ag.reasoned_fact_record("w003", ["w004", "w006"])

    # both are TRUE, non-abstained answers -- the difference is provenance framing, not correctness
    assert rec_perceived["hard_abstain"] is False
    assert rec_generated["hard_abstain"] is False

    # the judged label is read from the live spiking opponent comparator, not asserted by the call site
    assert rec_perceived["provenance"]["known"] is True
    assert rec_perceived["provenance"]["label"] == "perceived"
    assert rec_generated["provenance"]["known"] is True
    assert rec_generated["provenance"]["label"] == "generated"

    # the PERCEIVED case reads EXACTLY as the default-off text (the dominant, directly-taught case is unchanged)
    assert rec_perceived["answer_text"] == "w000 w001 w002."
    # the GENERATED case is FLAGGED
    assert "I believe" in rec_generated["answer_text"]
    assert "reasoned that myself" in rec_generated["answer_text"]
    assert rec_perceived["answer_text"] != rec_generated["answer_text"]


def test_lesioning_the_monitor_collapses_the_perceived_vs_generated_distinction():
    """The wire-in's own load-bearing control: lesion = the runner's OWN verified failing-direction anti-cheat
    (encode with the Hebbian plasticity gate shut). Across many independent (perceived, generated) fact keys the
    UN-lesioned monitor's judged label should track truth near-perfectly (matching the #129 6-seed GO's
    accuracy 1.000); the LESIONED monitor should be near chance -- i.e. the framing decision is driven by the
    LEARNED trace, not by a hardcoded branch."""
    from research.runners.source_provenance_honesty import (
        PROVENANCE_GENERATED,
        PROVENANCE_PERCEIVED,
        SourceProvenanceHonestyMonitor,
    )

    n_pairs = 10
    for lesion, expect_high_accuracy in ((False, True), (True, False)):
        mon = SourceProvenanceHonestyMonitor(seed=42, lesion=lesion)
        correct = 0
        for i in range(n_pairs):
            mon.encode_fact(("perc", i), PROVENANCE_PERCEIVED)
            mon.encode_fact(("gen", i), PROVENANCE_GENERATED)
        for i in range(n_pairs):
            if mon.judge_fact(("perc", i))["label"] == PROVENANCE_PERCEIVED:
                correct += 1
            if mon.judge_fact(("gen", i))["label"] == PROVENANCE_GENERATED:
                correct += 1
        acc = correct / (2 * n_pairs)
        if expect_high_accuracy:
            assert acc >= 0.9, f"un-lesioned accuracy too low: {acc}"
        else:
            # chance on a silent-pool host tie-break: expect roughly 0.5, well below the un-lesioned bar
            assert acc <= 0.75, f"lesioned accuracy suspiciously high (distinction should have collapsed): {acc}"


def test_unknown_key_never_fabricates_a_provenance_label():
    from research.runners.source_provenance_honesty import PROVENANCE_PERCEIVED, SourceProvenanceHonestyMonitor

    mon = SourceProvenanceHonestyMonitor(seed=42)
    mon.encode_fact(("known",), PROVENANCE_PERCEIVED)
    judged = mon.judge_fact(("never", "seen"))
    assert judged["known"] is False
    assert judged["label"] is None


def test_provenance_is_independent_of_self_schema_honesty():
    """The two faculties are orthogonal (source vs correctness-confidence); either may be on alone."""
    comp = _composer()
    ag = BrainConversationalAgent(
        seed=42,
        concepts={w: None for w in VOCAB},
        composer=comp,
        enable_neural_render=False,
        defer_parser=True,
        enable_self_schema_honesty=True,
        enable_source_provenance_honesty=True,
    )
    rec = ag.known_fact_record(("w000", "w001"))
    assert rec["self_schema_invoked"] is True
    assert rec["provenance"] is not None
    assert rec["provenance"]["label"] == "perceived"
