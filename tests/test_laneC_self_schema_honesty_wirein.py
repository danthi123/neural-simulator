"""CI guards for the Lane C self-schema honesty production wire-in."""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._fluidconv_graded_hedging import _build_stressed
from research.runners._communicable_turn_stageA_derisk import CommunicableTurn
from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.runners.rf_phasor_composer import RFPhasorComposer
from research.runners.self_schema_honesty import SelfSchemaHonestyMonitor


VOCAB = ["dog", "cat", "bird", "go", "come", "eat", "north", "south", "worm"]


def _agent(*, enable_self_schema_honesty=False, composer=None, vocab=None, **config):
    vocab = list(vocab or VOCAB)
    comp = composer or RFPhasorComposer(seed=42, D=64, vocab=vocab, trace=False)
    return BrainConversationalAgent(
        seed=42,
        concepts={w: None for w in vocab},
        composer=comp,
        enable_neural_render=False,
        defer_parser=True,
        enable_self_schema_honesty=enable_self_schema_honesty,
        self_schema_honesty_config=config or None,
    )


def test_default_off_raw_methods_are_unchanged():
    ag = _agent()
    ag.composer.store("dog", "go", "north", polarity="AFFIRM")
    ag.composer.store("cat", "come", "south", polarity="NEGATE")

    assert ag.enable_self_schema_honesty is False
    assert ag._self_schema_honesty is None
    assert ag.composer.trace is False
    assert ag.what_does("dog", "go") == "north"
    assert ag.who_does("come", "south") == "cat"
    assert ag.is_it_true("cat", "come", "south") == "no"
    assert ag.describe("dog") == "dog go north"
    assert ag.what_does("bird", "eat") is None

    rec = ag.known_fact_record(("dog", "go"))
    assert rec["band"] == "assert"
    assert rec["certain"] is True
    assert rec["self_schema_invoked"] is False


def test_hard_moat_stays_before_self_schema_honesty():
    ag = _agent(enable_self_schema_honesty=True)
    ag.composer.store("dog", "go", "north", polarity="AFFIRM")

    rec = ag.known_fact_record(("bird", "eat"))

    assert rec["hard_abstain"] is True
    assert rec["band"] == "MOAT"
    assert rec["answer_text"] == "I don't know about that."
    assert rec["self_schema_invoked"] is False
    assert ag._self_schema_honesty is None


def test_self_schema_monitor_tracks_input_confidence_and_lesions_collapse():
    intact = SelfSchemaHonestyMonitor(seed=42)
    lesion = SelfSchemaHonestyMonitor(seed=42, lesion_self_read=True)

    low = intact.read(0.30, familiar=True)
    high = intact.read(0.75, familiar=True)
    low_lesion = lesion.read(0.30, familiar=True)
    high_lesion = lesion.read(0.75, familiar=True)
    very_low = intact.read(0.29, familiar=True)

    assert high["self_schema_rate"] > low["self_schema_rate"]
    assert high["band"] == "assert"
    assert low["band"] in {"hedge", "soft_abstain"}
    assert very_low["band"] != "assert"
    assert high_lesion["self_schema_rate"] == low_lesion["self_schema_rate"] == 0.0


def test_familiar_but_wrong_recall_is_downgraded_not_claimed():
    comp, facts, unknown = _build_stressed(seed=100, D=16, n_facts=48, vocab_mode="synthetic")
    ag = _agent(enable_self_schema_honesty=True, composer=comp, vocab=comp.words)

    rows = []
    for a, v, gold in facts:
        rec = ag.known_fact_record((a, v))
        if rec["hard_abstain"]:
            continue
        rows.append((rec, gold))

    correct = [rec for rec, gold in rows if rec["raw_answer"] == gold]
    wrong = [rec for rec, gold in rows if rec["raw_answer"] != gold]
    low_conf_wrong = [rec for rec in wrong if rec["confidence_source"] is not None and rec["confidence_source"] < 0.48]

    assert correct
    assert wrong
    assert any(rec["band"] == "assert" and rec["certain"] for rec in correct)
    assert low_conf_wrong
    assert all(rec["band"] != "assert" and not rec["certain"] for rec in low_conf_wrong)
    assert all(str(rec["raw_answer"]) in rec["answer_text"] for rec in low_conf_wrong)

    hard_off = 0
    hard_on = 0
    for cue in unknown:
        if comp.query_patient(*cue) is None:
            hard_off += 1
        rec = ag.known_fact_record(cue)
        if rec["hard_abstain"]:
            hard_on += 1
            assert rec["self_schema_invoked"] is False
    assert hard_on == hard_off


def test_source_consistency_floor_downgrades_high_trace_wrong_recall():
    comp, facts, unknown = _build_stressed(seed=44, D=16, n_facts=48, vocab_mode="synthetic")
    ag = _agent(
        enable_self_schema_honesty=True,
        composer=comp,
        vocab=comp.words,
        confidence_source_mode="source_consistency_floor",
    )

    high_trace_wrong = []
    source_false_positive = []
    for a, v, gold in facts:
        rec = ag.known_fact_record((a, v))
        if rec["hard_abstain"]:
            continue
        ev = rec["confidence_evidence"]
        if rec["raw_answer"] == gold and ev["source_consistent"] is False:
            source_false_positive.append(rec)
        if (
            rec["raw_answer"] != gold
            and ev["raw_trace_confidence"] is not None
            and ev["raw_trace_confidence"] >= 0.55
        ):
            high_trace_wrong.append(rec)

    assert high_trace_wrong
    assert not source_false_positive
    assert all(rec["confidence_source"] == 0.0 for rec in high_trace_wrong)
    assert all(rec["band"] != "assert" and not rec["certain"] for rec in high_trace_wrong)
    assert all(rec["confidence_evidence"]["scaffold"] is True for rec in high_trace_wrong)
    assert all(rec["confidence_evidence"]["source_consistent"] is False for rec in high_trace_wrong)

    hard_cue = next(cue for cue in unknown if comp.query_patient(*cue) is None)
    hard = ag.known_fact_record(hard_cue)
    assert hard["band"] == "MOAT"
    assert hard["self_schema_invoked"] is False


def test_neural_source_consistency_uses_independent_echo_not_source_fact():
    comp, facts, unknown = _build_stressed(
        seed=44,
        D=16,
        n_facts=48,
        vocab_mode="synthetic",
        composer_kwargs={"enable_source_monitor": True, "source_monitor_D": 64},
    )
    ag = _agent(
        enable_self_schema_honesty=True,
        composer=comp,
        vocab=comp.words,
        confidence_source_mode="neural_source_consistency",
    )

    high_trace_wrong = []
    source_false_positive = []
    for a, v, gold in facts:
        rec = ag.known_fact_record((a, v))
        if rec["hard_abstain"]:
            continue
        ev = rec["confidence_evidence"]
        if rec["raw_answer"] == gold and ev["source_consistent"] is False:
            source_false_positive.append(rec)
        if (
            rec["raw_answer"] != gold
            and ev["raw_trace_confidence"] is not None
            and ev["raw_trace_confidence"] >= 0.55
        ):
            high_trace_wrong.append(rec)

    assert high_trace_wrong
    assert not source_false_positive
    assert all(rec["confidence_source"] == 0.0 for rec in high_trace_wrong)
    assert all(rec["band"] != "assert" and not rec["certain"] for rec in high_trace_wrong)
    assert all(
        rec["confidence_evidence"]["selected_consistency_source"] == "rf_independent_source_echo"
        for rec in high_trace_wrong
    )
    assert all(rec["confidence_evidence"]["scaffold"] is False for rec in high_trace_wrong)
    assert all(
        rec["confidence_evidence"]["neural_source_monitor"]["source"] == "rf_independent_source_echo"
        for rec in high_trace_wrong
    )

    hard_cue = next(cue for cue in unknown if comp.query_patient(*cue) is None)
    hard = ag.known_fact_record(hard_cue)
    assert hard["band"] == "MOAT"
    assert hard["self_schema_invoked"] is False


def test_neural_source_consistency_fails_closed_without_echo():
    comp = RFPhasorComposer(seed=42, D=64, vocab=VOCAB, trace=True)
    comp.store("dog", "go", "north", polarity="AFFIRM")
    ag = _agent(
        enable_self_schema_honesty=True,
        composer=comp,
        vocab=comp.words,
        confidence_source_mode="neural_source_consistency",
    )

    rec = ag.known_fact_record(("dog", "go"))

    assert rec["raw_answer"] == "north"
    assert rec["confidence_source"] == 0.0
    assert rec["band"] != "assert"
    assert rec["certain"] is False
    assert rec["confidence_evidence"]["selected_consistency_source"] == "rf_independent_source_echo"
    assert rec["confidence_evidence"]["neural_source_monitor"]["available"] is False
    assert rec["confidence_evidence"]["neural_source_monitor"]["source"] == "rf_independent_source_echo"
    assert rec["confidence_evidence"]["source_consistent"] is False


def test_plastic_source_consistency_requires_learning_and_feeds_self_schema():
    comp = RFPhasorComposer(
        seed=42,
        D=64,
        vocab=VOCAB,
        trace=True,
        enable_plastic_source_monitor=True,
        plastic_source_config={
            "n_banks": 4,
            "proposition_neurons_per_bank": 2048,
            "support_threshold": 0.25,
        },
    )
    comp.store("dog", "go", "north", polarity="AFFIRM")
    ag = _agent(
        enable_self_schema_honesty=True,
        composer=comp,
        vocab=comp.words,
        confidence_source_mode="plastic_source_consistency",
    )

    before = ag.known_fact_record(("dog", "go"))
    comp.observe_source_event(kind="what_does", cue=("dog", "go"), candidate="north")
    after = ag.known_fact_record(("dog", "go"))

    assert before["raw_answer"] == "north"
    assert before["confidence_source"] == 0.0
    assert before["band"] != "assert"
    assert after["confidence_evidence"]["source_consistent"] is True
    assert after["confidence_evidence"]["learned_source_association"] is True
    assert (
        after["confidence_evidence"]["selected_consistency_source"]
        == "plastic_hebbian_proposition_source"
    )
    assert after["confidence_source"] == after["confidence_evidence"]["raw_trace_confidence"]


def test_communicable_known_channel_uses_laneC_record_when_enabled():
    comp, facts, unknown = _build_stressed(seed=100, D=16, n_facts=48, vocab_mode="synthetic")
    ag = _agent(enable_self_schema_honesty=True, composer=comp, vocab=comp.words)
    turn = CommunicableTurn(
        comp,
        ag,
        proposer=None,
        accumulator=None,
        P=None,
        row={},
        vocab_sets=(set(), set(), set(), {}),
        faculty=None,
        value=None,
        codes={},
        full_pools=(set(), set(), set()),
    )

    low_conf_wrong = None
    for a, v, gold in facts:
        rec = ag.known_fact_record((a, v))
        if (not rec["hard_abstain"] and rec["raw_answer"] != gold
                and rec["confidence_source"] is not None and rec["confidence_source"] < 0.48):
            low_conf_wrong = (a, v, rec["raw_answer"])
            break
    assert low_conf_wrong is not None

    a, v, recalled = low_conf_wrong
    out = turn._known_fact_channel((a, v))
    assert out["channel"] == "known"
    assert out["abstained"] is False
    assert out["certain"] is False
    assert out["laneC_self_schema"]["enabled"] is True
    assert out["laneC_self_schema"]["band"] != "assert"
    assert recalled in out["answer"]

    hard_cue = next(cue for cue in unknown if comp.query_patient(*cue) is None)
    hard = turn._known_fact_channel(hard_cue)
    assert hard["abstained"] is True
    assert hard["answer"] == "I don't know about that."
    assert hard["laneC_self_schema"]["band"] == "MOAT"
    assert hard["laneC_self_schema"]["self_schema_invoked"] is False
