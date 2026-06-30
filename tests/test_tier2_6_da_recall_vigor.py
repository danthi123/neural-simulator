"""CI guard for the Tier-2 #6 DA-gated RECALL-VIGOR de-risk (research/runners/_tier2_6_da_recall_vigor_derisk.py).

The mechanism (deep-research scoping 2026-06-30): a value/salience PRIOR carried by the shared spiking dopamine
RE-RANKS WHICH stored fact wins the conversational composer's cue-match scan (recall vigor / drift-rate; Niv-2007
tonic-DA vigor, O.19 value scales the accumulator drift rate, Lisman-Grace). The prior re-ranks ONLY within the
familiarity-gated (no-confab-passing) candidate set -> MOAT-SAFE BY CONSTRUCTION (it cannot create a false-accept).

This guard pins the WIRING + the four anti-cheats on the CPU/numpy oracle (RFPhasorComposer):
  - DA-driven re-rank: the high-value fact wins the value-conflicted cue at high DA;
  - DA-LESION (DA at baseline): the value-driven pick collapses;
  - EQUAL-value: the prior is neutral (validate-by-function discriminator);
  - PERMUTED value->fact: the recalled fact FOLLOWS the value;
  - MOAT (HARD): an UNSTORED cue still abstains at every DA level (the prior re-ranks only within the gated set).

CPU/numpy (the RFPhasorComposer test oracle); no GPU required. The decisive multi-seed claim is the GPU 6-seed on
the merged bridge with the REAL shared dopamine (the runner's default path); this guard pins the read-layer plumbing
+ that the controls compute + point the right way at toy scale.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._tier2_6_da_recall_vigor_derisk import (  # noqa: E402
    DARecallVigorComposer, build_conflict_composer, VOCAB, FACT_HI, FACT_LO, UNSTORED_CUE, CUE_ROLE)


def _da_const(value):
    return (lambda: float(value))


def test_value_prior_reranks_within_gated_set_at_high_da():
    """At high DA the value prior re-ranks the value-conflicted cue's candidate set -> the HIGH-value fact's patient
    is recalled; the candidate set is exactly the familiarity-gated (cue-role-matching) facts."""
    c = build_conflict_composer(seed=42, D=64, value_hi=1.0, value_lo=0.0, beta=8.0,
                                da_fn=_da_const(0.84), da_baseline=0.5)
    # both facts share CUE_ROLE -> both clear familiarity for the cue; the prior picks the HI fact
    ans = c.valued_recall(FACT_HI[CUE_ROLE])
    assert ans == FACT_HI["patient"], f"high-DA value prior should recall the HI-value patient, got {ans!r}"
    # the candidate set is the gated set: exactly the two facts whose cue role matches
    cand = c.candidate_indices(FACT_HI[CUE_ROLE])
    assert sorted(cand) == [0, 1], f"candidate set should be the two cue-matching facts, got {cand}"


def test_da_lesion_makes_recall_value_invariant():
    """DA-LESION (DA held at baseline -> beta*(DA-baseline)*value contribution -> 0): the recall is INVARIANT to the
    value assignment -- normal (HI fact = high value) and permuted (LO fact = high value) give the SAME answer. The
    prior is the ONLY thing that made the read value-sensitive; lesioning it removes all value sensitivity. This is the
    content-controlled, geometry-robust lesion control (it does NOT assume the two facts have equal intrinsic match
    scores)."""
    les_normal = build_conflict_composer(seed=42, D=64, value_hi=1.0, value_lo=0.0, beta=8.0,
                                         da_fn=_da_const(0.5), da_baseline=0.5)   # DA == baseline -> prior off
    les_permuted = build_conflict_composer(seed=42, D=64, value_hi=0.0, value_lo=1.0, beta=8.0,
                                           da_fn=_da_const(0.5), da_baseline=0.5)
    assert les_normal.valued_recall(FACT_HI[CUE_ROLE]) == les_permuted.valued_recall(FACT_HI[CUE_ROLE]), \
        "under DA-lesion the recall must be INVARIANT to the value assignment (the value sensitivity is gone)"
    # and it must NOT be the value-driven (high-DA) winner -- the value-driven advantage genuinely vanished
    hi = build_conflict_composer(seed=42, D=64, value_hi=1.0, value_lo=0.0, beta=8.0,
                                 da_fn=_da_const(0.84), da_baseline=0.5)
    perm = build_conflict_composer(seed=42, D=64, value_hi=0.0, value_lo=1.0, beta=8.0,
                                   da_fn=_da_const(0.84), da_baseline=0.5)
    assert hi.valued_recall(FACT_HI[CUE_ROLE]) != perm.valued_recall(FACT_HI[CUE_ROLE]), \
        "at high DA the value assignment DOES change the answer (the prior is value-sensitive)"


def test_equal_value_is_neutral():
    """EQUAL-value discriminator: when both facts carry equal value, the prior contributes equally -> it is NEUTRAL
    (no spurious bias); the recall equals the DA-lesion (prior-off) answer even at high DA. Validate-by-function: this
    proves the lesion's effect is value-SPECIFIC, not a general lesion artifact."""
    eq = build_conflict_composer(seed=42, D=64, value_hi=0.5, value_lo=0.5, beta=8.0,
                                 da_fn=_da_const(0.84), da_baseline=0.5)
    lesion = build_conflict_composer(seed=42, D=64, value_hi=1.0, value_lo=0.0, beta=8.0,
                                     da_fn=_da_const(0.5), da_baseline=0.5)   # prior off
    assert eq.valued_recall(FACT_HI[CUE_ROLE]) == lesion.valued_recall(FACT_HI[CUE_ROLE]), \
        "equal-value -> the prior is neutral -> the recall equals the prior-off (lesion) pick"


def test_permuted_value_follows_the_value():
    """PERMUTED value->fact: at high DA, swap which fact is high-value -> the recalled fact FOLLOWS the value
    (normal->HI patient, permuted->LO patient). A content/match-score-driven read could NOT flip -- so this is the
    content-controlling test."""
    normal = build_conflict_composer(seed=42, D=64, value_hi=1.0, value_lo=0.0, beta=8.0,
                                     da_fn=_da_const(0.84), da_baseline=0.5)
    permuted = build_conflict_composer(seed=42, D=64, value_hi=0.0, value_lo=1.0, beta=8.0,
                                       da_fn=_da_const(0.84), da_baseline=0.5)   # the LO fact is now high-value
    assert normal.valued_recall(FACT_HI[CUE_ROLE]) == FACT_HI["patient"]
    assert permuted.valued_recall(FACT_HI[CUE_ROLE]) == FACT_LO["patient"], \
        "permuting the value assignment must flip which fact is recalled"


def test_moat_hard_unstored_cue_abstains_at_every_da():
    """MOAT (HARD): an UNSTORED cue returns None at BOTH DA levels -- the prior re-ranks ONLY within the familiarity-
    gated set, so an unstored cue has nothing to re-rank (an empty candidate set -> abstain). Moat-safe by
    construction."""
    for da in (0.5, 0.84):
        c = build_conflict_composer(seed=42, D=64, value_hi=1.0, value_lo=0.0, beta=8.0,
                                    da_fn=_da_const(da), da_baseline=0.5)
        assert c.candidate_indices(UNSTORED_CUE) == [], "unstored cue must yield an empty candidate set"
        assert c.valued_recall(UNSTORED_CUE) is None, f"moat breach: unstored cue not abstained at DA={da}"


def test_wrapper_is_a_thin_reuse_of_rfphasorcomposer():
    """The wrapper reuses RFPhasorComposer by composition (no sim/ edit): the inner composer holds the kb + does the
    on-substrate unbind/cleanup; the wrapper only adds the value-prior re-rank over the gated candidate set."""
    c = build_conflict_composer(seed=42, D=64, value_hi=1.0, value_lo=0.0, beta=8.0,
                                da_fn=_da_const(0.84), da_baseline=0.5)
    from research.runners.rf_phasor_composer import RFPhasorComposer
    assert isinstance(c.comp, RFPhasorComposer)
    assert len(c.comp.kb) == 2 and all(w in c.comp.words for w in VOCAB)
