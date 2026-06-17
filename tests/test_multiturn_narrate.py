"""Production gate for `MultiTurnAgentV2.narrate(topics)` -- COHERENT MULTI-SENTENCE NARRATION on the spiking
resonate-and-fire phasor substrate. Promotes two separately-validated, multi-seed-GO de-risk mechanisms into the
production agent as one method:

  * ORDERED EMISSION  (2026-06-17-multisentence-ordered-emission-derisk.md, GO 6/6): hold an ordered topic list in
    the order-encoded WM (gamma-slot POSITION phasors on the RF substrate), emit one sentence per slot IN SLOT
    ORDER -- so re-ordering the topics re-orders the output (the order is order-encoded, not a fixed storage order).
  * CROSS-SENTENCE COHERENCE (2026-06-17-cross-sentence-coherence-derisk.md, GO 6/6): a recurring subject is
    rendered as a PRONOUN ("it") that RESOLVES (validated by-slot slot-anaphora, `referent_at(antecedent_slot)`)
    back to the correct ANTECEDENT referent (the EARLIEST slot it occupied, not the most-recent).

THE FOUR LOAD-BEARING CONTROLS (each scored across 6 seeds 42 43 44 100 101 102), with the de-risk's FROZEN bars:
  1. ORDERED NARRATION   -- narrate([t0,t1,t2]) emits the 3 facts as correctly-ordered sentences (exact per topic).
  2. COHERENCE           -- a recurring topic is pronominalized AND the pronoun resolves to the correct antecedent.
  3. ORDER-CONTROL       -- permuting the topics permutes the sentences; swapping which referent recurs FLIPS the
                            resolved antecedent. (Load-bearing: a storage-order dump / fixed-entity resolver fails.)
  4. NO-CONFAB MOAT      -- a topic with NO stored fact -> abstain/skip, never a confabulated sentence.
Plus a side-effect-free check (a narration does not perturb an in-progress multi-turn dialogue) and a small fixed
transcript assertion (the exact validated surface string at seed 42).

GO bar: each control passes in >= ceil(5/6 * n_seeds) of the seeds run (a FRACTIONAL >= 5/6 bar, scaled to the seed
count -- never a hardcoded absolute). The companion regression suite `tests/test_multi_turn_ordered_wm.py` (the
31-assertion MultiTurnAgentV2 gate) must stay green -- run it alongside.

CPU/numpy backend (the spiking RF composer runs there; each op is a small SimulationBridge). NO `sim/` edit.
"""
import math
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import logging

import numpy as np
import pytest

# Quiet the per-bridge init spam (each RF op builds a small bridge): keep pytest output the assertions only.
logging.disable(logging.INFO)

from research.runners.multi_turn_agent_v2 import MultiTurnAgentV2

# ---------------------------------------------------------------------------
# Pre-registered constants (frozen; mirror the two validated de-risks; never tuned to a result).
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44, 100, 101, 102]

# The fact base of the two de-risks: each subject has exactly one stored SVO fact, so the correct rendered
# sentence per topic is well-defined. The subjects are the discourse topics narrate() holds + sequences.
FACTS = [
    ("dog", "ran", "north"),
    ("cat", "saw", "river"),
    ("bird", "ate", "worm"),
    ("fox", "found", "den"),
    ("frog", "crossed", "road"),
    ("hawk", "chased", "mouse"),
]
SUBJECTS = [s for (s, _, _) in FACTS]
OBJECTS = sorted(set(o for (_, _, o) in FACTS))
UNKNOWN_TOPIC = "owl"               # a referent the WM can hold, but NO fact has it as agent -> the no-confab probe

N_TRIALS = 40                       # per-control trials per seed (random topic subsets / discourses)
N_TRIALS_NOCONFAB = 30
PASS_BAR = 0.80                     # per-seed per-control accuracy bar (the de-risk's frozen bar)

# The frozen GO bar is a FRACTION ">= 5/6 of seeds", scaled to however many seeds are run (so a partial run is
# judged on the same fractional bar -- never a hardcoded absolute count).
GO_FRACTION = 5.0 / 6.0


def _go_thresh(n_seeds):
    return int(math.ceil(GO_FRACTION * n_seeds))


def _build_agent(seed):
    """A MultiTurnAgentV2 whose composer holds the fixed fact base and whose order-encoded discourse buffer shares
    the composer's concept codes (same seed/D/sorted-vocab). The buffer resolves slot reads against the referent
    subset (subjects + objects + the unknown probe) only. enable_neural_render=True -> each non-pronominalized
    sentence's word ORDER is produced by the de-risked spiking competitive-queuing serial-order generator (so the
    emitted output is neural in both inter-sentence order [the slots] and intra-sentence word order)."""
    referents = SUBJECTS + OBJECTS + [UNKNOWN_TOPIC]
    vocab = sorted(set([w for f in FACTS for w in f] + [UNKNOWN_TOPIC]))
    agent = MultiTurnAgentV2(referent_concepts=referents, concepts={w: None for w in vocab},
                             seed=seed, enable_neural_render=True)
    for (s, v, o) in FACTS:
        agent.agent.hear(f"{s} {v} {o}")        # active voice: pos0=agent, pos1=action, pos2=patient
    return agent


def _stored_fact_str(topic):
    """The single correct rendered sentence for a topic = its stored (subject, verb, object) in SVO order; None for
    a topic with no stored fact (the unknown probe)."""
    for (s, v, o) in FACTS:
        if s == topic:
            return f"{s} {v} {o}"
    return None


# ===========================================================================
# Per-seed control implementations (each returns an accuracy in [0,1]).
# ===========================================================================
def _ordered_narration_accuracy(agent, seed, k=3):
    """CONTROL 1. narrate() of a random ORDERED K-topic subset (all distinct -> no recurrence -> no pronouns) emits
    exactly the K stored-fact sentences in the SAME ORDER as the topic list. Score = fraction of trials whose
    surface string == the K stored sentences joined in topic order (exact content per topic, no abstain)."""
    rng = np.random.default_rng(seed + 7)
    ok = 0
    for _ in range(N_TRIALS):
        idx = list(rng.choice(len(SUBJECTS), size=k, replace=False))
        seq = [SUBJECTS[i] for i in idx]
        expected = ". ".join(_stored_fact_str(t) for t in seq) + "."
        ok += (agent.narrate(seq) == expected)
    return ok / N_TRIALS


def _coherence_accuracy(agent, seed):
    """CONTROL 2. Discourse [sA, sB, sA] -- sA recurs as the 3rd topic. The 3rd sentence must be pronominalized AND
    its pronoun must RESOLVE (spiking slot-anaphora) to sA (the antecedent at slot 0, NOT the most-recent slot).
    Score = fraction of trials where the recurring sentence is pronominalized AND resolves correctly."""
    rng = np.random.default_rng(seed + 11)
    ok = 0
    for _ in range(N_TRIALS):
        sA, sB = rng.choice(SUBJECTS, size=2, replace=False)
        _surface, det = agent.narrate([sA, sB, sA], return_details=True)
        rec = det[-1]                                   # the recurrence (sA again)
        ok += bool(rec["pronominalized"] and rec["resolved_correct"])
    return ok / N_TRIALS


def _order_control_permute_accuracy(agent, seed, k=3):
    """CONTROL 3a (emission order tracks the topic order). For a random ORDERED topic subset and a PERMUTATION of
    it, the narration of the permuted topics must equal the order-permuted narration of the base (the inter-sentence
    order comes from the ordered-WM SLOTS, not a fixed storage order -- a storage-dump would emit the same order
    regardless and FAIL). All-distinct topics so there are no pronouns to confound the order check. Score = fraction
    of (non-trivial-permutation) trials where the per-sentence emitted order permutes correspondingly."""
    rng = np.random.default_rng(seed + 21)
    ok = 0
    n = 0
    for _ in range(N_TRIALS):
        idx = list(rng.choice(len(SUBJECTS), size=k, replace=False))
        seq = [SUBJECTS[i] for i in idx]
        perm = list(rng.permutation(k))
        if perm == list(range(k)):
            continue                                    # trivial perm -> skip (must observe a genuine re-order)
        n += 1
        pseq = [seq[i] for i in perm]
        _, base = agent.narrate(seq, return_details=True)
        _, permd = agent.narrate(pseq, return_details=True)
        base_text = [d["text"] for d in base]
        permd_text = [d["text"] for d in permd]
        expected = [base_text[i] for i in perm]         # the order-permuted base emission
        ok += (permd_text == expected) and (None not in base_text)
    return ok / max(n, 1)


def _order_control_flip_accuracy(agent, seed):
    """CONTROL 3b (the resolved antecedent FLIPS with which referent recurs). discourse_A=[sA,sB,sA] (sA recurs) vs
    discourse_B=[sB,sA,sB] (sB recurs): the recurring sentence's resolved antecedent must be sA in A and sB in B --
    it FLIPS. Score CORRECT iff both resolve correctly AND the two resolved antecedents differ (a genuine flip). A
    fixed-entity resolver could not flip."""
    rng = np.random.default_rng(seed + 33)
    ok = 0
    for _ in range(N_TRIALS):
        sA, sB = rng.choice(SUBJECTS, size=2, replace=False)
        _, detA = agent.narrate([sA, sB, sA], return_details=True)
        _, detB = agent.narrate([sB, sA, sB], return_details=True)
        rA = detA[-1]["resolved_antecedent"]
        rB = detB[-1]["resolved_antecedent"]
        a_ok = detA[-1]["pronominalized"] and rA == sA
        b_ok = detB[-1]["pronominalized"] and rB == sB
        ok += bool(a_ok and b_ok and (rA != rB))
    return ok / N_TRIALS


def _no_confab_accuracy(agent, seed):
    """CONTROL 4. A length-3 narration with one UNKNOWN topic (no stored fact) placed at a random slot: that slot
    must ABSTAIN (no sentence -- skipped from the surface) while the OTHER topics emit their correct sentences in
    order. Score = fraction where the unknown slot abstained AND the known slots' sentences are correct + ordered +
    the surface contains no confabulated unknown sentence."""
    rng = np.random.default_rng(seed + 55)
    ok = 0
    for _ in range(N_TRIALS_NOCONFAB):
        knowns_idx = list(rng.choice(len(SUBJECTS), size=2, replace=False))
        knowns = [SUBJECTS[i] for i in knowns_idx]
        upos = int(rng.integers(0, 3))
        seq = list(knowns)
        seq.insert(upos, UNKNOWN_TOPIC)
        surface, det = agent.narrate(seq, return_details=True)
        unknown_abstained = (det[upos]["text"] is None) and bool(det[upos]["abstained"])
        # The known slots (positions != upos) emit their correct stored sentence, in order.
        knowns_ok = all(det[p]["text"] == _stored_fact_str(seq[p]) for p in range(len(seq)) if p != upos)
        # The surface (joined, abstains skipped) is exactly the two known sentences in order, no confabulation.
        expected_surface = ". ".join(_stored_fact_str(seq[p]) for p in range(len(seq)) if p != upos) + "."
        surface_ok = (surface == expected_surface)
        ok += bool(unknown_abstained and knowns_ok and surface_ok)
    return ok / N_TRIALS_NOCONFAB


# ===========================================================================
# Multi-seed aggregation: compute every control across the seeds ONCE, cache, and assert the fractional bar.
# ===========================================================================
def _run_all_seeds():
    """Build each seed's agent once and score all five controls. Cached so the parametrized tests do not rebuild
    the (somewhat slow) per-seed agents repeatedly."""
    results = {}
    for seed in SEEDS:
        agent = _build_agent(seed)
        results[seed] = {
            "ordered": _ordered_narration_accuracy(agent, seed),
            "coherence": _coherence_accuracy(agent, seed),
            "order_permute": _order_control_permute_accuracy(agent, seed),
            "order_flip": _order_control_flip_accuracy(agent, seed),
            "no_confab": _no_confab_accuracy(agent, seed),
        }
    return results


@pytest.fixture(scope="module")
def seed_results():
    return _run_all_seeds()


def _n_pass(seed_results, key):
    return sum(seed_results[s][key] >= PASS_BAR for s in SEEDS)


# ===========================================================================
# Test 1: ORDERED NARRATION -- 3 facts as correctly-ordered sentences (exact content per topic).
# ===========================================================================
def test_ordered_narration(seed_results):
    n = _n_pass(seed_results, "ordered")
    thr = _go_thresh(len(SEEDS))
    assert n >= thr, (f"ordered narration passed {n}/{len(SEEDS)} seeds (bar {thr}); "
                      f"per-seed={[round(seed_results[s]['ordered'], 3) for s in SEEDS]}")


# ===========================================================================
# Test 2: COHERENCE -- a recurring topic is pronominalized + resolves to the correct antecedent.
# ===========================================================================
def test_coherence(seed_results):
    n = _n_pass(seed_results, "coherence")
    thr = _go_thresh(len(SEEDS))
    assert n >= thr, (f"coherence passed {n}/{len(SEEDS)} seeds (bar {thr}); "
                      f"per-seed={[round(seed_results[s]['coherence'], 3) for s in SEEDS]}")


# ===========================================================================
# Test 3: ORDER-CONTROL (load-bearing) -- both the emission-order permute AND the antecedent flip.
# ===========================================================================
def test_order_control_permute(seed_results):
    n = _n_pass(seed_results, "order_permute")
    thr = _go_thresh(len(SEEDS))
    assert n >= thr, (f"order-control (permute) passed {n}/{len(SEEDS)} seeds (bar {thr}); "
                      f"per-seed={[round(seed_results[s]['order_permute'], 3) for s in SEEDS]}")


def test_order_control_flip(seed_results):
    n = _n_pass(seed_results, "order_flip")
    thr = _go_thresh(len(SEEDS))
    assert n >= thr, (f"order-control (antecedent flip) passed {n}/{len(SEEDS)} seeds (bar {thr}); "
                      f"per-seed={[round(seed_results[s]['order_flip'], 3) for s in SEEDS]}")


# ===========================================================================
# Test 4: NO-CONFAB MOAT -- a topic with no stored fact abstains/skips, no confabulated sentence.
# ===========================================================================
def test_no_confab(seed_results):
    n = _n_pass(seed_results, "no_confab")
    thr = _go_thresh(len(SEEDS))
    assert n >= thr, (f"no-confab moat passed {n}/{len(SEEDS)} seeds (bar {thr}); "
                      f"per-seed={[round(seed_results[s]['no_confab'], 3) for s in SEEDS]}")


# ===========================================================================
# Side-effect-free: a narration must not perturb an in-progress multi-turn dialogue.
# ===========================================================================
@pytest.mark.parametrize("seed", [42, 100])
def test_narrate_is_side_effect_free(seed):
    """narrate() uses a FRESH discourse buffer (saving + restoring the standing window/composite), so an
    in-progress dialogue's pronoun resolution + Q&A are unchanged across a narration."""
    NOUNS = ["dog", "cat", "fish", "bird", "worm", "ball"]
    VOCAB = NOUNS + ["chase", "eat", "see"]
    a = MultiTurnAgentV2(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=seed)
    c = a.agent.composer
    for ag, ob in [("cat", "fish"), ("dog", "worm"), ("fish", "worm"), ("bird", "ball")]:
        c.store(ag, "eat", ob)
    a.hear("dog see cat")                       # cat most-recent
    before_mr = a.most_recent_referent()
    before_qa = a.what_does("it", "eat")
    assert before_mr == "cat" and before_qa == "fish", f"seed {seed}: standing dialogue precondition"
    _ = a.narrate(["bird", "fish"])             # a narration in the middle of the dialogue
    assert a.most_recent_referent() == before_mr, f"seed {seed}: narrate must not change most-recent referent"
    assert a.what_does("it", "eat") == before_qa, f"seed {seed}: narrate must not change in-progress Q&A"
    assert a.held_referents() == ["dog", "cat"], f"seed {seed}: standing discourse window must be intact"


# ===========================================================================
# Empty + all-unknown edge cases.
# ===========================================================================
def test_narrate_empty_and_all_unknown():
    agent = _build_agent(42)
    assert agent.narrate([]) == "", "empty topics -> empty narration"
    # An agent with NO stored facts -> every topic abstains -> empty narration (no confabulation).
    referents = SUBJECTS + OBJECTS + [UNKNOWN_TOPIC]
    vocab = sorted(set([w for f in FACTS for w in f] + [UNKNOWN_TOPIC]))
    blank = MultiTurnAgentV2(referent_concepts=referents, concepts={w: None for w in vocab}, seed=42)
    assert blank.narrate(["dog", "cat", "bird"]) == "", "all-unknown topics -> empty narration (all abstain)"


# ===========================================================================
# Fixed transcript: the exact validated coherent surface string at seed 42 (matches the de-risk findings).
# ===========================================================================
def test_fixed_coherent_transcript_seed42():
    """A concrete, deterministic coherent multi-sentence narration at seed 42 (the validated mechanism's output):
    the recurring 'dog' is pronominalized and resolves to its antecedent at gamma-slot 0."""
    agent = _build_agent(42)
    surface, det = agent.narrate(["dog", "bird", "dog"], return_details=True)
    assert surface == "dog ran north. bird ate worm. then it ran north.", f"got {surface!r}"
    rec = det[-1]
    assert rec["pronominalized"] is True
    assert rec["antecedent_slot"] == 0
    assert rec["resolved_antecedent"] == "dog"
    assert rec["resolved_correct"] is True
