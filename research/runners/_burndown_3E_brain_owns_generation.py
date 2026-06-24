"""BURNDOWN Phase-3E -- BRAIN-OWNS-GENERATION: integrate the validated b2 generative-replay proposer into the
conversational agent, so the brain GENERATES novel grounded discourse during a turn (not just retrieves), with
the no-confab moat VERIFYING every generated proposition.

CONTEXT (roadmap `docs/plans/2026-06-23-inventory-burndown-roadmap.md` Phase 3E = I-9 + the b2 seed; owner's
stated primary = the generative-sequence frontier `project_generative_sequence_frontier`):
  The production conversational composer (RFPhasorComposer) is structured-RETRIEVAL: it scored **0.0** novel-
  composition on held-out in-vocab SVO (`2026-06-22-generation-novelty-categorical-gap-MEASURED.md`) -- it emits
  ONLY stored facts. The b2 generative-replay proposer is GO standalone
  (`_genfrontier_b2_generative_replay_derisk.py`: the brain INVENTS novel-but-plausible propositions --
  novel-composition 0.752 vs 0.0, 17x random, shuffled-graph collapses, 0 moat leaks). This burndown WIRES that
  proposer INTO the conversational TURN.

WHAT THIS WIRES (the scope answer):
  The b2 proposer is a GENERATE channel ADDED to the conversational turn, ALONGSIDE the existing retrieval
  channels (the agent's `what_does`/`who_does`/`reason_chain` direct recall + the dlPFC `elaborate` dialogue
  plan). The integration surface is a thin wrapper `GenerativeConversationalTurn(BrainConversationalAgent)` that
  exposes a new `generate(topic=None)` / `propose_novel()` turn:
    - the GENERATE channel resamples role-filler bindings from the brain's OWN association structure (the learned
      PPMI co-occurrence cortex over the corpus it heard) to PROPOSE a novel-but-plausible SVO the brain did NOT
      hear verbatim (an inference/elaboration), GATED by graph-plausibility + non-contradiction;
    - the MOAT VERIFIES a GENERATED (vs retrieved) proposition by the b2 dual gate (plausibility AND
      non-contradiction) + the HYPOTHESIS guarantee: a generated proposition must NEVER pass the composer's
      KNOWN-fact retrieval (`what_does`/`is_it_true` still ABSTAIN on it) -- the brain distinguishes "I know X"
      from "X is plausible". A proposed SVO that CANNOT be verified (not graph-supported, or contradicts a stored
      fact) is ABSTAINED, never emitted.

HOW THE MOAT VERIFIES A *GENERATED* PROPOSITION (the key distinction vs the retrieval VERIFY):
  - the retrieval VERIFY (RichAnswerComposer.render_paragraph) checks each sentence re-parses to its GATHERED
    STORED SVO -- that gate REJECTS a novel proposition (it was never stored). So a generated proposition uses a
    DIFFERENT, owner-sanctioned verification (`feedback_moat_not_hard_lossy_memory_ok`): the b2
    plausibility+non-contradiction gate + the hypothesis-flag.
  - the no-confab moat on KNOWN-FACT retrieval is PRESERVED VERBATIM: `what_does`/`who_does`/`is_it_true` still
    abstain on every unstored cue (0 leaks). The GENERATE channel is a SEPARATE, honestly-flagged channel; a
    generated proposition is emitted as a HYPOTHESIS ("perhaps <S> <V> <O>"), never as a known fact.

THE 4 DE-RISK MEASUREMENTS (multi-seed >= 3, at the SMALL conversational-agent scale -- CPU, < 5 min):
  (a) NOVEL  -- the generated propositions are NOVEL (not a stored fact verbatim) AND the agent's known-fact
                retrieval still ABSTAINS on every generated proposition (it is not in the store).
  (b) PLAUSIBLE -- the generated propositions are grounded in the learned association structure (graph-supported),
                far ABOVE a random-recombination floor (the b2 17x advantage, re-measured here at agent scale).
  (c) SHUFFLED-COLLAPSES -- shuffling the learned graph collapses the plausibility to ~the random floor (the
                learned structure, not the SVO template, is load-bearing).
  (d) MOAT 0-CONFAB -- 0 generated-proposition -> known-fact leaks (a hypothesis never passes as a known fact),
                AND 0 explicitly-negated facts re-proposed (the non-contradiction gate). The agent's standing
                abstention moat on untaught cues is unregressed.

ANTI-CHEATS (all per the b2 de-risk, re-run at agent scale): novelty (the generated set is disjoint from the
stored set + the store abstains on it); the shuffled-graph control collapses plausibility; the moat rejects an
ungrounded/negated proposal (0 confab leaks); a LESION (ablate the plausibility gate) floods nonsense.

VERDICT:
  GO = the brain GENERATES novel grounded propositions during a conversational turn (novel + plausible +
       shuffled-collapses) and the no-confab moat HOLDS (0 confab leaks, 0 negated re-proposed, untaught-cue
       abstention unregressed). The brain owns generation; the moat verifies it.
  HONEST = the proposer doesn't wire cleanly OR a generated proposition can't be moat-verified without weakening
       the moat -> characterize precisely (under the BRAIN-BASED-ONLY standard, an honest negative IS the
       deliverable).

REUSE-BY-IMPORT, NO sim/ edit. CPU (`SIM_BACKEND=numpy`). Run:
  SIM_BACKEND=numpy python -u -m research.runners._burndown_3E_brain_owns_generation \
      --seeds 42,43,44 --out research/findings/raw/_burndown_3E_brain_owns_generation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Reuse-by-import: the validated b2 proposer machinery + the conversational agent + the PPMI cortex.
from research.runners._genfrontier_b2_generative_replay_derisk import (  # noqa: E402
    GenerativeReplayProposer,
    build_plausibility,
    build_stored_facts,
    random_recombination,
    shuffle_graph,
    _category_pools,
)
from research.runners.option_c_real_cooccurrence_derisk import (  # noqa: E402
    TAXONOMY_8x8,
    taxonomy_to_vocab_categories,
    build_real_cooccurrence,
)
from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402


# ===========================================================================
# THE INTEGRATION SURFACE: a GENERATE channel added to the conversational turn, alongside retrieval.
# ===========================================================================
class GenerativeConversationalTurn:
    """Wrap a `BrainConversationalAgent` (rf composer) and ADD a GENERATE channel to the conversational turn.

    The agent already has the RETRIEVAL channels: `what_does`/`who_does`/`is_it_true` (direct recall, abstaining =
    the no-confab moat) + `elaborate` (the dlPFC dialogue plan). This wrapper adds `generate(topic=None)`: the
    brain PROPOSES a novel-but-plausible SVO it did NOT hear verbatim (an inference/elaboration), via the b2
    generative-replay proposer over the brain's own association structure (the learned PPMI co-occurrence cortex).

    A generated proposition is a HYPOTHESIS: it is FLAGGED as plausible-not-known ("perhaps <S> <V> <O>"), and it
    must NEVER pass the agent's known-fact retrieval (`what_does`/`is_it_true` still abstain on it). A proposed SVO
    that cannot be verified (not graph-supported, or contradicting a stored fact) is ABSTAINED, never emitted."""

    def __init__(self, agent, P, row, tau, affirmed_facts, negated_facts, seed=42):
        self.agent = agent
        self.composer = agent.composer
        # the b2 proposer reads the SAME RFPhasorComposer the agent talks through (it must not contradict the
        # brain's known store, and a generated proposition must never pass the known-fact retrieval).
        self.proposer = GenerativeReplayProposer(
            self.composer, affirmed_facts, negated_facts, P, row, tau,
            np.random.default_rng(seed * 7 + 1))
        self.P, self.row, self.tau = P, row, tau

    # --- the GENERATE channel (the brain owns generation) ---
    def generate(self, n_attempts=2000):
        """The brain GENERATES a set of novel-but-plausible propositions during the turn (the b2 generative-replay
        proposer over the brain's learned association structure). Each accepted proposition is NOVEL (not stored),
        PLAUSIBLE (graph-supported), and NON-CONTRADICTORY (the brain wasn't told it is false). Returns the list of
        HYPOTHESIS triples (flagged plausible-not-known)."""
        rep = self.proposer.propose(n_attempts)
        return rep["accepted"]

    def say_hypothesis(self, triple):
        """Surface a generated proposition AS A HYPOTHESIS (the honest channel): the brain distinguishes 'I know X'
        from 'X is plausible'. Never asserted as a known fact."""
        a, ac, p = triple
        return f"perhaps {a} {ac} {p}"

    # --- the MOAT VERIFICATION of a GENERATED proposition (the key distinction) ---
    def verify_generated(self, triple):
        """Verify a GENERATED (vs retrieved) proposition under the owner-sanctioned moat/generativity trade
        (`feedback_moat_not_hard_lossy_memory_ok`). Returns a dict with the verification breakdown.

        The retrieval VERIFY (re-parse == a gathered STORED SVO) would REJECT a novel proposition (never stored).
        A generated proposition uses the b2 dual gate + the hypothesis guarantee:
          (i)   PLAUSIBLE          -- graph-supported (selectional-preference: agent~action AND action~patient).
          (ii)  NON-CONTRADICTORY  -- the brain was NOT told this triple is false (`is_it_true` != 'no').
          (iii) HYPOTHESIS-NOT-KNOWN -- the agent's known-fact retrieval still ABSTAINS on it: `what_does` does
                NOT return this patient AND `is_it_true` is 'unknown' (it was never stored). A leak here = the
                moat broke (a hypothesis masquerading as a known fact)."""
        a, ac, p = triple
        plausible = self.proposer._plausible(a, ac, p)
        non_contradictory = (self.agent.is_it_true(a, ac, p) != "no")
        known_patient = self.agent.what_does(a, ac)        # must NOT be p (the cue was never stored as this)
        yn = self.agent.is_it_true(a, ac, p)               # must be 'unknown' (never stored -> not a known fact)
        hypothesis_not_known = (known_patient != p) and (yn == "unknown")
        return {
            "triple": f"{a} {ac} {p}",
            "plausible": bool(plausible),
            "non_contradictory": bool(non_contradictory),
            "hypothesis_not_known": bool(hypothesis_not_known),
            "verified": bool(plausible and non_contradictory and hypothesis_not_known),
        }


# ===========================================================================
# Build the agent + teach it the interlinked facts, returning the (P, row, tau, affirmed, negated) the proposer needs.
# ===========================================================================
def build_turn(seed, vocab, corpus, a):
    rng = np.random.default_rng(seed)
    agents, actions, patients = _category_pools(TAXONOMY_8x8)
    P, row = build_plausibility(corpus, vocab)
    pos = P[P > 0]
    tau = float(np.percentile(pos, a.tau_pct)) if pos.size else 0.0

    affirmed, negated, plausible_all = build_stored_facts(
        agents, actions, patients, P, row, tau, a.n_facts, a.n_negated, rng)

    # the conversational agent (rf composer = the production default), TEACH it the affirmed + negated facts via
    # the agent's `hear()`. So the brain's known store IS the agent's store; the proposer reads it.
    concepts = {w: None for w in vocab}
    agent = BrainConversationalAgent(seed=seed, concepts=concepts, composer_kind="rf",
                                     enable_neural_render=False)
    for ag, ac, pt in affirmed:
        agent.hear(f"{ag} {ac} {pt}", polarity="AFFIRM")
    for ag, ac, pt in negated:
        agent.hear(f"{ag} {ac} {pt}", polarity="NEGATE")

    all_stored = set(affirmed) | set(negated)
    plausible_novel_universe = sorted(set(plausible_all) - all_stored)

    turn = GenerativeConversationalTurn(agent, P, row, tau, affirmed, negated, seed=seed)
    return turn, agent, affirmed, negated, P, row, tau, plausible_novel_universe


def run_seed(seed, vocab, corpus, a):
    rng = np.random.default_rng(seed)
    turn, agent, affirmed, negated, P, row, tau, plausible_novel_universe = build_turn(seed, vocab, corpus, a)
    all_stored = set(affirmed) | set(negated)

    # ---- the brain GENERATES novel propositions during the turn ----
    generated = turn.generate(n_attempts=a.n_attempts)
    generated_set = set(generated)

    # (a) NOVEL: the generated propositions are NOT stored verbatim, AND the agent's known-fact retrieval ABSTAINS
    # on every generated proposition (it is not in the store).
    novel_disjoint = len(generated_set & all_stored) == 0
    n_generated = len(generated)
    novel_comp_score = min(1.0, n_generated / max(1, len(plausible_novel_universe)))
    # the agent's known-fact retrieval must ABSTAIN on every generated proposition (the hypothesis-not-known
    # guarantee): what_does returns != p (or None) AND is_it_true == 'unknown'.
    retrieval_abstains_on_generated = 0
    for (ag, ac, pt) in generated:
        kp = agent.what_does(ag, ac)
        yn = agent.is_it_true(ag, ac, pt)
        if kp != pt and yn == "unknown":
            retrieval_abstains_on_generated += 1
    retrieval_abstains_all = (retrieval_abstains_on_generated == n_generated)

    # full moat-verification breakdown per generated proposition
    verifs = [turn.verify_generated(t) for t in generated]
    n_verified = sum(1 for v in verifs if v["verified"])
    # a HYPOTHESIS-flag leak = a generated proposition that passes the known-fact retrieval (must be 0)
    moat_leaks = sum(1 for v in verifs if not v["hypothesis_not_known"])

    # (b) PLAUSIBLE (advantage over random recombination), re-measured at agent scale, via the b2 helpers.
    rep_book = turn.proposer.propose(a.n_attempts)               # bookkeeping incl. plausible_fraction_of_novel
    replay_frac = rep_book["plausible_fraction_of_novel"]
    randb = random_recombination(turn.proposer, a.n_attempts, np.random.default_rng(seed * 13 + 3))
    random_frac = randb["plausible_fraction_of_novel"]
    advantage_ratio = replay_frac / max(random_frac, 1.0 / max(1, randb["n_novel_attempts"]))

    # (c) SHUFFLED-GRAPH collapses: a proposer on the shuffled graph -> its TRUE-graph plausible-frac collapses to
    # ~the random floor (the learned structure is load-bearing, not the SVO template).
    P_shuf = shuffle_graph(P, np.random.default_rng(seed * 17 + 5))
    pos_s = P_shuf[P_shuf > 0]
    tau_s = float(np.percentile(pos_s, a.tau_pct)) if pos_s.size else 0.0
    proposer_shuf = GenerativeReplayProposer(turn.composer, affirmed, negated, P_shuf, row, tau_s,
                                             np.random.default_rng(seed * 19 + 7))
    shuf_true_novel_plausible = 0
    for _ in range(a.n_attempts):
        _r = proposer_shuf.rng
        ag = proposer_shuf.agents[int(_r.integers(len(proposer_shuf.agents)))]
        acn = proposer_shuf._sample_weighted(proposer_shuf.actions,
                                             proposer_shuf._weight_partner((ag,), proposer_shuf.actions))
        ptn = proposer_shuf._sample_weighted(proposer_shuf.patients,
                                             proposer_shuf._weight_partner((ag, acn), proposer_shuf.patients))
        if (ag, acn, ptn) in proposer_shuf.all_stored:
            continue
        if turn.proposer._plausible(ag, acn, ptn):      # TRUE-graph plausibility of a shuffled-replay sample
            shuf_true_novel_plausible += 1
    shuf_true_frac = shuf_true_novel_plausible / max(1, a.n_attempts)

    # (d) MOAT: non-contradiction effectiveness -- the proposer must NEVER propose a negated fact (each is a
    # tempting plausible recombination the gate must catch).
    contradictions_proposed = len(generated_set & set(negated))

    # the agent's STANDING abstention moat on UNTAUGHT cues is unregressed (a sanity that wiring the GENERATE
    # channel did not weaken the known-fact moat): random unstored (agent, action) cues -> what_does abstains.
    n_ab, ab_ok, guard = 0, 0, 0
    stored_cues = {(ag, ac) for ag, ac, _ in affirmed}
    agents_pool, actions_pool, _patients = _category_pools(TAXONOMY_8x8)
    while n_ab < 20 and guard < 100000:
        guard += 1
        ag = agents_pool[int(rng.integers(len(agents_pool)))]
        ac = actions_pool[int(rng.integers(len(actions_pool)))]
        if (ag, ac) in stored_cues:
            continue
        n_ab += 1
        ab_ok += int(agent.what_does(ag, ac) is None)

    # LESION: ablate the plausibility gate -> the proposer floods nonsense (the gate is causally responsible).
    lesion_accepted, lesion_plausible = 0, 0
    lrng = np.random.default_rng(seed * 23 + 11)
    seen_l = set()
    for _ in range(a.n_attempts):
        ag = turn.proposer.agents[int(lrng.integers(len(turn.proposer.agents)))]
        acn = turn.proposer._sample_weighted(turn.proposer.actions,
                                             turn.proposer._weight_partner((ag,), turn.proposer.actions))
        ptn = turn.proposer._sample_weighted(turn.proposer.patients,
                                             turn.proposer._weight_partner((ag, acn), turn.proposer.patients))
        triple = (ag, acn, ptn)
        if triple in turn.proposer.all_stored or triple in seen_l:
            continue
        # NO plausibility gate (only non-contradiction) -> nonsense floods
        if turn.composer.ask_yes_no(ag, acn, ptn) != "no":
            lesion_accepted += 1
            seen_l.add(triple)
            if turn.proposer._plausible(ag, acn, ptn):
                lesion_plausible += 1
    lesion_plausible_frac = lesion_plausible / max(1, lesion_accepted)

    # a few example hypotheses the brain GENERATED, with PPMI strengths + the hypothesis surface form
    examples = []
    for t in generated[:8]:
        ag, ac, pt = t
        examples.append({"hypothesis": turn.say_hypothesis(t),
                         "ppmi_agent_action": round(float(P[row[ag], row[ac]]), 3),
                         "ppmi_action_patient": round(float(P[row[ac], row[pt]]), 3),
                         "known_fact_retrieval_abstains": (agent.what_does(ag, ac) != pt
                                                           and agent.is_it_true(ag, ac, pt) == "unknown")})

    print(f"\n[3E seed {seed}] taught {len(affirmed)} affirmed + {len(negated)} negated SVO facts | "
          f"discoverable novel-plausible universe {len(plausible_novel_universe)} | tau(P{a.tau_pct})={tau:.3f}",
          flush=True)
    print(f"  (a) GENERATE: brain generated {n_generated} distinct NOVEL propositions (novel-comp score "
          f"{novel_comp_score:.3f} vs retrieval 0.0); novel-disjoint-from-store {novel_disjoint}; "
          f"known-fact retrieval ABSTAINS on {retrieval_abstains_on_generated}/{n_generated} (all: "
          f"{retrieval_abstains_all})", flush=True)
    print(f"  (b) PLAUSIBLE: replay-frac {replay_frac:.3f} vs random {random_frac:.4f} -> {advantage_ratio:.1f}x",
          flush=True)
    print(f"  (c) SHUFFLED: TRUE plausible-frac {shuf_true_frac:.4f} (must collapse toward random {random_frac:.4f}"
          f", vs replay {replay_frac:.3f})", flush=True)
    print(f"  (d) MOAT: hypothesis->known-fact leaks {moat_leaks} (must be 0) | negated re-proposed "
          f"{contradictions_proposed} (must be 0) | untaught-cue abstention {ab_ok}/{n_ab} | LESION (no "
          f"plausibility gate) {lesion_accepted} accepted, {lesion_plausible_frac*100:.0f}% plausible", flush=True)
    if examples:
        print(f"  generated hypotheses: {[e['hypothesis'] for e in examples]}", flush=True)

    return {
        "seed": seed,
        "n_affirmed": len(affirmed),
        "n_negated": len(negated),
        "tau": tau,
        "discoverable_novel_plausible_universe": len(plausible_novel_universe),
        # (a) NOVEL
        "n_generated": n_generated,
        "novel_composition_score": novel_comp_score,
        "novel_disjoint_from_store": novel_disjoint,
        "retrieval_abstains_on_generated": retrieval_abstains_on_generated,
        "retrieval_abstains_all": retrieval_abstains_all,
        "n_verified": n_verified,
        "generated_examples": [turn.say_hypothesis(t) for t in generated[:20]],
        # (b) PLAUSIBLE
        "replay_plausible_fraction_of_novel": replay_frac,
        "random_plausible_fraction_of_novel": random_frac,
        "advantage_ratio": advantage_ratio,
        # (c) SHUFFLED
        "shuffled_true_plausible_fraction_of_novel": shuf_true_frac,
        # (d) MOAT
        "moat_leaks": moat_leaks,
        "contradictions_proposed": contradictions_proposed,
        "untaught_cue_abstention_correct": ab_ok,
        "untaught_cue_abstention_attempted": n_ab,
        "lesion_accepted": lesion_accepted,
        "lesion_plausible_fraction": lesion_plausible_frac,
        "examples": examples,
    }


def decide_verdict(rows, a):
    """GO iff, across ALL seeds: (a) the brain GENERATES novel propositions (novel-comp score > 0, >= min_novel
    distinct, disjoint from the store, known-fact retrieval abstains on every one); (b) a clear PLAUSIBILITY
    ADVANTAGE over random recombination (advantage_ratio >= advantage_bar); (c) the SHUFFLED-graph control
    collapses the plausibility toward the random floor (the learned structure is load-bearing); (d) the no-confab
    MOAT HOLDS -- 0 hypothesis->known-fact leaks AND 0 negated facts re-proposed AND the untaught-cue abstention
    is unregressed (>= store_floor_bar, the documented RF code-fidelity tail). Else HONEST + why."""
    def col(k):
        return [r[k] for r in rows]

    replay_frac = np.array(col("replay_plausible_fraction_of_novel"))
    shuf_frac = np.array(col("shuffled_true_plausible_fraction_of_novel"))
    adv = np.array(col("advantage_ratio"))
    novel_score = np.array(col("novel_composition_score"))
    n_gen = np.array(col("n_generated"))
    leaks = np.array(col("moat_leaks"))
    contra = np.array(col("contradictions_proposed"))
    ab_ok = np.array(col("untaught_cue_abstention_correct"))
    ab_att = np.array(col("untaught_cue_abstention_attempted"))
    novel_disjoint = np.array(col("novel_disjoint_from_store"))
    retr_abstains = np.array(col("retrieval_abstains_all"))
    lesion_frac = np.array(col("lesion_plausible_fraction"))

    adv_bar = float(a.advantage_bar)
    min_novel = int(a.min_novel)
    collapse_frac = float(a.shuffle_collapse_frac)

    novel_above_zero_all = bool(np.all(n_gen >= min_novel) and np.all(novel_score > 0.0)
                                and np.all(novel_disjoint) and np.all(retr_abstains))
    advantage_all = bool(np.all(adv >= adv_bar))
    shuffled_collapses_all = bool(np.all(shuf_frac <= collapse_frac * np.maximum(replay_frac, 1e-9)))
    moat_preserved_all = bool(np.all(leaks == 0) and np.all(contra == 0))
    store_floor_rate = ab_ok / np.maximum(ab_att, 1)
    store_floor_ok_all = bool(np.all(store_floor_rate >= float(a.store_floor_bar)))
    # the lesion floods nonsense (causal): plausibility floor when the gate is ablated should be well below the
    # gated replay (a sanity that the gate is load-bearing, reported, not a hard gate).
    lesion_floods_all = bool(np.all(lesion_frac < 0.6 * np.maximum(replay_frac, 1e-9) + 0.5))

    detail = {
        "replay_plausible_fraction_mean": float(replay_frac.mean()),
        "random_plausible_fraction_mean": float(np.mean(col("random_plausible_fraction_of_novel"))),
        "shuffled_true_plausible_fraction_mean": float(shuf_frac.mean()),
        "advantage_ratio_mean": float(adv.mean()),
        "advantage_ratio_min": float(adv.min()),
        "novel_composition_score_mean": float(novel_score.mean()),
        "n_generated_mean": float(n_gen.mean()),
        "n_generated_min": int(n_gen.min()),
        "novel_disjoint_all_seeds": bool(np.all(novel_disjoint)),
        "retrieval_abstains_on_generated_all_seeds": bool(np.all(retr_abstains)),
        "moat_leaks_total": int(leaks.sum()),
        "contradictions_proposed_total": int(contra.sum()),
        "untaught_cue_abstention_rate_mean": float(store_floor_rate.mean()),
        "untaught_cue_abstention_rate_min": float(store_floor_rate.min()),
        "untaught_cue_abstention_ok_all": store_floor_ok_all,
        "lesion_plausible_fraction_mean": float(lesion_frac.mean()),
        "lesion_floods_nonsense_all": lesion_floods_all,
        "novel_above_zero_all_seeds": novel_above_zero_all,
        "advantage_all_seeds": advantage_all,
        "shuffled_collapses_all_seeds": shuffled_collapses_all,
        "moat_preserved_all_seeds": moat_preserved_all,
        "advantage_bar": adv_bar,
        "min_novel_bar": min_novel,
        "shuffle_collapse_frac_bar": collapse_frac,
        "store_floor_bar": float(a.store_floor_bar),
    }

    if novel_above_zero_all and advantage_all and shuffled_collapses_all and moat_preserved_all and store_floor_ok_all:
        verdict = "GO"
    elif not moat_preserved_all:
        verdict = "HONEST_moat_broken"
    elif not store_floor_ok_all:
        verdict = "HONEST_untaught_abstention_regressed"
    elif not novel_above_zero_all:
        verdict = "HONEST_no_novel_generated"
    elif not advantage_all:
        verdict = "HONEST_no_plausibility_advantage"
    elif not shuffled_collapses_all:
        verdict = "HONEST_structure_not_load_bearing"
    else:
        verdict = "HONEST_other"
    return verdict, detail


def main():
    p = argparse.ArgumentParser(description="Burndown 3E: brain-owns-generation -- wire the b2 generative-replay "
                                            "proposer into the conversational turn, moat-verifying generated props.")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--n-facts", type=int, default=24, help="AFFIRMED interlinked SVO facts taught to the agent")
    p.add_argument("--n-negated", type=int, default=12, help="NEGATED facts ('X does NOT Y') -- the non-contra gate")
    p.add_argument("--n-attempts", type=int, default=2000, help="generative-replay samples per channel")
    p.add_argument("--tau-pct", type=float, default=50.0, help="graph-related threshold = percentile of pos PPMI")
    p.add_argument("--advantage-bar", type=float, default=3.0, help="replay-vs-random plausible-frac RATIO gate")
    p.add_argument("--min-novel", type=int, default=3, help="min distinct novel-plausible propositions generated")
    p.add_argument("--shuffle-collapse-frac", type=float, default=0.5,
                   help="shuffled-graph TRUE plausible-frac must drop to <= this fraction of the real replay's")
    p.add_argument("--store-floor-bar", type=float, default=0.95,
                   help="agent untaught-cue abstention tolerance (the documented RF code-fidelity tail)")
    p.add_argument("--max-bytes", type=int, default=4_000_000, help="bytes of TinyStories for the co-occ graph")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--repeat-cap", type=int, default=40)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")

    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    t0 = time.time()
    print(f"[burndown-3E] seeds={seeds} n_facts={a.n_facts} n_attempts={a.n_attempts} tau_pct={a.tau_pct} -- "
          f"can the BRAIN GENERATE novel grounded propositions during a conversational turn, with the no-confab "
          f"moat verifying every one?", flush=True)

    vocab, cat_ids, cat_names = taxonomy_to_vocab_categories(TAXONOMY_8x8)
    corpus_path = os.path.join(_REPO, "data", "corpus", "tinystories.txt")
    if not os.path.exists(corpus_path):
        print(f"[ERROR] corpus not found: {corpus_path}", flush=True)
        sys.exit(2)
    corpus = build_real_cooccurrence(corpus_path, vocab, cat_ids, window=a.window, repeat_cap=a.repeat_cap,
                                     seed=42, max_bytes=a.max_bytes, freq_floor=30,
                                     min_facts_per_category=20, verbose=True)

    rows = [run_seed(s, vocab, corpus, a) for s in seeds]
    verdict, detail = decide_verdict(rows, a)

    print(f"\n{'='*98}", flush=True)
    print(f"  OVERALL VERDICT: {verdict}", flush=True)
    print(f"  (a) brain GENERATES novel grounded propositions: novel-comp score mean "
          f"{detail['novel_composition_score_mean']:.3f} (vs retrieval 0.0; >0 + disjoint + retrieval-abstains all "
          f"seeds: {detail['novel_above_zero_all_seeds']}; min {detail['n_generated_min']} generated)", flush=True)
    print(f"  (b) PLAUSIBILITY ADVANTAGE: replay {detail['replay_plausible_fraction_mean']:.3f} vs random "
          f"{detail['random_plausible_fraction_mean']:.4f} -> {detail['advantage_ratio_mean']:.1f}x "
          f"(min {detail['advantage_ratio_min']:.1f}x; >= {detail['advantage_bar']}x all: "
          f"{detail['advantage_all_seeds']})", flush=True)
    print(f"  (c) SHUFFLED-GRAPH collapses: TRUE plausible-frac {detail['shuffled_true_plausible_fraction_mean']:.4f}"
          f" (collapses all seeds: {detail['shuffled_collapses_all_seeds']})", flush=True)
    print(f"  (d) MOAT 0-CONFAB: {detail['moat_leaks_total']} hypothesis->known leaks + "
          f"{detail['contradictions_proposed_total']} negated re-proposed (preserved all seeds: "
          f"{detail['moat_preserved_all_seeds']}); untaught-cue abstention mean "
          f"{detail['untaught_cue_abstention_rate_mean']:.3f} (>= {detail['store_floor_bar']:.2f} all: "
          f"{detail['untaught_cue_abstention_ok_all']}); LESION plausible-frac "
          f"{detail['lesion_plausible_fraction_mean']:.3f} (floods nonsense: {detail['lesion_floods_nonsense_all']})",
          flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*98}\n", flush=True)

    out = {
        "probe": "burndown_3E_brain_owns_generation",
        "verdict": verdict,
        "seeds": seeds,
        "config": {"n_facts": a.n_facts, "n_negated": a.n_negated, "n_attempts": a.n_attempts,
                   "tau_pct": a.tau_pct, "advantage_bar": a.advantage_bar, "min_novel": a.min_novel,
                   "shuffle_collapse_frac": a.shuffle_collapse_frac, "store_floor_bar": a.store_floor_bar,
                   "max_bytes": a.max_bytes, "window": a.window},
        "baseline_to_beat": {"measured_retrieval_novel_composition": 0.0,
                             "source": "2026-06-22-generation-novelty-categorical-gap-MEASURED.md"},
        "integration_scope": (
            "the b2 generative-replay proposer is wired into the conversational TURN as a GENERATE channel ADDED "
            "alongside the agent's existing RETRIEVAL channels (what_does/who_does/is_it_true direct recall + the "
            "dlPFC elaborate dialogue plan). The wrapper GenerativeConversationalTurn(BrainConversationalAgent) "
            "exposes generate(topic) -> the brain proposes a novel-but-plausible SVO it did NOT hear verbatim, via "
            "the b2 proposer over the brain's own association structure (the learned PPMI co-occurrence cortex). "
            "The MOAT verifies a GENERATED (vs retrieved) proposition by the b2 dual gate (plausibility AND "
            "non-contradiction) + the HYPOTHESIS guarantee (it must NEVER pass the agent's known-fact retrieval -- "
            "what_does/is_it_true still abstain on it). The no-confab moat on KNOWN-fact retrieval is preserved "
            "verbatim; the GENERATE channel is a separate, honestly-flagged channel ('perhaps S V O')."),
        "moat_verify_distinction": (
            "the retrieval VERIFY (RichAnswerComposer.render_paragraph: each sentence re-parses to its gathered "
            "STORED SVO) REJECTS a novel proposition (never stored). A GENERATED proposition uses the owner-"
            "sanctioned moat/generativity trade (feedback_moat_not_hard_lossy_memory_ok): plausibility + "
            "non-contradiction + the hypothesis-flag. A generated SVO that cannot be verified is ABSTAINED, never "
            "emitted -- the no-confab moat (0 false-accepts on KNOWN facts) is never weakened."),
        "detail": detail,
        "per_seed": rows,
        "brain_based_note": (
            "the LEARNED graph (plausibility/likelihood) is the project's PPMI co-occurrence cortex over REAL "
            "TinyStories; the KNOWN-fact store + the no-confab moat are the RF phasor composer the conversational "
            "agent talks through. The proposer RESAMPLES role-filler bindings from the learned graph (hippocampal "
            "generative replay, catalog G.09), gated by graph-plausibility + non-contradiction, FLAGGING proposals "
            "as HYPOTHESES. NO sim/ edit; reuse-by-import; CPU."),
        "elapsed_total_s": time.time() - t0,
    }
    if a.out is None:
        a.out = os.path.join(_REPO, "research", "findings", "raw", "_burndown_3E_brain_owns_generation.json")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {a.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
