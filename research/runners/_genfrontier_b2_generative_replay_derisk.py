"""GENERATIVE-FRONTIER PATH (b2) DE-RISK -- can the BRAIN'S OWN generative replay INVENT novel-but-plausible
propositions (not retrieve)? The cheapest-first probe of "can the brain invent content" (tier G2).

CONTEXT (the scoping `research/findings/2026-06-23-generative-frontier-scoping.md`, Path (b) option b2 -- the most
"brain's own" route to G2 novel propositions):
  The production conversational composer is a structured-RETRIEVAL system: it scored **0.0** novel-composition on
  16 held-out in-vocabulary SVO triples (`2026-06-22-generation-novelty-categorical-gap-MEASURED.md`) -- it emits
  ONLY stored facts. The owner wants genuine generation that is THE BRAIN'S (not the LLM's). The biology-faithful
  route is **hippocampal generative replay** (catalog G.09 constructive imagination; the literature's finding that
  generative replay resamples FICTIVE sequences INCLUDING never-experienced recombinations -- Stoianov/Maisto/
  Pezzulo 2022; Barry/Love Nat Hum Behav 2023). During an offline "imagination" phase, RESAMPLE role-filler
  bindings from the LEARNED association graph to PROPOSE novel SVO triples, GATED by (i) graph-PLAUSIBILITY (the
  proposed triple's role-fillers co-occur / are graph-related) and (ii) NON-CONTRADICTION (it doesn't contradict a
  stored fact).

THE MOAT/GENERATIVITY TRADE (owner-sanctioned, `feedback_moat_not_hard_lossy_memory_ok`):
  A pure no-confab moat ("never assert the un-stored") FORBIDS G2 by construction. Generative replay DELIBERATELY
  asserts never-told (but plausible, graph-supported) propositions. The owner explicitly licenses this: the gate
  becomes "plausible given the learned graph + not contradicting a stored fact" rather than "exactly stored." The
  reconstructive-memory stance (real memory confabulates plausibly). CRITICAL: a proposal is flagged a HYPOTHESIS /
  'plausible', NEVER asserted as a known fact -- the brain distinguishes 'I know X' from 'X is plausible'. The
  no-confab moat on KNOWN-fact retrieval is preserved verbatim; the proposer is a SEPARATE, honestly-flagged channel.

REUSE-BY-IMPORT (NO sim/ edit):
  - the PPMI stream-cortex co-occurrence structure (`option_c_real_cooccurrence_derisk.build_real_cooccurrence` over
    REAL TinyStories + the 8x8 a-priori taxonomy) -> the LEARNED association graph + the plausibility signal;
  - the RF phasor composer (`rf_phasor_composer.RFPhasorComposer`: store + role-filler bindings + the no-confab
    moat `query_patient`/`ask_yes_no`) -> the brain's KNOWN-fact store the proposer reads + must not contradict.

THE 4 MEASUREMENTS (multi-seed, >=3):
  (1) NOVEL-COMPOSITION score -- novel-but-plausible triples proposed (never told, but graph-supported), vs the
      measured 0.0 retrieval baseline. The FIRST brain-mechanism novel-composition > 0.
  (2) RANDOM-RECOMBINATION baseline -- random vocab triples (chance plausibility). The proposer must beat this.
  (3) SHUFFLED-GRAPH control -- shuffle the learned edges -> the plausibility of the proposals MUST collapse to
      chance, proving the learned structure is load-bearing (not a string artifact / not the SVO template alone).
  (4) MOAT-HONESTY check -- proposals are flagged HYPOTHESES / 'plausible', NOT asserted as stored facts; a proposal
      must NEVER pass the composer's known-fact retrieval (`what_does`/`ask_yes_no` still abstain on it).

VERDICT:
  GO            = the brain proposes novel-but-plausible propositions ABOVE chance (random baseline), the learned
                  graph is load-bearing (shuffled collapses to chance), proposals honestly flagged plausible-not-
                  known (moat preserved) -> the FIRST brain-mechanism novel-composition > 0.
  HONEST-NEGATIVE = resampling doesn't beat chance / the plausibility gate fails to separate -> a real finding
                  (under the BRAIN-BASED-ONLY standard, an honest negative IS the deliverable) + precisely why.

GPU-FREE: SIM_BACKEND=numpy (CPU). Run:
  SIM_BACKEND=numpy python -u -m research.runners._genfrontier_b2_generative_replay_derisk \
      --seeds 42,43,44 --out research/findings/raw/_genfrontier_b2_generative_replay_derisk.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Reuse-by-import: the PPMI co-occurrence corpus + the a-priori taxonomy, and the RF composer (store + moat).
from research.runners.option_c_real_cooccurrence_derisk import (  # noqa: E402
    TAXONOMY_8x8,
    taxonomy_to_vocab_categories,
    build_real_cooccurrence,
)
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402


# ===========================================================================
# Role pools over the 8x8 taxonomy. The SVO world: animals/family ACT, actions are the VERBS, and
# food/places/toys/colors are the THINGS acted on. The role assignment is a-priori (NOT corpus-derived);
# the PLAUSIBILITY of a (role-filler, role-filler) pairing is what the LEARNED graph supplies.
# ===========================================================================
AGENT_CATS = ["animals", "family"]          # who does things
ACTION_CATS = ["actions"]                     # the verbs
PATIENT_CATS = ["food", "places", "toys", "colors", "body"]   # what is acted on / about


def _category_pools(taxonomy):
    agents, actions, patients = [], [], []
    for cat, words in taxonomy.items():
        if cat in AGENT_CATS:
            agents += words
        elif cat in ACTION_CATS:
            actions += words
        elif cat in PATIENT_CATS:
            patients += words
    return agents, actions, patients


# ===========================================================================
# The LEARNED association graph + plausibility signal, from REAL TinyStories co-occurrence (PPMI-style).
# build_real_cooccurrence returns `facts` = windowed co-occurrence SCENES (sets of in-vocab words). We turn
# those into a symmetric word-relatedness matrix P (positive pointwise mutual information over the scene
# co-occurrences). This is the brain's learned "how related are these two words" signal -- the generative
# model's likelihood the replay samples from / the plausibility gate thresholds on.
# ===========================================================================
def build_plausibility(corpus, vocab):
    """Symmetric PPMI word-relatedness P[i,j] (>=0) from the co-occurrence scenes. PPMI(i,j) =
    max(0, log( p(i,j) / (p(i) p(j)) )). p(i,j) = #scenes containing BOTH i and j / #scenes; p(i) =
    #scenes containing i / #scenes. The category structure (within-category words co-occur more in real
    text) is what makes a recombination plausible -- learned, not injected."""
    Nm = len(vocab)
    row = {w: i for i, w in enumerate(vocab)}
    facts = corpus["facts"]
    Ns = len(facts)
    co = np.zeros((Nm, Nm), dtype=np.float64)
    occ = np.zeros(Nm, dtype=np.float64)
    for scene in facts:
        idx = [row[w] for w in scene if w in row]
        for a in idx:
            occ[a] += 1.0
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                co[idx[a], idx[b]] += 1.0
                co[idx[b], idx[a]] += 1.0
    pi = occ / max(1.0, Ns)
    P = np.zeros((Nm, Nm), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(Nm):
            for j in range(Nm):
                if i == j or co[i, j] <= 0 or pi[i] <= 0 or pi[j] <= 0:
                    continue
                pij = co[i, j] / max(1.0, Ns)
                val = np.log(pij / (pi[i] * pi[j]))
                P[i, j] = max(0.0, val)
    return P, row


# ===========================================================================
# The BRAIN'S generative-replay PROPOSER (catalog G.09 constructive imagination).
# ===========================================================================
class GenerativeReplayProposer:
    """Offline 'imagination': resample role-filler bindings from the learned graph to PROPOSE novel SVO
    triples, biased toward graph-plausible recombinations (hippocampal generative replay samples from the
    learned generative model's likelihood). Each proposal is GATED by (i) plausibility (the triple's
    role-fillers are mutually graph-related above threshold) and (ii) non-contradiction (it doesn't
    contradict a stored fact). Proposals are flagged HYPOTHESES, never asserted as stored facts."""

    def __init__(self, composer, stored_facts, negated_facts, P, row, tau, rng):
        self.composer = composer
        # the brain's experienced facts -- AFFIRMED (true) and NEGATED ('X does NOT Y', stored via the
        # composer's polarity tag). Both are "stored" (the proposer must not resurface either as novel); the
        # NEGATED set is what the non-contradiction gate forbids re-asserting affirmatively.
        self.stored_set = {(a, ac, p) for a, ac, p in stored_facts}
        self.negated_set = {(a, ac, p) for a, ac, p in negated_facts}
        self.all_stored = self.stored_set | self.negated_set
        self.stored_cues = {(a, ac) for a, ac, _ in stored_facts}
        # the role-filler pools the brain has actually USED (replay recombines what it has experienced) --
        # drawn from BOTH affirmed and negated facts (the brain knows these words either way).
        allf = list(stored_facts) + list(negated_facts)
        self.agents = sorted({a for a, _, _ in allf})
        self.actions = sorted({ac for _, ac, _ in allf})
        self.patients = sorted({p for _, _, p in allf})
        self.P, self.row, self.tau, self.rng = P, row, tau, rng

    def _related(self, w1, w2):
        """graph-related = PPMI(w1, w2) >= tau (the learned co-occurrence threshold)."""
        return self.P[self.row[w1], self.row[w2]] >= self.tau

    def _plausible(self, a, ac, p):
        """A proposed triple is PLAUSIBLE iff the SELECTIONAL-PREFERENCE structure holds: the agent plausibly
        DOES the action (agent~action related) AND the action plausibly TAKES the patient (action~patient
        related) -- the verb selects its arguments, the standard subject-verb-object semantics. (The direct
        agent~patient relation is a graded BONUS used for ranking/weighting, not a hard gate -- in 'dog eat
        meat', dog~eat and eat~meat carry the sense; dog~meat may be weaker.) This is the learned generative
        model's likelihood saying the combination is sensible."""
        return self._related(a, ac) and self._related(ac, p)

    def _strong_plausible(self, a, ac, p):
        """STRICTER plausibility (reported secondary metric): all THREE pairs graph-related, including the
        direct agent~patient relation. A higher bar -- the fully-coherent recombination."""
        return self._related(a, ac) and self._related(ac, p) and self._related(a, p)

    def _contradicts(self, a, ac, p):
        """Non-contradiction: the proposal must not contradict a fact the brain was EXPLICITLY TOLD IS FALSE.
        The brain stored some NEGATED facts ('cat does NOT eat fish', polarity=NEGATE). Proposing that exact
        triple AFFIRMATIVELY contradicts the stored negation -> reject. (An agent doing MULTIPLE plausible
        things is NOT a contradiction -- 'dog plays ball' and 'dog plays toy' can both hold; single-valued
        cue->patient is the wrong assumption and would forbid the very recombinations that constitute
        generation.) Reads the composer's KNOWN store via the no-confab `ask_yes_no` (preserved verbatim):
        a 'no' means the brain was told this triple is false."""
        return self.composer.ask_yes_no(a, ac, p) == "no"

    def _weight_partner(self, partial, candidates):
        """Generative-model sampling weight: a candidate filler's weight is its graph-relatedness (PPMI) to
        every filler already in the partial triple -- so replay is BIASED toward plausible recombinations
        (it samples from the learned generative model's likelihood, not uniformly). Zero-weighted candidates
        (unrelated to the partial) are effectively never drawn -> the bias is the mechanism."""
        w = np.zeros(len(candidates), dtype=np.float64)
        for k, c in enumerate(candidates):
            s = 0.0
            for x in partial:
                s += self.P[self.row[x], self.row[c]]
            w[k] = s
        return w

    def _sample_weighted(self, candidates, weights):
        tot = float(weights.sum())
        if tot <= 0:                          # no graph signal -> fall back to uniform (rare)
            return candidates[int(self.rng.integers(len(candidates)))]
        p = weights / tot
        return candidates[int(self.rng.choice(len(candidates), p=p))]

    def propose(self, n_attempts):
        """Run `n_attempts` generative-replay samples. Each: pick a seed agent, then SAMPLE an action
        weighted by its graph-relatedness to the agent, then SAMPLE a patient weighted by its relatedness to
        {agent, action}. Accept the proposal iff it is NOVEL (never stored), PLAUSIBLE (gate i) and
        NON-CONTRADICTORY (gate ii). Returns the list of accepted HYPOTHESIS triples + bookkeeping."""
        accepted, attempts_novel, plausible_seen = [], 0, 0
        seen = set()
        for _ in range(n_attempts):
            a = self.agents[int(self.rng.integers(len(self.agents)))]
            ac = self._sample_weighted(self.actions, self._weight_partner((a,), self.actions))
            p = self._sample_weighted(self.patients, self._weight_partner((a, ac), self.patients))
            triple = (a, ac, p)
            if triple in self.all_stored:
                continue                       # replay can resurface a known fact; only NOVEL ones count
            attempts_novel += 1
            if self._plausible(a, ac, p):
                plausible_seen += 1
            if triple in seen:
                continue
            if self._plausible(a, ac, p) and not self._contradicts(a, ac, p):
                accepted.append(triple)
                seen.add(triple)
        return {
            "accepted": accepted,
            "n_accepted": len(accepted),
            "n_attempts": n_attempts,
            "n_novel_attempts": attempts_novel,
            "plausible_fraction_of_novel": plausible_seen / max(1, attempts_novel),
        }


# ===========================================================================
# The RANDOM-RECOMBINATION baseline (chance plausibility) -- uniform vocab triples.
# ===========================================================================
def random_recombination(proposer, n_attempts, rng):
    """Random vocab triples (uniform over the role pools, NO graph bias). Measures the chance rate at which
    a uniformly-sampled novel triple happens to be plausible -- the floor the brain's biased replay must
    beat. Same plausibility + novelty bookkeeping as propose()."""
    plausible, novel = 0, 0
    accepted = set()
    for _ in range(n_attempts):
        a = proposer.agents[int(rng.integers(len(proposer.agents)))]
        ac = proposer.actions[int(rng.integers(len(proposer.actions)))]
        p = proposer.patients[int(rng.integers(len(proposer.patients)))]
        if (a, ac, p) in proposer.all_stored:
            continue
        novel += 1
        if proposer._plausible(a, ac, p) and not proposer._contradicts(a, ac, p):
            plausible += 1
            accepted.add((a, ac, p))
    return {
        "n_plausible_novel": plausible,
        "n_novel_attempts": novel,
        "plausible_fraction_of_novel": plausible / max(1, novel),
        "n_distinct_accepted": len(accepted),
    }


# ===========================================================================
# The SHUFFLED-GRAPH control -- the load-bearing anti-cheat.
# ===========================================================================
def shuffle_graph(P, rng):
    """Shuffle the learned edges: randomly permute the OFF-DIAGONAL entries of the symmetric PPMI matrix.
    The marginal edge-weight distribution is preserved (same set of values), but every word's NEIGHBORHOOD
    is destroyed -> the category structure is gone. If the proposer's plausibility advantage survives this,
    it was a string/template artifact, not the learned structure. PASS = the shuffled-graph proposer's
    plausible-fraction collapses to ~the random-recombination floor."""
    Nm = P.shape[0]
    iu = np.triu_indices(Nm, k=1)
    vals = P[iu].copy()
    rng.shuffle(vals)
    Ps = np.zeros_like(P)
    Ps[iu] = vals
    Ps = Ps + Ps.T          # re-symmetrize
    return Ps


# ===========================================================================
# Build the stored knowledge graph: ~N interlinked PLAUSIBLE SVO facts (the brain was TOLD these).
# ===========================================================================
def enumerate_plausible(agents, actions, patients, P, row, tau):
    """All plausible (a, ac, p) triples (the recombination space) under SELECTIONAL-PREFERENCE plausibility
    (agent~action AND action~patient) -- matches GenerativeReplayProposer._plausible."""
    def related(w1, w2):
        return P[row[w1], row[w2]] >= tau
    out = []
    for a in agents:
        for ac in actions:
            if not related(a, ac):
                continue
            for p in patients:
                if related(ac, p):
                    out.append((a, ac, p))
    return out


def build_stored_facts(agents, actions, patients, P, row, tau, n_facts, n_negated, rng):
    """Draw the brain's experienced facts from the plausible recombination space: `n_facts` AFFIRMED facts +
    `n_negated` NEGATED facts ('the brain was told X does NOT Y'). Facts may SHARE an (agent, action) cue
    (an agent plausibly does several things -- 'dog plays ball' AND 'dog plays toy' -- so the world is NOT
    single-valued; that is what gives generative replay a rich recombination space to discover). The NEGATED
    facts are also plausible recombinations (so the proposer would otherwise be tempted by them) -- they are
    the non-contradiction gate's real work. Returns (affirmed, negated, plausible_all). The affirmed +
    negated sets are DISJOINT and together a strict subset of plausible_all, so the discoverable
    novel-plausible universe (plausible_all minus all stored) is non-empty and large."""
    plausible_all = enumerate_plausible(agents, actions, patients, P, row, tau)
    rng.shuffle(plausible_all)
    need = n_facts + n_negated
    chosen = plausible_all[:min(need, len(plausible_all))]
    affirmed = chosen[:n_facts]
    negated = chosen[n_facts:n_facts + n_negated]
    return affirmed, negated, plausible_all


def run_seed(seed, vocab, cat_ids, corpus, a):
    rng = np.random.default_rng(seed)
    agents, actions, patients = _category_pools(TAXONOMY_8x8)
    P, row = build_plausibility(corpus, vocab)

    # tau = a percentile over the POSITIVE PPMI values (a learned threshold, not fit to the test). A pair is
    # graph-related if its PPMI is in the top (100 - tau_pct)% of observed positive relations.
    pos = P[P > 0]
    tau = float(np.percentile(pos, a.tau_pct)) if pos.size else 0.0

    affirmed, negated, plausible_all = build_stored_facts(
        agents, actions, patients, P, row, tau, a.n_facts, a.n_negated, rng)
    n_stored = len(affirmed)
    all_stored = set(affirmed) | set(negated)
    # the ceiling: distinct plausible-novel triples that EXIST to be discovered (plausible, never stored in
    # either polarity, and NOT one of the explicitly-negated facts -> the non-contradictory novel-plausible set)
    plausible_novel_universe = sorted(set(plausible_all) - all_stored)

    # ---- store the facts in the BRAIN (RF composer; the no-confab moat intact) ----
    # AFFIRMED facts (polarity=AFFIRM) + NEGATED facts (polarity=NEGATE = 'the brain was told this is FALSE').
    comp = RFPhasorComposer(seed=seed, D=a.D, vocab=vocab)
    for ag, ac, pt in affirmed:
        comp.store(ag, ac, pt, polarity="AFFIRM")
    for ag, ac, pt in negated:
        comp.store(ag, ac, pt, polarity="NEGATE")

    # ---- (1) the brain's GENERATIVE-REPLAY proposer ----
    proposer = GenerativeReplayProposer(comp, affirmed, negated, P, row, tau,
                                        np.random.default_rng(seed * 7 + 1))
    rep = proposer.propose(a.n_attempts)
    # NON-CONTRADICTION effectiveness: the proposer must NEVER propose a triple the brain was told is FALSE
    # (a negated fact). Each negated fact is itself a plausible recombination -> a tempting nonsense the gate
    # must catch. Count any accepted proposal that is a negated fact (must be 0 -- the gate is load-bearing).
    proposed_set = set(rep["accepted"])
    contradictions_proposed = len(proposed_set & set(negated))
    # novel-composition score: distinct novel-plausible triples PROPOSED, normalized by the discoverable
    # universe (capped at 1.0). The measured retrieval baseline is 0.0 (it proposes NONE).
    novel_proposed = [t for t in rep["accepted"]]
    n_novel_proposed = len(novel_proposed)
    novel_comp_score = min(1.0, n_novel_proposed / max(1, len(plausible_novel_universe)))
    # the proposer's accepted set is plausible by gate-construction; how many are STRONGLY plausible (all 3
    # pairs related) is the higher-bar quality read.
    n_strong = sum(1 for t in novel_proposed if proposer._strong_plausible(*t))
    replay_frac = rep["plausible_fraction_of_novel"]   # fraction of NOVEL samples that pass the full gate

    # ---- (2) RANDOM-RECOMBINATION baseline (chance plausibility) ----
    randb = random_recombination(proposer, a.n_attempts, np.random.default_rng(seed * 13 + 3))
    random_frac = randb["plausible_fraction_of_novel"]
    # the plausibility ADVANTAGE = how many times more often the brain's biased replay lands a plausible
    # novel triple than uniform random recombination (the scale-robust measure; absolute fractions are both
    # small under conjunctive plausibility, so the RATIO is the honest advantage signal).
    advantage_ratio = replay_frac / max(random_frac, 1.0 / max(1, randb["n_novel_attempts"]))

    # ---- (3) SHUFFLED-GRAPH control (learned structure must be load-bearing) ----
    P_shuf = shuffle_graph(P, np.random.default_rng(seed * 17 + 5))
    # the shuffled threshold from the shuffled positive values (same marginal, destroyed neighborhoods)
    pos_s = P_shuf[P_shuf > 0]
    tau_s = float(np.percentile(pos_s, a.tau_pct)) if pos_s.size else 0.0
    proposer_shuf = GenerativeReplayProposer(comp, affirmed, negated, P_shuf, row, tau_s,
                                             np.random.default_rng(seed * 19 + 7))
    rep_shuf = proposer_shuf.propose(a.n_attempts)
    # The shuffled proposer believes its (shuffled) graph -> it accepts triples its shuffled-belief deems
    # plausible. Score those acceptances under the TRUE graph: a shuffled proposal is only REALLY plausible
    # if the TRUE graph agrees. If the learned structure is load-bearing, the shuffled proposer's TRUE
    # plausibility collapses to the random floor (its plausibility belief is now uncorrelated with reality).
    shuf_true_plausible = sum(1 for t in rep_shuf["accepted"] if proposer._plausible(*t))
    shuf_plausible_frac = shuf_true_plausible / max(1, rep_shuf["n_accepted"]) if rep_shuf["n_accepted"] else 0.0
    # ALSO measure the shuffled proposer's plausible-fraction-of-novel-attempts under the TRUE graph (the
    # apples-to-apples comparison to replay_frac / random_frac): does shuffling kill the advantage?
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
        if proposer._plausible(ag, acn, ptn):       # TRUE-graph plausibility of a shuffled-replay sample
            shuf_true_novel_plausible += 1
    shuf_true_frac = shuf_true_novel_plausible / max(1, a.n_attempts)

    # ---- (4) MOAT-HONESTY check: a proposal must NEVER pass the composer's KNOWN-fact retrieval ----
    # The proposals are HYPOTHESES. The brain must still distinguish "I know X" from "X is plausible":
    # query the composer's known-fact channel on every proposed triple -> it must ABSTAIN (it was never
    # stored). A single leak = the moat broke (a proposal masquerading as a known fact). The proposer is a
    # SEPARATE flagged channel; the known-fact moat is preserved verbatim.
    moat_leaks = 0
    for ag, ac, pt in novel_proposed:
        known = comp.query_patient(ag, ac)            # must be None OR != pt (the cue was never stored as this)
        yn = comp.ask_yes_no(ag, ac, pt)              # must be 'unknown' (never stored -> not a known fact)
        if known == pt:
            moat_leaks += 1
        if yn == "yes":
            moat_leaks += 1
    # ALSO: the composer's own held-out abstention floor still holds on UNSTORED cues (sanity on the store)
    n_ab, ab_ok = 0, 0
    pos_cues = set(proposer.stored_cues)
    guard = 0
    while n_ab < 20 and guard < 100000:
        guard += 1
        ag = agents[int(rng.integers(len(agents)))]
        ac = actions[int(rng.integers(len(actions)))]
        if (ag, ac) in pos_cues:
            continue
        n_ab += 1
        ab_ok += int(comp.query_patient(ag, ac) is None)

    # sample plausibility judgment: a few proposed triples, with their PPMI pair-strengths, for the record
    samples = []
    for t in novel_proposed[:8]:
        ag, ac, pt = t
        samples.append({"triple": f"{ag} {ac} {pt}",
                        "ppmi_agent_action": round(float(P[row[ag], row[ac]]), 3),
                        "ppmi_action_patient": round(float(P[row[ac], row[pt]]), 3),
                        "ppmi_agent_patient": round(float(P[row[ag], row[pt]]), 3)})

    print(f"\n[b2 seed {seed}] stored {n_stored} plausible SVO facts | discoverable novel-plausible "
          f"universe {len(plausible_novel_universe)} | tau(P{a.tau_pct})={tau:.3f}", flush=True)
    print(f"  (1) GENERATIVE REPLAY: proposed {n_novel_proposed} distinct NOVEL-plausible triples "
          f"({n_strong} strongly-plausible) | novel-comp score {novel_comp_score:.3f} (vs retrieval 0.0) | "
          f"plausible-frac-of-novel {replay_frac:.3f}", flush=True)
    print(f"  (2) RANDOM RECOMBINATION: plausible-frac-of-novel {random_frac:.4f} "
          f"({randb['n_distinct_accepted']} distinct) -> ADVANTAGE ratio {advantage_ratio:.1f}x", flush=True)
    print(f"  (3) SHUFFLED GRAPH: plausible-frac-of-novel (TRUE graph) {shuf_true_frac:.4f} "
          f"(must collapse toward random {random_frac:.4f}, vs replay {replay_frac:.3f}); accepted-set TRUE "
          f"plausibility {shuf_plausible_frac:.3f}", flush=True)
    print(f"  (4) MOAT HONESTY: proposal->known-fact leaks {moat_leaks} (must be 0) | "
          f"store abstention floor {ab_ok}/{n_ab} | negated-facts re-proposed {contradictions_proposed} "
          f"(must be 0)", flush=True)
    if samples:
        print(f"  novel-plausible examples: {[s['triple'] for s in samples]}", flush=True)

    return {
        "seed": seed,
        "n_stored": n_stored,
        "tau": tau,
        "discoverable_novel_plausible_universe": len(plausible_novel_universe),
        # (1)
        "novel_composition_score": novel_comp_score,
        "n_novel_proposed": n_novel_proposed,
        "n_strongly_plausible": n_strong,
        "replay_plausible_fraction_of_novel": replay_frac,
        "novel_proposed_examples": [f"{x[0]} {x[1]} {x[2]}" for x in novel_proposed[:20]],
        # (2)
        "random_plausible_fraction_of_novel": random_frac,
        "random_distinct_plausible": randb["n_distinct_accepted"],
        "advantage_ratio": advantage_ratio,
        # (3)
        "shuffled_n_accepted": rep_shuf["n_accepted"],
        "shuffled_true_plausible_fraction_accepted": shuf_plausible_frac,
        "shuffled_true_plausible_fraction_of_novel": shuf_true_frac,
        # (4)
        "moat_leaks": moat_leaks,
        "contradictions_proposed": contradictions_proposed,
        "n_negated": len(negated),
        "store_abstention_correct": ab_ok,
        "store_abstention_attempted": n_ab,
        "samples": samples,
    }


def decide_verdict(rows, a):
    """GO iff, across ALL seeds: (1) the proposer scores novel-composition > 0 (beats the measured 0.0
    retrieval baseline) and proposes >= min_novel distinct novel-plausible triples; (2) the brain's replay
    has a clear PLAUSIBILITY ADVANTAGE over random recombination (advantage_ratio >= advantage_bar -- the
    scale-robust measure, since conjunctive plausibility makes both absolute fractions small); (3) the
    SHUFFLED-graph control COLLAPSES the replay's TRUE-graph plausibility toward the random floor (the
    learned structure is load-bearing, not a string/template artifact); and (4) the PROPOSER-MOAT is
    preserved -- 0 proposal->known-fact leaks AND 0 explicitly-negated facts re-proposed (the load-bearing
    moat: a hypothesis never passes as a stored fact, and an explicitly-false recombination is never
    proposed). The COMPOSER's own baseline abstention floor on random unstored cues is reported as a SANITY
    metric (tolerance >= store_floor_bar = the documented RF code-fidelity tail at small D; a single
    false-accept out of 20 is the known tail, NOT a b2-mechanism failure) -- it does NOT gate the b2 verdict.
    Else HONEST-NEGATIVE + why."""
    def col(k):
        return [r[k] for r in rows]

    replay_frac = np.array(col("replay_plausible_fraction_of_novel"))
    rand_frac = np.array(col("random_plausible_fraction_of_novel"))
    shuf_frac = np.array(col("shuffled_true_plausible_fraction_of_novel"))
    adv = np.array(col("advantage_ratio"))
    novel_score = np.array(col("novel_composition_score"))
    n_novel = np.array(col("n_novel_proposed"))
    leaks = np.array(col("moat_leaks"))
    contra = np.array(col("contradictions_proposed"))
    ab_ok = np.array(col("store_abstention_correct"))
    ab_att = np.array(col("store_abstention_attempted"))

    adv_bar = float(a.advantage_bar)            # replay must out-plausible random by >= this ratio
    min_novel = int(a.min_novel)                # >= this many distinct novel-plausible triples proposed
    # the shuffled control "collapses" if shuffling removes most of the replay advantage: the shuffled-graph
    # replay's TRUE plausible-fraction must drop to <= collapse_frac of the real replay's (so the advantage
    # is destroyed by losing the learned neighborhoods).
    collapse_frac = float(a.shuffle_collapse_frac)

    novel_above_zero_all = bool(np.all(n_novel >= min_novel) and np.all(novel_score > 0.0))
    advantage_all = bool(np.all(adv >= adv_bar))
    # per-seed: shuffled TRUE-frac <= collapse_frac * replay-frac (advantage destroyed by the shuffle)
    shuffled_collapses_all = bool(np.all(shuf_frac <= collapse_frac * np.maximum(replay_frac, 1e-9)))
    # the PROPOSER-MOAT (the load-bearing b2 gate) is preserved iff: 0 proposal->known-fact leaks (a proposal
    # never passes as a stored fact) AND 0 negated facts re-proposed (the non-contradiction gate caught every
    # explicitly-false recombination).
    moat_preserved_all = bool(np.all(leaks == 0) and np.all(contra == 0))
    # the COMPOSER's own baseline abstention floor on random unstored cues (a SANITY metric, NOT a b2 gate):
    # the documented RF code-fidelity tail at small D allows a single false-accept out of 20 (>= store_floor_bar).
    store_floor_rate = ab_ok / np.maximum(ab_att, 1)
    store_floor_ok_all = bool(np.all(store_floor_rate >= float(a.store_floor_bar)))

    detail = {
        "replay_plausible_fraction_mean": float(replay_frac.mean()),
        "random_plausible_fraction_mean": float(rand_frac.mean()),
        "shuffled_true_plausible_fraction_mean": float(shuf_frac.mean()),
        "advantage_ratio_mean": float(adv.mean()),
        "advantage_ratio_min": float(adv.min()),
        "novel_composition_score_mean": float(novel_score.mean()),
        "n_novel_proposed_mean": float(n_novel.mean()),
        "n_novel_proposed_min": int(n_novel.min()),
        "n_strongly_plausible_mean": float(np.mean(col("n_strongly_plausible"))),
        "moat_leaks_total": int(leaks.sum()),
        "contradictions_proposed_total": int(contra.sum()),
        "store_abstention_floor_rate_mean": float(store_floor_rate.mean()),
        "store_abstention_floor_rate_min": float(store_floor_rate.min()),
        "store_abstention_floor_ok_all": store_floor_ok_all,   # sanity (>= store_floor_bar), NOT a b2 gate
        "novel_above_zero_all_seeds": novel_above_zero_all,
        "advantage_all_seeds": advantage_all,
        "shuffled_collapses_all_seeds": shuffled_collapses_all,
        "moat_preserved_all_seeds": moat_preserved_all,
        "advantage_bar": adv_bar,
        "min_novel_bar": min_novel,
        "shuffle_collapse_frac_bar": collapse_frac,
        "store_floor_bar": float(a.store_floor_bar),
    }

    # The b2 GO gates on the LOAD-BEARING proposer-moat (leaks + negated re-proposed), NOT the composer's
    # baseline abstention floor (a separate, documented RF code-fidelity property reported as sanity).
    if novel_above_zero_all and advantage_all and shuffled_collapses_all and moat_preserved_all:
        verdict = "GO"
    elif not moat_preserved_all:
        verdict = "HONEST_NEGATIVE_moat_broken"
    elif not novel_above_zero_all:
        verdict = "HONEST_NEGATIVE_no_novel_proposed"
    elif not advantage_all:
        verdict = "HONEST_NEGATIVE_no_plausibility_advantage"
    elif not shuffled_collapses_all:
        verdict = "HONEST_NEGATIVE_structure_not_load_bearing"
    else:
        verdict = "HONEST_NEGATIVE_other"
    return verdict, detail


def main():
    p = argparse.ArgumentParser(description="Generative-frontier (b2) generative-replay de-risk "
                                            "(can the brain INVENT novel-but-plausible propositions?)")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--D", type=int, default=64, help="phasor dimension for the RF composer store")
    p.add_argument("--n-facts", type=int, default=24, help="AFFIRMED interlinked SVO facts the brain is TOLD")
    p.add_argument("--n-negated", type=int, default=12,
                   help="NEGATED facts ('X does NOT Y') -- the non-contradiction gate's real work")
    p.add_argument("--n-attempts", type=int, default=3000, help="generative-replay samples per channel")
    p.add_argument("--tau-pct", type=float, default=50.0,
                   help="graph-related threshold = this percentile of positive PPMI values")
    p.add_argument("--advantage-bar", type=float, default=3.0,
                   help="replay plausible-frac must be >= this RATIO of the random baseline (scale-robust)")
    p.add_argument("--min-novel", type=int, default=3,
                   help="proposer must emit >= this many distinct novel-plausible triples")
    p.add_argument("--shuffle-collapse-frac", type=float, default=0.5,
                   help="shuffled-graph TRUE plausible-frac must drop to <= this fraction of the real replay's")
    p.add_argument("--store-floor-bar", type=float, default=0.95,
                   help="composer baseline abstention-floor tolerance (sanity only; documented RF tail)")
    p.add_argument("--max-bytes", type=int, default=4_000_000,
                   help="bytes of TinyStories to read for the co-occurrence graph (CPU speed)")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--repeat-cap", type=int, default=40)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")

    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    t0 = time.time()
    print(f"[genfrontier-b2] seeds={seeds} D={a.D} n_facts={a.n_facts} n_attempts={a.n_attempts} "
          f"tau_pct={a.tau_pct} -- can the BRAIN'S generative replay INVENT novel-but-plausible "
          f"propositions (vs the measured 0.0 retrieval baseline)?", flush=True)

    vocab, cat_ids, cat_names = taxonomy_to_vocab_categories(TAXONOMY_8x8)
    corpus_path = os.path.join(_REPO, "data", "corpus", "tinystories.txt")
    if not os.path.exists(corpus_path):
        print(f"[ERROR] corpus not found: {corpus_path}", flush=True)
        sys.exit(2)
    # build the co-occurrence corpus ONCE (seed-independent; the plausibility graph is shared) -- the brain's
    # learned structure of the language it heard.
    corpus = build_real_cooccurrence(corpus_path, vocab, cat_ids, window=a.window, repeat_cap=a.repeat_cap,
                                     seed=42, max_bytes=a.max_bytes, freq_floor=30,
                                     min_facts_per_category=20, verbose=True)

    rows = [run_seed(s, vocab, cat_ids, corpus, a) for s in seeds]
    verdict, detail = decide_verdict(rows, a)

    print(f"\n{'='*98}", flush=True)
    print(f"  OVERALL VERDICT: {verdict}", flush=True)
    print(f"  (1) novel-composition score (vs measured 0.0 RETRIEVAL baseline): mean "
          f"{detail['novel_composition_score_mean']:.3f}  (>0 all seeds: {detail['novel_above_zero_all_seeds']}; "
          f"min {detail['n_novel_proposed_min']} novel proposed, mean {detail['n_strongly_plausible_mean']:.1f} "
          f"strongly-plausible)", flush=True)
    print(f"  (2) PLAUSIBILITY ADVANTAGE: replay-frac {detail['replay_plausible_fraction_mean']:.3f} vs random "
          f"{detail['random_plausible_fraction_mean']:.4f}  -> ratio {detail['advantage_ratio_mean']:.1f}x "
          f"(min {detail['advantage_ratio_min']:.1f}x); >= {detail['advantage_bar']}x all seeds: "
          f"{detail['advantage_all_seeds']}", flush=True)
    print(f"  (3) SHUFFLED-GRAPH TRUE plausible-frac {detail['shuffled_true_plausible_fraction_mean']:.4f} "
          f"-> collapses (<= {detail['shuffle_collapse_frac_bar']}x replay) all seeds: "
          f"{detail['shuffled_collapses_all_seeds']}", flush=True)
    print(f"  (4) PROPOSER-MOAT: {detail['moat_leaks_total']} proposal->known leaks + "
          f"{detail['contradictions_proposed_total']} negated re-proposed "
          f"(preserved all seeds: {detail['moat_preserved_all_seeds']})", flush=True)
    print(f"      [sanity] composer baseline abstention floor: mean "
          f"{detail['store_abstention_floor_rate_mean']:.3f} min {detail['store_abstention_floor_rate_min']:.3f} "
          f"(>= {detail['store_floor_bar']:.2f} all seeds: {detail['store_abstention_floor_ok_all']}; the "
          f"documented RF code-fidelity tail, NOT a b2 gate)", flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*98}\n", flush=True)

    out = {
        "probe": "genfrontier_b2_generative_replay",
        "verdict": verdict,
        "seeds": seeds,
        "config": {"D": a.D, "n_facts": a.n_facts, "n_negated": a.n_negated, "n_attempts": a.n_attempts,
                   "tau_pct": a.tau_pct, "advantage_bar": a.advantage_bar, "min_novel": a.min_novel,
                   "shuffle_collapse_frac": a.shuffle_collapse_frac, "store_floor_bar": a.store_floor_bar,
                   "max_bytes": a.max_bytes, "window": a.window},
        "baseline_to_beat": {"measured_retrieval_novel_composition": 0.0,
                             "source": "2026-06-22-generation-novelty-categorical-gap-MEASURED.md"},
        "detail": detail,
        "per_seed": rows,
        "brain_based_note": (
            "the LEARNED graph (plausibility/likelihood) is the project's PPMI co-occurrence cortex over REAL "
            "TinyStories (option_c_real_cooccurrence_derisk.build_real_cooccurrence); the KNOWN-fact store + the "
            "no-confab moat are the RF phasor composer (rf_phasor_composer.RFPhasorComposer). The proposer "
            "RESAMPLES role-filler bindings from the learned graph (hippocampal generative replay, catalog G.09 "
            "constructive imagination), gated by graph-plausibility + non-contradiction, and FLAGS proposals as "
            "HYPOTHESES (the known-fact moat is preserved verbatim -- a proposal never passes as a stored fact). "
            "The host computes the recombination bookkeeping; the plausibility signal + the fact store are the "
            "brain's. NO sim/ edit; reuse-by-import; CPU."),
        "elapsed_total_s": time.time() - t0,
    }
    if a.out is None:
        a.out = os.path.join(_REPO, "research", "findings", "raw",
                             "_genfrontier_b2_generative_replay_derisk.json")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {a.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
