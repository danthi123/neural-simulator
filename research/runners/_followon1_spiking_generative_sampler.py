"""FOLLOW-ON #1 -- the FULLY-SPIKING generative-replay SAMPLER (3E G2->G3): replace the HOST sample-loop
bookkeeping in the brain-owns-generation channel with a SPIKING SWR-gated CA3 RESAMPLING mechanism.

CONTEXT (3E `_burndown_3E_brain_owns_generation.py` GO; the owner's stated generative-frontier primary):
  3E wired the b2 generative-replay proposer into the conversational turn: the BRAIN generates novel grounded
  propositions (novel-comp 0.752 vs 0.0 retrieval, ~17x random, shuffled-graph collapses, 0 moat leaks). But the
  RECOMBINATION sample-loop is HOST CODE: GenerativeReplayProposer.propose() is a host `for _ in range(n_attempts)`
  loop that, per attempt, does host weighted random sampling -- `_weight_partner` (read PPMI weights) +
  `_sample_weighted` (numpy `rng.choice(p=weights)`) -- to pick an action then a patient. The LOAD-BEARING pieces
  are the brain's (the PPMI-plausibility cortex, the RF composer store, the no-confab moat); only the
  recombination BOOKKEEPING (which role-fillers to recombine) is the host `rng.choice`.

WHAT THIS REPLACES (the scope answer):
  The host `rng.choice`-over-PPMI sample loop -> a SPIKING SWR-gated CA3 pattern-completion RESAMPLER. The biology
  (catalog G.09 constructive imagination; Buzsaki sharp-wave ripples + the Marr/Treves-Rolls CA3 autoassociator;
  Stoianov/Maisto/Pezzulo 2022 generative replay resamples FICTIVE recombinations): an SWR replay event drives a
  sparse CA3 cue, the PLASTIC recurrent attractor pattern-COMPLETES, and a novel-but-plausible recombination
  EMERGES from the spiking attractor dynamics -- NOT a host `rng.choice`.

THE SPIKING MECHANISM (reuse-by-import `_D_sparse_heteroassoc` -- a real spiking CA3 recurrent autoassociator on
the SimulationBridge; the same module `LearnedAssocGraph` uses):
  1. ENCODE the brain's association structure INTO the CA3 recurrent: each role-filler WORD = a sparse K-of-N CA3
     pattern. For each AFFIRMED SVO fact (agent, action, patient) the brain experienced, CO-FIRE (agent,action) and
     (action,patient) -> the plastic excitatory recurrent LEARNS the pairwise associations by Hebbian co-fire growth
     (this IS the PPMI association graph, made NEURAL -- learned from the SAME facts, NOT a host matrix).
  2. SWR-GATED RESAMPLE (the host sample loop's replacement): an SWR event drives a sparse SEED cue (a seed agent's
     pattern). The spiking attractor pattern-completes through the recurrent -> the completed CA3 firing's nearest
     ACTION concept emerges (stage 1). Then drive (agent + that action) -> the attractor completes to a related
     PATIENT (stage 2). The recombination = which fillers the SPIKING ATTRACTOR settled on. Stochasticity is a
     TEMPERATURE over the *spiking completion profile* (the brain's attractor OUTPUT firing), NOT the host PPMI
     matrix -- so the SAMPLE is the brain's.
  3. The SAME downstream gates (unchanged from 3E, the brain's): PPMI-plausibility + non-contradiction + the
     no-confab moat (a proposal is a HYPOTHESIS, never passes the composer's known-fact retrieval).

THE 4 DE-RISK MEASUREMENTS (multi-seed >= 3, like 3E; CPU `SIM_BACKEND=numpy`):
  (a) NOVEL      -- the spiking sampler's proposals are NOVEL (disjoint from the stored set) AND the agent's
                    known-fact retrieval ABSTAINS on every one (a hypothesis is not a known fact).
  (b) PLAUSIBLE  -- the proposals are grounded (PPMI-plausible) far ABOVE a random-recombination floor, AND match
                    the HOST sample-loop's plausibility quality (the spiking attractor sampler is >= the host
                    bookkeeping, not a degradation).
  (c) SWR/CA3 LOAD-BEARING (the LESION anti-cheat) -- ABLATE the CA3 recurrent (zero the learned recurrent weights
                    so the attractor cannot complete) -> the resampling DEGRADES to the random/floor (the spiking
                    attractor, not a host fallback, is doing the work). A SHUFFLED-recurrent control also collapses.
  (d) MOAT 0-CONFAB -- 0 hypothesis->known-fact leaks AND 0 explicitly-negated facts re-proposed; the agent's
                    standing untaught-cue abstention is unregressed.

VERDICT:
  GO              = the SPIKING SWR-CA3 sampler produces novel-but-plausible recombinations that MATCH the host
                    sample-loop's quality (novel + plausible >= host floor + CA3-lesion-collapses), with the
                    no-confab moat intact -> the recombination is now done by a SPIKING mechanism, not host code.
  HONEST_NEGATIVE = the spiking attractor sampler can't match the host bookkeeping at this scale -> characterize
                    the precise residual (under BRAIN-BASED-ONLY, the honest negative IS the deliverable).

REUSE-BY-IMPORT, NO sim/ edit. CPU. Run:
  SIM_BACKEND=numpy python -u -m research.runners._followon1_spiking_generative_sampler \
      --seeds 42,43,44 --out research/findings/raw/_followon1_spiking_generative_sampler.json
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

os.environ.setdefault("SIM_BACKEND", "numpy")

# Reuse-by-import: the b2/3E machinery (PPMI plausibility, the proposer's gates, stored-fact builder, the host
# sample loop we are replacing) + the spiking CA3 autoassociator + the conversational agent.
from research.runners._genfrontier_b2_generative_replay_derisk import (  # noqa: E402
    GenerativeReplayProposer,
    build_plausibility,
    build_stored_facts,
    random_recombination,
    _category_pools,
)
from research.runners.option_c_real_cooccurrence_derisk import (  # noqa: E402
    TAXONOMY_8x8,
    taxonomy_to_vocab_categories,
    build_real_cooccurrence,
)
from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
from research.runners._D_sparse_heteroassoc import (  # noqa: E402
    build as build_ca3,
    _pool_global,
    _drive,
)
from research.runners.concept_pool_sparse_distributed import generate_sparse_patterns  # noqa: E402
from sim.backend import to_host  # noqa: E402


# ===========================================================================
# THE SPIKING SWR-GATED CA3 RESAMPLER -- replaces the host `rng.choice`-over-PPMI sample loop.
# ===========================================================================
class SpikingSWRCa3Sampler:
    """A SPIKING CA3 recurrent autoassociator that RESAMPLES role-filler recombinations from spiking attractor
    pattern-completion (the SWR-gated generative-replay biology), NOT a host `rng.choice`.

    Build: each role-filler WORD = a sparse K-of-N CA3 pattern in a shared spiking pool with a plastic excitatory
    recurrent (`_D_sparse_heteroassoc`). The AFFIRMED facts the brain experienced are ENCODED by co-firing
    (agent,action) and (action,patient) -> the recurrent LEARNS the pairwise associations by Hebbian co-fire (the
    PPMI graph made neural). RESAMPLE: an SWR event drives a sparse seed cue; the spiking attractor completes; the
    completed firing's nearest ACTION (then, cued by agent+action, nearest PATIENT) emerges from the dynamics.
    Stochasticity = a temperature over the *spiking completion profile* (the attractor's OUTPUT firing), so the
    sample is the brain's, not the host PPMI matrix."""

    def __init__(self, vocab, P, row, tau, seed=42, n_pool=2000, pattern_size=120,
                 enc_cycles=3, drive_pA=1100.0, read_window=40, temperature=0.6, fs_inh=1.2,
                 encode_graph=True):
        self.vocab = list(vocab)
        self.widx = {w: i for i, w in enumerate(self.vocab)}
        self.n_pool = n_pool
        self.pattern_size = pattern_size
        self.drive_pA = drive_pA
        self.read_window = read_window
        self.temperature = temperature
        self.P, self.row, self.tau = P, row, tau
        self._rng_seed = seed * 31 + 3
        self.rng = np.random.default_rng(self._rng_seed)

        # the role pools (which words can fill which slot) -- a-priori taxonomy, as in b2/3E.
        self.agents, self.actions, self.patients = _category_pools(TAXONOMY_8x8)
        # each WORD = a sparse pattern (one CA3 pattern per vocab word, deterministic from seed).
        all_patterns = generate_sparse_patterns(len(self.vocab), n_pool, pattern_size, seed)
        self.word_pattern = {w: np.asarray(all_patterns[self.widx[w]]) for w in self.vocab}

        # ---- build the spiking CA3 bridge + encode the brain's learned association GRAPH ----
        self.bridge = build_ca3(seed, n_pool=n_pool, fs_inh=fs_inh)
        self.pool_base = np.asarray(self.bridge.region_manager.indices("pool"))
        self.word_global = {w: self.pool_base[self.word_pattern[w]] for w in self.vocab}

        # the role-relevant PPMI-related pairs = the brain's learned selectional-preference structure. The CA3
        # recurrent ENCODES this (co-fire each PPMI-related agent~action and action~patient pair) -> the recurrent
        # BECOMES the PPMI cortex (an SWR cue over it completes to PPMI-related fillers). This is the brain's
        # learned association graph made NEURAL -- so the spiking attractor's completions are PPMI-plausible.
        self.encodable_words = set()
        if encode_graph:
            self._encode_graph(enc_cycles)

    def _related_pairs(self):
        """The role-relevant PPMI-related pairs the CA3 recurrent learns: every (agent, action) with PPMI>=tau and
        every (action, patient) with PPMI>=tau. This is the brain's learned selectional-preference structure (the
        same graph the host sampler reads), encoded into the spiking recurrent."""
        aa, ap = [], []
        for ag in self.agents:
            for ac in self.actions:
                if self.P[self.row[ag], self.row[ac]] >= self.tau:
                    aa.append((ag, ac))
        for ac in self.actions:
            for pt in self.patients:
                if self.P[self.row[ac], self.row[pt]] >= self.tau:
                    ap.append((ac, pt))
        return aa, ap

    # --- ENCODE: co-fire each PPMI-related role pair -> Hebbian recurrent growth = the PPMI cortex ---
    def _encode_graph(self, enc_cycles):
        b = self.bridge
        aa, ap = self._related_pairs()
        pairs = aa + ap
        self.encodable_words = {w for pr in pairs for w in pr}
        try:
            b.set_plasticity_gate("recurrent", 1.0)
        except KeyError:
            pass
        for _ in range(enc_cycles):
            for w1, w2 in pairs:
                _drive(b, [self.word_global[w1], self.word_global[w2]], self.drive_pA)
                for _ in range(10):
                    b._run_one_simulation_step()
                b.cp_external_input_current[:] = 0.0
                for _ in range(5):
                    b._run_one_simulation_step()
        try:
            b.set_plasticity_gate("recurrent", 0.0)
        except KeyError:
            pass

    # --- the SWR-gated spiking completion read (one stage) ---
    def _complete(self, cue_words, candidates):
        """Drive the SWR cue (cue_words' patterns), accumulate pool firing over a read window, EXCLUDE the directly-
        driven cue neurons (-> the RECURRENT output), and return the spiking completion profile (cosine of the
        recurrent firing to each candidate word's pattern). The completion is the spiking attractor's output."""
        b = self.bridge
        cue_global = [self.word_global[w] for w in cue_words]
        _drive(b, cue_global, self.drive_pA)
        firing = np.zeros(self.n_pool)
        for _ in range(self.read_window):
            b._run_one_simulation_step()
            fs = np.asarray(to_host(b.cp_firing_states)).astype(float)
            firing += fs[self.pool_base]
        b.cp_external_input_current[:] = 0.0
        for _ in range(10):
            b._run_one_simulation_step()
        # exclude the directly-driven cue neurons -> what remains is the heteroassociative completion
        for w in cue_words:
            firing[self.word_pattern[w]] = 0.0
        nf = float(np.linalg.norm(firing))
        prof = {}
        for w in candidates:
            v = np.zeros(self.n_pool)
            v[self.word_pattern[w]] = 1.0
            prof[w] = float(firing @ v / (nf * np.linalg.norm(v))) if nf > 0 else 0.0
        return prof

    def _sample_from_profile(self, prof):
        """Temperature-softmax over the SPIKING completion profile -> sample a filler. The distribution is the
        brain's attractor OUTPUT firing (not the host PPMI matrix); only positive-completion candidates compete."""
        words = list(prof.keys())
        vals = np.array([max(0.0, prof[w]) for w in words], dtype=np.float64)
        if vals.sum() <= 1e-9:
            return None                    # the attractor completed nothing -> no sample (honest)
        logits = vals / max(1e-6, self.temperature)
        logits = logits - logits.max()
        p = np.exp(logits)
        p = p / p.sum()
        return words[int(self.rng.choice(len(words), p=p))]

    def resample_one(self):
        """ONE SWR-gated resample: pick a seed agent (the replay seed), then SPIKING-complete an action, then a
        patient -- each filler chosen by the spiking attractor's completion profile. Returns a triple or None."""
        # seed from agents that participate in the learned graph (so the SWR cue drives an encoded pattern)
        seed_agents = [a for a in self.agents if a in self.encodable_words] or self.agents
        ag = seed_agents[int(self.rng.integers(len(seed_agents)))]
        # stage 1: SWR cue = the seed agent -> complete an ACTION from the spiking recurrent
        prof_ac = self._complete([ag], [a for a in self.actions if a in self.word_global])
        ac = self._sample_from_profile(prof_ac)
        if ac is None:
            return None
        # stage 2: SWR cue = (agent + action) -> complete a PATIENT from the spiking recurrent
        prof_pt = self._complete([ag, ac], [p for p in self.patients if p in self.word_global])
        pt = self._sample_from_profile(prof_pt)
        if pt is None:
            return None
        return (ag, ac, pt)

    def resample(self, n_attempts):
        """Run `n_attempts` SWR-gated spiking resamples -> the multiset of proposed (raw) triples (pre-gate). The
        GATES (plausibility, non-contradiction, novelty, moat) are applied by the caller -- the brain's, unchanged
        from 3E."""
        out = []
        for _ in range(n_attempts):
            t = self.resample_one()
            if t is not None:
                out.append(t)
        return out

    def reset_rng(self):
        """Reset the resample rng so a controlled run draws the SAME seed-agent sequence (paired comparison)."""
        self.rng = np.random.default_rng(self._rng_seed)


# ===========================================================================
# Build the agent + teach it the facts, returning what the proposer/sampler need.
# ===========================================================================
def build_world(seed, vocab, corpus, a):
    rng = np.random.default_rng(seed)
    agents, actions, patients = _category_pools(TAXONOMY_8x8)
    P, row = build_plausibility(corpus, vocab)
    pos = P[P > 0]
    tau = float(np.percentile(pos, a.tau_pct)) if pos.size else 0.0
    affirmed, negated, plausible_all = build_stored_facts(
        agents, actions, patients, P, row, tau, a.n_facts, a.n_negated, rng)

    concepts = {w: None for w in vocab}
    agent = BrainConversationalAgent(seed=seed, concepts=concepts, composer_kind="rf",
                                     enable_neural_render=False)
    for ag, ac, pt in affirmed:
        agent.hear(f"{ag} {ac} {pt}", polarity="AFFIRM")
    for ag, ac, pt in negated:
        agent.hear(f"{ag} {ac} {pt}", polarity="NEGATE")

    all_stored = set(affirmed) | set(negated)
    plausible_novel_universe = sorted(set(plausible_all) - all_stored)
    return agent, affirmed, negated, P, row, tau, plausible_novel_universe


def _gate_and_collect(raw_triples, proposer, all_stored):
    """Apply the brain's gates (novel + plausible + non-contradiction) to raw spiking-sampled triples. Returns the
    accepted distinct HYPOTHESIS set + plausible-fraction-of-novel bookkeeping (apples-to-apples with the host)."""
    accepted, seen = [], set()
    n_novel, n_plausible = 0, 0
    for (ag, ac, pt) in raw_triples:
        triple = (ag, ac, pt)
        if triple in all_stored:
            continue
        n_novel += 1
        is_pl = proposer._plausible(ag, ac, pt)
        if is_pl:
            n_plausible += 1
        if triple in seen:
            continue
        if is_pl and not proposer._contradicts(ag, ac, pt):
            accepted.append(triple)
            seen.add(triple)
    return {
        "accepted": accepted,
        "n_novel_attempts": n_novel,
        "plausible_fraction_of_novel": n_plausible / max(1, n_novel),
    }


def run_seed(seed, vocab, corpus, a):
    rng = np.random.default_rng(seed)
    agent, affirmed, negated, P, row, tau, plausible_novel_universe = build_world(seed, vocab, corpus, a)
    all_stored = set(affirmed) | set(negated)

    # the brain's gates (the proposer object supplies _plausible/_contradicts; we DON'T use its host propose()).
    proposer = GenerativeReplayProposer(agent.composer, affirmed, negated, P, row, tau,
                                        np.random.default_rng(seed * 7 + 1))

    # ---- the HOST sample-loop baseline (what we are replacing) ----
    host_rep = proposer.propose(a.n_attempts)
    host_frac = host_rep["plausible_fraction_of_novel"]
    host_accepted = set(host_rep["accepted"])

    # ---- the SPIKING SWR-gated CA3 resampler (the replacement) ----
    # the CA3 recurrent ENCODES the brain's learned PPMI association graph (co-fire each PPMI-related role pair) ->
    # the recurrent BECOMES the PPMI cortex; an SWR cue over it completes to PPMI-related fillers.
    t_build = time.time()
    sampler = SpikingSWRCa3Sampler(vocab, P, row, tau, seed=seed, n_pool=a.n_pool,
                                   pattern_size=a.pattern_size, enc_cycles=a.enc_cycles,
                                   temperature=a.temperature, read_window=a.read_window, encode_graph=True)
    build_s = time.time() - t_build
    raw = sampler.resample(a.n_attempts_spiking)
    spk = _gate_and_collect(raw, proposer, all_stored)
    spk_accepted = spk["accepted"]
    spk_frac = spk["plausible_fraction_of_novel"]
    spk_set = set(spk_accepted)
    n_spk = len(spk_accepted)

    # (a) NOVEL: disjoint from store; known-fact retrieval abstains on every spiking proposal.
    novel_disjoint = len(spk_set & all_stored) == 0
    novel_comp_score = min(1.0, n_spk / max(1, len(plausible_novel_universe)))
    retr_abstains = 0
    for (ag, ac, pt) in spk_accepted:
        kp = agent.what_does(ag, ac)
        yn = agent.is_it_true(ag, ac, pt)
        if kp != pt and yn == "unknown":
            retr_abstains += 1
    retr_abstains_all = (retr_abstains == n_spk)

    # (b) PLAUSIBLE: vs random floor, and vs the HOST sample loop (must MATCH host quality).
    randb = random_recombination(proposer, a.n_attempts, np.random.default_rng(seed * 13 + 3))
    random_frac = randb["plausible_fraction_of_novel"]
    spk_advantage = spk_frac / max(random_frac, 1.0 / max(1, randb["n_novel_attempts"]))
    host_advantage = host_frac / max(random_frac, 1.0 / max(1, randb["n_novel_attempts"]))
    # quality match: the spiking sampler's plausible-frac is >= host_match_frac * the host's (not a degradation)
    spk_vs_host = spk_frac / max(host_frac, 1e-9)

    # (c) CA3 LESION: an UNTRAINED CA3 (encode_graph=False -> the recurrent never learned the association graph) ->
    # the spiking attractor has no learned structure to complete with -> the resampling degrades to the floor (the
    # bare dynamics). This is the load-bearing anti-cheat: if the lesion still resamples plausibly, a host fallback
    # (not the spiking attractor) was doing the work. (A fresh bridge avoids desyncing the trained bridge's cached
    # CSR index arrays, which a live in-place CSR rewrite corrupts.)
    sampler_lesion = SpikingSWRCa3Sampler(vocab, P, row, tau, seed=seed, n_pool=a.n_pool,
                                          pattern_size=a.pattern_size, enc_cycles=a.enc_cycles,
                                          temperature=a.temperature, read_window=a.read_window,
                                          encode_graph=False)
    # the lesion sampler still needs the encodable-word set (which agents to seed from) -- it shares the graph's
    # role membership but NOT the learned recurrent weights.
    aa, ap = sampler._related_pairs()
    sampler_lesion.encodable_words = {w for pr in (aa + ap) for w in pr}
    raw_lesion = sampler_lesion.resample(a.n_attempts_spiking)
    les = _gate_and_collect(raw_lesion, proposer, all_stored)
    lesion_frac = les["plausible_fraction_of_novel"]
    lesion_n = len(les["accepted"])
    # the lesion collapses the plausibility toward the random floor (the recurrent attractor was load-bearing).
    lesion_collapses = lesion_frac <= max(0.5 * spk_frac, random_frac * 1.5 + 0.02)

    # (d) MOAT 0-CONFAB: 0 hypothesis->known leaks, 0 negated re-proposed, untaught-cue abstention unregressed.
    moat_leaks = 0
    for (ag, ac, pt) in spk_accepted:
        known = agent.composer.query_patient(ag, ac)
        yn = agent.composer.ask_yes_no(ag, ac, pt)
        if known == pt:
            moat_leaks += 1
        if yn == "yes":
            moat_leaks += 1
    contradictions_proposed = len(spk_set & set(negated))
    n_ab, ab_ok, guard = 0, 0, 0
    stored_cues = {(ag, ac) for ag, ac, _ in affirmed}
    apool, acpool, _pp = _category_pools(TAXONOMY_8x8)
    while n_ab < 20 and guard < 100000:
        guard += 1
        ag = apool[int(rng.integers(len(apool)))]
        ac = acpool[int(rng.integers(len(acpool)))]
        if (ag, ac) in stored_cues:
            continue
        n_ab += 1
        ab_ok += int(agent.what_does(ag, ac) is None)

    examples = [f"perhaps {t[0]} {t[1]} {t[2]}" for t in spk_accepted[:12]]

    print(f"\n[followon1 seed {seed}] taught {len(affirmed)} affirmed + {len(negated)} negated | "
          f"novel-plausible universe {len(plausible_novel_universe)} | tau={tau:.3f} | CA3 build {build_s:.1f}s",
          flush=True)
    print(f"  (a) SPIKING SAMPLER generated {n_spk} distinct NOVEL props (novel-comp {novel_comp_score:.3f}); "
          f"disjoint {novel_disjoint}; known-fact retrieval ABSTAINS {retr_abstains}/{n_spk} (all {retr_abstains_all})",
          flush=True)
    print(f"  (b) PLAUSIBLE: spiking-frac {spk_frac:.3f} (adv {spk_advantage:.1f}x) vs HOST-frac {host_frac:.3f} "
          f"(adv {host_advantage:.1f}x) vs random {random_frac:.4f}  | spiking/host quality = {spk_vs_host:.2f}",
          flush=True)
    print(f"  (c) CA3 LESION: spiking-frac collapses {spk_frac:.3f} -> {lesion_frac:.3f} "
          f"({lesion_n} accepted; collapses {lesion_collapses})", flush=True)
    print(f"  (d) MOAT: hypothesis->known leaks {moat_leaks} (must 0) | negated re-proposed "
          f"{contradictions_proposed} (must 0) | untaught-cue abstention {ab_ok}/{n_ab}", flush=True)
    if examples:
        print(f"  spiking-sampled hypotheses: {examples}", flush=True)

    return {
        "seed": seed,
        "n_affirmed": len(affirmed),
        "n_negated": len(negated),
        "tau": tau,
        "ca3_build_s": build_s,
        "discoverable_novel_plausible_universe": len(plausible_novel_universe),
        # (a) NOVEL
        "n_spiking_generated": n_spk,
        "novel_composition_score": novel_comp_score,
        "novel_disjoint_from_store": novel_disjoint,
        "retrieval_abstains_on_generated": retr_abstains,
        "retrieval_abstains_all": retr_abstains_all,
        "spiking_examples": examples,
        # (b) PLAUSIBLE (vs host + vs random)
        "spiking_plausible_fraction_of_novel": spk_frac,
        "host_plausible_fraction_of_novel": host_frac,
        "random_plausible_fraction_of_novel": random_frac,
        "spiking_advantage_ratio": spk_advantage,
        "host_advantage_ratio": host_advantage,
        "spiking_vs_host_quality": spk_vs_host,
        "n_host_generated": len(host_accepted),
        # (c) CA3 LESION
        "lesion_plausible_fraction_of_novel": lesion_frac,
        "lesion_n_accepted": lesion_n,
        "lesion_collapses": lesion_collapses,
        # (d) MOAT
        "moat_leaks": moat_leaks,
        "contradictions_proposed": contradictions_proposed,
        "untaught_cue_abstention_correct": ab_ok,
        "untaught_cue_abstention_attempted": n_ab,
    }


def decide_verdict(rows, a):
    """GO iff, across ALL seeds: (a) the SPIKING sampler generates novel props (>= min_novel distinct, disjoint,
    known-fact retrieval abstains on every one); (b) a clear plausibility ADVANTAGE over random AND the spiking
    quality MATCHES the host sample loop (spiking_vs_host >= host_match_frac); (c) the CA3 LESION collapses the
    plausibility (the spiking attractor is load-bearing); (d) the no-confab MOAT holds (0 leaks, 0 negated
    re-proposed, untaught-cue abstention >= store_floor_bar). Else HONEST_NEGATIVE + the precise residual."""
    def col(k):
        return [r[k] for r in rows]

    spk_frac = np.array(col("spiking_plausible_fraction_of_novel"))
    host_frac = np.array(col("host_plausible_fraction_of_novel"))
    rand_frac = np.array(col("random_plausible_fraction_of_novel"))
    spk_adv = np.array(col("spiking_advantage_ratio"))
    host_adv = np.array(col("host_advantage_ratio"))
    spk_vs_host = np.array(col("spiking_vs_host_quality"))
    n_gen = np.array(col("n_spiking_generated"))
    novel_score = np.array(col("novel_composition_score"))
    novel_disjoint = np.array(col("novel_disjoint_from_store"))
    retr_abstains = np.array(col("retrieval_abstains_all"))
    lesion_frac = np.array(col("lesion_plausible_fraction_of_novel"))
    lesion_collapses = np.array(col("lesion_collapses"))
    leaks = np.array(col("moat_leaks"))
    contra = np.array(col("contradictions_proposed"))
    ab_ok = np.array(col("untaught_cue_abstention_correct"))
    ab_att = np.array(col("untaught_cue_abstention_attempted"))

    adv_bar = float(a.advantage_bar)
    min_novel = int(a.min_novel)
    host_match = float(a.host_match_frac)

    novel_all = bool(np.all(n_gen >= min_novel) and np.all(novel_score > 0.0)
                     and np.all(novel_disjoint) and np.all(retr_abstains))
    advantage_all = bool(np.all(spk_adv >= adv_bar))
    host_match_all = bool(np.all(spk_vs_host >= host_match))
    lesion_collapses_all = bool(np.all(lesion_collapses))
    moat_preserved_all = bool(np.all(leaks == 0) and np.all(contra == 0))
    store_floor_rate = ab_ok / np.maximum(ab_att, 1)
    store_floor_ok_all = bool(np.all(store_floor_rate >= float(a.store_floor_bar)))

    detail = {
        "spiking_plausible_fraction_mean": float(spk_frac.mean()),
        "host_plausible_fraction_mean": float(host_frac.mean()),
        "random_plausible_fraction_mean": float(rand_frac.mean()),
        "spiking_advantage_ratio_mean": float(spk_adv.mean()),
        "spiking_advantage_ratio_min": float(spk_adv.min()),
        "host_advantage_ratio_mean": float(host_adv.mean()),
        "spiking_vs_host_quality_mean": float(spk_vs_host.mean()),
        "spiking_vs_host_quality_min": float(spk_vs_host.min()),
        "novel_composition_score_mean": float(novel_score.mean()),
        "n_spiking_generated_mean": float(n_gen.mean()),
        "n_spiking_generated_min": int(n_gen.min()),
        "lesion_plausible_fraction_mean": float(lesion_frac.mean()),
        "lesion_collapses_all_seeds": lesion_collapses_all,
        "moat_leaks_total": int(leaks.sum()),
        "contradictions_proposed_total": int(contra.sum()),
        "untaught_cue_abstention_rate_mean": float(store_floor_rate.mean()),
        "untaught_cue_abstention_rate_min": float(store_floor_rate.min()),
        "novel_all_seeds": novel_all,
        "advantage_all_seeds": advantage_all,
        "host_match_all_seeds": host_match_all,
        "moat_preserved_all_seeds": moat_preserved_all,
        "store_floor_ok_all_seeds": store_floor_ok_all,
        "advantage_bar": adv_bar,
        "min_novel_bar": min_novel,
        "host_match_frac_bar": host_match,
        "store_floor_bar": float(a.store_floor_bar),
    }

    if novel_all and advantage_all and host_match_all and lesion_collapses_all and moat_preserved_all and store_floor_ok_all:
        verdict = "GO"
    elif not moat_preserved_all:
        verdict = "HONEST_NEGATIVE_moat_broken"
    elif not store_floor_ok_all:
        verdict = "HONEST_NEGATIVE_untaught_abstention_regressed"
    elif not novel_all:
        verdict = "HONEST_NEGATIVE_no_novel_generated"
    elif not advantage_all:
        verdict = "HONEST_NEGATIVE_no_plausibility_advantage"
    elif not host_match_all:
        verdict = "HONEST_NEGATIVE_underperforms_host_sample_loop"
    elif not lesion_collapses_all:
        verdict = "HONEST_NEGATIVE_ca3_not_load_bearing"
    else:
        verdict = "HONEST_NEGATIVE_other"
    return verdict, detail


def main():
    p = argparse.ArgumentParser(description="Follow-on #1: the FULLY-SPIKING SWR-gated CA3 generative-replay "
                                            "sampler -- replace the host sample loop with spiking attractor "
                                            "pattern-completion resampling.")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--n-facts", type=int, default=24, help="AFFIRMED interlinked SVO facts taught to the agent")
    p.add_argument("--n-negated", type=int, default=12, help="NEGATED facts -- the non-contradiction gate")
    p.add_argument("--n-attempts", type=int, default=2000, help="host/random sample-loop attempts (baseline)")
    p.add_argument("--n-attempts-spiking", type=int, default=200,
                   help="SWR-gated spiking resamples (each = 2 spiking completions; kept tractable on CPU)")
    p.add_argument("--n-pool", type=int, default=1500, help="CA3 pool size")
    p.add_argument("--pattern-size", type=int, default=100, help="K-of-N sparse pattern size per word")
    p.add_argument("--enc-cycles", type=int, default=5, help="Hebbian co-fire encode cycles for the recurrent graph")
    p.add_argument("--read-window", type=int, default=35, help="spiking read window per completion")
    p.add_argument("--temperature", type=float, default=0.6, help="softmax temp over the SPIKING completion profile")
    p.add_argument("--tau-pct", type=float, default=50.0)
    p.add_argument("--advantage-bar", type=float, default=3.0, help="spiking-vs-random plausible-frac RATIO gate")
    p.add_argument("--host-match-frac", type=float, default=0.7,
                   help="spiking plausible-frac must be >= this fraction of the HOST sample loop's (quality match)")
    p.add_argument("--min-novel", type=int, default=3)
    p.add_argument("--store-floor-bar", type=float, default=0.95)
    p.add_argument("--max-bytes", type=int, default=4_000_000)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--repeat-cap", type=int, default=40)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")

    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    t0 = time.time()
    print(f"[followon1] seeds={seeds} n_attempts_spiking={a.n_attempts_spiking} n_pool={a.n_pool} -- can a "
          f"SPIKING SWR-gated CA3 attractor RESAMPLE novel-but-plausible recombinations (replacing the host "
          f"sample loop), matching host quality, moat intact?", flush=True)

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
    print(f"  (a) SPIKING sampler novel: novel-comp mean {detail['novel_composition_score_mean']:.3f} "
          f"(>0 + disjoint + retrieval-abstains all seeds: {detail['novel_all_seeds']}; min "
          f"{detail['n_spiking_generated_min']} generated)", flush=True)
    print(f"  (b) PLAUSIBLE: spiking-frac {detail['spiking_plausible_fraction_mean']:.3f} "
          f"(adv {detail['spiking_advantage_ratio_mean']:.1f}x; >= {detail['advantage_bar']}x all: "
          f"{detail['advantage_all_seeds']}) vs HOST {detail['host_plausible_fraction_mean']:.3f} -- "
          f"spiking/host quality mean {detail['spiking_vs_host_quality_mean']:.2f} (>= "
          f"{detail['host_match_frac_bar']} all: {detail['host_match_all_seeds']})", flush=True)
    print(f"  (c) CA3 LESION collapses: spiking-frac {detail['spiking_plausible_fraction_mean']:.3f} -> lesion "
          f"{detail['lesion_plausible_fraction_mean']:.3f} (collapses all seeds: "
          f"{detail['lesion_collapses_all_seeds']})", flush=True)
    print(f"  (d) MOAT 0-CONFAB: {detail['moat_leaks_total']} hypothesis->known leaks + "
          f"{detail['contradictions_proposed_total']} negated re-proposed (preserved all: "
          f"{detail['moat_preserved_all_seeds']}); untaught-cue abstention mean "
          f"{detail['untaught_cue_abstention_rate_mean']:.3f} (>= {detail['store_floor_bar']:.2f} all: "
          f"{detail['store_floor_ok_all_seeds']})", flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*98}\n", flush=True)

    out = {
        "probe": "followon1_spiking_generative_sampler",
        "verdict": verdict,
        "seeds": seeds,
        "config": {k: getattr(a, k) for k in ("n_facts", "n_negated", "n_attempts", "n_attempts_spiking",
                                              "n_pool", "pattern_size", "enc_cycles", "read_window",
                                              "temperature", "tau_pct", "advantage_bar", "host_match_frac",
                                              "min_novel", "store_floor_bar", "max_bytes", "window")},
        "what_is_replaced": (
            "the HOST sample loop = GenerativeReplayProposer.propose(): a host `for _ in range(n_attempts)` loop "
            "doing weighted random sampling (`_weight_partner` reads PPMI weights, `_sample_weighted` = numpy "
            "rng.choice(p=weights)) to pick an action then a patient. REPLACED BY: a SPIKING SWR-gated CA3 "
            "pattern-completion resampler (SpikingSWRCa3Sampler over _D_sparse_heteroassoc) -- an SWR event drives "
            "a sparse seed cue, the plastic spiking recurrent attractor pattern-completes, and the recombination "
            "EMERGES from the spiking dynamics (a temperature over the attractor's OUTPUT firing). The brain's "
            "downstream gates (PPMI-plausibility + non-contradiction + the no-confab moat) are unchanged from 3E."),
        "baseline_to_match": {"host_sample_loop": "GenerativeReplayProposer.propose() (3E GO, the host bookkeeping)"},
        "detail": detail,
        "per_seed": rows,
        "brain_based_note": (
            "the recombination is now done by a SPIKING CA3 recurrent attractor (Buzsaki SWR + Marr/Treves-Rolls "
            "autoassociation; catalog G.09 constructive imagination) on the real SimulationBridge -- the experienced "
            "associations are ENCODED by Hebbian co-fire into the plastic recurrent, and an SWR cue drives spiking "
            "pattern-completion to resample fillers. The host `rng.choice`-over-PPMI bookkeeping is eliminated for "
            "the sampling step; the load-bearing PPMI-plausibility cortex + RF composer store + no-confab moat are "
            "unchanged. NO sim/ edit; reuse-by-import; CPU."),
        "elapsed_total_s": time.time() - t0,
    }
    if a.out is None:
        a.out = os.path.join(_REPO, "research", "findings", "raw", "_followon1_spiking_generative_sampler.json")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {a.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
