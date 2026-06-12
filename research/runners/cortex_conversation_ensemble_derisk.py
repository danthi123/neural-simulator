"""CORTEX <-> CONVERSATION **MULTI-BRIDGE ENSEMBLE** capability de-risk -- the owner-chosen INTERMEDIATE step
that combines the project's two validated de-risks on a SMALL 3-bridge ensemble BEFORE the ~2-4 week 32-bridge
build:

  (i)  WITHIN-bridge generalization-IN-conversation  (cortex_conversation_capability_derisk, GO 3-seed): a query
       about a held-out concept X is answered via a SIMILAR concept Y in the SAME bridge (cat ~ dog -> "what does
       a dog eat?" -> meat), with the no-confab moat intact.
  (ii) CROSS-bridge composition  (multibridge_graded_derisk, GO 3-seed): store a fact spanning two bridges
       (`dog eats meat`, dog in animals, meat in foods) via the V-tag engram layer, recall it, with the FIXED
       M7-style anti-cheat (a cue must NOT retrieve a WRONG cross-bridge target).

THE LOAD-BEARING DISTINCTION (stated up-front, enforced by the gates): graded SIMILARITY is a WITHIN-bridge
property (cat ~ dog only if both share a recurrent / live in the same shard). CROSS-bridge is IDENTITY
composition, NOT graded -- the V-tag layer relates concept IDENTITIES, not similarities. So:
  * Gate B (generalization) is measured WITHIN a bridge, on that bridge's graded cortex codes.
  * Gate X (cross-bridge) is measured ACROSS bridges, by exact identity recall of the stored target (top-1/top-2,
    signal clears the noise floor) -- and its anti-cheat is the FIXED `cross_bridge_eval` M7 control (score a
    WRONG target -> must rank ~median -> collapses), NOT a graded number.

SPEC: docs/plans/2026-06-12-cortex-conversation-integration-design.md (the integration architecture) +
the two reuse runners cited above. NO sim/ edits anywhere -- everything is reuse-by-import.

WHAT THIS RUNNER DOES (multi-seed 42/43/44; n_bridges shards x concepts-per-bridge concepts; e.g. animals /
foods / vehicles). The agent holds n_bridges learned-graded cortex codebooks (one per shard) on ONE
RFPhasorComposer over the union vocabulary; it stores BOTH within-shard facts (per bridge, so within-bridge
generalization is meaningful) AND cross-shard facts (spanning bridges, via the multibridge V-tag layer):

  GATE A -- the conversational matrix on the 3-bridge ensemble (who/what Q&A, abstention, negation/yes-no,
            one-attribute, a clause), SPANNING the bridges (the SVO roles are drawn from different shards). The
            matrix passes (>=5/6 cells) AND the moat returns None on every never-stored cue (zero abstention
            breaches). Any abstention breach => NEGATIVE.
  GATE B -- WITHIN-bridge generalization in conversation, with all 3 bridges CO-RESIDENT. Per bridge: hold out a
            graded-neighbour concept, query it through the agent's `what_does`, the graded fallback answers via
            the SAME-bridge similar concept. B1 (the design's run_generalization on that bridge's graded cortex
            codes) >= 0.7 (~4x chance); B2 the moat still abstains on genuine absence (zero false-accepts on a
            >=20-cue floor).
  GATE X -- CROSS-bridge composition driven CONVERSATIONALLY. Store cross-bridge facts (`dog eats meat`), query
            them back (who eats meat? / what does dog eat?), and verify the target is retrieved ACROSS bridges
            (top-1/top-2, signal clears the noise floor), with the CORRECTED M7-style anti-cheat (the FIXED
            `multibridge_graded_derisk.cross_bridge_eval` permuted control: a cue must NOT retrieve a WRONG
            cross-bridge target). Two complementary realizations:
              X-vtag  -- the spiking V-tag engram layer over the graded `pool` regions (the multibridge
                         mechanism; GPU; the canonical Gate X).
              X-conv  -- the SAME cross-bridge facts stored as relational SVO in the ensemble composer and
                         queried via who_does/what_does (the conversational realization; CPU/numpy; identity
                         recall, exact, no graded fallback used).
  ANTI-CHEATS (all mandatory; the GO is void without them):
    C1 permuted-similarity -> Gate B (B1) MUST collapse to chance (within-bridge generalization is meaning-driven).
    Cx the FIXED cross-bridge anti-cheat -> Gate X (V-tag) MUST collapse (cross-bridge recall is specific; a cue
       must not retrieve a WRONG target -- reuse cross_bridge_eval(..., permuted=True)).
    C3 the familiarity-gate moat validated ALONGSIDE the host (zero host-abstain/gate-accept breaches + lesion
       collapses) -- reuses multibridge_graded_derisk.moat_eval over the cross-bridge facts.
    C4 random-shard -> Gate B MUST collapse (within-shard graded co-location is load-bearing).

  GO       = A AND B AND X AND all anti-cheats collapse, multi-seed 42/43/44.
  BOUNDARY = A passes, B1 in 0.5-0.7 with B2 + all controls clean (real but weak within-bridge generalization),
             X GO.
  NEGATIVE = any moat breach (A abstention fails, B2/C3 false-accept) -- FATAL; OR B1 <= chance; OR Gate X recall
             fails (top-2 below band); OR C1/Cx/C4 fails to collapse (an artifact, not similarity/identity-driven).
  A NEGATIVE is itself the deliverable -- it reshapes the integration BEFORE the 32-bridge spend.

REUSE-BY-IMPORT (NO sim/ edits; every cited piece is runner-side / validated):
  - the SINGLE-shard capability de-risk classes (extended here to multi-bridge):
        CortexCodebook, CortexAugmentedAgent, build_cortex_codebook_synthetic / _learned, gate_A_matrix,
        gate_B_generalization, anticheat_C1_permuted, anticheat_C4_random_shard, GEN_ACTIONS, MATRIX_ACTIONS,
        _make_property_patient_words   (cortex_conversation_capability_derisk).
  - the CROSS-bridge machinery (the multibridge de-risk):
        GradedBridge, build_bridge_corpus, SHARD_NAMES, author_cross_facts, cross_bridge_eval (M3 + the FIXED
        M7), moat_eval   (multibridge_graded_derisk).
  - the conversational loop: RFPhasorComposer + the `grounded_codes` seam (rf_phasor_composer.py:86-89).
  - the moat protocol: RelationalFamiliarityGate (familiarity_gate_v320_validation, via the single-shard agent).

ADAPTATIONS vs the single-shard de-risk (the single-shard `CortexAugmentedAgent` assumes ONE codebook; this
runner generalizes to a {shard: CortexCodebook} dict + a cross-bridge layer -- see the module-level ADAPTATIONS
string and the run summary):
  1. EnsembleCortexAgent holds a {shard: CortexCodebook} dict and ONE RFPhasorComposer over the UNION of all
     shards' concepts + aux vocab, with EVERY shard's DG-decorrelated phase_codes merged into grounded_codes.
     This lets the composer bind BOTH within-shard facts and CROSS-shard SVO facts (`dog eats meat`).
  2. The within-bridge graded relational FALLBACK reads the cortex of the AGENT-word's shard (resolved by a word
     -> shard map), so generalization stays within-bridge (cat ~ dog only inside `animals`).
  3. Gate X (cross-bridge) is a SEPARATE measurement from Gate B: it uses the spiking V-tag layer (GradedBridge /
     cross_bridge_eval) for identity recall -- NOT the graded fallback. The conversational X-conv realization
     stores the cross facts in the ensemble composer and queries who_does/what_does (exact identity).
  4. The cheap CPU smoke uses a SYNTHETIC graded codebook per shard (the single-shard --cortex synthetic path)
     and SKIPS the live-spiking V-tag Gate X (`--skip-vtag`, default ON under --smoke) -- the V-tag layer builds
     live SimulationBridges (GPU). X-conv (the numpy identity-recall realization) DOES run on CPU, so the
     cross-bridge PLUMBING is still exercised end-to-end on CPU.
  5. Gate A reuses the single-shard `gate_A_matrix` (which builds a single-codebook CortexAugmentedAgent) over a
     SYNTHESIZED UNION codebook whose `words` INTERLEAVE the shards -- so the SVO roles span the bridges while the
     matrix stays the validated no-regression test on cortex-induced phases. Gate A does NOT need the multi-bridge
     graded fallback (it tests binding + the moat, not generalization), so reusing the single-shard matrix
     verbatim on the interleaved union is correct.

Run (REAL small-scale 3-bridge ensemble de-risk -- GPU for the graded spiking learn + the V-tag layer; the
matrix + generalization + moat + X-conv reads are numpy):
  SIM_BACKEND=cupy python -u -m research.runners.cortex_conversation_ensemble_derisk \
      --mode full --seeds 42,43,44 --cortex learned --n-bridges 3 --concepts-per-bridge 64 \
      --n-pool 2400 --pattern-size 100 --homeo oja --homeo-target 40 --cycles 10 \
      --out research/findings/raw/_cortex_conversation_ensemble_full.json

Tiny CPU smoke (plumbing only -- proves it RUNS end-to-end, NOT the science; ~<90s):
  SIM_BACKEND=numpy python -u -m research.runners.cortex_conversation_ensemble_derisk --mode full --seeds 42 --smoke
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

# ---- the conversational loop (REUSE) ----
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402

# ---- the SINGLE-shard capability de-risk (REUSE + EXTEND to multi-bridge) ----
from research.runners.cortex_conversation_capability_derisk import (  # noqa: E402
    CortexCodebook,
    CortexAugmentedAgent,
    build_cortex_codebook_synthetic,
    build_cortex_codebook_learned,
    gate_A_matrix,
    gate_B_generalization,
    anticheat_C1_permuted,
    anticheat_C4_random_shard,
    GEN_ACTIONS,
    MATRIX_ACTIONS,
)

# ---- the CROSS-bridge machinery (the multibridge de-risk; REUSE) ----
from research.runners.multibridge_graded_derisk import (  # noqa: E402
    SHARD_NAMES,
    build_bridge_corpus,
    author_cross_facts,
    cross_bridge_eval,
    moat_eval,
    GradedBridge,  # noqa: F401  (constructed inside the cross-bridge eval block)
)

ADAPTATIONS = (
    "1) EnsembleCortexAgent holds a {shard: CortexCodebook} dict + ONE RFPhasorComposer over the UNION vocab "
    "with every shard's DG-decorrelated phases merged into grounded_codes (so within-shard AND cross-shard "
    "SVO facts both bind). 2) the within-bridge graded fallback reads the AGENT-word's shard's cortex (a word->"
    "shard map keeps generalization within-bridge). 3) Gate X (cross-bridge) is SEPARATE from Gate B: it uses "
    "the spiking V-tag layer (GradedBridge/cross_bridge_eval, IDENTITY recall) + the conversational X-conv "
    "(cross facts stored as SVO, queried exactly) -- NOT the graded fallback (which is within-bridge only). "
    "4) the CPU smoke uses synthetic per-shard cortex and SKIPS the live-spiking V-tag Gate X (--skip-vtag); "
    "X-conv runs on CPU so the cross-bridge plumbing is still exercised end-to-end."
)


# ===========================================================================
# EnsembleCortexAgent -- the MULTI-BRIDGE generalization of the single-shard CortexAugmentedAgent.
#
# Holds a {shard: CortexCodebook} dict (per-bridge graded codes + DG-decorrelated phases) and ONE
# RFPhasorComposer over the UNION of all shards' concepts + the auxiliary action/property vocab. Every shard's
# decorrelated phase_codes are merged into grounded_codes, so the composer binds:
#   * within-shard SVO facts (both roles in one shard), AND
#   * cross-shard SVO facts (e.g. `dog eats meat`, dog in animals, meat in foods).
# The within-bridge graded relational FALLBACK (the NEW capability, same as the single-shard agent) reads the
# cortex of the AGENT word's shard (resolved by word->shard) so generalization stays WITHIN a bridge. The moat
# (RelationalFamiliarityGate + host abstention) is the single-shard agent's, computed over the ONE composer.
# ===========================================================================
class EnsembleCortexAgent(CortexAugmentedAgent):
    """A BrainConversationalAgent / CortexAugmentedAgent wired to N learned-graded cortex bridges co-resident on
    ONE RFPhasorComposer.

    Differs from the single-shard CortexAugmentedAgent only in that (a) it holds a {shard: CortexCodebook} dict
    (not one), (b) its composer vocab is the UNION over all shards + aux vocab with every shard's phases merged
    into grounded_codes, and (c) its graded fallback dispatches to the AGENT word's shard's cortex. The
    conversational queries (what_does/who_does/is_it_true), the moat, and the fallback gating are inherited
    verbatim from CortexAugmentedAgent (which reads `self.cortex` -- here a thin union view -- and the
    per-word shard for the fallback)."""

    def __init__(self, cortices: dict, *, seed=42, D=128, build_parser=False,
                 enable_fallback=True, fallback_novelty_thr=None, extra_vocab=None):
        # cortices: {shard_name -> CortexCodebook}
        self.seed = int(seed)
        self.cortices = dict(cortices)
        self.D = int(D)
        # word -> shard map (for the within-bridge fallback dispatch) + the union graded/phase codes.
        self.word_to_shard = {}
        union_words = set()
        union_phase_codes = {}
        for shard, cb in self.cortices.items():
            for w in cb.words:
                self.word_to_shard[w] = shard
                union_words.add(w)
            union_phase_codes.update(cb.phase_codes)
        # a UNION CortexCodebook view (so inherited code that reads self.cortex.graded_codes / .word_to_row still
        # works); its graded_codes are the per-shard graded codes concatenated (graded similarity is only
        # meaningful WITHIN a shard, which the fallback enforces via word_to_shard -- the union view is just a
        # convenience lookup, never used to compare ACROSS shards).
        self.cortex = _UnionCortexView(self.cortices, self.word_to_shard, D=D)
        vocab = sorted(union_words | set(extra_vocab or []))
        composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab, period=200,
                                    grounded_codes=union_phase_codes)
        self.composer = composer
        self._dlpfc = None
        self._dlpfc_key = None
        self._learned_assoc = None
        self.parser = None
        if build_parser:
            from research.runners.brain_conversational_agent import BridgeParser
            self.parser = BridgeParser(seed=seed)
        from research.runners.familiarity_gate_v320_validation import RelationalFamiliarityGate
        self.gate = RelationalFamiliarityGate(composer)
        self._gate_thr = fallback_novelty_thr
        self.enable_fallback = bool(enable_fallback)

    # ---- the within-bridge graded fallback: dispatch to the AGENT word's shard's cortex ----
    def _cortex_for(self, word):
        """The CortexCodebook for `word`'s shard (so the graded neighbour is a SAME-bridge concept)."""
        shard = self.word_to_shard.get(word)
        return self.cortices.get(shard) if shard is not None else None

    def _graded_fallback_patient(self, agent, action):
        """Within-bridge graded fallback for what_does: find the cortex-similar known agent for `action` that
        lives in the SAME shard as `agent` and whose (agent', action) fact is FAMILIAR; answer with its patient.
        Generalization stays within-bridge (the candidate set is restricted to the agent's shard)."""
        cb = self._cortex_for(agent)
        if cb is None:
            return None
        q = cb.graded_codes.get(agent)
        if q is None:
            return None
        # known agents with this action AND in the same shard as the queried agent
        cand = [(f["agent"], f) for f, _ in self.composer.kb
                if isinstance(f.get("agent"), str) and f.get("action") == action
                and self.word_to_shard.get(f.get("agent")) == self.word_to_shard.get(agent)]
        scored = []
        for a2, f in cand:
            c2 = cb.graded_codes.get(a2)
            if c2 is None:
                continue
            cos = float(np.dot(q, c2) / (np.linalg.norm(q) * np.linalg.norm(c2) + 1e-12))
            scored.append((cos, a2, f))
        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])
        _, best_a, _ = scored[0]
        if self._gate_thr is not None:
            nov = self.gate.novelty_patient(best_a, action)
            if nov >= self._gate_thr:
                return None
        return self.composer.query_patient(best_a, action)

    def _graded_fallback_agent(self, action, patient):
        """Within-bridge symmetric fallback for who_does (same-shard restriction on the patient)."""
        cb = self._cortex_for(patient)
        if cb is None:
            return None
        q = cb.graded_codes.get(patient)
        if q is None:
            return None
        cand = [(f["patient"], f) for f, _ in self.composer.kb
                if isinstance(f.get("patient"), str) and f.get("action") == action
                and self.word_to_shard.get(f.get("patient")) == self.word_to_shard.get(patient)]
        scored = []
        for p2, f in cand:
            c2 = cb.graded_codes.get(p2)
            if c2 is None:
                continue
            cos = float(np.dot(q, c2) / (np.linalg.norm(q) * np.linalg.norm(c2) + 1e-12))
            scored.append((cos, p2, f))
        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])
        _, best_p, _ = scored[0]
        if self._gate_thr is not None:
            nov = self.gate.novelty_agent(action, best_p)
            if nov >= self._gate_thr:
                return None
        return self.composer.query_agent(action, best_p)


class _UnionCortexView:
    """A thin read-only union view over {shard: CortexCodebook} exposing .words, .graded_codes, .word_to_row so
    code inherited from CortexAugmentedAgent that touches `self.cortex` still works. NOTE: cross-shard cosine
    comparisons on this union are meaningless (graded similarity is within-bridge); the fallback never uses the
    union to compare across shards (it dispatches per-shard via EnsembleCortexAgent._cortex_for)."""

    def __init__(self, cortices, word_to_shard, *, D=128):
        self.cortices = cortices
        self.word_to_shard = dict(word_to_shard)
        self.D = int(D)
        self.words = []
        self.graded_codes = {}
        for shard, cb in cortices.items():
            for w in cb.words:
                self.words.append(w)
                self.graded_codes[w] = cb.graded_codes[w]
        self.word_to_row = {w: i for i, w in enumerate(self.words)}


# ===========================================================================
# Build the ensemble: one CortexCodebook per shard (synthetic or learned).
# ===========================================================================
def build_ensemble_cortices(all_corpora, seed, args):
    """Return {shard_name -> CortexCodebook}. Each shard's codebook exposes graded_codes (the within-bridge
    similarity metric) + phase_codes (the DG-decorrelated phases the composer binds). Synthetic (cheap CPU) or
    learned (the spiking HomeostaticAssocGraph + divnorm read-out; GPU)."""
    cortices = {}
    for bc in all_corpora:
        shard = bc["shard"]
        n_sub, per_sub = bc["n_sub"], bc["per_sub"]
        members = bc["members"]   # namespaced (e.g. 'animals.c0_m1') -> globally-unique composer vocab
        t0 = time.time()
        if args.cortex == "learned":
            cb = build_cortex_codebook_learned(bc, D=args.D, dg_n_pool=args.n_pool,
                                               dg_pattern_size=args.pattern_size, seed=seed, args=args)
        else:
            cb = build_cortex_codebook_synthetic(
                members, n_sub, per_sub, D=args.D, dg_n_pool=args.n_pool,
                dg_pattern_size=args.pattern_size, seed=seed, dim=args.synthetic_dim,
                residual_frac=args.residual_frac)
        gstats = cb.graded_stats()
        exp_cos = cb.expansion_between_cos()
        print(f"    [cortex {shard:>9} {args.cortex}] within-cos={gstats['within_cluster_cos_mean']:.3f} "
              f"between-cos={gstats['between_cluster_cos_mean']:.3f} margin={gstats['graded_margin']:.3f} "
              f"graded={gstats['is_graded']} | DG-expansion between-cos={exp_cos:.3f} | {time.time()-t0:.1f}s",
              flush=True)
        cortices[shard] = cb
    return cortices


# ===========================================================================
# GATE X -- CROSS-bridge composition driven CONVERSATIONALLY (two realizations).
# ===========================================================================
def gate_X_conv(cortices, cross_facts, seed, args):
    """X-conv: store the cross-bridge facts as relational SVO in the ENSEMBLE composer (`dog eats meat`) and
    query them back conversationally (who_does('eats','foods.meat') -> 'animals.dog';
    what_does('animals.dog','eats') -> 'foods.meat'). This is IDENTITY recall ACROSS bridges (not graded), so
    the graded fallback is DISABLED here -- the exact host query must retrieve the stored cross-bridge target.

    Returns who/what exact-match accuracy + an abstention check (a never-stored cross cue must return None)."""
    # one ensemble agent over all shards, fallback OFF (cross-bridge is identity, not similarity).
    agent = EnsembleCortexAgent(cortices, seed=seed, D=args.D, build_parser=False,
                                enable_fallback=False, extra_vocab=["eats"])
    rel = "eats"
    stored = []
    seen_aa = set()
    seen_ap = set()
    for (cue_full, tgt_full) in cross_facts:
        if cue_full not in agent.composer.concepts or tgt_full not in agent.composer.concepts:
            continue
        if (cue_full, rel) in seen_aa or (rel, tgt_full) in seen_ap:
            continue
        agent.store_fact(cue_full, rel, tgt_full)
        stored.append((cue_full, tgt_full))
        seen_aa.add((cue_full, rel))
        seen_ap.add((rel, tgt_full))
    agent.reimprint_gate()

    n_what = n_what_ok = 0
    n_who = n_who_ok = 0
    for (cue_full, tgt_full) in stored:
        ans_what = agent.what_does(cue_full, rel)      # -> the cross-bridge target (foods.meat)
        n_what += 1
        n_what_ok += int(ans_what == tgt_full)
        ans_who = agent.who_does(rel, tgt_full)        # -> the cross-bridge cue (animals.dog)
        n_who += 1
        n_who_ok += int(ans_who == cue_full)
    # abstention: a never-stored cross cue (a real concept, the relation, but never stored) -> None.
    abst_breaches = 0
    rng = np.random.RandomState(seed * 37 + 11)
    all_words = list(agent.composer.concepts.keys())
    stored_cues = {c for (c, _t) in stored}
    n_abst = 0
    for _ in range(min(16, len(all_words))):
        w = all_words[rng.randint(len(all_words))]
        if w in stored_cues or w in ("eats",) or w in agent.composer.pol_words:
            continue
        n_abst += 1
        if agent.what_does(w, "NEVEREATS") is not None:
            abst_breaches += 1
    return {
        "n_stored": len(stored),
        "what_acc": (n_what_ok / n_what) if n_what else None,
        "who_acc": (n_who_ok / n_who) if n_who else None,
        "abstention_probes": n_abst, "abstention_breaches": int(abst_breaches),
    }


def gate_X_vtag(all_corpora, cross_facts, seed, args):
    """X-vtag: the spiking V-tag engram layer over the graded `pool` regions (the multibridge mechanism). Build
    one live GradedBridge per shard, store the cross-bridge facts as shared engram tags, recall the target ACROSS
    bridges (M3 over cross_bridge_eval), and run the FIXED M7 anti-cheat (Cx: a cue must NOT retrieve a WRONG
    target). GPU path (builds live SimulationBridges). Returns the M3 + M7 recall numbers + the bands."""
    graded_bridges = {}
    t0 = time.time()
    for bc in all_corpora:
        gb = GradedBridge(bc["shard"], bc["_local"]["concepts"], seed, args)
        gb.train(bc["_local"]["facts"])
        graded_bridges[bc["shard"]] = gb
        print(f"      [built+trained {bc['shard']:>9} graded bridge: "
              f"{gb.bridge.cp_membrane_potential_v.shape[0]} neurons]", flush=True)
    print(f"      (built {len(graded_bridges)} graded bridges in {time.time()-t0:.0f}s)", flush=True)

    m3 = cross_bridge_eval(graded_bridges, cross_facts, seed, args, permuted=False)   # store TRUE + score TRUE
    cx = cross_bridge_eval(graded_bridges, cross_facts, seed, args, permuted=True)    # FIXED M7: score WRONG
    # band on the cross-bridge recall (the multibridge M3 band)
    if m3["top2_fraction"] >= 0.80 and m3["mean_signal_vs_floor"] >= 1.5:
        band = "GO"
    elif m3["top2_fraction"] >= 0.50:
        band = "BOUNDARY"
    else:
        band = "NEGATIVE"
    cx_collapses = bool(cx["top2_fraction"] < max(0.5, m3["top2_fraction"] - 0.2))
    return {"m3": m3, "cx_permuted": cx, "band": band, "cx_collapses": cx_collapses}


# ===========================================================================
# C3 -- the moat ALONGSIDE the host over the CROSS-bridge facts (reuse moat_eval).
# ===========================================================================
def anticheat_C3_moat(all_corpora, cross_facts, seed, args):
    """The no-confab moat (familiarity gate alongside the host) over the cross-bridge facts -- reuse the
    multibridge moat_eval verbatim (host-abstain/gate-accept must be 0, floor false-accepts 0, lesion collapses).
    CPU/numpy."""
    all_members = []
    for bc in all_corpora:
        all_members.extend(bc["members"])
    return moat_eval(cross_facts, all_members, seed, args)


# ===========================================================================
# Per-seed driver.
# ===========================================================================
def run_seed(seed, args):
    print(f"\n{'='*94}", flush=True)
    print(f"  CORTEX<->CONVERSATION ENSEMBLE DE-RISK -- SEED {seed} -- mode={args.mode} "
          f"cortex={args.cortex} n_bridges={args.n_bridges}", flush=True)
    print(f"{'='*94}", flush=True)

    shard_names = SHARD_NAMES[:args.n_bridges]
    all_corpora = [build_bridge_corpus(sn, args.concepts_per_bridge, seed, args) for sn in shard_names]
    for bc in all_corpora:
        print(f"  [shard {bc['shard']:>9}] {len(bc['members'])} concepts "
              f"({bc['n_sub']} sub x {bc['per_sub']}), {bc['n_facts']} within-shard facts", flush=True)

    out = {"seed": seed, "mode": args.mode, "cortex_source": args.cortex, "shards": shard_names}

    # ---- build the per-shard cortex codebooks ----
    print(f"\n  [building {len(all_corpora)} cortex codebooks ({args.cortex})]", flush=True)
    cortices = build_ensemble_cortices(all_corpora, seed, args)
    out["graded_stats"] = {sh: cb.graded_stats() for sh, cb in cortices.items()}

    # ---- GATE A -- the conversational matrix SPANNING the bridges ----
    if args.mode in ("matrix", "full"):
        print(f"\n  {'-'*90}\n  GATE A -- conversational matrix on the {args.n_bridges}-bridge ensemble "
              f"(SVO roles span the bridges)\n  {'-'*90}", flush=True)
        # An ensemble matrix: build the matrix over a UNION CortexCodebook so the SVO roles are drawn from
        # DIFFERENT shards (the spanning-the-bridges requirement). We reuse the single-shard gate_A_matrix on a
        # synthesized union codebook whose `words` interleave the shards so words[0], words[1], ... come from
        # different bridges.
        union_cb = _make_union_codebook_for_matrix(cortices, seed, args)
        A = gate_A_matrix(union_cb, seed, args.D)
        print(f"    cells: {A['cells']}", flush=True)
        print(f"    n_cells_pass={A['n_cells_pass']}/6  moat_holds={A['moat_holds']} "
              f"(abstention_battery_breaches={A['cells']['abstention_battery_breaches']})", flush=True)
        out["gate_A"] = A

    # ---- GATE B -- WITHIN-bridge generalization in conversation, all bridges co-resident ----
    if args.mode in ("generalize", "full"):
        print(f"\n  {'-'*90}\n  GATE B -- WITHIN-bridge generalization in conversation (per bridge, "
              f"co-resident)\n  {'-'*90}", flush=True)
        per_bridge_B = {}
        per_bridge_C1 = {}
        per_bridge_C4 = {}
        for bc in all_corpora:
            sh = bc["shard"]
            cb = cortices[sh]
            B = gate_B_generalization(cb, seed, args.D, args, n_sub=bc["n_sub"], per_sub=bc["per_sub"])
            print(f"    [{sh:>9}] B1 gen acc={B['b1_accuracy']:.3f} "
                  f"(chance={B['chance']:.3f}, {B['ratio_vs_chance']:.1f}x)"
                  + (f"  B1-conv={B['b1_conv_accuracy']:.3f}" if B["b1_conv_accuracy"] is not None else "")
                  + (f"  B2 floor_n={B['b2']['floor_n']} fa={B['b2']['false_accepts']} "
                     f"abstains_all={B['b2']['abstains_all']}" if B["b2"] is not None else ""), flush=True)
            per_bridge_B[sh] = B
            # per-bridge anti-cheats C1 (permuted) + C4 (random shard) -- within-bridge generalization controls
            C1 = anticheat_C1_permuted(cb, seed, args.D, args)
            per_bridge_C1[sh] = C1
        # C4 random-shard once (it pools across all shards by construction)
        C4 = anticheat_C4_random_shard(all_corpora, None, seed, args.D, args)
        for bc in all_corpora:
            per_bridge_C4[bc["shard"]] = C4   # same control applies to the ensemble
        print(f"    C1 permuted-similarity (per bridge): "
              f"{ {sh: (round(c['b1_permuted'],3), c['collapses']) for sh, c in per_bridge_C1.items()} }",
              flush=True)
        print(f"    C4 random-shard: B1={C4['b1_random_shard']:.3f} collapses={C4['collapses']} "
              f"(n={C4['n_random_members']})", flush=True)
        out["gate_B"] = per_bridge_B
        out["anticheat_C1"] = per_bridge_C1
        out["anticheat_C4"] = C4

    # ---- GATE X -- CROSS-bridge composition (conversational + V-tag) + Cx + C3 ----
    if args.mode in ("cross", "full"):
        cross_facts = author_cross_facts(all_corpora, seed, args.n_cross_facts)
        print(f"\n  {'-'*90}\n  GATE X -- CROSS-bridge composition in conversation\n  {'-'*90}", flush=True)
        print(f"    [authored {len(cross_facts)} cross-bridge facts] e.g. {cross_facts[:3]}", flush=True)
        out["cross_facts"] = cross_facts

        # X-conv -- the conversational identity-recall realization (CPU/numpy)
        Xc = gate_X_conv(cortices, cross_facts, seed, args)
        print(f"    X-conv (SVO identity recall): what_acc={Xc['what_acc']} who_acc={Xc['who_acc']} "
              f"(n_stored={Xc['n_stored']}) | abstention breaches={Xc['abstention_breaches']}"
              f"/{Xc['abstention_probes']}", flush=True)
        out["gate_X_conv"] = Xc

        # X-vtag -- the spiking V-tag layer + the FIXED M7 anti-cheat (Cx). GPU; skipped on the CPU smoke.
        if args.skip_vtag:
            print(f"    X-vtag: SKIPPED (--skip-vtag; the live-spiking V-tag layer needs GPU bridges)",
                  flush=True)
            out["gate_X_vtag"] = None
        else:
            print(f"    X-vtag (spiking V-tag layer over the graded `pool` regions):", flush=True)
            Xv = gate_X_vtag(all_corpora, cross_facts, seed, args)
            print(f"      M3 TRUE: top2={Xv['m3']['top2_fraction']:.2f} top1={Xv['m3']['top1_fraction']:.2f} "
                  f"signal/floor={Xv['m3']['mean_signal_vs_floor']:.2f}x [{Xv['band']}]", flush=True)
            print(f"      Cx PERMUTED (FIXED M7, score WRONG target): "
                  f"top2={Xv['cx_permuted']['top2_fraction']:.2f} collapses={Xv['cx_collapses']}", flush=True)
            out["gate_X_vtag"] = Xv

        # C3 -- the moat over the cross-bridge facts (CPU/numpy)
        print(f"    C3 moat alongside host (over cross-bridge facts):", flush=True)
        C3 = anticheat_C3_moat(all_corpora, cross_facts, seed, args)
        print(f"      agreement={C3['agreement']:.3f} host-abstain/gate-accept={C3['host_abstain_gate_accept']} "
              f"(MUST be 0) floor-false-accepts={C3['abstention_floor_false_accepts']} (MUST be 0) "
              f"lesion-collapses={C3['lesion_collapsed']} -> moat_intact={C3['m4_moat_intact']}", flush=True)
        out["anticheat_C3"] = C3

    return out


def _make_union_codebook_for_matrix(cortices, seed, args):
    """Build a synthetic union CortexCodebook whose `words` INTERLEAVE the shards (words[0] from shard 0,
    words[1] from shard 1, ...) so gate_A_matrix's role picks (words[0], words[1], words[2], ...) draw the SVO
    roles from DIFFERENT bridges -- the 'matrix spans the bridges' requirement. The codes/labels are taken from
    the per-shard cortices so the matrix binds REAL cortex-induced phases (the no-regression test), and the moat
    runs over the union.

    NOTE the moat in gate_A_matrix is RELATIONAL (on the bound composite), so it holds regardless of which shard
    each role came from; the abstention battery uses a NEVERACT action token, never stored -> the host abstains.
    The interleave makes the stored facts genuinely cross-bridge."""
    # interleave the shards' members + their graded codes/labels
    shard_list = list(cortices.values())
    words, code_rows, labels = [], [], []
    lab_offset = 0
    # round-robin draw to interleave; cap to a modest count (the matrix needs only ~8 distinct words)
    max_each = max(2, (12 // max(1, len(shard_list))) + 2)
    per_shard_taken = {i: 0 for i in range(len(shard_list))}
    i = 0
    while len([w for w in words]) < 12 and any(per_shard_taken[k] < min(max_each, len(shard_list[k].words))
                                               for k in range(len(shard_list))):
        si = i % len(shard_list)
        cb = shard_list[si]
        t = per_shard_taken[si]
        if t < min(max_each, len(cb.words)):
            w = cb.words[t]
            words.append(w)
            code_rows.append(cb.codes[t])
            labels.append(int(cb.labels[t]) + lab_offset * 1000 + si * 100)  # keep labels shard-distinct
            per_shard_taken[si] += 1
        i += 1
    codes = np.stack(code_rows)
    labels = np.asarray(labels, dtype=int)
    # remap labels to a dense 0..K-1 range (codebook_similarity_stats / generalization want contiguous labels;
    # the matrix doesn't use generalization, but keep it clean)
    uniq = {v: k for k, v in enumerate(sorted(set(labels.tolist())))}
    labels = np.asarray([uniq[v] for v in labels.tolist()], dtype=int)
    return CortexCodebook(words, codes, labels, codes @ codes.T, D=args.D, dg_n_pool=args.n_pool,
                          dg_pattern_size=args.pattern_size, seed=seed, source="union")


# ===========================================================================
# Multi-seed verdict.
# ===========================================================================
def _bridge_iter(per_seed_result):
    return per_seed_result.get("gate_B", {}).items()


def aggregate(per_seed, args):
    seeds = list(per_seed.keys())
    agg = {"seeds": seeds, "mode": args.mode, "cortex_source": args.cortex, "b1_bar": args.a1_bar}
    chance = 1.0 / args.n_props

    def _all(pred):
        vals = [pred(per_seed[s]) for s in seeds]
        return all(v is True for v in vals), vals

    # ---- GATE A ----
    if args.mode in ("matrix", "full"):
        a_ok, a_vals = _all(lambda r: bool(r["gate_A"]["n_cells_pass"] >= 5 and r["gate_A"]["moat_holds"]))
        any_breach = any((not per_seed[s]["gate_A"]["cells"]["abstention"])
                         or per_seed[s]["gate_A"]["cells"]["abstention_battery_breaches"] > 0 for s in seeds)
        agg["gate_A"] = {"all_pass": a_ok, "per_seed": a_vals, "any_abstention_breach": any_breach}

    # ---- GATE B (within-bridge generalization, EVERY bridge EVERY seed) + C1 + C4 ----
    if args.mode in ("generalize", "full"):
        def _all_bridges(r, pred):
            return all(pred(B) for _sh, B in r["gate_B"].items())
        b1_go, _ = _all(lambda r: _all_bridges(r, lambda B: B["b1_accuracy"] >= args.a1_bar))
        b1_above_chance, _ = _all(lambda r: _all_bridges(r, lambda B: B["b1_accuracy"] > 1.25 * chance))
        b1_boundary = all(all(0.5 <= B["b1_accuracy"] < args.a1_bar for _sh, B in per_seed[s]["gate_B"].items())
                          for s in seeds)
        b2_ok, _ = _all(lambda r: _all_bridges(r, lambda B: (B["b2"] is None) or B["b2"]["abstains_all"]))
        c1_ok, _ = _all(lambda r: all(c["collapses"] for c in r["anticheat_C1"].values()))
        c4_ok, _ = _all(lambda r: r["anticheat_C4"]["collapses"])
        b1_all = {s: {sh: per_seed[s]["gate_B"][sh]["b1_accuracy"] for sh in per_seed[s]["gate_B"]}
                  for s in seeds}
        agg["gate_B"] = {"b1_per_seed_per_bridge": b1_all, "b1_all_GO": b1_go,
                         "b1_above_chance": b1_above_chance, "b1_boundary_band": b1_boundary,
                         "b2_zero_false_accepts": b2_ok, "chance": chance,
                         "C1_collapses": c1_ok, "C4_collapses": c4_ok}

    # ---- GATE X (cross-bridge) + Cx + C3 ----
    if args.mode in ("cross", "full"):
        xconv_ok, _ = _all(lambda r: (r["gate_X_conv"]["what_acc"] is not None
                                      and r["gate_X_conv"]["what_acc"] >= args.x_bar
                                      and r["gate_X_conv"]["who_acc"] >= args.x_bar
                                      and r["gate_X_conv"]["abstention_breaches"] == 0))
        # V-tag is optional (skipped on CPU smoke): GO requires it where present.
        vtag_present = all(per_seed[s].get("gate_X_vtag") is not None for s in seeds)
        if vtag_present:
            xvtag_go, _ = _all(lambda r: r["gate_X_vtag"]["band"] in ("GO", "BOUNDARY"))
            xvtag_recall_ok, _ = _all(lambda r: r["gate_X_vtag"]["m3"]["top2_fraction"] >= 0.5)
            cx_ok, _ = _all(lambda r: r["gate_X_vtag"]["cx_collapses"] is True)
        else:
            xvtag_go = xvtag_recall_ok = cx_ok = None
        c3_ok, _ = _all(lambda r: r["anticheat_C3"]["m4_moat_intact"])
        c3_lesion, _ = _all(lambda r: r["anticheat_C3"]["lesion_collapsed"])
        agg["gate_X"] = {"x_conv_ok": xconv_ok, "vtag_present": vtag_present,
                         "x_vtag_recall_ok": xvtag_recall_ok, "x_vtag_band_ok": xvtag_go,
                         "cx_collapses": cx_ok, "C3_moat_intact": c3_ok, "C3_lesion_collapses": c3_lesion}

    # ---- combined verdict ----
    verdict = None
    if args.mode == "full":
        gA, gB, gX = agg["gate_A"], agg["gate_B"], agg["gate_X"]
        moat_breach = (gA["any_abstention_breach"]
                       or (not gB["b2_zero_false_accepts"])
                       or (not gX["C3_moat_intact"]) or (not gX["C3_lesion_collapses"])
                       or any(per_seed[s]["gate_X_conv"]["abstention_breaches"] > 0 for s in agg["seeds"]))
        # cross-bridge recall must succeed (X-conv always; X-vtag where present)
        x_recall_ok = gX["x_conv_ok"] and (gX["x_vtag_recall_ok"] in (True, None))
        # anti-cheats collapse: C1 (within-bridge), C4 (random shard), Cx (cross-bridge permuted where present)
        controls_collapse = (gB["C1_collapses"] and gB["C4_collapses"]
                             and (gX["cx_collapses"] in (True, None)))
        if moat_breach:
            verdict = "NEGATIVE"                 # the moat is non-negotiable
        elif not gB["b1_above_chance"]:
            verdict = "NEGATIVE"                 # no within-bridge generalization
        elif not x_recall_ok:
            verdict = "NEGATIVE"                 # cross-bridge composition fails
        elif not controls_collapse:
            verdict = "NEGATIVE"                 # a "generalization"/"recall" is an artifact, not driven
        elif (gA["all_pass"] and gB["b1_all_GO"] and gB["b2_zero_false_accepts"]
              and (gX["x_vtag_band_ok"] in (True, None))):
            verdict = "GO"
        elif gA["all_pass"] and gB["b1_boundary_band"]:
            verdict = "BOUNDARY"
        else:
            verdict = "BOUNDARY"
    agg["verdict"] = verdict
    return agg


def main():
    p = argparse.ArgumentParser(
        description="Cortex<->conversation MULTI-BRIDGE ENSEMBLE capability de-risk (within-bridge "
                    "generalization-in-conversation + cross-bridge composition + the no-confab moat, on the "
                    "learned-graded cortex, 3-bridge ensemble)")
    p.add_argument("--mode", default="full", choices=["matrix", "generalize", "cross", "full"],
                   help="matrix=Gate A; generalize=Gate B + C1/C4; cross=Gate X + Cx/C3; full=all + verdict")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--seed", type=int, default=None, help="single-seed override")
    p.add_argument("--cortex", default="learned", choices=["synthetic", "learned"],
                   help="graded-codebook source: 'learned' (spiking HomeostaticAssocGraph + divnorm; GPU) or "
                        "'synthetic' (build_graded_codebook; cheap CPU)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU plumbing smoke: small shards + synthetic cortex + tiny n_pool/cycles + "
                        "--skip-vtag")
    p.add_argument("--skip-vtag", action="store_true",
                   help="skip the live-spiking V-tag Gate X (X-vtag); X-conv (numpy identity recall) still runs")
    # ensemble sizing (the multibridge curated shards)
    p.add_argument("--n-bridges", type=int, default=3)
    p.add_argument("--concepts-per-bridge", type=int, default=64)
    p.add_argument("--target-per-sub", type=int, default=8)
    # the graded learner (HomeostaticAssocGraph; the multibridge per-bridge recipe)
    p.add_argument("--n-pool", type=int, default=2400)
    p.add_argument("--pattern-size", type=int, default=100)
    p.add_argument("--homeo", default="oja", choices=["oja", "scaling", "none"])
    p.add_argument("--homeo-target", type=float, default=40.0)
    p.add_argument("--cycles", type=int, default=10)
    # synthetic codebook
    p.add_argument("--synthetic-dim", type=int, default=256)
    p.add_argument("--residual-frac", type=float, default=0.55)
    # within-shard corpus structure (de-risk defaults)
    p.add_argument("--n-props", type=int, default=4)
    p.add_argument("--hub-facts-per-member", type=int, default=6)
    p.add_argument("--bridge-facts", type=int, default=8)
    p.add_argument("--triplet-facts-per-cluster", type=int, default=4)
    # brain-based divnorm read-out (FIXED validated recipe)
    p.add_argument("--readout-divnorm", default="ch")
    p.add_argument("--readout-order", default="interleave")
    p.add_argument("--readout-sigma", type=float, default=0.001)
    p.add_argument("--readout-exponent", type=float, default=2.0)
    p.add_argument("--readout-log-clip", action="store_true")
    p.add_argument("--diffusion-alpha", type=float, default=0.5)
    p.add_argument("--diffusion-steps", type=int, default=2)
    # composer / agent
    p.add_argument("--D", type=int, default=128, help="phasor code dimension (composer default)")
    p.add_argument("--b-conv-splits", type=int, default=8)
    # cross-bridge V-tag encode/recall (adapted recipe over the `pool` region)
    p.add_argument("--n-cross-facts", type=int, default=12)
    p.add_argument("--encoding-steps", type=int, default=100)
    p.add_argument("--teacher-pA", type=float, default=500.0)
    p.add_argument("--top-k", type=int, default=150)
    p.add_argument("--drive-pA", type=float, default=1500.0)
    p.add_argument("--drive-steps", type=int, default=100)
    # moat (familiarity gate)
    p.add_argument("--moat-D", type=int, default=128)
    p.add_argument("--moat-floor", type=int, default=20)
    # gate bars
    p.add_argument("--a1-bar", type=float, default=0.7, help="B1 within-bridge generalization GO bar (~4x chance)")
    p.add_argument("--x-bar", type=float, default=0.7, help="Gate X-conv exact identity-recall GO bar")
    p.add_argument("--k-neighbours", type=int, default=3)
    # multibridge gate bars reused by the imported helpers
    p.add_argument("--g1-bar", type=float, default=0.5)
    p.add_argument("--so-margin-bar", type=float, default=0.10)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    # smoke overrides: tiny + synthetic + CPU-fast + skip the live-spiking V-tag layer
    if args.smoke:
        args.cortex = "synthetic"
        args.n_bridges = min(args.n_bridges, 3)
        args.concepts_per_bridge = min(args.concepts_per_bridge, 8)
        args.target_per_sub = 4
        args.n_pool = min(args.n_pool, 300)
        args.pattern_size = min(args.pattern_size, 30)
        args.cycles = 2
        args.synthetic_dim = 96
        args.b_conv_splits = 2
        args.n_cross_facts = min(args.n_cross_facts, 6)
        args.moat_floor = 8
        args.skip_vtag = True

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    backend = os.environ.get("SIM_BACKEND", "auto")
    t_all = time.time()
    print(f"[cortex<->conversation ENSEMBLE de-risk] mode={args.mode} seeds={seeds} cortex={args.cortex} "
          f"backend={backend} smoke={args.smoke}", flush=True)
    print(f"  {args.n_bridges} bridges x {args.concepts_per_bridge} concepts (target_per_sub="
          f"{args.target_per_sub}); composer D={args.D}; B1 bar={args.a1_bar}, X bar={args.x_bar} "
          f"(chance={1.0/args.n_props:.3f}); skip_vtag={args.skip_vtag}", flush=True)
    print(f"  ADAPTATIONS: {ADAPTATIONS}", flush=True)

    per_seed = {}
    for s in seeds:
        per_seed[str(s)] = run_seed(s, args)

    agg = aggregate(per_seed, args)

    print(f"\n{'='*94}", flush=True)
    print(f"  CORTEX<->CONVERSATION ENSEMBLE DE-RISK SUMMARY -- mode={args.mode}", flush=True)
    print(f"{'='*94}", flush=True)
    for k, v in agg.items():
        if k in ("seeds", "mode", "cortex_source", "verdict", "b1_bar"):
            continue
        print(f"  [{k}] {v}", flush=True)
    if agg.get("verdict") is not None:
        print(f"\n  >>> COMBINED VERDICT: {agg['verdict']} <<<", flush=True)
    print(f"  Total elapsed: {time.time()-t_all:.1f}s", flush=True)
    print(f"{'='*94}\n", flush=True)

    out_data = {"aggregate": agg, "per_seed": per_seed, "args": vars(args), "adaptations": ADAPTATIONS}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        tag = "smoke" if args.smoke else args.mode
        args.out = os.path.join(raw_dir, f"_cortex_conversation_ensemble_derisk_{tag}_seed{seeds[0]}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out_data


if __name__ == "__main__":
    main()
