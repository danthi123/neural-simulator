"""PHASE-1 COMPOSER-ARCHITECTURE A-vs-B DE-RISK -- the cheap-first run that DECIDES which composer architecture
carries the "step 3 true cortex" production build to 32 bridges / 2,048 concepts. Compares, at 8 bridges x 64 =
512 concepts, two composer architectures so the controller can pick the one that passes the full conversational
matrix at a cost that stays feasible at 2,048 concepts.

SPEC: docs/plans/2026-06-12-phase1-composer-architecture-design.md (the two architectures, the crux in SS3, the
cheap-first 8-bridge A-vs-B de-risk in SS5, the GO criteria SS5.4, the anti-cheats SS6). NO sim/ edits anywhere
-- this is a THIN sibling that imports the (just-validated, GO at 3 bridges) ensemble runner machinery verbatim
(cortex_conversation_ensemble_derisk) and adds ONLY the route-A per-bridge-composer agent + dispatch, route-A's
within-bridge Gate-A matrix, route-B's D-sweep, the per-route cost logging, and the A-vs-B comparison verdict.

THE TWO ROUTES (design SS2):
  ROUTE A -- per-bridge composers + cross-bridge V-tag identity layer (the design's RECOMMENDATION).
      Each of the N bridges owns its OWN small RFPhasorComposer (D ~ 256 over its 64 concepts + the shared aux
      vocab) for WITHIN-bridge generative binding (facts, clauses, attributes, who/what/negation). Within-bridge
      generalization reads that bridge's graded cortex codes directly (unchanged). CROSS-bridge facts use the
      validated V-tag IDENTITY recall (the ensemble's gate_X_vtag / multibridge cross_bridge_eval). Per-op cost is
      vocabulary-INDEPENDENT (D set by the 64-concept shard, cleanup argmax over 64, per-op RF bridges of a few
      hundred neurons) -> flat from 8 to 32 bridges.
  ROUTE B -- one scaled union composer (the EXISTING ensemble path).
      ONE RFPhasorComposer over all 512 concepts with a DIMENSION SWEEP (D in {512,1024,1536,2048}) to find the
      MINIMUM D that passes the full matrix (esp. the cross-bridge `clause` cell), so the controller can
      extrapolate the D needed at 2,048 and whether it stays feasible on the 24 GB GPU. The least-code-change
      path; carries the unvalidated-FHRR-SNR-at-2,048 risk + 10-20x larger per-op RF bridges.

THE CRUX (design SS3): does the validated conversational matrix NEED cross-bridge GENERATIVE binding, or is
within-bridge generative VSA + cross-bridge IDENTITY composition sufficient? The de-risk resolves it empirically:
route A's matrix forces the `clause` cell WITHIN a bridge (every clause word in ONE shard -- the single-shard
matrix GO is independent evidence it passes there); route B's matrix keeps the clause CROSS-bridge (interleaved
union). If route A's within-bridge matrix passes + Gate X (V-tag identity) GO + all anti-cheats, and route A
loses ONLY the cross-bridge generative clause (which no validated requirement demands), route A wins.

WHAT EACH ROUTE MEASURES (multi-seed 42/43/44, the SAME gate suite the ensemble already implements):
  GATE A   -- the conversational matrix (who/what, abstention, negation, one-attribute, clause). Route B: clause
              CROSS-bridge (interleaved). Route A: clause WITHIN-bridge (per-shard). Both >= 5/6 cells + zero
              abstention breach.
  GATE B   -- WITHIN-bridge graded generalization (B1 >= 0.7 ~ 4x chance; B2 moat zero false-accepts). IDENTICAL
              for both routes (generalization reads cortex codes; neither composer touches it).
  GATE X   -- cross-bridge composition. Route B: generative (X-conv, union composer). Route A: identity (V-tag,
              X-vtag) + a per-bridge-composer within-bridge plumbing check (X-pb). Both must retrieve the target
              above the noise floor with the Cx anti-cheat collapsing.
  ANTI-CHEATS -- C1 (permuted similarity -> B1 collapses), C4 (random shard -> B1 collapses), Cx (the FIXED M7:
              permuted cross-bridge -> Gate X collapses), C3 (moat alongside host, zero breaches + lesion
              collapses). All mandatory; the GO is void without them. PLUS a per-route abstention battery.
  COST     -- per route: composer D(s), codebook MB, per-op RF bridge neuron counts (bind / bundle / cleanup),
              cleanup argmax width, wall-clock per gate + total, and the route-B D/concept ratio for the 2,048
              extrapolation.

THE DECISION (design SS5.4): GO for the build = the route that passes the full matrix + Gate B + Gate X + all
anti-cheats at a cost that stays feasible at 2,048 concepts.
  * route A WINS if it passes A(within-bridge clause)+B+X(V-tag)+anti-cheats multi-seed AND its cost is
    per-shard-bounded (flat to 32 bridges) -- the EXPECTED outcome -> carry route A.
  * route B is REQUIRED if route A's matrix FAILS specifically for a cross-bridge generative structure the V-tag
    identity layer cannot stand in for, AND route B's measured D at 512 extrapolates feasibly to 2,048 -> carry
    route B (with the characterized SNR risk).
  * NEGATIVE (blocks the build) if a moat breach on EITHER route (fatal), OR neither route passes the matrix at
    512, OR route B's required D at 512 already extrapolates to an infeasible 2,048-concept composer AND route A
    cannot deliver a needed cross-bridge generative capability. A NEGATIVE is the scientific deliverable.

REUSE-BY-IMPORT (NO sim/ edits; every cited piece is the ensemble runner's / validated):
  - the ensemble (route B verbatim): EnsembleCortexAgent, build_ensemble_cortices, gate_X_conv, gate_X_vtag,
    anticheat_C3_moat, _make_union_codebook_for_matrix (cortex_conversation_ensemble_derisk).
  - the single-shard gates (route-independent, reused for both routes): gate_A_matrix, gate_B_generalization,
    anticheat_C1_permuted, anticheat_C4_random_shard, GEN_ACTIONS (cortex_conversation_capability_derisk).
  - the corpus + cross-bridge machinery: SHARD_NAMES, build_bridge_corpus, author_cross_facts, cross_bridge_eval,
    moat_eval, GradedBridge (multibridge_graded_derisk).
  - the composer + the moat: RFPhasorComposer, RelationalFamiliarityGate.

ADAPTATIONS vs the ensemble runner (the NEW code; noted because per-bridge composers have never been wired --
see the module-level ADAPTATIONS string + the run summary):
  A1. PerBridgeCortexAgent: a {shard: RFPhasorComposer} dict (each over its 64-concept shard vocab + the shared
      aux tokens) + a word->shard query router, instead of the ensemble's ONE union composer. WITHIN-shard facts
      bind in the agent-word's shard's composer; CROSS-shard facts go to the V-tag layer (route A does NOT bind
      them generatively). Each shard gets its OWN RelationalFamiliarityGate moat (or a shared host check) -- the
      moat surface is per-bridge, checked by the abstention battery.
  A2. Route A's Gate A forces the clause WITHIN a bridge: it runs the single-shard gate_A_matrix on EACH shard's
      OWN cortex codebook (so all SVO roles incl. the 3 clause words live in ONE shard) and reports the matrix
      per-bridge (a route-A bridge passes if its within-bridge matrix passes); the ensemble's interleaved-union
      cross-bridge clause is route B's matrix.
  A3. Route A's cross-bridge Gate X is the V-tag IDENTITY layer (gate_X_vtag, reused verbatim) -- the canonical
      route-A cross-bridge mechanism -- PLUS X-pb (a per-bridge-composer WITHIN-bridge fact-recall plumbing check
      that runs on CPU, so route A's NEW dispatch glue is exercised end-to-end even when --skip-vtag drops the
      GPU V-tag layer on the CPU smoke). Route B's Gate X is the generative X-conv (union composer).
  A4. COST LOGGING: a cheap analytic cost block per route (composer D(s), codebook bytes, per-op RF bridge neuron
      counts from the class's _bind/_bundle/_cleanup formulas, cleanup argmax width) + measured wall-clock per
      gate; route B additionally logs the D/concept ratio it MEASURED at V=512 for the 2,048 extrapolation.

Run (REAL small-scale 8-bridge A-vs-B de-risk -- GPU for the graded spiking learn + the V-tag layer; the matrix +
generalization + moat + per-bridge-composer reads are numpy). Route B (union composer + D-sweep) and route A
(per-bridge composers) are SEPARATE invocations via --composer:
  # ROUTE B (union composer, sweep D to find the clause-cell threshold at V=512):
  SIM_BACKEND=cupy python -u -m research.runners.phase1_composer_ab_derisk \
      --mode full --seeds 42,43,44 --cortex learned --composer union \
      --n-bridges 8 --concepts-per-bridge 64 --D-sweep 512,1024,1536,2048 \
      --n-pool 2400 --pattern-size 100 --homeo oja --homeo-target 40 --cycles 10 \
      --out research/findings/raw/_phase1_composer_routeB_512.json
  # ROUTE A (per-bridge composers + cross-bridge V-tag):
  SIM_BACKEND=cupy python -u -m research.runners.phase1_composer_ab_derisk \
      --mode full --seeds 42,43,44 --cortex learned --composer per-bridge --per-bridge-D 256 \
      --n-bridges 8 --concepts-per-bridge 64 \
      --n-pool 2400 --pattern-size 100 --homeo oja --homeo-target 40 --cycles 10 \
      --out research/findings/raw/_phase1_composer_routeA_512.json

Tiny CPU plumbing smoke (BOTH routes; proves they RUN end-to-end, NOT the science; ~<120s each):
  SIM_BACKEND=numpy python -u -m research.runners.phase1_composer_ab_derisk --mode full --seeds 42 \
      --composer per-bridge --smoke
  SIM_BACKEND=numpy python -u -m research.runners.phase1_composer_ab_derisk --mode full --seeds 42 \
      --composer union --D-sweep 96,128 --smoke
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

# ---- the composer + the moat (REUSE) ----
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402
from research.runners.familiarity_gate_v320_validation import RelationalFamiliarityGate  # noqa: E402

# ---- the ensemble runner (route B verbatim + the shared cortex/cross machinery; REUSE) ----
from research.runners.cortex_conversation_ensemble_derisk import (  # noqa: E402
    EnsembleCortexAgent,
    build_ensemble_cortices,
    gate_X_conv,
    gate_X_vtag,
    anticheat_C3_moat,
    _make_union_codebook_for_matrix,
)
# ---- the single-shard gates (route-independent; reused for BOTH routes; REUSE) ----
from research.runners.cortex_conversation_capability_derisk import (  # noqa: E402
    CortexCodebook,
    gate_A_matrix,
    gate_B_generalization,
    anticheat_C1_permuted,
    anticheat_C4_random_shard,
    GEN_ACTIONS,
)
# ---- the corpus + cross-bridge fact authoring (REUSE) ----
from research.runners.multibridge_graded_derisk import (  # noqa: E402
    SHARD_NAMES,
    build_bridge_corpus,
    author_cross_facts,
)

ADAPTATIONS = (
    "A1) PerBridgeCortexAgent: a {shard: RFPhasorComposer} dict (each over its 64-concept shard vocab + shared "
    "aux) + a word->shard router, replacing the ensemble's ONE union composer; within-shard facts bind in the "
    "agent-word's shard's composer, cross-shard facts go to the V-tag layer (route A does NOT bind them "
    "generatively); each shard gets its own moat (the per-bridge abstention surface). A2) route A's Gate A runs "
    "the single-shard gate_A_matrix on EACH shard's OWN cortex codebook so the clause's 3 words live in ONE "
    "shard (within-bridge clause); the ensemble's interleaved cross-bridge clause is route B's matrix. A3) route "
    "A's Gate X is the V-tag IDENTITY layer (gate_X_vtag) + X-pb (a per-bridge-composer within-bridge "
    "fact-recall plumbing check on CPU, so route A's dispatch is exercised even when --skip-vtag drops the GPU "
    "V-tag layer); route B's Gate X is the generative X-conv. A4) per-route COST logging (composer D(s), "
    "codebook bytes, per-op RF bridge neuron counts, cleanup argmax width, wall-clock per gate; route B also "
    "logs the V=512 D/concept ratio for the 2,048 extrapolation)."
)


# ===========================================================================
# Per-op RF bridge cost formulas, read DIRECTLY from RFPhasorComposer's op code (no bridge built):
#   _bind   (rf_phasor_composer.py:117-126)  -> a 2D-neuron RF bridge per bind.
#   _bundle (rf_phasor_composer.py:128-136)  -> an (L+1)D-neuron RF bridge per bundle of L roles.
#   _cleanup (rf_phasor_composer.py:247-252) -> argmax cosine over the codebook (width = #concepts the cleanup
#                                               ranks; O(D) per entry).
# A 3-role SVO fact binds 3 roles -> bundle of L=3.  Codebook bytes: #concepts * D * 8 (float64 phases dict).
# ===========================================================================
def composer_cost(D, n_concepts_cleanup, n_codebook_entries, *, label):
    """Analytic per-op cost of a composer at dimension D whose cleanup ranks `n_concepts_cleanup` concepts and
    whose codebook holds `n_codebook_entries` phasor codes. All counts from the class's op formulas; no bridge
    is constructed. Returns a dict the run summary logs + the A-vs-B comparison reads."""
    L_svo = 3  # an SVO fact binds agent+action+patient
    return {
        "label": label,
        "D": int(D),
        "cleanup_argmax_width": int(n_concepts_cleanup),       # the cleanup ranks this many concepts
        "n_codebook_entries": int(n_codebook_entries),
        "codebook_bytes": int(n_codebook_entries) * int(D) * 8,  # float64 phases dict
        "codebook_MB": round(int(n_codebook_entries) * int(D) * 8 / 1e6, 3),
        "per_bind_rf_neurons": 2 * int(D),                      # _bind: 2D-neuron RF bridge
        "per_bundle_rf_neurons": (L_svo + 1) * int(D),         # _bundle(L=3): 4D-neuron RF bridge
        "cleanup_float_ops": int(n_concepts_cleanup) * int(D),  # argmax over width, O(D) each
    }


# ===========================================================================
# Cortex-codebook dimension consistency.
#
# CRITICAL: a CortexCodebook's `phase_codes` (the DG-decorrelated phases the composer BINDS) are length-D, where
# D is the codebook's OWN dimension; the composer that consumes them MUST be built at the SAME D or the bind
# kick broadcast fails. The raw graded `codes` (and therefore Gate B's generalization, which reads them) are
# dimension-`dim`-INDEPENDENT of D. The ensemble runner sidesteps this by using ONE `args.D` for both the
# codebook and the composer; this de-risk uses DIFFERENT composer D per route (route A's per_bridge_D; route B's
# swept D), so we must re-wrap each codebook at the composer's D before a composer touches its phases. The
# re-wrap is a cheap numpy DG re-encode from the EXISTING raw `codes`/`labels` (NO re-learn -- the graded codes
# are unchanged; only the D-length decorrelated phases are re-derived).
# ===========================================================================
def rebuild_cortices_at_D(cortices, D, args, seed):
    """Return a {shard -> CortexCodebook} whose phase_codes are at dimension D (re-derived from the EXISTING raw
    graded codes; graded codes/labels unchanged -> Gate B generalization is identical). Cheap (numpy DG encode)."""
    out = {}
    for shard, cb in cortices.items():
        if int(cb.D) == int(D):
            out[shard] = cb
            continue
        out[shard] = CortexCodebook(cb.words, cb.codes, cb.labels, cb.S, D=int(D),
                                    dg_n_pool=cb.dg_n_pool, dg_pattern_size=cb.dg_pattern_size,
                                    seed=seed, source=cb.source)
    return out


# ===========================================================================
# ROUTE A -- the per-bridge-composer agent.
#
# Holds a {shard: RFPhasorComposer} dict (each composer over ONLY its 64-concept shard vocab + the shared aux
# tokens it needs -- action verbs, AFFIRM/NEGATE, attribute words) with that shard's DG-decorrelated phases as
# its grounded_codes. A word->shard router dispatches WITHIN-bridge facts/queries to the agent-word's shard's
# composer; CROSS-bridge facts are NOT bound generatively here (they go to the V-tag layer in Gate X). Each shard
# carries its OWN RelationalFamiliarityGate moat (the per-bridge abstention surface).
#
# This is the design's route-A object (SS2.1): the per-op cost is vocabulary-INDEPENDENT (D set by the 64-concept
# shard, cleanup over 64), so it is flat from 8 to 32 bridges. It does NOT subclass EnsembleCortexAgent (which
# builds one union composer); it composes the same validated pieces per shard.
# ===========================================================================
class PerBridgeCortexAgent:
    """Route A: per-bridge composers + a word->shard query router. NEW glue only -- every per-shard piece (the
    RFPhasorComposer, the cortex graded read, the RelationalFamiliarityGate moat) is validated; only the
    dispatch-by-shard is new."""

    def __init__(self, cortices: dict, *, seed=42, D=256, extra_vocab=None, enable_fallback=True,
                 fallback_novelty_thr=None):
        # cortices: {shard_name -> CortexCodebook}
        self.seed = int(seed)
        self.cortices = dict(cortices)
        self.D = int(D)
        self.enable_fallback = bool(enable_fallback)
        self._gate_thr = fallback_novelty_thr
        aux = set(extra_vocab or [])
        # word -> shard map (the query router)
        self.word_to_shard = {}
        for shard, cb in self.cortices.items():
            for w in cb.words:
                self.word_to_shard[w] = shard
        # one composer + one moat gate PER shard, each over THAT shard's concepts + the shared aux vocab.
        self.composers = {}
        self.gates = {}
        for shard, cb in self.cortices.items():
            vocab = sorted(set(cb.words) | aux)
            comp = RFPhasorComposer(seed=seed, D=D, vocab=vocab, period=200,
                                    grounded_codes=cb.phase_codes)
            self.composers[shard] = comp
            self.gates[shard] = RelationalFamiliarityGate(comp)

    # ---- dispatch helpers ----
    def _shard_of(self, word):
        return self.word_to_shard.get(word)

    def _composer_for(self, word):
        sh = self._shard_of(word)
        return self.composers.get(sh) if sh is not None else None

    def _cortex_for(self, word):
        sh = self._shard_of(word)
        return self.cortices.get(sh) if sh is not None else None

    def is_within_bridge(self, *words):
        """True iff every supplied concept word lives in the SAME shard (a within-bridge fact/query)."""
        shards = {self._shard_of(w) for w in words if isinstance(w, str)}
        shards.discard(None)
        return len(shards) == 1

    # ---- fact storage (within-bridge only; cross-bridge facts go to the V-tag layer) ----
    def store_fact(self, agent, action, patient, polarity=None):
        """Store a WITHIN-bridge SVO fact in the agent-word's shard's composer. The patient may be an attributed
        tuple (adj, noun) or a Clause -- those must also be within the same shard for route A to bind them."""
        comp = self._composer_for(agent)
        if comp is None:
            return False
        comp.store(agent, action, patient, polarity=polarity)
        return True

    def hear_clause_fact(self, agent, action, clause, polarity=None):
        comp = self._composer_for(agent)
        if comp is None:
            return False
        comp.store(agent, action, clause, polarity=polarity)
        return True

    def reimprint_gate(self):
        """Re-imprint every shard's familiarity gate over its composer's current kb (call after storing facts)."""
        for shard, comp in self.composers.items():
            g = RelationalFamiliarityGate(comp)
            g.imprint_facts()
            self.gates[shard] = g

    # ---- conversational queries (route to the agent/patient word's shard's composer; host moat + within-bridge
    #      graded fallback) ----
    def what_does(self, agent, action):
        comp = self._composer_for(agent)
        if comp is None:
            return None
        exact = comp.query_patient(agent, action)
        if exact is not None:
            return exact
        if not self.enable_fallback:
            return None
        return self._graded_fallback_patient(agent, action)

    def who_does(self, action, patient):
        comp = self._composer_for(patient)
        if comp is None:
            return None
        exact = comp.query_agent(action, patient)
        if exact is not None:
            return exact
        if not self.enable_fallback:
            return None
        return self._graded_fallback_agent(action, patient)

    def is_it_true(self, agent, action, patient):
        comp = self._composer_for(agent)
        if comp is None:
            return "unknown"
        return comp.ask_yes_no(agent, action, patient)

    def _graded_fallback_patient(self, agent, action):
        """Within-bridge graded fallback: the cortex-similar known agent for `action` in `agent`'s shard whose
        fact is familiar -> its patient. Same shard by construction (the composer + cortex are the agent's
        shard's)."""
        cb = self._cortex_for(agent)
        comp = self._composer_for(agent)
        if cb is None or comp is None:
            return None
        q = cb.graded_codes.get(agent)
        if q is None:
            return None
        gate = self.gates.get(self._shard_of(agent))
        cand = [(f["agent"], f) for f, _ in comp.kb
                if isinstance(f.get("agent"), str) and f.get("action") == action]
        scored = []
        for a2, f in cand:
            c2 = cb.graded_codes.get(a2)
            if c2 is None:
                continue
            cos = float(np.dot(q, c2) / (np.linalg.norm(q) * np.linalg.norm(c2) + 1e-12))
            scored.append((cos, a2))
        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])
        _, best_a = scored[0]
        if self._gate_thr is not None and gate is not None:
            if gate.novelty_patient(best_a, action) >= self._gate_thr:
                return None
        return comp.query_patient(best_a, action)

    def _graded_fallback_agent(self, action, patient):
        cb = self._cortex_for(patient)
        comp = self._composer_for(patient)
        if cb is None or comp is None:
            return None
        q = cb.graded_codes.get(patient)
        if q is None:
            return None
        gate = self.gates.get(self._shard_of(patient))
        cand = [(f["patient"], f) for f, _ in comp.kb
                if isinstance(f.get("patient"), str) and f.get("action") == action]
        scored = []
        for p2, f in cand:
            c2 = cb.graded_codes.get(p2)
            if c2 is None:
                continue
            cos = float(np.dot(q, c2) / (np.linalg.norm(q) * np.linalg.norm(c2) + 1e-12))
            scored.append((cos, p2))
        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])
        _, best_p = scored[0]
        if self._gate_thr is not None and gate is not None:
            if gate.novelty_agent(action, best_p) >= self._gate_thr:
                return None
        return comp.query_agent(action, best_p)


# ===========================================================================
# ROUTE A -- Gate A: the conversational matrix WITHIN a bridge (clause forced within one shard).
# Runs the single-shard gate_A_matrix on EACH shard's OWN cortex codebook (so all SVO roles, including the 3
# clause words, live in ONE shard -- the within-bridge generative clause). A route-A bridge passes if its
# within-bridge matrix passes (>= 5/6 cells + the moat). The per-bridge abstention battery is the route-A
# abstention surface (each composer has its own moat). Reports per-bridge + an aggregate (all bridges pass).
# ===========================================================================
def gate_A_routeA_per_bridge(cortices, seed, D):
    """Route A's Gate A: per-shard within-bridge conversational matrix (gate_A_matrix on each shard's cortex)."""
    per_bridge = {}
    for shard, cb in cortices.items():
        A = gate_A_matrix(cb, seed, D)
        per_bridge[shard] = A
    n_pass = sum(1 for A in per_bridge.values() if A["n_cells_pass"] >= 5 and A["moat_holds"])
    any_breach = any((not A["cells"]["abstention"]) or A["cells"]["abstention_battery_breaches"] > 0
                     for A in per_bridge.values())
    return {
        "per_bridge": per_bridge,
        "n_bridges_pass": int(n_pass),
        "n_bridges": len(per_bridge),
        "all_bridges_pass": bool(n_pass == len(per_bridge)),
        "any_abstention_breach": bool(any_breach),
    }


# ===========================================================================
# ROUTE A -- Gate X (X-pb): a per-bridge-composer WITHIN-bridge fact-recall plumbing check (CPU/numpy).
# Exercises route A's NEW dispatch glue end-to-end even when --skip-vtag drops the GPU V-tag layer (the CPU
# smoke). For each shard, store a within-shard SVO fact (`m0 eats m1`, both in the shard) through the agent's
# store_fact (which routes to the shard's composer), then query who/what back through the agent's router and
# verify exact identity recall + abstention on a never-stored within-shard cue. This is NOT the canonical Gate X
# (that is the cross-bridge V-tag identity); it is the route-A glue check so the per-bridge dispatch is proven on
# CPU. (Route A's cross-bridge Gate X proper is gate_X_vtag, the GPU V-tag layer, reused verbatim.)
# ===========================================================================
def gate_X_routeA_perbridge_plumbing(cortices, seed, D):
    """Route A within-bridge dispatch plumbing: store + recall a within-shard fact per bridge via the router."""
    agent = PerBridgeCortexAgent(cortices, seed=seed, D=D, extra_vocab=["eats"], enable_fallback=False)
    rel = "eats"
    n_what = n_what_ok = 0
    n_who = n_who_ok = 0
    abst_breaches = 0
    n_abst = 0
    stored = []
    for shard, cb in cortices.items():
        if len(cb.words) < 2:
            continue
        a, p = cb.words[0], cb.words[1]
        if not agent.is_within_bridge(a, p):
            continue
        agent.store_fact(a, rel, p)
        stored.append((shard, a, p))
    agent.reimprint_gate()
    for (shard, a, p) in stored:
        n_what += 1
        n_what_ok += int(agent.what_does(a, rel) == p)
        n_who += 1
        n_who_ok += int(agent.who_does(rel, p) == a)
        # abstention: a never-stored within-shard cue (a real concept in the shard, the relation, never stored)
        cb = cortices[shard]
        for w in cb.words[2:4]:
            n_abst += 1
            if agent.what_does(w, rel) is not None:
                abst_breaches += 1
    return {
        "n_bridges_exercised": len(stored),
        "what_acc": (n_what_ok / n_what) if n_what else None,
        "who_acc": (n_who_ok / n_who) if n_who else None,
        "abstention_probes": n_abst, "abstention_breaches": int(abst_breaches),
    }


# ===========================================================================
# Cost blocks per route (analytic; no bridge built).
# ===========================================================================
def routeA_cost(cortices, D):
    """Route A: per-bridge composer cost (each over its 64-concept shard) -- the cost is per-shard and
    vocabulary-INDEPENDENT. Reports the per-composer cost + the total across all bridges."""
    per_bridge = {}
    total_codebook_bytes = 0
    aux_n = 5  # the shared aux vocab (action verbs + AFFIRM/NEGATE + attribute words) -- small, ~independent
    for shard, cb in cortices.items():
        n_concepts = len(cb.words)
        # the composer's codebook = the shard concepts + aux + 2 polarity tags, each a D-phase code
        n_codebook = n_concepts + aux_n + 2
        c = composer_cost(D, n_concepts_cleanup=n_concepts, n_codebook_entries=n_codebook,
                          label=f"routeA[{shard}]")
        per_bridge[shard] = c
        total_codebook_bytes += c["codebook_bytes"]
    n_b = len(cortices)
    return {
        "route": "A",
        "per_bridge_composer_D": int(D),
        "n_bridges": int(n_b),
        "per_bridge_cost": per_bridge,
        "total_codebook_bytes": int(total_codebook_bytes),
        "total_codebook_MB": round(total_codebook_bytes / 1e6, 3),
        # the load-bearing facts: per-op cost is vocabulary-independent (set by the 64-concept shard).
        "per_bind_rf_neurons": 2 * int(D),
        "per_bundle_rf_neurons": 4 * int(D),
        "cleanup_argmax_width": (len(next(iter(cortices.values())).words) if cortices else 0),
        "vocabulary_independent_per_op": True,
    }


def routeB_cost(union_n_concepts, D):
    """Route B: one union composer over `union_n_concepts` at dimension D. The codebook + per-op RF bridge sizes +
    the cleanup argmax width ALL scale with the union vocabulary -- the cost route A avoids."""
    n_codebook = union_n_concepts + 5 + 2   # union concepts + aux + polarity
    c = composer_cost(D, n_concepts_cleanup=union_n_concepts, n_codebook_entries=n_codebook,
                      label="routeB[union]")
    return {
        "route": "B",
        "union_composer_D": int(D),
        "union_n_concepts": int(union_n_concepts),
        "codebook_MB": c["codebook_MB"],
        "per_bind_rf_neurons": c["per_bind_rf_neurons"],
        "per_bundle_rf_neurons": c["per_bundle_rf_neurons"],
        "cleanup_argmax_width": c["cleanup_argmax_width"],
        "cleanup_float_ops": c["cleanup_float_ops"],
        "vocabulary_independent_per_op": False,
    }


# ===========================================================================
# Shared Gate B + C1 + C4 (route-independent; generalization reads cortex codes, neither composer touches it).
# ===========================================================================
def run_gate_B_and_controls(all_corpora, cortices, seed, args, composer_D):
    """Gate B (within-bridge generalization) + C1 + C4. `composer_D` is the dimension of the cortices' phase
    codes (the B1-conv fallback builds a composer over them, so it MUST match): route A passes per_bridge_D,
    route B passes its build-time D. The B1 generalization number itself reads the raw graded codes (D-
    independent); only the B1-conv conversational realization touches a composer."""
    per_bridge_B = {}
    per_bridge_C1 = {}
    for bc in all_corpora:
        sh = bc["shard"]
        cb = cortices[sh]
        B = gate_B_generalization(cb, seed, composer_D, args, n_sub=bc["n_sub"], per_sub=bc["per_sub"])
        per_bridge_B[sh] = B
        C1 = anticheat_C1_permuted(cb, seed, composer_D, args)
        per_bridge_C1[sh] = C1
        print(f"    [{sh:>9}] B1 gen acc={B['b1_accuracy']:.3f} "
              f"(chance={B['chance']:.3f}, {B['ratio_vs_chance']:.1f}x)"
              + (f"  B2 fa={B['b2']['false_accepts']} abstains_all={B['b2']['abstains_all']}"
                 if B["b2"] is not None else "")
              + f"  | C1 perm B1={C1['b1_permuted']:.3f} collapses={C1['collapses']}", flush=True)
    C4 = anticheat_C4_random_shard(all_corpora, None, seed, composer_D, args)
    print(f"    C4 random-shard: B1={C4['b1_random_shard']:.3f} collapses={C4['collapses']} "
          f"(n={C4['n_random_members']})", flush=True)
    return per_bridge_B, per_bridge_C1, C4


# ===========================================================================
# Per-seed driver (dispatches on --composer).
# ===========================================================================
def run_seed(seed, args):
    print(f"\n{'='*100}", flush=True)
    print(f"  PHASE-1 COMPOSER A-vs-B DE-RISK -- SEED {seed} -- ROUTE {args.composer.upper()} "
          f"-- mode={args.mode} cortex={args.cortex} n_bridges={args.n_bridges}", flush=True)
    print(f"{'='*100}", flush=True)

    shard_names = SHARD_NAMES[:args.n_bridges]
    all_corpora = [build_bridge_corpus(sn, args.concepts_per_bridge, seed, args) for sn in shard_names]
    for bc in all_corpora:
        print(f"  [shard {bc['shard']:>9}] {len(bc['members'])} concepts "
              f"({bc['n_sub']} sub x {bc['per_sub']}), {bc['n_facts']} within-shard facts", flush=True)
    union_n_concepts = sum(len(bc["members"]) for bc in all_corpora)

    out = {"seed": seed, "mode": args.mode, "route": args.composer, "cortex_source": args.cortex,
           "shards": shard_names, "union_n_concepts": int(union_n_concepts)}

    # ---- build the per-shard cortex codebooks (shared by both routes) ----
    print(f"\n  [building {len(all_corpora)} cortex codebooks ({args.cortex})]", flush=True)
    t_cortex = time.time()
    cortices = build_ensemble_cortices(all_corpora, seed, args)
    out["cortex_build_seconds"] = round(time.time() - t_cortex, 1)
    out["graded_stats"] = {sh: cb.graded_stats() for sh, cb in cortices.items()}

    timings = {}

    if args.composer == "per-bridge":
        out.update(_run_route_A(all_corpora, cortices, seed, args, union_n_concepts, timings))
    else:
        out.update(_run_route_B(all_corpora, cortices, seed, args, union_n_concepts, timings))

    out["timings_seconds"] = {k: round(v, 2) for k, v in timings.items()}
    return out


def _run_route_A(all_corpora, cortices, seed, args, union_n_concepts, timings):
    out = {}
    # the per-bridge composers bind the cortices' phase_codes -> the codebooks must be at the per-bridge composer
    # D (a cheap numpy re-wrap of the SAME raw graded codes; Gate B's graded read is unchanged).
    cortices = rebuild_cortices_at_D(cortices, args.per_bridge_D, args, seed)

    # ---- GATE A -- the within-bridge conversational matrix (clause forced within one shard) ----
    if args.mode in ("matrix", "full"):
        print(f"\n  {'-'*96}\n  ROUTE A -- GATE A: within-bridge conversational matrix (per shard; "
              f"clause within ONE bridge)\n  {'-'*96}", flush=True)
        t0 = time.time()
        A = gate_A_routeA_per_bridge(cortices, seed, args.per_bridge_D)
        timings["gate_A"] = time.time() - t0
        for sh, a in A["per_bridge"].items():
            print(f"    [{sh:>9}] cells_pass={a['n_cells_pass']}/6 moat={a['moat_holds']} "
                  f"(abst_battery={a['cells']['abstention_battery_breaches']})", flush=True)
        print(f"    n_bridges_pass={A['n_bridges_pass']}/{A['n_bridges']}  "
              f"all_bridges_pass={A['all_bridges_pass']}  any_abstention_breach={A['any_abstention_breach']}",
              flush=True)
        out["gate_A"] = A

    # ---- GATE B -- within-bridge generalization (shared with route B) + C1 + C4 ----
    if args.mode in ("generalize", "full"):
        print(f"\n  {'-'*96}\n  GATE B -- within-bridge generalization (per bridge) + C1/C4 anti-cheats\n"
              f"  {'-'*96}", flush=True)
        t0 = time.time()
        per_bridge_B, per_bridge_C1, C4 = run_gate_B_and_controls(
            all_corpora, cortices, seed, args, composer_D=args.per_bridge_D)
        timings["gate_B"] = time.time() - t0
        out["gate_B"] = per_bridge_B
        out["anticheat_C1"] = per_bridge_C1
        out["anticheat_C4"] = C4

    # ---- GATE X -- cross-bridge composition: V-tag identity (canonical) + X-pb (per-bridge dispatch plumbing) ----
    if args.mode in ("cross", "full"):
        cross_facts = author_cross_facts(all_corpora, seed, args.n_cross_facts)
        print(f"\n  {'-'*96}\n  ROUTE A -- GATE X: cross-bridge IDENTITY (V-tag) + X-pb (per-bridge dispatch)\n"
              f"  {'-'*96}", flush=True)
        print(f"    [authored {len(cross_facts)} cross-bridge facts] e.g. {cross_facts[:3]}", flush=True)
        out["cross_facts"] = cross_facts

        # X-pb: the per-bridge-composer within-bridge fact-recall plumbing check (CPU/numpy; always runs)
        t0 = time.time()
        Xpb = gate_X_routeA_perbridge_plumbing(cortices, seed, args.per_bridge_D)
        timings["gate_X_pb"] = time.time() - t0
        print(f"    X-pb (per-bridge dispatch, within-bridge SVO recall): what_acc={Xpb['what_acc']} "
              f"who_acc={Xpb['who_acc']} (n_bridges={Xpb['n_bridges_exercised']}) | abstention breaches="
              f"{Xpb['abstention_breaches']}/{Xpb['abstention_probes']}", flush=True)
        out["gate_X_pb"] = Xpb

        # X-vtag: the canonical route-A cross-bridge identity layer (GPU; skipped on the CPU smoke)
        if args.skip_vtag:
            print(f"    X-vtag: SKIPPED (--skip-vtag; the live-spiking V-tag layer needs GPU bridges)",
                  flush=True)
            out["gate_X_vtag"] = None
        else:
            print(f"    X-vtag (spiking V-tag identity layer over the graded `pool` regions):", flush=True)
            t0 = time.time()
            Xv = gate_X_vtag(all_corpora, cross_facts, seed, args)
            timings["gate_X_vtag"] = time.time() - t0
            print(f"      M3 TRUE: top2={Xv['m3']['top2_fraction']:.2f} top1={Xv['m3']['top1_fraction']:.2f} "
                  f"signal/floor={Xv['m3']['mean_signal_vs_floor']:.2f}x [{Xv['band']}]", flush=True)
            print(f"      Cx PERMUTED (FIXED M7, score WRONG target): "
                  f"top2={Xv['cx_permuted']['top2_fraction']:.2f} collapses={Xv['cx_collapses']}", flush=True)
            out["gate_X_vtag"] = Xv

        # C3 moat over the cross-bridge facts (CPU/numpy)
        print(f"    C3 moat alongside host (over cross-bridge facts):", flush=True)
        t0 = time.time()
        C3 = anticheat_C3_moat(all_corpora, cross_facts, seed, args)
        timings["anticheat_C3"] = time.time() - t0
        print(f"      agreement={C3['agreement']:.3f} host-abstain/gate-accept={C3['host_abstain_gate_accept']} "
              f"(MUST be 0) floor-false-accepts={C3['abstention_floor_false_accepts']} (MUST be 0) "
              f"lesion-collapses={C3['lesion_collapsed']} -> moat_intact={C3['m4_moat_intact']}", flush=True)
        out["anticheat_C3"] = C3

    # ---- COST (analytic) ----
    cost = routeA_cost(cortices, args.per_bridge_D)
    print(f"\n  [ROUTE A COST] per-bridge D={cost['per_bridge_composer_D']} x {cost['n_bridges']} bridges | "
          f"total codebook={cost['total_codebook_MB']} MB | per-bind RF={cost['per_bind_rf_neurons']} neurons | "
          f"per-bundle RF={cost['per_bundle_rf_neurons']} | cleanup argmax width={cost['cleanup_argmax_width']} "
          f"| vocabulary-independent per-op={cost['vocabulary_independent_per_op']}", flush=True)
    out["cost"] = cost
    return out


def _run_route_B(all_corpora, cortices, seed, args, union_n_concepts, timings):
    """Route B: the union composer. Gate A's cross-bridge clause (interleaved union) is swept over --D-sweep to
    find the MINIMUM D that passes the full matrix; Gate B/X/moat run at the largest swept D (the operating point
    the matrix needs). Reuses the ensemble's interleaved-union matrix + X-conv + V-tag + moat verbatim."""
    out = {}
    D_sweep = args.D_sweep if args.D_sweep else [args.D]
    D_sweep = sorted(set(int(d) for d in D_sweep))

    # ---- GATE A -- the cross-bridge conversational matrix, swept over D ----
    if args.mode in ("matrix", "full"):
        print(f"\n  {'-'*96}\n  ROUTE B -- GATE A: cross-bridge conversational matrix (interleaved union), "
              f"D-sweep={D_sweep}\n  {'-'*96}", flush=True)
        sweep = {}
        t0 = time.time()
        # the interleaved-union codebook is D-independent in its CODES (graded codes), but the matrix BINDS at D,
        # so we rebuild the union codebook per D (its phase_codes depend on D) -- exactly the ensemble's path.
        for D in D_sweep:
            union_cb = _make_union_codebook_for_matrix(cortices, seed, _args_with_D(args, D))
            A = gate_A_matrix(union_cb, seed, D)
            clause_ok = bool(A["cells"]["clause"])
            passes = bool(A["n_cells_pass"] >= 5 and A["moat_holds"])
            sweep[str(D)] = {"n_cells_pass": A["n_cells_pass"], "moat_holds": A["moat_holds"],
                             "clause": clause_ok, "matrix_passes": passes,
                             "cells": A["cells"]}
            print(f"    D={D:>5}: cells_pass={A['n_cells_pass']}/6 clause={clause_ok} "
                  f"moat={A['moat_holds']} -> matrix_passes={passes}", flush=True)
        timings["gate_A_sweep"] = time.time() - t0
        # the minimum D at which the FULL matrix passes (incl. the cross-bridge clause)
        passing_Ds = [int(d) for d, s in sweep.items() if s["matrix_passes"]]
        min_D_pass = min(passing_Ds) if passing_Ds else None
        clause_Ds = [int(d) for d, s in sweep.items() if s["clause"]]
        min_D_clause = min(clause_Ds) if clause_Ds else None
        d_per_concept = (min_D_pass / union_n_concepts) if (min_D_pass is not None) else None
        out["gate_A_sweep"] = sweep
        out["min_D_matrix_passes"] = min_D_pass
        out["min_D_clause_passes"] = min_D_clause
        out["D_per_concept_at_V"] = (round(d_per_concept, 3) if d_per_concept is not None else None)
        out["extrapolated_D_at_2048"] = (int(round(d_per_concept * 2048)) if d_per_concept is not None else None)
        print(f"    => min D (full matrix) = {min_D_pass} | min D (clause cell) = {min_D_clause} | "
              f"D/concept @ V={union_n_concepts} = {out['D_per_concept_at_V']} | extrapolated D @ 2048 = "
              f"{out['extrapolated_D_at_2048']}", flush=True)
        # the operating D for the downstream gates (the smallest passing D, else the largest swept)
        D_op = min_D_pass if min_D_pass is not None else max(D_sweep)
    else:
        D_op = max(D_sweep)
    out["D_operating"] = int(D_op)

    # ---- GATE B -- within-bridge generalization (shared with route A; composer-independent) + C1 + C4 ----
    if args.mode in ("generalize", "full"):
        print(f"\n  {'-'*96}\n  GATE B -- within-bridge generalization (per bridge) + C1/C4 anti-cheats\n"
              f"  {'-'*96}", flush=True)
        t0 = time.time()
        # the cortices are built at args.D; Gate B's B1-conv composer binds their phase_codes, so pass args.D.
        per_bridge_B, per_bridge_C1, C4 = run_gate_B_and_controls(
            all_corpora, cortices, seed, args, composer_D=args.D)
        timings["gate_B"] = time.time() - t0
        out["gate_B"] = per_bridge_B
        out["anticheat_C1"] = per_bridge_C1
        out["anticheat_C4"] = C4

    # ---- GATE X -- cross-bridge composition: X-conv (generative, union composer at D_op) + V-tag + Cx + C3 ----
    if args.mode in ("cross", "full"):
        cross_facts = author_cross_facts(all_corpora, seed, args.n_cross_facts)
        print(f"\n  {'-'*96}\n  ROUTE B -- GATE X: cross-bridge GENERATIVE (X-conv, union D={D_op}) + V-tag + "
              f"Cx + C3\n  {'-'*96}", flush=True)
        print(f"    [authored {len(cross_facts)} cross-bridge facts] e.g. {cross_facts[:3]}", flush=True)
        out["cross_facts"] = cross_facts

        # X-conv: the generative identity-recall realization in the union composer at the OPERATING D (CPU/numpy).
        # The union composer binds the cortices' phase_codes, so re-wrap them to D_op (cheap numpy DG re-encode).
        cortices_op = rebuild_cortices_at_D(cortices, D_op, args, seed)
        t0 = time.time()
        Xc = gate_X_conv(cortices_op, cross_facts, seed, _args_with_D(args, D_op))
        timings["gate_X_conv"] = time.time() - t0
        print(f"    X-conv (SVO generative identity recall, union D={D_op}): what_acc={Xc['what_acc']} "
              f"who_acc={Xc['who_acc']} (n_stored={Xc['n_stored']}) | abstention breaches="
              f"{Xc['abstention_breaches']}/{Xc['abstention_probes']}", flush=True)
        out["gate_X_conv"] = Xc

        # X-vtag: the V-tag identity layer + the FIXED M7 anti-cheat (Cx). GPU; skipped on the CPU smoke.
        if args.skip_vtag:
            print(f"    X-vtag: SKIPPED (--skip-vtag; the live-spiking V-tag layer needs GPU bridges)",
                  flush=True)
            out["gate_X_vtag"] = None
        else:
            print(f"    X-vtag (spiking V-tag layer over the graded `pool` regions):", flush=True)
            t0 = time.time()
            Xv = gate_X_vtag(all_corpora, cross_facts, seed, args)
            timings["gate_X_vtag"] = time.time() - t0
            print(f"      M3 TRUE: top2={Xv['m3']['top2_fraction']:.2f} top1={Xv['m3']['top1_fraction']:.2f} "
                  f"signal/floor={Xv['m3']['mean_signal_vs_floor']:.2f}x [{Xv['band']}]", flush=True)
            print(f"      Cx PERMUTED (FIXED M7, score WRONG target): "
                  f"top2={Xv['cx_permuted']['top2_fraction']:.2f} collapses={Xv['cx_collapses']}", flush=True)
            out["gate_X_vtag"] = Xv

        # C3 moat over the cross-bridge facts (CPU/numpy)
        print(f"    C3 moat alongside host (over cross-bridge facts):", flush=True)
        t0 = time.time()
        C3 = anticheat_C3_moat(all_corpora, cross_facts, seed, args)
        timings["anticheat_C3"] = time.time() - t0
        print(f"      agreement={C3['agreement']:.3f} host-abstain/gate-accept={C3['host_abstain_gate_accept']} "
              f"(MUST be 0) floor-false-accepts={C3['abstention_floor_false_accepts']} (MUST be 0) "
              f"lesion-collapses={C3['lesion_collapsed']} -> moat_intact={C3['m4_moat_intact']}", flush=True)
        out["anticheat_C3"] = C3

    # ---- COST (analytic, at the operating D) ----
    cost = routeB_cost(union_n_concepts, D_op)
    print(f"\n  [ROUTE B COST] union D={cost['union_composer_D']} over {cost['union_n_concepts']} concepts | "
          f"codebook={cost['codebook_MB']} MB | per-bind RF={cost['per_bind_rf_neurons']} neurons | "
          f"per-bundle RF={cost['per_bundle_rf_neurons']} | cleanup argmax width={cost['cleanup_argmax_width']} "
          f"({cost['cleanup_float_ops']} float-ops) | vocabulary-independent per-op="
          f"{cost['vocabulary_independent_per_op']}", flush=True)
    out["cost"] = cost
    return out


def _args_with_D(args, D):
    """A shallow copy of args with .D set to D (so the reused ensemble helpers that read args.D bind at D)."""
    import copy
    a = copy.copy(args)
    a.D = int(D)
    return a


# ===========================================================================
# Multi-seed verdict (per route) + the A-vs-B comparison.
# ===========================================================================
def aggregate(per_seed, args):
    seeds = list(per_seed.keys())
    agg = {"seeds": seeds, "mode": args.mode, "route": args.composer, "cortex_source": args.cortex,
           "b1_bar": args.a1_bar, "x_bar": args.x_bar}
    chance = 1.0 / args.n_props

    def _all(pred):
        vals = [pred(per_seed[s]) for s in seeds]
        return all(v is True for v in vals), vals

    # ---- GATE A ----
    if args.mode in ("matrix", "full"):
        if args.composer == "per-bridge":
            a_ok, a_vals = _all(lambda r: bool(r["gate_A"]["all_bridges_pass"]))
            any_breach = any(per_seed[s]["gate_A"]["any_abstention_breach"] for s in seeds)
            agg["gate_A"] = {"all_pass": a_ok, "per_seed": a_vals, "any_abstention_breach": any_breach,
                             "criterion": "every bridge's within-bridge matrix passes (>=5/6 + moat)"}
        else:
            a_ok, a_vals = _all(lambda r: r.get("min_D_matrix_passes") is not None)
            min_Ds = [per_seed[s].get("min_D_matrix_passes") for s in seeds]
            dpc = [per_seed[s].get("D_per_concept_at_V") for s in seeds]
            extrap = [per_seed[s].get("extrapolated_D_at_2048") for s in seeds]
            # the matrix's abstention breach (the moat) at the operating D
            any_breach = any(
                (per_seed[s].get("min_D_matrix_passes") is not None)
                and (not per_seed[s]["gate_A_sweep"][str(per_seed[s]["min_D_matrix_passes"])]["cells"]
                     ["abstention"])
                for s in seeds)
            agg["gate_A"] = {"all_pass": a_ok, "per_seed_min_D": min_Ds, "any_abstention_breach": any_breach,
                             "D_per_concept_per_seed": dpc, "extrapolated_D_at_2048_per_seed": extrap,
                             "criterion": "the union matrix passes at SOME swept D (incl. cross-bridge clause)"}

    # ---- GATE B (route-independent) + C1 + C4 ----
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

    # ---- GATE X (route-specific) + Cx + C3 ----
    if args.mode in ("cross", "full"):
        if args.composer == "per-bridge":
            # X-pb (per-bridge dispatch plumbing) must recall + abstain; X-vtag (identity) where present.
            xpb_ok, _ = _all(lambda r: (r["gate_X_pb"]["what_acc"] is not None
                                        and r["gate_X_pb"]["what_acc"] >= args.x_bar
                                        and r["gate_X_pb"]["who_acc"] >= args.x_bar
                                        and r["gate_X_pb"]["abstention_breaches"] == 0))
            vtag_present = all(per_seed[s].get("gate_X_vtag") is not None for s in seeds)
            if vtag_present:
                xvtag_go, _ = _all(lambda r: r["gate_X_vtag"]["band"] in ("GO", "BOUNDARY"))
                xvtag_recall_ok, _ = _all(lambda r: r["gate_X_vtag"]["m3"]["top2_fraction"] >= 0.5)
                cx_ok, _ = _all(lambda r: r["gate_X_vtag"]["cx_collapses"] is True)
            else:
                xvtag_go = xvtag_recall_ok = cx_ok = None
            c3_ok, _ = _all(lambda r: r["anticheat_C3"]["m4_moat_intact"])
            c3_lesion, _ = _all(lambda r: r["anticheat_C3"]["lesion_collapsed"])
            x_abst_breach = any(per_seed[s]["gate_X_pb"]["abstention_breaches"] > 0 for s in seeds)
            agg["gate_X"] = {"x_pb_ok": xpb_ok, "vtag_present": vtag_present,
                             "x_vtag_recall_ok": xvtag_recall_ok, "x_vtag_band_ok": xvtag_go,
                             "cx_collapses": cx_ok, "C3_moat_intact": c3_ok, "C3_lesion_collapses": c3_lesion,
                             "x_abstention_breach": x_abst_breach, "x_primary_ok": xpb_ok}
        else:
            xconv_ok, _ = _all(lambda r: (r["gate_X_conv"]["what_acc"] is not None
                                          and r["gate_X_conv"]["what_acc"] >= args.x_bar
                                          and r["gate_X_conv"]["who_acc"] >= args.x_bar
                                          and r["gate_X_conv"]["abstention_breaches"] == 0))
            vtag_present = all(per_seed[s].get("gate_X_vtag") is not None for s in seeds)
            if vtag_present:
                xvtag_go, _ = _all(lambda r: r["gate_X_vtag"]["band"] in ("GO", "BOUNDARY"))
                xvtag_recall_ok, _ = _all(lambda r: r["gate_X_vtag"]["m3"]["top2_fraction"] >= 0.5)
                cx_ok, _ = _all(lambda r: r["gate_X_vtag"]["cx_collapses"] is True)
            else:
                xvtag_go = xvtag_recall_ok = cx_ok = None
            c3_ok, _ = _all(lambda r: r["anticheat_C3"]["m4_moat_intact"])
            c3_lesion, _ = _all(lambda r: r["anticheat_C3"]["lesion_collapsed"])
            x_abst_breach = any(per_seed[s]["gate_X_conv"]["abstention_breaches"] > 0 for s in seeds)
            agg["gate_X"] = {"x_conv_ok": xconv_ok, "vtag_present": vtag_present,
                             "x_vtag_recall_ok": xvtag_recall_ok, "x_vtag_band_ok": xvtag_go,
                             "cx_collapses": cx_ok, "C3_moat_intact": c3_ok, "C3_lesion_collapses": c3_lesion,
                             "x_abstention_breach": x_abst_breach, "x_primary_ok": xconv_ok}

    # ---- combined per-route verdict ----
    verdict = None
    if args.mode == "full":
        gA, gB, gX = agg["gate_A"], agg["gate_B"], agg["gate_X"]
        moat_breach = (gA["any_abstention_breach"]
                       or (not gB["b2_zero_false_accepts"])
                       or (not gX["C3_moat_intact"]) or (not gX["C3_lesion_collapses"])
                       or gX["x_abstention_breach"])
        x_recall_ok = gX["x_primary_ok"] and (gX["x_vtag_recall_ok"] in (True, None))
        controls_collapse = (gB["C1_collapses"] and gB["C4_collapses"]
                             and (gX["cx_collapses"] in (True, None)))
        if moat_breach:
            verdict = "NEGATIVE"
        elif not gB["b1_above_chance"]:
            verdict = "NEGATIVE"
        elif not x_recall_ok:
            verdict = "NEGATIVE"
        elif not controls_collapse:
            verdict = "NEGATIVE"
        elif (gA["all_pass"] and gB["b1_all_GO"] and gB["b2_zero_false_accepts"]
              and (gX["x_vtag_band_ok"] in (True, None))):
            verdict = "GO"
        elif gA["all_pass"] and gB["b1_boundary_band"]:
            verdict = "BOUNDARY"
        else:
            verdict = "BOUNDARY"
    agg["verdict"] = verdict

    # ---- cost summary (per route) ----
    cost0 = per_seed[seeds[0]].get("cost")
    if cost0 is not None:
        agg["cost"] = cost0
    return agg


def main():
    p = argparse.ArgumentParser(
        description="Phase-1 COMPOSER-ARCHITECTURE A-vs-B de-risk (route A = per-bridge composers + cross-bridge "
                    "V-tag identity; route B = one scaled union composer + D-sweep). Decides which composer "
                    "architecture carries the step-3 true-cortex build to 32 bridges / 2,048 concepts.")
    p.add_argument("--mode", default="full", choices=["matrix", "generalize", "cross", "full"],
                   help="matrix=Gate A; generalize=Gate B + C1/C4; cross=Gate X + Cx/C3; full=all + verdict")
    p.add_argument("--composer", default="per-bridge", choices=["per-bridge", "union"],
                   help="route A = per-bridge (one RFPhasorComposer per shard + V-tag cross-bridge); "
                        "route B = union (one RFPhasorComposer over all concepts + D-sweep)")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--seed", type=int, default=None, help="single-seed override")
    p.add_argument("--cortex", default="learned", choices=["synthetic", "learned"],
                   help="graded-codebook source: 'learned' (spiking HomeostaticAssocGraph + divnorm; GPU) or "
                        "'synthetic' (build_graded_codebook; cheap CPU)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU plumbing smoke: small shards + synthetic cortex + tiny n_pool/cycles + "
                        "--skip-vtag (proves BOTH routes RUN end-to-end, NOT the science)")
    p.add_argument("--skip-vtag", action="store_true",
                   help="skip the live-spiking V-tag Gate X (X-vtag); the per-bridge / X-conv numpy paths run")
    # ensemble sizing (the multibridge curated shards) -- DEFAULT 8 bridges = 512 concepts (the de-risk scale)
    p.add_argument("--n-bridges", type=int, default=8)
    p.add_argument("--concepts-per-bridge", type=int, default=64)
    p.add_argument("--target-per-sub", type=int, default=8)
    # the graded learner (HomeostaticAssocGraph; the multibridge per-bridge recipe)
    p.add_argument("--n-pool", type=int, default=2400)
    p.add_argument("--pattern-size", type=int, default=100)
    p.add_argument("--homeo", default="oja", choices=["oja", "scaling", "none"])
    p.add_argument("--homeo-target", type=float, default=40.0)
    p.add_argument("--cycles", type=int, default=10)
    # synthetic codebook (for the cheap CPU smoke)
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
    p.add_argument("--D", type=int, default=512,
                   help="route-B union composer dimension when --D-sweep is empty (the single operating D); "
                        "ALSO the generalization phase dimension shared by both routes (it reads cortex codes "
                        "-- composer-independent)")
    p.add_argument("--per-bridge-D", type=int, default=256, help="route-A per-bridge composer dimension")
    p.add_argument("--D-sweep", default=None,
                   help="route-B ONLY: comma-separated dimensions to sweep (e.g. 512,1024,1536,2048) to find the "
                        "minimum D that passes the full matrix (incl. the cross-bridge clause). Empty -> use --D.")
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
    p.add_argument("--x-bar", type=float, default=0.7, help="Gate X exact identity-recall GO bar")
    p.add_argument("--k-neighbours", type=int, default=3)
    # multibridge gate bars reused by the imported helpers
    p.add_argument("--g1-bar", type=float, default=0.5)
    p.add_argument("--so-margin-bar", type=float, default=0.10)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    # the generalization phase reads cortex codes (composer-independent), so its dimension is --D; keep a stable
    # alias the shared helpers read.
    args.D_gen = args.D

    # parse the route-B D-sweep
    if args.D_sweep:
        args.D_sweep = [int(s.strip()) for s in str(args.D_sweep).split(",") if s.strip()]
    else:
        args.D_sweep = None

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
        args.per_bridge_D = min(args.per_bridge_D, 96)
        args.D = min(args.D, 128)
        args.D_gen = args.D
        if args.D_sweep:
            args.D_sweep = [min(d, 128) for d in args.D_sweep]
        else:
            args.D_sweep = [96, 128] if args.composer == "union" else None

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    backend = os.environ.get("SIM_BACKEND", "auto")
    t_all = time.time()
    print(f"[PHASE-1 COMPOSER A-vs-B de-risk] ROUTE={args.composer.upper()} mode={args.mode} seeds={seeds} "
          f"cortex={args.cortex} backend={backend} smoke={args.smoke}", flush=True)
    print(f"  {args.n_bridges} bridges x {args.concepts_per_bridge} concepts "
          f"(= {args.n_bridges * args.concepts_per_bridge} concepts); "
          f"route-A per-bridge D={args.per_bridge_D}; route-B union D={args.D} sweep={args.D_sweep}; "
          f"B1 bar={args.a1_bar}, X bar={args.x_bar} (chance={1.0/args.n_props:.3f}); "
          f"skip_vtag={args.skip_vtag}", flush=True)
    print(f"  ADAPTATIONS: {ADAPTATIONS}", flush=True)

    per_seed = {}
    for s in seeds:
        per_seed[str(s)] = run_seed(s, args)

    agg = aggregate(per_seed, args)

    print(f"\n{'='*100}", flush=True)
    print(f"  PHASE-1 COMPOSER A-vs-B DE-RISK SUMMARY -- ROUTE {args.composer.upper()} -- mode={args.mode}",
          flush=True)
    print(f"{'='*100}", flush=True)
    for k, v in agg.items():
        if k in ("seeds", "mode", "route", "cortex_source", "verdict", "b1_bar", "x_bar"):
            continue
        print(f"  [{k}] {v}", flush=True)
    if agg.get("verdict") is not None:
        print(f"\n  >>> ROUTE {args.composer.upper()} VERDICT: {agg['verdict']} <<<", flush=True)
    print(f"  Total elapsed: {time.time()-t_all:.1f}s", flush=True)
    print(f"{'='*100}\n", flush=True)

    # the A-vs-B decision note (the controller compares the two route JSONs; we print the route-local guidance)
    print(f"  [A-vs-B] This run measured ROUTE {args.composer.upper()}. The decision (design SS5.4) compares it "
          f"against the OTHER route's JSON:\n"
          f"    * route A wins if it passes A(within-bridge clause)+B+X(V-tag)+anti-cheats at per-shard-bounded "
          f"cost (flat to 32 bridges).\n"
          f"    * route B is required only if route A fails a NEEDED cross-bridge generative structure AND "
          f"route B's measured D/concept extrapolates feasibly to 2,048.\n"
          f"    * NEGATIVE (blocks the build) on any moat breach, or neither route passing the matrix, or "
          f"route B infeasible at 2,048 with a real cross-bridge generative need.", flush=True)

    out_data = {"aggregate": agg, "per_seed": per_seed, "args": vars(args), "adaptations": ADAPTATIONS}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        tag = "smoke" if args.smoke else args.mode
        route_tag = "routeA" if args.composer == "per-bridge" else "routeB"
        args.out = os.path.join(raw_dir, f"_phase1_composer_{route_tag}_{tag}_seed{seeds[0]}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out_data


if __name__ == "__main__":
    main()
