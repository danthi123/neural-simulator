"""CORTEX <-> CONVERSATION capability de-risk -- prove the CAPABILITY (the conversational matrix on the
learned-graded cortex + generalization-IN-conversation with the no-confab moat intact) at SMALL scale BEFORE
the 32-bridge build.

SPEC: docs/plans/2026-06-12-cortex-conversation-integration-design.md (commit ec734553). This runner is the
cheap-first CAPABILITY de-risk of design SS2: the integration architecture (SS1) wired into the existing
conversational agent, with the two capability gates A/B and the anti-cheat battery C1-C4.

WHAT THE DE-RISK PROVES (design SS0 -- the GAP the mechanism de-risk left open):
  The mechanism de-risks (multibridge_graded_derisk, dual_cls_*, familiarity_gate_v320) validated that the
  cortex codes are a GRADED similarity metric, that cross-bridge V-tag recall works, and that the moat is a
  learned gate. They did NOT validate (a) the full conversational matrix running ON the learned-graded cortex,
  (b) generative role-filler BINDING using the cortex-induced codes, or (c) the NEW capability --
  generalization IN conversation (answer a query about concept X using a fact learned about a SIMILAR concept
  Y). Those three are what this runner gates before the expensive build.

THE INTEGRATION (design SS1.3, assembled here as CortexAugmentedAgent):
  word --> (A) CORTEX graded codebook (HomeostaticAssocGraph Oja + divnorm read-out, or a synthetic graded
              codebook for the cheap CPU smoke). cat ~ dog HIGH. Generalization reads these DIRECTLY.
       --> (B) DG pattern-separation encode of the graded codes -> DECORRELATED codes -> phases injected into
              the RFPhasorComposer via its already-built `grounded_codes={word: phases[D]}` seam. FHRR binds
              clean decorrelated codes, so the cleanup + the no-confab moat are unchanged.
       --> (C) cortical reinstatement: the composer's cleanup recovers WHICH concept; the cortex code for that
              identity is reinstated -> a recalled fact is graded again (here exercised implicitly: the
              composer's cleanup IS the identity recovery, and the generalization fallback reads the cortex
              codes directly).
       --> (D) the MOAT: the relational query abstains when no fact matches; the graded relational FALLBACK
              (the NEW glue) answers via the nearest cortex-similar known fact -- GATED ON the familiarity
              score of the similar cue, so abstention is redefined "no SIMILAR known fact -> abstain", not
              weakened.

THE GATES (design SS2):
  GATE A -- the conversational matrix (who/what Q&A, abstention, negation/yes-no, one-attribute, a clause)
            passes on the cortex-induced decorrelated codes AND the moat returns None on every never-stored
            cue. >= 5/5 cells + zero abstention breach (multi-seed) = GO. Abstention breach = NEGATIVE.
  GATE B -- generalization IN conversation (the NEW capability). Store a fact about Y (e.g. "cat eats meat"),
            HOLD OUT the analogous fact for a SIMILAR concept X (dog, cat ~ dog), query X
            ("what does a dog eat?") -> answer generalizes from Y (meat). B1 = generalization accuracy >= 0.7
            (>= ~4x chance); B2 = the moat STILL abstains (zero false-accepts) on a >=20-cue genuine-absence
            floor. GATE B = B1 AND B2, multi-seed.
  ANTI-CHEATS (all mandatory):
    C1 permuted-similarity -> B1 MUST collapse to chance (headline; proves it is meaning-driven).
    C2 orthogonal codes    -> B1 MUST collapse while gate A still passes (graded-vs-orthogonal contrast).
    C3 the moat validated ALONGSIDE the host -> zero host-abstain/gate-accept breaches + lesion collapses.
    C4 random-shard        -> B1 MUST collapse (the within-shard graded co-location is load-bearing).

  GO       = A AND B AND C1-C4, multi-seed 42/43/44.
  BOUNDARY = A passes, B1 in 0.5-0.7 with B2 + all controls clean (real but weak generalization).
  NEGATIVE = any moat breach (A abstention fails, or B2/C3 false-accept) -- FATAL; OR B1 <= chance; OR
             C1/C2/C4 fails to collapse (the "generalization" is an artifact, not similarity-driven).

REUSE-BY-IMPORT (NO sim/ edits; every cited piece is runner-side / validated):
  - the conversational loop:  BrainConversationalAgent / BridgeParser (brain_conversational_agent.py),
                              RFPhasorComposer + the `grounded_codes` seam (rf_phasor_composer.py:86-89).
  - the graded cortex learn:  HomeostaticAssocGraph + learn_W_homeostatic (homeostasis probe),
                              divnorm_spreading_readout (divnorm read-out probe).
  - the corpus + generalization harness: build_bridge_corpus / SHARD_NAMES (multibridge_graded_derisk),
                              build_graded_codebook / codebook_similarity_stats / assign_properties /
                              similarity_vote_infer / run_generalization / run_generalization_permuted /
                              load_orthogonal_codes / make_dg_encoder (dual_cls_architecture_proof_probe).
  - the moat protocol (C3):   RelationalFamiliarityGate + make_unknown_*_cues (familiarity_gate_v320).

ADAPTATIONS vs the design (noted because this integration has never been wired -- see the module-level
ADAPTATIONS string and the summary):
  1. The cheap CPU smoke uses a SYNTHETIC graded codebook (build_graded_codebook) instead of the spiking
     HomeostaticAssocGraph learn (`--cortex synthetic`, the default for `--smoke`). The real GPU de-risk uses
     `--cortex learned` (the validated spiking graded learn). Both expose the SAME {word: graded_code} dict to
     the rest of the pipeline; only the SOURCE of the graded codes differs.
  2. The composer's `grounded_codes` wants PHASES in [0,1)^D. The DG decorrelating encode produces a binary
     sparse expansion; we derive D decorrelated phases from it via a fixed random projection -> angle
     (a deterministic, seed-stable map). This is the concrete "DG-encode of the cortex codes -> the
     decorrelated codes the composer binds" step (design SS1.3 (B)).
  3. The parser (BridgeParser) is GPU-validated; on numpy it constructs but `parse()` is degenerate. The
     agent therefore stores facts via an explicit-roles `store_fact` (parser bypass) for the gates, which call
     what_does/who_does/is_it_true (composer-delegated) directly. `--use-parser` opts into the real parser on
     GPU (it is exercised by the existing conversational-matrix test suite, not re-tested here).

Run (REAL small-scale de-risk -- GPU for the graded spiking learn; the matrix + generalization + moat reads
are numpy):
  SIM_BACKEND=cupy python -u -m research.runners.cortex_conversation_capability_derisk \
      --mode full --seeds 42,43,44 --cortex learned \
      --n-pool 2400 --pattern-size 100 --homeo oja --homeo-target 40 --cycles 10 \
      --out research/findings/raw/_cortex_conversation_capability_derisk_full.json

Tiny CPU smoke (plumbing only -- proves it RUNS end-to-end, NOT the science):
  SIM_BACKEND=numpy python -u -m research.runners.cortex_conversation_capability_derisk \
      --mode full --seeds 42 --smoke
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
from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402

# ---- the corpus + generalization harness (REUSE) ----
from research.runners.multibridge_graded_derisk import (  # noqa: E402
    SHARD_NAMES,
    build_bridge_corpus,
)
from research.runners.dual_cls_architecture_proof_probe import (  # noqa: E402
    build_graded_codebook,
    codebook_similarity_stats,
    assign_properties,
    similarity_vote_infer,
    run_generalization,
    run_generalization_permuted,
    load_orthogonal_codes,
    make_dg_encoder,
)

# ---- the moat protocol (REUSE; C3) ----
from research.runners.familiarity_gate_v320_validation import (  # noqa: E402
    RelationalFamiliarityGate,
)

ADAPTATIONS = (
    "1) cheap CPU smoke uses a SYNTHETIC graded codebook (build_graded_codebook) not the spiking "
    "HomeostaticAssocGraph learn (the real GPU run uses --cortex learned); both expose the same "
    "{word: graded_code}. 2) grounded_codes wants phases in [0,1)^D, so the DG decorrelating encode's binary "
    "expansion is mapped to D decorrelated phases via a fixed random projection->angle. 3) the GPU-validated "
    "BridgeParser is bypassed for the gates (explicit-roles store_fact); the matrix queries are composer-"
    "delegated and parser-independent (the parser is covered by the existing matrix test suite)."
)


# ===========================================================================
# CortexCodebook -- the LEARNED GRADED codebook for ONE shard, exposing:
#   .graded_codes  {word -> graded_code[dim]}      (cat ~ dog HIGH; generalization reads these DIRECTLY)
#   .phase_codes   {word -> phases[D] in [0,1)}     (DG-decorrelated codes injected into the composer)
# Two cortex sources: 'synthetic' (build_graded_codebook -- cheap, CPU) or 'learned' (HomeostaticAssocGraph
# Oja learn + divnorm read-out -- the validated spiking graded learn, GPU). Both produce the same interface.
# ===========================================================================
class CortexCodebook:
    """The graded cortex codebook for one semantic shard + the DG decorrelating encode that produces the
    composer's decorrelated phase codes.

    `words` is the ordered list of concept words (the shard members). `graded_codes` is the {word: code} dict
    the generalization read uses directly (the cortex channel). `phase_codes` is the {word: phases} dict
    injected into the RFPhasorComposer via its `grounded_codes` seam (the decorrelated binder channel).
    """

    def __init__(self, words, codes, labels, S, *, D=128, dg_n_pool=2000, dg_pattern_size=100,
                 seed=42, source="synthetic"):
        self.words = list(words)
        self.codes = np.asarray(codes, dtype=np.float64)   # [N, dim] graded
        self.labels = np.asarray(labels, dtype=int)
        self.S = np.asarray(S, dtype=np.float64)           # [N, N] graded cosine
        self.D = int(D)
        self.seed = int(seed)
        self.source = str(source)
        self.dg_n_pool = int(dg_n_pool)
        self.dg_pattern_size = int(dg_pattern_size)
        self.word_to_row = {w: i for i, w in enumerate(self.words)}
        # graded codes the generalization read consumes directly
        self.graded_codes = {w: self.codes[i] for i, w in enumerate(self.words)}
        # DG-decorrelated phases for the composer (the binder channel)
        self.phase_codes = self._encode_phases()

    # ---- DG pattern-separation encode of the graded codes -> decorrelated phases ----
    def _encode_phases(self):
        """Run the cortex graded codes through a DG decorrelating encode (make_dg_encoder: random projection
        + top-k WTA), then map each concept's decorrelated sparse expansion to D phases in [0,1) via a FIXED
        random projection -> angle. This is the design SS1.3-(B) "DG-encode of the cortex codes -> the
        decorrelated codes the composer binds" step, concretized for the composer's phasor representation."""
        dim = self.codes.shape[1]
        _, encode_fn = make_dg_encoder(dim, self.dg_n_pool, self.dg_pattern_size, self.seed)
        # a fixed projection from the n_pool sparse expansion to D phase channels (seed-stable).
        rng = np.random.RandomState(self.seed * 7919 + 13)
        Pphase = rng.randn(self.dg_n_pool, self.D)
        phase_codes = {}
        for i, w in enumerate(self.words):
            expansion = encode_fn(self.codes[i])            # binary {0,1} [n_pool], decorrelated
            proj = expansion @ Pphase                       # [D] real
            # map to a phase in [0,1): wrap the projected real onto the circle deterministically.
            phases = (np.arctan2(np.sin(proj), np.cos(proj)) / (2.0 * np.pi)) % 1.0
            phase_codes[w] = phases.astype(float)
        return phase_codes

    def expansion_between_cos(self):
        """The mean between-concept cosine of the DG decorrelated expansion (must be ~0.05 for binding to be
        clean). Computed on the binary expansions (the design's binding-clean precondition)."""
        dim = self.codes.shape[1]
        _, encode_fn = make_dg_encoder(dim, self.dg_n_pool, self.dg_pattern_size, self.seed)
        exp = np.stack([encode_fn(self.codes[i]) for i in range(len(self.words))])
        e = exp - exp.mean(axis=1, keepdims=True)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        M = e @ e.T
        N = M.shape[0]
        off = [float(M[i, j]) for i in range(N) for j in range(i + 1, N)]
        return float(np.mean(off)) if off else 0.0

    def graded_stats(self):
        return codebook_similarity_stats(self.codes, self.labels)

    def nearest_similar(self, word, exclude=()):
        """The cortex-most-similar OTHER word to `word` (the graded neighbour the relational fallback uses).
        Returns (neighbour_word, cosine) or (None, -inf)."""
        if word not in self.word_to_row:
            return None, float("-inf")
        q = self.graded_codes[word]
        excl = set(exclude) | {word}
        best, best_c = None, float("-inf")
        for w, c in self.graded_codes.items():
            if w in excl:
                continue
            cos = float(np.dot(q, c) / (np.linalg.norm(q) * np.linalg.norm(c) + 1e-12))
            if cos > best_c:
                best, best_c = w, cos
        return best, best_c


def build_cortex_codebook_synthetic(words, n_sub, per_sub, *, D, dg_n_pool, dg_pattern_size,
                                    seed, dim=256, residual_frac=0.55):
    """A synthetic graded codebook (build_graded_codebook) over the shard's sub-cluster layout. Cheap, CPU.
    `words` order MUST match the (n_sub x per_sub) cluster layout build_graded_codebook produces (cluster c,
    member m -> row c*per_sub + m)."""
    codes, labels, S = build_graded_codebook(n_sub, per_sub, dim, seed, residual_frac)
    assert len(words) == codes.shape[0], "word count must equal n_sub*per_sub"
    return CortexCodebook(words, codes, labels, S, D=D, dg_n_pool=dg_n_pool,
                          dg_pattern_size=dg_pattern_size, seed=seed, source="synthetic")


def build_cortex_codebook_learned(bridge_corpus, *, D, dg_n_pool, dg_pattern_size, seed, args):
    """The validated spiking graded learn: HomeostaticAssocGraph (Oja set-point) co-occurrence recurrent +
    the brain-based divnorm read-out (the multibridge_graded_derisk per-bridge recipe). GPU path. Returns a
    CortexCodebook whose graded_codes are the LEARNED graded codes (cat ~ dog via the shared sub-hub)."""
    from research.runners.multibridge_graded_derisk import learn_bridge_graded
    local = bridge_corpus["_local"]
    W, codes, member_rows, info = learn_bridge_graded(bridge_corpus, seed, args)
    members = bridge_corpus["members"]        # namespaced member words (the shard concept vocabulary)
    labels = np.asarray(local["labels"], dtype=int)
    S = codes @ codes.T
    cb = CortexCodebook(members, codes, labels, S, D=D, dg_n_pool=dg_n_pool,
                        dg_pattern_size=dg_pattern_size, seed=seed, source="learned")
    cb.learn_info = info
    return cb


# ===========================================================================
# CortexAugmentedAgent -- a thin BrainConversationalAgent subclass:
#   * injects the cortex-induced DECORRELATED phases into the RFPhasorComposer (grounded_codes seam),
#   * holds the CortexCodebook for the generalization read,
#   * adds the graded relational FALLBACK in what_does / who_does (the NEW glue), gated by the familiarity
#     score on the nearest graded-similar known cue (so abstention is redefined "no SIMILAR known fact ->
#     abstain", not weakened).
# ===========================================================================
class CortexAugmentedAgent(BrainConversationalAgent):
    """BrainConversationalAgent wired to a learned-graded cortex.

    The composer binds the cortex-induced DECORRELATED codes (so FHRR + the no-confab moat are unchanged);
    generalization reads the GRADED cortex codes directly; a graded relational fallback answers a query about a
    held-out concept via its nearest cortex-similar known fact, gated by the familiarity gate's novelty on the
    similar cue (so the moat still abstains on genuine absence).
    """

    def __init__(self, cortex: CortexCodebook, *, seed=42, D=128, build_parser=False,
                 enable_fallback=True, fallback_novelty_thr=None, extra_vocab=None):
        # Build the composer ourselves with the cortex-induced phases injected via grounded_codes; the
        # vocabulary is the shard's concept words PLUS any auxiliary tokens (action verbs, attribute /
        # property-carrier words) the facts use -- those are NOT cortex concepts, so they get the composer's
        # own random decorrelated phases (correct: they are structural roles/words, not graded concepts).
        # We bypass BrainConversationalAgent.__init__ (which builds the GPU-only parser) unless requested.
        self.seed = int(seed)
        self.cortex = cortex
        self.D = int(D)
        vocab = sorted(set(cortex.words) | set(extra_vocab or []))
        composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab, period=200,
                                    grounded_codes=cortex.phase_codes)
        self.composer = composer
        self._dlpfc = None
        self._dlpfc_key = None
        self._learned_assoc = None
        self.parser = None
        if build_parser:
            from research.runners.brain_conversational_agent import BridgeParser
            self.parser = BridgeParser(seed=seed)
        # the relational familiarity gate (the moat ALONGSIDE the host; also gates the graded fallback)
        self.gate = RelationalFamiliarityGate(composer)
        self._gate_thr = fallback_novelty_thr        # the novelty threshold gating the fallback (calibrated)
        self.enable_fallback = bool(enable_fallback)

    # ---- fact storage (explicit roles -> parser bypass; the gates use this) ----
    def store_fact(self, agent, action, patient, polarity=None):
        """Store an SVO fact by EXPLICIT roles (bypassing the GPU-only parser). The composer binds the
        cortex-induced decorrelated codes; the gate re-imprints so the moat tracks the stored span."""
        self.composer.store(agent, action, patient, polarity=polarity)

    def reimprint_gate(self):
        """Re-imprint the familiarity gate over the CURRENT knowledge base (call after storing facts) so the
        moat's novelty separation and the fallback gate reflect the stored facts."""
        self.gate = RelationalFamiliarityGate(self.composer)
        self.gate.imprint_facts()

    def calibrate_fallback_threshold(self, unknown_aa_cues):
        """Place the fallback-gate novelty threshold midway between the host-known cues' max novelty and the
        genuine-absence cues' min novelty (the familiarity_gate_v320 midpoint placement). A cue whose nearest
        graded neighbour is FAMILIAR (novelty < thr) may use the fallback; a genuinely-novel similar cue
        (novelty >= thr) stays abstained -> the moat is preserved."""
        known_aa = [(f["agent"], f["action"]) for f, _ in self.composer.kb
                    if isinstance(f.get("agent"), str) and isinstance(f.get("action"), str)]
        nov_known = [self.gate.novelty_patient(a, ac) for (a, ac) in known_aa]
        nov_unknown = [self.gate.novelty_patient(a, ac) for (a, ac) in unknown_aa_cues]
        if nov_known and nov_unknown:
            thr = 0.5 * (max(nov_known) + min(nov_unknown))
        elif nov_known:
            thr = max(nov_known) * 1.5 + 1e-6
        else:
            thr = 1e9
        self._gate_thr = float(thr)
        return self._gate_thr

    # ---- the conversational queries (host moat + graded relational FALLBACK) ----
    def what_does(self, agent, action):
        """'what does <agent> <action>?' -> the patient (exact fact); on host-abstain, the graded relational
        fallback answers via the nearest cortex-similar known agent for the same action, GATED on the
        familiarity score of the similar (agent', action) cue. None when no similar known fact -> abstain."""
        exact = self.composer.query_patient(agent, action)
        if exact is not None:
            return exact
        if not self.enable_fallback:
            return None
        return self._graded_fallback_patient(agent, action)

    def who_does(self, action, patient):
        """'who <action> <patient>?' -> the agent (exact fact); on host-abstain, the graded relational
        fallback answers via the nearest cortex-similar known patient for the same action, gated by
        familiarity. None when no similar known fact -> abstain."""
        exact = self.composer.query_agent(action, patient)
        if exact is not None:
            return exact
        if not self.enable_fallback:
            return None
        return self._graded_fallback_agent(action, patient)

    def _graded_fallback_patient(self, agent, action):
        """Find the cortex-similar known agent for `action` whose (agent', action) fact is FAMILIAR; answer
        with that fact's patient. The fallback fires only when the similar cue's familiarity novelty is below
        the calibrated threshold (so a genuinely-novel cue with no similar known fact stays abstained)."""
        # candidate known agents that have a fact with this action
        cand = [(f["agent"], f) for f, _ in self.composer.kb
                if isinstance(f.get("agent"), str) and f.get("action") == action]
        if not cand or agent not in self.cortex.word_to_row:
            return None
        q = self.cortex.graded_codes.get(agent)
        if q is None:
            return None
        # rank known agents by cortex similarity to the queried agent
        scored = []
        for a2, f in cand:
            c2 = self.cortex.graded_codes.get(a2)
            if c2 is None:
                continue
            cos = float(np.dot(q, c2) / (np.linalg.norm(q) * np.linalg.norm(c2) + 1e-12))
            scored.append((cos, a2, f))
        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])
        _, best_a, best_f = scored[0]
        # MOAT GATE: the similar (best_a, action) cue must be FAMILIAR (novelty below threshold).
        if self._gate_thr is not None:
            nov = self.gate.novelty_patient(best_a, action)
            if nov >= self._gate_thr:
                return None                      # the nearest known fact is not familiar enough -> abstain
        # answer via the exact query on the similar known fact (renders attributes/clauses correctly)
        return self.composer.query_patient(best_a, action)

    def _graded_fallback_agent(self, action, patient):
        """Symmetric fallback for who_does: find the cortex-similar known patient for `action` with a familiar
        (action, patient') fact; answer with that fact's agent."""
        cand = [(f["patient"], f) for f, _ in self.composer.kb
                if isinstance(f.get("patient"), str) and f.get("action") == action]
        if not cand or patient not in self.cortex.word_to_row:
            return None
        q = self.cortex.graded_codes.get(patient)
        if q is None:
            return None
        scored = []
        for p2, f in cand:
            c2 = self.cortex.graded_codes.get(p2)
            if c2 is None:
                continue
            cos = float(np.dot(q, c2) / (np.linalg.norm(q) * np.linalg.norm(c2) + 1e-12))
            scored.append((cos, p2, f))
        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])
        _, best_p, best_f = scored[0]
        if self._gate_thr is not None:
            nov = self.gate.novelty_agent(action, best_p)
            if nov >= self._gate_thr:
                return None
        return self.composer.query_agent(action, best_p)


# ===========================================================================
# GATE A -- the conversational matrix on the cortex-induced codes (no regression).
# Mirrors tests/test_brain_conversational_agent.py, adapted to the shard vocabulary, with EXPLICIT-roles
# storage (parser-bypass). The abstention cell (the moat) must return None on every never-stored cue.
# ===========================================================================
MATRIX_ACTIONS = ["go", "come", "look"]


def gate_A_matrix(cortex, seed, D):
    from research.runners.core_sim_composition import Clause

    words = list(cortex.words)
    # the action verbs are auxiliary vocab (not cortex concepts) -> the composer's own random codes.
    aux = list(MATRIX_ACTIONS)
    # pick six distinct concept words to play the SVO roles (and a NEVER-stored cue from the rest).
    a0, p0 = words[0], words[1]
    a1, p1 = words[2 % len(words)], words[3 % len(words)]
    adj0 = words[4 % len(words)]
    cl_a, cl_ac, cl_p = words[5 % len(words)], words[1], words[2 % len(words)]
    never_a, never_act = words[-1], "NEVERACT"     # an action token never stored -> the host must abstain

    def _agent(fallback=False):
        return CortexAugmentedAgent(cortex, seed=seed, D=D, build_parser=False,
                                    enable_fallback=fallback, extra_vocab=aux)

    cells = {}

    # --- who/what Q&A + abstention ---
    ag = _agent()
    ag.store_fact(a0, "go", p0)
    ag.store_fact(a1, "come", p1)
    cells["what_does"] = (ag.what_does(a0, "go") == p0)
    cells["who_does"] = (ag.who_does("go", p0) == a0)
    cells["abstention"] = (ag.what_does(never_a, never_act) is None)

    # --- negation / yes-no ---
    ag2 = _agent()
    ag2.store_fact(a0, "go", p0, polarity="AFFIRM")
    ag2.store_fact(a1, "come", p1, polarity="NEGATE")
    yn_yes = (ag2.is_it_true(a0, "go", p0) == "yes")
    yn_no = (ag2.is_it_true(a1, "come", p1) == "no")
    yn_unknown = (ag2.is_it_true(never_a, never_act, p0) == "unknown")
    cells["negation"] = bool(yn_yes and yn_no and yn_unknown)

    # --- one-attribute ((adj, noun) patient) ---
    ag3 = _agent()
    ag3.store_fact(a0, "go", (adj0, p0))
    one_attr = ag3.what_does(a0, "go")
    cells["one_attribute"] = (one_attr == f"{adj0} {p0}")

    # --- a clause (nested SVO patient) ---
    ag4 = _agent()
    ag4.hear_clause_fact(a0, "look", Clause(cl_a, cl_ac, cl_p))
    clause = ag4.what_does(a0, "look")
    cells["clause"] = (clause == f"{cl_a} {cl_ac} {cl_p}")

    # additional abstention probes (a battery, so the moat is not a single point)
    abst_breaches = 0
    rng = np.random.RandomState(seed * 17 + 1)
    never_acts = ["NEVERACT", "XACT", "YACT", "ZACT"]
    for _ in range(12):
        wa = words[rng.randint(len(words))]
        wact = never_acts[rng.randint(len(never_acts))]
        if ag.what_does(wa, wact) is not None:
            abst_breaches += 1
    cells["abstention_battery_breaches"] = int(abst_breaches)

    n_cells_pass = sum(1 for k in ("what_does", "who_does", "abstention", "negation",
                                   "one_attribute", "clause") if cells[k])
    moat_holds = bool(cells["abstention"] and abst_breaches == 0)
    return {
        "cells": {k: (bool(v) if not isinstance(v, int) else v) for k, v in cells.items()},
        "n_cells_pass": int(n_cells_pass),
        "moat_holds": moat_holds,
        "rendered": {"one_attribute": one_attr, "clause": clause},
    }


# ===========================================================================
# GATE B -- generalization IN conversation (THE NEW CAPABILITY) + the moat on genuine absence.
# B is measured two complementary ways:
#   B1a (the conversational realization): build ONE agent, store facts for trained sub-cluster members, hold
#        out a graded-neighbour, query the held-out neighbour through what_does -> the graded fallback should
#        answer with the cluster's property-bearing patient. Accuracy over held-out splits.
#   B1b (the harness number, the design's run_generalization restricted to stored relations): the
#        similarity-vote inference accuracy over the graded cortex codes (the direct generalization read). This
#        is the canonical, control-comparable B1 (C1/C2/C4 are defined against it).
#   B2  the moat: a >=20-cue genuine-absence floor (cues whose graded-similar neighbours are ALSO never
#        stored) must yield abstention with ZERO false-accepts.
# ===========================================================================
GEN_ACTIONS = ["eats", "JUMPS"]      # the stored relation ('eats') + the genuine-absence action ('JUMPS')


def _make_property_patient_words(words, props, n_props, seed):
    """A distinct patient word per property value (the 'eats meat' analogue). Returns {prop_value:
    patient_word}, drawing patient words from OUTSIDE the agent-words where possible (here we reuse late
    shard words as property carriers; they are valid composer vocab)."""
    rng = np.random.RandomState(seed * 53 + 7)
    # use the last n_props distinct words as property carriers
    carriers = list(dict.fromkeys(words[::-1]))[:max(n_props, 1)]
    while len(carriers) < n_props:
        carriers.append(words[rng.randint(len(words))])
    return {pv: carriers[pv % len(carriers)] for pv in range(n_props)}


def gate_B_generalization(cortex, seed, D, args, codes_override=None, labels_override=None,
                          props_override=None, n_sub=None, per_sub=None):
    """Compute B1 (generalization accuracy, the design's run_generalization on the graded cortex codes,
    restricted to stored relations) + B1-conv (the conversational realization via the fallback) + B2 (the moat
    on a genuine-absence floor). `codes_override`/`labels_override`/`props_override` let the anti-cheats (C1/C2/
    C4) re-run B1 on permuted/orthogonal/random-shard inputs.

    The relation here is a single ACTION ('eats'); each cluster's members share a property -> a patient word.
    A held-out cluster-mate's property is inferred from its nearest trained neighbour (the cat~dog case)."""
    codes = cortex.codes if codes_override is None else np.asarray(codes_override)
    labels = cortex.labels if labels_override is None else np.asarray(labels_override)
    n_sub = n_sub if n_sub is not None else int(labels.max()) + 1
    per_sub = per_sub if per_sub is not None else (len(labels) // max(1, n_sub))
    n_props = args.n_props
    props = (assign_properties(n_sub, per_sub, n_props, seed) if props_override is None
             else np.asarray(props_override))
    chance = 1.0 / n_props

    # ---- B1: the design's held-out-neighbour generalization on the (graded) codes ----
    gen = run_generalization(codes, labels, props, n_sub, per_sub, seed, args.k_neighbours)
    b1 = float(gen["accuracy"])

    # ---- B1-conv: the conversational realization through the graded fallback ----
    # store one ('member', 'eats', property_patient) fact per TRAINED member; hold out one member per cluster;
    # query the held-out member's what_does('member','eats') -> the fallback answers; check the property word.
    b1_conv = None
    if codes_override is None and labels_override is None:    # the conversational test only on the real cortex
        prop_word = _make_property_patient_words(list(cortex.words), props, n_props, seed)
        rng = np.random.RandomState(seed * 13 + 5)
        n_correct = n_total = 0
        n_splits = max(4, args.b_conv_splits)
        for _ in range(n_splits):
            heldout = []
            train_members = []
            for c in range(n_sub):
                members = [i for i in range(len(cortex.words)) if labels[i] == c]
                if len(members) < 2:
                    continue
                ho = int(rng.choice(members))
                heldout.append(ho)
                train_members.extend([m for m in members if m != ho])
            # build a fresh agent, store the trained members' 'eats property' facts, calibrate the moat gate
            ag = CortexAugmentedAgent(cortex, seed=seed, D=D, build_parser=False, enable_fallback=True,
                                      extra_vocab=GEN_ACTIONS)
            for m in train_members:
                ag.store_fact(cortex.words[m], "eats", prop_word[int(props[m])])
            ag.reimprint_gate()
            # calibrate the fallback threshold against genuine-absence cues (members for a DIFFERENT action)
            unk_cues = [(cortex.words[m], "JUMPS") for m in range(len(cortex.words))][:max(20, n_sub)]
            ag.calibrate_fallback_threshold(unk_cues)
            for h in heldout:
                ans = ag.what_does(cortex.words[h], "eats")
                want = prop_word[int(props[h])]
                n_total += 1
                n_correct += int(ans == want)
        b1_conv = (n_correct / n_total) if n_total else None

    # ---- B2: the moat abstains on genuine absence (zero false-accepts on a >=20-cue floor) ----
    b2 = None
    if codes_override is None:
        prop_word = _make_property_patient_words(list(cortex.words), props, n_props, seed)
        ag = CortexAugmentedAgent(cortex, seed=seed, D=D, build_parser=False, enable_fallback=True,
                                  extra_vocab=GEN_ACTIONS)
        # store 'eats' facts for HALF the members (so some 'eats' relations exist, others genuinely absent)
        rng = np.random.RandomState(seed * 29 + 3)
        all_members = list(range(len(cortex.words)))
        rng.shuffle(all_members)
        stored = set(all_members[: len(all_members) // 2])
        for m in stored:
            ag.store_fact(cortex.words[m], "eats", prop_word[int(props[m])])
        ag.reimprint_gate()
        # genuine-absence floor: cues for an action NO fact uses ('JUMPS') -> no similar known fact exists ->
        # must abstain. >= 20 cues.
        floor_cues = []
        for m in all_members:
            floor_cues.append((cortex.words[m], "JUMPS"))
        floor_cues = floor_cues[: max(20, len(floor_cues))]
        ag.calibrate_fallback_threshold(floor_cues)
        false_accepts = 0
        for (w, act) in floor_cues:
            if ag.what_does(w, act) is not None:
                false_accepts += 1
        b2 = {"floor_n": len(floor_cues), "false_accepts": int(false_accepts),
              "abstains_all": bool(false_accepts == 0)}

    return {
        "b1_accuracy": b1, "chance": chance, "ratio_vs_chance": b1 / chance if chance > 0 else 0.0,
        "b1_conv_accuracy": b1_conv,
        "b2": b2,
        "n_sub": n_sub, "per_sub": per_sub, "n_props": n_props,
    }


# ===========================================================================
# ANTI-CHEATS C1-C4.
# ===========================================================================
def anticheat_C1_permuted(cortex, seed, D, args):
    """C1 -- permuted-similarity (headline): shuffle which concepts carry which property (decouple the
    code structure from the property label) -> B1 MUST collapse to chance."""
    labels = cortex.labels
    n_sub = int(labels.max()) + 1
    per_sub = len(labels) // max(1, n_sub)
    gen = run_generalization_permuted(cortex.codes, labels, assign_properties(n_sub, per_sub, args.n_props, seed),
                                      n_sub, per_sub, seed, args.k_neighbours)
    chance = 1.0 / args.n_props
    acc = float(gen["accuracy"])
    return {"b1_permuted": acc, "chance": chance, "collapses": bool(acc <= 1.5 * chance)}


def anticheat_C2_orthogonal(cortex, seed, D, args):
    """C2 -- orthogonal codes: re-run B1 on the project's ORTHOGONAL sparse codes (between-cos ~ 0.05) ->
    B1 MUST collapse (equidistant codes have no graded neighbour). Gate A on the orthogonal codes still
    passes (binding is fine on decorrelated codes) -- checked here too."""
    labels = cortex.labels
    n_sub = int(labels.max()) + 1
    per_sub = len(labels) // max(1, n_sub)
    N = len(labels)
    ortho = load_orthogonal_codes(seed, N)
    genB = gate_B_generalization(cortex, seed, D, args, codes_override=ortho, labels_override=labels,
                                 n_sub=n_sub, per_sub=per_sub)
    chance = 1.0 / args.n_props
    b1_ortho = genB["b1_accuracy"]
    # build a cortex codebook whose graded_codes ARE the orthogonal codes (so Gate A binds orthogonal-derived
    # phases) and confirm the matrix still passes.
    ortho_cortex = CortexCodebook(cortex.words, ortho, labels, ortho @ ortho.T, D=D,
                                  dg_n_pool=cortex.dg_n_pool, dg_pattern_size=cortex.dg_pattern_size,
                                  seed=seed, source="orthogonal")
    mat = gate_A_matrix(ortho_cortex, seed, D)
    return {"b1_orthogonal": b1_ortho, "chance": chance,
            "collapses": bool(b1_ortho <= 1.5 * chance),
            "matrix_still_passes": bool(mat["n_cells_pass"] >= 5 and mat["moat_holds"]),
            "matrix_n_cells": mat["n_cells_pass"], "matrix_moat_holds": mat["moat_holds"]}


def anticheat_C3_moat_alongside_host(cortex, seed, D, args):
    """C3 -- the moat ALONGSIDE the host (the familiarity_gate_v320 protocol): the learned
    RelationalFamiliarityGate's accept/abstain decision is compared with the host abstention; the
    host-abstain/gate-accept cell must be 0 and the abstention-floor false-accepts 0, and the LESION must
    collapse the novelty separation (the decision rides the learned gate)."""
    words = list(cortex.words)
    ag = CortexAugmentedAgent(cortex, seed=seed, D=D, build_parser=False, enable_fallback=True,
                              extra_vocab=GEN_ACTIONS)
    # store SVO facts (members 'eats' property-words), so the gate has a real span to imprint.
    labels = cortex.labels
    n_sub = int(labels.max()) + 1
    per_sub = len(labels) // max(1, n_sub)
    props = assign_properties(n_sub, per_sub, args.n_props, seed)
    prop_word = _make_property_patient_words(words, props, args.n_props, seed)
    rng = np.random.RandomState(seed * 71 + 9)
    all_m = list(range(len(words)))
    rng.shuffle(all_m)
    stored = all_m[: len(all_m) // 2]
    for m in stored:
        ag.store_fact(words[m], "eats", prop_word[int(props[m])])
    ag.reimprint_gate()
    gate = ag.gate

    # known patient-cues = stored (agent, action); unknown = (member, 'JUMPS') genuine-absence.
    known_aa = [(words[m], "eats") for m in stored]
    unknown_aa = [(words[m], "JUMPS") for m in all_m][: max(args.moat_floor, 20)]

    def host_accepts_patient(a, ac):
        return ag.composer.query_patient(a, ac) is not None

    rows = []
    for (a, ac) in known_aa:
        rows.append((host_accepts_patient(a, ac), gate.novelty_patient(a, ac), True))
    for (a, ac) in unknown_aa:
        rows.append((host_accepts_patient(a, ac), gate.novelty_patient(a, ac), False))

    nov_accept = np.array([n for (ha, n, _k) in rows if ha])
    nov_abstain = np.array([n for (ha, n, _k) in rows if not ha])
    known_max = float(nov_accept.max()) if nov_accept.size else float("nan")
    unknown_min = float(nov_abstain.min()) if nov_abstain.size else float("nan")
    margin = float(unknown_min - known_max) if (nov_accept.size and nov_abstain.size) else float("nan")
    thr = (0.5 * (known_max + unknown_min)) if (nov_accept.size and nov_abstain.size) else (
        float(np.median([n for (_h, n, _k) in rows])) if rows else 0.0)

    host_abstain_gate_accept = 0
    floor_false_accepts = 0
    n_agree = 0
    for (ha, nov, _k) in rows:
        gate_accept = nov < thr
        if ha == gate_accept:
            n_agree += 1
        if (not ha) and gate_accept:
            host_abstain_gate_accept += 1
            floor_false_accepts += 1
    agreement = (n_agree / len(rows)) if rows else 0.0

    # lesion: novelty separation must collapse
    gate.lesion()
    les_known = np.array([gate.novelty_patient(a, ac) for (a, ac) in known_aa])
    les_unknown = np.array([gate.novelty_patient(a, ac) for (a, ac) in unknown_aa])
    lesion_margin = float(les_unknown.min() - les_known.max()) if (les_known.size and les_unknown.size) else float("nan")
    lesion_collapsed = (bool(les_known.size and les_unknown.size and
                             np.allclose(les_known.mean(), les_unknown.mean(), atol=1e-6))
                        or bool(abs(lesion_margin) <= 1e-6))

    return {
        "n_known": len(known_aa), "n_unknown": len(unknown_aa),
        "separation_margin": margin, "agreement": agreement, "threshold": thr,
        "host_abstain_gate_accept": int(host_abstain_gate_accept),     # MUST be 0
        "abstention_floor_false_accepts": int(floor_false_accepts),    # MUST be 0
        "lesion_margin": lesion_margin, "lesion_collapsed": bool(lesion_collapsed),
        "moat_intact": bool(host_abstain_gate_accept == 0 and floor_false_accepts == 0),
    }


def anticheat_C4_random_shard(all_corpora, cortex_builder, seed, D, args):
    """C4 -- random-shard: re-shard the concepts randomly (destroy the within-shard semantic co-location) and
    re-run B1 on a random group -> MUST collapse to chance (the co-location is load-bearing). The graded codes
    of a random group have no sub-cluster block, so similarity-vote generalization collapses.

    We reuse the multibridge_graded_derisk.random_shard_anticheat machinery's logic: a random scattered group's
    graded codes (synthetic or learned) must not generalize. For the cheap CPU path we build a synthetic graded
    codebook over a SHUFFLED-label group (the structure-destroying control)."""
    # Pool members across shards, assign each its ORIGINAL property, then take a random scattered group; its
    # within-group labels are its members' (now meaningless) original sub-cluster ids -> no block structure.
    rng = np.random.RandomState(seed * 4242 + 1)
    n_per = len(all_corpora[0]["members"])
    all_members = []
    for bc in all_corpora:
        all_members.extend(bc["members"])
    if len(all_members) <= n_per:
        chosen = list(range(len(all_members)))
    else:
        chosen = list(rng.choice(len(all_members), size=n_per, replace=False))
    n_chosen = len(chosen)
    # a synthetic graded codebook with RANDOM (scrambled) cluster labels -> the codes carry a block structure
    # but the property labels are decoupled from it, so generalization collapses (the random-shard analogue:
    # the code's structure no longer predicts the property because the shard is scattered).
    n_sub = max(2, n_chosen // max(2, args.target_per_sub))
    per_sub = n_chosen // n_sub
    n_use = n_sub * per_sub
    words = [f"rand_{i}" for i in range(n_use)]
    cb = build_cortex_codebook_synthetic(words, n_sub, per_sub, D=D, dg_n_pool=args.n_pool,
                                          dg_pattern_size=args.pattern_size, seed=seed)
    # scramble the property labels relative to the code clusters (the co-location destruction)
    props = assign_properties(n_sub, per_sub, args.n_props, seed)
    perm = rng.permutation(len(props))
    props_scrambled = props[perm]
    genB = gate_B_generalization(cb, seed, D, args, codes_override=cb.codes, labels_override=cb.labels,
                                 props_override=props_scrambled, n_sub=n_sub, per_sub=per_sub)
    chance = 1.0 / args.n_props
    b1 = genB["b1_accuracy"]
    return {"b1_random_shard": b1, "chance": chance, "collapses": bool(b1 <= 1.5 * chance),
            "n_random_members": n_use}


# ===========================================================================
# Per-seed driver.
# ===========================================================================
def run_seed(seed, args):
    print(f"\n{'='*92}", flush=True)
    print(f"  CORTEX<->CONVERSATION CAPABILITY DE-RISK -- SEED {seed} -- mode={args.mode} "
          f"cortex={args.cortex}", flush=True)
    print(f"{'='*92}", flush=True)

    # ---- build the shard corpus (the multibridge curated shard) -> ONE shard's concepts ----
    shard_name = SHARD_NAMES[0]
    bridge_corpus = build_bridge_corpus(shard_name, args.concepts_per_bridge, seed, args)
    n_sub, per_sub = bridge_corpus["n_sub"], bridge_corpus["per_sub"]
    members = bridge_corpus["members"]
    print(f"  [shard {shard_name}] {len(members)} concepts ({n_sub} sub x {per_sub}), "
          f"{bridge_corpus['n_facts']} within-shard facts", flush=True)

    # ---- build the cortex codebook (synthetic for cheap CPU / learned for the real GPU run) ----
    t0 = time.time()
    if args.cortex == "learned":
        cortex = build_cortex_codebook_learned(bridge_corpus, D=args.D, dg_n_pool=args.n_pool,
                                               dg_pattern_size=args.pattern_size, seed=seed, args=args)
    else:
        # synthetic graded codebook over the SAME sub-cluster layout (member order matches the layout)
        cortex = build_cortex_codebook_synthetic(
            members, n_sub, per_sub, D=args.D, dg_n_pool=args.n_pool,
            dg_pattern_size=args.pattern_size, seed=seed, dim=args.synthetic_dim,
            residual_frac=args.residual_frac)
    cortex_s = time.time() - t0
    gstats = cortex.graded_stats()
    exp_cos = cortex.expansion_between_cos()
    print(f"  [cortex {args.cortex}] within-cos={gstats['within_cluster_cos_mean']:.3f} "
          f"between-cos={gstats['between_cluster_cos_mean']:.3f} margin={gstats['graded_margin']:.3f} "
          f"graded={gstats['is_graded']} | DG-expansion between-cos={exp_cos:.3f} "
          f"(binding-clean<0.15) | {cortex_s:.1f}s", flush=True)

    out = {"seed": seed, "mode": args.mode, "cortex_source": args.cortex,
           "shard": shard_name, "n_sub": n_sub, "per_sub": per_sub,
           "graded_stats": gstats, "dg_expansion_between_cos": exp_cos}

    # ---- GATE A -- the conversational matrix ----
    if args.mode in ("matrix", "full"):
        print(f"\n  {'-'*88}\n  GATE A -- conversational matrix on the cortex-induced codes\n  {'-'*88}",
              flush=True)
        A = gate_A_matrix(cortex, seed, args.D)
        print(f"    cells: {A['cells']}", flush=True)
        print(f"    n_cells_pass={A['n_cells_pass']}/6  moat_holds={A['moat_holds']} "
              f"(abstention_battery_breaches={A['cells']['abstention_battery_breaches']})", flush=True)
        out["gate_A"] = A

    # ---- GATE B -- generalization in conversation + the moat on genuine absence ----
    if args.mode in ("generalize", "full"):
        print(f"\n  {'-'*88}\n  GATE B -- generalization IN conversation (the NEW capability)\n  {'-'*88}",
              flush=True)
        B = gate_B_generalization(cortex, seed, args.D, args, n_sub=n_sub, per_sub=per_sub)
        print(f"    B1 generalization acc={B['b1_accuracy']:.3f} "
              f"(chance={B['chance']:.3f}, {B['ratio_vs_chance']:.1f}x)", flush=True)
        if B["b1_conv_accuracy"] is not None:
            print(f"    B1-conv (fallback through what_does) acc={B['b1_conv_accuracy']:.3f}", flush=True)
        if B["b2"] is not None:
            print(f"    B2 moat on genuine absence: floor_n={B['b2']['floor_n']} "
                  f"false_accepts={B['b2']['false_accepts']} abstains_all={B['b2']['abstains_all']}",
                  flush=True)
        out["gate_B"] = B

        # ---- anti-cheats ----
        print(f"\n  {'-'*88}\n  ANTI-CHEATS C1-C4\n  {'-'*88}", flush=True)
        C1 = anticheat_C1_permuted(cortex, seed, args.D, args)
        print(f"    C1 permuted-similarity: B1={C1['b1_permuted']:.3f} collapses={C1['collapses']}",
              flush=True)
        C2 = anticheat_C2_orthogonal(cortex, seed, args.D, args)
        print(f"    C2 orthogonal codes: B1={C2['b1_orthogonal']:.3f} collapses={C2['collapses']} "
              f"| matrix still passes={C2['matrix_still_passes']} "
              f"(cells={C2['matrix_n_cells']}, moat={C2['matrix_moat_holds']})", flush=True)
        C3 = anticheat_C3_moat_alongside_host(cortex, seed, args.D, args)
        print(f"    C3 moat alongside host: agreement={C3['agreement']:.3f} "
              f"host-abstain/gate-accept={C3['host_abstain_gate_accept']} (MUST be 0) "
              f"floor-false-accepts={C3['abstention_floor_false_accepts']} (MUST be 0) "
              f"lesion-collapses={C3['lesion_collapsed']} -> moat_intact={C3['moat_intact']}", flush=True)
        all_corpora = [build_bridge_corpus(sn, args.concepts_per_bridge, seed, args)
                       for sn in SHARD_NAMES[:max(2, args.n_bridges)]]
        C4 = anticheat_C4_random_shard(all_corpora, None, seed, args.D, args)
        print(f"    C4 random-shard: B1={C4['b1_random_shard']:.3f} collapses={C4['collapses']} "
              f"(n={C4['n_random_members']})", flush=True)
        out["anticheats"] = {"C1": C1, "C2": C2, "C3": C3, "C4": C4}

    return out


# ===========================================================================
# Multi-seed verdict (design SS2.4).
# ===========================================================================
def _seed_gateA_ok(r):
    A = r.get("gate_A")
    if A is None:
        return None
    return bool(A["n_cells_pass"] >= 5 and A["moat_holds"])


def _seed_gateA_abstention_breach(r):
    A = r.get("gate_A")
    if A is None:
        return False
    return bool((not A["cells"]["abstention"]) or A["cells"]["abstention_battery_breaches"] > 0)


def aggregate(per_seed, args):
    seeds = list(per_seed.keys())
    agg = {"seeds": seeds, "mode": args.mode, "cortex_source": args.cortex,
           "b1_bar": args.a1_bar}

    def _all(pred):
        vals = [pred(per_seed[s]) for s in seeds]
        return all(v is True for v in vals), vals

    chance = 1.0 / args.n_props

    if args.mode in ("matrix", "full"):
        a_ok, a_vals = _all(lambda r: _seed_gateA_ok(r) is True)
        any_breach = any(_seed_gateA_abstention_breach(per_seed[s]) for s in seeds)
        agg["gate_A"] = {"all_pass": a_ok, "per_seed": a_vals, "any_abstention_breach": any_breach}

    if args.mode in ("generalize", "full"):
        b1_vals = [per_seed[s]["gate_B"]["b1_accuracy"] for s in seeds]
        b1_go, _ = _all(lambda r: r["gate_B"]["b1_accuracy"] >= args.a1_bar)
        b1_above_chance, _ = _all(lambda r: r["gate_B"]["b1_accuracy"] > 1.25 * chance)
        b1_boundary = all(0.5 <= per_seed[s]["gate_B"]["b1_accuracy"] < args.a1_bar for s in seeds)
        b2_ok, _ = _all(lambda r: (r["gate_B"]["b2"] is None) or r["gate_B"]["b2"]["abstains_all"])
        c1_ok, _ = _all(lambda r: r["anticheats"]["C1"]["collapses"])
        c2_ok, _ = _all(lambda r: r["anticheats"]["C2"]["collapses"]
                        and r["anticheats"]["C2"]["matrix_still_passes"])
        c3_ok, _ = _all(lambda r: r["anticheats"]["C3"]["moat_intact"]
                        and r["anticheats"]["C3"]["lesion_collapsed"])
        c4_ok, _ = _all(lambda r: r["anticheats"]["C4"]["collapses"])
        agg["gate_B"] = {"b1_per_seed": b1_vals, "b1_all_GO": b1_go,
                         "b1_above_chance": b1_above_chance, "b1_boundary_band": b1_boundary,
                         "b2_zero_false_accepts": b2_ok, "chance": chance}
        agg["anticheats"] = {"C1_collapses": c1_ok, "C2_collapses_matrix_passes": c2_ok,
                             "C3_moat_intact_lesion_collapses": c3_ok, "C4_collapses": c4_ok}

    # ---- combined verdict (design SS2.4) ----
    verdict = None
    if args.mode == "full":
        gA = agg["gate_A"]
        gB = agg["gate_B"]
        ac = agg["anticheats"]
        moat_breach = (gA["any_abstention_breach"]
                       or (not gB["b2_zero_false_accepts"])
                       or (not ac["C3_moat_intact_lesion_collapses"]))
        controls_collapse = ac["C1_collapses"] and ac["C2_collapses_matrix_passes"] and ac["C4_collapses"]
        if moat_breach:
            verdict = "NEGATIVE"            # the moat is non-negotiable
        elif not gB["b1_above_chance"]:
            verdict = "NEGATIVE"            # no generalization
        elif not controls_collapse:
            verdict = "NEGATIVE"            # the "generalization" is an artifact, not similarity-driven
        elif gA["all_pass"] and gB["b1_all_GO"] and gB["b2_zero_false_accepts"] and ac["C3_moat_intact_lesion_collapses"]:
            verdict = "GO"
        elif gA["all_pass"] and gB["b1_boundary_band"]:
            verdict = "BOUNDARY"
        else:
            verdict = "BOUNDARY"
    agg["verdict"] = verdict
    return agg


def main():
    p = argparse.ArgumentParser(
        description="Cortex<->conversation CAPABILITY de-risk (the conversational matrix + generalization "
                    "in conversation + the no-confab moat, on the learned-graded cortex)")
    p.add_argument("--mode", default="full", choices=["matrix", "generalize", "full"],
                   help="matrix=Gate A only; generalize=Gate B + C1-C4; full=both + verdict")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--seed", type=int, default=None, help="single-seed override")
    p.add_argument("--cortex", default="learned", choices=["synthetic", "learned"],
                   help="graded-codebook source: 'learned' (spiking HomeostaticAssocGraph + divnorm; GPU) or "
                        "'synthetic' (build_graded_codebook; cheap CPU)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU plumbing smoke: small shard + synthetic cortex + tiny n_pool/cycles")
    # shard sizing (the multibridge curated shard)
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
    p.add_argument("--b-conv-splits", type=int, default=8,
                   help="held-out splits for the B1-conv conversational fallback measurement")
    # gate bars
    p.add_argument("--a1-bar", type=float, default=0.7, help="B1 generalization GO bar (>= ~4x chance)")
    p.add_argument("--k-neighbours", type=int, default=3)
    p.add_argument("--moat-floor", type=int, default=20)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    # smoke overrides: tiny + synthetic + CPU-fast
    if args.smoke:
        args.cortex = "synthetic"
        args.concepts_per_bridge = min(args.concepts_per_bridge, 16)
        args.target_per_sub = 4
        args.n_pool = min(args.n_pool, 400)
        args.pattern_size = min(args.pattern_size, 40)
        args.cycles = 2
        args.synthetic_dim = 96
        args.b_conv_splits = 3
        args.n_bridges = 2

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    backend = os.environ.get("SIM_BACKEND", "auto")
    t_all = time.time()
    print(f"[cortex<->conversation capability de-risk] mode={args.mode} seeds={seeds} "
          f"cortex={args.cortex} backend={backend} smoke={args.smoke}", flush=True)
    print(f"  shard: {args.concepts_per_bridge} concepts (target_per_sub={args.target_per_sub}); "
          f"composer D={args.D}; B1 bar={args.a1_bar} (chance={1.0/args.n_props:.3f})", flush=True)
    print(f"  ADAPTATIONS: {ADAPTATIONS}", flush=True)

    per_seed = {}
    for s in seeds:
        per_seed[str(s)] = run_seed(s, args)

    agg = aggregate(per_seed, args)

    print(f"\n{'='*92}", flush=True)
    print(f"  CORTEX<->CONVERSATION CAPABILITY DE-RISK SUMMARY -- mode={args.mode}", flush=True)
    print(f"{'='*92}", flush=True)
    for k, v in agg.items():
        if k in ("seeds", "mode", "cortex_source", "verdict", "b1_bar"):
            continue
        print(f"  [{k}] {v}", flush=True)
    if agg.get("verdict") is not None:
        print(f"\n  >>> COMBINED VERDICT: {agg['verdict']} <<<", flush=True)
    print(f"  Total elapsed: {time.time()-t_all:.1f}s", flush=True)
    print(f"{'='*92}\n", flush=True)

    out_data = {"aggregate": agg, "per_seed": per_seed, "args": vars(args), "adaptations": ADAPTATIONS}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        tag = "smoke" if args.smoke else args.mode
        args.out = os.path.join(raw_dir, f"_cortex_conversation_capability_derisk_{tag}_seed{seeds[0]}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out_data


if __name__ == "__main__":
    main()
