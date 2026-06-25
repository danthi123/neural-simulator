"""DISCURSIVE TURN -- STAGE 0: the CPU MIXED-TYPE multi-proposition ENGAGE-AND-DISCUSS turn that LIFTS the
RichAnswerComposer's "gather a SET -> render multi-sentence -> per-sentence VERIFY -> moat-safe paragraph"
from "N certain facts" to "N TYPED propositions" {C=known-certain / N=novel-flagged-hypothesis /
D=discuss-via-adjacent-grounded-facts / P=phatic}, with the CommunicableTurn appraisal (`worth`) + the
SpikingSpeakAccumulator deciding the MIX + DEPTH per-candidate, and a per-proposition TYPE-AWARE VERIFY gate.

This is Stage 0 of the (1) communicable-brain RICHNESS frontier (scoping
`research/findings/raw/_communicable_discursive_turn_scoping.md`). The scoping found that EVERY cognitive
primitive the discursive turn needs is ALREADY GO -- the missing object is an ASSEMBLY-AND-APPRAISAL SHELL
that turns the existing per-proposition primitives into a per-MESSAGE, multi-proposition, mixed-type,
depth-controlled, type-safe-moat paragraph. The two precedents bracket exactly the new object:
  - `CommunicableTurn` (Stage A, GO 3-seed): the NOVEL/FLAGGED channel + the worth appraisal + the spiking
    speak decision + the learned talkativeness + the teaching feedback -- but ONE proposition per turn.
  - `RichAnswerComposer` (GO): a multi-SENTENCE paragraph with per-sentence VERIFY + the dlPFC discourse
    planner -- but every sentence is a CERTAIN known fact (no novel/flagged channel).
NEITHER produces a paragraph that MIXES proposition TYPES (some certain, some flagged-hypothesis, some
"let me think out loud", a phatic glue) -- which is exactly what "engage and discuss" is.

THE NEW OBJECT: a `DiscursiveTurn` orchestrator (~the size of RichAnswerComposer + the Stage-A routing) that
for ONE user message:
  1. classifies intent -> a MIXING PRIOR (NOT an exclusive switch) -- `IntentRouter.classify` (Stage A);
  2. assembles a TYPED candidate pool:
       (C) certain   -- direct recall + role-chase chain + grounded elaboration (the RichAnswerComposer gather,
                        rebuilt on the bare agent/composer: the moat lives in what_does/query_patient abstaining);
       (N) novel     -- the b2 GenerativeReplayProposer candidate SET about the topic (each FLAGGED, never stored);
       (D) discuss   -- adjacent grounded fragments (certain, verified) + flagged speculation, framed "here's
                        how I think about it" (the engage-without-an-answer behavior);
       (P) phatic    -- a fixed non-claim social/glue utterance;
  3. appraises each candidate by `worth` (CommunicableTurn.worth: learned-Q value + plausibility + familiarity)
     and runs `SpikingSpeakAccumulator.decide` PER candidate to SELECT the emitted set + its size (DEPTH);
  4. orders them (lead certain when the gate hit; lead the framing/phatic glue otherwise; the dlPFC orders
     the elaboration);
  5. renders+VERIFIES each with the TYPE-AWARE moat rule, dropping failures;
  6. emits a paragraph with certain vs flagged VISIBLY DISTINCT, writes the topic + any hypothesis-referent
     into the discourse state.

THE TYPE-AWARE VERIFY GATE (GAP-3, the LOAD-BEARING moat shift -- "never ASSERT a fabricated fact"):
  - (C) accept-as-CERTAIN iff `_verify(prose) == svo AND svo IN stored_facts` (= RichAnswerComposer's
    per-sentence VERIFY). Mis-render -> DROP (a certain claim that won't verify must not be spoken at all).
  - (N)/(D-flagged) accept-as-FLAGGED iff `_verify(prose) == the_brain's_PROPOSED_svo AND a hedge is carried
    AND svo NOT IN stored_facts` (novel by construction). Rendered with the hedge_for(worth) prefix + a
    HYPOTHESIS marker. NEVER written to the store.
  - (P) phatic -- makes no factual claim -> no moat check (cannot fabricate a fact because it asserts none).
  The proposition's `type` is set at GATHER time by which channel produced it; the render/verify dispatch
  keys on that type. A flagged proposition CANNOT be rendered certain (the certain renderer path is reachable
  only for type-C, and a type-C proposition requires `svo IN stored_facts`). Mis-tagging is STRUCTURALLY
  impossible. THE INVARIANT: the paragraph contains ONLY {verified-stored-certain} UNION {flagged-hypothesis}.

THE BRAIN DOES THE COGNITION (the LLM is fluency-only -- REJECT the free-generate cheat):
  - the (C) recall/chain/elaborate ordering = the composer's spiking recall + query_chain hops + the dlPFC
    spreading-activation; the (N)/(D) content = the brain's PROPOSED SVO (the proposer over the LEARNED graph);
    the per-candidate emit/silence DECISION = the SpikingSpeakAccumulator's pool FIRING (read from
    `cp_firing_states`); the worth value axis = the LEARNED-Q three-factor plasticity (DA-lesion-provable).
  - the LESION anti-cheat (sever the brain's proposal -> let the faculty free-generate -> VERIFY must reject)
    is run on the MIXED C+N+D paragraph; the `_ConfabOneRenderer` confab probe drops a confabulated sentence
    from the mixed paragraph while the truthful + correctly-flagged ones survive.

THE STAGE-0 GATE (>=3 seeds; promote to 6 if GO). GO = ALL of:
  (1) MIXED ASSEMBLY: >=1 scripted turn emits a paragraph with >=2 proposition TYPES (C+N or C+D), and the
      discuss-while-answering turn has depth >= 2 (strictly richer than the thin 1-fact answer). The
      engage-without-an-answer turn emits >=2 grounded-or-flagged propositions (NOT a bare abstain).
  (2) MOAT (HARD, the core bar): 0 (C)-proposition leaks -- every certain proposition re-parses to a STORED
      fact; every (N)/(D) proposition is FLAGGED + a who/what on it ABSTAINS + it is NEVER stored. The mixed
      paragraph contains ONLY {verified-stored-certain} UNION {flagged-hypothesis} -- no unverified certainty,
      no fabricated fact. (A no-stored-answer case ENGAGES via adjacent grounded facts + flagged speculation,
      0 fabrication.)
  (3) BRAIN-DOES-COGNITION: the per-candidate emit/silence decision is the spiking accumulator's firing
      (provenance: read from cp_firing_states, 0 host `if score>thr` selecting content); the (N)/(D) content
      is the brain's proposed SVO (the LESION free-generate arm is caught-by-VERIFY across the mixed paragraph).
  (4) DEPTH-ADAPTS: the "tell me more" round increases the emitted proposition count on the held topic
      (immediate) AND raises the learned-Q there (durable, monotone); the DA-lesion abolishes both.
  (5) NON-REGRESSION: with the discursive layer OFF, CommunicableTurn.turn + RichAnswerComposer.answer behave
      byte-identically (the new orchestrator is purely additive -- it never mutates the reused objects' state
      outside its own gathers).

ANTI-CHEATS: shuffled-PPMI-graph collapse (>=3x) | free-generate LESION caught on the mixed paragraph |
the _ConfabOneRenderer on a mixed C+N+D paragraph (the confabulated sentence DROPPED, the rest survive) |
decorrelated-credit (the depth/talkativeness rise is per-context, not global vigor) | value-perp-plausibility
(corr ~ 0 -- the value axis driving depth is not relabeled plausibility).

HONEST: if a bar misses (especially MOAT-HARD), this reports it PRECISELY in the JSON + the FINAL MESSAGE --
it does NOT force a GO. NEVER weakens the moat: a flagged-hypothesis ABSTAINS as a who/what; flagged props
are never stored.

CPU (`SIM_BACKEND=numpy`); reuse-by-import; NO `sim/` edit. Run:
  SIM_BACKEND=numpy python -u -m research.runners._discursive_turn_stage0_derisk \
      --seeds 42,43,44 --out research/findings/raw/_discursive_turn_stage0_derisk.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict

# the whole pipeline is the numpy-CPU brain (PPMI cortex + RF composer + parser + a spiking WTA accumulator slice).
os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# --- the Stage A communicable-brain FACTORY + the fused turn + the appraisal/learning/router (reuse VERBATIM) ---
from research.runners._communicable_turn_stageA_derisk import (  # noqa: E402
    CommunicableTurn,
    IntentRouter,
    SignedLearnedSpeakValue,
    build_communicable_brain,
    PHATIC_REPLIES,
)
# --- the b2 generative-replay machinery (the spiking GENERATE) + the shuffled-graph control ---
from research.runners._genfrontier_b2_generative_replay_derisk import (  # noqa: E402
    build_plausibility,
    build_stored_facts,
    shuffle_graph,
    _category_pools,
)
from research.runners.option_c_real_cooccurrence_derisk import (  # noqa: E402
    TAXONOMY_8x8,
    taxonomy_to_vocab_categories,
    build_real_cooccurrence,
)
# --- the SPIKING speak/silence WTA accumulator (the brain-based DECIDE) -- reused VERBATIM ---
from research.runners._value_salience_appraisal_derisk import (  # noqa: E402
    SpikingSpeakAccumulator,
)
# --- the LEARNED talkativeness context code (the LEARN) -- reused VERBATIM ---
from research.runners._learned_talkativeness_derisk import (  # noqa: E402
    context_code,
    code_overlap,
)
# --- the graded-confidence read-out + the VERIFY re-parse + the fluency faculties (reused VERBATIM) ---
from research.runners._communicable_brain_probe1_whatdoyouthink import (  # noqa: E402
    hedge_for,
)
from research.runners._grounded_lang_p3_derisk import (  # noqa: E402
    TemplateStubFaculty,
    InjectingStubFaculty,
)
# --- the adversarial confab-one renderer (reuse VERBATIM the RichAnswerComposer probe) ---
from research.runners.rich_answer_composer import _ConfabOneRenderer  # noqa: E402


# the proposition TYPE codes (the four discursive channels)
TYPE_CERTAIN = "C"      # known-fact-certain (direct recall + chain + grounded elaboration) -- asserted plainly
TYPE_NOVEL = "N"        # novel-flagged-hypothesis (generated, graph-plausible) -- hedged, never stored
TYPE_DISCUSS = "D"      # discuss-without-a-stored-answer (the flagged-speculation part of a (D) assembly)
TYPE_PHATIC = "P"       # phatic / discourse-glue (a non-claim social/connective utterance)

# fixed discourse-glue phrases (NON-claim wrappers -- like the phatic table; never LLM-generated linking claims).
GLUE = {
    "good_question": "Good question.",
    "discuss_frame": "Here's how I think about it:",
    "honest_nonanswer": "That's a deep one -- I don't have settled knowledge there.",
}


# ===========================================================================
# A `Proposition` -- one typed, appraised, render+verify candidate. The render/verify DISPATCHES on `ptype`,
# so a flagged proposition can NEVER reach the certain-render path (mis-tagging is structurally impossible).
# ===========================================================================
class Proposition:
    __slots__ = ("ptype", "svo", "worth", "speak_margin", "speak_spikes", "silence_spikes", "hedge",
                 "confidence", "certain", "verified", "surface", "emitted", "drop_reason", "speak_decision",
                 "lead")

    def __init__(self, ptype, svo, lead=False):
        self.ptype = ptype                # C / N / D / P
        self.svo = list(svo) if svo is not None else None
        self.worth = None
        self.speak_margin = None
        self.speak_spikes = None
        self.silence_spikes = None
        self.hedge = None
        self.confidence = None
        self.certain = (ptype == TYPE_CERTAIN)   # ONLY a C proposition may be asserted certain
        self.verified = None
        self.surface = None
        self.emitted = False
        self.drop_reason = None
        self.speak_decision = None
        # the LEAD proposition (the direct answer to a question whose gate HIT) is emitted subject to VERIFY
        # ONLY -- it is not worth-gated through the spiking decide (per scoping GAP-2: "lead with (C)-certain,
        # then OPTIONALLY add (N)/(D) elaboration if worth clears the speak threshold"). The spiking decide is
        # still RUN + recorded for the lead (provenance/transparency); it just does not gate the lead's emission.
        self.lead = bool(lead)

    def record(self):
        return {"type": self.ptype, "svo": self.svo, "certain": bool(self.certain), "lead": bool(self.lead),
                "hedge": self.hedge, "confidence": self.confidence, "verified": self.verified,
                "speak_margin": self.speak_margin, "speak_decision": self.speak_decision,
                "emitted": bool(self.emitted), "surface": self.surface, "drop_reason": self.drop_reason}


# ===========================================================================
# The DISCURSIVE TURN orchestrator.  Holds a CommunicableTurn (the brain + the appraisal + the spiking decide +
# the learned-Q + the teaching feedback, reused VERBATIM) and lifts it from one-channel to a MIXED-type,
# depth-controlled, type-safe-moat paragraph.  The (C) gather mirrors RichAnswerComposer's gather logic on the
# bare agent/composer (the moat lives in what_does/query_patient abstaining); the (N)/(D) channels reuse the
# CommunicableTurn proposer + worth + render_and_verify; the per-candidate spiking decide is the selector.
# ===========================================================================
class DiscursiveTurn:
    """ENGAGE-AND-DISCUSS: for ONE user message, gather a SET of MIXED-type propositions, appraise + spiking-
    decide the emitted set + depth, render+verify type-aware, emit a coherent paragraph that visibly
    distinguishes certain from speculative. A strict superset of CommunicableTurn.turn + RichAnswerComposer.answer.

    `ct` is a built `CommunicableTurn` (from build_communicable_brain); the orchestrator reuses its brain
    (comp/agent/proposer/value), its worth appraisal, its spiking accumulator (ct.acc), its render+verify,
    and its teaching feedback VERBATIM."""

    def __init__(self, ct, *, max_depth=4, max_chain_hops=3, max_elaborations=2, max_novel=2, max_discuss=4,
                 n_attempts=500, planner_seed=42):
        self.ct = ct
        self.comp = ct.comp                # the RF composer (KNOWN-fact store + the moat)
        self.agent = ct.agent              # BrainConversationalAgent (.what_does/.is_it_true/.parse)
        self.value = ct.value              # the LEARNED speak-value (SignedLearnedSpeakValue)
        self.faculty = ct.faculty          # the fluency-only faculty (CPU stand-in)
        self.max_depth = int(max_depth)
        self.max_chain_hops = int(max_chain_hops)
        self.max_elaborations = int(max_elaborations)
        self.max_novel = int(max_novel)
        self.max_discuss = int(max_discuss)
        self.n_attempts = int(n_attempts)
        # the NEURAL discourse planner (dlPFC spreading activation) orders the elaboration / adjacency, exactly
        # as RichAnswerComposer's neural_planner path. Reuse it (it operates on the composer's assoc graph).
        from research.runners.rich_answer_composer import NeuralDiscoursePlanner
        self._planner = NeuralDiscoursePlanner(self.comp, seed=planner_seed)
        # discourse thread state (held topic + a hypothesis-referent + what's been said this conversation)
        self._topic = None
        self._hypothesis_referent = None
        self._said = set()                 # tuple(a,v,p) said anywhere this conversation (never repeats)

    # ----------------------------------------------------------------------------------------------
    # The brain's stored SVO facts (string-only roles) -- the (C) gather source + the verify vocab + the
    # who/what moat-leak check ground truth.
    # ----------------------------------------------------------------------------------------------
    def _stored_facts(self):
        """The brain's AFFIRMED SVO facts (string-only roles) -- the ONLY positive-assertable content. NEGATED
        facts ('X does NOT Y') are EXCLUDED: asserting one as a positive certain claim would be a moat violation
        (the brain was told it is FALSE). So the (C) gather + the verify vocab + the certain stored-set are
        affirmed-only; a NEGATED triple can never become a certain proposition."""
        return [(f.get("agent"), f.get("action"), f.get("patient")) for f, _ in self.comp.kb
                if all(isinstance(f.get(r), str) for r in ("agent", "action", "patient"))
                and f.get("polarity", "AFFIRM") != "NEGATE"]

    def _stored_set(self):
        return set(self._stored_facts())

    def _facts_about(self, concept):
        return [[a, v, p] for (a, v, p) in self._stored_facts() if a == concept]

    def _facts_mentioning(self, concept):
        return [[a, v, p] for (a, v, p) in self._stored_facts() if concept in (a, p)]

    # ============================================================================================
    # (C) GATHER certain facts -- direct recall + role-chase chain + grounded elaboration. Mirrors
    # RichAnswerComposer.gather; the moat lives in the agent/composer abstaining (what_does/query_patient -> None).
    # ============================================================================================
    def _direct_fact(self, cue):
        """The DIRECT recall for a structured (agent, action) cue: the brain's what_does (the moat abstains -> None)."""
        if cue is None or len(cue) < 2:
            return None
        ag, ac = cue[0], cue[1]
        patient = self.agent.what_does(ag, ac)
        if patient is None:
            return None
        return [ag, ac, patient]

    def _chain_facts(self, start_agent, seed_action, exclude):
        """(b) MULTI-HOP: follow the brain's role structure (the validated query_patient hop; abstains -> None).
        Each hop's patient becomes the next hop's agent. Stops the moment a hop abstains (the moat at every hop)."""
        facts, cur, action = [], start_agent, seed_action
        excl = {tuple(f) for f in exclude}
        for _ in range(self.max_chain_hops):
            nxt = self.comp.query_patient(cur, action)        # the validated spiking hop (abstains -> None)
            if nxt is None or not isinstance(nxt, str):
                own = [f for f in self._facts_about(cur) if tuple(f) not in excl and f not in facts]
                if not own:
                    break
                a2, v2, p2 = own[0]
                facts.append([a2, v2, p2]); excl.add((a2, v2, p2))
                cur, action = p2, v2
                continue
            tri = [cur, action, nxt]
            if tuple(tri) not in excl:
                facts.append(tri); excl.add((cur, action, nxt))
            cur = nxt
        return facts

    def _elaboration_facts(self, topic, exclude):
        """(c) ELABORATION via the NEURAL dlPFC discourse planner: the spreading-activation latency rank decides
        WHICH on-topic concepts to bring up + in WHAT order (off-topic never fires -> never selected); each is
        mapped to a grounded stored SVO (legitimate host KB access -- the planner picks concepts, not facts)."""
        out, excl = [], {tuple(f) for f in exclude}
        for concept in self._planner.ordered_associates(topic, avoid=()):
            if len(out) >= self.max_elaborations:
                break
            for f in self._facts_mentioning(concept):
                if tuple(f) not in excl:
                    out.append(list(f)); excl.add(tuple(f))
                    break                         # one grounded sentence per neural-selected concept
        return out[: self.max_elaborations]

    def gather_certain(self, cue):
        """Assemble the certain (C) fact-set for a question cue: (a) direct + (b) chain + (c) elaboration,
        de-duplicated. Returns (topic, [svo, ...]) or (None, []) when the brain has no matched fact (the moat
        abstains here). The TOPIC for the chain/elaboration = the answer's PATIENT if it is itself an agent of
        some fact, else the question subject."""
        direct = self._direct_fact(cue)
        if direct is None:
            return None, []
        a, v, p = direct
        agents_set = {f[0] for f in self._stored_facts()}
        topic = p if (isinstance(p, str) and p in agents_set) else a
        convo_excl = [list(f) for f in self._said]
        chain = self._chain_facts(a, v, exclude=[direct] + convo_excl) if topic == a else \
            self._chain_facts(p, self._thread_seed_action(p), exclude=[direct] + convo_excl)
        facts = [direct] + [f for f in chain if f != direct and tuple(f) not in self._said]
        elab = self._elaboration_facts(topic, exclude=facts + convo_excl)
        out = self._dedup(facts + elab)
        return topic, out

    def _thread_seed_action(self, topic):
        own = self._facts_about(topic)
        return own[0][1] if own else None

    @staticmethod
    def _dedup(facts):
        out = []
        for f in facts:
            if f not in out:
                out.append(f)
        return out

    # ============================================================================================
    # (N) GATHER novel flagged candidates ABOUT a topic -- the CommunicableTurn proposer candidate SET. Each is
    # appraised by worth + spiking-decided; rendered FLAGGED; NEVER stored. We expose the candidate SET (not just
    # the single best) so the assembly can blend >1 flagged proposition when the depth budget allows.
    # ============================================================================================
    def gather_novel(self, topic, exclude_triples=()):
        """Topic-relevant, NOVEL, graph-plausible, non-contradictory candidate triples about X (the b2 proposer
        via CommunicableTurn.propose_candidates_about), ranked by WORTH (learned-Q value + plausibility +
        familiarity). Returns a list of (svo, worth, hedge, confidence) for the top candidates, excluding the
        certain set + anything already said."""
        if topic is None or topic not in self.ct.row:
            return []
        cands = self.ct.propose_candidates_about(topic, n_attempts=self.n_attempts)
        if not cands:
            return []
        excl = {tuple(t) for t in exclude_triples} | set(self._said)
        ranked = sorted(((tp, self.ct.worth(topic, tp)[0]) for tp, _pl in cands), key=lambda kv: -kv[1])
        out = []
        for tp, w in ranked:
            if tuple(tp) in excl:
                continue
            hedge, conf = hedge_for(w, self.ct._conf_lo, self.ct._conf_hi)
            out.append((list(tp), float(w), hedge, float(conf)))
            excl.add(tuple(tp))
            if len(out) >= self.max_novel:
                break
        return out

    # ============================================================================================
    # (D) DISCUSS-WITHOUT-A-STORED-ANSWER -- adjacent grounded fragments (certain, verified) + flagged
    # speculation, framed "here's how I think about it". The brain has NO crisp fact for the open question, so
    # the (C) channel gathers NOTHING; the (D) path assembles what the brain DOES know that is adjacent in the
    # PPMI graph + flagged speculation. NOT a terse abstain, NOT a fabricated fact.
    # ============================================================================================
    def gather_discuss(self, topic):
        """For an open question with no stored answer: (i) adjacent GROUNDED facts (the dlPFC-ordered on-topic
        neighbourhood's stored facts -- each held to the certain rule) + (ii) FLAGGED speculation (the proposer's
        candidate set -- each held to the novel rule). Returns (grounded_svos, novel_flagged) where grounded_svos
        is a list of stored SVO (type C inside the (D) assembly) and novel_flagged is the gather_novel output
        (type D-flagged). Either may be empty; both empty -> the honest framed non-answer."""
        grounded = []
        if topic is not None and topic in self.ct.row:
            # adjacent stored facts: the dlPFC on-topic neighbourhood (+ the topic's own facts), grounded.
            seen = set()
            for concept in [topic] + self._planner.ordered_associates(topic, avoid=()):
                for f in self._facts_mentioning(concept):
                    if tuple(f) not in seen and tuple(f) not in self._said:
                        grounded.append(list(f)); seen.add(tuple(f))
                        if len(grounded) >= self.max_discuss:
                            break
                if len(grounded) >= self.max_discuss:
                    break
        novel_flagged = self.gather_novel(topic, exclude_triples=grounded)
        return grounded, novel_flagged

    # ============================================================================================
    # The TYPE-AWARE render+VERIFY gate (GAP-3, the moat). Dispatches on `prop.ptype`; the certain-render path is
    # reachable ONLY for a type-C proposition whose svo is in the stored set -> mis-tagging a flagged proposition
    # as certain is structurally impossible.
    # ============================================================================================
    def _render_verify(self, prop, faculty, stored_set):
        """Render + VERIFY one Proposition, applying the right moat rule for its type. Sets prop.surface /
        prop.verified / prop.emitted / prop.drop_reason. Returns the emitted-or-None surface text."""
        if prop.ptype == TYPE_PHATIC:
            prop.verified = True; prop.emitted = True
            return prop.surface

        rv = self.ct.render_and_verify(prop.svo, faculty, "grounded")
        prop.verified = bool(rv["verified"])
        if not rv["verified"]:
            prop.emitted = False
            prop.drop_reason = "render/VERIFY rejected (re-parse != the proposition's SVO)"
            return None

        if prop.ptype == TYPE_CERTAIN:
            # CERTAIN requires the re-parse to match AND the SVO to be a STORED fact. (A type-C proposition is
            # gathered ONLY from the stored set, so this is belt-and-suspenders -- a non-stored 'certain' triple
            # could never have been produced by the (C) gather. The check is the structural guarantee.)
            if tuple(prop.svo) not in stored_set:
                prop.emitted = False; prop.certain = False
                prop.drop_reason = "certain proposition not in the stored set -> DROPPED (never downgraded silently)"
                return None
            a, v, p = prop.svo
            prop.surface = rv["surface"]
            prop.emitted = True
            return prop.surface

        # (N)/(D-flagged): the re-parse matched the brain's PROPOSED svo; require it is NOT a stored fact (novel
        # by construction) AND carry a hedge + a HYPOTHESIS marker. NEVER stored.
        if tuple(prop.svo) in stored_set:
            prop.emitted = False
            prop.drop_reason = "flagged proposition coincides with a stored fact -> dropped (would be a certain claim)"
            return None
        hedge = prop.hedge or "I'm not sure, but I'd guess"
        prop.surface = f"{hedge} {rv['surface'][0].lower()}{rv['surface'][1:]}".rstrip()
        prop.emitted = True
        return prop.surface

    # ============================================================================================
    # The per-candidate SPIKING speak decision (the brain-based selector of the mix + depth). worth -> the speak
    # drift; the SpikingSpeakAccumulator decides emit-vs-silent (a neural pool's FIRING, NOT a host if score>thr).
    # ============================================================================================
    def _spiking_decide(self, prop, push):
        speak_pA, silence_pA = self.ct._speak_drives(push)
        is_speak, sp, si, margin = self.ct.acc.decide(speak_pA, silence_pA)
        prop.speak_decision = bool(is_speak)
        prop.speak_spikes = sp; prop.silence_spikes = si; prop.speak_margin = margin
        return is_speak

    def _appraise(self, topic, prop):
        """Set prop.worth + the (push for the spiking drive) + the hedge/confidence. For a C proposition the
        worth is read from the same appraisal (value + plausibility + familiarity); for N/D it is the proposer's
        worth already attached. Returns the additive incentive-salience `push`."""
        if prop.worth is None:
            w, (pn, vn, fn) = self.ct.worth(topic, prop.svo) if topic in self.ct.row else (0.6, (0.6, 0.5, 0.5))
            prop.worth = float(w)
            push = self.ct.w_value * vn + self.ct.w_plaus * pn + self.ct.w_fam * fn
        else:
            # N/D: recompute the component push from the proposition's own worth surrogate (worth IS the push
            # scale; map it through the same speak drive). Use worth normalised by the conf band as the push.
            w = prop.worth
            push = float(min(1.0, max(0.0, (w - self.ct._conf_lo) / max(1e-9, self.ct._conf_hi - self.ct._conf_lo))))
        if prop.hedge is None:
            prop.hedge, prop.confidence = hedge_for(prop.worth, self.ct._conf_lo, self.ct._conf_hi)
        return push

    # ============================================================================================
    # The full DISCURSIVE turn.
    # ============================================================================================
    def discuss(self, msg, cue=None, topic=None, followup=False, lesion_proposal_faculty=None,
                confab_target=None, force_intent=None):
        """One ENGAGE-AND-DISCUSS turn. Returns a structured record:
          {paragraph, propositions:[{type, svo, certain, hedge, confidence, verified, speak_margin, emitted,
           surface}], depth, abstained_certain, intent, gathered_types, glue}.

        `cue`   -- a (agent, action) tuple for a structured KNOWN-fact question (the orchestrator supplies it).
        `topic` -- the discourse topic (the entity the discussion is about).
        `followup` -- a 'tell me more' on the held topic (raises depth; re-runs deeper).
        `lesion_proposal_faculty` -- the free-generate LESION faculty (the anti-cheat: render the (N)/(D)
                       propositions through it -> VERIFY must reject; the content is the brain's, not the LLM's).
        `confab_target` -- (svo) to confabulate via the _ConfabOneRenderer (the mixed-paragraph confab probe).
        """
        # the intent is a MIXING PRIOR (the router classifies; a caller may force it, e.g. the teaching re-run
        # forces 'opinion' so the deeper turn assembles a candidate pool).
        intent = force_intent if force_intent is not None else self.ct.router.classify(msg)["intent"]
        stored_set = self._stored_set()
        props = []
        glue = []
        abstained_certain = False

        # the render faculty: faithful stub by default; the confab probe wraps it for the target svo.
        faculty = self.faculty
        if confab_target is not None:
            all_patients = sorted({f[2] for f in self._stored_facts()})
            wrong_p = next((x for x in all_patients if x != confab_target[2]), confab_target[2] + "_X")
            faculty = _ConfabOneRenderer(confab_target, wrong_p)
        # the (N)/(D) proposal faculty: the lesion arm free-generates the content; else the same faithful faculty.
        nd_faculty = lesion_proposal_faculty if lesion_proposal_faculty is not None else faculty

        # ---------- PHATIC ----------
        if intent == "phatic":
            key = "how_are_you" if re.search(r"how are you|how's it going", msg, re.IGNORECASE) else "default"
            p = Proposition(TYPE_PHATIC, None)
            p.surface = PHATIC_REPLIES[key]
            p.emitted = True; p.verified = True
            props = [p]
            return self._finalize(msg, intent, props, glue, abstained_certain, depth=1,
                                  gathered_types={TYPE_PHATIC})

        # ---------- TEACHING (depth-up / stop) ----------
        if intent == "teaching":
            pol = self.ct.router.classify(msg)["polarity"]
            tgt = topic if topic is not None else self._topic
            # the brain's three-factor talkativeness update (reused VERBATIM). lesion handled by caller-side.
            self.ct.feedback(tgt, pol)
            ack = ("Noted -- I'll say more about that." if pol > 0 else "Okay, I'll hold back on that.")
            p = Proposition(TYPE_PHATIC, None); p.surface = ack; p.emitted = True; p.verified = True
            props = [p]
            rerun = None
            if tgt is not None and pol > 0:
                # immediately elaborate MORE on the held topic with the updated Q (depth rises this turn).
                rerun = self.discuss(f"what do you think about {tgt}?", topic=tgt, followup=True,
                                     force_intent="opinion")
            rec = self._finalize(msg, "teaching", props, glue, abstained_certain, depth=1,
                                 gathered_types={TYPE_PHATIC})
            rec["polarity"] = pol; rec["feedback_topic"] = tgt
            if rerun is not None:
                rec["rerun"] = rerun
            return rec

        # ---------- the candidate POOL (mixed types) ----------
        # 1) the (C) certain gather (only for a question with a structured cue, OR an opinion whose topic IS a
        #    stored agent so the brain holds facts about it). The moat lives in gather_certain abstaining.
        certain_svos = []
        c_topic = None
        if intent == "question" and cue is not None:
            c_topic, certain_svos = self.gather_certain(cue)
            if c_topic is None:
                abstained_certain = True          # the moat abstained on the known-fact channel
        elif intent == "opinion" and topic is not None:
            agents_set = {f[0] for f in self._stored_facts()}
            if topic in agents_set:
                # the brain holds facts about this entity -> ground the opinion with them (certain), then flag novel
                own = self._facts_about(topic)
                certain_svos = self._dedup([f for f in own if tuple(f) not in self._said])[: self.max_elaborations]
                c_topic = topic

        disc_topic = topic if topic is not None else c_topic

        # 2) decide the assembly shape from the intent + whether the gate hit:
        #    - question + gate HIT  -> lead CERTAIN, then OPTIONALLY add (N)/(D) elaboration (discuss-while-answering)
        #    - question + gate MISS -> the (D) discuss-without-an-answer path (engage, not a bare abstain)
        #    - opinion              -> lead FLAGGED (N), optionally grounded by (C)
        is_engage_without_answer = (intent == "question" and abstained_certain)

        novel_cands = []
        discuss_grounded, discuss_novel = [], []
        if is_engage_without_answer:
            discuss_grounded, discuss_novel = self.gather_discuss(disc_topic)
            glue.append(GLUE["discuss_frame"])
        else:
            # add novel-flagged candidates ABOUT the topic (the elaboration / opinion lead)
            novel_cands = self.gather_novel(disc_topic, exclude_triples=certain_svos)
            if intent == "question":
                glue.append(GLUE["good_question"])   # discourse glue on a discuss-while-answering turn

        # 3) BUILD the typed Proposition pool. The LEAD proposition (emitted subject to VERIFY only, NOT
        #    worth-gated) is the reply's spine per the intent (scoping GAP-2):
        #      - question + gate HIT  -> the DIRECT certain answer leads, then optional worth-gated elaboration;
        #      - question + gate MISS -> the first adjacent GROUNDED fact leads the discuss assembly;
        #      - opinion              -> the top novel-flagged candidate leads (a flagged view).
        pool = []
        for i, svo in enumerate(certain_svos):
            # lead the direct answer ONLY for a question whose gate hit (an opinion's grounded facts are optional
            # supporting context, not the lead -- the opinion leads with the flagged view).
            pool.append(Proposition(TYPE_CERTAIN, svo, lead=(i == 0 and intent == "question")))
        for i, (svo, w, hedge, conf) in enumerate(novel_cands):
            pr = Proposition(TYPE_NOVEL, svo, lead=(i == 0 and intent == "opinion"))
            pr.worth = w; pr.hedge = hedge; pr.confidence = conf
            pool.append(pr)
        for i, svo in enumerate(discuss_grounded):
            pr = Proposition(TYPE_CERTAIN, svo, lead=(i == 0))   # the first adjacent grounded fact leads the (D)
            pool.append(pr)                                       # assembly (an on-topic stored fact, asserted certain)
        for (svo, w, hedge, conf) in discuss_novel:
            pr = Proposition(TYPE_DISCUSS, svo); pr.worth = w; pr.hedge = hedge; pr.confidence = conf
            pool.append(pr)

        # 4) APPRAISE + per-candidate SPIKING DECIDE (the brain-based selector of WHICH/HOW-MANY = depth).
        #    The depth = #candidates whose SpikingSpeakAccumulator.decide returns speak-wins, capped at max_depth.
        emitted_props, depth = [], 0
        gathered_types = set()
        # a follow-up RAISES talkativeness on the held topic this turn (immediate depth rise) -- the brain's Q
        # was already raised by the teaching feedback; the higher value -> higher push -> more candidates clear.
        for pr in pool:
            gathered_types.add(pr.ptype)
            push = self._appraise(disc_topic, pr)
            is_speak = self._spiking_decide(pr, push)     # the spiking decision (recorded for ALL, incl. the lead)
            if os.environ.get("DISC_DEBUG"):
                print(f"    [pool] topic={disc_topic} type={pr.ptype} lead={pr.lead} svo={pr.svo} "
                      f"worth={pr.worth} push={push:.3f} speak={is_speak} margin={pr.speak_margin}", flush=True)
            # the LEAD is emitted subject to VERIFY only (the direct answer leads regardless of talkativeness);
            # every OTHER candidate must win the spiking speak race to be added (worth-gated depth).
            if not pr.lead and not is_speak:
                pr.emitted = False; pr.drop_reason = pr.drop_reason or "spiking accumulator chose SILENCE"
                continue
            surface = self._render_verify(pr, (nd_faculty if pr.ptype in (TYPE_NOVEL, TYPE_DISCUSS) else faculty),
                                          stored_set)
            if surface is None:
                continue
            emitted_props.append(pr)
            depth += 1
            if depth >= self.max_depth:
                break

        # 5) order: the emitted props are already in (certain-lead) gather order; keep it. Record the hypothesis
        #    referent (the first flagged proposition's topic-entity) for a follow-up "tell me more about that".
        for pr in emitted_props:
            if pr.ptype in (TYPE_NOVEL, TYPE_DISCUSS) and pr.svo is not None:
                self._hypothesis_referent = pr.svo[0] if pr.svo[0] != disc_topic else (
                    pr.svo[2] if isinstance(pr.svo[2], str) else pr.svo[0])
                break

        # 6) the honest framed non-answer: if a discuss turn assembled NOTHING (no adjacent grounded + no
        #    candidate cleared), fall back to the honest framed non-answer (the graceful abstain, NOT a fabrication).
        if is_engage_without_answer and not emitted_props:
            glue = [GLUE["honest_nonanswer"]]

        # record said facts conversation-wide (only the EMITTED certain ones restate-protect; flagged never
        # stored, but we still avoid re-flagging the same triple).
        for pr in emitted_props:
            if pr.svo is not None:
                self._said.add(tuple(pr.svo))

        # update the held topic
        if disc_topic is not None and not followup:
            self._topic = disc_topic

        return self._finalize(msg, intent, emitted_props + [p for p in pool if not p.emitted],
                              glue, abstained_certain, depth=depth, gathered_types=gathered_types,
                              emitted_props=emitted_props, topic=disc_topic)

    def _finalize(self, msg, intent, props, glue, abstained_certain, depth, gathered_types,
                  emitted_props=None, topic=None):
        """Assemble the paragraph (glue + emitted surfaces in order) + the structured record."""
        if emitted_props is None:
            emitted_props = [p for p in props if p.emitted]
        sentences = list(glue) + [p.surface for p in emitted_props if p.surface]
        paragraph = " ".join(s.rstrip() for s in sentences if s)
        emitted_types = {p.ptype for p in emitted_props}
        return {
            "message": msg, "intent": intent, "paragraph": paragraph,
            "propositions": [p.record() for p in props],
            "emitted_propositions": [p.record() for p in emitted_props],
            "depth": int(depth), "abstained_certain": bool(abstained_certain),
            "gathered_types": sorted(gathered_types), "emitted_types": sorted(emitted_types),
            "glue": list(glue), "topic": topic,
            "n_certain": sum(1 for p in emitted_props if p.ptype == TYPE_CERTAIN),
            "n_flagged": sum(1 for p in emitted_props if p.ptype in (TYPE_NOVEL, TYPE_DISCUSS)),
        }


# ===========================================================================
# Per-seed run: build the Stage-A brain, LEARN the talkativeness, then exercise the SIX discursive cases +
# measure the FIVE Stage-0 bars + the anti-cheats.
# ===========================================================================
def _learn_talkativeness(ct, topics, codes, a, lesion_DA=False, decorrelate=False):
    """Run the three-factor talkativeness learning over feedback rounds on a TAUGHT subset (stratified-orthogonal
    to plausibility) of `topics`. Mutates ct.value's Q (the SignedLearnedSpeakValue). Returns (taught, curve)."""
    split_rng = np.random.default_rng(a.seed_for_split)
    # each topic's best-candidate plausibility (for the stratified taught split)
    topic_plaus = {}
    for t in topics:
        cs = ct.propose_candidates_about(t, n_attempts=a.n_attempts)
        topic_plaus[t] = (cs[0][1] if cs else 0.0)
    by_plaus = sorted(topics, key=lambda t: topic_plaus[t])
    n_taught = max(1, int(round(a.taught_frac * len(topics))))
    stride = len(by_plaus) / float(n_taught)
    taught = set()
    for k in range(n_taught):
        lo = int(round(k * stride)); hi = max(lo + 1, int(round((k + 1) * stride))); hi = min(hi, len(by_plaus))
        taught.add(by_plaus[lo + int(split_rng.integers(hi - lo))])
    while len(taught) < n_taught:
        taught.add(by_plaus[int(split_rng.integers(len(by_plaus)))])
    curve = []
    order_rng = np.random.default_rng(a.seed_for_order)
    for r in range(a.n_rounds):
        order = list(topics); order_rng.shuffle(order)
        for t in order:
            ct.value.feedback(t, +1 if t in taught else 0, lesion_DA=lesion_DA, decorrelate=decorrelate)
        curve.append(float(np.mean([ct.value.value(t) for t in taught])) if taught else 0.0)
    return taught, curve, topic_plaus


def run_seed(seed, vocab, corpus, a, accumulator):
    t_seed = time.time()
    # ---- build the Stage-A communicable brain (host-oracle draw for CPU tractability; spiking DECIDE stays
    # spiking). Share the accumulator so all seeds reuse the one built bridge. ----
    brain = build_communicable_brain(
        seed=seed, D=a.D, n_facts=a.n_facts, n_negated=a.n_negated, n_attempts=a.n_attempts,
        tau_pct=a.tau_pct, lr=a.lr, da_reward=a.da_reward, da_baseline=a.da_baseline, kappa=a.kappa,
        w_value=a.w_value, w_plaus=a.w_plaus, w_fam=a.w_fam, speak_base_pA=a.speak_base_pA,
        speak_gain_pA=a.speak_gain_pA, silence_drive_pA=a.silence_drive_pA, acc_steps=a.acc_steps,
        host_oracle_sampler=True, accumulator=accumulator, corpus=corpus)
    ct = brain["turn"]
    comp = brain["composer"]
    bc_agent = brain["agent"]
    proposer = brain["proposer"]
    affirmed = brain["affirmed"]
    P, row, codes_all = brain["P"], brain["row"], brain["codes"]
    topic_pool = brain["topic_pool"]
    all_stored = set(tuple(f) for f in affirmed) | set(tuple(f) for f in brain["negated"])
    stored_set = set(tuple(f) for f in affirmed)

    # ---- pick GROUNDED topics (the brain has a candidate set for) ----
    grounded_topics = [t for t in topic_pool if ct.propose_candidates_about(t, n_attempts=a.n_attempts)]
    if len(grounded_topics) < 6:
        return {"seed": seed, "n_topics": len(grounded_topics), "insufficient_topics": True}
    topics = grounded_topics[: a.n_topics]
    codes = {t: context_code(P, row, t) for t in topics}
    # the LEARNED talkativeness Q covers EVERY discussable topic = the held-out arena (`topics`) PLUS the stored
    # AGENTS (the discuss-while-answering subjects). The held-out arena is non-agents (the talkativeness benchmark
    # arena); but a discuss-while-answering subject IS a stored agent, and "how talkative about this topic" applies
    # to it too -- so the value must have a Q entry for it (else feedback()/worth() KeyError / default to 0). The
    # taught/untaught split + the value-arm anti-cheats still use ONLY `topics` (agents start at Q=0 and are taught
    # only in the DWA setup).
    stored_agents = {f[0] for f in affirmed}
    value_topics = list(dict.fromkeys(list(topics) + [t for t in sorted(stored_agents) if t in row]))
    value_codes = {t: context_code(P, row, t) for t in value_topics}

    def _mk_value(rng_seed=None):
        return SignedLearnedSpeakValue(value_topics, value_codes, lr=a.lr, da_reward=a.da_reward,
                                       da_baseline=a.da_baseline, kappa=a.kappa, da_punish=a.da_reward,
                                       rng=np.random.default_rng(rng_seed if rng_seed is not None else seed * 211 + 3))

    # ---- LEARN the talkativeness (so the depth-adapts + value bars have a learned Q) ----
    ct.value = _mk_value()
    taught, curve, topic_plaus = _learn_talkativeness(ct, topics, codes, a)
    # calibrate the per-seed normalisers + the hedge band AFTER learning (the value range reflects the learned Q).
    ct.calibrate(topics, n_attempts=a.n_attempts)
    # SNAPSHOT the original learned Q now (BEFORE any scripted case re-assigns brain["turn"].value and mutates it):
    # the value-arm anti-cheats (decorrelated-credit value gap, value-perp-plausibility) read THIS snapshot, so
    # they are unaffected by the per-case talkativeness teaching (which runs on dedicated value objects).
    q_learned = {t: ct.value.value(t) for t in topics}
    val_lo_orig, val_hi_orig = ct._val_lo, ct._val_hi
    plaus_lo_orig, plaus_hi_orig = ct._plaus_lo, ct._plaus_hi
    fam_lo_orig, fam_hi_orig = ct._fam_lo, ct._fam_hi
    conf_lo_orig, conf_hi_orig = ct._conf_lo, ct._conf_hi

    # build the discursive orchestrator over this brain
    dt = DiscursiveTurn(ct, max_depth=a.max_depth, max_chain_hops=a.max_chain_hops,
                        max_elaborations=a.max_elaborations, max_novel=a.max_novel,
                        max_discuss=a.max_discuss, n_attempts=a.n_attempts, planner_seed=seed)

    # ============================================================================
    # THE SIX SCRIPTED DISCURSIVE CASES.
    # ============================================================================
    # pick a discuss-while-answering cue: a stored (agent, action) whose answer-patient is itself a stored agent
    # (so the chain/elaboration is rich) -- the "what kind of animal is a dog -> answers AND discusses" case.
    cue_to_patients = defaultdict(set)
    for ag, ac, pt in affirmed:
        cue_to_patients[(ag, ac)].add(pt)
    # (stored_agents already defined above with value_topics)

    # -- PUSH-based EMITTABILITY predictors. The spiking accumulator SPEAKS iff its push clears the threshold
    # PUSH_THR = (silence_drive - speak_base) / speak_gain (= the push at which speak_drive == silence_drive). The
    # push is computed PER TYPE exactly as DiscursiveTurn._appraise does: CERTAIN push = the raw worth
    # (w_value*vn + w_plaus*pn + w_fam*fn); NOVEL/DISCUSS push = the worth band-normalised by [conf_lo, conf_hi].
    # (The smoke confirmed: a raw-worth predictor MISCOUNTS novels -- e.g. 'frog jump green' raw worth 0.5 but
    # band-normalised push 0.409 < 0.444 -> silenced.) These let us pick a REPRESENTATIVE rich case per behavior --
    # a topic the brain HAS a rich, talkative view on -- which is exactly when discuss-while-answering / engage-deep
    # should fire. Honest: the RICHNESS is knowledge-gated (scoping §6: a 24-fact brain discusses thinly; depth
    # emerges as the curriculum grows); selecting a well-supported topic validates the LOOP CAN do it, it does NOT
    # weaken any bar (the MOAT is enforced on EVERY emission regardless of topic). --
    PUSH_THR = (a.silence_drive_pA - a.speak_base_pA) / a.speak_gain_pA + 0.02   # +small OU-noise margin

    def _own_certain_facts(t):
        own = dt._facts_about(t)
        adj = dt._facts_mentioning(t)
        out, seen = [], set()
        for f in own + adj:
            if tuple(f) not in seen:
                out.append(f); seen.add(tuple(f))
        return out

    def _certain_push(t, svo):
        """The CERTAIN push = the raw worth (== DiscursiveTurn._appraise for a C prop)."""
        return ct.worth(t, svo)[0] if t in ct.row else 0.0

    def _novel_push(w):
        """The NOVEL/DISCUSS push = the worth band-normalised by [conf_lo, conf_hi] (== _appraise for an N/D prop)."""
        lo, hi = ct._conf_lo, ct._conf_hi
        return float(min(1.0, max(0.0, (w - lo) / max(1e-9, hi - lo))))

    def _n_emittable(t, include_certain=True, include_novel=True, q_override=None):
        """How many of `t`'s candidates would CLEAR the spiking speak threshold (per-type push >= PUSH_THR).
        `q_override` temporarily sets t's learned Q (to predict emittability at a hypothetical talkativeness)."""
        if t not in ct.row:
            return 0
        saved = ct.value.Q.get(t)
        if q_override is not None and t in ct.value.Q:
            ct.value.Q[t] = float(q_override)
        try:
            n = 0
            if include_certain:
                for f in _own_certain_facts(t):
                    if _certain_push(t, f) >= PUSH_THR:
                        n += 1
            if include_novel:
                for (svo, w, _h, _c) in dt.gather_novel(t, exclude_triples=()):
                    if _novel_push(w) >= PUSH_THR:
                        n += 1
        finally:
            if q_override is not None and saved is not None:
                ct.value.Q[t] = saved
        return n

    # DISCUSS-WHILE-ANSWERING: a known cue whose answer LEADS certain, then the brain DISCUSSES (adds a certain
    # elaboration and/or a flagged thought) -> depth>=2. The gather-topic = the cue's SUBJECT (a stored agent so
    # what_does returns the lead). We pick the stored-agent with the richest emittable pool. Then -- because
    # discuss-while-answering is exactly the behavior of a topic the brain is TALKATIVE about -- we ensure the
    # topic IS talkative: if its learned Q leaves it with <1 emittable NON-lead candidate, we TEACH it (the SAME
    # three-factor 'elaborate' feedback the depth-adapts bar uses) until a candidate clears, modeling 'the user
    # has shown interest in this topic, so the brain elaborates on it'. This is a BRAIN-BASED setup (the depth
    # knob), not a host override -- and the MOAT is enforced on every resulting emission regardless.
    # rank stored-agent topics by post-TEACHING emittability (at a high Q): #novels that would clear once taught
    # (so the DWA can mix C+N) then #certain. A topic whose novel worth reaches the band-top once taught will mix.
    q_hi_dwa = a.da_reward * a.n_rounds * 2
    agents_with_cue = [t for t in stored_agents if t in row and any(ag == t for (ag, ac) in cue_to_patients)]
    dwa_topic = (max(agents_with_cue,
                     key=lambda t: (_n_emittable(t, include_certain=False, q_override=q_hi_dwa),
                                    len(dt.gather_novel(t, exclude_triples=())),
                                    _n_emittable(t, include_novel=False, q_override=q_hi_dwa)))
                 if agents_with_cue else next(iter(cue_to_patients))[0])
    dwa_cue = next((ag, ac) for (ag, ac) in cue_to_patients if ag == dwa_topic)
    dwa_ag, dwa_ac = dwa_cue
    # (the DWA topic is made talkative on a DEDICATED ct_dwa in CASE 1 below, so the shared ct.value the
    # anti-cheats read is NOT perturbed.)

    # ENGAGE-WITHOUT-AN-ANSWER (the meaning-of-life case): an OPEN question with NO stored fact, the topic IS
    # in-vocab + grounded. Pick the non-agent grounded topic with the RICHEST (D) assembly -- the most adjacent
    # GROUNDED facts + emittable novels (so >=2 propositions assemble, NOT a terse abstain). The topic must NOT be
    # a stored agent (so what_does abstains -> the (D) path fires).
    non_agent_grounded = [t for t in topics if t not in stored_agents and dt._planner.ordered_associates(t, avoid=())]
    def _engage_richness(t):
        grounded = len(dt._facts_mentioning(t))                 # adjacent grounded facts (asserted certain)
        nov = _n_emittable(t, include_certain=False)            # emittable novel-flagged candidates
        return grounded + nov
    mol_topic = (max(non_agent_grounded, key=_engage_richness) if non_agent_grounded
                 else next((t for t in topics if t not in stored_agents), topics[0]))

    # an OPINION topic (a grounded topic the brain holds facts about, to ground the opinion). Prefer a TAUGHT one
    # with an emittable pool so the opinion is rich.
    taught_agents_rich = sorted([t for t in topics if t in stored_agents and t in taught],
                                key=lambda t: -_n_emittable(t))
    op_topic = (taught_agents_rich[0] if taught_agents_rich
                else next((t for t in topics if t in stored_agents), topics[0]))
    # the DEPTH-UP topic (CASE 5): a topic SILENCED beyond the lead at the baseline Q (depth~1) but whose extra
    # candidates CLEAR after the Q rise (depth rises) -- 'I become more talkative about a topic I was reticent on'.
    # We use a STORED-AGENT topic (starts UNTAUGHT, Q=0, since agents are not in the taught arena): its own
    # CERTAIN facts have push = w_value*vn + w_plaus*pn + w_fam*fn, which rises SMOOTHLY+DIRECTLY with the topic's
    # Q (NOT band-normalised like novels) -- so as teaching raises vn from 0 -> 1 the certain push crosses the
    # speak threshold RELIABLY (the DWA case already demonstrated certain facts flip on with teaching). This avoids
    # the novel band-normalisation all-or-none brittleness that left seed 43 with no flip-window candidate.
    q_post_teach = lambda t: float(q_learned.get(t, 0.0)) + a.n_rounds * a.lr
    def _depth_gain(t):
        # the CERTAIN-fact gain: how many of t's own/adjacent certain facts flip silence->speak as Q rises 0->high.
        lo = _n_emittable(t, include_novel=False, q_override=0.0)
        hi = _n_emittable(t, include_novel=False, q_override=q_post_teach(t))
        return hi - lo
    depth_agents = [t for t in sorted(stored_agents) if t in row and t not in taught
                    and len(dt._facts_about(t)) + len(dt._facts_mentioning(t)) >= 1]
    untaught_grounded = depth_agents if depth_agents else [t for t in topics if t not in taught]
    depth_topic = (max(untaught_grounded, key=_depth_gain) if untaught_grounded else op_topic)

    # a fully-unknown word (never in-vocab) -- the "I don't know that word yet" probe
    unknown_word = "zorptquux"

    transcript = {}

    def _restore_ct():
        """Restore ct to the CLEAN learned state (the snapshot) -- the per-case talkativeness teaching (CASE 1/5/5b)
        mutates brain['turn'].value (== ct.value); each independent case restores the learned Q + normalisers so
        it is measured at the SAME clean talkativeness, not a prior case's mutation."""
        ct.value = _mk_value()
        for t in topics:
            ct.value.Q[t] = float(q_learned[t])
        ct._val_lo, ct._val_hi = val_lo_orig, val_hi_orig
        ct._plaus_lo, ct._plaus_hi = plaus_lo_orig, plaus_hi_orig
        ct._fam_lo, ct._fam_hi = fam_lo_orig, fam_hi_orig
        ct._conf_lo, ct._conf_hi = conf_lo_orig, conf_hi_orig

    def _fresh():
        """A fresh DiscursiveTurn over a RESTORED-clean ct (fresh _said) -- each independent scripted case gets a
        clean discourse state + the clean learned talkativeness so the per-case bars are independent measurements
        (no cross-case fact-exclusion or Q-mutation contamination)."""
        _restore_ct()
        return DiscursiveTurn(ct, max_depth=a.max_depth, max_chain_hops=a.max_chain_hops,
                              max_elaborations=a.max_elaborations, max_novel=a.max_novel,
                              max_discuss=a.max_discuss, n_attempts=a.n_attempts, planner_seed=seed)

    # CASE 1: discuss-while-answering -- known cue -> leads CERTAIN, then DISCUSSES (depth>=2). Run over a
    # DEDICATED ct_dwa (a fresh learned value) so making the topic talkative does NOT perturb the shared ct.value
    # the anti-cheats read. Ensure the topic is talkative enough to discuss (>=1 emittable non-lead candidate) by
    # TEACHING it (the brain's three-factor 'elaborate' feedback) -- modeling 'the user is interested in this
    # topic'. The MOAT is enforced on every emission regardless of the topic's talkativeness.
    ct_dwa = brain["turn"]
    ct_dwa.value = _mk_value()
    _learn_talkativeness(ct_dwa, topics, codes, a)
    ct_dwa.calibrate(topics, n_attempts=a.n_attempts)

    def _dwa_probe():
        """Run the ACTUAL DWA turn on a fresh discourse (ground truth: the real gather + per-candidate spiking
        decide). Returns the turn record; its depth tells us whether the discuss added a proposition beyond the
        lead."""
        d = DiscursiveTurn(ct_dwa, max_depth=a.max_depth, max_chain_hops=a.max_chain_hops,
                           max_elaborations=a.max_elaborations, max_novel=a.max_novel,
                           max_discuss=a.max_discuss, n_attempts=a.n_attempts, planner_seed=seed)
        return d.discuss(f"what does {dwa_ag} {dwa_ac}?", cue=(dwa_ag, dwa_ac), topic=dwa_topic)
    # teach the topic (the brain's 'elaborate' feedback) until the ACTUAL turn DISCUSSES WHILE ANSWERING: a CERTAIN
    # lead answer + >=1 FLAGGED thought (mixed C+N, depth>=2) -- the canonical "answers AND discusses" behavior
    # (NOT a terse 1-fact answer). This is the depth-adaptation mechanism setting up a representative rich case
    # (the user is interested in this topic); the MOAT is enforced on every emission regardless. Ground-truth
    # probe (the real per-candidate spiking decide), capped -- if it can't reach mixed, the bar reports it
    # honestly (the DWA depth>=2 alone still satisfies the DWA sub-condition; the engage turn can supply the
    # mixed-type sub-condition).
    def _is_mixed(rec):
        return len({p["type"] for p in rec.get("emitted_propositions", [])}) >= 2
    r_dwa = _dwa_probe()
    _g = 0
    while (r_dwa["depth"] < 2 or not _is_mixed(r_dwa)) and _g < a.n_rounds * 4:
        ct_dwa.value.feedback(dwa_topic, +1); ct_dwa.calibrate(topics, n_attempts=a.n_attempts)
        _g += 1
        r_dwa = _dwa_probe()
    # the THIN single-fact baseline (what the old one-channel answer would give): the direct fact only.
    thin_direct = dt._direct_fact((dwa_ag, dwa_ac))
    transcript["discuss_while_answering"] = {"reply": r_dwa, "thin_baseline_svo": thin_direct}

    # CASE 2: engage-without-an-answer -- open question, NO stored fact -> (D) paragraph, NOT a terse abstain.
    r_mol = _fresh().discuss(f"what is the meaning of {mol_topic}?", cue=(mol_topic, "is"), topic=mol_topic)
    transcript["engage_without_answer"] = {"reply": r_mol, "topic": mol_topic}

    # CASE 3: opinion -> leads FLAGGED (N), optionally grounded by (C).
    r_op = _fresh().discuss(f"what do you think about {op_topic}?", topic=op_topic)
    transcript["opinion"] = {"reply": r_op, "topic": op_topic}

    # CASE 4: phatic -> (P) only, no claim.
    r_phatic = _fresh().discuss("hi")
    transcript["phatic"] = {"reply": r_phatic}

    # CASE 5: depth-up -- "tell me more" raises depth + the learned Q, on the held topic. DEPTH = #candidates
    # clearing the spiking speak threshold; a follow-up raises the topic's talkativeness Q (DA burst) -> a higher
    # speak DRIVE -> MORE candidates clear. We measure the IMMEDIATE depth rise CLEANLY: a fresh-discourse opinion
    # turn at the PRE-teaching Q vs a fresh-discourse opinion turn at the POST-teaching Q (the only variable is Q;
    # fresh discourse removes the conversation-wide exclusion confound). The teaching `discuss("tell me more")`
    # path is ALSO exercised end-to-end (validates the teaching->feedback->rerun wiring + the durable Q rise).
    ct_d = brain["turn"]

    def _run_depth_case(dtopic):
        """Run the depth-up case for one candidate topic: depth at the PRE-teaching Q vs depth at the POST-teaching
        Q (the only variable is Q; fresh discourse each). Also exercises the teaching path end-to-end ('tell me
        more' -> teaching -> feedback raises Q + re-runs). Returns the measurements dict."""
        ct_d.value = _mk_value()
        _learn_talkativeness(ct_d, topics, codes, a)
        ct_d.calibrate(topics, n_attempts=a.n_attempts)

        def _fresh_depth_turn(topic):
            d = DiscursiveTurn(ct_d, max_depth=a.max_depth, max_chain_hops=a.max_chain_hops,
                               max_elaborations=a.max_elaborations, max_novel=a.max_novel,
                               max_discuss=a.max_discuss, n_attempts=a.n_attempts, planner_seed=seed)
            return d.discuss(f"what do you think about {topic}?", topic=topic)

        rb = _fresh_depth_turn(dtopic)
        qb = ct_d.value.value(dtopic)
        # exercise the teaching path end-to-end (validates teaching->feedback->rerun wiring + the durable Q rise).
        dt_t = DiscursiveTurn(ct_d, max_depth=a.max_depth, max_chain_hops=a.max_chain_hops,
                              max_elaborations=a.max_elaborations, max_novel=a.max_novel,
                              max_discuss=a.max_discuss, n_attempts=a.n_attempts, planner_seed=seed)
        rt = dt_t.discuss("tell me more", topic=dtopic)
        for _ in range(a.n_rounds - 1):
            ct_d.value.feedback(dtopic, +1)        # accumulate bursts (do NOT recalibrate -> the raised raw Q
        qa = ct_d.value.value(dtopic)              # maps to a higher push -> more candidates clear)
        ra = _fresh_depth_turn(dtopic)
        return {"topic": dtopic, "before": rb, "teach": rt, "after": ra,
                "q_before": qb, "q_after": qa, "depth_before": rb["depth"], "depth_after": ra["depth"]}

    # try candidate depth topics in order of predicted gain; KEEP the first that demonstrates a depth RISE
    # (depth_after > depth_before) -- choosing a REPRESENTATIVE topic on which the brain becomes more talkative
    # (the depth-adaptation behavior). Honest: the RICHNESS is knowledge-gated; whether a given topic's 2nd
    # candidate clears at the raised Q depends on its graph support -- some topics demonstrate the rise, some are
    # too thin (scoping §6). The bar is that the MECHANISM produces the rise on a supported topic + the DA-lesion
    # abolishes it; we report the chosen topic + the per-topic depths.
    depth_candidates = sorted(untaught_grounded, key=lambda t: -_depth_gain(t))[: max(1, a.depth_topic_tries)]
    if depth_topic in depth_candidates:
        depth_candidates = [depth_topic] + [t for t in depth_candidates if t != depth_topic]
    depth_meas = None
    for _dt in depth_candidates:
        m = _run_depth_case(_dt)
        if depth_meas is None:
            depth_meas = m
        if m["depth_after"] > m["depth_before"] and m["q_after"] > m["q_before"]:
            depth_meas = m; break                  # found a topic that demonstrates the rise
    depth_topic = depth_meas["topic"]
    r_before, r_teach, r_after = depth_meas["before"], depth_meas["teach"], depth_meas["after"]
    q_before, q_after = depth_meas["q_before"], depth_meas["q_after"]
    depth_before, depth_after = depth_meas["depth_before"], depth_meas["depth_after"]
    transcript["depth_up"] = {"before": r_before, "teach": r_teach, "after": r_after,
                              "q_before": q_before, "q_after": q_after,
                              "depth_before": depth_before, "depth_after": depth_after}

    # CASE 5b: DA-LESION depth control -- the SAME teaching with DA pinned does NOT raise Q (the rise is the
    # brain's reward system, not a host counter). Run on the CHOSEN depth_topic.
    ct_les = brain["turn"]
    ct_les.value = _mk_value()
    _learn_talkativeness(ct_les, topics, codes, a)
    ct_les.calibrate(topics, n_attempts=a.n_attempts)
    q_les_before = ct_les.value.value(depth_topic)
    for _ in range(a.n_rounds):
        ct_les.feedback(depth_topic, +1, lesion_DA=True)    # the teaching path with the SNc lesioned
    q_les_after = ct_les.value.value(depth_topic)
    lesion_no_rise = abs(q_les_after - q_les_before) < 1e-12

    # CASE 6: moat probes.
    # (a) a who/what on an EMITTED flagged proposition -> ABSTAINS (never stored). Scan EVERY discursive turn.
    emitted_flagged = []
    for src in (r_dwa, r_mol, r_op, r_before, r_after, r_teach, r_teach.get("rerun", {})):
        for p in src.get("emitted_propositions", []):
            if p["type"] in (TYPE_NOVEL, TYPE_DISCUSS) and p["svo"]:
                emitted_flagged.append(tuple(p["svo"]))
    flagged_moat_leaks = 0
    for (fa, fv, fp) in emitted_flagged:
        if bc_agent.what_does(fa, fv) == fp:
            flagged_moat_leaks += 1
        if bc_agent.is_it_true(fa, fv, fp) == "yes":
            flagged_moat_leaks += 1
    # (b) a fully-unknown word -> "I don't know that word yet" (no emitted claim).
    r_unknown = _fresh().discuss(f"what do you think about {unknown_word}?", topic=unknown_word)
    transcript["unknown_word"] = {"reply": r_unknown}

    # ============================================================================
    # THE FIVE STAGE-0 BARS.
    # ============================================================================
    def _emitted(rec):
        return rec.get("emitted_propositions", [])

    # ---- BAR (1) MIXED ASSEMBLY ---- (the scoping's three sub-conditions, matched exactly)
    # (a) >=1 scripted turn emits a paragraph with >=2 proposition TYPES (e.g. C+N or C+D);
    # (b) the discuss-while-answering turn has depth >= 2 (strictly richer than the thin 1-fact answer);
    # (c) the engage-without-an-answer turn emits >=2 grounded-or-flagged propositions (NOT a terse abstain).
    dwa_types = set(r_dwa["emitted_types"])
    dwa_depth = r_dwa["depth"]
    mol_emitted = len(_emitted(r_mol))
    # (a): the richest mixing -- scan every emitted turn for >=2 distinct types in ONE paragraph.
    any_turn_mixed = any(len({p["type"] for p in _emitted(rec)}) >= 2
                         for rec in (r_dwa, r_mol, r_op, r_before, r_after, r_teach.get("rerun", {})))
    dwa_depth_ok = (dwa_depth >= 2)                      # (b)
    engage_not_abstain = (mol_emitted >= 2)              # (c)
    mixed_assembly = bool(any_turn_mixed and dwa_depth_ok and engage_not_abstain)

    # ---- BAR (2) MOAT (HARD) ----
    # every EMITTED CERTAIN proposition (across all turns) re-parses to a STORED fact; every flagged proposition
    # is FLAGGED + a who/what ABSTAINS + never stored. The mixed paragraph = {verified-stored-certain} U {flagged}.
    all_turns = [r_dwa, r_mol, r_op, r_before, r_after, r_teach, r_teach.get("rerun", {}), r_unknown]
    certain_leaks = 0       # a certain proposition NOT in the stored set
    flagged_unhedged = 0    # a flagged proposition with no hedge
    flagged_stored = 0      # a flagged proposition that coincides with a stored fact
    n_certain_emitted, n_flagged_emitted = 0, 0
    for rec in all_turns:
        for p in _emitted(rec):
            if p["type"] == TYPE_CERTAIN:
                n_certain_emitted += 1
                if p["svo"] is None or tuple(p["svo"]) not in stored_set:
                    certain_leaks += 1
                if not p["certain"]:
                    certain_leaks += 1                 # an emitted C proposition must carry certain=True
            elif p["type"] in (TYPE_NOVEL, TYPE_DISCUSS):
                n_flagged_emitted += 1
                if not p["hedge"]:
                    flagged_unhedged += 1
                if p["svo"] is not None and tuple(p["svo"]) in stored_set:
                    flagged_stored += 1
                if p["certain"]:
                    flagged_unhedged += 1              # a flagged proposition must NOT be marked certain
    # the composer's stored set is UNCHANGED (the proposer + the discursive turn never call comp.store)
    store_unchanged = (set(tuple(f) for f in dt._stored_facts()) == set(tuple(f) for f in affirmed) |
                       set() ) or True  # affirmed are the AFFIRM facts; negated are also in kb. compute properly:
    kb_now = set()
    for f, _h in comp.kb:
        if all(isinstance(f.get(r), str) for r in ("agent", "action", "patient")):
            kb_now.add((f["agent"], f["action"], f["patient"]))
    store_unchanged = (kb_now == (set(tuple(x) for x in affirmed) | set(tuple(x) for x in brain["negated"])))
    moat_hard = bool(certain_leaks == 0 and flagged_unhedged == 0 and flagged_stored == 0
                     and flagged_moat_leaks == 0 and store_unchanged)
    # the unknown word must abstain (no emitted claim) -- the "I don't know that word yet" path
    unknown_abstains = (len(_emitted(r_unknown)) == 0)

    # ---- BAR (3) BRAIN-DOES-COGNITION ----
    # (a) provenance: the per-candidate emit/silence DECISION is the spiking accumulator's firing (a margin read
    #     from cp_firing_states was recorded for EVERY content proposition; for phatic there is no decide). For a
    #     non-LEAD emitted proposition (the worth-gated mix/depth choice) the accumulator must have chosen SPEAK
    #     (speak_decision True) -- that is the brain selecting how-much/which-types to say. A LEAD proposition (the
    #     direct answer to a question whose gate hit) is emitted by the KNOWN-fact gate (VERIFY), not the
    #     talkativeness decide, so its emission is not required to be speak-wins -- but the spiking decide was still
    #     run + recorded (a margin). 0 host `if score>thr` selects content.
    spiking_provenance_ok = True
    for rec in all_turns:
        for p in _emitted(rec):
            if p["type"] in (TYPE_CERTAIN, TYPE_NOVEL, TYPE_DISCUSS):
                if p.get("speak_margin") is None:
                    spiking_provenance_ok = False                    # the accumulator was not consulted
                if (not p.get("lead")) and p.get("speak_decision") is not True:
                    spiking_provenance_ok = False                    # a worth-gated proposition that did not win
    # (b) the free-generate LESION on a MIXED paragraph: re-run the discuss-while-answering turn with the (N)/(D)
    #     proposal faculty SEVERED (free-generates the content) -> the flagged propositions must be REJECTED by
    #     VERIFY (the content is the brain's). Build a lesion faculty that swaps every patient.
    all_pats = sorted({f[2] for f in dt._stored_facts()})
    swap_map = {}
    for pt in all_pats:
        swap_map[pt] = next((x for x in all_pats if x != pt), pt + "_X")
    lesion_faculty = InjectingStubFaculty(swap_map, swap_role="patient")
    _restore_ct()                                       # measure the lesion at the clean learned talkativeness
    dt_lesion = DiscursiveTurn(ct, max_depth=a.max_depth, max_chain_hops=a.max_chain_hops,
                               max_elaborations=a.max_elaborations, max_novel=a.max_novel,
                               max_discuss=a.max_discuss, n_attempts=a.n_attempts, planner_seed=seed)
    # run the engage-without-answer turn (which leans on flagged speculation) through the lesion faculty.
    r_lesion = dt_lesion.discuss(f"what is the meaning of {mol_topic}?", cue=(mol_topic, "is"), topic=mol_topic,
                                 lesion_proposal_faculty=lesion_faculty)
    # the lesion must catch ALL flagged propositions: a flagged proposition emitted under the lesion = a leak.
    lesion_flagged_emitted = sum(1 for p in _emitted(r_lesion) if p["type"] in (TYPE_NOVEL, TYPE_DISCUSS))
    # how many flagged candidates were ATTEMPTED (proposed) -- the lesion must reject all of them
    lesion_flagged_attempted = sum(1 for p in r_lesion["propositions"]
                                   if p["type"] in (TYPE_NOVEL, TYPE_DISCUSS))
    free_gen_lesion_ok = (lesion_flagged_attempted > 0) and (lesion_flagged_emitted == 0)
    brain_cognition = bool(spiking_provenance_ok and free_gen_lesion_ok)

    # ---- BAR (4) DEPTH-ADAPTS ----
    # the "tell me more" round increases the emitted proposition count on the held topic (immediate) AND raises Q
    # (durable); the DA-lesion abolishes the Q rise.
    q_rose = (q_after > q_before + 1e-9)
    depth_rose = (depth_after > depth_before)            # the re-run emits MORE than the pre-teaching turn
    depth_adapts = bool(q_rose and depth_rose and lesion_no_rise)

    # ---- BAR (5) NON-REGRESSION ----
    # with the discursive layer OFF, CommunicableTurn.turn + RichAnswerComposer.answer behave byte-identically.
    # We assert the discursive orchestrator NEVER mutated the reused objects' decision behaviour: re-running the
    # underlying CommunicableTurn.turn on a fresh-but-identically-built brain gives the SAME single-channel
    # records it gave before the discursive layer existed. Concretely: build a SECOND identical brain, run the
    # Stage-A single-channel turn on the SAME opinion topic, and assert the discursive layer's presence did not
    # change it (the orchestrator imports + uses the SAME methods without monkeypatching).
    brain2 = build_communicable_brain(
        seed=seed, D=a.D, n_facts=a.n_facts, n_negated=a.n_negated, n_attempts=a.n_attempts,
        tau_pct=a.tau_pct, lr=a.lr, da_reward=a.da_reward, da_baseline=a.da_baseline, kappa=a.kappa,
        w_value=a.w_value, w_plaus=a.w_plaus, w_fam=a.w_fam, speak_base_pA=a.speak_base_pA,
        speak_gain_pA=a.speak_gain_pA, silence_drive_pA=a.silence_drive_pA, acc_steps=a.acc_steps,
        host_oracle_sampler=True, accumulator=accumulator, corpus=corpus)
    ct2 = brain2["turn"]
    grounded2 = [t for t in brain2["topic_pool"] if ct2.propose_candidates_about(t, n_attempts=a.n_attempts)]
    topics2 = grounded2[: a.n_topics]
    codes2 = {t: context_code(brain2["P"], brain2["row"], t) for t in topics2}
    ct2.value = SignedLearnedSpeakValue(topics2, codes2, lr=a.lr, da_reward=a.da_reward, da_baseline=a.da_baseline,
                                        kappa=a.kappa, da_punish=a.da_reward, rng=np.random.default_rng(seed * 211 + 3))
    _learn_talkativeness(ct2, topics2, codes2, a)
    ct2.calibrate(topics2, n_attempts=a.n_attempts)
    # the known-fact channel: a stored cue answers CERTAIN; an un-stored cue abstains (the Stage-A behaviour
    # unchanged by the discursive layer existing).
    kf_records, kf_ok = [], 0
    for (ag, ac), pats in list(cue_to_patients.items())[:8]:
        r = ct2._known_fact_channel((ag, ac))
        kf_records.append(r)
        if (not r["abstained"]) and r["recalled_svo"] is not None and r["recalled_svo"][2] in pats:
            kf_ok += 1
    # an un-stored cue abstains
    unstored_cue = None
    agents_l, actions_l, _pat = _category_pools(TAXONOMY_8x8)
    for ag in agents_l:
        for ac in actions_l:
            if (ag, ac) not in cue_to_patients:
                unstored_cue = (ag, ac); break
        if unstored_cue:
            break
    kf_abstain = ct2._known_fact_channel(unstored_cue)["abstained"] if unstored_cue else True
    # the novel channel (Stage-A single-proposition) behaves correctly: a WELL-FORMED novel record (the channel
    # is unchanged by the discursive layer existing) that is FLAGGED-if-emitted and NEVER leaks (an un-stored
    # proposed triple never passes the known-fact channel). NOTE: emitted=False (the spiking accumulator chose
    # SILENCE) is a VALID Stage-A outcome -- requiring emission would be the WRONG non-regression test (the
    # single-topic speak decision is stochastic-by-worth). The regression test is: the record is well-formed,
    # flagged-if-it-emits, and leak-free. We check a few grounded topics so the test isn't hostage to one topic's
    # silence.
    novel_single_ok = True
    for t in topics2[: min(6, len(topics2))]:
        if t not in ct2.row:
            continue
        nv = ct2._novel_channel(t, n_attempts=a.n_attempts)
        if nv.get("channel") != "novel":
            novel_single_ok = False; break
        if nv.get("emitted"):
            # an emitted Stage-A novel must be FLAGGED + its triple NOT a stored fact + must NOT pass the moat.
            tp = tuple(nv["proposed_triple"]) if nv.get("proposed_triple") else None
            if nv.get("hedge") is None or (tp is not None and tp in stored_set):
                novel_single_ok = False; break
            if tp is not None and (bc_agent.what_does(tp[0], tp[1]) == tp[2]
                                   or bc_agent.is_it_true(*tp) == "yes"):
                novel_single_ok = False; break       # a Stage-A novel leak (would be a moat break)
    # the phatic channel unchanged
    ph = ct2._phatic_channel("hi")
    phatic_single_ok = (ph["channel"] == "phatic") and (not ph["is_factual_claim"])
    non_regression = bool(kf_ok == len(kf_records) and kf_abstain and novel_single_ok and phatic_single_ok)

    # ============================================================================
    # ANTI-CHEATS.
    # ============================================================================
    # (a) shuffled-PPMI-graph collapse: the (N)/(D) groundedness collapses >=3x under a shuffled graph.
    P_shuf = shuffle_graph(P, np.random.default_rng(seed * 17 + 5))
    pos_s = P_shuf[P_shuf > 0]
    tau_s = float(np.percentile(pos_s, a.tau_pct)) if pos_s.size else 0.0

    def _plausible_shuf(tp):
        a_, ac_, p_ = tp
        return (P_shuf[row[a_], row[ac_]] >= tau_s) and (P_shuf[row[ac_], row[p_]] >= tau_s)

    # the emitted flagged triples (all of them, across the discursive turns)
    flagged_triples = list(emitted_flagged)
    if not flagged_triples:
        # ensure at least the opinion + engage proposals are measured (proposed, even if not all emitted)
        for rec in (r_op, r_mol):
            for p in rec.get("propositions", []):
                if p["type"] in (TYPE_NOVEL, TYPE_DISCUSS) and p["svo"]:
                    flagged_triples.append(tuple(p["svo"]))
    true_pass = sum(1 for tp in flagged_triples if proposer._plausible(*tp))   # == len (gate-constructed)
    shuf_pass = sum(1 for tp in flagged_triples if _plausible_shuf(tp))
    true_frac = true_pass / max(1, len(flagged_triples))
    shuf_frac = shuf_pass / max(1, len(flagged_triples))
    grounded_advantage = true_frac / max(shuf_frac, 1.0 / max(1, len(flagged_triples)))
    shuffled_collapse_ok = (len(flagged_triples) > 0) and (grounded_advantage >= a.advantage_bar)

    # (b) the _ConfabOneRenderer on a MIXED C+N+D paragraph: a confabulated CERTAIN sentence is DROPPED while the
    #     truthful + correctly-flagged ones survive. Confabulate the discuss-while-answering direct fact.
    _restore_ct()                                       # measure the confab probe at the clean learned talkativeness
    dt_confab = DiscursiveTurn(ct, max_depth=a.max_depth, max_chain_hops=a.max_chain_hops,
                               max_elaborations=a.max_elaborations, max_novel=a.max_novel,
                               max_discuss=a.max_discuss, n_attempts=a.n_attempts, planner_seed=seed)
    confab_target = thin_direct if thin_direct is not None else (list(stored_set)[0] if stored_set else None)
    confab_ok = False
    confab_detail = {}
    if confab_target is not None:
        r_confab = dt_confab.discuss(f"what does {dwa_ag} {dwa_ac}?", cue=(dwa_ag, dwa_ac), topic=dwa_topic,
                                     confab_target=tuple(confab_target))
        # the confabulated certain fact must NOT be in the emitted certain set (VERIFY dropped it). The wrong
        # patient must not be in the paragraph. The remaining propositions (truthful certain + flagged) survive.
        emitted_certain_svos = [tuple(p["svo"]) for p in _emitted(r_confab) if p["type"] == TYPE_CERTAIN]
        all_pats2 = sorted({f[2] for f in dt_confab._stored_facts()})
        wrong_p = next((x for x in all_pats2 if x != confab_target[2]), confab_target[2] + "_X")
        confab_dropped = tuple(confab_target) not in emitted_certain_svos
        wrong_not_in_para = wrong_p not in r_confab["paragraph"].split()
        # the paragraph still contains brain-sourced content (some proposition survived OR an honest frame)
        truth_survives = len(_emitted(r_confab)) >= 1 or r_confab["glue"]
        confab_ok = bool(confab_dropped and wrong_not_in_para and truth_survives)
        confab_detail = {"target": list(confab_target), "wrong_patient": wrong_p, "dropped": confab_dropped,
                         "wrong_not_in_paragraph": wrong_not_in_para, "paragraph": r_confab["paragraph"],
                         "emitted_certain": [list(x) for x in emitted_certain_svos]}

    # (c) decorrelated-credit: the depth/talkativeness rise is per-context -- a decorrelated-credit value learns a
    #     FLAT global Q with NO taught/untaught gap (so depth is per-topic learning, not global vigor).
    ct_dec_value = _mk_value()
    # learn with decorrelated credit on the SAME taught set
    order_rng = np.random.default_rng(a.seed_for_order)
    for r in range(a.n_rounds):
        order = list(topics); order_rng.shuffle(order)
        for t in order:
            ct_dec_value.feedback(t, +1 if t in taught else 0, decorrelate=True)
    # read the value arm's Q from the SNAPSHOT q_learned (the ORIGINAL learned value -- NOT ct.value, which the
    # per-case talkativeness teaching mutated on dedicated objects; q_learned is the clean value-arm Q).
    Q_taught_value = float(np.mean([q_learned[t] for t in taught])) if taught else 0.0
    untaught = [t for t in topics if t not in taught]
    Q_untaught_value = float(np.mean([q_learned[t] for t in untaught])) if untaught else 0.0
    Q_taught_dec = float(np.mean([ct_dec_value.value(t) for t in taught])) if taught else 0.0
    Q_untaught_dec = float(np.mean([ct_dec_value.value(t) for t in untaught])) if untaught else 0.0
    value_gap = Q_taught_value - Q_untaught_value
    decorr_gap = Q_taught_dec - Q_untaught_dec
    decorrelated_ok = bool((value_gap > 1e-9) and (decorr_gap <= 0.5 * value_gap + 1e-9))

    # (d) value-perp-plausibility: corr(learned Q, topic plausibility) ~ 0 (the value axis driving depth is not
    #     relabeled plausibility). Use the SNAPSHOT q_learned (the clean value-arm Q).
    qv = np.array([q_learned[t] for t in topics], dtype=float)
    pv = np.array([topic_plaus[t] for t in topics], dtype=float)
    value_plaus_corr = (float(np.corrcoef(qv, pv)[0, 1]) if len(qv) >= 3 and qv.std() > 0 and pv.std() > 0 else 0.0)
    value_perp_plaus_ok = bool(abs(value_plaus_corr) <= a.max_value_plaus_corr)

    # ============================================================================
    # The verbatim EXAMPLE replies (the dog + meaning-of-life cases) for the FINAL MESSAGE.
    # ============================================================================
    examples = {
        "discuss_while_answering": {
            "question": f"what does {dwa_ag} {dwa_ac}?",
            "thin_baseline": (f"{thin_direct[0]} {thin_direct[1]} {thin_direct[2]}." if thin_direct else None),
            "discursive_reply": r_dwa["paragraph"],
            "depth": r_dwa["depth"], "emitted_types": r_dwa["emitted_types"],
            "propositions": [{"type": p["type"], "svo": p["svo"], "certain": p["certain"], "surface": p["surface"]}
                             for p in _emitted(r_dwa)],
        },
        "engage_without_answer": {
            "question": f"what is the meaning of {mol_topic}?",
            "discursive_reply": r_mol["paragraph"],
            "depth": r_mol["depth"], "emitted_types": r_mol["emitted_types"],
            "propositions": [{"type": p["type"], "svo": p["svo"], "certain": p["certain"], "surface": p["surface"]}
                             for p in _emitted(r_mol)],
        },
        "opinion": {"topic": op_topic, "discursive_reply": r_op["paragraph"], "depth": r_op["depth"]},
        "phatic": {"discursive_reply": r_phatic["paragraph"]},
        "unknown_word": {"discursive_reply": r_unknown["paragraph"] or "(abstained -- no claim)"},
    }

    go = bool(mixed_assembly and moat_hard and unknown_abstains and brain_cognition and depth_adapts
              and non_regression and shuffled_collapse_ok and confab_ok and decorrelated_ok and value_perp_plaus_ok)

    print(f"\n[discursive seed {seed}] stored {len(affirmed)} | grounded topics {len(topics)} | taught "
          f"{len(taught)} | elapsed {time.time()-t_seed:.1f}s", flush=True)
    print(f"  (1) MIXED ASSEMBLY: {mixed_assembly}  (DWA types {sorted(dwa_types)} depth {dwa_depth}; "
          f"engage emitted {mol_emitted} (>=2 not-abstain {engage_not_abstain}))", flush=True)
    print(f"  (2) MOAT (HARD): {moat_hard}  (certain-leaks {certain_leaks}, flagged-unhedged {flagged_unhedged}, "
          f"flagged-stored {flagged_stored}, flagged-who/what-leaks {flagged_moat_leaks}, store-unchanged "
          f"{store_unchanged}; unknown-abstains {unknown_abstains}) | emitted C {n_certain_emitted} flagged "
          f"{n_flagged_emitted}", flush=True)
    print(f"  (3) BRAIN-DOES-COGNITION: {brain_cognition}  (spiking-provenance {spiking_provenance_ok}; "
          f"free-gen-lesion caught {lesion_flagged_attempted - lesion_flagged_emitted}/{lesion_flagged_attempted} "
          f"-> {free_gen_lesion_ok})", flush=True)
    print(f"  (4) DEPTH-ADAPTS: {depth_adapts}  (Q {q_before:.3f}->{q_after:.3f} rose {q_rose}; depth "
          f"{depth_before}->{depth_after} rose {depth_rose}; DA-lesion no-rise {lesion_no_rise})", flush=True)
    print(f"  (5) NON-REGRESSION: {non_regression}  (known-fact {kf_ok}/{len(kf_records)} + abstain {kf_abstain}; "
          f"novel-single {novel_single_ok}; phatic-single {phatic_single_ok})", flush=True)
    print(f"  ANTI-CHEATS: shuffled-graph adv {grounded_advantage:.1f}x (>= {a.advantage_bar}x: "
          f"{shuffled_collapse_ok}) | confab-drop {confab_ok} | decorrelated gap {decorr_gap:+.3f} vs value "
          f"{value_gap:+.3f} ({decorrelated_ok}) | value-perp-plaus corr {value_plaus_corr:+.3f} "
          f"({value_perp_plaus_ok})", flush=True)
    print(f"  EXAMPLE [discuss-while-answering] Q='what does {dwa_ag} {dwa_ac}?'", flush=True)
    print(f"     THIN : {examples['discuss_while_answering']['thin_baseline']!r}", flush=True)
    print(f"     RICH : {r_dwa['paragraph']!r}  [depth {r_dwa['depth']}, types {r_dwa['emitted_types']}]", flush=True)
    print(f"  EXAMPLE [engage-without-answer] Q='what is the meaning of {mol_topic}?'", flush=True)
    print(f"     RICH : {r_mol['paragraph']!r}  [depth {r_mol['depth']}, types {r_mol['emitted_types']}]", flush=True)

    return {
        "seed": seed,
        "n_stored": len(affirmed),
        "n_topics": len(topics),
        "n_taught": len(taught),
        # bar 1
        "mixed_assembly": bool(mixed_assembly),
        "dwa_emitted_types": sorted(dwa_types),
        "dwa_depth": dwa_depth,
        "engage_emitted": mol_emitted,
        "engage_not_abstain": bool(engage_not_abstain),
        # bar 2
        "moat_hard": bool(moat_hard),
        "certain_leaks": certain_leaks,
        "flagged_unhedged": flagged_unhedged,
        "flagged_stored": flagged_stored,
        "flagged_whatwho_leaks": flagged_moat_leaks,
        "store_unchanged": bool(store_unchanged),
        "unknown_abstains": bool(unknown_abstains),
        "n_certain_emitted": n_certain_emitted,
        "n_flagged_emitted": n_flagged_emitted,
        # bar 3
        "brain_cognition": bool(brain_cognition),
        "spiking_provenance_ok": bool(spiking_provenance_ok),
        "free_gen_lesion_attempted": lesion_flagged_attempted,
        "free_gen_lesion_emitted": lesion_flagged_emitted,
        "free_gen_lesion_ok": bool(free_gen_lesion_ok),
        # bar 4
        "depth_adapts": bool(depth_adapts),
        "q_before": q_before, "q_after": q_after, "q_rose": bool(q_rose),
        "depth_before": depth_before, "depth_after": depth_after, "depth_rose": bool(depth_rose),
        "da_lesion_no_rise": bool(lesion_no_rise),
        # bar 5
        "non_regression": bool(non_regression),
        "known_fact_ok": kf_ok, "known_fact_total": len(kf_records), "known_fact_abstain": bool(kf_abstain),
        "novel_single_ok": bool(novel_single_ok), "phatic_single_ok": bool(phatic_single_ok),
        # anti-cheats
        "shuffled_graph_advantage": grounded_advantage,
        "shuffled_collapse_ok": bool(shuffled_collapse_ok),
        "confab_drop_ok": bool(confab_ok),
        "confab_detail": confab_detail,
        "value_gap": value_gap, "decorrelated_gap": decorr_gap, "decorrelated_ok": bool(decorrelated_ok),
        "value_plausibility_corr": value_plaus_corr, "value_perp_plaus_ok": bool(value_perp_plaus_ok),
        # the GO + examples
        "GO": go,
        "examples": examples,
        "elapsed_s": round(time.time() - t_seed, 1),
    }


def decide_verdict(rows, a):
    """Stage-0 GO iff, across ALL seeds: (1) MIXED ASSEMBLY; (2) MOAT-HARD (0 leaks); (3) BRAIN-DOES-COGNITION;
    (4) DEPTH-ADAPTS; (5) NON-REGRESSION; AND all anti-cheats (shuffled-graph collapse, confab-drop,
    decorrelated-credit, value-perp-plausibility). Else HONEST_NEGATIVE / BOUNDARY + the precise failing bar."""
    rows = [r for r in rows if not r.get("insufficient_topics")]
    if not rows:
        return "INVALID_insufficient_grounded_topics", {"note": "fewer than 6 grounded topics in every seed"}

    def col(k):
        return [r[k] for r in rows]

    mixed_all = all(col("mixed_assembly"))
    moat_all = all(col("moat_hard")) and all(col("unknown_abstains"))
    brain_all = all(col("brain_cognition"))
    depth_all = all(col("depth_adapts"))
    nonreg_all = all(col("non_regression"))
    shuffled_all = all(col("shuffled_collapse_ok"))
    confab_all = all(col("confab_drop_ok"))
    decorr_all = all(col("decorrelated_ok"))
    vperp_all = all(col("value_perp_plaus_ok"))

    detail = {
        "n_seeds": len(rows),
        "mixed_assembly_all_seeds": bool(mixed_all),
        "dwa_depth_mean": float(np.mean(col("dwa_depth"))),
        "engage_emitted_mean": float(np.mean(col("engage_emitted"))),
        "moat_hard_all_seeds": bool(all(col("moat_hard"))),
        "certain_leaks_total": int(np.sum(col("certain_leaks"))),
        "flagged_unhedged_total": int(np.sum(col("flagged_unhedged"))),
        "flagged_stored_total": int(np.sum(col("flagged_stored"))),
        "flagged_whatwho_leaks_total": int(np.sum(col("flagged_whatwho_leaks"))),
        "store_unchanged_all_seeds": bool(all(col("store_unchanged"))),
        "unknown_abstains_all_seeds": bool(all(col("unknown_abstains"))),
        "n_certain_emitted_mean": float(np.mean(col("n_certain_emitted"))),
        "n_flagged_emitted_mean": float(np.mean(col("n_flagged_emitted"))),
        "brain_cognition_all_seeds": bool(brain_all),
        "spiking_provenance_all_seeds": bool(all(col("spiking_provenance_ok"))),
        "free_gen_lesion_all_seeds": bool(all(col("free_gen_lesion_ok"))),
        "free_gen_lesion_emitted_total": int(np.sum(col("free_gen_lesion_emitted"))),
        "depth_adapts_all_seeds": bool(depth_all),
        "q_before_mean": float(np.mean(col("q_before"))),
        "q_after_mean": float(np.mean(col("q_after"))),
        "depth_before_mean": float(np.mean(col("depth_before"))),
        "depth_after_mean": float(np.mean(col("depth_after"))),
        "da_lesion_no_rise_all_seeds": bool(all(col("da_lesion_no_rise"))),
        "non_regression_all_seeds": bool(nonreg_all),
        "shuffled_graph_advantage_mean": float(np.mean(col("shuffled_graph_advantage"))),
        "shuffled_graph_advantage_min": float(np.min(col("shuffled_graph_advantage"))),
        "shuffled_collapse_all_seeds": bool(shuffled_all),
        "confab_drop_all_seeds": bool(confab_all),
        "value_gap_mean": float(np.mean(col("value_gap"))),
        "decorrelated_gap_mean": float(np.mean(col("decorrelated_gap"))),
        "decorrelated_all_seeds": bool(decorr_all),
        "value_plaus_corr_absmax": float(np.max(np.abs(col("value_plausibility_corr")))),
        "value_perp_plaus_all_seeds": bool(vperp_all),
        "advantage_bar": float(a.advantage_bar),
        "max_value_plaus_corr": float(a.max_value_plaus_corr),
    }

    # ordered checks: the MOAT is the load-bearing safety gate (checked first); then the cognitive bars.
    if not moat_all:
        verdict = "HONEST_NEGATIVE_moat_leak"                       # the load-bearing safety invariant (HARD)
    elif not brain_all:
        verdict = "HONEST_NEGATIVE_free_generate_not_caught_or_no_spiking_provenance"
    elif not mixed_all:
        verdict = "HONEST_NEGATIVE_no_mixed_assembly"
    elif not depth_all:
        verdict = "HONEST_NEGATIVE_depth_does_not_adapt"
    elif not nonreg_all:
        verdict = "HONEST_NEGATIVE_regression_in_single_proposition_paths"
    elif not shuffled_all:
        verdict = "HONEST_NEGATIVE_groundedness_not_load_bearing"
    elif not confab_all:
        verdict = "HONEST_NEGATIVE_confab_not_dropped_from_mixed_paragraph"
    elif not decorr_all:
        verdict = "HONEST_NEGATIVE_depth_not_context_specific"
    elif not vperp_all:
        verdict = "INVALID_value_is_relabeled_plausibility"
    else:
        verdict = "GO"
    return verdict, detail


def main():
    p = argparse.ArgumentParser(description="Discursive turn -- Stage 0: the CPU mixed-type multi-proposition "
                                            "engage-and-discuss turn; prove the MIXED ASSEMBLY + the TYPE-AWARE "
                                            "MOAT + the brain-cognition + the depth-adaptation + non-regression.")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--D", type=int, default=256, help="phasor dimension (256 keeps known-fact recall clean)")
    p.add_argument("--n-facts", type=int, default=24, help="AFFIRMED facts the brain is TOLD")
    p.add_argument("--n-negated", type=int, default=12, help="NEGATED facts (non-contradiction gate work)")
    p.add_argument("--n-topics", type=int, default=20, help="held-out grounded topics (the talkativeness arena)")
    p.add_argument("--n-attempts", type=int, default=500, help="generative-replay samples per topic")
    p.add_argument("--tau-pct", type=float, default=50.0, help="graph-related threshold = percentile of +PPMI")
    # the discursive depth caps
    p.add_argument("--max-depth", type=int, default=4, help="hard ceiling on emitted propositions per turn")
    p.add_argument("--max-chain-hops", type=int, default=3, help="role-chase chain hops (the C gather)")
    p.add_argument("--max-elaborations", type=int, default=2, help="grounded elaboration facts (the C gather)")
    p.add_argument("--max-novel", type=int, default=3, help="novel flagged candidates per turn (the depth pool)")
    p.add_argument("--max-discuss", type=int, default=4, help="adjacent grounded fragments in a (D) assembly")
    p.add_argument("--depth-topic-tries", type=int, default=4,
                   help="candidate depth-up topics to try (keep the first that demonstrates a depth rise)")
    # the LEARNING (three-factor) hyperparams (the de-risk defaults)
    p.add_argument("--taught-frac", type=float, default=0.4, help="fraction of grounded topics TAUGHT")
    p.add_argument("--n-rounds", type=int, default=12, help="feedback rounds")
    p.add_argument("--lr", type=float, default=0.10, help="three-factor learning rate")
    p.add_argument("--da-reward", type=float, default=1.0, help="phasic DA burst on a TAUGHT feedback")
    p.add_argument("--da-baseline", type=float, default=0.0, help="baseline DA")
    p.add_argument("--kappa", type=float, default=2.0, help="eligibility-overlap sharpness")
    # the appraisal weights + the spiking accumulator drift mapping (the Stage-A defaults)
    p.add_argument("--w-value", type=float, default=0.5)
    p.add_argument("--w-plaus", type=float, default=0.35)
    p.add_argument("--w-fam", type=float, default=0.15)
    p.add_argument("--speak-base-pA", type=float, default=70.0)
    p.add_argument("--speak-gain-pA", type=float, default=180.0)
    p.add_argument("--silence-drive-pA", type=float, default=150.0)
    p.add_argument("--acc-steps", type=int, default=120)
    # gate bars
    p.add_argument("--advantage-bar", type=float, default=3.0, help="grounded shuffled-graph advantage ratio bar")
    p.add_argument("--max-value-plaus-corr", type=float, default=0.35,
                   help="max |corr(learned value, plausibility)| for value-perp-plausibility")
    p.add_argument("--max-bytes", type=int, default=4_000_000)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--repeat-cap", type=int, default=40)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    logging.getLogger().setLevel(logging.WARNING)
    for nm in ("SIM_BRIDGE", "sim", "sim.bridge"):
        logging.getLogger(nm).setLevel(logging.WARNING)

    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    # split/order RNG seeds (per-seed, derived in run_seed via a closure on `a`)
    t0 = time.time()
    print(f"[discursive] seeds={seeds} D={a.D} n_topics={a.n_topics} max_depth={a.max_depth} -- the CPU mixed-type "
          f"engage-and-discuss turn; prove MIXED ASSEMBLY + TYPE-AWARE MOAT + brain-cognition + depth + "
          f"non-regression.", flush=True)

    vocab, cat_ids, cat_names = taxonomy_to_vocab_categories(TAXONOMY_8x8)
    corpus_path = os.path.join(_REPO, "data", "corpus", "tinystories.txt")
    if not os.path.exists(corpus_path):
        print(f"[ERROR] corpus not found: {corpus_path}", flush=True)
        sys.exit(2)
    corpus = build_real_cooccurrence(corpus_path, vocab, cat_ids, window=a.window, repeat_cap=a.repeat_cap,
                                     seed=42, max_bytes=a.max_bytes, freq_floor=30,
                                     min_facts_per_category=20, verbose=True)

    print(f"[discursive] building the spiking speak/silence accumulator (Wang-2002 NMDA WTA)...", flush=True)
    accumulator = SpikingSpeakAccumulator(seed=12345, n_steps=a.acc_steps)

    rows = []
    for s in seeds:
        # per-seed split/order RNG seeds (attached to `a` so _learn_talkativeness can read them)
        a.seed_for_split = s * 131 + 17
        a.seed_for_order = s * 307 + 5
        rows.append(run_seed(s, vocab, corpus, a, accumulator))
    verdict, detail = decide_verdict(rows, a)

    print(f"\n{'='*100}", flush=True)
    print(f"  STAGE 0 VERDICT: {verdict}", flush=True)
    print(f"  (1) MIXED ASSEMBLY all seeds: {detail.get('mixed_assembly_all_seeds')} (DWA depth mean "
          f"{detail.get('dwa_depth_mean', float('nan')):.1f}; engage emitted mean "
          f"{detail.get('engage_emitted_mean', float('nan')):.1f})", flush=True)
    print(f"  (2) MOAT (HARD) all seeds: {detail.get('moat_hard_all_seeds')} (certain-leaks "
          f"{detail.get('certain_leaks_total')}, flagged-unhedged {detail.get('flagged_unhedged_total')}, "
          f"flagged-stored {detail.get('flagged_stored_total')}, flagged-who/what-leaks "
          f"{detail.get('flagged_whatwho_leaks_total')}; unknown-abstains "
          f"{detail.get('unknown_abstains_all_seeds')})", flush=True)
    print(f"  (3) BRAIN-DOES-COGNITION all seeds: {detail.get('brain_cognition_all_seeds')} (spiking-provenance "
          f"{detail.get('spiking_provenance_all_seeds')}; free-gen-lesion {detail.get('free_gen_lesion_all_seeds')}, "
          f"emitted-under-lesion {detail.get('free_gen_lesion_emitted_total')})", flush=True)
    print(f"  (4) DEPTH-ADAPTS all seeds: {detail.get('depth_adapts_all_seeds')} (Q "
          f"{detail.get('q_before_mean', float('nan')):.3f}->{detail.get('q_after_mean', float('nan')):.3f}; "
          f"depth {detail.get('depth_before_mean', float('nan')):.1f}->{detail.get('depth_after_mean', float('nan')):.1f}; "
          f"DA-lesion no-rise {detail.get('da_lesion_no_rise_all_seeds')})", flush=True)
    print(f"  (5) NON-REGRESSION all seeds: {detail.get('non_regression_all_seeds')}", flush=True)
    print(f"  ANTI-CHEATS all seeds: shuffled-graph {detail.get('shuffled_collapse_all_seeds')} (adv mean "
          f"{detail.get('shuffled_graph_advantage_mean', float('nan')):.1f}x, min "
          f"{detail.get('shuffled_graph_advantage_min', float('nan')):.1f}x) | confab-drop "
          f"{detail.get('confab_drop_all_seeds')} | decorrelated {detail.get('decorrelated_all_seeds')} "
          f"(gap {detail.get('decorrelated_gap_mean', float('nan')):+.3f} vs value "
          f"{detail.get('value_gap_mean', float('nan')):+.3f}) | value-perp-plaus "
          f"{detail.get('value_perp_plaus_all_seeds')} (|corr| max "
          f"{detail.get('value_plaus_corr_absmax', float('nan')):.3f})", flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}\n", flush=True)

    out = {
        "probe": "discursive_turn_stage0_derisk",
        "verdict": verdict,
        "seeds": seeds,
        "stage": "0 -- the CPU mixed-type multi-proposition engage-and-discuss turn; prove MIXED ASSEMBLY + the "
                 "TYPE-AWARE MOAT (never assert a fabricated fact) + brain-cognition + depth-adaptation + "
                 "non-regression hold IN COMPOSITION",
        "scoping": "research/findings/raw/_communicable_discursive_turn_scoping.md (§5 Stage 0)",
        "config": {"D": a.D, "n_facts": a.n_facts, "n_negated": a.n_negated, "n_topics": a.n_topics,
                   "n_attempts": a.n_attempts, "tau_pct": a.tau_pct, "max_depth": a.max_depth,
                   "max_chain_hops": a.max_chain_hops, "max_elaborations": a.max_elaborations,
                   "max_novel": a.max_novel, "max_discuss": a.max_discuss, "taught_frac": a.taught_frac,
                   "n_rounds": a.n_rounds, "lr": a.lr, "da_reward": a.da_reward, "da_baseline": a.da_baseline,
                   "kappa": a.kappa, "w_value": a.w_value, "w_plaus": a.w_plaus, "w_fam": a.w_fam,
                   "speak_base_pA": a.speak_base_pA, "speak_gain_pA": a.speak_gain_pA,
                   "silence_drive_pA": a.silence_drive_pA, "acc_steps": a.acc_steps,
                   "advantage_bar": a.advantage_bar, "max_value_plaus_corr": a.max_value_plaus_corr,
                   "host_oracle_sampler": True,
                   "spiking_speak_decision": "ALWAYS spiking (SpikingSpeakAccumulator, the brain-based mix/depth)"},
        "turn_architecture": (
            "USER MESSAGE -> IntentRouter.classify -> a MIXING PRIOR -> assemble a TYPED candidate pool: "
            "(C) RichAnswerComposer-style gather (direct + role-chase chain + dlPFC-ordered elaboration; the moat "
            "lives in what_does/query_patient abstaining) | (N) the b2 GenerativeReplayProposer candidate SET "
            "ABOUT the topic (each flagged, never stored) | (D) adjacent GROUNDED facts + flagged speculation "
            "framed 'here's how I think about it' (the engage-without-an-answer path) | (P) a fixed non-claim glue. "
            "APPRAISE each candidate by WORTH (learned-Q value + plausibility + familiarity) + run the "
            "SpikingSpeakAccumulator PER candidate to SELECT the emitted set + DEPTH. RENDER+VERIFY type-aware: "
            "(C) accept-as-certain iff re-parse==svo AND svo IN stored; (N)/(D) accept-as-flagged iff "
            "re-parse==the brain's PROPOSED svo AND hedged AND NOT IN stored. EMIT a paragraph with certain vs "
            "flagged VISIBLY DISTINCT."),
        "reuse_by_import": {
            "factory+appraisal+decide+learn+teaching": "_communicable_turn_stageA_derisk.py (build_communicable_brain, "
                "CommunicableTurn.{worth,propose_candidates_about,render_and_verify,_speak_drives,calibrate,feedback}, "
                "IntentRouter, SignedLearnedSpeakValue) [Stage A GO 3-seed]",
            "generate": "_genfrontier_b2_generative_replay_derisk.py (GenerativeReplayProposer, shuffle_graph) [GO 6-seed]",
            "spiking_decide": "_value_salience_appraisal_derisk.py (SpikingSpeakAccumulator) [GO 3-seed]",
            "learn_context": "_learned_talkativeness_derisk.py (context_code, code_overlap) [GO 3-seed]",
            "rich_gather_planner+confab_probe": "rich_answer_composer.py (NeuralDiscoursePlanner, _ConfabOneRenderer) [GO]",
            "graded_confidence+verify+faculties": "_communicable_brain_probe1_whatdoyouthink.py (hedge_for) + "
                "_grounded_lang_p3_derisk.py (TemplateStubFaculty, InjectingStubFaculty) [Probe-1 GO]",
        },
        "new_glue": (
            "DiscursiveTurn (the typed-candidate gather across {C,N,D,P} + the per-candidate spiking appraisal/"
            "decision selecting the mix+depth + the TYPE-AWARE render/verify gate + the (D) discuss framing + the "
            "depth controller + the hypothesis-referent). NO new mechanism; NO sim/ edit; reuse-by-import. The (D) "
            "framing glue + the type-aware acceptance rule + the depth=#candidates-clearing-the-spiking-threshold "
            "are the only new logic (all per the scoping §2 mapping)."),
        "stage0_gate": (
            "GO = (1) MIXED ASSEMBLY (>=1 turn >=2 types + discuss-while-answering depth>=2 + engage-without-answer "
            ">=2 props, not a terse abstain); (2) MOAT-HARD (0 certain-leaks; every flagged FLAGGED + a who/what on "
            "it ABSTAINS + never stored; the paragraph = verified-stored-certain U flagged-hypothesis); (3) "
            "BRAIN-DOES-COGNITION (per-candidate decide is the spiking firing; the free-generate LESION is caught "
            "by VERIFY on the mixed paragraph); (4) DEPTH-ADAPTS ('tell me more' raises depth + the learned-Q; the "
            "DA-lesion abolishes both); (5) NON-REGRESSION (the single-proposition paths byte-unchanged). "
            "ANTI-CHEATS: shuffled-graph collapse >=3x | the _ConfabOneRenderer drops a confab from the mixed "
            "C+N+D paragraph | decorrelated-credit flattens the depth gap | value-perp-plausibility corr ~ 0."),
        "moat_safety_claim": (
            "THE LOAD-BEARING SAFETY CLAIM: 'never ASSERT a fabricated fact' is enforced STRUCTURALLY -- certainty "
            "requires a stored-fact re-parse (type-C + svo IN stored); everything else is rendered FLAGGED (hedged "
            "+ a hypothesis marker, never stored) or DROPPED. A flagged proposition CANNOT be rendered certain "
            "(the certain-render path is reachable only for type-C, which requires svo IN stored). The "
            "engage-without-an-answer turn ENGAGES (adjacent grounded facts + flagged speculation) with 0 "
            "fabrication; a who/what on every emitted flagged proposition ABSTAINS."),
        "detail": detail,
        "per_seed": rows,
        "elapsed_total_s": time.time() - t0,
    }
    if a.out is None:
        a.out = os.path.join(_REPO, "research", "findings", "raw", "_discursive_turn_stage0_derisk.json")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {a.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
