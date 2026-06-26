"""COMMUNICABLE TURN -- STAGE A: the CPU UNIFIED TURN that FUSES the three de-risked communicable-brain
mechanisms (GENERATE / DECIDE-TO-SPEAK / LEARN-TALKATIVENESS) + Probe-1's render+VERIFY into ONE
`CommunicableTurn` orchestrator on ONE CPU brain, and PROVES every safety invariant holds IN COMPOSITION.

This is Stage A of the communicable-brain console integration (scoping
`research/findings/raw/_communicable_brain_console_integration_scoping.md` §4 Stage A). The scoping found that
the three GO mechanisms ALREADY run on the SAME brain object (the b2 PPMI cortex over the 8x8 taxonomy +
`RFPhasorComposer` at D=64 + a `BrainConversationalAgent` + the spiking speak/silence WTA accumulator + the CPU
`TemplateStubFaculty`), so the fusion is a ROUTING SHELL, NOT a new mechanism. Stage A builds that shell and
runs the four turn cases (question / opinion / phatic / teaching) end-to-end, then asserts that each component's
GO REPRODUCES IN COMPOSITION while every load-bearing safety invariant survives.

THE UNIFIED TURN (scoping §1):

    USER MESSAGE -> classify intent {question, opinion, phatic, teaching} (a rule classifier; Stage A is fine
                    with a simple one), then route to one of three CHANNELS + a teaching path:
      - KNOWN-FACT (hard-gated, CERTAIN): the unchanged BrainConversationalAgent.what_does / is_it_true; the
        no-confab moat abstains when nothing matches.  (A bare factual question that the moat abstains on stays
        a clean "I don't know" -- the scoping's PROPOSED DEFAULT: fall-through OFF for bare factual questions.)
      - NOVEL / GENERATIVE (FLAGGED hypothesis, graded): ASSIMILATE(X) -> the b2 generative-replay proposer
        draws a NOVEL, graph-plausible, non-contradictory proposition ABOUT X (each filler a SPIKING draw) ->
        APPRAISE by WORTH = talkativeness * (w_v * VALUE[learned Q] + w_p * plausibility + w_f * familiarity)
        -> the SPIKING speak/silence WTA accumulator decides emit-vs-silent -> RENDER (fluency-only faculty) +
        VERIFY (re-parse the prose -> SVO; require == the brain's SVO) -> EMIT a FLAGGED hypothesis with a
        graded hedge.  NEVER stored.  (An OPINION framing that has no grounded candidate -> the honest "I don't
        really have a view on that"; an OPINION about an unknown word -> "I don't know that word yet".)
      - PHATIC ("hi"): a tiny canned NON-factual reply.  Makes no claim -> no moat risk.
      - TEACHING ("tell me more" / "elaborate" / "stop" / "too much"): the LEARN update -- a perceived
        conversational reward (+) or a perceived negative (-) fires a phasic DA burst/dip -> the three-factor
        rule raises (or lowers) the per-context speak-value Q for that topic (and PPMI-similar contexts).  THEN
        the prior topic is re-run with the updated Q (so "tell me more" immediately elaborates more there).

THE THREE GO PIECES COMPOSED (reuse-by-import, VERBATIM, NO sim/ edit):
  - GENERATE      -- `_genfrontier_b2_generative_replay_derisk.GenerativeReplayProposer` (the spiking soft-WTA
                     draw is default-on); ASSIMILATE + PROPOSE.  [GO 6-seed]
  - DECIDE        -- `_value_salience_appraisal_derisk.SpikingSpeakAccumulator` (a real Izhikevich WTA on a
                     numpy SimulationBridge slice; the speak DECISION is a neural pool's FIRING).  [GO 3-seed]
  - LEARN         -- `_learned_talkativeness_derisk.LearnedSpeakValue` (a per-context speak-value Q updated by
                     the three-factor rule from perceived feedback; non-circular by construction).  [GO 3-seed]
  - RENDER+VERIFY -- Probe-1's `plausibility_score`/`hedge_for` + `_grounded_lang_integration_derisk`'s
                     `_extract_svo_from_prose`/`_build_inflection_map` + `BrainConversationalAgent.parse`.  [GO]
  - KNOWN-FACT + MOAT -- `BrainConversationalAgent.what_does/who_does/is_it_true` + the `RFPhasorComposer`
                     no-confab moat.  [the production known-fact channel, unchanged]

THE ONLY GENUINELY NEW CODE (the integration glue):
  - `IntentRouter.classify(msg)` -- a transparent rule classifier {question, opinion, phatic, teaching}.
  - `SignedLearnedSpeakValue` -- a tiny subclass of LearnedSpeakValue that adds the scoping's signed-NEGATIVE
    feedback extension ("stop"/"too much" -> a DA DIP -> a NEGATIVE RPE -> Q DECREASES for that context).  The
    three-factor rule already handles rpe<0; this just lets a single feedback carry a signed delta.  (No edit to
    the de-risk file; reuse-by-subclass.)
  - `CommunicableTurn` -- the §1 decision logic + the candidate-set propose/appraise (LearnedTalkativenessTurn
    mechanics) + the phatic table + `feedback(signal, topic)`.

THE STAGE A GATE (>=3 seeds; promote to 6 if GO).  GO = the fused turn keeps EVERY invariant:
  (1) EACH COMPONENT'S GO REPRODUCES IN COMPOSITION:
        - KNOWN-FACT recall works (a stored cue answers CERTAIN);
        - the NOVEL channel emits GENERATED + GROUNDED + FLAGGED propositions (novel, topic-relevant, the
          shuffled-PPMI-graph control collapses groundedness >= 3x);
        - the value/talkativeness appraisal SPEAKS-MORE-WHERE-TAUGHT (post-teaching the brain emits on MORE
          topics than the pre-teaching baseline, and the taught speak-rate > untaught);
        - the LEARNING curve rises monotonically on the taught bin.
  (2) 0 MOAT LEAKS ACROSS THE WHOLE TURN (HARD): the generative channel NEVER calls store; the known-fact
      channel is hard-gated; a never-stored cue ABSTAINS-or-FLAGS, NEVER asserts as certain.  Every novel
      emission is FLAGGED.  Stored facts still answer.
  (3) A FEEDBACK ROUND ('elaborate' on a context) RAISES the next-turn talkativeness THERE (the teaching loop
      closes end-to-end); the negative ('stop') extension LOWERS Q there.
  (4) THE DA-LESION ABOLISHES the talkativeness change + the value-driven extra speaking (pin DA during
      learning -> dQ=0 -> no extra emissions; the composition is the BRAIN's reward system, not a host counter).

ANTI-CHEATS:
  - SHUFFLED-PPMI-GRAPH: the novel-channel groundedness collapses (>= 3x advantage of true-graph over shuffled).
  - FREE-GENERATE LESION: sever the brain's proposal -> let the faculty free-generate the content -> VERIFY
    REJECTS it (the content is the BRAIN's, not the LLM's).  Run on the composed turn's emitted propositions.
  - NON-CIRCULARITY: corr(learned Q, plausibility) ~ 0 (value is a separate axis, not relabeled plausibility).
  - DECORRELATED-CREDIT: the SAME total DA but the eligibility shuffled across contexts -> a FLAT global rise,
    no taught/untaught gap (the gap is per-context learning, not global vigor).

HONEST: if composing the three REGRESSES any invariant (a leak, a lost gate, the lesion not abolishing), this
reports it PRECISELY in the JSON + the FINAL MESSAGE -- it does NOT fake a GO.  NEVER weakens the moat.

CPU (`SIM_BACKEND=numpy`); reuse-by-import; NO `sim/` edit.  Run:
  SIM_BACKEND=numpy python -u -m research.runners._communicable_turn_stageA_derisk \
      --seeds 42,43,44 --out research/findings/raw/_communicable_turn_stageA_derisk.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict

# the whole pipeline is the numpy-CPU brain (PPMI cortex + RF composer + parser + a spiking WTA accumulator slice).
os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# --- the b2 generative-replay machinery (the GO PROPOSE piece) -- reused VERBATIM ---
from research.runners._genfrontier_b2_generative_replay_derisk import (  # noqa: E402
    GenerativeReplayProposer,
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
# --- the RF composer + the no-confab moat (the brain's KNOWN-fact store) ---
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402
# --- the gate->constrain->VERIFY loop (the GO RENDER+VERIFY piece) ---
from research.runners._grounded_lang_p3_derisk import (  # noqa: E402
    TemplateStubFaculty,
    InjectingStubFaculty,
)
from research.runners._grounded_lang_integration_derisk import (  # noqa: E402
    _build_inflection_map,
    _extract_svo_from_prose,
)
from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
# --- Probe-1's graded-confidence read-out (the SAME plausibility->hedge mapping) -- reused VERBATIM ---
from research.runners._communicable_brain_probe1_whatdoyouthink import (  # noqa: E402
    plausibility_score,
    hedge_for,
)
# --- the SPIKING speak/silence WTA accumulator + the value helpers (reused VERBATIM) ---
from research.runners._value_salience_appraisal_derisk import (  # noqa: E402
    SpikingSpeakAccumulator,
)
# --- the LEARNED talkativeness Q + the context code (reused VERBATIM) ---
from research.runners._learned_talkativeness_derisk import (  # noqa: E402
    LearnedSpeakValue,
    context_code,
    code_overlap,
)


# ===========================================================================
# GLUE #1 -- the INTENT ROUTER.  A transparent rule classifier {question, opinion, phatic, teaching}.  Stage A
# only needs it to be transparent + correct on the test phrasings; the production console router (the scoping
# §3.4 QuestionRouter extension) is a later stage.  We pre-tag each test message with its intent so the
# classifier's job is verifiable, AND we expose `classify(msg)` so the orchestrator routes on the message text.
# ===========================================================================
_PHATIC_RE = re.compile(r"^\s*(hi|hey|hello|yo|howdy|how are you|how's it going|good morning|good evening)\b",
                        re.IGNORECASE)
_OPINION_RE = re.compile(r"\b(what do you think|your (view|opinion|take)|do you think|tell me about|"
                         r"thoughts on|how about|what about)\b", re.IGNORECASE)
_TEACH_MORE_RE = re.compile(r"\b(tell me more|elaborate|go on|say more|more about|keep going)\b", re.IGNORECASE)
_TEACH_STOP_RE = re.compile(r"\b(stop|that's enough|thats enough|too much|enough|less)\b", re.IGNORECASE)
# a who/what/yes-no QUESTION about a known cue (the existing known-fact router patterns)
_QUESTION_RE = re.compile(r"^\s*(what does|who does|is it true|does|do|what|who|is)\b", re.IGNORECASE)


class IntentRouter:
    """Classify a user message into one of {question, opinion, phatic, teaching}.  Teaching is split into a
    polarity (+1 elaborate / -1 stop).  The order matters: teaching + phatic are checked before the generic
    question pattern (a teaching signal or a greeting is not a content question)."""

    def classify(self, msg):
        m = msg.strip()
        if _TEACH_MORE_RE.search(m):
            return {"intent": "teaching", "polarity": +1}
        if _TEACH_STOP_RE.search(m):
            return {"intent": "teaching", "polarity": -1}
        if _PHATIC_RE.search(m):
            return {"intent": "phatic", "polarity": 0}
        if _OPINION_RE.search(m):
            return {"intent": "opinion", "polarity": 0}
        if _QUESTION_RE.search(m):
            return {"intent": "question", "polarity": 0}
        # default: treat a bare topic mention as an opinion request (the chatty default) -- but a bare factual
        # cue tuple is handled by the orchestrator's structured-question entry, not the free-text classifier.
        return {"intent": "opinion", "polarity": 0}


# ===========================================================================
# GLUE #2 -- the SIGNED learned speak-value.  The scoping §3.3 reward-policy needs a NEGATIVE feedback arm
# ("stop"/"too much" -> a DA DIP -> a negative RPE -> Q DECREASES) in addition to the positive ("elaborate")
# arm the de-risk modeled.  The three-factor rule already handles rpe<0; this subclass just lets a single
# feedback carry a SIGNED delta (a +burst or a -dip), and clamps Q at >= 0 (a context can be silenced back to
# its no-preference baseline but not driven negative-talkative -- a floor, not a new mechanism).  Reuse-by-
# subclass; NO edit to the de-risk file, NO sim/ edit.
# ===========================================================================
class SignedLearnedSpeakValue(LearnedSpeakValue):
    """LearnedSpeakValue + a signed feedback path.  `feedback(active_topic, polarity)` fires a phasic DA burst
    (polarity=+1, the 'elaborate' reward) or a DA dip (polarity=-1, the 'stop' negative) and runs the SAME
    three-factor update (dQ = lr * (DA - baseline) * eligibility[c'|active]); Q is floored at 0."""

    def __init__(self, *args, da_punish=None, **kwargs):
        super().__init__(*args, **kwargs)
        # the DA dip magnitude on a negative feedback -- symmetric to the reward burst by default.
        self.da_punish = float(da_punish) if da_punish is not None else self.da_reward

    def feedback(self, active_topic, polarity, lesion_DA=False, decorrelate=False):
        """One PERCEIVED-feedback round on `active_topic`.  polarity=+1 -> 'elaborate' (DA burst); polarity=-1 ->
        'stop' (DA dip); polarity=0 -> neutral (no update).  Mirrors `update()` but the DA is signed.  The
        DA-LESION pins DA to baseline (no burst/dip ever) -> dQ=0 -> no learning."""
        if polarity == 0:
            return
        if lesion_DA:
            return                                   # the SNc lesion: no phasic DA -> no learning, ever
        da = (self.da_baseline + self.da_reward) if polarity > 0 else (self.da_baseline - self.da_punish)
        rpe = da - self.da_baseline                  # the three-factor reward factor (signed)
        if rpe == 0.0:
            return
        credit_ctx = active_topic
        if decorrelate:
            credit_ctx = self.topics[int(self.rng.integers(len(self.topics)))]
        for tgt in self.topics:
            self.Q[tgt] += self.lr * rpe * self.eligibility(credit_ctx, tgt)
            if self.Q[tgt] < 0.0:
                self.Q[tgt] = 0.0                    # floor: silence back to no-preference, never negative


# ===========================================================================
# GLUE #3 -- the CommunicableTurn orchestrator.  Holds the brain (composer + agent + proposer + accumulator +
# the learned-Q value system) and the router, and exposes the unified per-message turn + the feedback path.
# The candidate-propose/appraise mechanics are the LearnedTalkativenessTurn / AppraisalTurn mechanics, inlined
# here so the single class IS the production-grade fusion (per scoping §3.1).
# ===========================================================================
PHATIC_REPLIES = {
    # a small fixed social repertoire -- makes NO factual claim, so no moat risk (scoping §1 case 3).
    "default": "Hi! Ask me about what I know, or what I think about something.",
    "how_are_you": "I'm doing well, thanks. Ask me about what I know, or what I think about something.",
}


class CommunicableTurn:
    """The fused communicable turn on ONE CPU brain.  Routes a user message to the known-fact / novel-generative
    / phatic channel (or the teaching path), with the spiking speak decision + the learned talkativeness + the
    no-confab moat all composed.

    Channels (scoping §1):
      - KNOWN-FACT  : agent.what_does / is_it_true (hard-gated, CERTAIN; abstains -> "I don't know about that").
      - NOVEL       : propose candidate set ABOUT X -> appraise by WORTH (learned Q + plausibility + familiarity)
                      -> spiking speak decision -> render+VERIFY -> EMIT a FLAGGED hypothesis (NOT stored).
      - PHATIC      : a canned non-factual reply.
      - TEACHING    : feedback() updates the learned Q (+ re-runs the prior topic with the new Q).
    """

    def __init__(self, comp, agent, proposer, accumulator, P, row, vocab_sets, faculty, value, codes,
                 full_pools=None, w_value=0.5, w_plaus=0.35, w_fam=0.15,
                 speak_base_pA=70.0, speak_gain_pA=180.0, silence_drive_pA=150.0,
                 fall_through_opinion=True, fall_through_question=False, cand_cache=None):
        self.comp = comp                              # the RF composer (KNOWN-fact store + the moat)
        self.agent = agent                            # BrainConversationalAgent (.what_does/.is_it_true/.parse)
        self.proposer = proposer                      # the b2 generative-replay proposer (spiking draw)
        self.acc = accumulator                        # the spiking speak/silence WTA accumulator
        self.P, self.row = P, row
        self.agents_set, self.actions_set, self.patients_set, self.inflect = vocab_sets
        self.faculty = faculty                        # the fluency-only faculty (CPU stand-in)
        self.value = value                            # the LEARNED speak-value (SignedLearnedSpeakValue)
        self.codes = codes                            # the per-topic context codes
        self.router = IntentRouter()
        self.w_value, self.w_plaus, self.w_fam = float(w_value), float(w_plaus), float(w_fam)
        self.speak_base_pA, self.speak_gain_pA = float(speak_base_pA), float(speak_gain_pA)
        self.silence_drive_pA = float(silence_drive_pA)
        # the scoping's PROPOSED DEFAULTS for the known-fact-abstain -> generative fall-through (owner-steer #1):
        # ON for opinion framings, OFF for bare factual questions.
        self.fall_through_opinion = bool(fall_through_opinion)
        self.fall_through_question = bool(fall_through_question)
        fa, fac, fp = full_pools if full_pools else (set(proposer.agents), set(proposer.actions),
                                                     set(proposer.patients))
        self.full_agents, self.full_actions, self.full_patients = set(fa), set(fac), set(fp)
        self._fa_list = sorted(self.full_agents)
        self._fac_list = sorted(self.full_actions)
        self._fp_list = sorted(self.full_patients)
        # per-seed normalisers (set by calibrate())
        self._plaus_lo = self._plaus_hi = None
        self._val_lo = self._val_hi = None
        self._fam_lo = self._fam_hi = None
        self._conf_lo = self._conf_hi = 0.0, 1.0
        self._last_topic = None                       # for the "tell me more" re-run
        # per-topic candidate-set CACHE: propose_candidates_about is deterministic-enough per topic and its
        # non-contradiction gate runs a resonate (composer.ask_yes_no) per candidate, so re-proposing the SAME
        # topic across calibrate + the value/pre/lesion/decorr arms repeats that cost N-fold. Caching the
        # candidate SET per topic (the SET is independent of the learned Q -- only the worth RANKING + the speak
        # DRIVE depend on Q) makes the fused turn tractable on CPU with NO change to any decision. The cache is
        # keyed by topic.  A SHARED cache (passed in) lets the scratch + the value/pre/lesion/decorr arms reuse
        # one another's proposals (they all use the SAME proposer + graph -> the same candidate SET per topic).
        self._cand_cache = cand_cache if cand_cache is not None else {}

    # -------------------- ASSIMILATE (Probe-1) --------------------
    def assimilate(self, topic):
        if topic not in self.row:
            return {"in_graph": False, "related_actions": [], "related_patients": []}
        ti = self.row[topic]
        rel_ac = sorted(self.proposer.actions, key=lambda w: -self.P[ti, self.row[w]])
        rel_pt = sorted(self.proposer.patients, key=lambda w: -self.P[ti, self.row[w]])
        return {"in_graph": True,
                "related_actions": [(w, round(float(self.P[ti, self.row[w]]), 3)) for w in rel_ac[:4]],
                "related_patients": [(w, round(float(self.P[ti, self.row[w]]), 3)) for w in rel_pt[:4]]}

    # -------------------- the FAMILIARITY axis (Probe-1/appraisal) --------------------
    def familiarity(self, triple):
        a, ac, p = triple
        return (float(self.P[self.row[a], self.row[ac]]) + float(self.P[self.row[ac], self.row[p]])
                + float(self.P[self.row[a], self.row[p]])) / 3.0

    # -------------------- PROPOSE the CANDIDATE SET about X (appraisal de-risk mechanic) --------------------
    def propose_candidates_about(self, topic, n_attempts=500):
        """All distinct topic-relevant, NOVEL, graph-plausible, non-contradictory candidate triples about X, each
        with its plausibility.  Empty = the honest 'no graph-supported candidate'."""
        if topic not in self.row:
            return []
        if topic in self._cand_cache:
            return self._cand_cache[topic]
        topic_is_agent = topic in self.full_agents
        topic_is_patient = topic in self.full_patients
        if not (topic_is_agent or topic_is_patient):
            self._cand_cache[topic] = []
            return []
        seen = {}
        rejected = set()                              # triples already gate-checked (don't re-run the resonate)
        for _ in range(n_attempts):
            if topic_is_agent:
                a = topic
                ac = self.proposer._sample_weighted(
                    self._fac_list, self.proposer._weight_partner((a,), self._fac_list))
                p = self.proposer._sample_weighted(
                    self._fp_list, self.proposer._weight_partner((a, ac), self._fp_list))
            else:
                p = topic
                a = self.proposer._sample_weighted(
                    self._fa_list, self.proposer._weight_partner((p,), self._fa_list))
                ac = self.proposer._sample_weighted(
                    self._fac_list, self.proposer._weight_partner((a, p), self._fac_list))
            triple = (a, ac, p)
            if topic not in triple or triple in seen or triple in rejected:
                continue                              # already accepted or gate-rejected -> skip (no re-resonate)
            if triple in self.proposer.all_stored:
                rejected.add(triple); continue
            if not self.proposer._plausible(a, ac, p):
                rejected.add(triple); continue
            # _contradicts runs a composer resonate (ask_yes_no) -> only reached for novel, plausible, NEW triples
            if self.proposer._contradicts(a, ac, p):
                rejected.add(triple); continue
            seen[triple] = plausibility_score(self.P, self.row, a, ac, p)
            # Stage-0 latency: stop once we have enough accepted candidates. The DiscursiveTurn only consumes the
            # top few (max_discuss/max_novel), so enumerating ALL distinct plausible triples just burns
            # _contradicts composer-resonates. `_cand_cap`=None (default) keeps the original exhaustive behavior;
            # each kept candidate is still VERIFY-gated downstream, so the no-confab moat is unaffected.
            _cap = getattr(self, "_cand_cap", None)
            if _cap and len(seen) >= _cap:
                break
        out = sorted(seen.items(), key=lambda kv: -kv[1])
        self._cand_cache[topic] = out
        return out

    # -------------------- the WORTH appraisal (learned-talkativeness mechanic) --------------------
    def _norm(self, x, lo, hi):
        if lo is None or hi is None or hi <= lo:
            return 0.5
        return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))

    def worth(self, topic, triple):
        """WORTH = w_v * VALUE[learned Q for topic] + w_p * plausibility + w_f * familiarity (each axis
        normalised per-seed).  The LEARNED Q is the value axis (incentive salience / how readily the brain
        volunteers a view on this context)."""
        plaus = plausibility_score(self.P, self.row, *triple)
        val = self.value.value(topic)
        fam = self.familiarity(triple)
        pn = self._norm(plaus, self._plaus_lo, self._plaus_hi)
        vn = self._norm(val, self._val_lo, self._val_hi)
        fn = self._norm(fam, self._fam_lo, self._fam_hi)
        return self.w_value * vn + self.w_plaus * pn + self.w_fam * fn, (pn, vn, fn)

    def _speak_drives(self, push):
        """The additive incentive-salience speak drive (the learned-talkativeness mechanic): the speak pool gets
        an additive push from (value + plausibility + familiarity); the silence pool has a fixed reticence."""
        speak = self.speak_base_pA + self.speak_gain_pA * push
        return float(speak), float(self.silence_drive_pA)

    # -------------------- RENDER + VERIFY (Probe-1 contract) --------------------
    def render_and_verify(self, triple, faculty, faculty_mode="grounded"):
        a, v, p = triple
        surface, asserted = faculty.render_svo(a, v, p)
        csvo = _extract_svo_from_prose(surface, self.agents_set, self.actions_set, self.patients_set, self.inflect)
        if csvo is None:
            return {"surface": surface, "asserted_svo": asserted, "reparse_svo": None, "verified": False}
        parsed = self.agent.parse(csvo, voice="active")
        rsvo = [parsed.get("agent"), parsed.get("action"), parsed.get("patient")]
        return {"surface": surface, "asserted_svo": asserted, "reparse_svo": rsvo,
                "verified": (rsvo == list(triple))}

    # ==================== the THREE CHANNELS ====================
    def _known_fact_channel(self, cue):
        """The hard-gated KNOWN-fact channel: cue = (agent, action) for a what-does, or (agent, action, patient)
        for a yes/no.  Returns a structured record; abstains (the no-confab moat) when nothing matches."""
        if len(cue) == 2:
            ag, ac = cue
            patient = self.agent.what_does(ag, ac)
            if patient is None:
                return {"channel": "known", "certain": True, "abstained": True,
                        "answer": "I don't know about that.", "recalled_svo": None}
            return {"channel": "known", "certain": True, "abstained": False,
                    "answer": f"{ag} {ac} {patient}.", "recalled_svo": [ag, ac, patient]}
        else:
            ag, ac, pt = cue
            yn = self.agent.is_it_true(ag, ac, pt)
            if yn == "unknown":
                return {"channel": "known", "certain": True, "abstained": True,
                        "answer": "I don't know about that.", "recalled_svo": None, "yesno": "unknown"}
            return {"channel": "known", "certain": True, "abstained": False,
                    "answer": ("Yes." if yn == "yes" else "No."), "recalled_svo": [ag, ac, pt], "yesno": yn}

    def _novel_channel(self, topic, n_attempts=500):
        """The NOVEL / generative channel: propose candidates about X -> appraise by worth -> spiking speak
        decision -> render+VERIFY -> EMIT a FLAGGED hypothesis (NOT stored).  Returns a structured record."""
        if topic not in self.row:
            return {"channel": "novel", "certain": False, "emitted": False, "abstained_opinion": True,
                    "reply": "I don't know that word yet.", "proposed_triple": None, "unknown_word": True}
        cands = self.propose_candidates_about(topic, n_attempts=n_attempts)
        if not cands:
            return {"channel": "novel", "certain": False, "emitted": False, "abstained_opinion": True,
                    "reply": "I don't really have a view on that.", "proposed_triple": None}
        # APPRAISE + RANK by worth (the committed candidate = highest worth)
        ranked = sorted(cands, key=lambda tp: -self.worth(topic, tp[0])[0])
        best_triple, best_plaus = ranked[0]
        w, (pn, vn, fn) = self.worth(topic, best_triple)
        speak_pA, silence_pA = self._speak_drives(self.w_value * vn + self.w_plaus * pn + self.w_fam * fn)
        is_speak, sp_spk, si_spk, margin = self.acc.decide(speak_pA, silence_pA)
        rv = self.render_and_verify(best_triple, self.faculty, "grounded") if is_speak else None
        # the graded-confidence hedge tracks the committed candidate's WORTH
        hedge, conf = hedge_for(w, self._conf_lo, self._conf_hi)
        emitted = bool(is_speak and rv is not None and rv["verified"])
        rec = {"channel": "novel", "certain": False, "topic": topic,
               "proposed_triple": list(best_triple), "topic_in_proposition": (topic in best_triple),
               "plausibility": round(best_plaus, 4), "worth": round(float(w), 4),
               "speak_decision": bool(is_speak), "speak_spikes": sp_spk, "silence_spikes": si_spk,
               "decision_margin": margin, "verified": bool(rv and rv["verified"]),
               "emitted": emitted, "hedge": hedge, "confidence": round(conf, 3),
               "abstained_opinion": False}
        if emitted:
            rec["reply"] = f"{hedge} {' '.join(best_triple)}."
        elif not is_speak:
            rec["reply"] = None
            rec["silence_reason"] = "spiking accumulator chose SILENCE"
        else:
            rec["reply"] = None
            rec["silence_reason"] = "render/VERIFY rejected"
        return rec

    def _phatic_channel(self, msg):
        key = "how_are_you" if re.search(r"how are you|how's it going", msg, re.IGNORECASE) else "default"
        return {"channel": "phatic", "certain": False, "emitted": True, "abstained": False,
                "reply": PHATIC_REPLIES[key], "is_factual_claim": False}

    # ==================== the unified per-message turn ====================
    def turn(self, msg, cue=None, topic=None, n_attempts=500):
        """Run the full unified turn for a user message.

        `cue`   -- a (agent, action) or (agent, action, patient) tuple for a structured KNOWN-fact question
                   (the orchestrator supplies it; a real console parses it from the message text).
        `topic` -- the OPINION topic (the entity the view is about); for an opinion the orchestrator supplies it.
        For phatic/teaching, neither is needed.  Returns a structured channel record."""
        intent = self.router.classify(msg)
        rec = {"message": msg, "intent": intent["intent"]}

        if intent["intent"] == "phatic":
            rec.update(self._phatic_channel(msg))
            return rec

        if intent["intent"] == "teaching":
            # the teaching path is handled by feedback(); the turn just acknowledges + re-runs the prior topic.
            tgt = topic if topic is not None else self._last_topic
            self.feedback(tgt, intent["polarity"])
            ack = ("Noted -- I'll say more about that." if intent["polarity"] > 0
                   else "Okay, I'll hold back on that.")
            rec.update({"channel": "teaching", "certain": False, "emitted": False, "polarity": intent["polarity"],
                        "feedback_topic": tgt, "reply": ack})
            if tgt is not None and intent["polarity"] > 0:
                # immediately elaborate more on the taught topic with the updated Q (the scoping's "tell me more
                # immediately elaborates").
                rec["rerun"] = self._novel_channel(tgt, n_attempts=n_attempts)
            return rec

        if intent["intent"] == "question":
            # the KNOWN-fact channel (hard-gated, CERTAIN).
            assert cue is not None, "a structured question needs a cue tuple"
            kf = self._known_fact_channel(cue)
            rec.update(kf)
            # fall-through (owner-steer #1, default OFF for bare factual questions): if the moat abstained AND
            # fall-through is enabled, offer a flagged guess from the novel channel.
            if kf["abstained"] and self.fall_through_question and topic is not None:
                nv = self._novel_channel(topic, n_attempts=n_attempts)
                rec["fall_through"] = nv
                if nv.get("emitted"):
                    rec["reply"] = "I wasn't told, but " + nv["reply"][0].lower() + nv["reply"][1:]
            return rec

        # OPINION -> the novel/generative channel.
        assert topic is not None, "an opinion needs a topic"
        self._last_topic = topic
        nv = self._novel_channel(topic, n_attempts=n_attempts)
        rec.update(nv)
        return rec

    # ==================== the TEACHING / feedback path (the LEARN update) ====================
    def feedback(self, topic, polarity, lesion_DA=False, decorrelate=False):
        """Deliver a perceived conversational feedback on `topic`: polarity=+1 ('elaborate' -> a DA burst raises
        Q there + at PPMI-similar contexts), polarity=-1 ('stop' -> a DA dip lowers Q), polarity=0 (neutral).
        The DA-LESION pins DA to baseline (no learning).  This is the brain's reward-modulated three-factor
        plasticity (NOT a host counter)."""
        if topic is None:
            return
        self.value.feedback(topic, polarity, lesion_DA=lesion_DA, decorrelate=decorrelate)

    # ==================== per-seed calibration of the normalisers + hedge bands ====================
    def calibrate(self, topics, n_attempts=500):
        """Pre-pass: set the per-seed [lo,hi] normalisers for {plausibility, value, familiarity} and the
        worth-based hedge band, over the topic population (the appraisal/learned mechanic).  Called AFTER the Q
        is learned so the value normaliser reflects the learned range."""
        plaus_v, val_v, fam_v = [], [], []
        for t in topics:
            cands = self.propose_candidates_about(t, n_attempts=n_attempts)
            if not cands:
                continue
            for tp, pl in cands:
                plaus_v.append(pl)
                fam_v.append(self.familiarity(tp))
            val_v.append(self.value.value(t))
        if plaus_v:
            self._plaus_lo, self._plaus_hi = float(min(plaus_v)), float(max(plaus_v))
            self._fam_lo, self._fam_hi = float(min(fam_v)), float(max(fam_v))
        if val_v and max(val_v) > min(val_v):
            self._val_lo, self._val_hi = float(min(val_v)), float(max(val_v))
        else:
            self._val_lo, self._val_hi = 0.0, 1.0
        # worth population over the per-topic best candidate (for the hedge band)
        worths = []
        for t in topics:
            cs = self.propose_candidates_about(t, n_attempts=n_attempts)
            if not cs:
                continue
            worths.append(max(self.worth(t, tp)[0] for tp, _pl in cs))
        self._conf_lo = float(min(worths)) if worths else 0.0
        self._conf_hi = float(max(worths)) if worths else 1.0
        return self._conf_lo, self._conf_hi


# ===========================================================================
# STAGE B FACTORY -- assemble the WHOLE communicable brain (the same construction run_seed does inline) into ONE
# `CommunicableTurn`, so the PRODUCTION agent can ATTACH it without duplicating the mechanism (the agent wire-in
# `BrainConversationalAgent.enable_communicable_mode` / the `communicable_mode=True` constructor flag call this).
# Pure ADDITION: run_seed is unchanged; this just hoists its brain-assembly into a reusable builder that returns
# the same objects the gate exercises (composer / agent / proposer / accumulator / value / the CommunicableTurn).
# NO sim/ edit; reuse-by-import VERBATIM.
# ===========================================================================
_CORPUS_CACHE = {}


def _default_corpus(vocab, cat_ids, *, window=5, repeat_cap=40, max_bytes=4_000_000):
    """Build (or reuse a cached) PPMI co-occurrence corpus over the 8x8-taxonomy vocab from the project's
    TinyStories corpus -- the SAME `build_real_cooccurrence` Stage A uses. Cached per (window, repeat_cap,
    max_bytes) so attaching communicable-mode to several agents in one process pays the corpus pass once."""
    key = (window, repeat_cap, max_bytes)
    if key not in _CORPUS_CACHE:
        corpus_path = os.path.join(_REPO, "data", "corpus", "tinystories.txt")
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"corpus not found: {corpus_path} (needed to build the communicable PPMI graph)")
        _CORPUS_CACHE[key] = build_real_cooccurrence(corpus_path, vocab, cat_ids, window=window,
                                                     repeat_cap=repeat_cap, seed=42, max_bytes=max_bytes,
                                                     freq_floor=30, min_facts_per_category=20, verbose=False)
    return _CORPUS_CACHE[key]


def build_communicable_brain(seed=42, *, D=256, n_facts=24, n_negated=12, n_attempts=500, tau_pct=50.0,
                             lr=0.10, da_reward=1.0, da_baseline=0.0, kappa=2.0,
                             w_value=0.5, w_plaus=0.35, w_fam=0.15,
                             speak_base_pA=70.0, speak_gain_pA=180.0, silence_drive_pA=150.0,
                             acc_steps=120, host_oracle_sampler=False,
                             composer=None, bc_agent=None, accumulator=None,
                             stored_facts=None, speak_value_Q=None, corpus=None):
    """Assemble the full communicable brain into a `CommunicableTurn` (the Stage A fusion), reusing every GO piece
    VERBATIM. Returns a dict with the turn + every component (so a caller can persist the value Q or inspect).

    The DRAW selector (scoping/owner-steer #3): `host_oracle_sampler` picks the generative draw --
      - False (DEFAULT, the production path): the validated SPIKING soft-WTA generative draw (each filler a
        spiking Izhikevich-bank sample of the brain's PPMI likelihood). ~40s/topic in the fused turn on CPU (the
        per-candidate spiking draw x ~500 attempts); the megakernel perf lever is the Stage-C fix.
      - True (the fast-interactive / numpy-CPU / test oracle): the HOST sample from the SAME PPMI likelihood
        (`_weight_partner`); answer-distribution-identical, fast. The load-bearing SPIKING speak DECISION (the
        SpikingSpeakAccumulator) stays spiking either way -- the brain-based speak choice is unchanged.

    `composer`/`bc_agent`/`accumulator` (optional) let a caller SHARE its already-built brain objects (e.g. the
    production agent passes ITS composer + itself, so the known-fact channel reads the agent's OWN facts + the
    moat). When `composer` is given, `stored_facts` is ignored (the composer already holds the agent's facts).
    `speak_value_Q` (optional) seeds the LEARNED talkativeness Q (a {topic: float} dict, e.g. restored from a
    bundle) so the talkativeness learned across sessions carries forward."""
    rng = np.random.default_rng(seed)
    agents, actions, patients = _category_pools(TAXONOMY_8x8)
    vocab, cat_ids, _cat_names = taxonomy_to_vocab_categories(TAXONOMY_8x8)
    if corpus is None:
        corpus = _default_corpus(vocab, cat_ids)
    P, row = build_plausibility(corpus, vocab)
    pos = P[P > 0]
    tau = float(np.percentile(pos, tau_pct)) if pos.size else 0.0

    # the KNOWN-fact store + the moat: reuse the caller's composer/agent when given (so the channel reads the
    # agent's OWN facts); else build a self-contained communicable brain (the Stage A construction).
    if composer is None:
        comp = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
        if stored_facts is None:
            affirmed, negated, _ = build_stored_facts(agents, actions, patients, P, row, tau, n_facts, n_negated, rng)
        else:
            affirmed = [tuple(f) for f in stored_facts.get("affirmed", [])]
            negated = [tuple(f) for f in stored_facts.get("negated", [])]
        for ag, ac, pt in affirmed:
            comp.store(ag, ac, pt, polarity="AFFIRM")
        for ag, ac, pt in negated:
            comp.store(ag, ac, pt, polarity="NEGATE")
    else:
        comp = composer
        # derive affirmed/negated from the composer's OWN kb (the agent's facts) so the proposer's non-contradiction
        # gate + the all_stored novelty filter use the real stored set.
        affirmed, negated = [], []
        for fact, _h in getattr(comp, "kb", []):
            a_, v_, p_ = fact.get("agent"), fact.get("action"), fact.get("patient")
            if not (isinstance(a_, str) and isinstance(v_, str) and isinstance(p_, str)):
                continue  # clause / attributed patients are not flat SVO triples -> skip for the proposer gate
            (negated if fact.get("polarity") == "NEGATE" else affirmed).append((a_, v_, p_))
    all_stored = set(affirmed) | set(negated)

    if bc_agent is None:
        bc_agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab},
                                            composer=comp, composer_kind="rf", enable_neural_render=False)
    proposer = GenerativeReplayProposer(comp, affirmed, negated, P, row, tau,
                                        np.random.default_rng(seed * 7 + 1),
                                        use_spiking_sampler=(not host_oracle_sampler))

    agents_set, actions_set, patients_set = set(agents), set(actions), set(patients)
    inflect = _build_inflection_map(sorted(actions_set))
    vocab_sets = (agents_set, actions_set, patients_set, inflect)
    grounded_faculty = TemplateStubFaculty()
    full_pools = (set(agents), set(actions), set(patients))

    if accumulator is None:
        accumulator = SpikingSpeakAccumulator(seed=12345, n_steps=acc_steps)

    # the talkativeness arena = the held-out grounded topics (words NOT the agent of any stored fact). The LEARNED
    # speak-value Q is over this topic set; `speak_value_Q` (optional) restores it from a bundle.
    stored_agents = {f[0] for f in affirmed}
    topic_pool = [w for w in (agents + patients) if w not in stored_agents]
    codes = {w: context_code(P, row, w) for w in topic_pool}
    value = SignedLearnedSpeakValue(topic_pool, codes, lr=lr, da_reward=da_reward, da_baseline=da_baseline,
                                    kappa=kappa, da_punish=da_reward, rng=np.random.default_rng(seed * 211 + 3))
    if speak_value_Q:
        for t, q in speak_value_Q.items():
            if t in value.Q:
                value.Q[t] = float(q)

    turn = CommunicableTurn(comp, bc_agent, proposer, accumulator, P, row, vocab_sets, grounded_faculty,
                            value, codes, full_pools=full_pools, w_value=w_value, w_plaus=w_plaus, w_fam=w_fam,
                            speak_base_pA=speak_base_pA, speak_gain_pA=speak_gain_pA,
                            silence_drive_pA=silence_drive_pA)
    return {"turn": turn, "composer": comp, "agent": bc_agent, "proposer": proposer,
            "accumulator": accumulator, "value": value, "P": P, "row": row, "codes": codes,
            "topic_pool": topic_pool, "affirmed": affirmed, "negated": negated,
            "host_oracle_sampler": bool(host_oracle_sampler)}


# ===========================================================================
# Per-seed run: build the ONE brain, LEARN the talkativeness over taught/untaught feedback, then exercise the
# FUSED turn across all four cases + measure the Stage A gate IN COMPOSITION.
# ===========================================================================
def run_seed(seed, vocab, corpus, a, accumulator):
    rng = np.random.default_rng(seed)
    agents, actions, patients = _category_pools(TAXONOMY_8x8)
    P, row = build_plausibility(corpus, vocab)
    pos = P[P > 0]
    tau = float(np.percentile(pos, a.tau_pct)) if pos.size else 0.0

    affirmed, negated, _ = build_stored_facts(agents, actions, patients, P, row, tau,
                                              a.n_facts, a.n_negated, rng)
    all_stored = set(affirmed) | set(negated)

    # the ONE brain: KNOWN-fact store (RF composer; the no-confab moat) + the agent + the proposer.
    comp = RFPhasorComposer(seed=seed, D=a.D, vocab=vocab)
    for ag, ac, pt in affirmed:
        comp.store(ag, ac, pt, polarity="AFFIRM")
    for ag, ac, pt in negated:
        comp.store(ag, ac, pt, polarity="NEGATE")
    # enable_neural_render=False: the agent's NeuralSerialOrderRenderer (an 18432-neuron serial-order word-ORDER
    # pool) is irrelevant to the Stage A invariants (moat / channels / learning) -- the novel channel's surface
    # word order is the SVO frame (' '.join(triple)), and the known-fact channel reads flat facts (no clause
    # ordering). Skipping its heavy CPU build keeps the known-fact + VERIFY behaviour byte-identical to the
    # de-risks (which carry it only for describe()/clause ordering, never exercised here). NO behaviour change to
    # what_does / is_it_true / parse (those don't touch _neural_render for flat facts).
    bc_agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab},
                                        composer=comp, composer_kind="rf", enable_neural_render=False)
    # use_spiking_sampler=False (the documented numpy-CPU / test / reproducibility oracle): the generative DRAW is
    # the HOST sample from the brain's SAME PPMI likelihood (`_weight_partner`), not the spiking soft-WTA bank.
    # Stage A composes the CHANNELS + proves the SAFETY INVARIANTS; the spiking DRAW being the brain's generative
    # act is SEPARATELY validated GO 6-seed (_followon2_spiking_wta_sampler_derisk) and is the DEFAULT-ON
    # production path. On this CPU de-risk the fused turn runs MANY proposes (the scratch grounded-filter over all
    # topics + calibrate x3 + the 4 appraisal arms x n_topics); each spiking draw steps an Izhikevich WTA bank
    # (~15ms), so the host-oracle draw (identical PPMI distribution) keeps the de-risk tractable. The load-bearing
    # SPIKING decision (the speak/silence accumulator) STAYS spiking (the brain-based speak choice). The full
    # production CommunicableTurn (Stage B/C) uses the default spiking draw.
    proposer = GenerativeReplayProposer(comp, affirmed, negated, P, row, tau,
                                        np.random.default_rng(seed * 7 + 1),
                                        use_spiking_sampler=(not a.host_oracle_sampler))

    agents_set, actions_set, patients_set = set(agents), set(actions), set(patients)
    inflect = _build_inflection_map(sorted(actions_set))
    vocab_sets = (agents_set, actions_set, patients_set, inflect)
    grounded_faculty = TemplateStubFaculty()
    full_pools = (set(agents), set(actions), set(patients))
    # ONE shared per-topic candidate cache, used by the scratch grounded-filter + all four appraisal arms (they
    # share the SAME proposer + graph, so the candidate SET per topic is identical; only the worth ranking + the
    # speak drive depend on the learned Q). Eliminates the N-fold repeat of the per-candidate non-contradiction
    # resonate (composer.ask_yes_no) -- the CPU bottleneck. No change to any decision.
    cand_cache = {}

    # held-out topics (words NOT the agent of any stored fact); keep only ones the brain HAS a grounded view about.
    stored_agents = {f[0] for f in affirmed}
    topic_pool = [w for w in (agents + patients) if w not in stored_agents]
    rng.shuffle(topic_pool)

    # a scratch turn to filter grounded topics (uses an empty Q; only the propose path is exercised)
    scratch_value = SignedLearnedSpeakValue([w for w in topic_pool], {w: context_code(P, row, w) for w in topic_pool},
                                            lr=a.lr, da_reward=a.da_reward, da_baseline=a.da_baseline,
                                            kappa=a.kappa, rng=np.random.default_rng(seed * 211 + 3))
    scratch = CommunicableTurn(comp, bc_agent, proposer, accumulator, P, row, vocab_sets, grounded_faculty,
                               scratch_value, {w: context_code(P, row, w) for w in topic_pool}, full_pools=full_pools,
                               w_value=a.w_value, w_plaus=a.w_plaus, w_fam=a.w_fam,
                               speak_base_pA=a.speak_base_pA, speak_gain_pA=a.speak_gain_pA,
                               silence_drive_pA=a.silence_drive_pA, cand_cache=cand_cache)
    grounded_topics = [t for t in topic_pool if scratch.propose_candidates_about(t, n_attempts=a.n_attempts)]
    topics = grounded_topics[:a.n_topics]
    if len(topics) < 6:
        return {"seed": seed, "n_topics": len(topics), "insufficient_topics": True}

    codes = {t: context_code(P, row, t) for t in topics}

    # =====================================================================
    # THE TAUGHT/UNTAUGHT SPLIT -- stratified-random orthogonal to plausibility (the learned-talkativeness fix).
    # =====================================================================
    split_rng = np.random.default_rng(seed * 131 + 17)
    topic_plaus = {}
    for t in topics:
        cs = scratch.propose_candidates_about(t, n_attempts=a.n_attempts)
        topic_plaus[t] = (cs[0][1] if cs else 0.0)
    by_plaus = sorted(topics, key=lambda t: topic_plaus[t])
    n_taught = max(1, int(round(a.taught_frac * len(topics))))
    stride = len(by_plaus) / float(n_taught)
    taught = set()
    for k in range(n_taught):
        lo = int(round(k * stride))
        hi = max(lo + 1, int(round((k + 1) * stride)))
        hi = min(hi, len(by_plaus))
        taught.add(by_plaus[lo + int(split_rng.integers(hi - lo))])
    while len(taught) < n_taught:
        taught.add(by_plaus[int(split_rng.integers(len(by_plaus)))])
    untaught_all = [t for t in topics if t not in taught]
    untaught_overlap = {t: (max(code_overlap(codes[t], codes[tt]) for tt in taught) if taught else 0.0)
                        for t in untaught_all}
    ov_sorted = sorted(untaught_all, key=lambda t: -untaught_overlap[t]) if untaught_all else []
    half = max(1, len(ov_sorted) // 2)
    similar_untaught = set(ov_sorted[:half])
    dissimilar_untaught = set(ov_sorted[half:])

    # =====================================================================
    # LEARN the talkativeness over feedback ROUNDS (the three-factor rule).  Three arms: value-intact / DA-lesion
    # / decorrelated-credit.  The teaching path is `value.feedback(topic, +1)`; we run it through the SIGNED
    # update (positive on taught).  Record the taught-bin curve (the monotonicity check).
    # =====================================================================
    def learn(lesion_DA=False, decorrelate=False, record_curve=False):
        lv = SignedLearnedSpeakValue(topics, codes, lr=a.lr, da_reward=a.da_reward, da_baseline=a.da_baseline,
                                     kappa=a.kappa, da_punish=a.da_reward,
                                     rng=np.random.default_rng(seed * 211 + 3))
        curve = []
        order_rng = np.random.default_rng(seed * 307 + 5)
        for r in range(a.n_rounds):
            order = list(topics)
            order_rng.shuffle(order)
            for t in order:
                # the perceived feedback: TAUGHT -> the owner asked to elaborate (a +1 reward); UNTAUGHT -> neutral.
                lv.feedback(t, +1 if t in taught else 0, lesion_DA=lesion_DA, decorrelate=decorrelate)
            if record_curve:
                curve.append({
                    "round": r + 1,
                    "Q_taught_mean": float(np.mean([lv.value(t) for t in taught])) if taught else 0.0,
                    "Q_similar_untaught_mean": (float(np.mean([lv.value(t) for t in similar_untaught]))
                                                if similar_untaught else 0.0),
                    "Q_dissimilar_untaught_mean": (float(np.mean([lv.value(t) for t in dissimilar_untaught]))
                                                   if dissimilar_untaught else 0.0),
                })
        return lv, curve

    lv_value, curve = learn(lesion_DA=False, decorrelate=False, record_curve=True)
    lv_pre = SignedLearnedSpeakValue(topics, codes, lr=a.lr, da_reward=a.da_reward, da_baseline=a.da_baseline,
                                     kappa=a.kappa, rng=np.random.default_rng(seed * 211 + 3))  # the PRE-teaching Q (all 0)
    lv_lesion, _ = learn(lesion_DA=True, decorrelate=False)
    lv_decorr, _ = learn(lesion_DA=False, decorrelate=True)

    # =====================================================================
    # Build the FUSED turn over the value-intact Q + calibrate the normalisers (after learning).
    # =====================================================================
    def make_turn(value_obj):
        t = CommunicableTurn(comp, bc_agent, proposer, accumulator, P, row, vocab_sets, grounded_faculty,
                             value_obj, codes, full_pools=full_pools,
                             w_value=a.w_value, w_plaus=a.w_plaus, w_fam=a.w_fam,
                             speak_base_pA=a.speak_base_pA, speak_gain_pA=a.speak_gain_pA,
                             silence_drive_pA=a.silence_drive_pA, cand_cache=cand_cache)
        return t

    turn_value = make_turn(lv_value)
    conf_lo, conf_hi = turn_value.calibrate(topics, n_attempts=a.n_attempts)
    # the PRE-teaching / lesion / decorrelated arms reuse the SAME normalisers (so the gap is purely the learned Q).
    turn_pre = make_turn(lv_pre)
    turn_pre._plaus_lo, turn_pre._plaus_hi = turn_value._plaus_lo, turn_value._plaus_hi
    turn_pre._val_lo, turn_pre._val_hi = turn_value._val_lo, turn_value._val_hi
    turn_pre._fam_lo, turn_pre._fam_hi = turn_value._fam_lo, turn_value._fam_hi
    turn_pre._conf_lo, turn_pre._conf_hi = conf_lo, conf_hi
    turn_lesion = make_turn(lv_lesion)
    turn_lesion._plaus_lo, turn_lesion._plaus_hi = turn_value._plaus_lo, turn_value._plaus_hi
    turn_lesion._val_lo, turn_lesion._val_hi = turn_value._val_lo, turn_value._val_hi
    turn_lesion._fam_lo, turn_lesion._fam_hi = turn_value._fam_lo, turn_value._fam_hi
    turn_lesion._conf_lo, turn_lesion._conf_hi = conf_lo, conf_hi
    turn_decorr = make_turn(lv_decorr)
    turn_decorr._plaus_lo, turn_decorr._plaus_hi = turn_value._plaus_lo, turn_value._plaus_hi
    turn_decorr._val_lo, turn_decorr._val_hi = turn_value._val_lo, turn_value._val_hi
    turn_decorr._fam_lo, turn_decorr._fam_hi = turn_value._fam_lo, turn_value._fam_hi
    turn_decorr._conf_lo, turn_decorr._conf_hi = conf_lo, conf_hi

    # =====================================================================
    # EXERCISE THE FUSED TURN -- the four cases, end-to-end (the composition).
    # =====================================================================
    # (A) KNOWN-FACT (question) -- a stored cue must answer CERTAIN; an un-stored cue must abstain.
    cue_to_patients = defaultdict(set)
    for ag, ac, pt in affirmed:
        cue_to_patients[(ag, ac)].add(pt)
    known_records = []
    for (ag, ac), pats in cue_to_patients.items():
        r = turn_value.turn(f"what does {ag} {ac}?", cue=(ag, ac))
        known_records.append((r, pats))
    # yes/no on a few affirmed facts (full SVO -> handles many-to-one cues)
    yesno_records = [turn_value.turn(f"is it true {ag} {ac} {pt}?", cue=(ag, ac, pt))
                     for ag, ac, pt in affirmed[: min(8, len(affirmed))]]

    # (B) OPINION (novel) -- the value-intact arm vs the PRE-teaching arm (the teaching-loop comparison).
    opinion_value = [turn_value.turn(f"what do you think about {t}?", topic=t, n_attempts=a.n_attempts)
                     for t in topics]
    opinion_pre = [turn_pre.turn(f"what do you think about {t}?", topic=t, n_attempts=a.n_attempts)
                   for t in topics]
    opinion_lesion = [turn_lesion.turn(f"what do you think about {t}?", topic=t, n_attempts=a.n_attempts)
                      for t in topics]
    opinion_decorr = [turn_decorr.turn(f"what do you think about {t}?", topic=t, n_attempts=a.n_attempts)
                      for t in topics]

    # (C) PHATIC -- "hi" makes a non-factual reply.
    phatic_rec = turn_value.turn("hi")
    phatic_rec2 = turn_value.turn("how are you?")

    # (D) TEACHING -- a live feedback round on a topic that was SILENT pre-teaching; verify it raises Q + can flip
    # the emit decision.  Then the negative "stop" lowers it.  This exercises the turn() teaching path end-to-end.
    emit_topics_value = {r["topic"] for r in opinion_value if r.get("emitted")}
    emit_topics_pre = {r["topic"] for r in opinion_pre if r.get("emitted")}

    # =====================================================================
    # GATE (1): EACH COMPONENT'S GO REPRODUCES IN COMPOSITION.
    # =====================================================================
    # known-fact recall: every distinct stored cue answers CERTAIN with a genuine stored patient (small-D tol),
    # and the answer is flagged certain (never a hedge).
    known_ok = 0
    for r, pats in known_records:
        if (not r["abstained"]) and r["recalled_svo"] is not None and r["recalled_svo"][2] in pats and r["certain"]:
            known_ok += 1
    known_answer_rate = known_ok / max(1, len(known_records))
    yesno_ok = sum(1 for r in yesno_records if (not r["abstained"]) and r.get("yesno") == "yes")
    known_recall_ok = (len(known_records) > 0 and known_answer_rate >= a.stored_answer_bar
                       and yesno_ok == len(yesno_records))

    # novel channel: emitted propositions are GENERATED (novel) + topic-relevant + flagged.
    emitted_value = [r for r in opinion_value if r.get("emitted")]
    n_emit_value = len(emitted_value)
    n_novel = sum(1 for r in emitted_value if tuple(r["proposed_triple"]) not in all_stored)
    n_topic_rel = sum(1 for r in emitted_value if r.get("topic_in_proposition"))
    all_novel = (n_emit_value > 0) and (n_novel == n_emit_value)
    all_topic_rel = (n_emit_value > 0) and (n_topic_rel == n_emit_value)
    all_flagged = (n_emit_value > 0) and all(r.get("hedge") is not None for r in emitted_value)

    # appraisal speaks-more-where-taught: the value-intact arm emits on MORE topics than the PRE-teaching arm,
    # AND the taught speak-rate > untaught speak-rate (the behavioral teaching effect through the spiking gate).
    n_emit_pre = len(emit_topics_pre)
    speaks_more_than_pre = n_emit_value > n_emit_pre

    def _rate(records, bin_topics):
        bset = set(bin_topics)
        rel = [r for r in records if r.get("topic") in bset]
        return (sum(1 for r in rel if r.get("emitted")) / max(1, len(rel))), len(rel)

    rate_taught_v, _ = _rate(opinion_value, taught)
    rate_untaught_v, _ = _rate(opinion_value, untaught_all)
    rate_behavioral_ok = rate_taught_v > rate_untaught_v + 1e-9

    # learning curve monotone + rose on the taught bin
    q_taught_curve = [c["Q_taught_mean"] for c in curve]
    curve_monotonic = all(q_taught_curve[i + 1] >= q_taught_curve[i] - 1e-9
                          for i in range(len(q_taught_curve) - 1))
    curve_rose = (len(q_taught_curve) >= 2) and (q_taught_curve[-1] > q_taught_curve[0] + 1e-9)
    # the learned-Q similarity gradient (clean, threshold-independent)
    Q_taught = float(np.mean([lv_value.value(t) for t in taught])) if taught else 0.0
    Q_simU = float(np.mean([lv_value.value(t) for t in similar_untaught])) if similar_untaught else 0.0
    Q_disU = float(np.mean([lv_value.value(t) for t in dissimilar_untaught])) if dissimilar_untaught else 0.0
    q_gradient_ok = (Q_taught > Q_simU + 1e-9) and (Q_simU >= Q_disU - 1e-9) and (Q_taught > Q_disU + 1e-9)

    components_reproduce = bool(known_recall_ok and all_novel and all_topic_rel
                                and speaks_more_than_pre and rate_behavioral_ok
                                and curve_monotonic and curve_rose and q_gradient_ok)

    # =====================================================================
    # GATE (2): 0 MOAT LEAKS ACROSS THE WHOLE TURN (HARD).
    #   - every novel emission's un-stored proposition still ABSTAINS on the known-fact channel;
    #   - the generative channel NEVER stored (the composer's stored-fact set is unchanged);
    #   - every novel emission is FLAGGED;
    #   - the known-fact channel never asserted an un-stored fact as certain (abstained instead).
    # =====================================================================
    moat_leaks = 0
    for r in emitted_value:
        a_, v_, p_ = r["proposed_triple"]
        if bc_agent.what_does(a_, v_) == p_:
            moat_leaks += 1
        if bc_agent.is_it_true(a_, v_, p_) == "yes":
            moat_leaks += 1
    # the generative channel did NOT write to the store: assert the composer's stored set is exactly all_stored
    # (the b2 proposer + the novel channel never call comp.store; we re-verify by querying that NONE of the
    # emitted novel triples became a known fact -- the leak check above -- AND the affirmed facts still answer).
    store_unchanged = (moat_leaks == 0)
    moat_ok = (moat_leaks == 0) and all_flagged and known_recall_ok

    # =====================================================================
    # GATE (3): A FEEDBACK ROUND RAISES NEXT-TURN TALKATIVENESS THERE (+ the negative LOWERS it).
    # We do this LIVE through the turn() teaching path on a FRESH value object so the effect is the orchestrator's,
    # not the batch learn(): pick a grounded topic, record its Q + emit decision pre-feedback, deliver "tell me
    # more" via turn(), then re-check.  The negative "stop" then lowers it back.
    # =====================================================================
    live_value = SignedLearnedSpeakValue(topics, codes, lr=a.lr, da_reward=a.da_reward, da_baseline=a.da_baseline,
                                         kappa=a.kappa, da_punish=a.da_reward,
                                         rng=np.random.default_rng(seed * 211 + 3))
    live_turn = make_turn(live_value)
    live_turn._plaus_lo, live_turn._plaus_hi = turn_value._plaus_lo, turn_value._plaus_hi
    live_turn._val_lo, live_turn._val_hi = turn_value._val_lo, turn_value._val_hi
    live_turn._fam_lo, live_turn._fam_hi = turn_value._fam_lo, turn_value._fam_hi
    live_turn._conf_lo, live_turn._conf_hi = conf_lo, conf_hi
    # pick a topic the brain DOES have a grounded view about; ask an opinion (sets _last_topic), then teach.
    fb_topic = topics[0]
    _ = live_turn.turn(f"what do you think about {fb_topic}?", topic=fb_topic, n_attempts=a.n_attempts)
    q_before = live_value.value(fb_topic)
    # validate the FULL teaching path end-to-end ONCE through turn() (router classifies 'tell me more' as
    # teaching -> feedback() raises Q + the "immediately elaborate" rerun fires); then accumulate the remaining
    # rounds via the cheap feedback() directly (the rerun's expensive propose+spiking-decide is UX, not part of
    # the Q-rise measurement, which reads live_value.value()). This keeps the orchestration validation while
    # avoiding 2*(n_rounds-1) redundant spiking decisions.
    teach_rec = live_turn.turn("tell me more", topic=fb_topic, n_attempts=a.n_attempts)
    teaching_path_ok = (teach_rec.get("channel") == "teaching" and teach_rec.get("polarity") == +1
                        and teach_rec.get("feedback_topic") == fb_topic and "rerun" in teach_rec)
    for _ in range(a.n_rounds - 1):
        live_turn.feedback(fb_topic, +1)            # cheap signed update (the brain's three-factor plasticity)
    q_after_teach = live_value.value(fb_topic)
    feedback_raises = q_after_teach > q_before + 1e-9
    # the negative extension: "stop" lowers it back -- validate the path once, then accumulate cheaply.
    stop_rec = live_turn.turn("that's enough", topic=fb_topic, n_attempts=a.n_attempts)
    stop_path_ok = (stop_rec.get("channel") == "teaching" and stop_rec.get("polarity") == -1)
    for _ in range(a.n_rounds - 1):
        live_turn.feedback(fb_topic, -1)
    q_after_stop = live_value.value(fb_topic)
    negative_lowers = q_after_stop < q_after_teach - 1e-9

    # =====================================================================
    # GATE (4): THE DA-LESION ABOLISHES the talkativeness change + the value-driven extra speaking.
    #   - lesion arm: Q never moved (pin DA) -> emits on the SAME or FEWER topics than the PRE-teaching arm, and
    #     the value arm emits MORE than the lesion arm (the extra emissions require the value system);
    #   - the value arm's taught>untaught gap VANISHES under lesion.
    #   - live lesion: a "tell me more" through turn() with DA pinned does NOT raise Q.
    # =====================================================================
    n_emit_lesion = sum(1 for r in opinion_lesion if r.get("emitted"))
    rate_taught_les, _ = _rate(opinion_lesion, taught)
    rate_untaught_les, _ = _rate(opinion_lesion, untaught_all)
    value_gap = rate_taught_v - rate_untaught_v
    lesion_gap = rate_taught_les - rate_untaught_les
    lesion_abolishes_batch = (n_emit_value > n_emit_lesion) and (value_gap > 1e-9) \
        and (lesion_gap <= 0.5 * value_gap + 1e-9)
    # live DA-lesion: a teaching round with DA pinned does NOT raise Q
    lesion_value_live = SignedLearnedSpeakValue(topics, codes, lr=a.lr, da_reward=a.da_reward,
                                                da_baseline=a.da_baseline, kappa=a.kappa, da_punish=a.da_reward,
                                                rng=np.random.default_rng(seed * 211 + 3))
    lesion_turn_live = make_turn(lesion_value_live)
    lesion_turn_live._last_topic = fb_topic
    q_les_before = lesion_value_live.value(fb_topic)
    for _ in range(a.n_rounds):
        lesion_turn_live.feedback(fb_topic, +1, lesion_DA=True)   # the teaching path with the SNc lesioned
    q_les_after = lesion_value_live.value(fb_topic)
    lesion_no_learn_live = abs(q_les_after - q_les_before) < 1e-12
    lesion_abolishes = bool(lesion_abolishes_batch and lesion_no_learn_live)

    # =====================================================================
    # ANTI-CHEAT: SHUFFLED-PPMI-GRAPH (the novel channel's groundedness collapses).
    # =====================================================================
    P_shuf = shuffle_graph(P, np.random.default_rng(seed * 17 + 5))
    pos_s = P_shuf[P_shuf > 0]
    tau_s = float(np.percentile(pos_s, a.tau_pct)) if pos_s.size else 0.0

    def _plausible_shuf(tp):
        a_, ac_, p_ = tp
        return (P_shuf[row[a_], row[ac_]] >= tau_s) and (P_shuf[row[ac_], row[p_]] >= tau_s)

    emit_triples = [tuple(r["proposed_triple"]) for r in emitted_value]
    true_pass = sum(1 for tp in emit_triples if proposer._plausible(*tp))     # == len (gate-constructed)
    shuf_pass = sum(1 for tp in emit_triples if _plausible_shuf(tp))
    true_frac = true_pass / max(1, len(emit_triples))
    shuf_frac = shuf_pass / max(1, len(emit_triples))
    grounded_advantage = true_frac / max(shuf_frac, 1.0 / max(1, len(emit_triples)))
    grounded_ok = (len(emit_triples) > 0) and (grounded_advantage >= a.advantage_bar)

    # =====================================================================
    # ANTI-CHEAT: FREE-GENERATE LESION (sever the brain's proposal -> the faculty free-generates -> VERIFY rejects).
    # Run on the composed turn's emitted propositions.
    # =====================================================================
    all_pats = sorted(patients_set)
    lesion_caught, lesion_total = 0, 0
    for r in emitted_value:
        a_, v_, p_ = r["proposed_triple"]
        wrong = next((x for x in all_pats if x != p_), p_ + "_X")
        lesion_faculty = InjectingStubFaculty({p_: wrong}, swap_role="patient")
        rv = turn_value.render_and_verify((a_, v_, p_), lesion_faculty, faculty_mode="lesion")
        caught = (not rv["verified"])
        lesion_caught += int(caught)
        lesion_total += 1
    free_gen_lesion_ok = (lesion_total > 0) and (lesion_caught == lesion_total)

    # =====================================================================
    # ANTI-CHEAT: NON-CIRCULARITY corr(learned Q, plausibility) ~ 0.
    # =====================================================================
    qv = np.array([lv_value.value(t) for t in topics], dtype=float)
    pv = np.array([topic_plaus[t] for t in topics], dtype=float)
    value_plaus_corr = (float(np.corrcoef(qv, pv)[0, 1]) if len(qv) >= 3 and qv.std() > 0 and pv.std() > 0 else 0.0)
    noncircular_ok = abs(value_plaus_corr) <= a.max_value_plaus_corr

    # =====================================================================
    # ANTI-CHEAT: DECORRELATED-CREDIT (a FLAT global rise, no taught/untaught gap).
    # =====================================================================
    rate_taught_dec, _ = _rate(opinion_decorr, taught)
    rate_untaught_dec, _ = _rate(opinion_decorr, untaught_all)
    decorr_gap = rate_taught_dec - rate_untaught_dec
    context_specific_ok = bool((value_gap > 1e-9) and (decorr_gap <= 0.5 * value_gap + 1e-9))

    # phatic: makes a non-factual reply, not flagged-or-stored
    phatic_ok = (phatic_rec.get("channel") == "phatic" and phatic_rec.get("reply")
                 and not phatic_rec.get("is_factual_claim", True)
                 and phatic_rec2.get("channel") == "phatic")

    print(f"\n[stageA seed {seed}] stored {len(affirmed)} ({len(negated)} neg) | grounded topics {len(topics)} | "
          f"taught {len(taught)} (simU {len(similar_untaught)} disU {len(dissimilar_untaught)}) | "
          f"tau(P{a.tau_pct})={tau:.3f}", flush=True)
    print(f"  CHANNELS exercised: known-fact {len(known_records)} cues + {len(yesno_records)} yes/no | opinion "
          f"{len(topics)} | phatic 2 | teaching (live)", flush=True)
    print(f"  (1) COMPONENTS REPRODUCE: {components_reproduce}", flush=True)
    print(f"      known-fact recall {known_ok}/{len(known_records)} ({known_answer_rate:.2f}) + yes/no "
          f"{yesno_ok}/{len(yesno_records)} -> {known_recall_ok}", flush=True)
    print(f"      novel: emitted {n_emit_value} | novel {n_novel}/{n_emit_value} ({all_novel}) | topic-rel "
          f"{n_topic_rel}/{n_emit_value} ({all_topic_rel})", flush=True)
    print(f"      speaks-more-than-pre {speaks_more_than_pre} ({n_emit_value} > {n_emit_pre}) | speak-rate taught "
          f"{rate_taught_v:.2f} > untaught {rate_untaught_v:.2f} ({rate_behavioral_ok})", flush=True)
    print(f"      learned-Q gradient taught {Q_taught:.3f} > simU {Q_simU:.3f} >= disU {Q_disU:.3f} "
          f"({q_gradient_ok}) | curve monotone {curve_monotonic} rose {curve_rose}", flush=True)
    print(f"  (2) MOAT (0 leaks across the whole turn): {moat_leaks} leaks | all-flagged {all_flagged} | "
          f"store-unchanged {store_unchanged} -> {moat_ok}", flush=True)
    print(f"  (3) FEEDBACK raises talkativeness: Q[{fb_topic}] {q_before:.3f} -> teach {q_after_teach:.3f} "
          f"({feedback_raises}) -> stop {q_after_stop:.3f} (negative-lowers {negative_lowers}) | teaching-path "
          f"{teaching_path_ok} stop-path {stop_path_ok}", flush=True)
    print(f"  (4) DA-LESION abolishes: batch (value {n_emit_value} > lesion {n_emit_lesion}; value-gap "
          f"{value_gap:+.2f} lesion-gap {lesion_gap:+.2f}) {lesion_abolishes_batch} | live (Q unchanged) "
          f"{lesion_no_learn_live} -> {lesion_abolishes}", flush=True)
    print(f"  ANTI-CHEATS: shuffled-graph adv {grounded_advantage:.1f}x (>= {a.advantage_bar}x: {grounded_ok}) | "
          f"free-gen-lesion caught {lesion_caught}/{lesion_total} ({free_gen_lesion_ok}) | non-circular "
          f"corr={value_plaus_corr:+.3f} ({noncircular_ok}) | decorrelated gap {decorr_gap:+.2f} "
          f"({context_specific_ok}) | phatic {phatic_ok}", flush=True)
    if emitted_value:
        print(f"  example fused-turn flagged hypotheses (opinion channel):", flush=True)
        for r in emitted_value[:5]:
            tag = " [TAUGHT]" if r["topic"] in taught else ""
            print(f"     X={r['topic']!r:>10} -> {r['reply']!r}  (worth {r['worth']}, conf {r['confidence']}, "
                  f"plaus {r['plausibility']}){tag}", flush=True)
    # show one of each channel surface
    ex_known = next((r for (r, _p) in known_records if not r["abstained"]), None)
    print(f"  channel examples: KNOWN={ex_known['answer'] if ex_known else None!r} | "
          f"PHATIC={phatic_rec.get('reply')!r}", flush=True)

    return {
        "seed": seed,
        "n_stored": len(affirmed),
        "n_negated": len(negated),
        "n_topics": len(topics),
        "n_taught": len(taught),
        "n_similar_untaught": len(similar_untaught),
        "n_dissimilar_untaught": len(dissimilar_untaught),
        "tau": tau,
        # gate (1) components reproduce
        "known_recall_rate": known_answer_rate,
        "known_recall_ok": bool(known_recall_ok),
        "yesno_ok": yesno_ok,
        "yesno_total": len(yesno_records),
        "n_emitted_value": n_emit_value,
        "n_emitted_pre": n_emit_pre,
        "all_novel": bool(all_novel),
        "all_topic_relevant": bool(all_topic_rel),
        "speaks_more_than_pre": bool(speaks_more_than_pre),
        "rate_taught_value": rate_taught_v,
        "rate_untaught_value": rate_untaught_v,
        "rate_behavioral_ok": bool(rate_behavioral_ok),
        "Q_taught": Q_taught,
        "Q_similar_untaught": Q_simU,
        "Q_dissimilar_untaught": Q_disU,
        "q_gradient_ok": bool(q_gradient_ok),
        "q_taught_curve": q_taught_curve,
        "curve_monotonic": bool(curve_monotonic),
        "curve_rose": bool(curve_rose),
        "components_reproduce": bool(components_reproduce),
        # gate (2) moat
        "moat_leaks": moat_leaks,
        "all_flagged": bool(all_flagged),
        "store_unchanged": bool(store_unchanged),
        "moat_ok": bool(moat_ok),
        # gate (3) feedback
        "feedback_topic": fb_topic,
        "q_before_feedback": q_before,
        "q_after_teach": q_after_teach,
        "q_after_stop": q_after_stop,
        "feedback_raises_talkativeness": bool(feedback_raises),
        "negative_feedback_lowers": bool(negative_lowers),
        "teaching_path_ok": bool(teaching_path_ok),
        "stop_path_ok": bool(stop_path_ok),
        # gate (4) lesion
        "n_emitted_lesion": n_emit_lesion,
        "value_gap": value_gap,
        "lesion_gap": lesion_gap,
        "lesion_abolishes_batch": bool(lesion_abolishes_batch),
        "lesion_no_learn_live": bool(lesion_no_learn_live),
        "lesion_abolishes": bool(lesion_abolishes),
        # anti-cheats
        "grounded_advantage_ratio": grounded_advantage,
        "grounded_ok": bool(grounded_ok),
        "free_gen_lesion_caught": lesion_caught,
        "free_gen_lesion_total": lesion_total,
        "free_gen_lesion_ok": bool(free_gen_lesion_ok),
        "value_plausibility_corr": value_plaus_corr,
        "noncircular_ok": bool(noncircular_ok),
        "decorrelated_gap": decorr_gap,
        "context_specific_ok": bool(context_specific_ok),
        "phatic_ok": bool(phatic_ok),
        # trail
        "emitted_examples": [{"topic": r["topic"], "reply": r["reply"], "worth": r["worth"],
                              "confidence": r["confidence"], "plausibility": r["plausibility"],
                              "taught": r["topic"] in taught}
                             for r in emitted_value[:12]],
        "known_example": (ex_known["answer"] if ex_known else None),
        "phatic_example": phatic_rec.get("reply"),
    }


def decide_verdict(rows, a):
    """STAGE A GO iff, across ALL seeds: (1) each component's GO reproduces in composition; (2) 0 moat leaks
    across the whole turn (+ all flagged + known recall); (3) a feedback round raises next-turn talkativeness
    (+ the negative lowers it); (4) the DA-lesion abolishes the talkativeness change + the extra speaking; AND
    all anti-cheats hold (shuffled-graph groundedness collapses, free-gen lesion caught, non-circular,
    context-specific/decorrelated, phatic clean).  Else HONEST_NEGATIVE / BOUNDARY + the precise failing gate."""
    rows = [r for r in rows if not r.get("insufficient_topics")]
    if not rows:
        return "INVALID_insufficient_grounded_topics", {"note": "fewer than 6 grounded topics in every seed"}

    def col(k):
        return [r[k] for r in rows]

    components_all = all(col("components_reproduce"))
    moat_all = all(col("moat_ok"))
    feedback_all = (all(col("feedback_raises_talkativeness")) and all(col("negative_feedback_lowers"))
                    and all(col("teaching_path_ok")) and all(col("stop_path_ok")))
    lesion_all = all(col("lesion_abolishes"))
    grounded_all = all(col("grounded_ok"))
    freegen_all = all(col("free_gen_lesion_ok"))
    noncirc_all = all(col("noncircular_ok"))
    context_all = all(col("context_specific_ok"))
    phatic_all = all(col("phatic_ok"))

    detail = {
        "n_seeds": len(rows),
        "components_reproduce_all_seeds": bool(components_all),
        "known_recall_rate_mean": float(np.mean(col("known_recall_rate"))),
        "n_emitted_value_mean": float(np.mean(col("n_emitted_value"))),
        "n_emitted_pre_mean": float(np.mean(col("n_emitted_pre"))),
        "all_novel_all_seeds": bool(all(col("all_novel"))),
        "all_topic_relevant_all_seeds": bool(all(col("all_topic_relevant"))),
        "speaks_more_than_pre_all_seeds": bool(all(col("speaks_more_than_pre"))),
        "rate_taught_value_mean": float(np.mean(col("rate_taught_value"))),
        "rate_untaught_value_mean": float(np.mean(col("rate_untaught_value"))),
        "q_gradient_all_seeds": bool(all(col("q_gradient_ok"))),
        "curve_monotonic_all_seeds": bool(all(col("curve_monotonic"))),
        "curve_rose_all_seeds": bool(all(col("curve_rose"))),
        "moat_all_seeds": bool(moat_all),
        "moat_leaks_total": int(np.sum(col("moat_leaks"))),
        "all_flagged_all_seeds": bool(all(col("all_flagged"))),
        "feedback_raises_all_seeds": bool(all(col("feedback_raises_talkativeness"))),
        "negative_lowers_all_seeds": bool(all(col("negative_feedback_lowers"))),
        "teaching_path_ok_all_seeds": bool(all(col("teaching_path_ok"))),
        "stop_path_ok_all_seeds": bool(all(col("stop_path_ok"))),
        "q_before_feedback_mean": float(np.mean(col("q_before_feedback"))),
        "q_after_teach_mean": float(np.mean(col("q_after_teach"))),
        "q_after_stop_mean": float(np.mean(col("q_after_stop"))),
        "lesion_abolishes_all_seeds": bool(lesion_all),
        "lesion_no_learn_live_all_seeds": bool(all(col("lesion_no_learn_live"))),
        "n_emitted_lesion_mean": float(np.mean(col("n_emitted_lesion"))),
        "value_gap_mean": float(np.mean(col("value_gap"))),
        "lesion_gap_mean": float(np.mean(col("lesion_gap"))),
        "grounded_advantage_mean": float(np.mean(col("grounded_advantage_ratio"))),
        "grounded_advantage_min": float(np.min(col("grounded_advantage_ratio"))),
        "grounded_all_seeds": bool(grounded_all),
        "free_gen_lesion_caught_total": int(np.sum(col("free_gen_lesion_caught"))),
        "free_gen_lesion_total": int(np.sum(col("free_gen_lesion_total"))),
        "free_gen_lesion_all_seeds": bool(freegen_all),
        "value_plaus_corr_absmax": float(np.max(np.abs(col("value_plausibility_corr")))),
        "noncircular_all_seeds": bool(noncirc_all),
        "decorrelated_gap_mean": float(np.mean(col("decorrelated_gap"))),
        "context_specific_all_seeds": bool(context_all),
        "phatic_all_seeds": bool(phatic_all),
        "advantage_bar": float(a.advantage_bar),
        "max_value_plaus_corr": float(a.max_value_plaus_corr),
        "stored_answer_bar": float(a.stored_answer_bar),
    }

    # ordered checks: the precise failing gate localizes the next bounded fix.
    if not noncirc_all:
        verdict = "INVALID_value_is_relabeled_plausibility"
    elif not moat_all:
        verdict = "HONEST_NEGATIVE_moat_leak_in_composition"        # the load-bearing safety invariant
    elif not freegen_all:
        verdict = "HONEST_NEGATIVE_free_generate_lesion_not_caught"  # the LLM is doing the cognition
    elif not components_all:
        verdict = "HONEST_NEGATIVE_component_regressed_in_composition"
    elif not feedback_all:
        verdict = "HONEST_NEGATIVE_feedback_does_not_raise_talkativeness"
    elif not lesion_all:
        verdict = "HONEST_NEGATIVE_lesion_does_not_abolish"
    elif not grounded_all:
        verdict = "HONEST_NEGATIVE_novel_channel_not_grounded"
    elif not context_all:
        verdict = "HONEST_NEGATIVE_not_context_specific"
    elif not phatic_all:
        verdict = "HONEST_NEGATIVE_phatic_channel_broken"
    else:
        verdict = "GO"
    return verdict, detail


def main():
    p = argparse.ArgumentParser(description="Communicable turn -- Stage A: fuse GENERATE/DECIDE/LEARN + known-fact "
                                            "+ phatic into ONE CommunicableTurn on one CPU brain; prove every "
                                            "safety invariant holds in composition.")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--D", type=int, default=256,
                   help="phasor dimension for the RF composer store (256 keeps the known-fact recall clean)")
    p.add_argument("--n-facts", type=int, default=24, help="AFFIRMED facts the brain is TOLD")
    p.add_argument("--n-negated", type=int, default=12, help="NEGATED facts (non-contradiction gate work)")
    p.add_argument("--n-topics", type=int, default=24, help="held-out grounded topics (the talkativeness arena)")
    p.add_argument("--n-attempts", type=int, default=500, help="generative-replay samples per topic")
    p.add_argument("--tau-pct", type=float, default=50.0, help="graph-related threshold = percentile of +PPMI")
    # the LEARNING (three-factor) hyperparams (the de-risk's defaults)
    p.add_argument("--taught-frac", type=float, default=0.4,
                   help="fraction of grounded topics TAUGHT (random, orthogonal to plausibility)")
    p.add_argument("--n-rounds", type=int, default=12, help="feedback rounds (each presents every topic once)")
    p.add_argument("--lr", type=float, default=0.10, help="three-factor learning rate")
    p.add_argument("--da-reward", type=float, default=1.0, help="phasic DA burst on a TAUGHT 'elaborate' feedback")
    p.add_argument("--da-baseline", type=float, default=0.0, help="baseline DA (no reward)")
    p.add_argument("--kappa", type=float, default=2.0, help="eligibility-overlap sharpness")
    # the appraisal weights + the spiking accumulator drift mapping (the PROPOSED DEFAULTS)
    p.add_argument("--w-value", type=float, default=0.5, help="weight on the LEARNED speak-value axis")
    p.add_argument("--w-plaus", type=float, default=0.35, help="weight on the plausibility axis")
    p.add_argument("--w-fam", type=float, default=0.15, help="weight on the familiarity axis")
    p.add_argument("--speak-base-pA", type=float, default=70.0, help="speak-pool base drive")
    p.add_argument("--speak-gain-pA", type=float, default=180.0, help="component-push -> speak drift gain")
    p.add_argument("--silence-drive-pA", type=float, default=150.0, help="silence-pool fixed reticence drive")
    p.add_argument("--acc-steps", type=int, default=120, help="spiking integration window (steps)")
    # gate bars
    p.add_argument("--advantage-bar", type=float, default=3.0, help="grounded shuffled-graph advantage ratio bar")
    p.add_argument("--max-value-plaus-corr", type=float, default=0.35,
                   help="max |corr(learned value, plausibility)| for NON-circular")
    p.add_argument("--stored-answer-bar", type=float, default=0.9,
                   help="min known-fact recall rate (small-D tolerant)")
    p.add_argument("--host-oracle-sampler", dest="host_oracle_sampler", action="store_true", default=True,
                   help="generative DRAW = the host sample from the brain's PPMI likelihood (the numpy-CPU/test "
                        "oracle; DEFAULT ON for this CPU de-risk -- the spiking draw is separately GO 6-seed). The "
                        "load-bearing SPIKING speak DECISION stays spiking regardless.")
    p.add_argument("--spiking-sampler", dest="host_oracle_sampler", action="store_false",
                   help="use the validated spiking soft-WTA generative draw (default-on production path; slower on CPU)")
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
    t0 = time.time()
    print(f"[stageA] seeds={seeds} D={a.D} n_topics={a.n_topics} taught_frac={a.taught_frac} -- fuse "
          f"GENERATE/DECIDE/LEARN + known-fact + phatic into ONE CommunicableTurn; prove every safety invariant "
          f"holds in composition.", flush=True)

    vocab, cat_ids, cat_names = taxonomy_to_vocab_categories(TAXONOMY_8x8)
    corpus_path = os.path.join(_REPO, "data", "corpus", "tinystories.txt")
    if not os.path.exists(corpus_path):
        print(f"[ERROR] corpus not found: {corpus_path}", flush=True)
        sys.exit(2)
    corpus = build_real_cooccurrence(corpus_path, vocab, cat_ids, window=a.window, repeat_cap=a.repeat_cap,
                                     seed=42, max_bytes=a.max_bytes, freq_floor=30,
                                     min_facts_per_category=20, verbose=True)

    print(f"[stageA] building the spiking speak/silence accumulator (Wang-2002 NMDA WTA)...", flush=True)
    accumulator = SpikingSpeakAccumulator(seed=12345, n_steps=a.acc_steps)

    rows = [run_seed(s, vocab, corpus, a, accumulator) for s in seeds]
    verdict, detail = decide_verdict(rows, a)

    print(f"\n{'='*100}", flush=True)
    print(f"  STAGE A VERDICT: {verdict}", flush=True)
    print(f"  (1) COMPONENTS REPRODUCE all seeds: {detail.get('components_reproduce_all_seeds')} "
          f"(known recall {detail.get('known_recall_rate_mean', float('nan')):.2f}; novel emitted mean "
          f"{detail.get('n_emitted_value_mean', float('nan')):.1f} vs pre {detail.get('n_emitted_pre_mean', float('nan')):.1f}; "
          f"speak-rate taught {detail.get('rate_taught_value_mean', float('nan')):.2f} > untaught "
          f"{detail.get('rate_untaught_value_mean', float('nan')):.2f})", flush=True)
    print(f"  (2) MOAT (0 leaks across the whole turn) all seeds: {detail.get('moat_all_seeds')} "
          f"({detail.get('moat_leaks_total')} leaks; all-flagged {detail.get('all_flagged_all_seeds')})", flush=True)
    print(f"  (3) FEEDBACK raises talkativeness all seeds: {detail.get('feedback_raises_all_seeds')} "
          f"(Q {detail.get('q_before_feedback_mean', float('nan')):.3f} -> teach "
          f"{detail.get('q_after_teach_mean', float('nan')):.3f} -> stop "
          f"{detail.get('q_after_stop_mean', float('nan')):.3f}; negative-lowers "
          f"{detail.get('negative_lowers_all_seeds')})", flush=True)
    print(f"  (4) DA-LESION abolishes all seeds: {detail.get('lesion_abolishes_all_seeds')} "
          f"(value emit mean {detail.get('n_emitted_value_mean', float('nan')):.1f} vs lesion "
          f"{detail.get('n_emitted_lesion_mean', float('nan')):.1f}; live-no-learn "
          f"{detail.get('lesion_no_learn_live_all_seeds')})", flush=True)
    print(f"  ANTI-CHEATS all seeds: shuffled-graph {detail.get('grounded_all_seeds')} (adv mean "
          f"{detail.get('grounded_advantage_mean', float('nan')):.1f}x, min "
          f"{detail.get('grounded_advantage_min', float('nan')):.1f}x) | free-gen-lesion "
          f"{detail.get('free_gen_lesion_all_seeds')} ({detail.get('free_gen_lesion_caught_total')}/"
          f"{detail.get('free_gen_lesion_total')}) | non-circular {detail.get('noncircular_all_seeds')} "
          f"(|corr| max {detail.get('value_plaus_corr_absmax', float('nan')):.3f}) | context-specific "
          f"{detail.get('context_specific_all_seeds')} | phatic {detail.get('phatic_all_seeds')}", flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}\n", flush=True)

    out = {
        "probe": "communicable_turn_stageA_derisk",
        "verdict": verdict,
        "seeds": seeds,
        "stage": "A -- the CPU unified turn (the fused mechanism); prove every safety invariant holds IN COMPOSITION",
        "scoping": "research/findings/raw/_communicable_brain_console_integration_scoping.md (§4 Stage A)",
        "config": {"D": a.D, "n_facts": a.n_facts, "n_negated": a.n_negated, "n_topics": a.n_topics,
                   "n_attempts": a.n_attempts, "tau_pct": a.tau_pct, "taught_frac": a.taught_frac,
                   "n_rounds": a.n_rounds, "lr": a.lr, "da_reward": a.da_reward, "da_baseline": a.da_baseline,
                   "kappa": a.kappa, "w_value": a.w_value, "w_plaus": a.w_plaus, "w_fam": a.w_fam,
                   "speak_base_pA": a.speak_base_pA, "speak_gain_pA": a.speak_gain_pA,
                   "silence_drive_pA": a.silence_drive_pA, "acc_steps": a.acc_steps,
                   "advantage_bar": a.advantage_bar, "max_value_plaus_corr": a.max_value_plaus_corr,
                   "stored_answer_bar": a.stored_answer_bar, "max_bytes": a.max_bytes,
                   "host_oracle_sampler": a.host_oracle_sampler,
                   "generative_draw": ("host-oracle (numpy-CPU/test path; PPMI likelihood unchanged)"
                                       if a.host_oracle_sampler else "spiking soft-WTA (production)"),
                   "spiking_speak_decision": "ALWAYS spiking (SpikingSpeakAccumulator, the brain-based speak choice)"},
        "turn_architecture": (
            "USER MESSAGE -> IntentRouter.classify {question, opinion, phatic, teaching} -> route: "
            "KNOWN-FACT (BrainConversationalAgent.what_does/is_it_true, hard-gated CERTAIN, moat abstains) | "
            "NOVEL (assimilate -> b2 GenerativeReplayProposer candidate set -> WORTH appraisal [learned Q + "
            "plausibility + familiarity] -> SpikingSpeakAccumulator decision -> render+VERIFY -> EMIT FLAGGED "
            "hypothesis, NOT stored) | PHATIC (canned non-factual reply) | TEACHING (SignedLearnedSpeakValue."
            "feedback -> three-factor Q update + re-run the topic).  The scoping's PROPOSED DEFAULTS: fall-through "
            "ON for opinion framings, OFF for bare factual questions; the de-risks' talkativeness prior + weights."),
        "reuse_by_import": {
            "generate": "research/runners/_genfrontier_b2_generative_replay_derisk.py (GenerativeReplayProposer) "
                        "[GO 6-seed]",
            "decide": "research/runners/_value_salience_appraisal_derisk.py (SpikingSpeakAccumulator) [GO 3-seed]",
            "learn": "research/runners/_learned_talkativeness_derisk.py (LearnedSpeakValue, context_code) [GO 3-seed]",
            "render_verify": "research/runners/_communicable_brain_probe1_whatdoyouthink.py (plausibility_score, "
                             "hedge_for) + _grounded_lang_integration_derisk._extract_svo_from_prose + "
                             "BrainConversationalAgent.parse [Probe-1 GO]",
            "known_fact_moat": "research/runners/brain_conversational_agent.py (BrainConversationalAgent) + "
                               "rf_phasor_composer.py (RFPhasorComposer no-confab moat)",
        },
        "new_glue": (
            "IntentRouter.classify (the rule classifier) + SignedLearnedSpeakValue (the scoping's signed-negative "
            "'stop' feedback extension: a DA dip -> negative RPE -> Q decreases, floored at 0; reuse-by-subclass, "
            "NO de-risk edit) + CommunicableTurn (the §1 decision logic + the phatic table + feedback()). NO sim/ edit."),
        "stage_a_gate": (
            "GO = the fused turn keeps EVERY invariant: (1) each component's GO reproduces in composition; (2) 0 "
            "moat leaks across the whole turn (HARD) + all flagged + known recall; (3) a feedback round raises "
            "next-turn talkativeness there (+ the negative lowers it); (4) the DA-lesion abolishes the talkativeness "
            "change + the value-driven extra speaking. ANTI-CHEATS: shuffled-PPMI-graph collapses groundedness; the "
            "free-generate lesion is caught-by-VERIFY; value is non-circular; the decorrelated-credit control "
            "flattens the taught/untaught gap; the phatic channel makes no claim."),
        "detail": detail,
        "per_seed": rows,
        "elapsed_total_s": time.time() - t0,
    }
    if a.out is None:
        a.out = os.path.join(_REPO, "research", "findings", "raw", "_communicable_turn_stageA_derisk.json")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {a.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
