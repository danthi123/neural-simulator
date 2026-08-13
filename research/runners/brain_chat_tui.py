"""brain_chat_tui — an easy TUI to LOAD a developed/trained brain and hold a MULTI-TURN conversation with it.

The owner uses this to TALK to a developed brain (e.g. the self-knowledge brain). It LOADS the EXACT developed
brain (its grounded concept codes + its stored facts + its vocab) and runs a multi-turn chat loop:

    prompt -> parse the question (self-aliases resolved, anaphora resolved from the discourse buffer)
           -> RECALL from the brain (who/what/yes-no/describe/reason via the agent; the no-confab GATE)
           -> RENDER fluently (default: the OFF-BRIDGE Qwen grounded-language faculty, gate->constrain->verify,
              loaded ONCE + kept warm; --stub-renderer uses the template-stub, GPU-FREE, for the CPU smoke)
           -> print the answer, OR "I don't know about that." on abstention (the MOAT).

The render is GATED + VERIFIED: the brain supplies + verifies the CONTENT (the moat holds EVEN WITH a real
generative LLM in the loop); the faculty's only job is fluent surface form.

GENERATE channel (#3E) SURFACE -- brain-native SPIKING mouth (production default): when the brain VOLUNTEERS a
novel grounded HYPOTHESIS (an open-ended "what might a dog do" turn -> a moat-verified `HypothesisSVO`), its
surface is rendered grammatically ON FIRING NEURONS by the composed spiking BROCA ("perhaps the <S> <V-3sg> the
<O>": word order = the per-pool spiking-RATE ranking on a real Izhikevich SimulationBridge, EMERGE-59/61 x the
#3E draw, `_spiking_fluent_surface_derisk`, 6-seed GO), TRANSFORMER-FREE -- replacing the agrammatic host
f-string 'perhaps bear walk foot'. It is re-parse VERIFIED (the same moat the recall path uses) so it recovers
the drawn SVO; a verify miss falls back to the raw flagged template (NEVER a leak); the guess stays clearly
FLAGGED either way. Escape: `BRAIN_SPIKING_MOUTH=0` reverts to the pre-spiking mouth (Qwen / stub / template).
Open ARBITRARY prose the spiking Broca can't frame still falls back to the Qwen mouth -- the banked A1 residual.

LOAD SOURCES (auto-detected from --load):
  * a `developed_brain_io` BUNDLE directory (brain.json + grounded_codes.npz + facts.json + lineage/) -- the
    self-contained "developed brain" the develop loop / a save_developed_brain call writes. THE GENERIC PATH.
  * the SELF-KNOWLEDGE brain: a `_self_knowledge_grounded_codes.json` codes blob (+ the curriculum it was
    developed on) -- the brain reconstructs on the learned codes and re-teaches the curriculum facts. Pass the
    codes .json (or just `--self-knowledge` to use the default codes path).
  * NOTHING / a tiny fallback (the GPU-FREE smoke): build a tiny CPU brain from a handful of facts.

COMMANDS in the chat loop:
  /raw      toggle the brain's OWN neural renderer (no LLM) -- the unvarnished brain (raw recalled triple).
  /facts    list what the brain knows (its stored facts).
  /help     show the commands.
  /quit     exit (also: /exit, /q, Ctrl-D).

SELF-REFERENCE: 'you'/'your'/'I'/'me'/'it' map to the agent 'brain' so 'what are you?' / 'how do you learn?'
resolve against the brain's self-facts.

REUSE-BY-IMPORT, NO `sim/` edit. The OFF-BRIDGE Qwen faculty is the runtime fluent renderer (used when the owner
runs it for real with a free GPU); the GPU-FREE smoke validates the BRAIN side on CPU with the template-stub.

Usage:
    # talk to a saved developed brain (real, with the off-bridge Qwen renderer, free GPU):
    SIM_BACKEND=cupy python -m research.runners.brain_chat_tui --load <developed-brain-dir-or-codes.json>

    # the self-knowledge brain (after `_self_knowledge_demo` saved its codes):
    SIM_BACKEND=cupy python -m research.runners.brain_chat_tui --self-knowledge

    # GPU-FREE smoke (template-stub renderer, scripted stdin):
    SIM_BACKEND=numpy python -m research.runners.brain_chat_tui --stub-renderer --tiny-demo
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.developed_brain_io import (  # noqa: E402
    load_developed_brain, is_developed_brain_bundle,
)

# default self-knowledge artifacts (so `--self-knowledge` works with no path)
_SK_CODES = os.path.join(_REPO, "research", "findings", "raw", "_self_knowledge_grounded_codes.json")
_SK_CURRICULUM = os.path.join(_REPO, "research", "findings", "raw", "_curriculum_self_knowledge.json")


# ============================================================================================================
# OPEN-ENDED GENERATION (production wire-in of the 6-seed-GO #3E "brain owns open-ended generation" faculty).
# On an EXPLICIT open-ended prompt ("what might X ...", "tell me something new about X", "what else about X",
# "guess ..."), the brain VOLUNTEERS a NOVEL grounded proposition via generative replay over its OWN learned
# association graph (the substrate-learned Hebbian co-occurrence on the onebrain path), gated by the validated
# #3E/b2 plausibility + non-contradiction gate, moat-verified (a proposal that contradicts a stored fact or
# passes known-fact retrieval is REJECTED -> abstain), and rendered as a FLAGGED hypothesis ("perhaps a v p").
# `_NOT_OPEN_ENDED` is the sentinel `_parse_open_ended` returns for EVERY non-matching turn, so gate() stays
# byte-identical on the recall / abstain / learn / anaphora paths. `HypothesisSVO` is a list subclass so it
# still behaves as an [a, v, p] triple everywhere a plain gate result flows (JSON, transcript), while render()
# can recognise it and mark it as a guess (never asserted as knowledge).
# ============================================================================================================

_NOT_OPEN_ENDED = object()


class HypothesisSVO(list):
    """A GENERATED, moat-verified HYPOTHESIS triple [a, v, p] (plausible + non-contradictory + NOT a known
    fact). A `list` subclass so it flows unchanged through everything that treats a gate result as [a, v, p]
    (the webapp JSON `recalled_svo`, the smoke transcript), while `render()` recognises it and renders an
    explicit guess rather than an asserted fact."""
    __slots__ = ()


# Each entry: (compiled regex on the lowercased/stripped question, has_topic). Named groups: `topic` (the
# subject to generate about) and, for "what might", an optional `action`. These fixed lead-ins are the WHOLE
# trigger surface — a normal recall ("what does dog chase"), teach ("dog eat bone"), yes/no, or anaphora turn
# matches NONE of them, so it never enters the generation branch (gate() stays byte-identical).
_OPEN_ENDED_PATTERNS = [
    (re.compile(r"^what might (?:a |an |the )?(?P<topic>[a-z]+)(?:\s+(?P<action>[a-z]+))?\b"), True),
    (re.compile(r"^tell me something (?:new |else |more )?about (?:a |an |the )?(?P<topic>[a-z]+)\b"), True),
    (re.compile(r"^what else (?:about|can you (?:tell me|say)(?: something)? about|do you know about) "
                r"(?:a |an |the )?(?P<topic>[a-z]+)\b"), True),
    (re.compile(r"^(?:make something up|imagine something|dream up something) about "
                r"(?:a |an |the )?(?P<topic>[a-z]+)\b"), True),
    (re.compile(r"^guess(?:\s+.*?\babout (?:a |an |the )?(?P<topic>[a-z]+)\b)?"), True),
]


# ============================================================================================================
# Self-reference + a free-text question -> a (kind, cue) the brain answers against its stored SVO facts.
# (The keyword->fact matcher is faithful: it routes a question to the stored fact whose WORDS the question
# mentions, synonym-resolved; an unmatched question ABSTAINS -- the no-confab moat. Ported from the
# self-knowledge demo's router so a plain English question resolves, while carrying ZERO project knowledge.)
# ============================================================================================================

DEFAULT_SELF_ALIASES = {"you", "your", "yours", "i", "me", "my", "it", "its", "yourself", "itself"}

_STOP = {"what", "who", "does", "do", "the", "a", "an", "is", "are", "of", "to", "from", "that", "how",
         "did", "will", "can", "and", "with", "in", "on", "for", "by", "as", "be", "this", "these",
         "those", "there", "here", "prevent", "prevents", "tell", "about", "say", "know", "knows"}

_QUESTION_SYNONYMS = {
    "learn": {"learns", "learning"}, "learns": {"learns"},
    "forget": {"forgetting", "replays", "replay", "remembers"}, "forgetting": {"forgetting"},
    "remember": {"remembers", "memory"}, "memory": {"memory", "remembers", "consolidates"},
    "lie": {"moat", "confabulation", "abstains", "refuses", "honest"},
    "lying": {"moat", "confabulation", "abstains", "refuses", "honest"},
    "guess": {"moat", "confabulation", "refuses", "guessing"},
    "use": {"uses"}, "uses": {"uses"}, "using": {"uses"},
    "teach": {"teaches"}, "teaches": {"teaches"}, "taught": {"teaches"},
    "store": {"stores", "remembers", "composer"}, "speak": {"phrases", "faculty", "answers"},
    "answer": {"answers", "remembers"}, "think": {"uses", "neurons"}, "work": {"uses", "runs"},
    "consolidate": {"consolidates"}, "grow": {"grows", "develops", "tiers"},
    "develop": {"develops", "daily"}, "made": {"has", "uses", "neurons", "spikes"}, "make": {"has", "uses"},
}


class QuestionRouter:
    """Map a free-text question to a stored SVO fact (the GATE cue), resolving self-aliases. Decisive only when a
    CONTENT keyword of the question appears in some fact (a bare self-alias match is not enough -> abstain)."""

    def __init__(self, self_aliases=None):
        self.self_aliases = set(self_aliases) if self_aliases else set(DEFAULT_SELF_ALIASES)

    def _resolve_self(self, word):
        w = word.lower().strip(".,!?")
        return "brain" if w in self.self_aliases else w

    def keywords(self, question):
        toks = [self._resolve_self(t) for t in re.findall(r"[a-zA-Z]+", question.lower())]
        kws = set()
        for t in toks:
            if t in _STOP and t != "brain":
                continue
            kws.add(t)
            kws |= _QUESTION_SYNONYMS.get(t, set())
        return kws, toks

    def match_fact(self, question, stored_facts):
        """Return (gate_svo or None, score). The best stored fact by content-keyword overlap; an identity question
        ('what are you') routes to a defining 'brain has/is/uses ...' fact."""
        kws, toks = self.keywords(question)
        content_kws = kws - {"brain"}
        is_identity_q = ("brain" in kws and not content_kws
                         and any(w in {"be", "are", "is", "am"} for w in toks))
        if is_identity_q:
            # a defining fact about the brain, in preference order (covers base + 3rd-person inflected verbs)
            for want in ("has", "have", "is", "uses", "use"):
                for (a, v, p) in stored_facts:
                    if a == "brain" and v == want:
                        return [a, v, p], 1
            # fall back to ANY fact whose agent is 'brain' (the brain's own self-statement)
            for (a, v, p) in stored_facts:
                if a == "brain":
                    return [a, v, p], 1
        best, best_score = None, 0
        for (a, v, p) in stored_facts:
            ftoks = {a, v, p}
            content_hits = len(content_kws & ftoks)
            brain_hit = 1 if ("brain" in kws and "brain" in ftoks) else 0
            score = content_hits * 10 + brain_hit
            if content_hits >= 1 and score > best_score:
                best, best_score = (a, v, p), score
        return (list(best) if best is not None else None), best_score


# ============================================================================================================
# The fluent renderers (default = the off-bridge Qwen; --stub-renderer = the template-stub, GPU-free).
# Both expose `render_svo(a, v, p) -> (surface, asserted_svo_or_None)`; the TUI gate->constrain->verify wraps them.
# ============================================================================================================

class StubRenderer:
    """The GPU-FREE template-stub faculty (the P3 `TemplateStubFaculty`): renders a gated SVO into a fluent
    surface form CONSTRAINED to the fact's own words, and exposes the canonical content SVO it asserts (what
    VERIFY re-parses). Stands in for the real Qwen renderer in the CPU smoke -- NO model download, deterministic."""

    name = "template-stub (GPU-free)"

    def __init__(self):
        from research.runners._grounded_lang_p3_derisk import TemplateStubFaculty
        self._fac = TemplateStubFaculty()

    def render_svo(self, a, v, p):
        surface, asserted = self._fac.render_svo(a, v, p)
        return surface, asserted


class QwenRenderer:
    """The OFF-BRIDGE Qwen-0.5B grounded-language faculty (the spiking forward, reused-by-import from the
    integration de-risk). Loaded ONCE + kept warm. `render_svo` returns the generated prose + None for the
    asserted SVO (the TUI re-parses the PROSE to recover the asserted content -- the genuine VERIFY of a real
    generative model's output)."""

    name = "off-bridge Qwen-0.5B (spiking forward)"

    def __init__(self, T=16, max_new_tokens=24, seed=42):
        from research.runners._grounded_lang_integration_derisk import SpikingQwenFaculty
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("[tui] WARNING: CUDA not available -- the Qwen renderer will be slow on CPU.", flush=True)
        self._fac = SpikingQwenFaculty(T=T, max_new_tokens=max_new_tokens, seed=seed, device=device)
        self.load_seconds = self._fac.load_seconds

    def render_svo(self, a, v, p):
        surface, _surface_full, _gen_s = self._fac.render_svo(a, v, p)
        return surface, None      # asserted SVO recovered by the TUI's re-parse of the prose

    def render_svo_regen(self, a, v, p):
        surface, _surface_full, _gen_s = self._fac.render_svo_regen(a, v, p)
        return surface, None


# ============================================================================================================
# The chat brain: wraps a loaded conversational agent + the router + the renderer + the gate/constrain/verify.
# ============================================================================================================

class ChatBrain:
    def __init__(self, agent, *, self_aliases=None, renderer=None, verbose_thinking=True):
        # agent is a MultiTurnAgent (preferred, for anaphora) or a BrainConversationalAgent
        self.agent = agent
        self.inner = getattr(agent, "agent", agent)             # the BrainConversationalAgent
        self.is_multiturn = hasattr(agent, "held_referent")     # MultiTurnAgent exposes this
        self.router = QuestionRouter(self_aliases=self_aliases)
        self.renderer = renderer
        self.verbose_thinking = verbose_thinking
        self.raw_mode = False                                   # /raw toggles the brain's own renderer (no LLM)
        # DISCOURSE EVENT TRACKING (2026-07-10): if the agent carries an event register, the console can HEAR a
        # multi-clause discourse and answer "who was doing it before?" across a connective (the D3 event-register arc,
        # deployed on MultiTurnAgent but previously unreachable in any console). Backward-compatible: no register -> off.
        self.has_event_register = self.is_multiturn and getattr(agent, "_event_register", None) is not None
        self._boundary_seen = False          # "who was doing it before?" only has meaning AFTER a discourse boundary
        self._heard_any_clause = False
        # OPEN-ENDED GENERATION (#3E wire-in): a lazily-built, fact-count-cached b2 generative-replay proposer over
        # the brain's OWN association structure. Fires ONLY on an explicit open-ended prompt (see gate()); every
        # other turn is untouched. Config below matches the #3E de-risk operating point (tau = 50th pctile of the
        # positive co-occurrence edges; see `_gen_spiking` below for the DRAW choice).
        self._gen_proposer = None
        self._gen_nfacts = None
        self._gen_tau_pct = 50.0
        self._gen_n_attempts = 400
        self._gen_min_facts = 3
        self._gen_seed = int(getattr(self.inner, "seed", 42)) * 7 + 1
        # the generative DRAW is the b2 HOST oracle (numpy weighted sampling). The b2 SPIKING soft-WTA sampler
        # (SpikingWTASampler) hardcodes the 8x8-taxonomy role pools and KeyErrors on an arbitrary conversational
        # vocab, so it cannot encode a runtime-grown lexicon; the host oracle is the b2-sanctioned numpy path
        # (`_genfrontier_b2` retains it for exactly the numpy-CPU/reproducibility case). The LOAD-BEARING part is
        # the plausibility SIGNAL — the brain's own learned fact-association graph — which is the brain's here.
        self._gen_spiking = False
        # VOCAB-AGNOSTIC SPIKING DRAW organ (B1 burn-down, 2026-08-13): converts the #3E generative DRAW above from
        # the host oracle to a genuinely-SPIKING soft-WTA read off cp_firing_states. The b2 taxonomy SpikingWTASampler
        # KeyErrors on a runtime lexicon (see comment above); this organ induces role pools from the brain's OWN
        # stored-fact concepts (no taxonomy) and pre-injects a taxonomy-free VocabAgnosticSpikingSampler. Built LAZILY
        # on the first open-ended generation turn (per-ChatBrain, avoiding process-singleton cache thrashing across
        # sessions/brains), so a session that never generates never imports it and its non-generation turns are
        # untouched. Default-ON; BRAIN_SPIKING_DRAW=0 leaves `_gen_spiking=False` -> the host oracle draw
        # (byte-identical). See research/runners/vocab_agnostic_spiking_generation_production_organ.py.
        self._spiking_draw_organ = None
        # the brain's stored facts (string-only roles) + content-token sets for the VERIFY re-parse
        self._refresh_facts()

    def _refresh_facts(self):
        comp = self.inner.composer
        self.stored_facts = [(f.get("agent"), f.get("action"), f.get("patient")) for f, _ in comp.kb
                             if all(isinstance(f.get(r), str) for r in ("agent", "action", "patient"))]
        self.agents_set = {a for a, _, _ in self.stored_facts}
        self.actions_set = {v for _, v, _ in self.stored_facts}
        self.patients_set = {p for _, _, p in self.stored_facts}
        from research.runners._grounded_lang_integration_derisk import _build_inflection_map
        self.inflect = _build_inflection_map(sorted(self.actions_set))

    # --- the GATE: a free-text question -> a verified stored SVO fact, or None (abstain) ---
    def gate(self, question):
        """Resolve the question to a stored fact and VERIFY it against the spiking recall. Returns
        (gate_svo or None). An anaphor in the question is resolved from the discourse WM (multi-turn)."""
        # OPEN-ENDED GENERATION (#3E production wire-in) — fires ONLY on an EXPLICIT open-ended prompt pattern
        # ("what might X ...", "tell me something new about X", "what else about X", "guess ..."). On a match, the
        # brain VOLUNTEERS a novel grounded proposition via generative replay over its own learned association
        # graph, moat-verified + FLAGGED as a hypothesis (or abstains -> None; never confabulates). For EVERY
        # OTHER question `_parse_open_ended` returns the `_NOT_OPEN_ENDED` sentinel and gate() falls through to the
        # unchanged recall/abstain/learn/anaphora pipeline below — byte-identical.
        oe = self._parse_open_ended(question)
        if oe is not _NOT_OPEN_ENDED:
            return self._generate_hypothesis(*oe)
        # resolve anaphora in the question FIRST (multi-turn): replace a leading 'it'/'that'/'they' with the held
        # referent, so a follow-up 'what does it eat' uses the prior turn's referent.
        acq = self._maybe_acquire(question)      # IN-LOOP LEARNING (production path): an SVO ASSERTION is TAUGHT here in
        if acq is not None:                      # gate() so the /api/brain-chat endpoint (which calls gate(), NOT
            return acq                           # answer()) reaches it; gate returns the acquired SVO -> render confirms.
        q = self._resolve_anaphora(question)
        anaphora_used = (q != question)          # the extracted agent came from the (noisy) discourse WM, not the user
        # SUBSTRATE-FIRST recall (production-integration #2, in-loop learning). For a well-formed "what does AGENT
        # ACTION?" question where AGENT+ACTION are known, recall the patient FROM THE SPIKING SUBSTRATE
        # (`inner.what_does`) — which is ROLE-AWARE (it queries the specific (agent, action) binding, not the host
        # router's role-blind keyword overlap) AND sees a fact HEARD this conversation. `what_does` returns the stored
        # patient only if the binding is genuinely in the substrate, so this cannot confabulate (the no-confab moat
        # holds). The host QuestionRouter remains the fallback for self/identity questions and anything not in this form.
        sub = self._substrate_recall(q)
        if sub == "__ABSTAIN__" and not anaphora_used:
            return None                          # DIRECT well-formed query, substrate has no fact -> honest abstain
                                                 # (fixes the host-router keyword CONFAB, e.g. "what does fish fly?").
        if sub not in (None, "__ABSTAIN__"):     # anaphora abstain falls through: the WM referent may be noisy, so let
            return sub                           # the host router try (its keyword match masks a bad WM pick).
        gate_svo, _score = self.router.match_fact(q, self.stored_facts)
        if gate_svo is None:
            return None
        a, v, p = gate_svo
        # VERIFY the matcher's pick against the brain's SPIKING recall (the answer must be the spiking memory's)
        recalled = self.inner.what_does(a, v)
        if recalled == p:
            # write the answer's salient referent (the PATIENT/object) into the discourse WM so a NEXT-turn pronoun
            # resolves to it -- exactly as MultiTurnAgent.hear() writes only the patient. We treat a CONCRETE entity
            # (one that is itself the AGENT of some fact -- i.e. something the brain can say more about) as the
            # discourse referent; this matches the validated single-referent anaphora pattern (a fresh referent
            # dominates the WM) and avoids polluting the WM with abstract patients (e.g. 'spikes'/'words') that are
            # not salient pronoun antecedents. The no-confab moat is unaffected.
            if isinstance(p, str) and p in self.agents_set:
                self._note_referent(p)
            return [a, v, p]
        return None

    # --- OPEN-ENDED GENERATION (#3E: the brain VOLUNTEERS novel grounded propositions via generative replay) ---
    def _parse_open_ended(self, question):
        """Detect an EXPLICIT open-ended generation prompt. Returns `(topic, action)` on a match (either may be
        None: a bare 'guess' -> free generation), else the `_NOT_OPEN_ENDED` sentinel. Deliberately conservative:
        only the fixed lead-ins in `_OPEN_ENDED_PATTERNS` match, so a normal recall/teach/yes-no/anaphora turn
        never enters the generation branch and gate() stays byte-identical."""
        ql = question.lower().strip()
        for rx, _has_topic in _OPEN_ENDED_PATTERNS:
            m = rx.match(ql)
            if m is None:
                continue
            gd = m.groupdict()
            topic = (gd.get("topic") or "").strip(".,!? ") or None
            action = (gd.get("action") or "").strip(".,!? ") or None
            if topic in self.router.self_aliases:      # 'you'/'it' -> the brain's self-facts
                topic = "brain"
            return (topic, action)
        return _NOT_OPEN_ENDED

    def _build_generation_proposer(self):
        """Build (and fact-count cache) the #3E/b2 `GenerativeReplayProposer` over the brain's OWN association
        structure. The plausibility graph P is the brain's CLEAN concept co-occurrence over its stored facts (the
        agent's association structure — every fact's agent/action/patient co-occur), which is what the dlPFC
        `_assoc_graph` learned graph approximates but WITHOUT that graph's dense reserve-slot noise (which floods
        implausible recombinations) and WITH the runtime-taught facts the fixed-vocab `_learned_assoc` never sees.
        tau = the 50th percentile of the positive edges (the #3E operating point → 'related' = co-occurred). This
        is the same host-computed selectional-preference plausibility signal #3E used (there a corpus PPMI; here
        the brain's own heard facts), gating the reused b2 proposer's replay. Returns the proposer, or None when
        the brain knows too few facts (-> the caller abstains, never confabulates)."""
        facts = list(self.stored_facts)
        if len(facts) < self._gen_min_facts:
            return None
        if self._gen_proposer is not None and self._gen_nfacts == len(facts):
            return self._gen_proposer
        # clean concept co-occurrence over the brain's stored facts (agent/action/patient of each fact co-occur).
        graph = {}
        for a, v, p in facts:
            cs = [c for c in (a, v, p) if isinstance(c, str)]
            for x in cs:
                for y in cs:
                    if x != y:
                        graph.setdefault(x, {})[y] = graph.get(x, {}).get(y, 0.0) + 1.0
        vocab = sorted(graph.keys())
        if len(vocab) < 3:
            return None
        row = {w: i for i, w in enumerate(vocab)}
        P = np.zeros((len(vocab), len(vocab)), dtype=float)
        for a, nbrs in graph.items():
            for b, w in nbrs.items():
                P[row[a], row[b]] = float(w)          # symmetric by construction of the co-occurrence
        pos = P[P > 0]
        if pos.size == 0:
            return None
        tau = float(np.percentile(pos, self._gen_tau_pct))
        from research.runners._genfrontier_b2_generative_replay_derisk import GenerativeReplayProposer
        # the proposer reads the SAME composer the brain answers through (so a generated proposition must not
        # contradict a stored fact, and must never pass known-fact retrieval). negated=[] here: the composer's own
        # `ask_yes_no` (which the proposer's non-contradiction gate reads) still catches any stored negation.
        self._gen_proposer = GenerativeReplayProposer(
            self.inner.composer, facts, [], P, row, tau,
            np.random.default_rng(self._gen_seed), use_spiking_sampler=self._gen_spiking)
        self._gen_nfacts = len(facts)
        return self._gen_proposer

    def _generate_hypothesis(self, topic=None, action=None, n_attempts=None):
        """GENERATE a novel grounded proposition (the #3E faculty), optionally about `topic` (its agent) and/or
        `action`. Draws role-fillers with the reused b2 proposer's OWN weighted sampler (`_sample_weighted` /
        `_weight_partner`), gates each candidate with the reused b2 `_plausible` (selectional-preference over the
        brain's learned association graph) + `_contradicts` (non-contradiction vs the composer's store), then
        MOAT-VERIFIES it (not a degenerate self-loop; matches the requested topic/action; and — the no-confab
        guarantee — must NOT pass known-fact retrieval: `what_does` != patient AND `is_it_true` == 'unknown').
        EARLY-STOPS at the first passing proposal (so a turn runs only a few spiking moat queries, not a full
        exhaustive replay) and returns it as a FLAGGED `HypothesisSVO`; returns None (honest abstain) when no
        plausible grounded proposal exists. An unknown topic (not a known agent) ABSTAINS — the brain does not
        invent about what it has never heard of."""
        if topic is not None and topic not in self.agents_set:
            return None                                # unknown subject -> abstain (no confabulation)
        prop = self._build_generation_proposer()
        if prop is None:
            return None
        # Route the #3E generative DRAW through the VOCAB-AGNOSTIC spiking soft-WTA (default-ON, B1 burn-down): the b2
        # taxonomy sampler KeyErrors on runtime vocab, so install() induces role pools from the brain's OWN stored-fact
        # concepts and pre-injects a taxonomy-free VocabAgnosticSpikingSampler onto `prop` (flips use_spiking_sampler=True).
        # The UNCHANGED loop below then draws on FIRING NEURONS (prop._sample_weighted -> the injected sampler ->
        # draw_from_weights reads cp_firing_states). BRAIN_SPIKING_DRAW=0 -> install() is a no-op -> the host oracle draw
        # (byte-identical). BRAIN_SPIKING_DRAW_LESION=1 -> likelihood ablated (uniform drive) -> plausibility collapses.
        # Every downstream gate (_plausible / _contradicts) + the #3E moat verify are UNTOUCHED.
        if self._spiking_draw_organ is None:
            from research.runners.vocab_agnostic_spiking_generation_production_organ import (
                VocabAgnosticSpikingDrawOrgan,
            )
            self._spiking_draw_organ = VocabAgnosticSpikingDrawOrgan(seed=self._gen_seed)
        self._spiking_draw_organ.install(prop)
        if action is not None and action not in self.actions_set:
            action = None                              # a requested action the brain doesn't know -> don't hard-filter
        agents = [topic] if topic is not None else list(prop.agents)
        if not agents or not prop.actions or not prop.patients:
            return None
        n = int(self._gen_n_attempts if n_attempts is None else n_attempts)
        rng = prop.rng
        seen = set()
        for _ in range(n):
            a = agents[0] if len(agents) == 1 else agents[int(rng.integers(len(agents)))]
            ac = action if action is not None else prop._sample_weighted(
                prop.actions, prop._weight_partner((a,), prop.actions))
            p = prop._sample_weighted(prop.patients, prop._weight_partner((a, ac), prop.patients))
            triple = (a, ac, p)
            if a == p or triple in seen or triple in prop.all_stored:
                continue                               # degenerate / repeat / a stored fact (only NOVEL counts)
            seen.add(triple)
            if not prop._plausible(a, ac, p):          # b2 selectional-preference plausibility gate (reused)
                continue
            if prop._contradicts(a, ac, p):            # b2 non-contradiction gate (reads the composer's ask_yes_no)
                continue
            # MOAT VERIFY (the #3E hypothesis-not-known guarantee): a HYPOTHESIS never passes as a known fact.
            if self.inner.what_does(a, ac) == p or self.inner.is_it_true(a, ac, p) != "unknown":
                continue
            return HypothesisSVO([a, ac, p])
        return None

    def _resolve_anaphora(self, question):
        """If the question's first content token is a pronoun and the discourse WM holds a referent, substitute it
        (multi-turn anaphora). Only the MultiTurnAgent has a WM loop; otherwise pass the question through."""
        if not self.is_multiturn:
            return question
        anaphors = {"it", "that", "they", "them", "this"}
        toks = question.split()
        for i, t in enumerate(toks):
            tl = t.lower().strip(".,!?")
            if tl in anaphors:
                ref = self.agent.held_referent()[0]
                if ref is not None:
                    toks[i] = ref
                    return " ".join(toks)
        return question

    def _maybe_generate(self, question):
        """GENERATION: for an open-ended TOPIC prompt ('tell me about X' / 'describe X' / 'what about X'), VOLUNTEER what
        the brain knows about X by CHAINING ASSOCIATIONS on the substrate — describe(X) plus the dlPFC spiking
        `elaborate` (spreading-activation content-selection) to a related concept and describe THAT. This is generation
        from the brain's own knowledge, beyond single-fact recall. Returns (answer, abstained) or None. No confab:
        describe() returns None for an unknown topic (-> falls through to abstain)."""
        ql = question.lower().strip().rstrip("?. ")
        topic = None
        for pat in ("tell me about ", "describe ", "what about ", "what do you know about ", "say something about "):
            if ql.startswith(pat):
                topic = ql[len(pat):].strip().split()[-1] if ql[len(pat):].strip() else None
                break
        if not topic:
            return None
        topic = topic.strip(".,!?")
        if topic in self.router.self_aliases:
            topic = "brain"
        try:
            primary = self.inner.describe(topic)
        except Exception:
            primary = None
        if not primary:
            return None                          # unknown topic -> let the pipeline abstain (no confabulation)
        parts = [primary]
        try:                                     # ONE associative hop via the dlPFC spiking spreading-activation control
            assoc = self.inner.elaborate(topic)
            if assoc and assoc != topic:
                more = self.inner.describe(assoc)
                if more and more != primary:
                    parts.append(more)
        except Exception:
            pass
        return " ".join(p.rstrip(".") + "." for p in parts), False

    def _maybe_acquire(self, question):
        """IN-LOOP LEARNING acquisition: if the input is a declarative 3-word SVO ASSERTION (not a question), TEACH it to
        the spiking substrate (`inner.hear` -> composer.store with runtime code allocation for any new word) and refresh
        the recallable vocabulary, then acknowledge. Returns (answer, abstained) or None (not an assertion). This is what
        lets the owner grow the brain's knowledge by talking to it."""
        q = question.strip()
        ql = q.lower()
        if "?" in q or ql.split()[:1] and ql.split()[0] in (
                "what", "who", "whom", "where", "when", "why", "how", "is", "are", "was", "were", "does", "do", "did"):
            return None
        # ── NON-CONTRADICTION STORE-SIDE (Gate-B, B3, 2026-08-12) ──────────────────────────────────────────
        # So the non-contradiction gate has NEGATIONS to fire against (today the console stores ZERO negations — the
        # legacy path below hard-codes polarity="AFFIRM" and only acquires an EXACTLY-3-whitespace-token input),
        # acquire a heard assertion with its DETECTED polarity via the B3 organ's extractor: it strips negation cues +
        # function words to expose the 3-token SVO content and tags a heard negation ("the dog does not eat grass") as
        # NEGATE, using the SAME function-word-strip the gate's recall uses (so store + recall AGREE). Additive +
        # guarded: falls back to the EXACT legacy 3-token / AFFIRM path when B3 is unavailable OR disabled
        # (BRAIN_NONCONTRADICTION_GATE=0) -> byte-identical acquisition. (This edits the host conversational scaffold,
        # NOT sim/.)
        try:
            import research.runners.b3_noncontradiction_production_organ as _b3nc
            _b3nc_on = _b3nc.noncontradiction_enabled()
        except Exception:
            _b3nc = None
            _b3nc_on = False
        if _b3nc_on:
            parsed = _b3nc.extract_polar_assertion(q)   # (agent, action, patient, polarity) or None (out of scope)
            if parsed is None:
                return None
            a, v, p, pol = parsed
            try:
                self.inner.hear("%s %s %s" % (a, v, p), polarity=pol)   # a heard NEGATION stores as NEGATE
            except Exception:
                return None
            self._refresh_facts()
            return [a, v, p]
        # (B3 unavailable / disabled) — the EXACT legacy path (byte-identical acquisition)
        toks = [t.strip(".,!?") for t in q.split() if t.strip(".,!?")]
        if len(toks) != 3:                       # the minimal SVO assertion the parser handles
            return None
        a, v, p = toks
        try:
            self.inner.hear("%s %s %s" % (a, v, p), polarity="AFFIRM")
        except Exception:
            return None
        self._refresh_facts()                    # pick up the new fact -> agents_set/actions_set now include it
        return [a, v, p]                         # the acquired SVO; gate() returns it so the endpoint renders a confirm

    def _neural_question_parse(self, content):
        """CHOOSE (#1) — comprehend the question's (agent, action) NEURALLY. Present the stripped content words
        (position-padded to SVO, the queried patient a placeholder) to the ON-BRAIN BridgeParser, whose (position,
        voice)->role conjunction FIRES the role assignment on Izhikevich neurons — the SAME parser `hear()` uses to
        comprehend a stored sentence. Returns (agent, action) or None. This replaces the host first-known-token /
        positional heuristic so the question COMPREHENSION is on the substrate, not a Python vocabulary lookup. Requires
        the composer to carry a parser (the onebrain default does); returns None otherwise, so the caller falls back to
        the host heuristic (the rf escape path). Lesioning the parser -> role_of returns junk -> None -> the fact is not
        recalled (the load-bearing test)."""
        parser = getattr(getattr(self.inner, "composer", None), "parser", None)
        if parser is None or len(content) < 2:
            return None
        padded = [content[0], content[1], "__q__"]           # SVO with the queried patient a placeholder
        try:
            role_map = {}
            for pos in range(3):
                role_map[parser.role_of(pos)] = padded[pos]  # each position's role FIRES on the parser ensembles
        except Exception:
            return None
        a, v = role_map.get("agent"), role_map.get("action")
        if not (a and v) or a == v or a == "__q__" or v == "__q__":
            return None                                       # degenerate/lesioned parse -> let the caller fall back
        return a, v

    def _substrate_recall(self, question):
        """IN-LOOP LEARNING recall: resolve (agent, action) from the question and recall the patient FROM THE SPIKING
        SUBSTRATE (`inner.what_does`), so a fact heard this conversation is answerable even though it is not in the
        build-time host snapshot. Returns [a, v, p] or None. No confabulation: `what_does` returns nothing unless the
        binding is genuinely stored. The (agent, action) COMPREHENSION is NEURAL (the on-brain BridgeParser) on the
        onebrain default, with a host heuristic fallback (the rf escape path)."""
        _STOP = {"what", "who", "whom", "does", "do", "did", "is", "are", "was", "were", "the", "a", "an",
                 "to", "it", "that", "this", "they", "them", "of", "about"}
        toks = [t.lower().strip(".,!?") for t in question.split()]
        content = [t for t in toks if t and t not in _STOP]
        # CHOOSE (#1): the on-brain parser OWNS a factual-SVO-shaped question (>=2 content words, none a self-alias).
        # When it comprehends -> (agent, action) on FIRING neurons; when it DECLINES on such a question -> honest
        # "__ABSTAIN__" (do NOT fall to the host router's role-blind keyword confab). This makes the comprehension
        # genuinely on the substrate + LESION-LOAD-BEARING: lesion the parser -> role_of returns junk -> the factual
        # CHOOSE abstains (the answer CHANGES). A self/identity/short question (or the rf escape — NO parser) keeps the
        # host heuristic (prefer a KNOWN agent/action, else STRUCTURAL position) + the router fallback in gate().
        has_self_alias = any(t in self.router.self_aliases for t in content)
        parser_present = getattr(getattr(self.inner, "composer", None), "parser", None) is not None
        if parser_present and len(content) >= 2 and not has_self_alias:
            nq = self._neural_question_parse(content)
            if nq is None:
                return "__ABSTAIN__"             # factual-shaped question the on-brain parser could not comprehend -> abstain
            a, v = nq
        else:
            a = next((t for t in content if t in self.agents_set), None) or (content[0] if content else None)
            v = next((t for t in content if t in self.actions_set), None) or (content[1] if len(content) > 1 else None)
        if not (a and v) or a == v:
            return None                          # could not extract a query -> let the host router try (self/identity)
        # a self/identity query (a or v is a self-alias) is the host router's job, not the substrate's.
        if a in self.router.self_aliases or v in self.router.self_aliases:
            return None
        try:
            p = self.inner.what_does(a, v)
        except Exception:
            return None
        if not p:
            # a WELL-FORMED question the substrate cannot answer -> ABSTAIN honestly. Do NOT fall through to the host
            # router's role-blind keyword guess (that is the confabulation the CHOOSE gap produced, e.g. "what does fish
            # fly?" -> "cat eat fish"). This retires the host router's CONFAB for well-formed queries.
            return "__ABSTAIN__"
        if isinstance(p, str) and p in self.agents_set:
            self._note_referent(p)
        return [a, v, p]

    def _note_referent(self, word):
        """Write a referent into the discourse WM (multi-turn), so a later pronoun resolves to it."""
        if self.is_multiturn and isinstance(word, str):
            try:
                self.agent._write_referent(word)
            except Exception:
                pass

    # --- the CONSTRAIN + VERIFY render of a gated fact into fluent prose ---
    def render(self, gate_svo):
        """Render the gated SVO into a fluent sentence (CONSTRAIN) and VERIFY the content re-parses to the gated
        fact. Returns the verified fluent string, or the brain's raw triple on a verify miss / raw mode / no
        renderer. NEVER emits unverified generative prose as the answer."""
        # OPEN-ENDED GENERATION (#3E): a moat-verified HYPOTHESIS is rendered as an EXPLICIT, clearly-FLAGGED guess
        # (the honesty boundary is a deliverable). We now speak it as FLUENT prose via the mouth — SVO-VERIFIED so
        # the mouth cannot swap the content — but framed 'Maybe ... -- that's a guess ...' so it can never be
        # mistaken for asserted knowledge; a mouth-verify miss / GPU-free host falls back to the raw flagged
        # template. The proposal was already gated (plausible + non-contradictory) + moat-verified in gate().
        if isinstance(gate_svo, HypothesisSVO):
            return self.render_hypothesis(gate_svo)
        a, v, p = gate_svo
        if self.raw_mode or self.renderer is None:
            return self._raw(gate_svo)
        surface, asserted = self.renderer.render_svo(a, v, p)
        if self._verify(surface, asserted, gate_svo):
            return surface
        # a generative renderer can DRIFT: try a tighter re-prompt once (if supported), else speak the raw fact
        if hasattr(self.renderer, "render_svo_regen"):
            surface2, asserted2 = self.renderer.render_svo_regen(a, v, p)
            if self._verify(surface2, asserted2, gate_svo):
                return surface2
        return self._raw(gate_svo) + "   [unverified render -> spoke the brain's raw fact]"

    def _verify(self, surface, asserted, gate_svo):
        """VERIFY: re-parse the rendered content back into an SVO and require it to MATCH the gated fact. For the
        stub, `asserted` is the canonical content SVO; for Qwen, `asserted` is None -> re-parse the PROSE."""
        if asserted is None:
            from research.runners._grounded_lang_integration_derisk import _extract_svo_from_prose
            asserted = _extract_svo_from_prose(surface, self.agents_set, self.actions_set,
                                               self.patients_set, self.inflect)
            if asserted is None:
                return False
        parsed = self.inner.parse(asserted, voice="active")
        rsvo = [parsed.get("agent"), parsed.get("action"), parsed.get("patient")]
        return rsvo == list(gate_svo)

    # --- the CLAIM-LEVEL moat generalization (de-risked ClaimEntailmentVerifier, wired for the multi-fact turn) ---
    @staticmethod
    def _claim_moat_enabled():
        """The escape hatch. `BRAIN_CLAIM_MOAT=0` reverts to the exact single-triple `_verify` per rendered
        sentence (the pre-generalization behaviour). Any other value (incl. unset) keeps the claim-level moat on
        -- the production default, so genuinely free-form MULTI-CLAUSE grounded prose survives the moat."""
        return os.environ.get("BRAIN_CLAIM_MOAT", "1") != "0"

    def _build_claim_verifier(self, gated_facts):
        """Build (and lazily cache on the gated SET) the de-risked ClaimEntailmentVerifier, REUSED BY IMPORT --
        NOT reimplemented. It decomposes multi-clause prose into its asserted proposition set, role-parses EACH
        clause on THIS brain's on-substrate parser (`self.inner.parse`, the same spiking role parser the
        single-triple `_verify` uses), and accepts IFF every asserted proposition is entailed by `gated_facts`
        (with the flagged-hypothesis carve-out + the coverage invariant). Returns None when the set is empty or
        has a role-permutation collision (the verifier's own well-formedness guard) -> the caller falls back to
        the single-triple `_verify` in the SAFE direction."""
        key = frozenset(tuple(f) for f in gated_facts
                        if isinstance(f, (list, tuple)) and len(f) == 3
                        and all(isinstance(x, str) for x in f))
        if not key:
            return None
        cache = getattr(self, "_claim_verifier_cache", None)
        if cache is None:
            cache = self._claim_verifier_cache = {}
        if key in cache:
            return cache[key]
        from research.runners._moat_claim_entailment_derisk import (
            ClaimEntailmentVerifier, VERB_SYNONYMS, _build_inflection_map)
        gated = [list(f) for f in key]
        nouns = {t for f in gated for t in (f[0], f[2])}
        verbs = {f[1] for f in gated}
        inflect = _build_inflection_map(sorted(verbs))
        try:
            ver = ClaimEntailmentVerifier(self.inner, gated, nouns, verbs, VERB_SYNONYMS, inflect)
        except AssertionError:
            ver = None                                   # role-permutation collision -> fall back (SAFE)
        cache[key] = ver
        return ver

    def _verify_claim_set(self, surface, gated_facts):
        """CLAIM-LEVEL VERIFY: does the rendered PROSE `surface` assert ONLY facts entailed by `gated_facts` (the
        set the turn gathered)? Returns (accepted: bool, result: dict) via the de-risked ClaimEntailmentVerifier,
        or (None, None) when the claim moat is DISABLED (escape flag) or the verifier is unbuildable -> the caller
        must fall back to the single-triple `_verify` (byte-identical old behaviour). This is a strict SUPERSET of
        `_verify`: a single grounded sentence still passes, AND multi-clause grounded prose passes, while any
        response carrying even one ungrounded/contradictory asserted clause is rejected (0 confab leaks)."""
        if not self._claim_moat_enabled():
            return None, None
        ver = self._build_claim_verifier(gated_facts)
        if ver is None:
            return None, None
        res = ver.verify(surface)
        return bool(res["accepted"]), res

    def _raw(self, gate_svo):
        """The brain's OWN renderer: the raw recalled triple as a plain sentence (no LLM)."""
        return " ".join(str(x) for x in gate_svo)

    # --- OPEN-ENDED GENERATION (#3E): render a generated HYPOTHESIS as a FLUENT, clearly-FLAGGED guess ---
    def render_hypothesis(self, hyp):
        """Render a GENERATED, moat-verified HYPOTHESIS (a #3E novel proposition) as a clearly-FLAGGED guess.
        Prefer FLUENT prose via the mouth, framed 'Maybe <fluent> -- that's a guess ...', VERIFYING that the
        fluent sentence re-parses to the SAME (a, v, p) the hypothesis asserts (so the mouth cannot swap the
        content). On a verify miss / no renderer / raw mode, fall back to the raw FLAGGED template 'perhaps a v
        p'. The guess is NEVER surfaced as an asserted fact -- the honesty framing is explicit in the surface text
        either way."""
        return self.render_hypothesis_verified(hyp)[0]

    def render_hypothesis_verified(self, hyp):
        """As `render_hypothesis`, but also report whether the FLUENT surface VERIFIED (True) or the raw flagged
        template FALLBACK was used (False). Returns (surface, fluent_verified). The VERIFY is the same re-parse the
        recall path uses: the fluent sentence must carry the hypothesis's exact (a, v, p).

        SURFACE ORDER OF PREFERENCE (production default): the BRAIN-NATIVE SPIKING BROCA mouth renders a supported
        structured hypothesis (a transitive SVO) grammatically ON FIRING NEURONS -- word order = the per-pool
        spiking-RATE ranking on a real Izhikevich SimulationBridge (EMERGE-59/61, composed with the #3E draw in
        `_spiking_fluent_surface_derisk`, 6-seed GO) -- transformer-FREE, replacing the agrammatic host f-string.
        It is re-parse VERIFIED (the moat) so it recovers the DRAWN SVO; a verify miss falls back to the raw flagged
        template (NEVER a leak). The escape flag `BRAIN_SPIKING_MOUTH=0`, OR content the spiking Broca can't frame
        (open/multi-word prose), falls through to the pre-spiking mouth (off-bridge Qwen / template-stub / raw
        flagged template) -- the documented A1 residual (open arbitrary prose = the banked deep-context wall)."""
        a, v, p = hyp
        template = self._hypothesis_template(a, v, p)
        # (1) BRAIN-NATIVE SPIKING MOUTH -- the production default for a structured (transitive SVO) hypothesis.
        if self._spiking_mouth_enabled() and self._hyp_frame_supported(hyp):
            spk = self._render_hypothesis_spiking(hyp)
            if spk is not None:
                return spk, True                      # grammatical, moat-verified, flagged -- produced on spikes
            return template, False                    # spiking verify miss -> honest flagged fallback (NO leak)
        # (2) the PRE-SPIKING mouth (escape flag BRAIN_SPIKING_MOUTH=0, or content the spiking Broca can't frame).
        if self.raw_mode or self.renderer is None:
            return template, False                    # GPU-free / --raw: the honest raw flagged guess
        surface, asserted = self.renderer.render_svo(a, v, p)
        if self._verify(surface, asserted, hyp):
            return self._frame_guess(surface), True
        # a generative mouth can DRIFT: one tighter re-prompt (if supported), else the raw flagged template
        if hasattr(self.renderer, "render_svo_regen"):
            surface2, asserted2 = self.renderer.render_svo_regen(a, v, p)
            if self._verify(surface2, asserted2, hyp):
                return self._frame_guess(surface2), True
        return template, False                        # mouth swapped/garbled the content -> honest flagged fallback

    @staticmethod
    def _hypothesis_template(a, v, p):
        """The raw FLAGGED-guess surface (GPU-free fallback / a mouth-verify miss). Byte-identical to the pre-fluent
        template so the moat framing is unchanged when the mouth is unavailable."""
        return f"perhaps {a} {v} {p}  [a guess from what I've learned -- not something I was taught]"

    @staticmethod
    def _frame_guess(surface):
        """Frame a VERIFIED fluent sentence as an EXPLICIT guess (the honesty boundary is a deliverable): the
        fluent content is kept verbatim, lower-cased into a 'Maybe ...' lead so it can never be read as asserted
        knowledge, with the not-taught disclaimer appended."""
        g = surface.strip().rstrip(".")
        if g[:1].isupper():
            g = g[0].lower() + g[1:]
        return f"Maybe {g} -- that's a guess from what I've learned, not something I was taught."

    # --- BRAIN-NATIVE SPIKING BROCA mouth for the GENERATE channel (#3E surface; REUSE-BY-IMPORT, NO sim/ edit) ---
    @staticmethod
    def _spiking_mouth_enabled():
        """Escape hatch. `BRAIN_SPIKING_MOUTH=0` reverts the GENERATE-channel hypothesis SURFACE to the pre-spiking
        mouth (off-bridge Qwen / template-stub / raw flagged template) -- byte-identical to the pre-wire behaviour.
        Any other value (incl. unset) keeps the brain-native SPIKING Broca render ON -- the production default."""
        return os.environ.get("BRAIN_SPIKING_MOUTH", "1") != "0"

    @staticmethod
    def _hyp_frame_supported(hyp):
        """True iff the hypothesis fits a structured frame the spiking BROCA supports (a transitive SVO with single-
        WORD alphabetic roles, subject != object). Open/arbitrary content (empty / multi-word roles) is NOT frameable
        here -> the caller falls back to the current mouth (the documented A1 residual = open arbitrary prose)."""
        if not isinstance(hyp, (list, tuple)) or len(hyp) != 3:
            return False
        a, v, p = hyp
        return all(isinstance(x, str) and x.isalpha() for x in (a, v, p)) and a != p

    def _spiking_broca_producer(self):
        """Lazily build + cache the reused spiking BROCA clause producer (EMERGE-59/61 order read-out on a real
        Izhikevich SimulationBridge). Built ONCE (bridge build + competitive-queuing learn of the 6-slot hedged-
        transitive order, ~0.35 s CPU); each hypothesis then emits in ~5 ms via the EMERGE-61 inter-utterance
        wash-out (the producer's `emit` restores the post-init substrate state before every clause). REUSE-BY-IMPORT
        from `_spiking_fluent_surface_derisk` -- NO reimplementation, NO sim/ edit."""
        prod = getattr(self, "_spk_producer", None)
        if prod is None:
            from research.runners._spiking_fluent_surface_derisk import SpikingClauseProducer, HEDGED_TRANSITIVE
            seed = int(getattr(self.inner, "seed", 42))
            prod = SpikingClauseProducer(seed)
            prod.learn(len(HEDGED_TRANSITIVE))         # competitive-queuing learn of the hedged-transitive slot order
            self._spk_producer = prod
        return prod

    def _render_hypothesis_spiking(self, hyp):
        """Render a GENERATED hypothesis SVO grammatically ON FIRING NEURONS (the composed spiking BROCA render:
        'perhaps the <S> <V-3sg> the <O>', word order = the per-pool spiking-RATE ranking on the SimulationBridge),
        then re-parse VERIFY (the SAME moat the recall path uses -> `_verify` re-parses the surface PROSE) that the
        rendered sentence recovers the DRAWN (a, v, p). Returns the framed FLAGGED guess on a verify PASS, or None on
        a verify miss (-> the caller uses the raw flagged template; NEVER a leak). Transformer-FREE: this path never
        touches the Qwen mouth."""
        from research.runners._spiking_fluent_surface_derisk import HEDGED_TRANSITIVE
        from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import emerge_v3
        a, v, p = hyp
        dctx = {"subject": a, "verb_3sg": emerge_v3(v), "object": p}
        surface = " ".join(self._spiking_broca_producer().emit(HEDGED_TRANSITIVE, dctx))
        if self._verify(surface, None, hyp):           # the moat: the spiking sentence must recover THIS (a, v, p)
            return self._frame_guess_spiking(surface)
        return None                                    # verify miss -> caller falls back to the flagged template

    @staticmethod
    def _frame_guess_spiking(surface):
        """Frame the SPIKING-Broca hedged surface ('perhaps the S V-3sg the O') as an EXPLICIT flagged guess. The
        surface already leads with the epistemic hedge 'perhaps' (the spiking Broca's own CONN slot); we append the
        SAME not-taught disclaimer the raw template uses, so the honesty framing is identical whichever mouth spoke."""
        return f"{surface.strip()}  [a guess from what I've learned -- not something I was taught]"

    # --- discourse event tracking (who is doing it now / who was doing it before) ---
    def _discourse_turn(self, line):
        """Route a discourse turn if the agent carries an event register: hear an SVO clause (updating the running
        event), or answer 'who was doing it before/now?'. Returns (answer, abstained) or None (not a discourse turn)."""
        if not self.has_event_register:
            return None
        reg = self.agent._event_register
        ql = line.lower().strip().rstrip(".!?").strip()
        toks = ql.split()
        if ql in ("who was doing it before", "who was before", "who did it before", "who was doing that before"):
            if not self._boundary_seen:      # no earlier event yet -> the no-confab moat abstains
                return ("I don't know who was doing it before -- no earlier event yet.", True)
            a = self.agent.who_agent_before()
            return (f"{a} was.", False) if a else ("I don't know who was doing it before.", True)
        if ql in ("who is doing it now", "who is doing it", "who is now", "who did it now"):
            if not self._heard_any_clause:
                return ("I don't know who is doing it now -- nothing said yet.", True)
            a = self.agent.who_agent_now()
            return (f"{a} is.", False) if a else ("I don't know who is doing it now.", True)
        # a discourse SVO clause (optionally with a leading connective): subject action object, where the action is
        # a known verb OR the subject is a pronoun the register tracks -> HEAR it (fold into the running event).
        w = list(toks)
        had_connective = bool(w) and w[0] in ("then", "but", "meanwhile", "and")
        if had_connective:
            w = w[1:]
        if len(w) == 3 and (w[1] in self.actions_set or w[1] == "chase"):
            self.agent.hear(line.rstrip(".!?"))
            self._heard_any_clause = True
            if had_connective:                       # a connective marks a discourse boundary (an earlier event now exists)
                self._boundary_seen = True
            now = self.agent.who_agent_now()
            return (f"ok -- now {now} is doing it." if now else "ok, i heard that.", False)
        return None

    # --- the full turn ---
    def answer(self, question):
        """One conversational turn: DISCOURSE (event tracking) -> GATE (recall + abstain) -> CONSTRAIN+VERIFY render.
        Returns (answer_string, abstained_bool)."""
        disc = self._discourse_turn(question)
        if disc is not None:
            return disc
        gen = self._maybe_generate(question)     # GENERATION (TUI/answer path): multi-fact associative topic reply
        if gen is not None:
            return gen
        gate_svo = self.gate(question)           # gate() now also handles ACQUISITION (assertions) -> reaches the webapp
        if gate_svo is None:
            return "I don't know about that.", True
        return self.render(gate_svo), False

    def list_facts(self):
        """The brain's stored facts (for /facts)."""
        self._refresh_facts()
        return list(self.stored_facts)


# ============================================================================================================
# Loading a developed brain from the various sources.
# ============================================================================================================

def _load_self_knowledge(codes_path, curriculum_path, seed, use_multiturn, enable_neural_render):
    """Reconstruct the self-knowledge brain: build a BrainConversationalAgent/MultiTurnAgent on the saved learned
    grounded codes + teach the curriculum facts. Returns (agent, self_aliases, n_facts)."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    with open(os.path.abspath(curriculum_path), "r", encoding="utf-8") as fh:
        cur = json.load(fh)
    # the full taught fact set as SVO (facts + attribute_facts as (noun, 'is', adj))
    facts = [tuple(f) for f in cur.get("facts", [])]
    facts += [(noun, "is", adj) for noun, adj in cur.get("attribute_facts", [])]
    # vocab: the concept set + general-knowledge + untaught fall-backs (so the moat abstains STRUCTURALLY)
    vocab = set(["is"])
    for a, v, p in facts:
        vocab.update([a, v, p])
    vocab |= {"france", "paris", "two", "plus", "four", "romeo", "juliet", "wrote", "shakespeare",
              "color", "blue", "legs", "has", "many"}
    for probe in cur.get("deliberately_untaught_project_facts", {}).get("probes", []):
        for w in probe:
            if isinstance(w, str) and w != "?":
                vocab.add(w)
    vocab = sorted(vocab)
    grounded = None
    if codes_path and os.path.exists(codes_path):
        with open(codes_path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        grounded = {w: np.asarray(v, dtype=float) for w, v in blob.get("grounded_codes", {}).items()}
        print(f"[tui] loaded {len(grounded)} developed grounded codes from "
              f"{os.path.relpath(codes_path, _REPO)}", flush=True)
    else:
        print("[tui] no developed codes file found -- the brain answers the taught facts on its own seed codes "
              "(run _self_knowledge_demo to develop + save the learned codes).", flush=True)
    concepts = {w: None for w in vocab}
    if use_multiturn:
        from research.runners.multi_turn_agent import MultiTurnAgent
        actions = {v for _a, v, _p in facts} | {"is"}
        referents = [w for w in vocab if w not in actions]
        # size the WM loop to hold every referent (2x headroom) so a large vocabulary does NOT overrun the
        # pattern budget (the SpikingLoopContextBuffer holds n/pattern_size patterns) -- same rule as
        # _longitudinal_develop_loop.build_agent.
        pattern_size = 40
        wm_n = max(600, 2 * pattern_size * max(1, len(referents)))
        # DISCOURSE EVENT REGISTER (2026-07-10): the running FACTORED (agent, patient) event so the real developed
        # brain can also answer "who was doing it before?" across a connective. Built on up to 6 of the brain's own
        # referents (the D3 arc's validated K=6 scale; a larger register is best-effort). numpy (spiking=False).
        ev_reg = None
        try:
            from research.runners._d3_event_pair_agent_derisk import PairEventRegister
            reg_refs = referents[:6] if len(referents) >= 6 else (referents + ["dog", "cat", "fish", "bird", "worm", "ball"])[:6]
            ev_reg = PairEventRegister(reg_refs, seed=seed, spiking=False)
        except Exception as _e:
            print(f"[tui] discourse event register unavailable ({_e!r}); who-was-before disabled.", flush=True)
        # defer_planner=True: the persistent discourse WM loop is built lazily on the first multi-turn referent
        # (the curriculum teach below uses BrainConversationalAgent.hear, which does NOT write WM referents, so a
        # loaded self-knowledge brain never pays the ~681s WM build at load -- only when a console turn actually
        # introduces a pronoun antecedent). Byte-identical otherwise.
        agent = MultiTurnAgent(referent_concepts=referents, concepts=concepts,
                               grounded_codes=grounded if grounded else None, seed=seed,
                               wm_n=wm_n, wm_pattern_size=pattern_size,
                               enable_neural_render=enable_neural_render, composer_kind="rf",
                               enable_biased_competition=False, defer_planner=True, event_register=ev_reg)
    else:
        agent = BrainConversationalAgent(seed=seed, concepts=concepts,
                                         grounded_codes=grounded if grounded else None,
                                         composer_kind="rf", enable_neural_render=enable_neural_render)
    inner = getattr(agent, "agent", agent)
    n = 0
    for a, v, p in facts:
        inner.hear(f"{a} {v} {p}", polarity="AFFIRM")
        n += 1
    aliases = set(cur.get("self_reference", {}).get("agent_aliases", [])) | DEFAULT_SELF_ALIASES
    return agent, aliases, n


def _build_tiny_demo(seed, use_multiturn, enable_neural_render, composer_kind="rf"):
    """A tiny CPU brain for the GPU-FREE smoke: a handful of self-facts + a couple of object facts. Mirrors the
    self-knowledge shape so the smoke exercises self-reference + the moat + multi-turn anaphora.

    `composer_kind` (default 'rf' = the numpy fast-path recall, byte-identical to before): pass 'onebrain' for the
    GENUINELY-SPIKING recall (resonate-and-fire per query + the on-substrate cleanup/store; runtime new-word LEARN
    works via the vocab_headroom recruit-an-assembly path). The onebrain build is much slower (~180s) but is the
    brain-based-only recall the mission requires; speed is secondary."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    # base-form verbs so the template-stub's 3rd-person inflection reads cleanly (use->uses, learn->learns).
    # 'cat' is the OBJECT of (dog chase cat) AND the SUBJECT of (cat eat fish) -- the validated chainable-referent
    # pattern so 'what does it eat' resolves 'it'->cat (the dog's chase-object) and answers 'fish'.
    facts = [
        ("brain", "use", "spikes"),
        ("brain", "learn", "words"),
        ("brain", "store", "memory"),
        ("dog", "chase", "cat"),
        ("cat", "eat", "fish"),
    ]
    actions = {v for _a, v, _p in facts}
    # include the discourse event register's animal referents (worm/ball/river/bird) so a multi-clause discourse
    # ("dog chase cat. then bird chase worm.") folds into BOTH the running event AND the composer without a code miss.
    vocab = sorted({w for f in facts for w in f} | {"river", "bird", "fish", "worm", "ball"})
    concepts = {w: None for w in vocab}
    if use_multiturn:
        from research.runners.multi_turn_agent import MultiTurnAgent
        referents = [w for w in vocab if w not in actions]
        # DISCOURSE EVENT REGISTER (2026-07-10): a running FACTORED (agent, patient) event so the console can answer
        # "who was doing it before?" across a connective ("dog chase cat. THEN bird chase worm. who was before?" -> dog).
        # The labelled PairEventRegister (0.928, its validated animal referents), numpy (spiking=False) for the CPU path.
        ev_reg = None
        try:
            from research.runners._d3_event_pair_agent_derisk import PairEventRegister
            ev_reg = PairEventRegister(["dog", "cat", "fish", "bird", "worm", "ball"], seed=seed, spiking=False)
        except Exception as _e:
            print(f"[tui] discourse event register unavailable ({_e!r}); who-was-before disabled.", flush=True)
        agent = MultiTurnAgent(referent_concepts=referents, concepts=concepts, seed=seed,
                               enable_neural_render=enable_neural_render, composer_kind=composer_kind,
                               enable_biased_competition=False, defer_planner=True, event_register=ev_reg)
    else:
        agent = BrainConversationalAgent(seed=seed, concepts=concepts, composer_kind=composer_kind,
                                         enable_neural_render=enable_neural_render)
    inner = getattr(agent, "agent", agent)
    for a, v, p in facts:
        inner.hear(f"{a} {v} {p}", polarity="AFFIRM")
    return agent, DEFAULT_SELF_ALIASES, len(facts)


def _resolve_composer_kind(args):
    """The tiny-demo recall substrate for the TUI. Interactive default = 'onebrain' (the GENUINELY-SPIKING recall,
    resonate-and-fire per query, runtime new-word LEARN — the same production default the webapp uses), so the owner
    gets the full spiking brain in the TERMINAL too, not only the web UI. Resolution order: explicit --composer wins;
    then the automated --smoke path forces 'rf' (the GPU-free smoke must stay fast + byte-identical); then the
    BRAIN_COMPOSER_KIND env (shared with the webapp); else 'onebrain'. Pass --composer rf for the fast numpy path."""
    if getattr(args, "composer", None):
        return args.composer
    if getattr(args, "smoke", False):
        return "rf"
    return os.environ.get("BRAIN_COMPOSER_KIND", "onebrain")


def load_brain(args):
    """Resolve --load / --self-knowledge / --tiny-demo into (agent, self_aliases, n_facts, source_desc)."""
    use_mt = not args.no_multiturn
    nr = args.neural_render
    # explicit developed-brain bundle directory
    if args.load and is_developed_brain_bundle(args.load):
        agent, manifest = load_developed_brain(args.load, use_multiturn=use_mt, enable_neural_render=nr)
        aliases = set(manifest.get("self_aliases") or []) | DEFAULT_SELF_ALIASES
        n = manifest.get("n_facts", len(getattr(agent, "agent", agent).composer.kb))
        return agent, aliases, n, f"developed-brain bundle: {args.load}"
    # self-knowledge brain (explicit flag, or a --load pointing at a codes .json)
    if args.self_knowledge or (args.load and str(args.load).endswith(".json")):
        codes = args.load if (args.load and str(args.load).endswith(".json")) else _SK_CODES
        curriculum = args.curriculum or _SK_CURRICULUM
        agent, aliases, n = _load_self_knowledge(codes, curriculum, args.seed, use_mt, nr)
        return agent, aliases, n, f"self-knowledge brain (codes={os.path.relpath(codes, _REPO) if os.path.exists(codes) else 'seed-codes'})"
    # tiny CPU demo — interactive default is the genuinely-SPIKING onebrain recall (the --smoke path stays 'rf' fast)
    if args.tiny_demo or not args.load:
        ck = _resolve_composer_kind(args)
        agent, aliases, n = _build_tiny_demo(args.seed, use_mt, nr, composer_kind=ck)
        return agent, aliases, n, f"tiny CPU demo brain (composer={ck})"
    raise FileNotFoundError(f"--load {args.load!r} is neither a developed-brain bundle nor a codes .json")


# ============================================================================================================
# The renderer factory.
# ============================================================================================================

def build_renderer(args):
    """Build the fluent renderer: the off-bridge Qwen (default) or the template-stub (--stub-renderer / smoke)."""
    if args.stub_renderer:
        return StubRenderer()
    if args.no_renderer:
        return None
    return QwenRenderer(T=args.T, max_new_tokens=args.max_new_tokens, seed=args.seed)


# ============================================================================================================
# The interactive REPL.
# ============================================================================================================

_BANNER = """\
============================================================================
  BRAIN CHAT  --  talk to a developed brain about what it knows
============================================================================
  Source : {source}
  Knows  : {n_facts} facts   |   Renderer: {renderer}
  Self   : 'you'/'your'/'I'/'me'/'it' map to the brain (ask 'what are you?')
  Moat   : the brain ABSTAINS ('I don't know about that.') on anything it
           was not taught -- it never makes things up.
  Commands: /facts  /raw  /help  /quit
============================================================================
"""

_HELP = """\
  /facts   list the facts the brain knows
  /raw     toggle the brain's OWN renderer (no LLM) -- raw recalled triple
  /help    show this help
  /quit    exit  (also /exit, /q, Ctrl-D)
"""


def _print_facts(chat):
    facts = chat.list_facts()
    if not facts:
        print("  (the brain knows no facts.)", flush=True)
        return
    print(f"  the brain knows {len(facts)} facts:", flush=True)
    for a, v, p in facts:
        print(f"    - {a} {v} {p}", flush=True)


def run_repl(chat, source, n_facts, rich=None):
    """The interactive chat loop. When `rich` is a RichAnswerComposer, each turn produces a SUBSTANTIVE
    multi-sentence GROUNDED reply (direct recall + multi-hop chain + elaboration, each sentence verify-checked);
    'tell me more' / 'why?' elaborates the held topic further. Otherwise the default single-fact answer."""
    rname = chat.renderer.name if chat.renderer is not None else "(none -- raw brain triples)"
    mode = "RICH (multi-sentence grounded; 'tell me more' elaborates)" if rich is not None else "single-fact"
    print(_BANNER.format(source=source, n_facts=n_facts, renderer=f"{rname}   |   answers: {mode}"), flush=True)
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[tui] bye.", flush=True)
            break
        if not line:
            continue
        low = line.lower()
        if low in ("/quit", "/exit", "/q", "quit", "exit"):
            print("[tui] bye.", flush=True)
            break
        if low in ("/help", "help", "?"):
            print(_HELP, flush=True)
            continue
        if low == "/facts":
            _print_facts(chat)
            continue
        if low == "/raw":
            chat.raw_mode = not chat.raw_mode
            print(f"  [raw mode {'ON -- the brain speaks its own raw triples (no LLM)' if chat.raw_mode else 'OFF -- fluent rendering'}]",
                  flush=True)
            continue
        if chat.verbose_thinking and chat.renderer is not None and not chat.raw_mode:
            print("  brain> thinking...", flush=True)
        if rich is not None:
            r = rich.answer(line)
            tag = "  (abstained -- the moat)" if r["abstained"] else f"  [{r['n_sentences']} grounded sentences]"
            print(f"brain> {r['answer']}{tag}\n", flush=True)
        else:
            ans, abstained = chat.answer(line)
            tag = "  (abstained -- the moat)" if abstained else ""
            print(f"brain> {ans}{tag}\n", flush=True)


# ============================================================================================================
# The GPU-FREE scripted SMOKE.
# ============================================================================================================

def run_smoke(chat, source, n_facts, out_path):
    """Scripted multi-turn turns (incl. anaphora + abstention + self-reference) on the tiny CPU brain with the
    template-stub renderer. Verifies the TUI loads + converses + the moat abstains + multi-turn anaphora works."""
    # the scripted multi-turn conversation. Each entry: (utterance, expectation-kind).
    # 'anaphora' uses the prior turn's referent; 'abstain' must hit the moat; 'self' is a self-reference question.
    script = [
        ("what are you", "answer"),              # self-reference: 'you' -> brain ('brain uses spikes')
        ("how do you learn", "answer"),          # self-reference synonym: learn -> learns ('brain learns words')
        ("what does the brain store", "answer"),  # direct self-fact ('brain store memory')
        ("what does the dog chase", "answer"),   # object fact -> the answer 'cat' is a chainable referent -> WM
        ("what does it eat", "anaphora"),        # anaphora: 'it' -> cat (the dog's chase-object) -> 'fish'
        ("what does the dragon do", "abstain"),  # untaught subject -> the moat abstains
        ("who wrote romeo and juliet", "abstain"),  # general knowledge never taught -> abstain (the firewall)
        ("what is the capital of france", "abstain"),  # Qwen knows this; the brain must NOT (firewall)
    ]
    transcript = []
    for utterance, kind in script:
        gate_svo = chat.gate(utterance)          # peek the gate so the transcript records what the brain recalled
        ans, abstained = (chat.answer(utterance) if gate_svo is None
                          else (chat.render(gate_svo), False))
        transcript.append({"you": utterance, "kind": kind, "gate_svo": gate_svo,
                           "brain": ans, "abstained": abstained})

    # DISCOURSE EVENT TRACKING (2026-07-10): hear a multi-clause discourse across a connective, then answer
    # "who was doing it before?" -- the deployed D3 event-register capability, now reachable in the console.
    disc_ok = None
    if getattr(chat, "has_event_register", False):
        fresh_before, fresh_abst = chat.answer("who was doing it before?")   # nothing said yet -> the moat abstains
        chat.answer("dog chase cat")
        chat.answer("then bird chase worm")                                  # the connective pushes dog's event
        before, _ = chat.answer("who was doing it before?")                  # -> dog
        now, _ = chat.answer("who is doing it now?")                         # -> bird
        disc_ok = bool(fresh_abst and ("dog" in before.lower()) and ("bird" in now.lower()))
        transcript.append({"you": "[discourse] dog chase cat / then bird chase worm / who was before? / now?",
                           "kind": "discourse", "gate_svo": None, "abstained": False,
                           "brain": f"before={before!r} now={now!r} (fresh-moat abstained={fresh_abst})",
                           "discourse_ok": disc_ok})

    # checks
    self_q = transcript[0]
    self_answered = (not self_q["abstained"]) and self_q["gate_svo"] is not None and self_q["gate_svo"][0] == "brain"
    learn_q = next((t for t in transcript if t["you"] == "how do you learn"), None)
    learn_answered = bool(learn_q and not learn_q["abstained"]
                          and learn_q["gate_svo"] is not None and learn_q["gate_svo"][1] == "learn")
    # anaphora (RIGOROUS): the 'what does it eat' turn must have RESOLVED 'it' to the EXACT prior referent ('cat',
    # the dog's chase-object) AND answered the cat-eat-fish fact. A resolution to anything but 'cat', or an
    # abstention, FAILS -- so a spurious WM read cannot pass.
    anaphora_turn = next(t for t in transcript if t["you"] == "what does it eat")
    resolved_to = chat._resolve_anaphora("what does it eat")
    anaphora_resolved = (("cat" in resolved_to.split()) and ("it" not in resolved_to.split())
                         and (not anaphora_turn["abstained"])
                         and anaphora_turn["gate_svo"] == ["cat", "eat", "fish"])
    # abstention turns must abstain (the moat)
    abstain_turns = [t for t in transcript if t["kind"] == "abstain"]
    moat_held = all(t["abstained"] for t in abstain_turns)
    # at least the self + object facts answered (the brain converses)
    answered = [t for t in transcript if t["kind"] == "answer" and not t["abstained"]]
    converses = len(answered) >= 3

    discourse_ok = (disc_ok is None) or bool(disc_ok)   # if a register is present it must track; else neutral
    go = bool(self_answered and learn_answered and anaphora_resolved and moat_held and converses and discourse_ok)

    verdict = (
        f"GO -- the TUI loads a saved/tiny brain + holds a multi-turn conversation: self-reference resolves "
        f"('what are you' -> {self_q['gate_svo']}), learn-synonym resolves to the 'brain learn words' fact, "
        f"multi-turn anaphora binds 'it' -> {resolved_to!r} (the dog's chase-object 'cat') and answers "
        f"['cat','eat','fish'], the no-confab moat abstains on all {len(abstain_turns)} untaught/general cues "
        f"(incl. 'capital of France' the LLM knows but the brain must not), and {len(answered)} fact turns "
        f"answered. Renderer={chat.renderer.name if chat.renderer else 'raw'}. READY for the owner to --load the "
        f"real developed brain (with the off-bridge Qwen renderer)."
        if go else
        f"PARTIAL/SNAG -- self_answered={self_answered} learn_answered={learn_answered} "
        f"anaphora_resolved={anaphora_resolved} (resolved={resolved_to!r}) moat_held={moat_held} "
        f"converses={converses} ({len(answered)} fact turns). See the transcript for the localize."
    )

    res = {
        "go": go,
        "verdict": verdict,
        "backend": os.environ.get("SIM_BACKEND"),
        "source": source,
        "renderer": (chat.renderer.name if chat.renderer is not None else "raw brain triples"),
        "n_facts": n_facts,
        "self_reference_answered": self_answered,
        "learn_synonym_answered": learn_answered,
        "multiturn_anaphora_resolved": anaphora_resolved,
        "anaphora_resolved_to": resolved_to,
        "moat_held": moat_held,
        "n_abstain_turns": len(abstain_turns),
        "n_answer_turns": len(answered),
        "converses": converses,
        "transcript": transcript,
        "tui_features": [
            "load a developed brain (codes + facts + vocab) from a developed_brain_io bundle, OR the self-knowledge "
            "codes+curriculum, OR a tiny CPU fallback",
            "multi-turn chat: GATE (recall + abstain) -> CONSTRAIN+VERIFY fluent render (off-bridge Qwen default; "
            "template-stub for the GPU-free smoke) -> answer or 'I don't know about that.'",
            "multi-turn anaphora (it/that/they -> the prior referent via the MultiTurnAgent discourse WM)",
            "self-reference (you/your/I/me/it -> the brain) so 'what are you' / 'how do you learn' resolve",
            "commands: /raw (brain's own renderer, no LLM), /facts (list knowledge), /help, /quit",
            "the no-confab moat: the brain abstains on anything it was not taught (verified at the recall layer)",
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    # print the transcript
    print("\n" + "=" * 90, flush=True)
    print("[tui SMOKE] scripted multi-turn transcript:", flush=True)
    print("=" * 90, flush=True)
    for t in transcript:
        gate = "" if t["gate_svo"] is None else f"   (recalled: {t['gate_svo']})"
        atag = "  [ABSTAIN]" if t["abstained"] else ""
        print(f"  you>   {t['you']}", flush=True)
        print(f"  brain> {t['brain']}{atag}{gate}", flush=True)
    print("=" * 90, flush=True)
    print(f"[tui SMOKE] VERDICT: {verdict}", flush=True)
    print(f"[tui SMOKE] saved {os.path.relpath(out_path, _REPO)}", flush=True)
    return res


# ============================================================================================================
# main.
# ============================================================================================================

def main():
    ap = argparse.ArgumentParser(description="Talk to a developed/trained brain (multi-turn).")
    ap.add_argument("--load", default=None,
                    help="a developed-brain bundle DIR (brain.json+...) OR a grounded-codes .json (self-knowledge).")
    ap.add_argument("--self-knowledge", action="store_true",
                    help="load the self-knowledge brain (default codes + curriculum).")
    ap.add_argument("--curriculum", default=None,
                    help="curriculum .json for the self-knowledge brain (default: _curriculum_self_knowledge.json).")
    ap.add_argument("--tiny-demo", action="store_true",
                    help="build a tiny CPU brain from a handful of facts (GPU-free fallback / smoke).")
    ap.add_argument("--composer", choices=["rf", "onebrain"], default=None,
                    help="tiny-demo recall substrate: 'onebrain' (GENUINELY SPIKING, resonate-and-fire, the interactive "
                         "default) or 'rf' (numpy fast path). Default: onebrain interactively, rf under --smoke; the "
                         "BRAIN_COMPOSER_KIND env is honored when this is unset. The onebrain build is ~180s (speed secondary).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-multiturn", action="store_true",
                    help="use the bare BrainConversationalAgent (no discourse WM / anaphora).")
    ap.add_argument("--neural-render", action="store_true",
                    help="enable the brain's own spiking serial-order renderer (slow).")
    # renderer
    ap.add_argument("--stub-renderer", action="store_true",
                    help="use the GPU-FREE template-stub renderer (the CPU smoke); default is the off-bridge Qwen.")
    ap.add_argument("--no-renderer", action="store_true",
                    help="no fluent renderer (the brain speaks its own raw triples).")
    ap.add_argument("--T", type=int, default=16, help="off-bridge Qwen rate-code pool budget (16=GO).")
    ap.add_argument("--max-new-tokens", type=int, default=24, help="Qwen surface-form length cap.")
    # rich answers (opt-in)
    ap.add_argument("--rich", action="store_true",
                    help="SUBSTANTIVE multi-sentence GROUNDED replies (direct recall + multi-hop chain + "
                         "elaboration, each sentence verify-checked); 'tell me more'/'why?' elaborates further. "
                         "Default OFF = the single-fact oracle answer.")
    ap.add_argument("--rich-max-sentences", type=int, default=4, help="max sentences per rich reply.")
    ap.add_argument("--no-neural-planner", action="store_true",
                    help="(--rich only) DISABLE the spiking dlPFC discourse-planner; use the HOST gather/order/"
                         "stop heuristics instead. Default = neural-ON (the brain-based-purity version: the dlPFC "
                         "spreading-activation latency rank drives WHICH grounded facts to bring up, in WHAT order, "
                         "and WHEN to stop). The escape exists for the numpy-CPU / reproducibility / test-oracle "
                         "path (the host planner avoids building a per-topic SimulationBridge).")
    # smoke
    ap.add_argument("--smoke", action="store_true",
                    help="run the scripted GPU-FREE smoke (no interactive input) + write the JSON verdict.")
    ap.add_argument("--out", default="research/findings/raw/_brain_chat_tui_smoke.json",
                    help="smoke JSON output path.")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    # load the brain
    agent, aliases, n_facts, source = load_brain(a)
    # build the renderer (the smoke forces the stub if neither flag set)
    if a.smoke and not a.stub_renderer and not a.no_renderer:
        a.stub_renderer = True   # the GPU-free smoke uses the template-stub by default
    renderer = build_renderer(a)
    chat = ChatBrain(agent, self_aliases=aliases, renderer=renderer)

    if a.smoke:
        res = run_smoke(chat, source, n_facts, os.path.join(_REPO, a.out) if not os.path.isabs(a.out) else a.out)
        return 0 if res["go"] else 1

    rich = None
    if a.rich:
        from research.runners.rich_answer_composer import RichAnswerComposer
        # DEFAULT = the NEURAL discourse-planner (brain-based-purity): the spiking dlPFC content-selection
        # (SpikingSpreadingController) drives WHICH grounded facts to bring up, in WHAT neural-relevance order,
        # and WHEN to stop -- the GO 3G replacement for the host gather/order/stop heuristics (quality-parity,
        # lesion-load-bearing, on-topic, moat 0-FA). `--no-neural-planner` is the host escape.
        # numpy-CPU nuance (mirrors the 1A sentinel): the planner builds + steps a per-topic SimulationBridge,
        # which is heavy on the CPU smoke path -- so on the numpy backend we keep the HOST default for
        # portability/speed (neural-on stays the GPU default). The explicit `--no-neural-planner` always forces
        # host regardless of backend.
        try:
            from sim.backend import is_gpu_backend
            _on_gpu = bool(is_gpu_backend())
        except Exception:
            _on_gpu = (os.environ.get("SIM_BACKEND", "").lower() == "cupy")
        neural_planner = (not a.no_neural_planner) and _on_gpu
        if a.no_neural_planner:
            print("[rich] neural discourse-planner DISABLED (--no-neural-planner): host gather/order/stop.",
                  flush=True)
        elif not _on_gpu:
            print("[rich] neural discourse-planner: HOST default on the numpy-CPU backend "
                  "(the spiking dlPFC planner needs a bridge; use SIM_BACKEND=cupy for neural-ON).", flush=True)
        else:
            print("[rich] neural discourse-planner ON (default): spiking dlPFC content-selection drives "
                  "gather/order/stop.", flush=True)
        rich = RichAnswerComposer(chat, max_sentences=a.rich_max_sentences,
                                  neural_planner=neural_planner, planner_seed=a.seed)
    run_repl(chat, source, n_facts, rich=rich)
    return 0


if __name__ == "__main__":
    sys.exit(main())
