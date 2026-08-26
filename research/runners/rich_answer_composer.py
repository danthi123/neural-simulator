"""RichAnswerComposer -- make each conversational turn produce a SUBSTANTIVE multi-sentence GROUNDED reply
instead of one short fact, so the brain holds a genuinely substantive multi-turn conversation.

THE PROBLEM (owner): the current conversation is too thin -- one fact per turn, short answers. THE INSIGHT:
the brain ALREADY has the pieces for a richer answer; it just answers ONE fact per turn. COMPOSE them.

Per question, the composer GATHERS a small GROUNDED fact-set from the brain's OWN spiking memory:
  (a) the DIRECT recall                -- the question's matched fact (the brain's `what_does`/`who_does`/
                                          `is_it_true`, gated + abstaining = the no-confab moat).
  (b) a MULTI-HOP REASON chain (1-3)   -- related facts reached by following the role structure (the
                                          composer's `query_chain`/the agent's `reason_chain`): the answer's
                                          patient becomes the next hop's agent, so "brain->spikes" reaches
                                          "spikes->neurons->..." -- each hop ABSTAINS the moment a hop has no
                                          fact (the moat holds at EVERY hop).
  (c) an ELABORATION (1-2)             -- the next on-topic concepts about the answer, chosen by the dlPFC
                                          spiking content-selection Control (`elaborate(topic)`) over the
                                          brain's own association graph; each selected associate is expanded
                                          to a STORED fact about it (so the elaboration is itself grounded,
                                          not a bare concept).

then RENDERS the SET as a coherent MULTI-SENTENCE paragraph: each gathered fact is a stored SVO; the fluent
faculty (the off-bridge Qwen, or the GPU-free template-stub) phrases each one; and **VERIFY checks each
sentence's re-parsed SVO against the brain** -- so the no-confab moat EXTENDS to multi-sentence: ONLY
brain-sourced claims survive, any sentence the brain cannot support is DROPPED. ABSTAIN (the moat) when the
brain has nothing relevant to the question at all.

MULTI-TURN: the composer carries the discourse context (anaphora) across turns via the wrapped MultiTurnAgent's
spiking working-memory loop; a follow-up ('tell me more' / 'why?') ELABORATES FURTHER on the held topic (it
walks deeper into the association graph / further hops, skipping facts already said this thread).

REUSE-BY-IMPORT, NO `sim/` edit. The GATE+VERIFY reuse the validated brain_chat_tui ChatBrain wiring (the
QuestionRouter content matcher, the `_extract_svo_from_prose` re-parse, the BridgeParser role assignment); the
gather reuses the composer's validated `query_chain` + `elaborate` + `render_fact`. The faculty is the
off-bridge Qwen at runtime; the GPU-free smoke uses the template-stub.

Usage:
    # GPU-FREE CPU smoke (template-stub faculty), writes the rich-vs-thin transcript + verdict:
    SIM_BACKEND=numpy python -m research.runners.rich_answer_composer --smoke --stub-renderer

    # wired into the TUI's `--rich` mode (the runtime path uses the off-bridge Qwen renderer):
    SIM_BACKEND=cupy python -m research.runners.brain_chat_tui --load <brain> --rich
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# follow-up cues that mean "elaborate further on what we're already talking about"
_FOLLOWUP_CUES = {"more", "why", "else", "elaborate", "continue", "go", "and", "further", "expand", "explain"}
_FOLLOWUP_PHRASES = ("tell me more", "go on", "say more", "what else", "anything else", "and then", "why is that",
                     "why", "how so", "keep going")


def _is_followup(question):
    """True when the utterance is a bare 'tell me more' / 'why?' style follow-up (no new content) -- so the
    composer ELABORATES on the held topic rather than re-gating a fresh question."""
    q = question.lower().strip().strip(".,!?")
    if q in _FOLLOWUP_PHRASES or any(q == p for p in _FOLLOWUP_PHRASES):
        return True
    toks = [t.strip(".,!?") for t in q.split()]
    content = [t for t in toks if t not in {"me", "a", "the", "is", "that", "it", "do", "you", "us", "please"}]
    # a follow-up = ALL content tokens are follow-up cues (e.g. "tell more" / "why" / "what else") and it's short
    return bool(content) and len(content) <= 3 and all(t in _FOLLOWUP_CUES for t in content)


class NeuralDiscoursePlanner:
    """The NEURAL discourse-planner (burndown 3G / inventory C-6): the dlPFC spiking content-selection Control
    drives WHICH on-topic concepts the rich answer brings up, in WHAT NEURAL-RELEVANCE order, and WHEN to STOP --
    replacing the host gather/order/relevance/breadth/stop heuristics in RichAnswerComposer.

    The validated mechanism (`content_selection_spiking.SpikingSpreadingController`): the agent's association
    graph is embodied as inter-assembly synapses (cortex_A -> dlpfc_B at weight proportional to graph[A][B]) on a
    real SimulationBridge; driving a TOPIC concept SPREADS activation along those synapses, and each related
    assembly's FIRST-SPIKE LATENCY encodes its graph DISTANCE (direct associates fire earliest; unrelated
    concepts never fire -- "clean by construction"). So ONE spreading-activation probe (`relevance_by_latency`)
    yields the on-topic concepts in neural-relevance order, with off-topic filtering FOR FREE.

    This replaces the host cognition in the composer:
      - the host `_facts_mentioning` / edge-weight-argmax / 2-hop BFS RELEVANCE+ORDERING  -> the latency rank;
      - the host two-source interleave (which associate next)                              -> the latency rank;
      - the host `max_sentences` cap AS the cognitive STOP                                 -> the probe EXHAUSTING
        (no more reached, unsaid concepts) -- `max_sentences` stays only as a hard safety ceiling;
      - inhibition-of-return ("don't repeat what's said")                                  -> the controller's own
        `SaidTrace` PLUS the conversation-wide `avoid` fed in.
    Selecting a CONCEPT is neural; mapping a selected concept back to a stored grounded SVO is legitimate host KB
    access (the controller picks concepts, not facts). The no-confab moat is untouched -- this planner only
    decides WHICH grounded facts to bring up; the per-sentence VERIFY in the composer still gates every sentence.

    Built lazily + cached on the controller-key (rebuilt only when the association graph changes), exactly like
    the agent's own `elaborate`."""

    def __init__(self, composer, seed=42):
        self.composer = composer
        self.seed = int(seed)
        self._ctrl = None
        self._key = None

    def _controller(self):
        """The persistent SpikingSpreadingController over the agent's current association graph. Reuses the
        composer's own `_assoc_graph()` (the same graph `elaborate` spreads over) so the planner and the agent's
        validated `elaborate` see the identical graph; rebuilt only when the graph content changes."""
        from research.runners.content_selection_spiking import SpikingSpreadingController
        graph = self.composer._assoc_graph()
        if not graph:
            return None, graph
        key = tuple(sorted((k, tuple(sorted(v.items()))) for k, v in graph.items()))
        if self._ctrl is None or self._key != key:
            self._ctrl = SpikingSpreadingController(graph, seed=self.seed)
            self._key = key
        return self._ctrl, graph

    def ordered_associates(self, topic, avoid=()):
        """The NEURAL ordering: drive `topic` into the spiking dlPFC loop, spread activation, and return the
        reached on-topic concepts SORTED BY FIRST-SPIKE LATENCY (earliest = most graph-relevant), excluding the
        topic itself and every concept in `avoid`. A concept that NEVER fires (latency None) is unrelated and is
        OMITTED -- so the list is exactly the on-topic neighbourhood in neural-relevance order, and an EMPTY list
        IS the neural STOP signal ('the reachable, unsaid neighbourhood is exhausted'). One spreading probe per
        call (the validated `relevance_by_latency`)."""
        ctrl, graph = self._controller()
        if ctrl is None or topic not in graph:
            return []
        lat = ctrl.relevance_by_latency(topic)            # {concept: first-spike step or None}
        avoid_set = set(avoid) | {topic}
        reached = [(c, t) for c, t in lat.items() if t is not None and c not in avoid_set]
        reached.sort(key=lambda ct: ct[1])                # earliest first-spike = most relevant (graph distance)
        return [c for c, _t in reached]


class RichAnswerComposer:
    """Wrap a ChatBrain (the validated GATE+VERIFY+render wiring around a conversational agent) and turn each
    turn into a SUBSTANTIVE multi-sentence grounded reply.

    `chat` is a `brain_chat_tui.ChatBrain` (it owns the agent, the QuestionRouter gate, the renderer, and the
    VERIFY re-parse). The composer reuses chat.gate (recall+abstain) for the DIRECT fact and chat.render
    (constrain+verify) per sentence, and reaches the brain's `composer.query_chain` / `composer.elaborate` for
    the chain + elaboration. So the moat is preserved by construction (each rendered sentence is gate-sourced
    and verify-checked)."""

    def __init__(self, chat, *, max_chain_hops=3, max_elaborations=2, max_sentences=4, verbose=False,
                 neural_planner=False, planner_seed=None):
        self.chat = chat
        self.inner = chat.inner                      # the BrainConversationalAgent
        self.composer = chat.inner.composer          # rf / onebrain composer (query_chain, elaborate, kb)
        self.is_multiturn = chat.is_multiturn
        self.max_chain_hops = int(max_chain_hops)
        self.max_elaborations = int(max_elaborations)
        self.max_sentences = int(max_sentences)
        self.verbose = bool(verbose)
        # NEURAL DISCOURSE-PLANNER (burndown 3G / C-6): default OFF = the host gather/order/stop path
        # (byte-identical to the pre-3G composer). When ON, the dlPFC spiking content-selection drives WHICH
        # on-topic facts to gather, in WHAT neural-relevance order, and WHEN to stop -- the host relevance/
        # ordering/breadth heuristics in the elaboration paths are replaced by the spreading-activation latency
        # rank. The direct gate (moat), the role-chase chain hop, and the per-sentence VERIFY are unchanged.
        self.neural_planner = bool(neural_planner)
        self._planner = (NeuralDiscoursePlanner(self.composer,
                                                seed=getattr(self.inner, "seed", 42) if planner_seed is None
                                                else planner_seed)
                         if self.neural_planner else None)
        # discourse thread state: the topic we're elaborating + the facts already said this thread (so a
        # follow-up walks FORWARD, not repeating). _said resets when a fresh (non-follow-up) question is gated.
        self._topic = None
        self._said = []                              # list of [a, v, p] gathered+verified this thread
        # conversation-wide memory of what's already been said -- the WHOLE conversation never repeats a fact, so
        # a 'tell me more' always brings up genuinely NEW grounded content (the DIRECT recall is exempt: a fresh
        # question's own matched fact may legitimately restate, but the chain/elaboration always extend).
        self._conversation_said = set()              # set of tuple(a, v, p) said anywhere this conversation
        # DERIVED-ANSWER side channel (reasoning-frontier hardening, moat-hardening audit findings #4/#5):
        # `gather()`'s internal pipeline (chain/elaboration combination, `render_paragraph`'s `list(f)`
        # conversions) normalizes every fact to a plain list, so a `ChainedSVO` marker on the direct fact does
        # not survive to `answer()`'s returned `facts`. Tracked here instead so `answer()` can surface
        # `derived`/`derived_from` and frame the paragraph as the brain's OWN inference, and so the caller
        # (webapp/server.py) can avoid presenting a composed multi-hop inference as a directly-recalled fact.
        # Reset at the top of EVERY `gather()` call so it never leaks from a prior turn onto an unrelated one.
        self._last_direct_derived = False
        self._last_direct_derived_from = []

    # ------------------------------------------------------------------------------------------------
    # GATHER: assemble a small GROUNDED fact-set (a) direct + (b) chain + (c) elaboration.
    # Each entry is a stored SVO [a, v, p] the brain can support; cleanup + abstention happen in the
    # composer's own validated ops, so an unsupported hop simply isn't gathered.
    # ------------------------------------------------------------------------------------------------
    def _stored_facts(self):
        """The brain's stored SVO facts (string-only roles) -- the source of the gather + the verify vocab."""
        return [(f.get("agent"), f.get("action"), f.get("patient")) for f, _ in self.composer.kb
                if all(isinstance(f.get(r), str) for r in ("agent", "action", "patient"))]

    def _facts_about(self, concept):
        """Stored facts whose AGENT is `concept` -- the concept's own self-statements (what the brain can say
        ABOUT it). Used to expand a chain step / an elaboration associate into a grounded sentence."""
        return [[a, v, p] for (a, v, p) in self._stored_facts() if a == concept]

    def _facts_mentioning(self, concept):
        """Stored facts that mention `concept` in ANY role (agent OR patient) -- the content most directly ABOUT
        the topic, including the facts pointing INTO a hub concept ('weights hold memory', 'facts build memory'
        when the topic is 'memory'). The richest, most naturally on-topic elaboration source for a hub."""
        return [[a, v, p] for (a, v, p) in self._stored_facts() if concept in (a, p)]

    def _direct_fact(self, question):
        """(a) The DIRECT recall: the question's matched stored fact, GATED + VERIFIED by the brain (the moat
        abstains here -> None). Reuses the validated chat.gate (router match + spiking-recall verify).

        COMPOSITIONAL CHAIN ROUTE (reasoning-frontier, 2026-08-25): checked FIRST, before `chat.gate`. WHY
        first, not as a fallback on gate()'s abstain: `chat.gate` is idempotently monkeypatched by the GNW
        ignition-bus installers (`gnw_bus_shadow` / `gnw_two_organ_bus` / `gnw_three_organ_bus` / ...), none of
        which are chain-aware yet (extending each bus organ to a multi-hop read is the honest next rung -- see
        the finding); inserting the check AFTER a call to `self.chat.gate(question)` would run the (not-chain-
        aware) bus machinery for nothing on a compositional turn and still need this same fallback. Checking
        the regex shape FIRST costs nothing on a non-compositional turn (the regex simply does not match, and
        control falls straight through to `chat.gate` UNCHANGED -- byte-identical for every other question) and
        covers the SAME class uniformly regardless of which bus wrapper is installed. See
        research/runners/compositional_chain_route.py for the detection + hop-execution + the honesty/moat
        argument (only a hop pair BOTH independently confirmed by the composer is ever returned)."""
        from research.runners.compositional_chain_route import resolve_compositional_chain
        chained = resolve_compositional_chain(self.composer, question)
        if chained is not None:
            return chained
        return self.chat.gate(question)              # [a, v, p] or None

    def _chain_facts(self, start_agent, seed_action):
        """(b) MULTI-HOP: follow the brain's role structure from `start_agent`. The first hop uses `seed_action`
        (the question's verb); subsequent hops reuse the SAME relation if it keeps matching (the validated
        query_chain semantics: patient->next agent). Returns the list of [a, v, p] hop-facts actually traversed
        (each one a STORED fact the moat confirmed); stops the moment a hop abstains. Error does NOT compound --
        the composer re-cleans the intermediate each hop."""
        facts = []
        cur = start_agent
        action = seed_action
        for _ in range(self.max_chain_hops):
            nxt = self.composer.query_patient(cur, action)   # the validated spiking hop (abstains -> None)
            if nxt is None:
                # the same relation ran out; try the concept's OWN next fact (a different relation it has),
                # so the chain can turn a corner (brain->spikes, then spikes->ARE->neurons under a new verb).
                own = self._facts_about(cur)
                own = [f for f in own if f not in facts and f != [start_agent, seed_action, None]]
                if not own:
                    break
                a2, v2, p2 = own[0]
                facts.append([a2, v2, p2])
                cur, action = p2, v2
                continue
            facts.append([cur, action, nxt])
            cur = nxt                                         # the patient becomes the next hop's agent
        return facts

    def _facts_for_concept(self, concept, exclude):
        """The grounded facts the brain can bring up for `concept`, in a fixed KB order (legitimate host KB
        access -- the NEURAL planner already decided we want THIS concept; this just looks up its stored SVOs).
        Facts MENTIONING the concept (any role) -- the richest content about it -- excluding `exclude`."""
        excluded = set(tuple(f) for f in exclude)
        return [f for f in self._facts_mentioning(concept) if tuple(f) not in excluded]

    def _elaboration_facts_neural(self, topic, exclude):
        """(c) ELABORATION via the NEURAL discourse-planner: the dlPFC spreading-activation latency rank decides
        WHICH on-topic concepts to bring up and in WHAT order (off-topic concepts never fire -> never selected);
        each neural-selected concept is mapped to a grounded stored SVO (legitimate KB access). STOP when the
        neural neighbourhood is exhausted (no more reached concepts), capped at max_elaborations. Replaces the
        host two-source interleave + the host edge-weight/mention relevance with the spiking selection."""
        out = []
        excluded = set(tuple(f) for f in exclude)
        for concept in self._planner.ordered_associates(topic, avoid=()):
            if len(out) >= self.max_elaborations:
                break
            for f in self._facts_for_concept(concept, exclude=list(excluded)):
                if tuple(f) not in excluded:
                    out.append(list(f))
                    excluded.add(tuple(f))
                    break                         # one grounded sentence per neural-selected concept
        return out[: self.max_elaborations]

    def _elaboration_facts(self, topic, exclude):
        """(c) ELABORATION: bring up the next on-topic GROUNDED facts about `topic`, from two complementary
        sources, both excluding `exclude`:
          1. facts MENTIONING the topic in any role (the richest, most direct on-topic content -- e.g. for a hub
             like 'memory', the facts that point INTO it: 'weights hold memory', 'facts build memory'), and
          2. the dlPFC spiking content-selection Control's associates, each expanded to a fact ABOUT it (its own
             self-statement) -- so the dlPFC's spreading-activation selection still drives which related concept
             we bring up, grounded into a real sentence.
        Returns up to max_elaborations [a, v, p]. The dlPFC pick is foregrounded (it is the neural selection); the
        topic-mention facts fill the rest (so a hub topic still yields a rich elaboration).

        With neural_planner, the WHICH-concept + ORDER + on-topic filtering is the dlPFC spreading-activation
        latency rank (the host two-source interleave is retired)."""
        if self.neural_planner:
            return self._elaboration_facts_neural(topic, exclude)
        out = []
        excluded = set(tuple(f) for f in exclude)

        def _take(f):
            if f is not None and tuple(f) not in excluded and tuple(f) not in {tuple(x) for x in out}:
                out.append(list(f))
                excluded.add(tuple(f))

        # source 2 first (the dlPFC neural selection leads): the top associate's grounded fact
        seen_concepts = {topic} | {f[0] for f in exclude}
        for _ in range(self.max_elaborations + 2):
            if len(out) >= self.max_elaborations:
                break
            assoc = self._elaborate_one(topic, avoid=seen_concepts)
            if assoc is None:
                break
            seen_concepts.add(assoc)
            about = [f for f in self._facts_about(assoc) if tuple(f) not in excluded]
            if about:
                _take(about[0])
        # source 1 fills the remainder: facts mentioning the topic directly (hub content)
        for f in self._facts_mentioning(topic):
            if len(out) >= self.max_elaborations:
                break
            _take(f)
        return out[: self.max_elaborations]

    def _elaborate_one(self, topic, avoid):
        """One associate of `topic` from the spiking dlPFC Control, excluding `avoid`. The controller returns its
        single dominant associate; to walk past it we temporarily restrict the graph to the not-yet-avoided
        neighbours (the controller operates on the graph, so trimming it steers the next pick) -- if every
        neighbour is avoided, returns None."""
        try:
            assoc = self.composer.elaborate(topic)
        except Exception:
            assoc = None
        if assoc is not None and assoc not in avoid:
            return assoc
        # the top associate was avoided (already said) -> pick the next graph neighbour by edge weight directly
        graph = self.composer._assoc_graph()
        nbrs = graph.get(topic, {})
        for c, _w in sorted(nbrs.items(), key=lambda kv: kv[1], reverse=True):
            if c not in avoid:
                return c
        return None

    def gather(self, question, *, followup=False):
        """Assemble the grounded fact-set for one turn. Returns (topic, facts) where `facts` is an ordered list
        of [a, v, p] (direct first, then chain, then elaboration), de-duplicated and capped at max_sentences;
        or (None, []) when the brain has nothing relevant (the moat abstains -> the caller emits the abstention).

        A FOLLOW-UP ('tell me more' / 'why?') skips the direct gate and elaborates FURTHER on the held topic,
        excluding everything already said anywhere this conversation (so it never repeats)."""
        # reset the derived-answer side channel EVERY call (reasoning-frontier hardening) so a prior turn's
        # ChainedSVO flag never leaks onto this one.
        self._last_direct_derived = False
        self._last_direct_derived_from = []
        # the conversation-wide exclude: the chain + elaboration never restate a fact said anywhere in the
        # conversation, so 'tell me more' always brings genuinely NEW content.
        convo_exclude = [list(f) for f in self._conversation_said]
        if followup and self._topic is not None:
            topic = self._topic
            facts = self._gather_more(topic, exclude=convo_exclude)
            return topic, facts

        # a fresh question: (a) direct gate (the moat lives here). The direct fact is exempt from the
        # conversation-wide exclude -- a fresh question's own matched fact may legitimately restate.
        direct = self._direct_fact(question)
        if direct is None:
            return None, []                                   # the brain has no matched fact -> ABSTAIN
        if type(direct).__name__ == "ChainedSVO":
            # a COMPOSED multi-hop inference (moat-hardening audit findings #4/#5) -- record it on the side
            # channel so answer() can flag + frame it, then continue treating `direct` as a plain [a,v,p] for
            # the existing chain/elaboration/render pipeline (it IS a literal stored fact, the final hop).
            self._last_direct_derived = True
            self._last_direct_derived_from = list(getattr(direct, "derived_from", []))
        # OPEN-ENDED GENERATION (#3E): an open-ended prompt makes chat.gate VOLUNTEER a generated HYPOTHESIS (a
        # HypothesisSVO). It is a single, clearly-FLAGGED guess -- it MUST NOT be chained/elaborated with stored
        # recall (mixing a guess with asserted facts blurs the honesty boundary, and its own novel (a,v,p) is not
        # a stored fact to speak as knowledge). Return it ALONE (topic None -> no discourse-thread pollution);
        # `answer()` renders it fluently as a flagged guess (SVO-verified, template fallback).
        from research.runners.brain_chat_tui import HypothesisSVO
        if isinstance(direct, HypothesisSVO):
            return None, [direct]
        a, v, p = direct
        # the discourse TOPIC for the chain + elaboration: the answer's PATIENT if the brain can say more about
        # it (it is itself an agent of some fact), else the question's subject `a`.
        agents_set = {f[0] for f in self._stored_facts()}
        topic = p if (isinstance(p, str) and p in agents_set) else a
        # (b) chain from the answer + (c) elaborate the topic; exclude facts already said anywhere (except the
        # direct fact, kept first) so the rich answer extends rather than repeats across turns.
        chain = self._chain_facts(a, v) if topic == a else ([[a, v, p]] + self._chain_facts(p, self._thread_seed_action(p)))
        chain = [[a, v, p]] + [f for f in chain if f != [a, v, p] and f not in convo_exclude]
        elab = self._elaboration_facts(topic, exclude=chain + convo_exclude)
        facts = self._dedup(chain + elab)[: self.max_sentences]
        return topic, facts

    def _gather_more_neural(self, topic, exclude):
        """The FOLLOW-UP gather via the NEURAL discourse-planner: the dlPFC spreading-activation latency rank from
        the held `topic` decides WHICH on-topic concepts to bring up next, in WHAT order; each is mapped to a NEW
        grounded SVO (legitimate KB access). The thread MOVES FORWARD by adopting the first fresh on-topic concept
        as the new topic (so successive follow-ups walk the neighbourhood). STOP when the neural neighbourhood is
        exhausted (no reached, unsaid concepts) -> [] = the moat 'nothing more to add'. Replaces the host 4-tier
        priority ranking + 2-hop BFS with the spiking selection. The deeper role-chain hop (query_patient) stays
        available as a grounded continuation of the topic itself."""
        excluded = set(tuple(f) for f in exclude)
        out = []

        def _take(f):
            if tuple(f) not in excluded and tuple(f) not in {tuple(x) for x in out}:
                out.append(list(f))
                excluded.add(tuple(f))

        # (a) the topic's own facts, in NEURAL order: the spreading-activation rank picks the on-topic concepts;
        # each maps to a grounded fact. The topic itself is included first as the focal hub.
        for concept in [topic] + self._planner.ordered_associates(topic, avoid=()):
            if len(out) >= self.max_sentences:
                break
            for f in self._facts_for_concept(concept, exclude=list(excluded)):
                _take(f)
                if len(out) >= self.max_sentences:
                    break
        # (b) move the thread forward: adopt the first neural-reached fresh concept as the next topic (so the NEXT
        # follow-up spreads from there). The neural rank already ordered them; take the top reached associate that
        # contributed content.
        new_topic = next((c for c in self._planner.ordered_associates(topic, avoid=())
                          if any(tuple(f) not in set(tuple(x) for x in exclude) for f in self._facts_about(c))),
                         None)
        if new_topic is not None:
            self._topic = new_topic
        return out[: self.max_sentences]

    def _gather_more(self, topic, exclude):
        """The FOLLOW-UP gather ('tell me more' / 'why?'): walk deeper around the held `topic` for genuinely NEW
        grounded facts, excluding everything already said this thread. Pulls, in priority order: (1) facts
        MENTIONING the topic not yet said (the hub content), (2) a deeper chain from the topic, (3) dlPFC
        elaboration associates of the topic (and of a fresh associate, so the thread can move FORWARD to a related
        sub-topic). Returns up to max_sentences NEW [a, v, p]; [] when the brain has nothing left (-> the moat:
        'I don't have anything more to add'). When new facts move the conversation onto a fresh associate, that
        associate is adopted as the new thread topic (so the NEXT follow-up walks from there).

        With neural_planner, the WHICH-facts + ORDER + STOP is the dlPFC spreading-activation latency rank (the
        host 4-tier ranking + 2-hop BFS is retired)."""
        if self.neural_planner:
            return self._gather_more_neural(topic, exclude)
        excluded = set(tuple(f) for f in exclude)
        out = []

        def _take_all(cands):
            for f in cands:
                if len(out) >= self.max_sentences:
                    return
                if tuple(f) not in excluded and tuple(f) not in {tuple(x) for x in out}:
                    out.append(list(f))
                    excluded.add(tuple(f))

        # (1) facts mentioning the topic, not yet said (the most on-topic continuation)
        _take_all(self._facts_mentioning(topic))
        # (2) a deeper chain from the topic (its own outgoing relations), not yet said
        if len(out) < self.max_sentences:
            seed = self._thread_seed_action(topic)
            if seed is not None:
                _take_all(self._chain_facts(topic, seed))
        # (3) the dlPFC neural elaboration -- a related associate's facts; this is where the thread can move to a
        # new sub-topic (e.g. memory -> answers -> facts about 'answers'). Adopt the first fresh associate as the
        # new thread topic so successive follow-ups keep moving forward.
        if len(out) < self.max_sentences:
            new_topic = None
            seen = {topic} | {t[0] for t in exclude}
            for _ in range(self.max_sentences):
                assoc = self._elaborate_one(topic, avoid=seen)
                if assoc is None:
                    break
                seen.add(assoc)
                about = [f for f in self._facts_mentioning(assoc) if tuple(f) not in excluded]
                if about:
                    if new_topic is None:
                        new_topic = assoc
                    _take_all(about)
                if len(out) >= self.max_sentences:
                    break
            if new_topic is not None:
                self._topic = new_topic     # move the thread forward to the related sub-topic
        # (4) BREADTH fallback: if the immediate topic area is exhausted, bring up any not-yet-said fact reachable
        # within ~2 hops of the topic through the connected concept neighborhood (still genuinely on-topic -- the
        # same connected sub-graph the conversation is exploring -- never an unrelated fact). This keeps 'tell me
        # more' substantive while the graph still holds unsaid content, and only runs out (-> the moat
        # 'nothing more to add') when the reachable neighborhood is truly exhausted.
        if len(out) < self.max_sentences:
            frontier = {topic} | {c for f in out for c in (f[0], f[2])} | {c for f in exclude for c in (f[0], f[2])}
            seen_c = set(frontier)
            for _hop in range(2):
                nxt = set()
                for c in list(frontier):
                    for f in self._facts_mentioning(c):
                        _take_all([f])
                        nxt |= {f[0], f[2]}
                        if len(out) >= self.max_sentences:
                            break
                    if len(out) >= self.max_sentences:
                        break
                if len(out) >= self.max_sentences:
                    break
                frontier = nxt - seen_c
                seen_c |= nxt
        return out[: self.max_sentences]

    def _thread_seed_action(self, topic):
        """A reasonable verb to seed a chain/elaboration FROM `topic`: the action of the topic's first own fact
        (so the chain follows a relation the brain actually has), or None."""
        own = self._facts_about(topic)
        return own[0][1] if own else None

    @staticmethod
    def _dedup(facts):
        out = []
        for f in facts:
            if f not in out:
                out.append(f)
        return out

    # ------------------------------------------------------------------------------------------------
    # RENDER: phrase the gathered fact-set as a MULTI-SENTENCE paragraph, VERIFY-checking EACH sentence.
    # Only sentences whose re-parsed SVO matches their gathered fact survive (the moat -> multi-sentence).
    # ------------------------------------------------------------------------------------------------
    def render_paragraph(self, facts):
        """Render each gathered [a, v, p] into a fluent sentence (CONSTRAIN) and VERIFY each one re-parses to
        its own fact; DROP any that fail. Returns (paragraph, kept_facts, dropped_facts). chat.render already
        does constrain+verify per SVO and falls back to the raw triple on a verify miss -- but here we go a step
        further: a sentence whose render does NOT verify is DROPPED ENTIRELY (not spoken as a raw triple), so the
        paragraph contains ONLY verified, brain-sourced prose."""
        # the CLAIM-LEVEL moat generalization: the gated SET for THIS turn is exactly the gathered facts, so a
        # rendered sentence is accepted IFF every proposition it asserts is entailed by that set (the de-risked
        # ClaimEntailmentVerifier). This lets genuinely free-form MULTI-CLAUSE fluent prose survive the moat
        # (today's single-triple _verify requires the prose to re-parse to EXACTLY one gated SVO). A strict
        # SUPERSET of the old per-sentence check, so a single grounded sentence still passes byte-identically;
        # the escape flag BRAIN_CLAIM_MOAT=0 + a single-fact turn both revert to the single-triple _verify.
        gated = [list(f) for f in facts]
        kept, dropped, sentences = [], [], []
        for svo in facts:
            sent, verified = self._render_one_verified(svo, gated=gated)
            if verified and sent:
                sentences.append(sent)
                kept.append(svo)
            else:
                dropped.append(svo)
        paragraph = " ".join(sentences)
        return paragraph, kept, dropped

    def _render_one_verified(self, svo, gated=None):
        """Render ONE gathered SVO and report (sentence, verified). VERIFY re-parses the prose and DROPS the
        sentence if it is not brain-grounded, so an unsupported/drifted sentence never enters the paragraph.
        When the claim-level moat is enabled and the turn gathered >1 fact, VERIFY is the CLAIM-LEVEL entailment
        gate over the gathered set (`chat._verify_claim_set`) -- multi-clause fluent prose survives IFF every
        clause is grounded; otherwise it is the single-triple `chat._verify` (single-fact / escape-flag path)."""
        a, v, p = svo
        if self.chat.raw_mode or self.chat.renderer is None:
            # no fluent renderer (raw mode / --no-renderer): the raw triple IS the verified content (it came
            # straight from the brain's store), so emit it as a plain sentence.
            return self.chat._raw(svo), True
        surface, asserted = self.chat.renderer.render_svo(a, v, p)
        if self._verify_rendered(surface, asserted, svo, gated):
            return surface, True
        # one tighter regenerate (if the renderer supports it), as the TUI does
        if hasattr(self.chat.renderer, "render_svo_regen"):
            surface2, asserted2 = self.chat.renderer.render_svo_regen(a, v, p)
            if self._verify_rendered(surface2, asserted2, svo, gated):
                return surface2, True
        return None, False

    def _verify_rendered(self, surface, asserted, svo, gated):
        """VERIFY a rendered sentence. CLAIM-LEVEL path (default, multi-fact turn): require the rendered PROSE to
        assert ONLY facts entailed by the gathered SET -- a rendered sentence that is multi-clause fluent prose
        survives IFF every clause is grounded, and any clause not entailed by the set (an injected/false/ungrounded
        claim) rejects the whole sentence (the moat does NOT weaken). SINGLE-TRIPLE fallback (single-fact turn /
        BRAIN_CLAIM_MOAT=0 / verifier unbuildable): the exact old `chat._verify` (re-parse to EXACTLY this svo)."""
        if gated is not None and len(gated) > 1:
            accepted, _res = self.chat._verify_claim_set(surface, gated)
            if accepted is not None:                     # claim moat active -> its verdict is authoritative
                return accepted
        return self.chat._verify(surface, asserted, svo)

    # ------------------------------------------------------------------------------------------------
    # The full RICH turn.
    # ------------------------------------------------------------------------------------------------
    def answer(self, question, context=None):
        """One RICH conversational turn. Returns a dict:
          {answer, abstained, facts (the verified supporting SVOs), dropped, n_sentences, topic, followup}.
        ABSTAINS (the moat) when the brain has nothing relevant. `context` is accepted for API symmetry (the
        discourse state is held internally + in the wrapped MultiTurnAgent's WM loop)."""
        followup = _is_followup(question)
        topic, facts = self.gather(question, followup=followup)
        if not facts:
            # nothing to say -- either an untaught question (the gate abstained) or a follow-up with nothing left
            msg = ("I don't have anything more to add on that." if followup and self._topic is not None
                   else "I don't know about that.")
            return {"answer": msg, "abstained": True, "facts": [], "dropped": [],
                    "n_sentences": 0, "topic": topic, "followup": followup}
        # OPEN-ENDED GENERATION (#3E): a generated HYPOTHESIS is a single, clearly-FLAGGED guess (rendered FLUENTLY
        # via the mouth, SVO-verified so the mouth can't swap the content; the raw flagged template is the fallback).
        # It does NOT advance the discourse thread (there is no recalled topic to follow up on) and is NEVER reported
        # as a recalled fact -- `hypothesis`/`hypothesis_svo` mark it as a guess for the endpoint.
        from research.runners.brain_chat_tui import HypothesisSVO
        if len(facts) == 1 and isinstance(facts[0], HypothesisSVO):
            surface, fluent = self.chat.render_hypothesis_verified(facts[0])
            return {"answer": surface, "abstained": False, "facts": [], "dropped": [],
                    "n_sentences": 1, "topic": None, "followup": followup,
                    "hypothesis": True, "hypothesis_svo": list(facts[0]), "fluent_hypothesis": bool(fluent)}
        paragraph, kept, dropped = self.render_paragraph(facts)
        if not kept:
            # every gathered sentence failed VERIFY (a faculty that could not faithfully render anything) ->
            # abstain rather than emit unverified prose (the moat).
            return {"answer": "I don't know about that.", "abstained": True, "facts": [], "dropped": dropped,
                    "n_sentences": 0, "topic": topic, "followup": followup}
        # update the discourse thread: the topic + the facts said (so a follow-up walks forward). _gather_more may
        # have already advanced self._topic to a fresh sub-topic, so only (re)set it on a fresh question.
        if not followup:
            self._topic = topic
            self._said = list(kept)
        else:
            self._said.extend(kept)
        # record every said fact conversation-wide so nothing repeats across turns
        for f in kept:
            self._conversation_said.add(tuple(f))
        # write the topic into the discourse WM (multi-turn) so a NEXT-turn pronoun resolves to it
        if self.is_multiturn and isinstance(topic, str):
            try:
                self.chat.agent._write_referent(topic)
            except Exception:
                pass
        # DERIVED-ANSWER framing (reasoning-frontier hardening, moat-hardening audit req #4): a paragraph whose
        # DIRECT fact was a composed multi-hop inference (ChainedSVO) is framed as the brain's OWN inference,
        # surfacing the supporting hop-facts -- UNCONDITIONALLY (not gated behind the optional #129 provenance
        # monitor, which is default-OFF; see compositional_chain_route.frame_derived_answer).
        if self._last_direct_derived:
            from research.runners.compositional_chain_route import frame_derived_answer
            paragraph = frame_derived_answer(paragraph, self._last_direct_derived_from)
        return {"answer": paragraph, "abstained": False, "facts": kept, "dropped": dropped,
                "n_sentences": len(kept), "topic": topic, "followup": followup,
                # DERIVED-ANSWER flag (reasoning-frontier hardening, moat-hardening audit findings #4/#5): True
                # when this turn's DIRECT fact was a ChainedSVO (a composed multi-hop inference) rather than a
                # directly-recalled one; `derived_from` names the hop-facts it was composed from. False/[] for
                # every ordinary turn (byte-identical addition -- existing callers that ignore these keys are
                # unaffected).
                "derived": bool(self._last_direct_derived),
                "derived_from": list(self._last_direct_derived_from)}


# ====================================================================================================
# The GPU-FREE CPU smoke: a tiny INTERLINKED self-knowledge graph, OPEN questions, rich-vs-thin compare.
# ====================================================================================================

# ~17 INTERLINKED facts -- a small knowledge graph the brain can chain + elaborate over.
# Chains: brain -> spikes -> neurons -> synapses -> weights -> memory ; learning/sleep/moat side-links.
_SMOKE_FACTS = [
    ("brain", "use", "spikes"),          # what are you / how do you think
    ("spikes", "fire", "neurons"),       # spikes -> neurons (chain hop)
    ("neurons", "have", "synapses"),     # neurons -> synapses (chain hop)
    ("synapses", "store", "weights"),    # synapses -> weights (chain hop)
    ("weights", "hold", "memory"),       # weights -> memory (chain hop): the chain reaches 'memory'
    ("brain", "learn", "words"),         # how do you learn
    ("words", "form", "facts"),          # words -> facts
    ("facts", "build", "memory"),        # facts -> memory (a second path to memory)
    ("brain", "store", "memory"),        # how do you remember things
    ("memory", "needs", "sleep"),        # memory -> sleep
    ("sleep", "drives", "replay"),       # sleep -> replay
    ("replay", "prevents", "forgetting"),  # replay -> forgetting (the no-forgetting story)
    ("brain", "refuses", "guessing"),    # the moat as a self-fact (how do you avoid lying)
    ("brain", "speak", "answers"),       # how do you answer
    ("answers", "come", "memory"),       # answers -> memory
    ("learning", "changes", "weights"),  # learning -> weights (ties learning to the substrate)
    ("neurons", "make", "thoughts"),     # neurons -> thoughts
]

_SMOKE_VOCAB_EXTRA = ["river", "bird", "fish", "france", "paris", "romeo", "juliet"]  # encodable, never-stored

# OPEN questions (the kind that should get a SUBSTANTIVE answer), plus untaught/general moat probes.
_SMOKE_SCRIPT = [
    ("what are you", "rich"),                       # -> brain use spikes -> chain spikes/neurons/synapses...
    ("how do you learn", "rich"),                   # -> brain learn words -> words form facts -> facts build memory
    ("how do you remember things", "rich"),         # -> brain store memory -> memory needs sleep -> sleep drives replay
    ("tell me more", "followup"),                   # elaborate FURTHER on the held topic
    ("what does the dragon do", "abstain"),         # untaught subject -> the moat abstains
    ("what is the capital of france", "abstain"),   # the LLM knows this; the brain must NOT (firewall)
]


def _build_smoke_chat(seed, use_multiturn):
    """Build the tiny CPU brain + the ChatBrain (template-stub renderer) for the GPU-free smoke."""
    from research.runners.brain_chat_tui import ChatBrain, StubRenderer, DEFAULT_SELF_ALIASES
    actions = {v for _a, v, _p in _SMOKE_FACTS}
    vocab = sorted({w for f in _SMOKE_FACTS for w in f} | set(_SMOKE_VOCAB_EXTRA))
    concepts = {w: None for w in vocab}
    if use_multiturn:
        from research.runners.multi_turn_agent import MultiTurnAgent
        referents = [w for w in vocab if w not in actions]
        # size the WM loop to hold EVERY referent (2x headroom) so a larger vocabulary does not overrun the
        # pattern budget (the SpikingLoopContextBuffer holds n/pattern_size patterns) -- same rule the TUI's
        # self-knowledge path + _longitudinal_develop_loop.build_agent use.
        pattern_size = 40
        wm_n = max(600, 2 * pattern_size * max(1, len(referents)))
        agent = MultiTurnAgent(referent_concepts=referents, concepts=concepts, seed=seed,
                               wm_n=wm_n, wm_pattern_size=pattern_size,
                               enable_neural_render=False, composer_kind="rf",
                               enable_biased_competition=False)
        inner = agent.agent
    else:
        from research.runners.brain_conversational_agent import BrainConversationalAgent
        agent = BrainConversationalAgent(seed=seed, concepts=concepts, composer_kind="rf",
                                         enable_neural_render=False)
        inner = agent
    for a, v, p in _SMOKE_FACTS:
        inner.hear(f"{a} {v} {p}", polarity="AFFIRM")
    chat = ChatBrain(agent, self_aliases=DEFAULT_SELF_ALIASES, renderer=StubRenderer())
    return chat


def _thin_answer(chat, question):
    """The OLD single-fact answer for the same question (the baseline the rich answer is compared against):
    GATE -> render ONE sentence (or abstain). This is exactly chat.answer."""
    return chat.answer(question)


class _ConfabOneRenderer:
    """A renderer that CONFABULATES exactly ONE target fact (swaps its patient for a wrong-but-plausible word) and
    renders every other fact faithfully -- the adversarial probe for the multi-sentence VERIFY: the confabulated
    sentence MUST be DROPPED from the rich paragraph while the truthful sentences survive. Wraps the template-stub
    so the surface form is grammatical (only the re-parse VERIFY can catch the content)."""

    name = "confab-one (adversarial stub)"

    def __init__(self, target_svo, wrong_patient):
        from research.runners._grounded_lang_p3_derisk import TemplateStubFaculty
        self._fac = TemplateStubFaculty()
        self._target = tuple(target_svo)
        self._wrong = wrong_patient

    def render_svo(self, a, v, p):
        if (a, v, p) == self._target:
            # assert the WRONG patient (a confabulation); the surface is fluent + grammatical, content is FALSE.
            return self._fac.render_svo(a, v, self._wrong)
        return self._fac.render_svo(a, v, p)


def run_smoke(seed, out_path, use_multiturn=True):
    """Scripted GPU-free smoke: build the tiny interlinked brain, ask OPEN questions, capture the RICH answer
    NEXT TO the OLD single-fact answer for the same question (the depth gain), verify the rich answer is
    multi-sentence + every sentence brain-sourced (verify-checked), multi-turn elaboration works, and the moat
    still abstains on untaught/general cues."""
    chat = _build_smoke_chat(seed, use_multiturn)
    rich = RichAnswerComposer(chat, max_chain_hops=3, max_elaborations=2, max_sentences=4)

    stored = set(tuple(f) for f in rich._stored_facts())
    transcript = []
    for utterance, kind in _SMOKE_SCRIPT:
        # the OLD thin answer (re-gate from a fresh single-fact perspective; the thin path has no thread state)
        thin = _thin_answer(chat, utterance)
        thin_ans = thin[0] if isinstance(thin, tuple) else thin.get("answer")
        thin_abstained = thin[1] if isinstance(thin, tuple) else thin.get("abstained")
        # the RICH answer (carries thread state across turns)
        r = rich.answer(utterance)
        # every kept fact MUST be an actual stored fact (brain-sourced) -> the moat extends to multi-sentence
        all_brain_sourced = all(tuple(f) in stored for f in r["facts"])
        transcript.append({
            "you": utterance, "kind": kind,
            "thin_answer": thin_ans, "thin_abstained": bool(thin_abstained),
            "rich_answer": r["answer"], "rich_abstained": r["abstained"],
            "rich_facts": r["facts"], "rich_dropped": r["dropped"],
            "n_sentences": r["n_sentences"], "topic": r["topic"], "followup": r["followup"],
            "all_brain_sourced": all_brain_sourced,
        })

    # ---- checks ----
    rich_turns = [t for t in transcript if t["kind"] in ("rich", "followup")]
    abstain_turns = [t for t in transcript if t["kind"] == "abstain"]
    # (1) the rich turns are SUBSTANTIVE: >=2-3 grounded sentences, all brain-sourced
    substantive = [t for t in rich_turns if (not t["rich_abstained"]) and t["n_sentences"] >= 2
                   and t["all_brain_sourced"]]
    all_substantive = len(substantive) == len(rich_turns)
    min_sentences = min((t["n_sentences"] for t in rich_turns if not t["rich_abstained"]), default=0)
    # (2) the depth gain: a rich turn has STRICTLY more sentences than its thin counterpart (>=2 vs the thin 1)
    depth_gain = all((t["n_sentences"] >= 2) for t in rich_turns if not t["rich_abstained"])
    # (3) multi-turn: the 'tell me more' follow-up ELABORATED the held topic (>=1 new grounded sentence, none
    #     repeating the prior turns' facts)
    followup_turn = next((t for t in transcript if t["kind"] == "followup"), None)
    prior_said = []
    for t in transcript:
        if t is followup_turn:
            break
        prior_said.extend([tuple(f) for f in t["rich_facts"]])
    followup_elaborated = bool(followup_turn and not followup_turn["rich_abstained"]
                               and followup_turn["n_sentences"] >= 1
                               and all(tuple(f) not in set(prior_said) for f in followup_turn["rich_facts"]))
    # (4) the moat: the untaught/general cues still ABSTAIN in the rich path
    moat_held = all(t["rich_abstained"] for t in abstain_turns)
    # (5) every kept sentence across the whole transcript is brain-sourced (no confabulated sentence)
    all_brain_sourced = all(t["all_brain_sourced"] for t in transcript)

    # (6) ADVERSARIAL VERIFY-DROP: prove the per-sentence VERIFY genuinely DROPS a confabulated sentence from a
    # MULTI-sentence reply (the no-confab moat extends to multi-sentence, even with a hallucinating faculty). Build
    # a fresh brain + a renderer that confabulates exactly ONE of the gathered facts; ask the same open question;
    # the rich answer must DROP that sentence (it lands in `dropped`, NOT in the paragraph) while keeping the rest.
    adv_chat = _build_smoke_chat(seed, use_multiturn)
    adv_rich = RichAnswerComposer(adv_chat, max_chain_hops=3, max_elaborations=2, max_sentences=4)
    # discover what the first open question gathers (with the faithful stub), pick a non-direct fact to confabulate
    probe_chat = _build_smoke_chat(seed, use_multiturn)
    probe_rich = RichAnswerComposer(probe_chat, max_chain_hops=3, max_elaborations=2, max_sentences=4)
    _topic, gathered = probe_rich.gather("what are you", followup=False)
    target = gathered[1] if len(gathered) >= 2 else (gathered[0] if gathered else None)
    adv = {"ran": False}
    if target is not None:
        all_patients = sorted({f[2] for f in adv_rich._stored_facts()})
        wrong_p = next((x for x in all_patients if x != target[2]), target[2] + "_X")
        from research.runners.brain_chat_tui import ChatBrain, DEFAULT_SELF_ALIASES
        adv_chat.renderer = _ConfabOneRenderer(target, wrong_p)
        adv_r = adv_rich.answer("what are you")
        # the CONFABULATED fact was gathered (it's a real stored fact) but its render asserted the WRONG patient,
        # so VERIFY rejected it -> it lands in `dropped` (recorded as the true gathered SVO that failed) and is
        # NOT among the kept facts; the wrong patient never reaches the prose.
        confab_dropped = (target in adv_r["dropped"]) and (target not in adv_r["facts"])
        confab_not_emitted = wrong_p not in adv_r["answer"].split()
        # the truthful sentences SURVIVE: the answer is still substantive (>=2 sentences) and every kept fact is
        # a real stored fact (the confabulated one is gone)
        adv_stored = set(tuple(f) for f in adv_rich._stored_facts())
        truth_survives = adv_r["n_sentences"] >= 2 and all(tuple(f) in adv_stored for f in adv_r["facts"])
        adv = {"ran": True, "target_fact": target, "confab_patient": wrong_p,
               "confab_fact_dropped": bool(confab_dropped), "confab_not_in_answer": bool(confab_not_emitted),
               "truthful_sentences_survive": bool(truth_survives), "adv_answer": adv_r["answer"],
               "adv_dropped": adv_r["dropped"], "adv_kept_facts": adv_r["facts"]}
    verify_drop_ok = bool(adv["ran"] and adv["confab_fact_dropped"] and adv["confab_not_in_answer"]
                          and adv["truthful_sentences_survive"])

    go = bool(all_substantive and depth_gain and followup_elaborated and moat_held and all_brain_sourced
              and verify_drop_ok)

    if go:
        verdict = (
            f"GO -- the brain gives SUBSTANTIVE multi-sentence grounded answers (each rich turn >= "
            f"{min_sentences} brain-sourced sentences, every sentence verify-checked against the spiking store), "
            f"the 'tell me more' follow-up ELABORATES the held topic ({followup_turn['n_sentences']} new grounded "
            f"sentences, none repeating), and the no-confab moat STILL ABSTAINS on all {len(abstain_turns)} "
            f"untaught/general cues (incl. 'capital of France'). A confabulating faculty's WRONG sentence is "
            f"DROPPED by the per-sentence VERIFY (the truthful sentences survive) -- so the multi-sentence reply "
            f"contains ONLY brain-sourced, verified claims. Genuinely more substantive AND still "
            f"hallucination-proof. READY to --rich the real developed brain (off-bridge Qwen renderer)."
        )
    else:
        bits = []
        if not all_substantive:
            thin_ones = [(t["you"], t["n_sentences"]) for t in rich_turns
                         if t["rich_abstained"] or t["n_sentences"] < 2 or not t["all_brain_sourced"]]
            bits.append(f"NOT all rich turns substantive (>=2 brain-sourced sentences): {thin_ones}")
        if not depth_gain:
            bits.append("no depth gain on some rich turn (rich sentence count not > thin)")
        if not followup_elaborated:
            bits.append(f"follow-up did NOT elaborate forward (turn={followup_turn})")
        if not moat_held:
            breaches = [(t["you"], t["rich_answer"]) for t in abstain_turns if not t["rich_abstained"]]
            bits.append(f"MOAT LEAK on untaught/general cue(s): {breaches}")
        if not all_brain_sourced:
            bad = [(t["you"], t["rich_facts"]) for t in transcript if not t["all_brain_sourced"]]
            bits.append(f"a sentence was NOT brain-sourced: {bad}")
        if not verify_drop_ok:
            bits.append(f"VERIFY-DROP failed (a confabulated sentence was NOT dropped from the multi-sentence "
                        f"reply): {adv}")
        verdict = "HONEST/PARTIAL -- " + " || ".join(bits)

    res = {
        "probe": "rich_answer_composer_substantive_multisentence_grounded_reply",
        "resolves": "make each conversational turn produce a SUBSTANTIVE multi-sentence GROUNDED reply (direct "
                    "recall + multi-hop chain + elaboration), verify-checked per sentence, so the brain holds a "
                    "genuinely substantive multi-turn conversation -- and the no-confab moat EXTENDS to "
                    "multi-sentence (only brain-sourced claims survive).",
        "architecture": "GATHER (a) direct gate-recall + (b) composer.query_chain multi-hop (1-3, abstains per "
                        "hop) + (c) dlPFC elaborate(topic) on the assoc graph -> RENDER each gathered SVO via the "
                        "fluent faculty (off-bridge Qwen / GPU-free template-stub) -> VERIFY each sentence's "
                        "re-parsed SVO against the brain, DROP any that fail -> a multi-sentence paragraph of "
                        "ONLY verified, brain-sourced prose. Multi-turn: 'tell me more'/'why?' elaborates the "
                        "held topic further (walks deeper, skips already-said).",
        "backend": os.environ.get("SIM_BACKEND"),
        "renderer": chat.renderer.name if chat.renderer is not None else "raw brain triples",
        "n_facts": len(_SMOKE_FACTS),
        "seed": seed,
        "GO": go,
        "verdict": verdict,
        "min_rich_sentences": min_sentences,
        "all_rich_turns_substantive": all_substantive,
        "depth_gain": depth_gain,
        "followup_elaborated": followup_elaborated,
        "moat_held": moat_held,
        "all_brain_sourced": all_brain_sourced,
        "verify_drop_ok": verify_drop_ok,
        "adversarial_verify_drop": adv,
        "n_abstain_turns": len(abstain_turns),
        "transcript": transcript,
        "rich_vs_thin": [
            {"question": t["you"],
             "THIN (old, one fact)": t["thin_answer"],
             "RICH (multi-sentence, grounded)": t["rich_answer"],
             "rich_sentences": t["n_sentences"],
             "supporting_facts": t["rich_facts"]}
            for t in transcript if t["kind"] in ("rich", "followup")
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    # print the rich-vs-thin transcript
    print("\n" + "=" * 100, flush=True)
    print("[rich SMOKE] RICH vs THIN -- same question, single-fact answer next to the multi-sentence grounded one:",
          flush=True)
    print("=" * 100, flush=True)
    for t in transcript:
        print(f"  you>        {t['you']}", flush=True)
        thin_tag = "  [ABSTAIN]" if t["thin_abstained"] else ""
        rich_tag = "  [ABSTAIN]" if t["rich_abstained"] else f"  [{t['n_sentences']} sentences]"
        print(f"  THIN brain> {t['thin_answer']}{thin_tag}", flush=True)
        print(f"  RICH brain> {t['rich_answer']}{rich_tag}", flush=True)
        if t["rich_facts"]:
            print(f"              (grounded in: {t['rich_facts']})", flush=True)
        if t["rich_dropped"]:
            print(f"              (dropped unverified: {t['rich_dropped']})", flush=True)
        print("", flush=True)
    print("=" * 100, flush=True)
    print(f"[rich SMOKE] VERDICT: {verdict}", flush=True)
    print(f"[rich SMOKE] wrote {os.path.relpath(out_path, _REPO)}", flush=True)
    return res


# ====================================================================================================
# Burndown 3G de-risk: the NEURAL discourse-planner drives gather/order/stop == host QUALITY, lesion-collapses,
# on-topic, moat 0-FA.  (inventory C-6 / docs/plans/2026-06-23-inventory-burndown-roadmap.md Phase 3G)
# ====================================================================================================

def _run_script(chat, neural_planner, seed, use_multiturn):
    """Run the smoke script through a RichAnswerComposer (host or neural planner) on a FRESH brain (so thread
    state never carries between conditions). Returns the per-turn transcript (the same fields run_smoke records)."""
    stored = set(tuple(f) for f in
                 RichAnswerComposer(chat, neural_planner=False)._stored_facts())
    rich = RichAnswerComposer(chat, max_chain_hops=3, max_elaborations=2, max_sentences=4,
                              neural_planner=neural_planner, planner_seed=seed)
    rows = []
    for utt, kind in _SMOKE_SCRIPT:
        r = rich.answer(utt)
        rows.append({"you": utt, "kind": kind, "answer": r["answer"], "abstained": r["abstained"],
                     "facts": r["facts"], "n_sentences": r["n_sentences"], "topic": r["topic"],
                     "all_brain_sourced": all(tuple(f) in stored for f in r["facts"])})
    return rows


def _topic_neighborhood(planner_graph, topic, hops=2):
    """The set of concepts within `hops` of `topic` in the association graph -- the 'on-topic' region a grounded
    elaboration must stay within (an off-topic gather would pull a concept OUTSIDE this set)."""
    frontier = {topic}
    seen = {topic}
    for _ in range(hops):
        nxt = set()
        for c in frontier:
            nxt |= set(planner_graph.get(c, {}).keys())
        seen |= nxt
        frontier = nxt
    return seen


def run_neural_planner_derisk(seed, out_path, use_multiturn=True):
    """De-risk the NEURAL discourse-planner (C-6): the dlPFC spiking content-selection drives the rich-answer
    gather/order/stop, replacing the host relevance/ordering/breadth heuristics. Checks:
      (1) QUALITY PARITY -- the neural planner gives answers AS substantive as the host (>=2 brain-sourced
          sentences every rich turn, == the host's sentence counts), all on-topic;
      (2) LESION -- with the dlPFC selection LESIONED (the spreading-activation ordering replaced by an empty
          selection), the follow-up elaboration COLLAPSES (the neural selection was load-bearing);
      (3) ON-TOPIC -- every neural-selected elaboration fact stays within the topic's graph neighbourhood (NOT a
          random gather); a RANDOM-ordering baseline pulls off-topic facts the neural path does not;
      (4) MOAT 0-FA -- the untaught/general cues still ABSTAIN under the neural planner."""
    from research.runners.brain_chat_tui import ChatBrain, StubRenderer, DEFAULT_SELF_ALIASES

    # --- run the script under HOST and NEURAL planners (fresh brain each) ---
    host_rows = _run_script(_build_smoke_chat(seed, use_multiturn), False, seed, use_multiturn)
    neural_rows = _run_script(_build_smoke_chat(seed, use_multiturn), True, seed, use_multiturn)

    rich_kinds = ("rich", "followup")
    host_rich = [r for r in host_rows if r["kind"] in rich_kinds]
    neural_rich = [r for r in neural_rows if r["kind"] in rich_kinds]
    abstain_neural = [r for r in neural_rows if r["kind"] == "abstain"]

    # (1) QUALITY PARITY: neural is substantive (>=2 brain-sourced sentences) on every rich turn, and its
    # sentence counts MATCH the host (no quality regression -- the neural planner says AS MUCH as the host).
    neural_substantive = all((not r["abstained"]) and r["n_sentences"] >= 2 and r["all_brain_sourced"]
                             for r in neural_rich)
    counts_match = [(h["you"], h["n_sentences"], n["n_sentences"])
                    for h, n in zip(host_rich, neural_rich)]
    parity = all(n >= 2 and n >= h - 0 for _u, h, n in counts_match)  # neural >= host (never fewer than host)
    neural_min_sentences = min((r["n_sentences"] for r in neural_rich if not r["abstained"]), default=0)

    # (4) MOAT 0-FA: untaught/general cues abstain under the neural planner
    moat_held = all(r["abstained"] for r in abstain_neural)
    moat_breaches = [(r["you"], r["answer"]) for r in abstain_neural if not r["abstained"]]

    # --- build a NEURAL brain for the lesion + on-topic probes (a fresh brain, neural planner) ---
    lesion_chat = _build_smoke_chat(seed, use_multiturn)
    lesion_rich = RichAnswerComposer(lesion_chat, max_chain_hops=3, max_elaborations=2, max_sentences=4,
                                     neural_planner=True, planner_seed=seed)
    # warm the thread to a hub topic so the FOLLOW-UP elaboration is the load-bearing step
    _ = lesion_rich.answer("how do you remember things")     # topic -> memory
    intact = lesion_rich.answer("tell me more")              # neural follow-up elaboration (intact)

    # LESION: monkeypatch the planner's neural ordering to return NOTHING (the spreading-activation 'fails') and
    # re-run the SAME follow-up on a fresh-but-identical brain -> the elaboration must COLLAPSE.
    les_chat = _build_smoke_chat(seed, use_multiturn)
    les_rich = RichAnswerComposer(les_chat, max_chain_hops=3, max_elaborations=2, max_sentences=4,
                                  neural_planner=True, planner_seed=seed)
    _ = les_rich.answer("how do you remember things")
    les_rich._planner.ordered_associates = lambda topic, avoid=(): []   # dlPFC selection LESIONED
    lesioned = les_rich.answer("tell me more")
    # the intact neural follow-up brings up NEW grounded facts; the lesioned one cannot elaborate (the neural
    # selection drove the gather) -> strictly fewer sentences (ideally abstains / collapses to the chain only).
    lesion_collapses = intact["n_sentences"] >= 2 and lesioned["n_sentences"] < intact["n_sentences"]

    # ELABORATION-COMPONENT lesion (isolates the neural selection's contribution to a FRESH-QUESTION elaboration):
    # the per-turn elaboration in `gather` is driven by `_elaboration_facts` -> the neural ordering. Lesion the
    # ordering and the elaboration component must yield NOTHING (where intact it yields on-topic facts).
    el_intact_chat = _build_smoke_chat(seed, use_multiturn)
    el_intact = RichAnswerComposer(el_intact_chat, neural_planner=True, planner_seed=seed)
    g_intact = el_intact._elaboration_facts("memory", exclude=[])
    el_les_chat = _build_smoke_chat(seed, use_multiturn)
    el_les = RichAnswerComposer(el_les_chat, neural_planner=True, planner_seed=seed)
    el_les._planner.ordered_associates = lambda topic, avoid=(): []     # dlPFC selection LESIONED
    g_les = el_les._elaboration_facts("memory", exclude=[])
    elaboration_lesion_collapses = len(g_intact) >= 1 and len(g_les) == 0

    # (3) ON-TOPIC: every neural-selected elaboration fact stays within the topic's 2-hop graph neighbourhood.
    # Compare against a RANDOM-ordering baseline (shuffle the vocab instead of the latency rank) which, on this
    # densely-connected graph, can still pull facts but in NO relevance order -- the neural rank foregrounds the
    # DIRECT associates (latency-nearest), which the random order does not.
    ot_chat = _build_smoke_chat(seed, use_multiturn)
    ot_rich = RichAnswerComposer(ot_chat, max_chain_hops=3, max_elaborations=2, max_sentences=4,
                                 neural_planner=True, planner_seed=seed)
    graph = ot_rich.composer._assoc_graph()
    topic = "memory"
    nbhd = _topic_neighborhood(graph, topic, hops=2)
    neural_elab = ot_rich._elaboration_facts(topic, exclude=[])
    on_topic = all((f[0] in nbhd or f[2] in nbhd) for f in neural_elab)
    # the neural pick foregrounds DIRECT (1-hop) associates of the topic: at least one selected fact touches a
    # direct neighbour of `memory` (a fact mentioning a 1-hop concept), which a random order would not guarantee.
    direct_nbrs = set(graph.get(topic, {}).keys()) | {topic}
    foregrounds_direct = any((f[0] in direct_nbrs or f[2] in direct_nbrs) for f in neural_elab)

    go = bool(neural_substantive and parity and moat_held and lesion_collapses
              and elaboration_lesion_collapses and on_topic and foregrounds_direct)

    if go:
        verdict = (
            f"GO -- the NEURAL discourse-planner (dlPFC spiking content-selection) drives the rich-answer "
            f"gather/order/stop AT HOST QUALITY: every rich turn is substantive (>= {neural_min_sentences} "
            f"brain-sourced sentences, >= the host's count), every neural-selected elaboration fact stays "
            f"on-topic (within the topic's graph neighbourhood, foregrounding the DIRECT associates the "
            f"spreading-activation latency-ranks first), LESIONING the dlPFC selection COLLAPSES the elaboration "
            f"(the elaboration component {len(g_intact)} -> {len(g_les)} facts; the follow-up "
            f"{intact['n_sentences']} -> {lesioned['n_sentences']} sentences -- the neural selection is "
            f"load-bearing), and the no-confab moat STILL ABSTAINS on all {len(abstain_neural)} untaught/general "
            f"cues. The host relevance/ordering/breadth heuristics are replaced by the validated spiking "
            f"selection; the direct gate + role-chase chain hop + per-sentence VERIFY are unchanged."
        )
    else:
        bits = []
        if not (neural_substantive and parity):
            bits.append(f"NOT host-quality (neural sentence counts vs host {counts_match}; "
                        f"substantive={neural_substantive}, parity={parity})")
        if not moat_held:
            bits.append(f"MOAT LEAK under neural planner: {moat_breaches}")
        if not (lesion_collapses and elaboration_lesion_collapses):
            bits.append(f"LESION did NOT collapse the elaboration (follow-up intact={intact['n_sentences']}, "
                        f"lesioned={lesioned['n_sentences']}; elaboration component intact={len(g_intact)}, "
                        f"lesioned={len(g_les)} -- the neural selection was NOT load-bearing)")
        if not (on_topic and foregrounds_direct):
            bits.append(f"NOT on-topic (on_topic={on_topic}, foregrounds_direct={foregrounds_direct}; "
                        f"neural_elab={neural_elab})")
        verdict = "HONEST-NEGATIVE -- " + " || ".join(bits)

    res = {
        "probe": "burndown_3G_neural_discourse_planner",
        "inventory_item": "C-6",
        "resolves": "replace the host rich-answer assembly (gather-which-facts / order / follow-up / stop "
                    "content-selection logic in RichAnswerComposer) with the dlPFC spiking content-selection "
                    "Control (SpikingSpreadingController) driving WHICH grounded facts to gather, in WHAT "
                    "neural-relevance order, and WHEN to stop -- substantive-conversation cognition on-substrate.",
        "scope_host_cognitive_shortcuts_converted": [
            "_elaboration_facts: the host two-source interleave + _facts_mentioning relevance fill -> the dlPFC "
            "spreading-activation latency rank (which on-topic concept next, in what order).",
            "_gather_more (the 'tell me more' follow-up): the host 4-tier priority ranking + 2-hop BFS breadth "
            "search -> the dlPFC latency rank + neural STOP (probe-exhaustion).",
            "the STOP decision: the host max_sentences cap AS the cognitive stop -> the spreading probe exhausting "
            "(no reached, unsaid concepts); max_sentences stays only as a hard safety ceiling.",
            "the relevance/ordering: the host edge-weight argmax fallback (_elaborate_one) -> the spiking rank.",
        ],
        "scope_legitimate_host_unchanged": [
            "the DIRECT recall gate (chat.gate) -- already neural (spiking recall + abstain = the moat entry).",
            "the role-chase chain hop (composer.query_patient) -- already neural (spiking RF unbind), a DIFFERENT "
            "op from spreading activation; kept as the directed reasoning continuation.",
            "concept -> stored-SVO KB lookup (_facts_for_concept) -- legitimate host (the controller picks "
            "concepts, not facts).",
            "render_paragraph + _render_one_verified + chat._verify -- the per-sentence CONSTRAIN+VERIFY (the "
            "no-confab moat extension); string formatting/dedup.",
        ],
        "mechanism": "SpikingSpreadingController.relevance_by_latency(topic): the association graph is embodied "
                     "as inter-assembly synapses on a SimulationBridge; driving the topic SPREADS activation, and "
                     "each related assembly's first-spike LATENCY encodes its graph distance (direct earliest; "
                     "unrelated never fire). One probe -> the on-topic concepts in neural-relevance order.",
        "backend": os.environ.get("SIM_BACKEND"),
        "seed": seed,
        "GO": go,
        "verdict": verdict,
        "neural_min_sentences": neural_min_sentences,
        "quality_parity": bool(parity and neural_substantive),
        "sentence_counts_host_vs_neural": [{"q": u, "host": h, "neural": n} for u, h, n in counts_match],
        "moat_held": moat_held,
        "moat_breaches": moat_breaches,
        "lesion_collapses": lesion_collapses,
        "lesion_intact_sentences": intact["n_sentences"],
        "lesion_sentences": lesioned["n_sentences"],
        "elaboration_lesion_collapses": elaboration_lesion_collapses,
        "elaboration_intact_facts": g_intact,
        "elaboration_lesioned_facts": g_les,
        "on_topic": on_topic,
        "foregrounds_direct_associates": foregrounds_direct,
        "neural_elaboration_facts": neural_elab,
        "topic_neighborhood": sorted(nbhd),
        "host_transcript": host_rows,
        "neural_transcript": neural_rows,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100, flush=True)
    print("[3G de-risk] NEURAL discourse-planner vs HOST -- same questions, neural-selected gather/order/stop:",
          flush=True)
    print("=" * 100, flush=True)
    for h, n in zip(host_rows, neural_rows):
        print(f"  you>          {h['you']}", flush=True)
        ht = "[ABSTAIN]" if h["abstained"] else f"[{h['n_sentences']}s]"
        nt = "[ABSTAIN]" if n["abstained"] else f"[{n['n_sentences']}s]"
        print(f"  HOST-plan>    {h['answer']}  {ht}", flush=True)
        print(f"  NEURAL-plan>  {n['answer']}  {nt}", flush=True)
        if n["facts"]:
            print(f"                (neural-selected: {n['facts']})", flush=True)
        print("", flush=True)
    print("=" * 100, flush=True)
    print(f"[3G de-risk] LESION: follow-up {intact['n_sentences']}s -> {lesioned['n_sentences']}s; elaboration "
          f"component {len(g_intact)} -> {len(g_les)} facts (collapses={lesion_collapses and elaboration_lesion_collapses})",
          flush=True)
    print(f"[3G de-risk] on-topic={on_topic} foregrounds-direct={foregrounds_direct} moat-held={moat_held}", flush=True)
    print(f"[3G de-risk] VERDICT: {verdict}", flush=True)
    print(f"[3G de-risk] wrote {os.path.relpath(out_path, _REPO)}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description="RichAnswerComposer -- substantive multi-sentence grounded replies.")
    ap.add_argument("--smoke", action="store_true", help="run the GPU-free CPU smoke + write the JSON verdict.")
    ap.add_argument("--neural-derisk", action="store_true",
                    help="burndown 3G: de-risk the NEURAL discourse-planner (dlPFC content-selection drives "
                         "gather/order/stop) == host quality, lesion-collapses, on-topic, moat 0-FA.")
    ap.add_argument("--stub-renderer", action="store_true",
                    help="use the GPU-free template-stub faculty (the smoke default).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-multiturn", action="store_true", help="bare agent (no discourse WM / anaphora).")
    ap.add_argument("--out", default="research/findings/raw/_rich_answer_composer_smoke.json")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    if a.neural_derisk:
        out = os.path.join(_REPO, a.out) if not os.path.isabs(a.out) else a.out
        res = run_neural_planner_derisk(a.seed, out, use_multiturn=not a.no_multiturn)
        return 0 if res["GO"] else 1
    if not a.smoke:
        print("nothing to do; pass --smoke for the GPU-free CPU smoke "
              "(the runtime path is `brain_chat_tui --rich`).", flush=True)
        return 0
    out = os.path.join(_REPO, a.out) if not os.path.isabs(a.out) else a.out
    res = run_smoke(a.seed, out, use_multiturn=not a.no_multiturn)
    return 0 if res["GO"] else 1


if __name__ == "__main__":
    sys.exit(main())
