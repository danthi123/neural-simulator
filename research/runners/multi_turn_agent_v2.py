"""MultiTurnAgentV2 -- the production multi-turn conversational agent with an ORDER-ENCODED discourse buffer.

WHY V2 EXISTS. The production `MultiTurnAgent` (research/runners/multi_turn_agent.py) holds discourse referents
in the rate-attractor `SpikingLoopContextBuffer` -- a SET with NO order, whose winner is fixed by intrinsic
basin asymmetry. That substrate is GO for the SINGLE-referent case (one held antecedent -> a turn-2 pronoun
resolves), but it is exactly the substrate that FAILED multi-referent disambiguation across three converging
negatives -- recency, salience-boost, biased-competition WTA (2026-06-17-multireferent-disambiguation-NEGATIVE.md):
with several referents held, a bare pronoun cannot be resolved to the foregrounded one, because the buffer has no
notion of "which slot / how recent".

THE FIX (validated CYCLE 135, 2026-06-17-ordered-wm-position-binding-derisk.md). Replace the discourse buffer with
the ORDER-ENCODED working memory `OrderedPositionWM` (research/runners/ordered_position_wm.py): each referent is
bound to a successive gamma-slot POSITION phasor on the spiking resonate-and-fire substrate; a bare pronoun
resolves to the MOST-RECENT slot's referent via spiking `unbind`. The winner is "which slot you read", so it
FLIPS deterministically when the discourse order changes -- the order-control the rate buffer could not pass
(0/6 -> 6/6). The no-confab moat is FREE from the composer's familiarity gate: a pronoun with no referent held
(empty discourse) -> ABSTAIN (None).

DESIGN. `MultiTurnAgentV2` wraps the same `BrainConversationalAgent` (parser + composer + dlPFC) as V1. Its
discourse buffer is an `OrderedPositionWM` built with the SAME (seed, D=128, vocab) as the agent's composer -- so
the WM's concept codes are byte-identical to the composer's (both draw from `default_rng(seed)` over the sorted
vocab), and a word read out of a slot is a genuine composer concept the Q&A path can use directly. The WM cleans
up slot reads against the REFERENT subset only (so a discourse pronoun never resolves to an action word). The
familiarity abstention threshold is set by the PRINCIPLED separation rule (`OrderedPositionWM.calibrate_threshold`,
the `cleanup_separated` placement), NOT the de-risk's marginal frozen 0.15.

Reuse-by-import; NO `sim/` edit. The single-referent production path is preserved (one referent -> the most-recent
slot is that referent -> resolves), so V2 is a strict superset of V1's anaphora capability.
"""
from __future__ import annotations

from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.runners.ordered_position_wm import OrderedPositionWM

_ANAPHORS = {"it", "that", "them", "they", "this"}
_CORRECTION_MARKERS = {"actually", "no", "wait", "correction"}   # optional leading cue for a reconsolidation turn

# Surface pronoun emitted for a recurring (singular) subject in narrate(). The matching anaphor "it" is in
# _ANAPHORS, so the emitted pronoun is the same token the agent resolves on the substrate.
_NARRATE_PRONOUN = "it"


class MultiTurnAgentV2:
    """Multi-turn dialogue with an order-encoded (gamma-slot position-binding) discourse buffer on the validated
    spiking substrate. Resolves a bare pronoun to the FOREGROUNDED (most-recent slot) referent among SEVERAL held.

    Args:
        referent_concepts: the nouns the discourse buffer can hold (a slot read cleans up against these only).
        concepts: the full composer vocabulary (nouns + actions). ``{word: None}`` selects the vocab; a
            ``{word: code}`` dict supplies grounded codes to the composer (the WM shares the composer's codes).
        grounded_codes: optional learned phasor codes passed to the composer.
        seed: per-seed determinism. The discourse WM is built with the SAME seed/D/vocab as the composer so their
            concept codes match.
        wm_n_slots: gamma slots in the discourse buffer (default 7, the Lisman-Idiart ceiling). The buffer holds a
            sliding window of the last ``wm_n_slots`` introduced referents.
        enable_neural_render: forwarded to the agent (neural serial-order rendering of generated sentences).
    """

    def __init__(self, referent_concepts, concepts=None, grounded_codes=None, seed=42,
                 wm_n_slots=7, enable_neural_render=False, composer_kind="rf"):
        self.seed = int(seed)
        # composer_kind passes through to the inner agent: "rf" (default, the production numpy composer) or
        # "onebrain" (the integrated one-brain composer -- the cleanup arc validates correction + anaphora on it).
        self.agent = BrainConversationalAgent(seed=seed, concepts=concepts, grounded_codes=grounded_codes,
                                              enable_neural_render=enable_neural_render, composer_kind=composer_kind)
        self.referents = list(referent_concepts)
        # The discourse buffer shares the composer's concept codes: same seed, same D (=128, the agent composer's
        # D), same vocab (the composer's sorted vocab). A slot read cleans up against the referent subset only, so
        # a discourse pronoun resolves to a referent, never an action word. Familiarity threshold = the principled
        # calibrated value (cleanup_separated placement), NOT the de-risk's frozen 0.15.
        comp = self.agent.composer
        self.wm = OrderedPositionWM(seed=seed, D=comp.D, vocab=comp.words, n_slots=int(wm_n_slots),
                                    cleanup_words=self.referents)
        # The sliding window of introduced referents (most-recent last) and the current encoded composite.
        self._window = []                 # list[str], the referents currently bound to slots 0..len-1
        self._composite = None            # the bundled position-binding composite of self._window (None = empty)

    # --- discourse state (order-encoded) -------------------------------------
    def _write_referent(self, ref):
        """Introduce a referent into the order-encoded discourse buffer: append it to the sliding window (most
        recent last, capped at n_slots) and re-encode the position-binding composite on the spiking RF substrate.
        Non-referent or None inputs are ignored (only nouns the buffer can hold become discourse referents)."""
        if not (isinstance(ref, str) and ref in self.referents):
            return
        self._window.append(ref)
        if len(self._window) > self.wm.n_slots:
            self._window = self._window[-self.wm.n_slots:]      # keep the most-recent n_slots (gamma ceiling)
        self._composite = self.wm.encode_sequence(self._window)

    def held_referents(self):
        """Return the current ordered list of held referents (slot 0 .. most-recent). Diagnostic / introspection."""
        return list(self._window)

    def most_recent_referent(self):
        """Resolve the FOREGROUNDED referent = the item at the most-recent occupied slot, read on the spiking
        substrate (spiking unbind of the highest slot, familiarity-gated). Returns the referent word, or None if
        the discourse is empty / the read does not ground (the no-confab moat). The winner is decided by WHICH
        SLOT is read (the highest), so it flips deterministically when the discourse order changes."""
        if not self._window or self._composite is None:
            return None
        last_slot = len(self._window) - 1
        word, _match = self.wm.read_slot(self._composite, f"pos{last_slot}", gate=True)
        return word

    def referent_at(self, slot):
        """Read the referent at gamma-slot ``slot`` on the spiking substrate (familiarity-gated). None if the slot
        is unoccupied or does not ground. Lets a caller address any held referent by position, not just the most
        recent -- the capability the rate-attractor set structurally lacks."""
        if self._composite is None or not (0 <= slot < len(self._window)):
            return None
        return self.wm.read_slot(self._composite, f"pos{slot}", gate=True)[0]

    def _resolve(self, word):
        """If ``word`` is an anaphor, resolve it from the most-recent discourse referent (None if unresolved);
        else return ``word`` unchanged."""
        if isinstance(word, str) and word.lower() in _ANAPHORS:
            return self.most_recent_referent()
        return word

    # --- turns ---------------------------------------------------------------
    def hear(self, sentence, voice="active", polarity=None):
        """Comprehend + store a statement, and introduce its salient referents into the order-encoded discourse
        buffer. Both the agent (subject) and the patient (object) of the sentence are foregrounded, in surface
        order (agent then patient), so the MOST-RECENT discourse referent is the object -- the natural antecedent
        of a following pronoun -- while earlier-mentioned entities remain addressable at earlier slots."""
        roles = self.agent.hear(sentence, voice, polarity)
        # Introduce in surface order: subject first, object last -> object is most-recent (the usual antecedent),
        # but the subject is still held at an earlier slot (addressable, order-bearing).
        self._write_referent(roles.get("agent"))
        self._write_referent(roles.get("patient"))
        return roles

    def correct(self, sentence, voice="active"):
        """RECONSOLIDATION turn (the opt-in reconsolidation entry point; `hear` stays append-only and byte-identical).
        A corrective utterance -- 'actually <agent> <action> <new_patient>' -- reactivates the cued fact and updates
        its patient IN PLACE (no contradictory duplicate), gated by the prediction error: a re-statement re-stabilizes
        unchanged, and a NEVER-stored cue ABSTAINS (the no-confab moat -- update a reactivated trace, never fabricate a
        missing one). A leading correction marker (actually/no/wait) is optional; an anaphor agent ('actually it went
        south') resolves from the discourse buffer. Returns the composer's update result {action, wrote, pe}. De-risked
        6/6 multi-seed: research/findings/2026-06-17-reconsolidation-update-derisk-GO.md."""
        words = [w for w in sentence.split() if w]
        if words and words[0].lower() in _CORRECTION_MARKERS:
            words = words[1:]
        roles = self.agent.parse(words, voice)              # parser-agnostic: the agent's own parser OR the onebrain one
        agent = self._resolve(roles["agent"])               # resolve an agent pronoun from the discourse buffer
        if agent is None:
            return {"action": "abstain", "wrote": False, "pe": None, "reason": "unresolved_pronoun"}
        res = self.agent.composer.update_on_mismatch(agent, roles["action"], roles["patient"])
        if res.get("wrote"):                                 # foreground the corrected fact's referents (like hear)
            self._write_referent(agent)
            self._write_referent(roles["patient"])
        return res

    def what_does(self, agent_word, action):
        """'what does <agent|it> <action>?' -> patient or None. Resolves a pronoun agent from the most-recent
        discourse referent (abstains if the pronoun is unresolved)."""
        a = self._resolve(agent_word)
        return self.agent.what_does(a, action) if a is not None else None

    def who_does(self, action, patient_word):
        """'who <action> <patient|it>?' -> agent or None. Resolves a pronoun patient from the discourse."""
        p = self._resolve(patient_word)
        return self.agent.who_does(action, p) if p is not None else None

    def is_it_true(self, agent_word, action, patient_word):
        """'does <agent|it> <action> <patient|it>?' -> 'yes'/'no'/'unknown'. 'unknown' if a pronoun is unresolved
        or no fact matches."""
        a, p = self._resolve(agent_word), self._resolve(patient_word)
        if a is None or p is None:
            return "unknown"
        return self.agent.is_it_true(a, action, p)

    def reason_chain(self, cue_word, actions):
        """Multi-hop reasoning from a cue that may be a pronoun resolved from the discourse. The chain's
        intermediate concepts are introduced into the SAME order-encoded buffer as they are produced (the chain's
        working state is then genuinely on the spiking substrate, addressable by slot). Returns the terminal
        concept, or None (abstain at any hop / unresolved cue)."""
        cue = self._resolve(cue_word)
        if cue is None:
            return None
        x = cue
        for act in actions:
            x = self.agent.composer.query_patient(x, act)
            if x is None:
                return None
            self._write_referent(x)        # carry the hop's intermediate in the order-encoded buffer
        return x

    def describe(self, agent_word):
        """'describe <agent|it>' -> a generated sentence about the (possibly pronoun-resolved) subject, or None."""
        a = self._resolve(agent_word)
        return self.agent.describe(a) if a is not None else None

    # --- multi-sentence narration (ordered emission + cross-sentence coherence) -----------------
    def narrate(self, topics, return_details=False):
        """Produce a COHERENT MULTI-SENTENCE narration of an ordered list of topics, on the spiking substrate.

        This composes two separately-validated, multi-seed-GO mechanisms (NO new mechanism; reuse-by-import):
          * ORDERED EMISSION (2026-06-17-multisentence-ordered-emission-derisk.md, 6/6): hold the topics in the
            order-encoded WM (each topic bound to a successive gamma-slot POSITION phasor on the resonate-and-fire
            substrate), then emit one sentence per slot IN SLOT ORDER -- so re-ordering `topics` re-orders the
            output (the order is order-encoded, not a fixed storage order).
          * CROSS-SENTENCE COHERENCE (2026-06-17-cross-sentence-coherence-derisk.md, 6/6): when a topic RECURS as
            a later subject, render it as a PRONOUN ("it") and RESOLVE the pronoun (validated by-slot slot-anaphora,
            `referent_at(antecedent_slot)`) back to the correct ANTECEDENT referent -- the antecedent = the EARLIEST
            slot that referent was introduced at, NOT the most-recent slot.

        For each topic (a subject the agent has a stored fact about), in order: on the FIRST mention, render the
        full-noun sentence via the validated `describe` path (neural word order when `enable_neural_render=True`);
        on a RECURRENCE, emit a pronominalized sentence whose pronoun resolves on the substrate to the antecedent.
        Each sentence's object is then introduced into the same order-encoded WM, so the buffer holds the full
        surface-order referent stream (and the antecedent slots stay addressable).

        The no-confab moat holds: a topic with NO stored fact -> the slot ABSTAINS (no sentence, no confabulation);
        the slot is skipped in the surface string. (A topic that the WM read does not even ground -- the familiarity
        gate -- is also skipped.)

        Args:
            topics: an ordered list of topic words (subjects the agent may have stored facts about). A topic that
                is not a referent the buffer can hold is skipped.
            return_details: if True, also return the per-sentence structured detail list (subject, whether it was
                pronominalized, the antecedent slot, the substrate-resolved antecedent, and the rendered text) --
                used by the test/control harness and for transcripts. Default False -> returns just the joined
                surface string.

        Returns:
            the coherent multi-sentence surface string (e.g. "dog ran north. bird ate worm. then it ran north.");
            empty string if no topic produced a sentence. If `return_details=True`, returns
            ``(surface_string, sentences)`` where ``sentences`` is the per-sentence detail list.

        Existing MultiTurnAgentV2 capabilities (multi-referent resolution, single-referent anaphora, the Q&A /
        reason-chain paths) are untouched; narrate() uses a FRESH discourse buffer per call (it saves + restores the
        agent's standing discourse window/composite), so a narration does not perturb an in-progress dialogue."""
        narration = _CoherentNarration(self)
        sentences = narration.emit(list(topics))
        surface = _join_sentences(sentences)
        if return_details:
            return surface, sentences
        return surface


def _join_sentences(sentences):
    """Join the per-sentence detail dicts into the surface narration string, skipping abstained (None-text) slots.
    Each rendered sentence is terminated with a period; an all-abstain narration yields the empty string."""
    texts = [d["text"] for d in sentences if d.get("text")]
    if not texts:
        return ""
    return ". ".join(texts) + "."


class _CoherentNarration:
    """The production cross-sentence-coherence loop, lifted verbatim from the validated de-risk
    (`research/runners/_phaseB_cross_sentence_coherence_derisk.py`, class `CoherentDiscourse`, GO 6/6).

    It drives a MultiTurnAgentV2's order-encoded discourse buffer (`agent._window` / `agent._composite` /
    `agent.wm`) on the spiking RF substrate: accumulate referents in surface order as each topic is processed,
    tracking `_slot_of[referent]` = the EARLIEST gamma-slot each referent occupied (its ANTECEDENT slot). A
    recurring subject is pronominalized and resolved by reading its antecedent slot
    (`agent.referent_at(antecedent_slot)`, a familiarity-gated spiking unbind); a first mention is the validated
    full-noun `describe` path. A topic with no stored fact ABSTAINS (no sentence). NO new mechanism; reuse only.

    narrate() uses a FRESH buffer per call: the agent's standing discourse window/composite are SAVED on entry and
    RESTORED on exit, so a narration is side-effect-free with respect to an in-progress multi-turn dialogue.
    """

    def __init__(self, agent):
        self.agent = agent
        # Save the agent's standing discourse state so narration is side-effect-free for in-progress dialogue.
        self._saved_window = list(agent._window)
        self._saved_composite = agent._composite
        self._reset_discourse()

    def _reset_discourse(self):
        """Start a fresh discourse buffer for this narration: empty WM window + antecedent-slot bookkeeping."""
        self.agent._window = []
        self.agent._composite = None
        self._slot_of = {}                 # referent -> earliest gamma-slot it occupied (its antecedent slot)

    def _restore(self):
        """Restore the agent's standing discourse state (called after emission); narration leaves no trace."""
        self.agent._window = self._saved_window
        self.agent._composite = self._saved_composite

    def _introduce(self, referent):
        """Append a referent to the order-encoded discourse buffer (re-encoding the position-binding composite on
        the RF substrate), recording its EARLIEST slot. Mirrors MultiTurnAgentV2._write_referent (same spiking
        encode) but also tracks the antecedent slot so a later recurrence can be resolved BY that slot. Non-referent
        / over-capacity inputs are ignored (the WM holds at most n_slots; an over-cap referent is not bound)."""
        if not (isinstance(referent, str) and referent in self.agent.referents):
            return
        if len(self.agent._window) >= self.agent.wm.n_slots:
            return                          # gamma-slot ceiling: do not exceed the ordered-WM capacity
        slot = len(self.agent._window)      # the slot this referent will occupy (pre-append window length)
        self.agent._window.append(referent)
        if referent not in self._slot_of:
            self._slot_of[referent] = slot
        self.agent._composite = self.agent.wm.encode_sequence(self.agent._window)

    def _fact_for(self, subject):
        """The agent's stored (subject, verb, object) fact for `subject`, or None if the agent knows no fact about
        it. Read from the composer's own flat fact memory (the validated store) -- this is the no-confab probe:
        None => the topic abstains (no sentence)."""
        for fact, _ in self.agent.agent.composer.kb:
            if fact.get("agent") == subject and isinstance(fact.get("patient"), str):
                return (subject, fact.get("action"), fact.get("patient"))
        return None

    def emit(self, topics):
        """Emit one sentence per topic IN ORDER (the validated coherence loop). For each topic's fact (s, v, o):
        if `s` was introduced at an earlier slot (recurs), emit a PRONOUN and RESOLVE it via the antecedent slot on
        the spiking substrate; else render the full-noun sentence (validated `describe`). The object is introduced
        into the order-encoded buffer after the sentence. A topic with NO stored fact ABSTAINS (text=None). Returns
        the per-sentence detail list. Always restores the agent's standing discourse state on exit."""
        try:
            self._reset_discourse()
            out = []
            for topic in topics:
                if not isinstance(topic, str):
                    continue
                fact = self._fact_for(topic)
                if fact is None:
                    # No-confab moat: a topic the agent has no fact about -> abstain (no sentence, no confabulation).
                    out.append({"subject": topic, "pronominalized": False, "antecedent_slot": None,
                                "resolved_antecedent": None, "true_antecedent": None,
                                "resolved_correct": None, "abstained": True, "text": None})
                    continue
                (subj, verb, obj) = fact
                recurs = subj in self._slot_of                 # already introduced at an earlier slot?
                if recurs:
                    antecedent_slot = self._slot_of[subj]
                    # RESOLVE the pronoun on the spiking substrate: read the antecedent's gamma slot (familiarity-
                    # gated spiking unbind). This is the validated MultiTurnAgentV2 by-slot resolution.
                    resolved = self.agent.referent_at(antecedent_slot)
                    text = f"then {_NARRATE_PRONOUN} {verb} {obj}"   # the pronominalized, coherent sentence
                    out.append({"subject": subj, "pronominalized": True, "antecedent_slot": antecedent_slot,
                                "resolved_antecedent": resolved, "true_antecedent": subj,
                                "resolved_correct": (resolved == subj), "abstained": False, "text": text})
                else:
                    # First mention -> full noun, rendered by the validated single-sentence describe path.
                    sentence = self.agent.agent.describe(subj)
                    out.append({"subject": subj, "pronominalized": False, "antecedent_slot": None,
                                "resolved_antecedent": None, "true_antecedent": None,
                                "resolved_correct": None, "abstained": (sentence is None), "text": sentence})
                    self._introduce(subj)                      # introduce the subject AFTER its full-noun mention
                # The object is part of the surface discourse stream (held at a slot), introduced after the sentence.
                self._introduce(obj)
            return out
        finally:
            self._restore()
