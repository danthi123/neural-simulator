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
                 wm_n_slots=7, enable_neural_render=False):
        self.seed = int(seed)
        self.agent = BrainConversationalAgent(seed=seed, concepts=concepts, grounded_codes=grounded_codes,
                                              enable_neural_render=enable_neural_render)
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
