"""MultiTurnAgent — the production multi-turn conversational agent: a `BrainConversationalAgent` plus a PERSISTENT
spiking working-memory loop that holds discourse referents across turns. This unites the two de-risked pieces:

  * multi-hop reasoning  (2026-06-17-multihop-query-chain-GO.md, now production: composer.query_chain)
  * multi-turn anaphora  (2026-06-17-multiturn-anaphora-derisk-GO.md: a persistent SpikingLoopContextBuffer
                          carries the salient referent across a turn boundary so 'it' resolves)

so that a pronoun in a later turn resolves to the held referent, AND a multi-hop chain's intermediate concept is
carried in the SAME working-memory loop (the chain's working state is then genuinely neural, not a Python
variable). The no-confabulation moat is preserved everywhere: an unresolved pronoun (empty / ambiguous WM) yields
None, and every fact query abstains when no fact matches.

Reuse-by-import; NO `sim/` edit. The WM loop, the composer, the parser are all already validated.
"""
from __future__ import annotations

import numpy as np

from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.runners.content_selection_spiking import SpikingLoopContextBuffer

_ANAPHORS = {"it", "that", "them", "they", "this"}


class MultiTurnAgent:
    """Multi-turn dialogue on the validated substrate.

    `referent_concepts` = the nouns the working-memory loop can hold (it installs one attractor per concept).
    `concepts` = the full composer vocabulary (nouns + actions); `grounded_codes` optionally supplies learned
    phasor codes (e.g. the 320 stream-learned cortex codes). Everything else mirrors BrainConversationalAgent."""

    def __init__(self, referent_concepts, concepts=None, grounded_codes=None, seed=42,
                 wm_n=600, wm_pattern_size=40, enable_neural_render=False, spec_threshold=1.5):
        self.seed = int(seed)
        self.agent = BrainConversationalAgent(seed=seed, concepts=concepts, grounded_codes=grounded_codes,
                                              enable_neural_render=enable_neural_render)
        self.referents = list(referent_concepts)
        self.wm = SpikingLoopContextBuffer(self.referents, n=wm_n, pattern_size=wm_pattern_size,
                                           seed=seed, enable_ou=False)
        self._spec = float(spec_threshold)

    # --- discourse state -----------------------------------------------------
    def _write_referent(self, ref):
        """Write a salient referent into the persistent WM loop (held by its attractor across turns)."""
        if isinstance(ref, str) and ref in self.referents:
            self.wm.update([ref])

    def held_referent(self, window=20):
        """Read the WM loop; return (referent, specificity). The referent is the concept whose attractor dominates
        the read by > spec_threshold; otherwise None (ambiguous / empty WM -> no antecedent)."""
        rates = self.wm.read(window=window)
        items = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)
        if not items or items[0][1] <= 1e-6:
            return None, 0.0
        top, top_r = items[0]
        rest = float(np.mean([r for _, r in items[1:]])) if len(items) > 1 else 0.0
        spec = top_r / (rest + 1e-9)
        return (top if spec > self._spec else None), spec

    def _resolve(self, word):
        """If `word` is an anaphor, resolve it from the held WM referent (None if unresolved); else return `word`."""
        if isinstance(word, str) and word.lower() in _ANAPHORS:
            return self.held_referent()[0]
        return word

    # --- turns ---------------------------------------------------------------
    def hear(self, sentence, voice="active", polarity=None):
        """Comprehend + store a statement, and write its salient referent (the object/patient) into the WM."""
        roles = self.agent.hear(sentence, voice, polarity)
        self._write_referent(roles.get("patient"))
        return roles

    def what_does(self, agent_word, action):
        """'what does <agent|it> <action>?' -> patient or None. Resolves a pronoun agent from the held referent."""
        a = self._resolve(agent_word)
        return self.agent.what_does(a, action) if a is not None else None

    def who_does(self, action, patient_word):
        p = self._resolve(patient_word)
        return self.agent.who_does(action, p) if p is not None else None

    def is_it_true(self, agent_word, action, patient_word):
        a, p = self._resolve(agent_word), self._resolve(patient_word)
        if a is None or p is None:
            return "unknown"
        return self.agent.is_it_true(a, action, p)

    def reason_chain(self, cue_word, actions):
        """Multi-hop reasoning from a cue that may be a pronoun resolved from the WM. The intermediate concepts of
        the chain are written into the SAME persistent loop as they are produced (the chain's working state is
        neural). Returns the terminal concept or None (abstain at any hop)."""
        cue = self._resolve(cue_word)
        if cue is None:
            return None
        x = cue
        for act in actions:
            x = self.agent.composer.query_patient(x, act)
            if x is None:
                return None
            self._write_referent(x)        # carry the hop's intermediate in the WM loop
        return x

    def describe(self, agent_word):
        a = self._resolve(agent_word)
        return self.agent.describe(a) if a is not None else None
