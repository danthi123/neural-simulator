"""MultiTurnAgent — the production multi-turn conversational agent: a `BrainConversationalAgent` plus a PERSISTENT
spiking working-memory loop that holds discourse referents across turns. This unites the two de-risked pieces:

  * multi-hop reasoning  (2026-06-17-multihop-query-chain-GO.md, now production: composer.query_chain)
  * multi-turn anaphora  (2026-06-17-multiturn-anaphora-derisk-GO.md: a persistent SpikingLoopContextBuffer
                          carries the salient referent across a turn boundary so 'it' resolves)

so that a pronoun in a later turn resolves to the held referent, AND a multi-hop chain's intermediate concept is
carried in the SAME working-memory loop (the chain's working state is then genuinely neural, not a Python
variable). The no-confabulation moat is preserved everywhere: an unresolved pronoun (empty / ambiguous WM) yields
None, and every fact query abstains when no fact matches.

MULTI-REFERENT DISAMBIGUATION (opt-in, default OFF; de-risk GO 2026-06-19-multireferent-biased-competition-derisk.md):
the plain anaphora path (`held_referent`) reads the single dominant WM attractor, which cannot pick among >=2 held
referents (a tie -> abstain). With `enable_biased_competition=True`, a pronoun query over >=2 held referents routes
through a `BiasedCompetitionContextBuffer` (WTA mutual inhibition + a small CONTENT bias from the query verb's
selectional restriction + the candidate animacy) so the pronoun resolves to the content-favored referent — exactly
where recency and a salience boost both failed. Default OFF == byte-identical to the prior behavior (the
biased-competition buffer is never even constructed). See
`research/findings/2026-06-19-multireferent-integration-multiturnagent.md`.

Reuse-by-import; NO `sim/` edit. The WM loop, the composer, the parser, the biased-competition buffer are all
already validated.
"""
from __future__ import annotations

import numpy as np

from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.runners.biased_competition_buffer import (
    BiasedCompetitionContextBuffer,
    content_bias_target,
    resolve_referent,
)
from research.runners.content_selection_spiking import SpikingLoopContextBuffer

_ANAPHORS = {"it", "that", "them", "they", "this"}


class MultiTurnAgent:
    """Multi-turn dialogue on the validated substrate.

    `referent_concepts` = the nouns the working-memory loop can hold (it installs one attractor per concept).
    `concepts` = the full composer vocabulary (nouns + actions); `grounded_codes` optionally supplies learned
    phasor codes (e.g. the 320 stream-learned cortex codes). Everything else mirrors BrainConversationalAgent."""

    def __init__(self, referent_concepts, concepts=None, grounded_codes=None, seed=42,
                 wm_n=600, wm_pattern_size=40, enable_neural_render=False, spec_threshold=1.5,
                 composer_kind="rf", enable_biased_competition=False,
                 biased_competition_bias_pA=2500.0, biased_competition_spec_threshold=1.3,
                 biased_competition_window=20):
        self.seed = int(seed)
        # composer_kind passes through to the inner agent: "rf" (default) or "onebrain" (the integrated one-brain
        # composer -- the cleanup arc validates multi-turn anaphora + cued multi-hop on it).
        self.agent = BrainConversationalAgent(seed=seed, concepts=concepts, grounded_codes=grounded_codes,
                                              enable_neural_render=enable_neural_render, composer_kind=composer_kind)
        self.referents = list(referent_concepts)
        self.wm = SpikingLoopContextBuffer(self.referents, n=wm_n, pattern_size=wm_pattern_size,
                                           seed=seed, enable_ou=False)
        self._spec = float(spec_threshold)
        # Agent-owned discourse-referent registry (the plain SpikingLoopContextBuffer does NOT track which
        # referents were introduced; the biased-competition path needs the held SET to know when >=2 referents
        # are co-present). Appended in _write_referent; mirrors exactly what is written into the WM loop(s).
        self._referent_history = []

        # --- multi-referent biased-competition (opt-in, default OFF) -------------------------------------------
        self.enable_biased_competition = bool(enable_biased_competition)
        self._bc_bias_pA = float(biased_competition_bias_pA)
        self._bc_spec = float(biased_competition_spec_threshold)
        self._bc_window = int(biased_competition_window)
        self._wm_n = int(wm_n)
        self._wm_pattern_size = int(wm_pattern_size)
        # The biased-competition buffer mirrors the held discourse-referent registry; it is built ONLY when the
        # flag is ON (default OFF -> never constructed -> byte-identical to the prior behavior). It holds the same
        # attractor per referent (same seed, same n/pattern_size) PLUS the per-referent WTA accumulator + selective
        # inhibition that the plain SpikingLoopContextBuffer lacks.
        self.bcw = None
        if self.enable_biased_competition:
            self.bcw = BiasedCompetitionContextBuffer(
                self.referents, n=self._wm_n, pattern_size=self._wm_pattern_size,
                seed=seed, enable_ou=False, competition=True)

    # --- discourse state -----------------------------------------------------
    def _write_referent(self, ref):
        """Write a salient referent into the persistent WM loop (held by its attractor across turns). When biased
        competition is enabled, mirror the write into the biased-competition buffer's registry so the same held
        referents compete during a pronoun query."""
        if isinstance(ref, str) and ref in self.referents:
            self.wm.update([ref])
            self._referent_history.append(ref)
            if self.bcw is not None:
                self.bcw.update([ref])

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

    def _held_set(self):
        """The set of referents introduced into the discourse (deduplicated). The agent-owned registry is the
        source of truth (the plain WM buffer mirrors the same writes but does not expose a registry)."""
        return sorted(set(self._referent_history))

    def _resolve_biased(self, query_verb):
        """Resolve a pronoun over the held referents via WTA biased competition steered by the query verb's
        content (selectional restriction x candidate animacy). Returns the resolved referent or None (abstain):
          - empty / single held referent  -> defer to the plain held_referent (nothing to arbitrate);
          - content is silent (verb has no selectional restriction, or 0 / >1 compatible candidates) -> abstain
            (the no-confab moat: refuse to pick by intrinsic strength);
          - else run the biased-competition read (re-present the held referents + bias the content-favored sel
            pool) and return the moat-gated WTA winner.
        This is the de-risked decision (resolve_pronoun in the de-risk runner), here driven by the live registry."""
        held = self._held_set()
        if len(held) < 2 or self.bcw is None:
            return None  # <2 held -> let the plain single-attractor path decide (no competition needed)
        fav = content_bias_target(held, query_verb)
        if fav is None:
            return None  # content silent -> abstain (moat)
        rates = self.bcw.read(window=self._bc_window, bias_concept=fav, bias_pA=self._bc_bias_pA)
        return resolve_referent(rates, spec_threshold=self._bc_spec)

    def _resolve(self, word, query_verb=None):
        """If `word` is an anaphor, resolve it from the held WM referent (None if unresolved); else return `word`.

        When biased competition is enabled AND a query verb is available AND >=2 referents are held, route the
        resolution through the WTA biased competition (content-steered) instead of the single-attractor read.
        Default (flag OFF, or no verb, or <2 held) -> the plain held_referent path, byte-identical to before."""
        if not (isinstance(word, str) and word.lower() in _ANAPHORS):
            return word
        if self.enable_biased_competition and query_verb is not None and len(self._held_set()) >= 2:
            return self._resolve_biased(query_verb)
        return self.held_referent()[0]

    # --- turns ---------------------------------------------------------------
    def hear(self, sentence, voice="active", polarity=None):
        """Comprehend + store a statement, and write its salient referent (the object/patient) into the WM."""
        roles = self.agent.hear(sentence, voice, polarity)
        self._write_referent(roles.get("patient"))
        return roles

    def what_does(self, agent_word, action):
        """'what does <agent|it> <action>?' -> patient or None. Resolves a pronoun agent from the held referent;
        the query verb (`action`) steers the biased competition when that mode is enabled + >=2 referents held."""
        a = self._resolve(agent_word, query_verb=action)
        return self.agent.what_does(a, action) if a is not None else None

    def who_does(self, action, patient_word):
        p = self._resolve(patient_word, query_verb=action)
        return self.agent.who_does(action, p) if p is not None else None

    def is_it_true(self, agent_word, action, patient_word):
        a, p = self._resolve(agent_word, query_verb=action), self._resolve(patient_word, query_verb=action)
        if a is None or p is None:
            return "unknown"
        return self.agent.is_it_true(a, action, p)

    def reason_chain(self, cue_word, actions):
        """Multi-hop reasoning from a cue that may be a pronoun resolved from the WM. The intermediate concepts of
        the chain are written into the SAME persistent loop as they are produced (the chain's working state is
        neural). Returns the terminal concept or None (abstain at any hop). The first hop's verb steers the biased
        competition when that mode is enabled + >=2 referents held."""
        cue = self._resolve(cue_word, query_verb=(actions[0] if actions else None))
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
