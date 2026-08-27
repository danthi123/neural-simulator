"""BRAIN_SPIKING_MOUTH_RECALL production organ -- route the GROUNDED bounded-SVO RECALL / RICH-answer surface
through the spiking BROCA word-order render (EMERGE-59/61: per-pool spiking-RATE ranking on a real Izhikevich
SimulationBridge) instead of the Qwen/template surface, for the bounded transitive-SVO frame inventory ONLY.

WHAT THIS EXTENDS. The GENERATE channel already renders a structured (hedged-transitive) HYPOTHESIS on the
spiking Broca by default (brain_chat_tui.render_hypothesis_verified / _render_hypothesis_spiking, flag
BRAIN_SPIKING_MOUTH default ON, EMERGE-59/61 6-seed GO). This organ carries the SAME spiking mouth to the
ASSERTED RECALL / RICH answer surface (the `PLAIN_TRANSITIVE` frame "the <S> <V-3sg> the <O>"), which today is
authored by the off-bridge Qwen / template-stub. It BURNS DOWN the recall half of the mouth scaffold for the
bounded transitive-SVO case; open prose the frames cannot cover keeps the Qwen fallback (owner-sanctioned).

FLAG (additive; 2026-08-26 FLIPPED DEFAULT-ON, wave 3, 6/6 flip-soak GO):

    BRAIN_SPIKING_MOUTH_RECALL  -- unset => ON (bounded transitive-SVO recall frames render on the spiking Broca,
                                   verify-gated); an explicit off (0/false/off/no/'') => OFF (recall surface = the
                                   exact pre-flip Qwen/template path, byte-identical). The pre-existing generate-channel
                                   BRAIN_SPIKING_MOUTH (default ON) is UNTOUCHED.

    The master mouth kill-switch BRAIN_SPIKING_MOUTH=0 ALSO silences this recall surface (so the literal lesion
    oracle "BRAIN_SPIKING_MOUTH lesion -> the recall SVO surface reverts to Qwen/template" holds directly).

SCOPE-GUARD. Only a bounded transitive SVO (single-word alphabetic roles, subject != object, non-copula verb)
whose spiking render RE-PARSE VERIFIES to the exact recalled (a, v, p) is ever routed here; every other fact
(open/multi-word prose, copula, a verify miss) falls straight through to the current mouth, byte-identical. So
the moat (recall CONTENT) never weakens: the surface either carries EXACTLY the recalled SVO or is not used.

REUSE-BY-IMPORT, NO sim/ edit. The producer + frame + wash-out are imported verbatim from
`_spiking_fluent_surface_derisk` (EMERGE-59/61); the re-parse VERIFY is the ChatBrain's own `_verify` (the same
independent grammar re-parse the recall + generate paths already use).

LOAD-BEARING (the lesion oracle, proven in `_spiking_mouth_recall_soak`):
  - flag lesion   : flag ON  -> surface authored on spikes ("the brain uses the spikes");
                    flag OFF  -> surface reverts to Qwen/template ("The brain uses spikes.") -- the word ORDER /
                    surface changes while the recalled CONTENT SVO is byte-identical.
  - rate-read lesion: intact per-pool spiking-RATE ranking -> correct slot ORDER; the EMERGE anti-cheats
                    (equal_drive: rates tie / permute_order: fixed wrong order) SCRAMBLE the ORDER with the same
                    content words -> proves the SPIKING READ authored the ORDER, not a fixed host template.
"""
from __future__ import annotations

import os

# copula / auxiliary verbs that do not fit the transitive "the S <V-3sg> the O" frame (would render as
# "the sky is the blue"); scope them OUT so they keep the current mouth. (A verify miss would also reject them,
# but excluding up front avoids a wasted spiking emit + keeps the frame inventory honest.)
_COPULA = frozenset({"is", "are", "was", "were", "be", "been", "being", "am", "'s", "s"})

# The DEFAULT-ON master switch (the production-integration anchor; 2026-08-26 flip, wave 3, 6/6 flip-soak GO).
# BRAIN_SPIKING_MOUTH_RECALL=0 disables the bounded transitive-SVO recall surface byte-identically (the row STAYS
# on_by_default:YES). Flipping this to False would turn the faculty OFF by default.
_RECALL_MOUTH_DEFAULT_ON = True


def recall_mouth_enabled():
    """The escape hatch / flip gate. Returns True IFF the bounded transitive-SVO recall surface should render on
    the spiking Broca. DEFAULT-ON (2026-08-26 flip, wave 3, 6/6 flip-soak GO): unset `BRAIN_SPIKING_MOUTH_RECALL`
    -> ON; an explicit off (0/false/off/no/'') -> the recall surface stays on the current Qwen/template path,
    byte-identical to pre-flip. The master generate-channel kill-switch BRAIN_SPIKING_MOUTH=0 also forces this OFF
    (the literal lesion oracle)."""
    if os.environ.get("BRAIN_SPIKING_MOUTH", "1") == "0":
        return False                                              # master mouth kill also silences recall
    v = os.environ.get("BRAIN_SPIKING_MOUTH_RECALL")
    if _RECALL_MOUTH_DEFAULT_ON:
        return not (v is not None and v.strip().lower() in ("0", "false", "off", "no", ""))
    return v is not None and v.strip().lower() not in ("0", "false", "off", "no", "")


def frame_supported(a, v, p):
    """True iff (a, v, p) fits the bounded transitive-SVO frame the spiking Broca renders: single-WORD alphabetic
    roles, subject != object, and a non-copula verb. Everything else (open/multi-word prose, copula/attribute
    facts) is NOT frameable here -> the caller keeps the current mouth (the documented open-prose residual)."""
    if not isinstance(a, str) or not isinstance(v, str) or not isinstance(p, str):
        return False
    if not (a.isalpha() and v.isalpha() and p.isalpha()):
        return False
    if a == p:
        return False
    if v.lower() in _COPULA:
        return False
    return True


class SpikingRecallMouth:
    """Lazily-built, cached spiking BROCA clause producer for the ASSERTED recall/rich surface. Reuses the
    EMERGE-59/61 `SpikingClauseProducer` learning the 5-slot `PLAIN_TRANSITIVE` order; each recall SVO then emits
    "the <S> <V-3sg> the <O>" in ~5 ms via the EMERGE-61 inter-utterance wash-out (the producer restores the
    post-init substrate state before every clause, so productions are position-independent).

    `mode`:
      - None          -> the production render (per-pool spiking-RATE ranking authors the slot ORDER);
      - "equal_drive" -> LESION: equal drive, rates tie, the ORDER read collapses to noise (anti-cheat);
      - "permute"     -> LESION: a fixed WRONG learned order (anti-cheat).
    The lesion modes exist only for the load-bearing proof (`_spiking_mouth_recall_soak`); production builds mode=None.
    """

    def __init__(self, seed=42, mode=None):
        self.seed = int(seed)
        self.mode = mode
        self._prod = None

    def _producer(self):
        if self._prod is None:
            from research.runners._spiking_fluent_surface_derisk import (
                SpikingClauseProducer, PLAIN_TRANSITIVE)
            kw = {}
            if self.mode == "equal_drive":
                kw["equal_drive"] = True
            elif self.mode == "permute":
                kw["permute_order"] = True
            prod = SpikingClauseProducer(self.seed, **kw)
            prod.learn(len(PLAIN_TRANSITIVE))          # competitive-queuing learn of the 5-slot transitive order
            self._prod = prod
        return self._prod

    def render(self, a, v, p):
        """Render the recall SVO grammatically ON FIRING NEURONS: order the `PLAIN_TRANSITIVE` slots by the per-pool
        spiking-RATE ranking on the real bridge, realize each slot ("the <S> <V-3sg> the <O>"). Returns the surface
        string. Does NOT verify -- the caller runs the independent re-parse VERIFY (the moat)."""
        from research.runners._spiking_fluent_surface_derisk import PLAIN_TRANSITIVE
        from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import emerge_v3
        dctx = {"subject": a, "verb_3sg": emerge_v3(v), "object": p}
        return " ".join(self._producer().emit(PLAIN_TRANSITIVE, dctx))

    @property
    def spiked(self):
        """True once any emit read a positive spiking rate (a genuinely-spiking assertion, not a silent default)."""
        return bool(self._prod is not None and getattr(self._prod, "spiked", False))
