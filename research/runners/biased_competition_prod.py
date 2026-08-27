"""Production wire-in ORGAN for SELECTIVE ATTENTION — biased competition (Wong-Wang / Desimone-Duncan lateral
inhibition between held discourse-referent attractors).

WHAT THIS DOES. The validated faculty already exists as an organ — `BiasedCompetitionContextBuffer`
(`research/runners/biased_competition_buffer.py`) — and is already wired into `MultiTurnAgent` behind its
`enable_biased_competition` constructor flag. But EVERY live build site (the console TUI's self-knowledge /
tiny-demo brains, the rich-answer smoke, and the WEBAPP production loader `developed_brain_io.load_developed_brain`)
hard-codes `enable_biased_competition=False`, so the faculty is dark in production. This module is the single
place that decides — from a NEW env flag — whether those live sites turn the faculty ON. It adds NO new mechanism;
it only routes the existing, de-risk-validated organ into the live pipeline behind one gate.

THE FLAG (2026-08-26 FLIPPED DEFAULT-ON — wave 1/2, 6-seed pool soak GO: NO_REGRESSION 6/6, FACULTY_LIVE 2/6 expected
per the finding's seed-100 extreme-intrinsic-asymmetry abstain-under-both-arms case):
    BRAIN_BIASED_COMPETITION   unset                                      -> ON  (the production default)
                               "0" / "false" / "off" / "no" / ""          -> OFF (byte-identical escape to pre-flip)
                               "1" / "true" / "on" / "yes"                -> ON  (explicit, redundant now)

BYTE-IDENTITY GUARANTEE (flag OFF == today). Each wired build site substitutes the literal `False` it holds today
with `biased_competition_enabled()`. When the flag is unset this call returns exactly `False`, i.e. the SAME value
the code passes today, so `MultiTurnAgent.enable_biased_competition is False`, the biased-competition buffer is
never constructed (`self.bcw is None`), and `_write_referent` / `_resolve` follow the plain single-attractor
anaphora path unchanged. Byte-identity here is therefore STRUCTURAL (identical argument value), not merely
empirical — and it is additionally proven empirically by the wire-in verifier + the 6-seed soak runner.

WHAT CHANGES WHEN ON (the faculty, load-bearing). With the flag ON, a bare pronoun over >=2 held discourse
referents ("the cat and the ball ... it ...") routes through the WTA biased competition: mutual inhibition between
the referent assemblies + a small CONTENT bias from the query verb's selectional restriction resolves the pronoun
to the SALIENT / content-favored referent (e.g. "what does it eat?" -> the animate cat; "where does it roll?" ->
the inanimate ball). UNIQUE-referent turns (< 2 held) never enter the biased path -> byte-identical to OFF; only
MULTI-referent turns change. The no-confab moat is preserved (empty WM / content-silent verb -> abstain).

LESION ORACLE (the coupling is load-bearing iff lesioning it makes the content-tracking VANISH). The de-risk's own
bias-lesion: zero the content bias current injected into the winning sel pool
(`MultiTurnAgent(..., biased_competition_bias_pA=0.0)`) -> the WTA reverts to the SEED-DEPENDENT intrinsic
attractor, so the verb no longer steers the winner (both verbs collapse to the same intrinsic referent, or
abstain). A lesion that changed nothing would be a FAIL; the wire-in verifier asserts the difference vanishes.

De-risk GO (the mechanism this routes): `research/findings/2026-06-19-multireferent-biased-competition-derisk.md`
(GO-arm 5/6 seeds, all anti-cheat controls 6/6, on the spiking `SimulationBridge`). Integration into MultiTurnAgent:
`research/findings/2026-06-19-multireferent-integration-multiturnagent.md`. CI guard:
`tests/test_multireferent_biased_competition.py`.

Reuse-by-import; NO `sim/` edit; the organ + its lesion oracle are unchanged.
"""
from __future__ import annotations

import os

#: The environment flag that gates selective-attention biased competition at the live build sites.
BRAIN_BIASED_COMPETITION_ENV = "BRAIN_BIASED_COMPETITION"

#: Truthy spellings (case-insensitive, whitespace-stripped). Anything else -> OFF.
_TRUTHY = frozenset({"1", "true", "on", "yes"})
#: Explicit-OFF spellings (case-insensitive, whitespace-stripped) for the default-ON anchor's escape.
_FALSY = frozenset({"0", "false", "off", "no", ""})

# 2026-08-26 FLIPPED DEFAULT-ON (wave 1/2 flip, 6-seed pool soak GO: NO_REGRESSION 6/6 — the flip-safety gate;
# FACULTY_LIVE 2/6 is EXPECTED and does not block, per the finding: a seed with extreme intrinsic asymmetry abstains
# under both OFF and ON, moat-preserving). The production-integration anchor.
_BIASED_COMPETITION_DEFAULT_ON = True


def biased_competition_enabled(env=None) -> bool:
    """Return True iff selective-attention biased competition is armed at the live build sites.

    Default-ON anchor (current, post-flip): unset -> ``_BIASED_COMPETITION_DEFAULT_ON`` (``True``);
    ``BRAIN_BIASED_COMPETITION`` in {0,false,off,no,''} (explicitly set) is the byte-identical escape back to the
    pre-flip OFF oracle; any other explicit value (1/true/on/yes/anything-else) stays ON. This mirrors the
    _SWAP_DRIVES_DEFAULT_ON / _AFFECTIVE_TOM_DEFAULT_ON convention in webapp/server.py.

    ``env`` defaults to ``os.environ``; an explicit mapping is accepted so a test/soak can toggle the flag without
    mutating the process environment.
    """
    src = os.environ if env is None else env
    raw = src.get(BRAIN_BIASED_COMPETITION_ENV)
    if _BIASED_COMPETITION_DEFAULT_ON:
        if raw is None:
            return True
        return str(raw).strip().lower() not in _FALSY
    return str(raw if raw is not None else "").strip().lower() in _TRUTHY
