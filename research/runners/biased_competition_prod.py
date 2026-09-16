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


# ---------------------------------------------------------------------------------------------------------------
# GAP #3 RESIDUAL A1 RETIREMENT (2026-09-16 wire-in de-risk) — the referent-bias FEATURE-COMPATIBILITY chooser.
#
# `biased_competition_enabled()` above gates whether the WTA itself is built at all. Independently of that, once the
# WTA IS active, `MultiTurnAgent._resolve_biased` still asks the HOST `content_bias_target` lexicon
# (`biased_competition_buffer.ANIMACY` / `VERB_SELECTS`) which held referent gets the content bias current — a host
# lookup, not a brain-based decision (BRAIN-BASED-ONLY standard, CLAUDE.md). An already-6-seed-GO, already-CI-pinned
# (`tests/test_gap3_spiking_feature_compat.py`) spiking replacement exists —
# `research/runners/_gap3_spiking_feature_compat_derisk.SpikingFeatureCompat`, exposed as
# `MultiTurnAgent(feat_compat_source=...)` — plus a DEPLOYMENT helper,
# `MultiTurnAgent.build_referent_bias_from_experience()`, that LEARNS the concept-animacy / verb-selection
# compatibility map from the agent's OWN heard SVO facts (its conversational experience), then installs the
# resulting `SpikingFeatureCompat` as `_feat_compat_source`. Finding:
# `research/findings/2026-07-18-gap3-A1-learned-feature-compatibility-cheap-first-GO.md`.
#
# THIS FLAG (default OFF — this wire-in's own de-risk scope, NOT yet flipped default-on): every live build site
# constructs the agent, teaches/restores its facts, then calls `maybe_install_learned_referent_bias(agent)` below.
# Flag OFF (unset, or any FALSY spelling) -> the call is a byte-identical no-op (`learned_bias_enabled()` returns
# False -> the helper returns False immediately without touching `agent` at all) -> `_feat_compat_source` stays
# `None` -> `_resolve_biased` keeps asking the host `content_bias_target` lexicon, EXACTLY as before this flag
# existed. Flag ON -> the helper calls `agent.build_referent_bias_from_experience()`, which itself no-ops (returns
# False, host lexicon keeps answering) unless the agent has heard >= `min_facts` SVO facts — so even ON, a
# freshly-built agent with few facts is unaffected until it has accumulated enough conversational experience.
#
#     BRAIN_BIASED_COMPETITION_LEARNED_BIAS   unset / "0"/"false"/"off"/"no"/""   -> OFF (default; host lexicon)
#                                              "1"/"true"/"on"/"yes"              -> ON  (learn from own experience)
#
# Independent of `BRAIN_BIASED_COMPETITION`: the learned-bias chooser only matters once the WTA itself is active
# (>= 2 held referents), same as the host lexicon it replaces — turning this ON with the WTA OFF is inert.
BRAIN_BIASED_COMPETITION_LEARNED_BIAS_ENV = "BRAIN_BIASED_COMPETITION_LEARNED_BIAS"

_LEARNED_BIAS_DEFAULT_ON = False  # default OFF -- the flip+delete of the host lexicon is a reviewed follow-on


def learned_bias_enabled(env=None) -> bool:
    """Return True iff the LEARNED spiking feature-compatibility referent-bias chooser (gap #3 residual A1) should
    be installed in place of the host `content_bias_target` lexicon lookup. Default OFF: unset or any of
    {0,false,off,no,''} -> False (byte-identical to before this flag existed); any other value -> True. Mirrors
    `biased_competition_enabled`'s spelling rules but with the OPPOSITE (OFF) default anchor -- this capability has
    not yet earned its own production flip."""
    src = os.environ if env is None else env
    raw = src.get(BRAIN_BIASED_COMPETITION_LEARNED_BIAS_ENV)
    if _LEARNED_BIAS_DEFAULT_ON:
        if raw is None:
            return True
        return str(raw).strip().lower() not in _FALSY
    return str(raw if raw is not None else "").strip().lower() in _TRUTHY


def maybe_install_learned_referent_bias(agent, min_facts=40, seed=None, env=None) -> bool:
    """If `learned_bias_enabled()`, try to LEARN the referent-bias feature-compatibility from `agent`'s own heard
    facts and install it as `agent._feat_compat_source` (gap #3 residual A1 deployment), replacing the host
    `content_bias_target` lexicon lookup for THIS agent's pronoun resolution. Call once, AFTER the agent's facts have
    been taught/restored (so `agent.heard_facts()` sees them), at every production build site.

    Returns True iff the learned chooser was installed. Every non-installing path is a documented no-op that leaves
    the host lexicon (or whatever `feat_compat_source` the caller already passed) answering exactly as before:
      - flag OFF (default)                                   -> returns False, `agent` untouched.
      - `agent` has no `build_referent_bias_from_experience`  -> returns False (e.g. a plain `BrainConversationalAgent`
        build site, which some callers use when `use_multiturn=False`; this helper is safe to call unconditionally).
      - fewer than `min_facts` heard SVO facts                -> `build_referent_bias_from_experience` itself
        returns False (its own documented floor); this helper propagates that.
      - any exception while learning (a malformed/degenerate heard-fact corpus) -> caught, returns False, so a
        production build never crashes because the learned-bias experiment misbehaves on real conversational data.
    """
    if not learned_bias_enabled(env):
        return False
    fn = getattr(agent, "build_referent_bias_from_experience", None)
    if fn is None:
        return False
    try:
        return bool(fn(min_facts=min_facts, seed=seed))
    except Exception:
        return False
