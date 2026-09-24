"""Production wire-in of the #129 source-provenance opponent monitor (board #129, Vikunja #137).

THE FACULTY: did the brain SEE a fact (PERCEIVED -- directly taught) or did it INFER/COMPOSE it itself
(GENERATED -- produced by the brain's own multi-hop reasoning, never itself a single stored fact)? This is
reality monitoring (Johnson-Hashtroudi-Lindsay 1993); its failure is confabulation -- misattributing an
inferred/imagined claim to direct experience.

This module is a THIN, ADDITIVE, DEFAULT-OFF wrapper around the validated 6-seed GO mechanism in
`_laneC_source_provenance_opponent_derisk.ProvenanceBrain` -- REUSED BY IMPORT, not re-derived, so the
production wire-in carries the SAME learned, context-gated opponent-comparator substrate the de-risk verified
(research/findings/2026-08-25-laneC-source-provenance-opponent-perceived-vs-generated-6seed-GO.md): two
neuromodulatory encoding-context lines (ctx_perceived / ctx_generated) each gate a separate zero-init Hebbian
episode->provenance trace; at recall the contexts are silent and the content cue alone drives the learned
trace; the read-out is the SIGN of an opponent (mutually-inhibiting) comparator, reported as a normalized
discriminability d = (r_true - r_false) / (r_true + r_false).

What this module ADDS on top of the de-risk (which only ever judged its own 4 fixed calibration pairs):
  - a stable content-pattern generator keyed on an arbitrary hashable "fact key" (a production fact does not
    arrive pre-registered as one of the de-risk's calibration pairs);
  - encode-once-per-key idempotence (an episodic trace is written the first time a fact is EXPERIENCED, not
    re-written on every later recall of the same key -- exactly how a real episodic memory works);
  - a judge() -> label API a conversational agent can call at answer time, which NEVER fabricates a label for
    a key it was never shown (anti-cheat (3) of the de-risk: "a never-encoded pattern must leave both prov
    pools ~silent");
  - honesty-framing text helpers that turn a judged GENERATED claim into a flagged sentence, while a judged
    PERCEIVED claim (and an unjudged/unknown claim) renders exactly as the pipeline already renders it today.

Everything here is OFF unless a caller explicitly builds a `SourceProvenanceHonestyMonitor` and calls
`encode_fact`/`judge_fact`; importing this module builds no substrate and runs no simulation step.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Mapping

import numpy as np

from research.runners._laneC_source_provenance_opponent_derisk import (
    EP_PATTERN,
    N_EPISODE,
    ProvenanceBrain,
    _judge,
)

PROVENANCE_PERCEIVED = "perceived"
PROVENANCE_GENERATED = "generated"
PROVENANCES = (PROVENANCE_PERCEIVED, PROVENANCE_GENERATED)

# 2026-09-22 #5 STABILIZER FIX (research/lbf-fix-source-provenance-abstain; LOCATED by
# research/findings/2026-09-22-borderline-separability-stabilizer-is-buildable.md): source-provenance's opponent
# read `d` separates PERFECTLY intact-vs-lesion on every seed (min-intact 1.000, max-lesion 0.000 -- see the
# finding's separability table), yet the production battery reads it load-bearing only 4/6 seeds (off@s44,s102).
# The fragile step is NOT the read -- it is `_judge()`'s HOST TIE-BREAK: at the lesion's genuine no-signal
# collapse (both provenance pools silent, `rate_perceived == rate_generated`, `d == 0.0`), `_judge()` coin-flips
# `winner` from a per-monitor RNG seeded off `seed` (`_laneC_source_provenance_opponent_derisk._judge`, kept
# UNCHANGED -- it is the validated 6-seed-GO de-risk primitive and its own docstring is explicit that the coin-flip
# is deliberate, "so a no-signal control ... is GENUINE chance, not a degenerate constant"). That per-seed coin
# lands the lesion's label on the SAME side as the intact arm's confident "perceived" on 2 of 6 seeds (s44, s102),
# masking the lesion there while correctly exposing it elsewhere (s100) -- seed-dependence in the INTEGRATION
# step that reads the (perfectly-separated) opponent output, not in the substrate's own discrimination.
#
# FIX, gated behind BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE (default ON since 2026-09-23; an explicit 0/false/no/off/""
# reads as off, byte-identical to pre-fix): when the judged |d| falls below TIE_D_EPS -- i.e. the opponent produced NO signal,
# not merely a close call -- `judge_fact()` reports `label=None` DETERMINISTICALLY instead of forwarding the
# coin-flipped `winner`. `label=None` is not a new sentinel invented for this fix: it is the EXACT value
# `judge_fact()` already returns for a never-encoded key (`known=False`), and `provenance_framed_text()`'s own
# docstring already names this branch ("label is None ... the judgment ties/is undecided -> UNCHANGED") though
# `_judge()` never actually produced it before now -- the mapping existed as a documented intent, unwired. This
# is an HONEST functional read-out (a collapsed opponent genuinely cannot report "I saw this" or "I inferred
# this" -- abstaining is the truthful state, not a coin flip dressed as a judgment), not tuning: TIE_D_EPS is set
# from the SAME float-exact-tie floor `_judge()` itself already uses on the raw margin (1e-9), nothing was fit to
# any seed's count, and the intact arm's `d` (1.000 on every observed seed, per the finding) sits nowhere near
# this epsilon, so the fix only ever engages on the already-degenerate no-signal state -- it can never turn a
# confident intact read into an abstain.
TIE_D_EPS = 1e-6

# PRODUCTION DEFAULT-ON since 2026-09-23 (owner-authorized validated flip; evidence: finding
# 2026-09-22-source-provenance-abstain-at-tie-load-bearing-6-6 + allfixes2 robust core). An EXPLICIT falsy value
# (BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE=0/false/no/off/"") is the reversible escape -> the pre-fix coin-flip forward,
# byte-identical.
_ABSTAIN_AT_TIE_DEFAULT_ON = True


def source_prov_abstain_at_tie_enabled() -> bool:
    """`BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE` unset -> `_ABSTAIN_AT_TIE_DEFAULT_ON` (True since 2026-09-23); set -> ON iff
    in {1,true,on,yes} (so an explicit `=0` keeps the OFF arm reachable). ON -> `judge_fact()` reports a
    deterministic abstain (`label=None`) at a genuine opponent no-signal collapse (`|d| < TIE_D_EPS`) instead of
    forwarding `_judge()`'s host coin-flip. Mirrors the enabled()/lesioned() env-flag convention used by the sibling
    production organ (`source_provenance_production_organ.source_provenance_enabled` / `.source_provenance_lesioned`)."""
    v = os.environ.get("BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE")
    if v is None:
        return _ABSTAIN_AT_TIE_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "on", "yes")


def _stable_pattern(key: Any, seed: int, *, ep: int = EP_PATTERN, n_episode: int = N_EPISODE) -> np.ndarray:
    """A deterministic content pattern for `key`: `ep` neuron indices in [0, n_episode), stable across calls
    (the same key always maps to the same episode assembly -- a fact re-encountered is the SAME episodic
    memory, not a fresh one) and well-spread across different keys. This is a good-faith hash, not a
    cryptographic guarantee of zero collision between two distinct keys -- production key spaces are small
    relative to N_EPISODE choose EP_PATTERN, so collision risk is a measured, not assumed-away, quantity (see
    the wire-in de-risk's collision-rate check). `seed` lets two monitors over the same key space (e.g. a
    lesioned twin built for a load-bearing control) draw IDENTICAL patterns, so only the learning differs."""
    digest = hashlib.sha256(f"{int(seed)}:{key!r}".encode("utf-8")).digest()
    rng = np.random.default_rng(np.frombuffer(digest[:8], dtype=np.uint64)[0])
    return np.sort(rng.choice(n_episode, size=ep, replace=False)).astype(np.int64)


class SourceProvenanceHonestyMonitor:
    """One `ProvenanceBrain` (the #129 spiking opponent comparator) plus a key -> content-pattern episodic map.

    `lesion=True` builds the runner's OWN VERIFIED failing-direction control: every `encode_fact` call runs
    with the Hebbian plasticity gate SHUT (`ProvenanceBrain.encode(..., learning=False)`), exactly anti-cheat
    (1) of the 6-seed GO ("LEARNING-OFF -> no discrimination ... accuracy collapses to chance"). The wire-in
    de-risk uses this to prove the honesty framing is driven by the LEARNED trace, not by a Python if/else on
    a caller-supplied label: lesioned, `judge_fact` reads back ~silent prov pools and the discrimination
    collapses toward chance, so the text framing can no longer reliably distinguish perceived from generated.
    """

    def __init__(self, seed: int = 42, *, lesion: bool = False):
        self.seed = int(seed)
        self.lesion = bool(lesion)
        self._brain = ProvenanceBrain(self.seed)
        self._rng = np.random.default_rng(self.seed)
        self._patterns: dict[Any, np.ndarray] = {}
        self._encoded_as: dict[Any, str] = {}

    def is_known(self, key: Any) -> bool:
        return key in self._patterns

    def encode_fact(self, key: Any, provenance: str) -> None:
        """Bind `key` to a fresh content pattern and Hebbian-teach it under `provenance`'s encoding context.
        Idempotent: a key already encoded here keeps its FIRST provenance -- an episodic trace records how the
        brain first came to hold this content, not the last caller's re-claim about it."""
        if provenance not in PROVENANCES:
            raise ValueError(f"provenance must be one of {PROVENANCES!r}, got {provenance!r}")
        if key in self._patterns:
            return
        pattern = _stable_pattern(key, self.seed)
        self._patterns[key] = pattern
        self._encoded_as[key] = provenance
        self._brain.encode(pattern, provenance, learning=not self.lesion)

    def judge_fact(self, key: Any) -> dict[str, Any]:
        """Recall `key` from CONTENT ALONE (the encoding-context lines are silent, exactly as at recall in the
        de-risk) and read the opponent sign. `known=False` (label=None) for a key never encoded here -- this
        monitor never fabricates a provenance judgment for content it was never shown (the de-risk's anti-cheat
        (3): 'a never-encoded pattern must leave both prov pools ~silent').

        `BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE` (default ON since 2026-09-23, `source_prov_abstain_at_tie_enabled()`): at a genuine
        opponent no-signal collapse (`|d| < TIE_D_EPS`) the label reads a deterministic abstain (`None`) rather
        than `_judge()`'s seed-dependent host coin-flip -- see the module-level note above `TIE_D_EPS`. OFF, this
        method is byte-identical to its pre-fix behavior (`label` is always `_judge()`'s raw `winner`)."""
        pattern = self._patterns.get(key)
        if pattern is None:
            return {"known": False, "label": None, "d": None, "encoded_as": None, "agrees_with_encoded": None}
        rec = self._brain.recall(pattern)
        winner, d = _judge(rec, self._rng)
        label = winner
        if source_prov_abstain_at_tie_enabled() and abs(d) < TIE_D_EPS:
            label = None   # honest "no clean provenance signal" -- not a coin-flipped perceived/generated guess
        encoded_as = self._encoded_as.get(key)
        return {
            "known": True,
            "label": label,
            "d": float(d),
            "rate_perceived": float(rec["rate_perceived"]),
            "rate_generated": float(rec["rate_generated"]),
            "encoded_as": encoded_as,
            "agrees_with_encoded": bool(label == encoded_as),
        }


def provenance_framed_text(kind: str, raw_text: str, label: str | None, *, cue: tuple[Any, ...] | None = None) -> str:
    """Wrap `raw_text` (the assertion the answer pipeline already produced) in a provenance-honest frame, driven
    by the JUDGED label read back from the live spiking monitor -- not by a caller-supplied claim about how the
    fact was obtained:

      - label == PROVENANCE_PERCEIVED  -> UNCHANGED. The dominant, directly-taught case reads exactly as it
        does today (byte-identical to the pre-existing text for the common case).
      - label == PROVENANCE_GENERATED  -> FLAGGED. The brain marks the claim as its own inference rather than
        something it was told.
      - label is None (never presented to the monitor, or the judgment ties/is undecided) -> UNCHANGED. Absent
        provenance evidence is not evidence of either source: flagging it as "generated" here would itself be a
        confabulated hedge, and asserting it as confidently "perceived" would overclaim a read the monitor
        never took.
    """
    if label == PROVENANCE_GENERATED:
        text = (raw_text[:1].lower() + raw_text[1:]) if raw_text else raw_text
        text = text.rstrip(".")
        return f"I believe {text}, but I reasoned that myself rather than being told it directly."
    return raw_text
