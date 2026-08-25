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
        (3): 'a never-encoded pattern must leave both prov pools ~silent')."""
        pattern = self._patterns.get(key)
        if pattern is None:
            return {"known": False, "label": None, "d": None, "encoded_as": None, "agrees_with_encoded": None}
        rec = self._brain.recall(pattern)
        winner, d = _judge(rec, self._rng)
        encoded_as = self._encoded_as.get(key)
        return {
            "known": True,
            "label": winner,
            "d": float(d),
            "rate_perceived": float(rec["rate_perceived"]),
            "rate_generated": float(rec["rate_generated"]),
            "encoded_as": encoded_as,
            "agrees_with_encoded": bool(winner == encoded_as),
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
