"""SOURCE-PROVENANCE HONESTY — did the brain SEE a fact or INFER it, wired into the PRODUCTION turn (board #129,
Vikunja #137, 2026-08-25).

The owner's named faculty: "I saw this fact vs I inferred/imagined it" — a functional honesty read the brain's
reply should reflect (hedges or flags a generated-source claim) rather than asserting every claim with the
same flat confidence regardless of how the brain came to hold it.

This is the process-shared PRODUCTION ORGAN wrapper (mirrors `metacog_production_organ.py` /
`curiosity_production_organ.py`'s `enabled()` / `lesioned()` / `get_organ()` convention) around the validated
6-seed GO mechanism (`research/runners/_laneC_source_provenance_opponent_derisk.py`,
research/findings/2026-08-25-laneC-source-provenance-opponent-perceived-vs-generated-6seed-GO.md) via the thin
production wrapper `research/runners/source_provenance_honesty.py`.

BRAIN-BASED: the provenance judgment is a live read of an opponent-comparator spiking circuit (two
neuromodulatory encoding-context lines each gating a separate zero-init Hebbian episode->provenance trace,
mutually-inhibiting FS interneurons, a sign/ratio read-out) — not a host `if` on a caller-supplied claim. The
host boundary (declared, unchanged from the de-risk): which encoding context a fact is taught under (PERCEIVED
for every directly-taught/recalled fact; GENERATED for a multi-hop composed conclusion) is supplied by the
CALLER (`BrainConversationalAgent.known_fact_record` / `.reasoned_fact_record`), exactly as the de-risk's own
encoding context is externally timed. The monitor's readback of WHICH label a given content pattern carries is
the genuine spiking read, and it is what decides the reply's framing.

MOAT-SAFE + ADDITIVE: this organ NEVER produces an answer or flips an abstain — it only reframes the TEXT of an
already-produced, moat-verified answer (assert -> flagged-generated), or leaves it untouched (assert -> assert,
the dominant perceived case). PRODUCTION DEFAULT-ON (2026-09-01, `_DEFAULT_ON`): `BRAIN_SOURCE_PROVENANCE_HONESTY`
unset now builds the organ; `=0` (explicit off) is the byte-identical escape to the pre-flip oracle (the organ
is never built, no substrate step is taken).

LESION-LOAD-BEARING: `BRAIN_SOURCE_PROVENANCE_HONESTY_LESION=1` rebuilds the organ with its Hebbian plasticity
gate held shut at encode (the de-risk's own verified failing-direction anti-cheat: "LEARNING-OFF -> no
discrimination ... accuracy collapses to chance"), so the framing decision demonstrably stops tracking true
provenance under the lesion — proving the live-chat framing is driven by the LEARNED trace, not a host flag.

FUNCTIONAL CORRELATE, NOT phenomenal: this measures + reports a source-monitoring CORRELATE (a learned
perceived-vs-generated opponent read). It makes no claim of subjective experience.

NO `sim/` edit; reuse-by-import; numpy-CPU backend (the #129 de-risk's own validated lane).
"""
from __future__ import annotations

import os

from research.runners.source_provenance_honesty import SourceProvenanceHonestyMonitor

_ORGAN: SourceProvenanceHonestyMonitor | None = None
_ORGAN_KEY: tuple | None = None


# 2026-09-01 FLIPPED DEFAULT-ON (owner auto-flip directive; the #140 rung -- webapp/source_monitoring_honesty_
# chat.py's `BRAIN_SOURCE_MONITORING_FRAMES_HONESTY`, itself flipped default-ON the same session -- nests
# ENTIRELY inside this organ's `source_provenance_enabled()` gate in webapp/server.py, so #140 was hollow until
# THIS flag was also default-ON; independently re-verified through the real `webapp.server.brain_chat` handler,
# 6-seed, before commit: research/findings/raw/_source_provenance_honesty_flip/flip_verify.json. `BRAIN_
# SOURCE_PROVENANCE_HONESTY=0` is the byte-identical escape to the pre-flip (organ never built) behavior.
_DEFAULT_ON = True


def source_provenance_enabled() -> bool:
    """DEFAULT-ON anchor (`_DEFAULT_ON`, board #129 + #140 rung): enabled UNLESS `BRAIN_SOURCE_PROVENANCE_HONESTY`
    is an explicit off (0/false/no/off/""); `=0` reverts byte-identically to the pre-flip (organ never built,
    no substrate step taken, no `provenance` key added) behavior. Mirrors the `_LEARNED_ANIMACY_CUE_DEFAULT_ON`
    / `_MERGE2_DEFAULT_ON` convention used elsewhere for a validated default-ON flip."""
    v = os.environ.get("BRAIN_SOURCE_PROVENANCE_HONESTY")
    if _DEFAULT_ON:
        return not (v is not None and v.strip().lower() in ("0", "false", "no", "off", ""))
    return v is not None and v.strip().lower() in ("1", "true", "on", "yes")


def source_provenance_lesioned() -> bool:
    """`BRAIN_SOURCE_PROVENANCE_HONESTY_LESION` in {1,true,on,yes} -> the load-bearing lesion (Hebbian
    plasticity gate held shut at encode; the de-risk's own verified failing-direction anti-cheat)."""
    v = os.environ.get("BRAIN_SOURCE_PROVENANCE_HONESTY_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "on", "yes")


def get_organ(seed: int = 42, *, lesion: bool = False) -> SourceProvenanceHonestyMonitor:
    """The process-shared #129 spiking opponent-comparator provenance monitor (built once per (seed, lesion)
    pair; rebuilt if the lesion flag changes, e.g. between a normal request and a lesion-verification probe)."""
    global _ORGAN, _ORGAN_KEY
    key = (int(seed), bool(lesion))
    if _ORGAN is None or _ORGAN_KEY != key:
        _ORGAN = SourceProvenanceHonestyMonitor(seed=seed, lesion=bool(lesion))
        _ORGAN_KEY = key
    return _ORGAN
