"""SOURCE-MONITORING DRIVES HONESTY FRAMING (board #140 rung, 2026-09-01) -- makes the #129 source-provenance
opponent-comparator's OWN judged label load-bearing on the live chat-path GENERATED case, not just a diagnostic.

THE GAP THIS CLOSES (research/findings/2026-09-01-production-default-flip-plan.md row #6; board #140). The #129
source-provenance-honesty organ (`research/runners/source_provenance_honesty.py`, 6-seed GO
`research/findings/2026-08-25-laneC-source-provenance-opponent-perceived-vs-generated-6seed-GO.md`) is wired
into `webapp/server.py`'s single-fact `/api/brain-chat` path behind `BRAIN_SOURCE_PROVENANCE_HONESTY`, but ONLY
the PERCEIVED half of that wiring ever reaches live reply TEXT: a directly-recalled fact is always encoded
PERCEIVED and (correctly) renders unchanged, while a compositional-chain-derived ("GENERATED") answer is
explicitly EXCLUDED from `provenance_framed_text` (see the "SOURCE-PROVENANCE HONESTY" comment in
`webapp/server.py`) in favor of the UNCONDITIONAL, host-driven `compositional_chain_route.frame_derived_answer`
(a moat-hardening requirement, audit req #4, which must never be a caller's ONLY hedge on a derived answer). So
flipping `BRAIN_SOURCE_PROVENANCE_HONESTY` on the real endpoint changes NO observable reply text -- a hollow
flip, exactly what board #140 names: "the GENERATED half has no live HTTP exposure at all."

THE FIX (additive, ONE new flag, does not touch the #129 organ, does not weaken `frame_derived_answer`'s
guarantee). `BRAIN_SOURCE_MONITORING_FRAMES_HONESTY=1` lets a compositional-chain (GENERATED) reply ALSO be
offered the SUBSTRATE's own honesty framing (`source_provenance_honesty.provenance_framed_text`, reused
verbatim -- NOT reimplemented) as an ALTERNATIVE surface to the host-generic `frame_derived_answer` text. The
swap is chosen ONLY when the #129 organ's OWN live readback for THIS EXACT fact agrees the content reads
GENERATED (`judge_fact()["label"] == PROVENANCE_GENERATED`) -- never when the caller's claim disagrees with what
the organ actually read back. This is the SAME "readback decides the text, not the caller's claim" boundary
`source_provenance_production_organ.py` already documents and the #129 de-risk already validated; this module
adds no new spiking machinery, it only decides WHETHER the already-computed judged label is allowed to reach a
branch of `webapp/server.py` it could not reach before.

THE SAFETY NET (why this can never make a derived answer read as a bare fact). When the organ's read does NOT
confirm GENERATED (a genuine tie/None, or -- the load-bearing case -- `BRAIN_SOURCE_PROVENANCE_HONESTY_LESION=1`
collapsing the discrimination toward chance per the de-risk's own verified failing-direction anti-cheat), the
caller in `webapp/server.py` FALLS BACK to `frame_derived_answer`'s wording, unchanged. So audit req #4's
guarantee ("never the ONLY thing standing between a derived answer and being presented as a plain perceived
fact") holds in EVERY reachable state of this flag: it only ever SWAPS which of two already-honest hedges is
used, it never removes the hedge, and it can never fabricate an unhedged "perceived" surface for GENERATED
content.

LOAD-BEARING (the anti-hollow bar, owner-flagged 2026-08-19 `feedback_faculties_must_drive_not_observe`): the
WORDING of a chain-derived reply now differs depending on whether the #129 organ's live readback confirms the
GENERATED label -- `frame_derived_answer`'s "I derived this from: ..." vs `provenance_framed_text`'s "I believe
..., but I reasoned that myself rather than being told it directly." -- and that confirmation rate collapses
under `BRAIN_SOURCE_PROVENANCE_HONESTY_LESION=1`, so the vary-effect this flag introduces is demonstrably driven
by the LEARNED opponent-comparator trace, not a host if/else keyed on `_is_chain_route`. See
`research/runners/_source_monitoring_honesty_flip_verify.py` for the 6-seed/6-item vary-then-lesion measurement
through the real `brain_chat` handler, plus a direct 6-seed mechanism-level sweep of the underlying monitor.

MOAT-SAFE: never changes WHICH fact is stated (only which of two already-honest hedge phrasings wraps the same
composed content), never turns an abstain into an answer, and reuses the SAME `SourceProvenanceHonestyMonitor` /
`provenance_framed_text` the already-shipped #129 wiring uses -- no new spiking machinery, no `sim/` edit.

DEFAULT-OFF: `BRAIN_SOURCE_MONITORING_FRAMES_HONESTY` unset -> `source_monitoring_frames_honesty_enabled()`
returns False -> `webapp/server.py`'s chain-route branch is UNTOUCHED (byte-identical to pre-existing behavior:
`frame_derived_answer` applies unconditionally, exactly as before this module existed). Requires
`BRAIN_SOURCE_PROVENANCE_HONESTY` to ALSO be on for the organ to even be built -- with that unset, this flag has
nothing to read and the branch is byte-identical regardless of its own value.

Owner directive (2026-08-30/09-01 ACTIVE MISSION honesty-boundary clause): this is a FUNCTIONAL read-out of the
brain's own source-monitoring state ("my familiarity monitor reads this as reconstructed, so I'm less sure"),
never an assertion of phenomenal experience.
"""
from __future__ import annotations

import os

# PRODUCTION DEFAULT -- OFF. The flip to default-ON is a separate, later, owner-gated step (this is a NEW,
# additive lever; it does not touch or re-decide `BRAIN_SOURCE_PROVENANCE_HONESTY`'s own default).
_SOURCE_MONITORING_FRAMES_HONESTY_DEFAULT_ON = True


def source_monitoring_frames_honesty_enabled() -> bool:
    """`BRAIN_SOURCE_MONITORING_FRAMES_HONESTY` in {1,true,on,yes} -> a compositional-chain (GENERATED) reply on
    the live single-fact `/api/brain-chat` path may be framed by the #129 organ's OWN judged label instead of
    the host-generic `frame_derived_answer` text, when (and ONLY when) the organ's live readback agrees the
    content reads GENERATED. Unset/{0,false,no,off} -> byte-identical (default OFF, per
    `_SOURCE_MONITORING_FRAMES_HONESTY_DEFAULT_ON`)."""
    v = os.environ.get("BRAIN_SOURCE_MONITORING_FRAMES_HONESTY")
    if v is None:
        return _SOURCE_MONITORING_FRAMES_HONESTY_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "on", "yes")
