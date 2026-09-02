"""ONE-BRAIN INTEGRATION PROGRAM, PHASE 3 WAVE 2 — pool accessor (opt-in, DEFAULT-OFF).

Extends the shipped 6-organ Wave-1 pool (`onebrain_wave1_pool_production.get_wave1_pool`: surprise + world-model
+ metacog + pragmatic + comprehension + source_provenance) with self_schema + curiosity + causal_whatif onto ONE
shared `merge_organs` pool, behind a NEW, DISTINCT flag (`BRAIN_ONEBRAIN_WAVE2_POOL`, default-OFF) — never touches
the Wave-1 flag/pool object or the base `BRAIN_ONEBRAIN_SINGLE_POOL` flag/pool object.

DE-RISKED by `research/runners/_onebrain_wave2_organread_verify.py` (read it first — this module is the tiny
memoized accessor, not the reconciliation; `_wave2_descriptors()` there is the single source of truth for the
exact reconciled 9-organ family, reused by import here so the pool this module builds can never drift from what
the organ-read gate validated).

SCOPE (honest, deliberately narrow — mirrors Wave 1's own scope decision exactly): this module is PURELY ADDITIVE
and is NOT wired into any of the 9 organs' live `get_organ()` singletons (`surprise_production_organ.py` /
`worldmodel_production_organ.py` / `metacog_production_organ.py` / `pragmatic_production_organ.py` /
`comprehension_production_organ.py` / `source_provenance_production_organ.py` /
`self_schema_production_organ.py` / `curiosity_production_organ.py` /
`causal_whatif_production_organ.py` — every one of those files is UNCHANGED by this landing, verify with
`git diff`). Consequently "byte-identical when the flag is OFF" is not merely asserted — it is TRIVIALLY true by
construction: nothing in the live chat path calls `get_wave2_pool()` regardless of the flag, so setting
`BRAIN_ONEBRAIN_WAVE2_POOL` has ZERO effect on production today. The flag/accessor exist so a FUTURE, separate
landing can wire the 9 organs onto this ONE pool without re-deriving the reconciliation — the same deferred-rung
pattern Wave 1 used (its own production `get_organ()` wiring is still a separate, later, not-yet-landed commit).
NO `sim/` edit; all state lives in `research/runners/`.
"""
from __future__ import annotations

import os

_WAVE2_POOL_DEFAULT_ON = False


def wave2_pool_enabled() -> bool:
    """Opt-in (DEFAULT per `_WAVE2_POOL_DEFAULT_ON` = OFF). `BRAIN_ONEBRAIN_WAVE2_POOL` in {1,true,yes,on} -> the
    flag reads as ON, but see the module docstring: NOTHING in the live chat path currently checks this function,
    so flipping it has no production effect yet — it exists for the pool accessor + a future, separate
    get_organ() wiring landing."""
    v = os.environ.get("BRAIN_ONEBRAIN_WAVE2_POOL")
    if v is None:
        return _WAVE2_POOL_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


# process-shared, built once per seed on first use — the same lifetime discipline as
# `onebrain_wave1_pool_production.get_wave1_pool` / `onebrain_single_pool_production.get_single_pool`, so a
# future 9-organ get_organ() wiring resolves the SAME pool object across all nine organs.
_POOL: dict = {}


def get_wave2_pool(seed: int = 42):
    """The process-shared 9-organ Wave-2 merged pool (built once on first use, memoized by seed). ONE
    `merge_organs([surprise, worldmodel, metacog, pragmatic, comprehension, source_provenance, self_schema,
    curiosity, causal_whatif], wire=True)` spiking bridge — the shipped Wave-1 pool's 6-organ reconciliation
    UNCHANGED (`_onebrain_wave1_organread_verify._wave1_descriptors`, imported not re-derived) plus self_schema
    (region-renamed to avoid the metacog `workspace`/`workspace_fs` name collision + wiring-key collision),
    curiosity (region-renamed to avoid the surprise `cue` name collision), and causal_whatif (standard
    hebbian-pop + gain-0-freeze, no rename needed) — see `_onebrain_wave2_organread_verify._wave2_descriptors`'s
    docstring for the full seam reasoning.

    Reuse-by-import: `_wave2_descriptors()` is imported from the organ-read verify runner (the codebase's
    standard single-source-of-truth pattern for a validated de-risk artefact), so this production pool is BUILT
    FROM THE EXACT reconciled family the organ-read gate validated — no re-declaration, no drift. Lazy import: no
    import-time cost when the flag is off (nothing calls this while `wave2_pool_enabled()` is unused)."""
    key = int(seed)
    if key not in _POOL:
        from research.runners.onebrain_merge_framework import merge_organs
        from research.runners._onebrain_wave2_organread_verify import _wave2_descriptors
        _POOL[key] = merge_organs(_wave2_descriptors(), key, wire=True)
    return _POOL[key]
