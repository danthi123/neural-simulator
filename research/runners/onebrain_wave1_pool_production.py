"""ONE-BRAIN INTEGRATION PROGRAM, PHASE 3 WAVE 1 — pool accessor (opt-in, DEFAULT-OFF).

Extends the shipped 4-organ single pool (`onebrain_single_pool_production.get_single_pool`: surprise +
world-model + metacog + pragmatic) with comprehension + source_provenance onto ONE shared `merge_organs`
pool, behind a NEW, DISTINCT flag (`BRAIN_ONEBRAIN_WAVE1_POOL`, default-OFF) — never touches the base
`BRAIN_ONEBRAIN_SINGLE_POOL` flag or its pool object.

DE-RISKED by `research/runners/_onebrain_wave1_organread_verify.py` (read it first — this module is the tiny
memoized accessor, not the reconciliation; `_wave1_descriptors()` there is the single source of truth for the
exact reconciled 6-organ family, reused by import here so the pool this module builds can never drift from
what the organ-read gate validated).

SCOPE (honest, deliberately narrow — Wave 1's "true next step" per
docs/plans/2026-09-02-onebrain-integration-program.md §Phase 3 is the MIGRATION-SAFETY organ-read rung, not
the production flip): this module is PURELY ADDITIVE and is NOT wired into any of the 6 organs' live
`get_organ()` singletons (`surprise_production_organ.py` / `worldmodel_production_organ.py` /
`metacog_production_organ.py` / `pragmatic_production_organ.py` / `comprehension_production_organ.py` /
`source_provenance_production_organ.py` — every one of those files is UNCHANGED by this landing, verify with
`git diff`). Consequently "byte-identical when the flag is OFF" is not merely asserted — it is TRIVIALLY true
by construction: nothing in the live chat path calls `get_wave1_pool()` regardless of the flag, so setting
`BRAIN_ONEBRAIN_WAVE1_POOL` has ZERO effect on production today. The flag/accessor exist so a FUTURE, separate
landing (mirroring how the base single pool's own `get_organ()` wiring was a later commit after ITS organ-read
GO) can wire the 6 organs onto this ONE pool without re-deriving the reconciliation. NO `sim/` edit; all state
lives in `research/runners/`.
"""
from __future__ import annotations

import os

_WAVE1_POOL_DEFAULT_ON = False


def wave1_pool_enabled() -> bool:
    """Opt-in (DEFAULT per `_WAVE1_POOL_DEFAULT_ON` = OFF). `BRAIN_ONEBRAIN_WAVE1_POOL` in {1,true,yes,on} ->
    the flag reads as ON, but see the module docstring: NOTHING in the live chat path currently checks this
    function, so flipping it has no production effect yet — it exists for the pool accessor + a future,
    separate get_organ() wiring landing."""
    v = os.environ.get("BRAIN_ONEBRAIN_WAVE1_POOL")
    if v is None:
        return _WAVE1_POOL_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


# process-shared, built once per seed on first use — the same lifetime discipline as
# `onebrain_single_pool_production.get_single_pool`, so a future 6-organ get_organ() wiring resolves the SAME
# pool object across all six organs.
_POOL: dict = {}


def get_wave1_pool(seed: int = 42):
    """The process-shared 6-organ Wave-1 merged pool (built once on first use, memoized by seed). ONE
    `merge_organs([surprise, worldmodel, metacog, pragmatic, comprehension, source_provenance], wire=True)`
    spiking bridge — the shipped single pool's 4-organ reconciliation UNCHANGED
    (`_onebrain_twopool_merge_organread_verify._recon_descriptors`, imported not re-derived) plus comprehension
    + source_provenance reconciled the SAME way metacog/pragmatic already are (pop the conflicting
    `enable_hebbian_learning` + gain-0 freeze the organ's own regions — see
    `_onebrain_wave1_organread_verify._wave1_descriptors`'s docstring for the full reasoning + the config-key
    inspection that found no other conflict).

    Reuse-by-import: `_wave1_descriptors()` is imported from the organ-read verify runner (the codebase's
    standard single-source-of-truth pattern for a validated de-risk artefact), so this production pool is BUILT
    FROM THE EXACT reconciled family the organ-read gate validated — no re-declaration, no drift. Lazy import:
    no import-time cost when the flag is off (nothing calls this while `wave1_pool_enabled()` is unused)."""
    key = int(seed)
    if key not in _POOL:
        from research.runners.onebrain_merge_framework import merge_organs
        from research.runners._onebrain_wave1_organread_verify import _wave1_descriptors
        _POOL[key] = merge_organs(_wave1_descriptors(), key, wire=True)
    return _POOL[key]
