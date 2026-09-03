"""ONE-BRAIN INTEGRATION PROGRAM, PHASE 3 WAVE 3 (the FINAL merge wave) — pool accessor (opt-in, DEFAULT-OFF).

Extends the shipped 9-organ Wave-2 pool (`onebrain_wave2_pool_production.get_wave2_pool`: surprise + world-model
+ metacog + pragmatic + comprehension + source_provenance + self_schema + curiosity + causal_whatif) with
prospective_memory + d6_multiref_wm onto ONE shared `merge_organs` pool, behind a NEW, DISTINCT flag
(`BRAIN_ONEBRAIN_WAVE3_POOL`, default-OFF) — never touches the Wave-2 flag/pool object or the base
`BRAIN_ONEBRAIN_SINGLE_POOL`/`BRAIN_ONEBRAIN_WAVE1_POOL` flags/pool objects.

DE-RISKED by `research/runners/_onebrain_wave3_organread_verify.py` (read it first — this module is the tiny
memoized accessor, not the reconciliation; `_wave3_descriptors()` there is the single source of truth for the
exact reconciled 11-organ family, reused by import here so the pool this module builds can never drift from what
the organ-read gate validated).

SCOPE (honest, deliberately narrow — mirrors Wave 1/Wave 2's own scope decision exactly): this module is PURELY
ADDITIVE and is NOT wired into any of the 11 organs' live `get_organ()` singletons (`surprise_production_organ.py`
/ `worldmodel_production_organ.py` / `metacog_production_organ.py` / `pragmatic_production_organ.py` /
`comprehension_production_organ.py` / `source_provenance_production_organ.py` /
`self_schema_production_organ.py` / `curiosity_production_organ.py` / `causal_whatif_production_organ.py` (via
`_causal_forward_model_derisk.py`) / `prospective_memory_production_organ.py` /
`d6_multiref_wm_production_organ.py` — every one of those files is UNCHANGED by this landing, verify with
`git diff`). Consequently "byte-identical when the flag is OFF" is not merely asserted — it is TRIVIALLY true by
construction: nothing in the live chat path calls `get_wave3_pool()` regardless of the flag, so setting
`BRAIN_ONEBRAIN_WAVE3_POOL` has ZERO effect on production today. The flag/accessor exist so a FUTURE, separate
landing can wire the 11 organs onto this ONE pool without re-deriving the reconciliation — the same deferred-rung
pattern Wave 1/2 used (their own production `get_organ()` wiring is still a separate, later, not-yet-landed
commit). NO `sim/` edit; all state lives in `research/runners/`.

This is the LARGEST and FINAL merge wave in the program's sequenced plan (docs/plans/2026-09-02-onebrain-
integration-program.md, Phase 3): it completes the d6->comprehension cross-edge TEMPLATE onto the single pool
(the same substrate d6's already-shipped `onebrain_xedge_curiosity_d6_production.py` cross-edge reads from, now
co-resident with the other 9 organs rather than its own separate two-organ pool).
"""
from __future__ import annotations

import os

_WAVE3_POOL_DEFAULT_ON = False


def wave3_pool_enabled() -> bool:
    """Opt-in (DEFAULT per `_WAVE3_POOL_DEFAULT_ON` = OFF). `BRAIN_ONEBRAIN_WAVE3_POOL` in {1,true,yes,on} -> the
    flag reads as ON, but see the module docstring: NOTHING in the live chat path currently checks this function,
    so flipping it has no production effect yet — it exists for the pool accessor + a future, separate
    get_organ() wiring landing."""
    v = os.environ.get("BRAIN_ONEBRAIN_WAVE3_POOL")
    if v is None:
        return _WAVE3_POOL_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


# process-shared, built once per seed on first use — the same lifetime discipline as
# `onebrain_wave2_pool_production.get_wave2_pool` / `onebrain_wave1_pool_production.get_wave1_pool`, so a
# future 11-organ get_organ() wiring resolves the SAME pool object across all eleven organs.
_POOL: dict = {}


def get_wave3_pool(seed: int = 42):
    """The process-shared 11-organ Wave-3 merged pool (built once on first use, memoized by seed). ONE
    `merge_organs([surprise, worldmodel, metacog, pragmatic, comprehension, source_provenance, self_schema,
    curiosity, causal_whatif, prospective_memory, d6_multiref_wm], wire=True)` spiking bridge — the shipped
    Wave-2 pool's 9-organ reconciliation UNCHANGED (`_onebrain_wave2_organread_verify._wave2_descriptors`,
    imported not re-derived) plus prospective_memory (hebbian pop + full-region gain-0 freeze; its pool-gained
    ~300-weight attractor survives the freeze unclipped vs the pool's 45 hebbian_max_weight ceiling — verified,
    not assumed) and d6_multiref_wm (same hebbian pop + freeze seam; its region names are DISCOVERED at build —
    no rename needed, zero collision with any of the other 10 organs' region/wiring-key namespaces) — see
    `_onebrain_wave3_organread_verify._wave3_descriptors`'s docstring for the full seam reasoning.

    Reuse-by-import: `_wave3_descriptors()` is imported from the organ-read verify runner (the codebase's
    standard single-source-of-truth pattern for a validated de-risk artefact), so this production pool is BUILT
    FROM THE EXACT reconciled family the organ-read gate validated — no re-declaration, no drift. Lazy import: no
    import-time cost when the flag is off (nothing calls this while `wave3_pool_enabled()` is unused)."""
    key = int(seed)
    if key not in _POOL:
        from research.runners.onebrain_merge_framework import merge_organs
        from research.runners._onebrain_wave3_organread_verify import _wave3_descriptors
        _POOL[key] = merge_organs(_wave3_descriptors(), key, wire=True)
    return _POOL[key]
