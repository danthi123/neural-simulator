"""ONE-BRAIN INTEGRATION PROGRAM, PHASE 3 WAVE 3 (the FINAL merge wave) — pool accessor + the PRODUCTION resolver.

Extends the shipped 9-organ Wave-2 pool (`onebrain_wave2_pool_production.get_wave2_pool`: surprise + world-model
+ metacog + pragmatic + comprehension + source_provenance + self_schema + curiosity + causal_whatif) with
prospective_memory + d6_multiref_wm onto ONE shared `merge_organs` pool. `BRAIN_ONEBRAIN_WAVE3_POOL` is now the
PRODUCTION default (`_WAVE3_POOL_DEFAULT_ON = True`, flipped 2026-09-17). `get_wave3_pool()` stays a PURE 11-organ
builder (the organ-read gates depend on its exact N — see the coherence note below); the production routing lives
in `get_merged_cortical_pool()`.

DE-RISKED by `research/runners/_onebrain_wave3_organread_verify.py` (read it first — the accessor is the tiny
memoized builder, not the reconciliation; `_wave3_descriptors()` there is the single source of truth for the
exact reconciled 11-organ family, reused by import here so the pool this module builds can never drift from what
the organ-read gate validated). Organ-read GO 6/6: all 11 organs read BYTE-IDENTICALLY off the 11-organ pool vs
their standalone builds (`2026-09-17-onebrain-wave3-organ-merge-ALL-11-organs-one-pool-GO.md`).

⭐ THE COHERENCE NOTE (the verify-first crux of the DEFAULT-ON flip). The pools NEST: single(4) ⊂ wave1(6) ⊂
wave2(9) ⊂ wave3(11), each a strict SUPERSET (`get_wave{2,3}_pool` reuse the lower descriptors by import). Naively
flipping wave1/wave2/wave3 all default-ON while the base 4 organs still route to `get_single_pool` would build
FOUR different pool objects and split the organs across them — surprise on the 4-organ pool, comprehension on the
6-organ pool, self_schema on the 9-organ pool, d6 on the 11-organ pool — i.e. DOUBLE-BUILD (surprise's regions in
all four) and organs NOT co-resident (the opposite of one brain). So the flip is coherent ONLY if every WIRED
organ resolves to the SAME pool object. `get_merged_cortical_pool()` is that single routing point: it returns the
highest-enabled pool that CONTAINS the caller (`min_wave` floor), so in production (only wave3 default-ON) all
wired organs get the ONE 11-organ pool. The builders (`get_wave{1,2,3}_pool`) are NOT delegated — the organread
verify gates call them as their N-organ BASELINES (`_onebrain_wave3_organread_verify` compares vs `get_wave2_pool`,
etc.), so delegating a builder would confound those gates.

PRODUCTION WIRING (this flip's landing rung — additive, escape-preserving; `NO sim/ edit`, all in
`research/runners/`). Eight process-shared organs route their `get_organ().shared=` through
`get_merged_cortical_pool()`: surprise / world-model / metacog / pragmatic (min_wave=1) + comprehension
(min_wave=1) + self_schema / curiosity / causal_whatif (min_wave=2). THREE organs are documented EXCLUSIONS, left
on their own path (their descriptor sits in the pool but their live wrapper does not read it — a documented
residual, not a double-build regression):
  * source_provenance — its SHIPPED wrapper (`SourceProvenanceHonestyMonitor`) has an incremental per-turn
    `encode_fact()` API the pool's build-time-only gain re-open would silently no-op; needs a separate online
    re-open mechanism (see that module's `get_organ` docstring). Its POOL read organ is validated; the wrapper
    re-wire is a separate rung.
  * prospective_memory — scope-reduced (its read organ validated in-pool; production wrapper wiring separate).
  * d6_multiref_wm — production builds it PER SESSION (`webapp/server.py::_get_multiref_organ`, cache_key-isolated
    to avoid cross-session referent leak); the process-shared pool cannot be shared per-session safely yet (a
    future per-session-safe landing). d6's module `get_organ()` is DEAD for production and left unchanged.

ESCAPE (byte-identical): `BRAIN_ONEBRAIN_WAVE3_POOL=0` (the only default-ON pool flag) -> `wave3_pool_enabled()`
False -> `get_merged_cortical_pool()` returns None for every organ -> each falls back to its EXACT pre-flip path
(surprise/wm/metacog/pragmatic -> `single_pool` which is default-on; comprehension -> xedge/standalone;
self_schema/curiosity/causal -> standalone) -> byte-identical to `main` before this flip. wave1/wave2 stay
default-OFF de-risk builders; setting one manually (with wave3 off) opts the organs it contains onto that pool.

This is the LARGEST and FINAL merge wave in the program's sequenced plan (docs/plans/2026-09-02-onebrain-
integration-program.md, Phase 3). PREP: this flip is de-risked (organread GO 6/6) but is NOT to be merged to main
until the integrator runs the answer-preservation battery (`_onebrain_11organ_pool_flip_regression` 6-seed, or the
integrated `onebrain_regression_battery --flag BRAIN_ONEBRAIN_WAVE3_POOL`).
"""
from __future__ import annotations

import os

_WAVE3_POOL_DEFAULT_ON = True    # LANDED 2026-09-19: the 11-organ pool flip is validated at 3 levels — DR-1 6/6 under the
                                 # faithful/calmer curiosity regime, engagement ON==OFF EXACT (salience max|Δ|=0.0000 at
                                 # intermediate raw points), and the 38-faculty cross-faculty no-regression battery all_pass
                                 # (0 regressed, run on AWS r7i to avoid the local OOM). This is the one-brain culmination.


def wave3_pool_enabled() -> bool:
    """DEFAULT-ON per `_WAVE3_POOL_DEFAULT_ON` = True (the PRODUCTION one-brain cortical pool; flipped 2026-09-17,
    de-risk organread GO 6/6). `BRAIN_ONEBRAIN_WAVE3_POOL=0` reverts byte-identical (every wired organ falls back to
    its pre-flip path — see the module docstring's ESCAPE note). `BRAIN_ONEBRAIN_WAVE3_POOL` in {1,true,yes,on} ->
    ON. Production routing goes through `get_merged_cortical_pool()`, not `wave3_pool_enabled()` directly."""
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


def get_merged_cortical_pool(seed: int = 42, min_wave: int = 1):
    """THE PRODUCTION ROUTING POINT (the DEFAULT-ON flip's single source of truth). Return the ONE merged spiking
    pool a WIRED cortical organ should attach its `shared=` to, or None (the organ then takes its pre-flip
    standalone path -> byte-identical). Every wired organ calls THIS (never a builder directly), so all wired
    organs on the same process land on the SAME pool object -> one brain, no double-build.

    Resolution: the HIGHEST-enabled pool that CONTAINS the caller. `min_wave` is the caller's membership floor:
      * min_wave=1 — organs in every pool (surprise, world-model, metacog, pragmatic, comprehension): accept
        wave3(11) > wave2(9) > wave1(6).
      * min_wave=2 — organs first introduced in Wave 2 (self_schema, curiosity, causal_whatif): accept wave3 > wave2
        ONLY (never wave1, which lacks their descriptors -> would KeyError/garbage-read). This floor is why the
        resolver is `min_wave`-aware rather than a blind "highest wins".
    In PRODUCTION only wave3 is default-ON, so every wired organ resolves to the ONE 11-organ pool. wave1/wave2 are
    reachable only if an operator sets them manually (with wave3 off) — a de-risk opt-in, kept faithful here.

    The builders (`get_wave{1,2,3}_pool`) are deliberately NOT delegated: the organread verify gates call them as
    their fixed N-organ BASELINES, so delegating one would confound the gate that proves the merge is clean."""
    if wave3_pool_enabled():
        # D3 (2026-09-23, DEFAULT-OFF): with `BRAIN_ONEBRAIN_AFFECT_POOL=1` every wired organ resolves to the
        # 12-organ pool (the 11 Wave-3 organs + the affect ladder) so affect and the cortical organs share ONE
        # object. Unset -> the lazy import below returns False -> the unchanged Wave-3 path.
        from research.runners.onebrain_affect_pool_flags import affect_pool_enabled
        if affect_pool_enabled():
            from research.runners.onebrain_affect_pool import get_affect_pool
            return get_affect_pool(seed)
        # BRAIN_XEDGE_IN_WAVE3 (2026-09-24, DEFAULT-OFF): the d6 w{k}->sel cross-edge grown INSIDE this pool (the
        # Wave-3 organs + the R3 da_credit organ + the declared cross-edges), so comprehension and the per-session d6
        # organ share ONE object. Unset -> the import-light reader returns False -> the unchanged Wave-3 path.
        from research.runners.onebrain_xedge_wave3_flags import xedge_in_wave3_enabled
        if xedge_in_wave3_enabled():
            from research.runners.onebrain_xedge_wave3 import get_wave3_xedge_pool
            return get_wave3_xedge_pool(seed)
        return get_wave3_pool(seed)
    from research.runners.onebrain_wave2_pool_production import wave2_pool_enabled, get_wave2_pool
    if min_wave <= 2 and wave2_pool_enabled():
        return get_wave2_pool(seed)
    from research.runners.onebrain_wave1_pool_production import wave1_pool_enabled, get_wave1_pool
    if min_wave <= 1 and wave1_pool_enabled():
        return get_wave1_pool(seed)
    return None
