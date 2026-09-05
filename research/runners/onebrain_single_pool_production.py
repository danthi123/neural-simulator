"""ONE-BRAIN SINGLE-POOL MERGE — production wiring (opt-in, DEFAULT-OFF).

Retire the TWO production merged pools for ONE shared spiking pool in the LIVE chat path:
  * pool #1 (`onebrain_merge_production.MergedSubstrate`)  = D2 SURPRISE + E2 WORLD-MODEL
  * pool #2 (`onebrain_merge_production2.MergedSubstrate2`) = E1 METACOG + D PRAGMATIC
->  ONE `merge_organs([surprise, worldmodel, metacog, pragmatic], wire=True)` pool (N=2034), behind the
    opt-in `BRAIN_ONEBRAIN_SINGLE_POOL` flag. All 4 core cortical organs' `shared=` point at this ONE pool.

DE-RISKED by `research/findings/2026-09-02-onebrain-twopool-merge-organ-read-GO.md` (all 4 organs read
BYTE-IDENTICALLY off ONE `merge_organs` pool — 6/6 seeds — both co-residence-invariant AND byte-identical to
the shipped 2 production pools, under the per-organ read-isolation protocol). That is the ISOLATION migration
gate; THIS module is the production flip it named as its next rung. The flip stays DEFAULT-OFF pending the
`webapp/server.py` brain-chat 6-seed regression soak (metacog + pragmatic are default-ON in live chat).

THE 5 RECONCILIATION SEAMS are supplied verbatim by `_recon_descriptors()` (reuse-by-import from the
organ-read verify — the SINGLE SOURCE OF TRUTH for the exact reconciled 4-organ family the 6-seed GO
validated, so the production pool == the validated pool with ZERO drift): (1) global `enable_hebbian_learning`
True + a per-synapse gain-0 FREEZE on every pool-2 (metacog/pragmatic) internal edge (`freeze_regions`);
(2) the param-het MASK on metacog/pragmatic only (`param_het=True`, global het OFF); (3) `hebbian_max_weight`
45 (pool-1's; pool-2 edges frozen => never clipped); (4) the per-region HOMEOSTASIS mask on every
surprise/world-model region (the SILENT world-model killer if dropped); (5) the full-snapshot read isolation
(the pool's post-build settle-to-rest `snap`, restored before each organ read — metacog/pragmatic already
consume `shared.snap`, surprise/world-model use `shared.read_isolation`).

NAMING NOTE (a genuine verify-first finding). The task named this `BRAIN_ONEBRAIN_MERGE`, but that env var
ALREADY EXISTS on `main` as pool #1's PAIRWISE merge flag (default-ON, `onebrain_merge_production.merge_enabled`),
and `BRAIN_ONEBRAIN_MERGE2` is pool #2's — reusing that name would silently CHANGE pool #1's semantics. So the
single-pool flip takes a DISTINCT name, `BRAIN_ONEBRAIN_SINGLE_POOL`, layered ABOVE the two pairwise flags.

DEFAULT-OFF + ADDITIVE: `BRAIN_ONEBRAIN_SINGLE_POOL` unset/0 -> `single_pool_enabled()` False -> every organ's
`get_organ()` takes its EXACT current two-pool path (byte-identical to before this module existed). NO `sim/`
edit; all state lives in `research/runners/`; the pool is the tiny numpy/cupy net the organ-read GO validated.
"""
from __future__ import annotations

import os

# De-risked (organ-read GO 6/6) but NOT flipped: the production default stays OFF until the brain-chat 6-seed
# regression soak (metacog + pragmatic default-ON in live chat) confirms answer-preservation through the LIVE
# organ read paths (which use the organs' OWN read isolation, not the verify's harness-driven full-restore).
_SINGLE_POOL_DEFAULT_ON = True


def single_pool_enabled() -> bool:
    """DEFAULT-ON per `_SINGLE_POOL_DEFAULT_ON` = True (flipped 2026-09-05: goal-b one-brain merge, de-risk GO 6/6 on current main; `BRAIN_ONEBRAIN_SINGLE_POOL=0` reverts byte-identical). `BRAIN_ONEBRAIN_SINGLE_POOL` in {1,true,yes,on} ->
    all 4 core cortical organs (surprise, world-model, metacog, pragmatic) share ONE `merge_organs` pool (the
    single-pool merge), retiring the two production pools for the turn. ABSENT/0 -> the current two-pool path.

    Layered ABOVE the two pairwise flags: when this is ON it WINS over `BRAIN_ONEBRAIN_MERGE`/`_MERGE2` (all four
    organs go on the ONE pool); when OFF, each organ resolves its `shared=` exactly as today (the pairwise flags)."""
    v = os.environ.get("BRAIN_ONEBRAIN_SINGLE_POOL")
    if v is None:
        return _SINGLE_POOL_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


# process-shared, built once per seed on first use (the same lifetime discipline as
# `onebrain_merge_production.get_merged_substrate` / `..2.get_merged_substrate2`), so ALL FOUR organs' get_organ()
# resolve the SAME pool object.
_POOL: dict = {}


def get_single_pool(seed: int = 42):
    """The process-shared single 4-organ merged pool (built once on first use, memoized by seed). ONE
    `merge_organs([surprise, worldmodel, metacog, pragmatic], wire=True)` spiking bridge with the 5 organ-read-GO
    reconciliation seams applied (see the module docstring). Every organ's `get_organ()` points its `shared=`
    here when `single_pool_enabled()`, so all 4 co-inhabit ONE bridge — the literal single-pool merge.

    `_recon_descriptors()` is imported from the organ-read verify runner (reuse-by-import; the codebase's standard
    single-source-of-truth pattern for a validated de-risk artefact) so the production pool is BUILT FROM THE EXACT
    reconciled family the 6-seed organ-read GO validated — no re-declaration, no drift. Lazy import: no import-time
    cost when the flag is off (the two-pool path never calls this)."""
    key = int(seed)
    if key not in _POOL:
        from research.runners.onebrain_merge_framework import merge_organs
        from research.runners._onebrain_twopool_merge_organread_verify import _recon_descriptors
        _POOL[key] = merge_organs(_recon_descriptors(), key, wire=True)
    return _POOL[key]
