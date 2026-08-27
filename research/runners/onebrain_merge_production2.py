"""ONE-BRAIN MERGE — the SECOND production pool: the PARAM-HET-ON cluster on ONE shared spiking bridge.

THE GAP (rung after pool #1). Pool #1 (`onebrain_merge_production.py`, DEFAULT-ON 2026-08-13) put the D2 surprise +
E2 world-model production organs on ONE shared bridge, but ONLY those two — they run `enable_parameter_heterogeneity
=False`. The other three proven Gate-B organs (metacog / pragmatic / affect) REQUIRE `enable_parameter_heterogeneity=
True` for their graded rate codes, a DIFFERENT global config, so they cannot join pool #1 byte-identically
(`2026-08-13-onebrain-production-default-flip-SCOPED.md`). Their named next rung is a SECOND shared pool with EACH
OTHER.

THIS POOL. The two param-het-ON CORTICAL-MICROCIRCUIT organs — the E1 METACOG balance-of-evidence confidence monitor
(`metacog_production_organ`, `build_metacog_bridge`, confidence_read="balance") and the D-pragmatics SCALAR-IMPLICATURE
RSA belief organ (`pragmatic_production_organ`, `build_rsa_bridge` + `_rsa_recursion`) — onto ONE shared
`SimulationBridge` (one `cp_membrane_potential_v`). Both are param-het-ON, plasticity-OFF, OU-OFF, neuromod-OFF, fixed
frozen operating points, with DISJOINT region names (metacog: workspace / workspace_fs / meta_schema; pragmatic: item /
item_fs) and NO cross-synapse.

AFFECT IS SCOPED OUT (measured, structural — NOT a shortcut). The affect production organ builds a WHOLE co-resident
brain (`_stageA_full_integration_derisk.build_one_brain(with_faculties=True, co_resident_affect_ladder=True)`) whose
honesty relay defines regions NAMED `workspace` / `workspace_fs` / `meta_schema` — a HARD NAME COLLISION with metacog on
one `region_manager`. Renaming is not an option: every merge seam (per-region param-het / threshold / OU / wiring) keys
its name-invariant RNG on the region NAME (zlib.crc32), so a rename changes the slice's init + wiring and breaks
byte-identity to the standalone organ. Affect ALSO needs a GLOBAL `enable_ou_process=True` + the neuromodulator
subsystem, which the OU-off / neuromod-off microcircuits do not share in ONE cfg. So affect merges onto its OWN pool /
the recall-composer bridge (it is itself a "one brain"), a distinct rung — the same "flip the clean subset, map the rest
honestly" rule pool #1 followed. See `2026-08-13-onebrain-second-pool-*.md`.

BYTE-IDENTITY, why it is EXACT. The merge needs THREE region-scoped seams ON so each organ's slice is invariant to
co-residence: `per_region_parameter_heterogeneity` (name-keyed Izhikevich jitter), `per_region_threshold_heterogeneity`
(name-keyed firing thresholds), and — the seam pool #1 did NOT need — `per_region_wiring_seed` (each pathway's sparse
synapse placement drawn from a name-keyed RNG, so BOTH organs' pathways sample order-INVARIANTLY on the shared
`region_manager`; the rung named as "not yet exercised end-to-end in a two-fully-wired-organ merge" by
`2026-08-13-per-region-ou-wiring-affect-GO.md`, now exercised here). Each organ trains NOTHING (frozen operating points)
and reads ONLY its own regions; every read RESTORES the full global rest snapshot first (the reused
`_gnw_rung1._restore_state`), so a co-resident organ's transient firing during a read leaves NO footprint — read
isolation is INHERENT to the full-snapshot-restore protocol, no per-slice guard needed. With no cross-synapse and no
global per-step coupling (homeostasis / neuromod / OU all off), each slice's evolution depends only on its OWN reads ->
byte-identical to the standalone-with-flags (co-resident) organ.

GUARDED. `BRAIN_ONEBRAIN_MERGE2` gates this pool (independent of pool #1's `BRAIN_ONEBRAIN_MERGE`). Default per
`_MERGE2_DEFAULT_ON`. `BRAIN_ONEBRAIN_MERGE2=0` -> each organ builds its own bridge exactly as the pre-flip production.

NO NEW `sim/` behavior — the three region-scoped flags already exist on `main` (guarded, default-off;
`per_region_parameter_heterogeneity`, `per_region_threshold_heterogeneity`, `per_region_wiring_seed`). Reuse-by-import:
the region / pathway SPECS + the assembly-loop wiring are pulled from each de-risk builder; each production organ reads
its own slice on the shared bridge. Process backend (cupy in production, numpy in tests).

RETIRED-TO-A-SHIM (2026-08-27). `MergedSubstrate2.ensure_built()` no longer builds this bridge itself — it
DELEGATES to `onebrain_merge_framework.merge_organs([METACOG, PRAGMATIC-or-subset], seed, wire=True)`, the same
declarative engine every other registered organ (surprise/world-model + the 7 Group-A organs) builds through. The
config/wiring rationale documented above is now encoded in `onebrain_merge_framework.py`'s `METACOG`/`PRAGMATIC`
`OrganDescriptor`s (kept here as biology/architecture documentation, not as duplicated executable logic). This
class's public surface (`.bridge`/`.cfg`/`.xp`/`.snap`/`.metacog_idx()`/`.pragmatic_item_dev()`/`.ensure_built()`,
the `organs=` constructor for BOTH the 2-organ production combo and the 1-organ CORESIDENT-baseline combos its
other callers use) is preserved UNCHANGED — verified byte-identical (whole-bridge SHA1 hash + the real
`judge()`/`interpret()` production reads) against the pre-retirement bespoke build, 6/6 seeds, all three combos:
`_onebrain_merge2_retire_verify.py`, `research/findings/2026-08-27-merged-substrate2-retirement-framework-backed.md`.
"""
from __future__ import annotations

import os

# reuse-by-import: metacog geometry (still needed by `_metacog_specs`/`metacog_idx`, both kept as the SINGLE
# definition the framework's `_pool2_metacog_specs` reuses via a throwaway instance of this class).
from research.runners._second_order_metacog_monitor_derisk import (
    ASSEMBLY_SIZE, K_CLASSES, WORKSPACE_FS_N, META_SIZE, WS_TO_FS_WEIGHT, FS_TO_WS_WEIGHT,
)
# reuse-by-import: RSA geometry (still needed by `_pragmatic_specs`/`pragmatic_item_dev`).
from research.runners._recursive_tom_rsa_derisk import (
    RSA_ITEM_SIZE, RSA_FS_N, RSA_EXC_FS_W, RSA_FS_EXC_W,
)

from sim.regions import BrainRegion, RegionPathway


# PRODUCTION DEFAULT for pool #2 — DEFAULT-ON (2026-08-13). The genuine two-organ merge is byte-identical
# (merged == coresident-with-flags, 6/6) and BOTH organs are now answer-preserving vs pre-flip: PRAGMATIC was 6/6;
# METACOG became 6/6 once its confidence read was made robust to the per-region heterogeneity re-draw. The blocker
# was metacog's ABSOLUTE spike-rate margin sitting at the workspace's ~0.1%-firing noise floor (near-random,
# seed-fragile), which the per-region param/threshold seams reshuffled -> the confident/uncertain flip at mid-range
# evidence. RESOLVED by adopting the DIVISIVE-NORMALIZED NMDA-CONDUCTANCE balance read as the metacog production
# default (`metacog_production_organ.nmda_norm_margin`, GO `2026-08-13-metacog-robust-confidence-GO.md`): it tracks
# evidence monotonically in both the standalone and merged build, so the self-calibrated threshold lands at the same
# boundary -> the decision is invariant to the re-draw (answer-preservation 1/6 -> 6/6). `BRAIN_ONEBRAIN_MERGE2=0`
# (or false/no/off) is the escape to two separate bridges (byte-identical to the pre-flip production).
_MERGE2_DEFAULT_ON = True


def merge2_enabled() -> bool:
    """Production-DEFAULT (`_MERGE2_DEFAULT_ON`). `BRAIN_ONEBRAIN_MERGE2` in {1,true,yes,on} -> the metacog +
    pragmatic organs share ONE spiking bridge; in {0,false,no,off} -> each builds its own bridge (the escape,
    byte-identical to the pre-flip production); ABSENT -> the production default (`_MERGE2_DEFAULT_ON`)."""
    v = os.environ.get("BRAIN_ONEBRAIN_MERGE2")
    if v is None:
        return _MERGE2_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


_N_WS = ASSEMBLY_SIZE * K_CLASSES


class MergedSubstrate2:
    """ONE `SimulationBridge` holding the METACOG organ's regions and/or the PRAGMATIC (RSA) organ's regions, with the
    THREE region-scoped merge seams ON (`per_region_parameter_heterogeneity`, `per_region_threshold_heterogeneity`,
    `per_region_wiring_seed`). Built ONCE (lazily), then SHARED: each production organ reads its own region slice on
    `self.bridge`.

    `organs` selects which organs' regions are present — ("metacog", "pragmatic") for the real production merge, or a
    single-organ tuple for the byte-identity CO-RESIDENT baseline (an organ on its own bridge, same construction path,
    all three flags ON — so merged-vs-solo isolates the merge itself, exactly as pool #1's MergedSubstrate does)."""

    _METACOG_REGIONS = ("workspace", "workspace_fs", "meta_schema")
    _PRAGMATIC_REGIONS = ("item", "item_fs")

    def __init__(self, seed: int = 42, organs=("metacog", "pragmatic")):
        self.seed = int(seed)
        self.organs = tuple(organs)
        self.bridge = self.cfg = self.xp = self.snap = None
        self._pool = None    # the underlying framework MergedPool, once built (ensure_built delegates to it)
        self._built = False

    def _metacog_specs(self):
        regions = [
            BrainRegion(name="workspace", n_neurons=_N_WS, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
            BrainRegion(name="workspace_fs", n_neurons=WORKSPACE_FS_N, exc_fraction=0.0, internal_density=0.0,
                        enable_nmda=False),
            BrainRegion(name="meta_schema", n_neurons=META_SIZE, exc_fraction=1.0, internal_density=0.0,
                        enable_nmda=True),
        ]
        pathways = [
            RegionPathway(from_region="workspace", to_region="workspace_fs", density=0.5,
                          weight_mean=WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False),
            RegionPathway(from_region="workspace_fs", to_region="workspace", density=0.5,
                          weight_mean=FS_TO_WS_WEIGHT, weight_jitter=0.0, plastic=False),
        ]
        return regions, pathways

    def _pragmatic_specs(self):
        regions = [
            BrainRegion(name="item", n_neurons=RSA_ITEM_SIZE * 3, exc_fraction=1.0, internal_density=0.0,
                        enable_nmda=False),
            BrainRegion(name="item_fs", n_neurons=RSA_FS_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        ]
        pathways = [
            RegionPathway(from_region="item", to_region="item_fs", density=0.6, weight_mean=RSA_EXC_FS_W,
                          weight_jitter=0.0, plastic=False),
            RegionPathway(from_region="item_fs", to_region="item", density=0.6, weight_mean=RSA_FS_EXC_W,
                          weight_jitter=0.0, plastic=False),
        ]
        return regions, pathways

    def ensure_built(self):
        """DELEGATES to `onebrain_merge_framework.merge_organs()` — the declarative engine's `METACOG`/`PRAGMATIC`
        `OrganDescriptor`s encode exactly the geometry/config/wiring this method used to hand-build (reused
        BY CALLING `self._metacog_specs`/`self._pragmatic_specs` below, so there remains exactly ONE definition
        of the geometry). `wire=True` reproduces the ALWAYS-ON wiring inject (base pathways + metacog's dense
        self-recurrent assembly loops, per-region-seamed) this class used to build directly. Verified
        byte-identical to the pre-retirement bespoke build (this docstring's history, preserved in the module
        docstring above) for every `organs=` combination this class's callers use — 6/6 seeds, whole-bridge SHA1
        hash + the real `judge()`/`interpret()` production reads —
        `_onebrain_merge2_retire_verify.py` / `2026-08-27-merged-substrate2-retirement-framework-backed.md`."""
        if self._built:
            return
        from research.runners.onebrain_merge_framework import merge_organs, METACOG, PRAGMATIC
        descs = []
        if "metacog" in self.organs:
            descs.append(METACOG)
        if "pragmatic" in self.organs:
            descs.append(PRAGMATIC)
        pool = merge_organs(descs, seed=self.seed, wire=True)
        self._pool = pool
        self.bridge = pool.bridge
        self.cfg = pool.cfg
        self.xp = pool.xp
        self.snap = pool.snap
        self._built = True

    # ── per-organ read contexts (the slice indices each organ's real read path consumes) — dispatch to the
    #    framework pool's own idx_fn (the SAME computation this class used to run inline; see
    #    `onebrain_merge_framework._metacog_idx_fn`/`_pragmatic_idx_fn`), so there is exactly ONE definition. ──
    def metacog_idx(self):
        """The metacog organ's region->neuron-index map on the shared bridge, in `_run_trial`'s expected shape."""
        self.ensure_built()
        return self._pool.metacog_idx()

    def pragmatic_item_dev(self):
        """The pragmatic (RSA) organ's 3 item-assembly index arrays on the shared bridge (`_rsa_recursion` shape)."""
        self.ensure_built()
        return self._pool.pragmatic_item_dev()


# The process-shared pool #2 substrate (built once on first use; holds BOTH organs).
_MERGED_SUBSTRATE2: "MergedSubstrate2 | None" = None


def get_merged_substrate2(seed: int = 42) -> MergedSubstrate2:
    """The process-shared metacog+pragmatic merged substrate (pool #2, both organs on one pool)."""
    global _MERGED_SUBSTRATE2
    if _MERGED_SUBSTRATE2 is None:
        _MERGED_SUBSTRATE2 = MergedSubstrate2(seed=seed, organs=("metacog", "pragmatic"))
    return _MERGED_SUBSTRATE2
