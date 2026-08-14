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
"""
from __future__ import annotations

import os

import numpy as np

# reuse-by-import: metacog geometry + wiring helper + the shared GNW snapshot/settle machinery.
from research.runners._second_order_metacog_monitor_derisk import (
    ASSEMBLY_SIZE, K_CLASSES, WORKSPACE_FS_N, META_SIZE,
    WS_TO_FS_WEIGHT, FS_TO_WS_WEIGHT, WS_LOOP_GATE, DEFAULT_ATTRACTOR_WEIGHT, DEFAULT_NMDA_TAU,
)
from research.runners._gnw_rung1_ignition_curve_derisk import (
    _build_assembly_loop_population, _snapshot_state, SETTLE_STEPS,
)
# reuse-by-import: RSA geometry.
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


# metacog uses its production build parameters (must match MetacogProductionOrgan.ensure_built:
# build_metacog_bridge(confidence_read="balance") -> DEFAULT_ATTRACTOR_WEIGHT, DEFAULT_NMDA_TAU).
_METACOG_ATTRACTOR_W = float(DEFAULT_ATTRACTOR_WEIGHT)
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
        if self._built:
            return
        from sim.bridge import SimulationBridge
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.enums import NeuronModel
        from sim.backend import get_backend
        xp, _ = get_backend()

        regions, pathways = [], []
        if "metacog" in self.organs:
            r, p = self._metacog_specs()
            regions += r
            pathways += p
        if "pragmatic" in self.organs:
            r, p = self._pragmatic_specs()
            regions += r
            pathways += p

        # ── THE MERGED (or single-organ baseline) CONFIG SUPERSET. Globals replicate build_metacog_bridge /
        #    build_rsa_bridge where they matter; the additions are the THREE region-scoped merge seams + the region /
        #    pathway UNION. enable_nmda is ON iff metacog is present (its workspace / meta_schema slices are NMDA;
        #    build_metacog_bridge sets enable_nmda=True). Pragmatic's item / item_fs carry region enable_nmda=False, so
        #    the per-neuron NMDA mask zeroes their NMDA current even under a global enable_nmda=True -> byte-identical
        #    to build_rsa_bridge's enable_nmda=False. When only pragmatic is present, enable_nmda stays False. ──
        has_metacog = "metacog" in self.organs
        cfg = CoreSimConfig()
        cfg.seed = int(self.seed)
        cfg.heterogeneity_seed = int(self.seed)
        cfg.ou_seed = int(self.seed)
        cfg.per_region_parameter_heterogeneity = True    # merge seam #1 (Izhikevich param jitter, name-keyed)
        cfg.per_region_threshold_heterogeneity = True    # merge seam #2 (firing thresholds, name-keyed)
        cfg.per_region_wiring_seed = True                # merge seam #3 (sparse-pathway placement, order-invariant)
        cfg.dt_ms = 1.0
        cfg.num_traits = 1
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.connections_per_neuron = 0
        cfg.enable_brain_region_framework = True
        cfg.enable_nmda = bool(has_metacog)
        cfg.nmda_ratio = 0.5
        cfg.nmda_tau_decay = float(DEFAULT_NMDA_TAU)
        cfg.nmda_recurrent_tau_decay_ms = float(DEFAULT_NMDA_TAU)
        for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
                  "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process"):
            setattr(cfg, f, False)
        cfg.enable_parameter_heterogeneity = True         # both organs' graded rate codes REQUIRE it (seeded)
        cfg.stdp_w_max = max(400.0, _METACOG_ATTRACTOR_W * 4.0)
        cfg.hebbian_max_weight = max(400.0, _METACOG_ATTRACTOR_W * 4.0)
        cfg.brain_regions = regions
        cfg.region_pathways = pathways

        bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                  runtime_state=RuntimeState(), gpu_config=GPUConfig())
        bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        bridge.runtime_state.actual_seed_used = self.seed
        bridge._initialize_simulation_data(called_from_playback_init=False)

        rm = bridge.region_manager
        # ONE union: the framework wiring plan (both organs' declared pathways, sampled per-region-name-keyed via
        # per_region_seed=True -> order-invariant) + metacog's dense self-recurrent assembly loops. inject ONCE.
        union = dict(rm.build_wiring_plan(seed=int(self.seed), per_region_seed=True))
        if has_metacog:
            ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
            for k in range(K_CLASSES):
                member = ws[k * ASSEMBLY_SIZE:(k + 1) * ASSEMBLY_SIZE]
                union[f"loop_{k}"] = _build_assembly_loop_population(member, _METACOG_ATTRACTOR_W)
        inh = []
        for region in rm.regions():
            inh.extend(rm.inhibitory_indices(region.name))
        bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
        if has_metacog:
            bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)   # freeze the assembly loop (balance mode installs no META_GATE)

        # settle to a quiescent rest, snapshot ONCE (each organ's read restores this global snapshot -> read isolation).
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(SETTLE_STEPS):
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[:] = 0.0
        self.snap = _snapshot_state(bridge, xp)

        self.bridge = bridge
        self.cfg = cfg
        self.xp = xp
        self._built = True

    # ── per-organ read contexts (the slice indices each organ's real read path consumes) ─────────────────────────
    def metacog_idx(self):
        """The metacog organ's region->neuron-index map on the shared bridge, in `_run_trial`'s expected shape."""
        self.ensure_built()
        rm = self.bridge.region_manager
        xp = self.xp
        ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
        fs = np.asarray(rm.indices("workspace_fs"), dtype=np.int64)
        meta = np.asarray(rm.indices("meta_schema"), dtype=np.int64)
        member_idx = {k: ws[k * ASSEMBLY_SIZE:(k + 1) * ASSEMBLY_SIZE] for k in range(K_CLASSES)}
        meta_sub = META_SIZE // K_CLASSES
        meta_member_idx = {k: meta[k * meta_sub:(k + 1) * meta_sub] for k in range(K_CLASSES)}
        return {
            "member_dev": {k: xp.asarray(v) for k, v in member_idx.items()},
            "meta_dev": xp.asarray(meta),
            "meta_member_dev": {k: xp.asarray(v) for k, v in meta_member_idx.items()},
            "fs_dev": xp.asarray(fs),
            "confidence_read": "balance",
        }

    def pragmatic_item_dev(self):
        """The pragmatic (RSA) organ's 3 item-assembly index arrays on the shared bridge (`_rsa_recursion` shape)."""
        self.ensure_built()
        rm = self.bridge.region_manager
        xp = self.xp
        base = np.asarray(rm.indices("item"), dtype=np.int64)
        return {i: xp.asarray(base[i * RSA_ITEM_SIZE:(i + 1) * RSA_ITEM_SIZE]) for i in range(3)}


# The process-shared pool #2 substrate (built once on first use; holds BOTH organs).
_MERGED_SUBSTRATE2: "MergedSubstrate2 | None" = None


def get_merged_substrate2(seed: int = 42) -> MergedSubstrate2:
    """The process-shared metacog+pragmatic merged substrate (pool #2, both organs on one pool)."""
    global _MERGED_SUBSTRATE2
    if _MERGED_SUBSTRATE2 is None:
        _MERGED_SUBSTRATE2 = MergedSubstrate2(seed=seed, organs=("metacog", "pragmatic"))
    return _MERGED_SUBSTRATE2
