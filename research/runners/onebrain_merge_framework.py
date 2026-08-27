"""ONE-BRAIN MERGE FRAMEWORK — a DECLARATIVE, batched N-organ merge engine (PROTOTYPE SKELETON).

DESIGN DOC: research/findings/2026-08-27-onebrain-merge-framework-DESIGN.md (read it first — this file is the
§7 skeleton, not the migration). It packages the merge that is done BESPOKE today (one hand-written
`MergedSubstrate` per pool in onebrain_merge_production{,2}.py) as ONE `OrganDescriptor` registry + ONE
`merge_organs` engine, so registering an organ becomes a data ROW, not a class. The generic form is already
proven for 4 organs on the `research/onebrain-twopool-merge` branch (`build_pool`, 6/6 GO substrate-init +
organ-read smoke GO); this skeleton demonstrates the schema on the two on-`main` pool-#1 organs and proves
one round-trips BYTE-IDENTICALLY against the shipped `MergedSubstrate`.

SCOPE (honest): 2 organs registered (surprise, world-model), a numpy INIT-byte-identity smoke. NOT the full
migration, NOT the metacog/pragmatic config-conflict reconciliation (that is the twopool branch's per-region
seams), NOT the functional-integration gate (design §4). NO sim/ edit; reuse-by-import; the smoke runs on the
numpy backend (tiny nets, no GPU).

Run the smoke (CPU, bit-exact):
    SIM_BACKEND=numpy python -m research.runners.onebrain_merge_framework --smoke
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# reuse-by-import: the SAME de-risk builders the shipped MergedSubstrate uses for spec extraction, and the
# SAME topographic-wiring + index/host helpers. Nothing here is new mechanism.
from research.runners._spiking_expectation_rpe_derisk import (
    build_expectation_circuit, _install_block_diagonal, _idx, _host,
)
from research.runners._affective_world_model_derisk import build_world_model_circuit
from research.runners.onebrain_merge_production import _SURPRISE_KW, _WORLDMODEL_KW


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  1. THE DESCRIPTOR SCHEMA — the minimal declarative record to register any organ for merge.
# ─────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class OrganDescriptor:
    """Everything the bespoke MergedSubstrate did BY HAND for one organ, as data. See DESIGN §1 for the
    field->bespoke-code mapping. A trivial organ needs only key/regions/spec_fn/config; the rest default off."""
    key: str                                   # stable id; keys the per-region name RNG + the isolation mask
    regions: tuple                             # region NAMES this organ owns (DISJOINT across the pool)
    spec_fn: Callable                          # seed -> (regions, pathways, meta); reuse the de-risk builder
    config: dict = field(default_factory=dict) # cfg field -> value REQUIRED (unioned; a clash -> MergeConflict)
    region_flags: dict = field(default_factory=dict)   # {"workspace": {"enable_nmda": True}} per-region overrides
    post_build: Callable = None                # (bridge, meta) -> None; topographic wiring AFTER init
    freeze_regions: tuple = ()                 # regions whose INTERNAL edges get cp_plasticity_rate_gain=0
    isolation: str = "per_slice"               # "per_slice" | "full_snapshot" (the two existing protocols)
    idx_fn: Callable = None                    # (bridge) -> the idx map the organ's shared= read path consumes
    explicit_wiring_fn: Callable = None        # (bridge, region_manager) -> dict of extra wiring populations UNIONED
                                               #   into the pool's ONE per-region-seamed inject (assembly loops /
                                               #   member->attend / block-diagonal edges that are NOT plain pathways)
    post_inject_fn: Callable = None            # (bridge) -> None; runs AFTER the wiring inject (e.g. set_plasticity_gate)
    organ_cls: type = None                     # the shipped *_ProductionOrgan (constructed with shared=<pool>)
    read_fn: Callable = None                   # (organ_instance) -> dict of numeric reads (organ-read byte battery)
    answer_fn: Callable = None                 # (organ_instance) -> the rendered chat answer(s) (answer-preservation)
    supports_shared: bool = False              # True == the shipped class runs (with a `shared=` kwarg) on a MergedPool
    param_het: bool = False                    # organ's standalone uses param-het -> reconcile via the name-keyed
                                               #   per-region seam (global OFF + enable_heterogeneity on ITS regions)
    scaffold_residuals: tuple = ()             # host-scaffold flagged for self-organization burn-down (DESIGN §6)
    defer_reason: str = ""                     # non-empty == Group-B/C deferred; why (the engine seam it needs)


class MergeConflict(ValueError):
    """Two descriptors REQUIRE a config key at different values — a genuine global-config incompatibility
    (param-het ON vs OFF; OU on vs off). Raised at BUILD so it fails loudly at registration, never silently
    corrupts a slice. The reconciliation is a per-region seam (see the twopool branch), declared per organ."""


# spec extraction re-runs an organ's de-risk builder (a throwaway bridge just to read its BrainRegion specs),
# which is the batched verify's dominant cost. Cache the raw specs per (key, seed); deepcopy on use so the
# per-region flag masking in ensure_built mutates a FRESH copy, never the cached originals.
_SPEC_CACHE: dict = {}


def _cached_spec(descriptor, seed):
    ck = (descriptor.key, int(seed))
    if ck not in _SPEC_CACHE:
        _SPEC_CACHE[ck] = descriptor.spec_fn(int(seed))
    r, p, m = _SPEC_CACHE[ck]
    return copy.deepcopy(list(r)), copy.deepcopy(list(p)), m


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  2. THE MERGE ENGINE — merge_organs([descriptors], seed) -> MergedPool.
# ─────────────────────────────────────────────────────────────────────────────────────────────
# The engine-UNIVERSAL config (dt / model / profile / seeds / the always-on per-region seam + framework).
# Everything organ-family-specific (hebbian block, gabab, the disable flags) is declared PER DESCRIPTOR and
# UNIONED, so a new family is a descriptor's `config`, not an engine edit.
def _base_config(seed: int, legacy: bool = False):
    from sim.config import CoreSimConfig
    from sim.enums import NeuronModel
    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    # merge seam #1 — the name-keyed firing-threshold draw. ON: a region's thresholds key on its NAME
    # (crc32), so a slice is co-residence + RNG-order invariant (the substrate-init byte-identity property).
    # legacy -> OFF (the DISCRIMINATOR): the global-RNG threshold draw depends on total-N + build order, so
    # merged-vs-coresident DIVERGE -> proves the seam-ON byte-identity is NOT vacuous.
    cfg.per_region_threshold_heterogeneity = not legacy
    return cfg


class MergedPool:
    """The shared N-organ substrate. Exposes exactly the surface a shipped organ's `shared=` path expects:
    .bridge/.cfg/.xp, .ensure_built(), .read_isolation(key), and per-organ idx accessors dispatched by key."""

    _PER_NEURON_STATE = (
        "cp_membrane_potential_v", "cp_recovery_variable_u",
        "cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab", "cp_conductance_g_nmda",
        "cp_firing_states", "cp_prev_firing_states", "cp_refractory_timers", "cp_refractory",
        "cp_neuron_firing_thresholds", "cp_neuron_activity_ema", "cp_external_input_current",
    )

    def __init__(self, seed, descriptors, config_descriptors=None, legacy=False, force_het_off=False, wire=False):
        self.seed = int(seed)
        # wire=True == the ORGAN-READ substrate: after the base build, rebuild cp_connections from ONE
        # per-region-seamed wiring plan (`build_wiring_plan(per_region_seed=True)`, co-residence + order INVARIANT)
        # UNIONed with every descriptor's explicit_wiring_fn (assembly loops / member->attend / block-diagonals),
        # then settle-to-rest + snapshot (`self.snap`). Both the MERGED and the CORESIDENT pools take this same
        # path, so an organ's slice + its read is byte-identical merged-vs-coresident. Left OFF for the
        # substrate-INIT gate (init arrays are inject-invariant, so that gate needs no wiring).
        self.wire = bool(wire)
        self.snap = None
        self.descriptors = list(descriptors)                       # which regions the pool INSTANTIATES
        # which descriptors' `config` dicts UNION into the global config. For the MERGED pool this is the
        # same list; for a CORESIDENT baseline it is the FULL registry, so a solo organ sits on the SAME
        # superset config as the merged pool -> a non-zero slice delta isolates CO-RESIDENCE, not config.
        self.config_descriptors = list(config_descriptors) if config_descriptors is not None else list(descriptors)
        self.legacy = bool(legacy)
        # LOAD-BEARING control: same reconciled config but CLEAR the per-region param-het mask, so a param-het
        # organ's izh params revert to the non-jittered preset. A non-zero delta vs the normal merged pool proves
        # the mask is genuinely DOING WORK (the reconciliation is not a vacuous all-zero het).
        self.force_het_off = bool(force_het_off)
        self._by_key = {d.key: d for d in descriptors}
        self.bridge = self.cfg = self.xp = None
        self.meta = {}
        self.organ_regions = {}
        self._keep_mask_cache = {}
        self._built = False

    def ensure_built(self):
        if self._built:
            return
        from sim.bridge import SimulationBridge
        from sim.config import RuntimeState, GPUConfig, VisualizationConfig
        from sim.backend import get_backend
        xp, _ = get_backend()

        # (1) SPEC EXTRACTION — reuse-by-import; UNION regions/pathways in descriptor order. Every seam keys on
        #     region NAME (crc32), so per-neuron init is co-residence + RNG-order invariant (the whole point).
        regions, pathways = [], []
        owner = {}                                  # region-name -> owning descriptor key
        het_regions = set()                         # regions whose organ opts into the param-het seam
        self.organ_regions = {}                     # descriptor key -> [region names it owns] (build-discovered)
        for d in self.descriptors:
            r, p, m = _cached_spec(d, self.seed)
            self.organ_regions[d.key] = [rg.name for rg in r]
            for rg in r:
                if rg.name in owner:
                    raise MergeConflict(
                        f"region-name collision {rg.name!r}: {owner[rg.name]} vs {d.key} (rename forbidden — "
                        f"the seams key on the name; a shared name changes a slice's init -> own-pool, DESIGN §5)")
                owner[rg.name] = d.key
                if d.param_het:
                    het_regions.add(rg.name)
            regions += list(r); pathways += list(p); self.meta[d.key] = m

        # (3) CONFIG UNION — base + each config-descriptor's requirements; a key at two values is a real
        #     conflict. config_descriptors (not descriptors) so a coresident baseline unions the FULL registry.
        cfg = _base_config(self.seed, legacy=self.legacy)
        union, provenance = {}, {}
        for d in self.config_descriptors:
            for k, v in d.config.items():
                if k in union and union[k] != v:
                    raise MergeConflict(f"{k!r}: {provenance[k]}={union[k]!r} vs {d.key}={v!r}")
                union[k] = v; provenance[k] = d.key
        for k, v in union.items():
            setattr(cfg, k, v)
        # PARAM-HET SEAM — an organ whose standalone uses param-het reconciles it the twopool way: GLOBAL
        #   enable_parameter_heterogeneity OFF + name-keyed per-region draw ON, masked to ITS regions only. The
        #   name-keyed draw is co-residence-invariant, so the masked slice carries byte-identical het merged-vs-solo.
        #   The GLOBAL seam flag keys on config_descriptors (so a coresident baseline sets the SAME global config as
        #   the merged pool); the per-region MASK (het_regions) keys on the instantiated descriptors.
        het_config = any(getattr(d, "param_het", False) for d in self.config_descriptors)
        if het_config and not self.legacy:
            cfg.enable_parameter_heterogeneity = False
            cfg.per_region_parameter_heterogeneity = True

        if self.legacy:
            # DISCRIMINATOR: force the name-keyed seams OFF regardless of what a descriptor declared, so a
            # co-resident slice's init draws depend on total-N + build order and DIVERGE from the solo slice.
            cfg.per_region_threshold_heterogeneity = False
            cfg.per_region_parameter_heterogeneity = False
            cfg.per_region_wiring_seed = False

        # (4) PER-REGION FLAGS — the diffbuilder pattern: reconcile a would-be global conflict into a masked one.
        for rg in regions:
            if rg.name in het_regions and not self.legacy and not self.force_het_off:
                rg.enable_heterogeneity = True            # opt this slice into the name-keyed param-het draw
        for d in self.descriptors:
            for rname, flags in (d.region_flags or {}).items():
                for rg in regions:
                    if rg.name == rname:
                        for fk, fv in flags.items():
                            setattr(rg, fk, fv)

        cfg.brain_regions = regions
        cfg.region_pathways = pathways

        # (5) BUILD — one bridge; per-region seeding is already generic (reads region names).
        bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                  runtime_state=RuntimeState(), gpu_config=GPUConfig())
        bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        bridge.runtime_state.actual_seed_used = self.seed
        bridge._initialize_simulation_data(called_from_playback_init=False)

        # (5b) ORGAN-READ WIRING (wire=True only) — ONE order-INVARIANT inject that replaces cp_connections with the
        #      base pathways REGENERATED per-region-seamed (`build_wiring_plan(per_region_seed=True)`, so every edge
        #      keys on its endpoints' NAMES, not build order) UNION each organ's explicit_wiring_fn (assembly loops /
        #      member->attend). The CORESIDENT pool runs the identical path, so a slice's WEIGHTS (not just its init
        #      arrays) are byte-identical merged-vs-coresident -- the precondition the organ-read reads need. Skipped
        #      for the substrate-init gate (cp_connections does not enter those init arrays).
        if self.wire:
            self._install_organ_read_wiring(bridge, xp)

        # (6) POST-BUILD WIRING — block-diagonal / assembly loops, in descriptor order (weights, not init arrays).
        for d in self.descriptors:
            if d.post_build is not None:
                d.post_build(bridge, self.meta[d.key])

        # (7) GAIN-0 FREEZE — every edge with BOTH endpoints in a frozen region; assert NO edge has EXACTLY one
        #     (an unintended cross-synapse). Generic form promoted from organread_verify (DESIGN §2 step 7).
        frozen = set()
        for d in self.descriptors:
            frozen |= set(d.freeze_regions)
        if frozen:
            self._apply_gain0_freeze(bridge, frozen, xp)

        # (8) SNAPSHOT for the per_slice organs' hard resets.
        bridge._rest_v = bridge.cp_membrane_potential_v.copy()
        bridge._rest_u = bridge.cp_recovery_variable_u.copy()

        # (9) ORGAN-READ REST SNAPSHOT (wire=True) — settle every region to a true quiescent rest under zero input,
        #     then snapshot the full per-neuron state (the EMERGE-61 / GNW Rung-1 wash-out an organ's read restores
        #     before each trial). Deterministic (the pool config has OU + conductance noise OFF), so the snapshot's
        #     per-organ slice is byte-identical merged-vs-coresident (a slice's dynamics never couple across organs
        #     -- zero cross-synapses). Exposed as `self.snap` for a `shared=` organ whose read restores it.
        if self.wire:
            from research.runners._gnw_rung1_ignition_curve_derisk import _snapshot_state, SETTLE_STEPS
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(SETTLE_STEPS):
                bridge._run_one_simulation_step()
            bridge.cp_external_input_current[:] = 0.0
            self.snap = _snapshot_state(bridge, xp)

        self.bridge, self.cfg, self.xp, self._built = bridge, cfg, xp, True

    def _install_organ_read_wiring(self, bridge, xp):
        """The wire=True inject: rebuild cp_connections from the per-region-seamed base plan UNION every descriptor's
        explicit wiring, then run each descriptor's post_inject_fn (plasticity gates). Mirrors the standalone build of
        an inject-organ (self_schema / MergedSubstrate4) but for the WHOLE pool, so every organ's slice is wired
        order-invariantly and identically whether it is alone or co-resident."""
        rm = bridge.region_manager
        union = dict(rm.build_wiring_plan(seed=self.seed, per_region_seed=True))
        for d in self.descriptors:
            if d.explicit_wiring_fn is not None:
                extra = d.explicit_wiring_fn(bridge, rm)
                if extra:
                    union.update(extra)
        inh = []
        for region in rm.regions():
            inh.extend(rm.inhibitory_indices(region.name))
        bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
        for d in self.descriptors:
            if d.post_inject_fn is not None:
                d.post_inject_fn(bridge)

    def _apply_gain0_freeze(self, bridge, frozen_regions, xp):
        idx = set()
        for name in frozen_regions:
            idx |= set(int(i) for i in _idx(bridge, name))
        arr = np.asarray(sorted(idx), dtype=np.int64)
        coo = bridge.cp_connections.tocoo()
        row = np.asarray(_host(coo.row)); col = np.asarray(_host(coo.col))
        row_in = np.isin(row, arr); col_in = np.isin(col, arr)
        cross = row_in ^ col_in
        if bool(cross.any()):
            raise MergeConflict(f"{int(cross.sum())} edge(s) with exactly one endpoint in a frozen region "
                                f"(an unintended cross-synapse) -- the no-cross premise broke")
        in_frozen = row_in & col_in
        nnz = int(bridge.cp_connections.nnz)
        if bridge.cp_plasticity_rate_gain is None:
            bridge.cp_plasticity_rate_gain = xp.ones(nnz, dtype=xp.float32)
        g = np.asarray(_host(bridge.cp_plasticity_rate_gain)).copy()
        g[in_frozen] = 0.0
        bridge.cp_plasticity_rate_gain = xp.asarray(g, dtype=xp.float32)

    # ── per-organ idx accessors, dispatched by key to the descriptor's idx_fn ──
    def idx(self, key):
        self.ensure_built()
        return self._by_key[key].idx_fn(self.bridge)

    # The shipped surprise organ calls `shared.surprise_idx_map()`; provide it by dispatch so the UNMODIFIED
    # organ class works against this pool (the injection contract is already universal).
    def surprise_idx_map(self):
        return self.idx("surprise")

    def _keep_mask(self, active):
        if active not in self._keep_mask_cache:
            self.ensure_built()
            n = int(self.bridge.cp_membrane_potential_v.shape[0])
            m = self.xp.zeros(n, dtype=bool)
            for r in self._by_key[active].regions:
                m[self.xp.asarray(_idx(self.bridge, r))] = True
            self._keep_mask_cache[active] = m
        return self._keep_mask_cache[active]

    def read_isolation(self, active):
        """Name-keyed over N organs (identical mechanism to onebrain_merge_production.MergedSubstrate): snapshot
        the full per-neuron state, let `active`'s slice evolve, restore every OTHER organ's slice at the end."""
        import contextlib
        @contextlib.contextmanager
        def _guard():
            b = self.bridge
            snaps = [(nm, getattr(b, nm).copy() if getattr(b, nm, None) is not None else None)
                     for nm in self._PER_NEURON_STATE]
            try:
                yield
            finally:
                keep = self._keep_mask(active)
                for nm, snap in snaps:
                    if snap is None:
                        continue
                    setattr(b, nm, self.xp.where(keep, getattr(b, nm), snap))
        return _guard()


def merge_organs(descriptors, seed: int = 42, config_descriptors=None, legacy: bool = False,
                 force_het_off: bool = False, wire: bool = False) -> MergedPool:
    """Build ONE shared spiking bridge holding all `descriptors`' regions. The N-organ generalization of the
    bespoke per-pool MergedSubstrate (DESIGN §2). Returns a MergedPool ready for `desc.organ_cls(shared=pool)`.

    config_descriptors -> whose `config` dicts union into the global cfg (default: `descriptors`). Pass the
    FULL registry to build a CORESIDENT baseline (one organ's regions on the merged pool's superset config).
    legacy=True -> the DISCRIMINATOR (name-keyed seams OFF; merged-vs-coresident should diverge).
    force_het_off=True -> the LOAD-BEARING control (per-region param-het mask cleared).
    wire=True -> the ORGAN-READ substrate (one per-region-seamed wiring inject + settle-to-rest snapshot; needed
    when a `shared=` organ actually RUNS its read/judge pipeline on the pool, not just the substrate-init gate)."""
    pool = MergedPool(seed, descriptors, config_descriptors=config_descriptors, legacy=legacy,
                      force_het_off=force_het_off, wire=wire)
    pool.ensure_built()
    return pool


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  SUBSTRATE-INIT byte-identity — the MIGRATION-SAFETY gate that scales to ANY registered organ
#  (needs only spec_fn+config; NO shared= plumbing). Promoted from _onebrain_twopool_merge_derisk.byte_identity.
# ─────────────────────────────────────────────────────────────────────────────────────────────
# every per-neuron INIT array a merge seam could perturb (the two gate masks included).
_INIT_ARRAYS = (
    "cp_neuron_firing_thresholds", "cp_membrane_potential_v", "cp_recovery_variable_u",
    "cp_izh_a", "cp_izh_b", "cp_izh_C", "cp_izh_c_reset", "cp_izh_d_increment",
    "cp_izh_vpeak", "cp_izh_vt", "cp_izh_vr",
    "cp_heterogeneity_neuron_mask", "cp_homeostasis_neuron_mask",
)
# a gate mask is None when NO region opts in; semantically None == all-False, so coerce before slicing
# (else merged (mask present, this slice False) vs solo (mask None) would spuriously read as a mismatch).
_MASK_ARRAYS = ("cp_heterogeneity_neuron_mask", "cp_homeostasis_neuron_mask")


def _region_indices(bridge, name):
    return np.asarray(sorted(int(i) for i in _idx(bridge, name)), dtype=np.int64)


def _slice_arrays(bridge, idx):
    n = int(_host(bridge.cp_membrane_potential_v).shape[0])
    out = {}
    for a in _INIT_ARRAYS:
        arr = _host(getattr(bridge, a, None))
        if arr is None and a in _MASK_ARRAYS:
            arr = np.zeros(n, dtype=np.float64)
        out[a] = None if arr is None else np.asarray(arr)[idx]
    return out


def _maxerr(x, y):
    if x is None and y is None:
        return 0.0, "both-none"
    if (x is None) != (y is None):
        return float("inf"), "one-none"
    x = x.astype(np.float64); y = y.astype(np.float64)
    if x.shape != y.shape:
        return float("inf"), f"shape {x.shape}!={y.shape}"
    return (float(np.max(np.abs(x - y))) if x.size else 0.0), "ok"


def substrate_byte_identity(merged: MergedPool, coresident: MergedPool, regions) -> dict:
    """Per-array max |delta| between an organ's region slice in the MERGED pool vs the same organ's slice in
    its CORESIDENT pool (alone, same superset config). 0.0 over every region+array == migration byte-identity."""
    organ_max, detail = 0.0, {}
    for rname in regions:
        mi = _region_indices(merged.bridge, rname)
        si = _region_indices(coresident.bridge, rname)
        if mi.size != si.size:
            detail[rname] = {"maxerr": float("inf"), "note": f"size {mi.size}!={si.size}"}
            organ_max = float("inf"); continue
        ma = _slice_arrays(merged.bridge, mi)
        sa = _slice_arrays(coresident.bridge, si)
        rmax, worst = 0.0, None
        for a in _INIT_ARRAYS:
            e, note = _maxerr(ma[a], sa[a])
            if e > rmax:
                rmax, worst = e, (a, note)
        detail[rname] = {"maxerr": rmax, "worst": worst}
        organ_max = max(organ_max, rmax)
    return {"maxerr": organ_max, "regions": detail}


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Reuse-by-import spec/wiring/idx callables for the two registered organs (pool #1 family).
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _surprise_spec(seed):
    _br, cfgS, metaS = build_expectation_circuit(seed, per_region_thresh=True, **_SURPRISE_KW)
    return list(cfgS.brain_regions), list(cfgS.region_pathways), metaS


def _worldmodel_spec(seed):
    _br, cfgW, metaW = build_world_model_circuit(seed, **_WORLDMODEL_KW)
    return list(cfgW.brain_regions), list(cfgW.region_pathways), metaW


def _surprise_post_build(bridge, meta):
    blk = meta["blk"]
    _install_block_diagonal(bridge, "patient_asserted", "surprise", blk, meta["W_exc"])
    _install_block_diagonal(bridge, "patient_expected", "surprise", blk, meta["W_inh"])
    _install_block_diagonal(bridge, "cue", "patient_expected", blk, float(_SURPRISE_KW["cue_to_expected_weight"]))
    bridge._blk = blk


def _name_idx(bridge, names):
    from sim.backend import get_backend
    xp, _ = get_backend()
    return {n: xp.asarray(_idx(bridge, n)) for n in names}


def _cfg_of(obj):
    """Find the CoreSimConfig on whatever a builder returned/produced: the object itself, a `.cfg`/`.core_config`
    attribute, or (for a bridge) `.core_config`. Returns None if none carries `brain_regions`."""
    if obj is None:
        return None
    if hasattr(obj, "brain_regions") and hasattr(obj, "region_pathways"):
        return obj
    for attr in ("cfg", "core_config", "config"):
        c = getattr(obj, attr, None)
        if c is not None and hasattr(c, "brain_regions"):
            return c
    return None


def _spec_from_builder(builder, **kw):
    """Adapt an organ's existing de-risk builder into a descriptor `spec_fn` (reuse-by-import; no new mechanism).
    Robust to every return shape a builder here uses: `(bridge, cfg, meta)`, `(bridge, cfg)`, a bare cfg, or a
    class instance whose `.bridge`/`.cfg`/`.core_config` carries the config. spec_fn(seed) -> (regions, pathways, meta)."""
    def spec(seed):
        out = builder(int(seed), **kw)
        elts = list(out) if isinstance(out, tuple) else [out]
        cfg = None
        for e in elts:                                             # a returned cfg / bridge / instance ...
            cfg = _cfg_of(e) or (_cfg_of(getattr(e, "bridge", None)) if e is not None else None)
            if cfg is not None:
                break
        if cfg is None:
            raise MergeConflict(f"{getattr(builder, '__name__', builder)}: no CoreSimConfig with brain_regions "
                                f"in the builder's return -- give this organ a custom spec_fn")
        meta = next((e for e in elts if isinstance(e, dict)), {})
        return list(cfg.brain_regions), list(cfg.region_pathways), meta
    return spec


def _spec_from_instance(factory):
    """spec_fn for an organ whose 'builder' is a CLASS constructed with args (e.g. ProvenanceBrain(seed)); `factory`
    is a callable seed -> object, and we read the config off the object or its `.bridge`/`.sb`/`._bridge`."""
    def spec(seed):
        obj = factory(int(seed))
        cfg = _cfg_of(obj)
        for attr in ("bridge", "sb", "_bridge"):
            if cfg is None:
                cfg = _cfg_of(getattr(obj, attr, None))
        if cfg is None:
            raise MergeConflict("instance builder exposed no CoreSimConfig with brain_regions")
        return list(cfg.brain_regions), list(cfg.region_pathways), {}
    return spec


# The pool-#1 organ-family config (surprise + world-model share it verbatim -> a clean union). Mirrors
# onebrain_merge_production.MergedSubstrate.ensure_built's cfg block so the merged bridge is byte-identical.
_POOL1_CONFIG = {
    "per_region_homeostasis_isolation": True,
    "enable_stdp": False, "enable_hebbian_learning": True, "hebbian_learning_rate": 0.06,
    "hebbian_min_weight": 0.0, "hebbian_max_weight": 45.0, "hebbian_weight_decay": 0.0,
    "hebbian_rate_window": True, "hebbian_coactivity_decay": 0.85, "hebbian_coactivity_thresh": 0.20,
    "hebbian_mean_subtract": 1.0, "enable_reward_modulation": False, "enable_short_term_plasticity": False,
    "enable_structural_plasticity": False, "enable_parameter_heterogeneity": False,
    "enable_ou_process": False, "enable_conductance_noise": False,
    "current_reward_signal": 0.0, "reward_baseline": 0.0,
    "enable_gabab": True, "gabab_reversal_potential": -90.0, "gabab_tau_decay": 150.0,
    "gabab_propagation_strength": 0.22, "gabab_conductance_max": 0.0,
}

SURPRISE = OrganDescriptor(
    key="surprise",
    regions=("cue", "patient_expected", "patient_asserted", "surprise"),
    spec_fn=_surprise_spec,
    config=_POOL1_CONFIG,
    post_build=_surprise_post_build,
    isolation="per_slice",
    idx_fn=lambda b: _name_idx(b, ("cue", "patient_expected", "patient_asserted", "surprise")),
)

WORLDMODEL = OrganDescriptor(
    key="worldmodel",
    regions=("state", "pred_pos", "pred_neg", "obs_pos", "obs_neg", "surprise_pos", "surprise_neg"),
    spec_fn=_worldmodel_spec,
    config=_POOL1_CONFIG,
    post_build=None,
    isolation="per_slice",
    idx_fn=lambda b: _name_idx(b, ("state", "pred_pos", "pred_neg", "obs_pos", "obs_neg",
                                   "surprise_pos", "surprise_neg")),
)

# ─────────────────────────────────────────────────────────────────────────────────────────────
#  GROUP A — the declarative-NOW organs (DESIGN §5). Each is a registry ROW: a `spec_fn` that reuses the
#  organ's OWN de-risk builder (throwaway bridge, we read its BrainRegion specs) + `param_het` where the
#  organ's standalone uses parameter-heterogeneity (reconciled by the name-keyed per-region seam).
#  ORGAN-READ status (2026-08-27): two of these — self_schema + d6_multiref_wm — now take a `shared=` kwarg and
#  RUN their real read pipeline on the wired pool byte-identically (supports_shared=True; the descriptors carry
#  organ_cls/read_fn/answer_fn + idx_fn + explicit_wiring_fn/region_flags). The other five are substrate-init GO
#  but their read still needs a seam (GROUP_A_ORGANREAD_DEFERRED names each) — for those the batched gate stays the
#  SUBSTRATE-INIT co-residence-invariance migration gate.
#
#  Builders are imported LAZILY inside each spec_fn (avoid a heavy import at module load + circular imports).
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _spec_causal_whatif(seed):
    from research.runners._causal_forward_model_derisk import build_forward_model
    _b, cfg, meta = build_forward_model(int(seed))
    return list(cfg.brain_regions), list(cfg.region_pathways), meta


def _spec_comprehension(seed):
    from research.runners._spiking_comprehension_monitor_derisk import _build_comp
    comp = _build_comp(int(seed))
    cfg = _cfg_of(comp) or _cfg_of(getattr(comp, "bridge", None))
    return list(cfg.brain_regions), list(cfg.region_pathways), {}


def _spec_self_schema(seed):
    from research.runners._self_schema_region_derisk import build_self_schema_bridge
    bridge = build_self_schema_bridge(seed=int(seed))[0]
    cfg = _cfg_of(bridge)
    return list(cfg.brain_regions), list(cfg.region_pathways), {}


def _spec_source_provenance(seed):
    from research.runners._laneC_source_provenance_opponent_derisk import ProvenanceBrain
    pb = ProvenanceBrain(int(seed))
    cfg = _cfg_of(getattr(pb, "_bridge", None)) or _cfg_of(pb)
    return list(cfg.brain_regions), list(cfg.region_pathways), {}


def _spec_curiosity(seed):
    from research.runners._curiosity_seek_learn_onbridge_derisk import build_curiosity_bridge
    out = build_curiosity_bridge(int(seed), 4)
    cfg = _cfg_of(out[1])
    return list(cfg.brain_regions), list(cfg.region_pathways), {}


def _spec_prospective_memory(seed):
    from research.runners._pmem_sfa_nmda_amplifier_derisk import SFANmdaProspectiveMemory
    pm = SFANmdaProspectiveMemory(["A", "B"], ["d0", "d1", "d2", "d3"], seed=int(seed))
    cfg = _cfg_of(getattr(pm, "bridge", None)) or _cfg_of(pm)
    return list(cfg.brain_regions), list(cfg.region_pathways), {}


def _spec_d6_multiref_wm(seed):
    from research.runners._multi_slot_binding_derisk import MultiSlotHold
    hold = MultiSlotHold(int(seed), 5, 6)
    cfg = _cfg_of(getattr(hold, "sb", None)) or _cfg_of(hold)
    return list(cfg.brain_regions), list(cfg.region_pathways), {}


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  GROUP-A ORGAN-READ plumbing — the `shared=` read surface for the two organs whose read pipeline is a
#  CLEAN function of the substrate (frozen plasticity, no neuromodulator subsystem, config-compatible):
#  DR-3 self_schema authorship + D6 multi-referent WM. Each descriptor supplies: a config (its cfg needs,
#  UNIONed compatibly), an idx_fn (region-name -> the dev-index map its read consumes), and for the
#  inject-organ (self_schema) an explicit_wiring_fn + post_inject_fn (assembly loops / member->attend /
#  the frozen loop gate). The read_fn/answer_fn run the organ's REAL production read battery.
# ─────────────────────────────────────────────────────────────────────────────────────────────
# config UNIONs (compatible: both frozen + NMDA-on; only self_schema declares nmda_ratio, only d6 the
# nmda_recurrent slow-hold, so the keys they share (enable_nmda + the plasticity-OFF flags) agree exactly).
# enable_conductance_noise + enable_ou_process draw from a SINGLE global RNG stream in neuron-index order, so a
# neuron's noise depends on its ABSOLUTE index -> it is inherently co-residence-DEPENDENT (an organ built second
# sits at a different offset -> a different draw). Making it invariant needs a per-neuron-seeded noise stream (a
# sim/ engine edit, out of scope here), so the frozen migration pool reconciles it by running the noise OFF -- a
# per-pool config decision exactly like the per-region threshold-het seam. Both organs declare the SAME value, so
# there is no MergeConflict, and every read stays deterministic + byte-identical merged-vs-coresident.
_NOISE_OFF = {"enable_conductance_noise": False, "enable_ou_process": False, "ou_std_current_pA": 0.0}
# nmda_ratio is a GLOBAL scalar, but the engine applies it PER-REGION once any region opts into parameter
# heterogeneity (self_schema does): a non-het co-resident (d6) then reads the DEFAULT ratio in the merged arm but
# the global override in the alone arm -> the override bleeds across co-residence. self_schema's attractor works
# at the default ratio (0.4) too (author read still self>>heard, verified), so we DO NOT override it -- keeping the
# global at its default makes d6's nmda-ratio identical in both arms. (A per-region nmda_ratio field is the faithful
# long-term seam; the default is the correct value here, so no override is needed.)
_SELF_SCHEMA_CONFIG = {
    "enable_nmda": True,
    "enable_stdp": False, "enable_hebbian_learning": False, "enable_homeostasis": False,
    "enable_short_term_plasticity": False, "enable_structural_plasticity": False,
    "enable_reward_modulation": False, **_NOISE_OFF,
}
_D6_CONFIG = {
    "enable_nmda": True, "enable_nmda_recurrent": True, "nmda_recurrent_tau_decay_ms": 100.0,
    "enable_stdp": False, "enable_hebbian_learning": False, "enable_homeostasis": False,
    "enable_short_term_plasticity": False, "enable_structural_plasticity": False,
    "enable_reward_modulation": False, "enable_input_divisive_norm": False, **_NOISE_OFF,
}
# The engine builds a per-neuron NMDA mask the moment ANY region sets BrainRegion.enable_nmda=True (self_schema's
# `workspace` does), after which regular NMDA applies ONLY to masked neurons. d6's standalone has NO per-region
# enable_nmda, so it runs GLOBAL NMDA (every neuron). Co-resident with self_schema the mask would EXCLUDE d6 ->
# d6 silently loses its regular NMDA in the merged arm only. Marking every d6 region enable_nmda=True restores
# d6's faithful global-NMDA operating point AND makes the mask identical merged-vs-coresident. (K = R_MAX*N_SLOT
# = 5*6 slot pools `w0..w29` + the shared `fs`, from d6_multiref_wm_production_organ.)
_D6_REGION_FLAGS = {**{f"w{k}": {"enable_nmda": True} for k in range(5 * 6)}, "fs": {"enable_nmda": True}}


def _ordered_region_idx(bridge, name):
    """Region indices in the ORDER the standalone self_schema build reads them (`rm.indices(name)`), so the
    dev-index map + the explicit loop wiring reference the SAME neuron ordering the read expects."""
    return np.asarray(bridge.region_manager.indices(name), dtype=np.int64)


def _self_schema_geom():
    from research.runners._self_schema_region_derisk import (
        ASSEMBLY_SIZE, K_CONTENTS, ATTEND_SIZE, CONFID_SIZE, AUTHOR_SIZE, MEMBER_TO_ATTEND_W, WS_LOOP_GATE)
    from research.runners._gnw_rung1_ignition_curve_derisk import DEFAULT_ATTRACTOR_WEIGHT
    return dict(A=ASSEMBLY_SIZE, K=K_CONTENTS, AT=ATTEND_SIZE, CF=CONFID_SIZE, AU=AUTHOR_SIZE,
                MTA=float(MEMBER_TO_ATTEND_W), GATE=WS_LOOP_GATE, LOOP_W=float(DEFAULT_ATTRACTOR_WEIGHT))


def _self_schema_member_attend(bridge):
    g = _self_schema_geom()
    ws = _ordered_region_idx(bridge, "workspace")
    ss = _ordered_region_idx(bridge, "self_schema")
    member = {k: ws[k * g["A"]:(k + 1) * g["A"]] for k in range(g["K"])}
    attend = {k: ss[k * g["AT"]:(k + 1) * g["AT"]] for k in range(g["K"])}
    base = g["AT"] * g["K"]
    confid = ss[base:base + g["CF"]]
    author = ss[base + g["CF"]:base + g["CF"] + g["AU"]]
    return g, member, attend, confid, author


def _self_schema_idx(bridge):
    """The dev-index map `SelfSchemaAuthorshipOrgan._author_rate` consumes (member/attend/confid/author),
    computed from the pool's region slices exactly as `build_self_schema_bridge` computes it standalone."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    _g, member, attend, confid, author = _self_schema_member_attend(bridge)
    return {
        "member_dev": {k: xp.asarray(v) for k, v in member.items()},
        "attend_dev": {k: xp.asarray(v) for k, v in attend.items()},
        "confid_dev": xp.asarray(confid),
        "author_dev": xp.asarray(author),
    }


def _self_schema_wiring(bridge, rm):
    """The explicit edges the base pathways do NOT carry: the K dense self-recurrent workspace assembly loops
    (the GNW Rung-1 attractor) + the fixed member->attend read projection. Reproduces exactly the `union`
    `build_self_schema_bridge` adds before its own inject, keyed on the pool's region slices."""
    from research.runners._gnw_rung1_ignition_curve_derisk import _build_assembly_loop_population
    from research.runners._gnw_rung3_report_reasoning_identity_derisk import _dense_projection
    g, member, attend, _confid, _author = _self_schema_member_attend(bridge)
    union = {}
    for k in range(g["K"]):
        union[f"loop_{k}"] = _build_assembly_loop_population(member[k], g["LOOP_W"])
        union[f"member{k}_to_attend"] = _dense_projection(member[k], attend[k], g["MTA"], g["GATE"])
    return union


def _self_schema_post_inject(bridge):
    from research.runners._self_schema_region_derisk import WS_LOOP_GATE
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)   # freeze the loop (per-turn reads never learn)


def _self_schema_reads(organ):
    """The organ's REAL author-pool read battery: the SELF (volunteered) + HEARD (recalled) author firing
    rates + the calibrated threshold. Byte-identical merged-vs-coresident == the read is co-residence-invariant."""
    organ.ensure_built()
    r_self = organ._author_rate(authored=True, lesion=False)
    r_heard = organ._author_rate(authored=False, lesion=False)
    ra = organ.read_author(authored=True)
    return {
        "author_rate_self": float(r_self), "author_rate_heard": float(r_heard),
        "threshold": float(organ.threshold),
        "calib.self_rate": float(organ.calib["self_rate"]),
        "calib.heard_rate": float(organ.calib["heard_rate"]),
        "read_author.rate": float(ra["author_rate"]),
        "read_author.is_self": float(bool(ra["is_self"])),
    }


def _self_schema_answer(organ):
    organ.ensure_built()
    return (organ.read_author(authored=True)["label"], organ.read_author(authored=False)["label"])


# D6 multi-referent WM — a bare-bridge MultiSlotHold whose reads run directly on the pool slice.
_D6_BATTERY = ("dog", "cat", "bird")


def _d6_organ(seed, shared):
    from research.runners.d6_multiref_wm_production_organ import MultiReferentWMOrgan
    return MultiReferentWMOrgan(seed=seed, shared=shared)


def _self_schema_organ(seed, shared):
    from research.runners.self_schema_production_organ import SelfSchemaAuthorshipOrgan
    return SelfSchemaAuthorshipOrgan(seed=seed, shared=shared)


def _d6_reads(organ):
    """The organ's REAL multi-referent LOAD+HOLD+READ pipeline over a fixed 3-referent battery: the min held
    bump amplitude, the recovery success + the per-register bump amplitudes read off the spiking buffer."""
    organ.ensure_built()
    res = organ.load(list(_D6_BATTERY), lesion=False)
    out = {
        "n_referents": float(res["n_referents"]),
        "hold_alive_min": float(res["hold_alive_min"]),
        "all_recovered": float(bool(res["all_recovered"])),
        "zero_input_ok": float(bool(res["zero_input_ok"])),
    }
    return out


def _d6_answer(organ):
    organ.ensure_built()
    res = organ.load(list(_D6_BATTERY), lesion=False)
    return tuple(res["recovered"].get(r) for r in range(res["n_referents"]))


GROUP_A = [
    OrganDescriptor(key="causal_whatif",
                    regions=("evt",),
                    spec_fn=_spec_causal_whatif,
                    scaffold_residuals=("host-injected DA sign at train time (declared teacher signal)",)),
    OrganDescriptor(key="comprehension",
                    regions=("sel_agent", "sel_FS_agent", "sel_patient", "sel_FS_patient",
                             "cue_position_pos", "cue_position_neg", "cue_animacy_pos", "cue_animacy_neg",
                             "cue_verbfit_pos", "cue_verbfit_neg", "cue_lexbias_pos", "cue_lexbias_neg"),
                    spec_fn=_spec_comprehension),
    OrganDescriptor(key="self_schema",
                    regions=("workspace", "workspace_fs", "self_schema"),
                    spec_fn=_spec_self_schema, param_het=True,
                    config=_SELF_SCHEMA_CONFIG,
                    idx_fn=_self_schema_idx,
                    explicit_wiring_fn=_self_schema_wiring, post_inject_fn=_self_schema_post_inject,
                    organ_cls=_self_schema_organ, read_fn=_self_schema_reads, answer_fn=_self_schema_answer,
                    supports_shared=True,
                    scaffold_residuals=("hand-declared GNW assembly loops (explicit_wiring_fn, self-organize later)",)),
    OrganDescriptor(key="source_provenance",
                    regions=("episode", "content_readout", "ctx_perceived", "ctx_generated",
                             "prov_perceived", "prov_generated", "inh_perceived", "inh_generated"),
                    spec_fn=_spec_source_provenance, param_het=True),
    OrganDescriptor(key="curiosity",
                    regions=("cue", "striosome_value", "reward_us", "snc", "ask"),
                    spec_fn=_spec_curiosity, param_het=True),
    OrganDescriptor(key="prospective_memory",
                    regions=("cortex_ctx", "dlpfc_wm", "rel_A", "rel_B"),
                    spec_fn=_spec_prospective_memory),
    OrganDescriptor(key="d6_multiref_wm",
                    regions=(),                          # region names live in build_persistent_slot; discovered at build
                    spec_fn=_spec_d6_multiref_wm,
                    config=_D6_CONFIG, param_het=True,   # co-reside under the SAME name-keyed per-region param-het
                    region_flags=_D6_REGION_FLAGS,       # seam self_schema uses, so a co-resident's het STATE is
                    organ_cls=_d6_organ, read_fn=_d6_reads, answer_fn=_d6_answer,  # symmetric (co-residence-invariant);
                    supports_shared=True),               # region_flags keep d6's NMDA-mask membership identical too
]

# Group-B/C DEFERRALS (registered as data so the report + board carry the honest reason + the seam each needs).
GROUP_A_DEFERRED = {
    "b3_noncontradiction": "STATELESS — owns no substrate; rides the live composer's spiking polarity recall "
                           "via a `recall` callable. Nothing to co-locate (no BrainRegions).",
    "reconsolidation": "Owns no circuit — reuses the D2 SURPRISE organ's slice + rewrites the composer store. "
                       "Its substrate migrates WHEN surprise does; no distinct regions.",
    "repair": "No class — functions composing the D4 COMPREHENSION organ. Its substrate == comprehension's; "
              "migrates when comprehension does.",
    "d3_discourse_event_register": "Multi-bridge — builds FOUR FS-WTA discretizer bridges + a host rate-RNN "
                                   "transition. Not a single shared-pool slice; needs a multi-bridge seam.",
    "d5_episodic": "Heavy own-pool — a ~2000-neuron CA3 with two-compartment apical dendritic-dAP + slow-NMDA "
                   "reverberation + BTSP formation. Group-C own-pool + apical/NMDA-slow seam.",
    "affective_tom": "OU + NEUROMODULATOR-subsystem seam — enable_ou_process=True + a bespoke `appraisal` "
                     "neuromodulator triad drives the read. Group-B OU/neuromod seam.",
}

# ORGAN-READ deferrals — the 5 Group-A organs that ARE substrate-init byte-identical (migration-safe) but whose
# READ pipeline is not YET a clean function of the frozen shared substrate. Each names the concrete engine/wrapper
# seam it needs (honest boundary: substrate-init is the migration gate; organ-read is this rung, closed for the 2
# frozen bare-substrate organs; these 5 need a further seam and DO NOT block the batch).
GROUP_A_ORGANREAD_DEFERRED = {
    "curiosity": "NEUROMODULATOR-SUBSYSTEM + plasticity config seam — the read needs the `curiosity` neuromodulator "
                 "(from_novelty -> ASK excitability_drive) + a spiking-SNc RPE critic (enable_stdp + "
                 "enable_reward_modulation + gabab). Those global plasticity/neuromod flags CONFLICT with the frozen "
                 "pool; needs a per-region neuromodulator/plasticity seam (Group-B neuromod).",
    "source_provenance": "NEUROMODULATOR-CONTEXT-LINE seam — the read rides two encoding-context neuromod lines "
                         "(ctx_perceived/ctx_generated) each gating a zero-init Hebbian episode->provenance trace "
                         "(enable_hebbian at encode). The `ProvenanceBrain` wrapper must accept an injected bridge, "
                         "and the hebbian-encode config conflicts with the frozen pool.",
    "comprehension": "WRAPPER + OPERATING-POINT seam — the read is tied to the `SpikingRoleCompetition` wrapper "
                     "(installed cue weights + per-cue index maps); it must accept an injected bridge+slice, and its "
                     "merge operating point (dt=1.0, homeostasis ON, per-region-thresh ON) must reconcile the global "
                     "enable_homeostasis the frozen pool holds OFF.",
    "prospective_memory": "STATEFUL WRAPPER seam — a `SFANmdaProspectiveMemory` (HomeostaticProspectiveMemory "
                          "hierarchy) with a homeostatic-bias calibration + SFA + dendritic plateau + a one-shot "
                          "Hebbian FORMATION event, read across MULTIPLE turns (form -> hold -> present-cue). Needs "
                          "the hierarchy to accept an injected bridge + a stateful multi-turn read protocol.",
    "causal_whatif": "LIVE-COMPOSER GROUNDING + DA/STDP seam — the read enumerates events + moat-confirms answers "
                     "against a live RFPhasorComposer, and the forward model trains temporal-order STDP + phasic-DA "
                     "at build (enable_stdp + enable_reward_modulation, conflicting with the frozen pool). Needs a "
                     "shared/stub composer surface + a per-region plasticity/DA seam.",
}

GROUP_A_KEYS = [d.key for d in GROUP_A]

REGISTRY = {d.key: d for d in (SURPRISE, WORLDMODEL, *GROUP_A)}


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Smoke: the descriptor->engine round-trip is BYTE-IDENTICAL to the shipped MergedSubstrate at init.
# ─────────────────────────────────────────────────────────────────────────────────────────────
_INIT_ARRAYS = (
    "cp_neuron_firing_thresholds", "cp_membrane_potential_v", "cp_recovery_variable_u",
    "cp_izh_a", "cp_izh_b", "cp_izh_C", "cp_izh_c_reset", "cp_izh_d_increment",
    "cp_izh_vpeak", "cp_izh_vt", "cp_izh_vr",
)


def _region_slice(bridge, name):
    return np.asarray(sorted(int(i) for i in _idx(bridge, name)), dtype=np.int64)


def _smoke(seed: int = 42) -> dict:
    """Build the engine pool (surprise+world-model via the REGISTRY) and the shipped MergedSubstrate, and
    compare every per-neuron INIT array over both organs' regions. Byte-identity (max delta 0.0) proves the
    declarative descriptor path reproduces the bespoke pool exactly -- the round-trip the design claims."""
    from research.runners.onebrain_merge_production import MergedSubstrate

    pool = merge_organs([SURPRISE, WORLDMODEL], seed=seed)
    shipped = MergedSubstrate(seed=seed, organs=("surprise", "worldmodel"))
    shipped.ensure_built()

    all_regions = list(SURPRISE.regions) + list(WORLDMODEL.regions)
    worst, worst_where = 0.0, None
    for rname in all_regions:
        ei = _region_slice(pool.bridge, rname)
        si = _region_slice(shipped.bridge, rname)
        if ei.size != si.size:
            return {"seed": seed, "byte_identical": False, "reason": f"{rname} size {ei.size}!={si.size}"}
        for a in _INIT_ARRAYS:
            ea = _host(getattr(pool.bridge, a, None)); sa = _host(getattr(shipped.bridge, a, None))
            if ea is None or sa is None:
                continue
            d = float(np.max(np.abs(ea[ei].astype(np.float64) - sa[si].astype(np.float64)))) if ei.size else 0.0
            if d > worst:
                worst, worst_where = d, (rname, a)
    n_pool = int(pool.bridge.cp_membrane_potential_v.shape[0])
    n_ship = int(shipped.bridge.cp_membrane_potential_v.shape[0])
    ok = bool(worst == 0.0 and n_pool == n_ship)
    return {"seed": seed, "byte_identical": ok, "max_init_delta": worst, "worst_where": worst_where,
            "n_engine": n_pool, "n_shipped": n_ship, "organs": all_regions}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_onebrain_merge_framework_smoke_s42.json")
    args = ap.parse_args()
    if args.smoke:
        r = _smoke(args.seed)
        print("=== onebrain_merge_framework SMOKE (descriptor->engine round-trip vs shipped MergedSubstrate) ===")
        print(f"  seed={r['seed']} engine_N={r.get('n_engine')} shipped_N={r.get('n_shipped')} "
              f"max_init_delta={r.get('max_init_delta')} worst={r.get('worst_where')}")
        print(f"  BYTE-IDENTICAL: {r['byte_identical']}  ->  {'PASS' if r['byte_identical'] else 'FAIL'}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(r, indent=2))
            print(f"  wrote {args.out}")
    else:
        print("registered organs:", list(REGISTRY))
        print("run the round-trip smoke with:  SIM_BACKEND=numpy python -m "
              "research.runners.onebrain_merge_framework --smoke")


if __name__ == "__main__":
    main()
