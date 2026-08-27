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
# reuse-by-import: pool #2's (metacog + pragmatic) geometry constants -- the SAME ones the shipped
# `MergedSubstrate2` / `MetacogProductionOrgan` import, so a descriptor's config/wiring can never drift from them.
from research.runners._second_order_metacog_monitor_derisk import (
    ASSEMBLY_SIZE, K_CLASSES, META_SIZE, WS_LOOP_GATE, DEFAULT_ATTRACTOR_WEIGHT, DEFAULT_NMDA_TAU,
)
from research.runners._recursive_tom_rsa_derisk import RSA_ITEM_SIZE


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
    # merge seam #2 (2026-08-27, the MULTI-TURN-READ arc) — DETERMINISTIC transpose SpMV for the synaptic input.
    # The default synaptic-input path is `connections.T @ fired_2col`, a TRANSPOSE sparse matmul whose FP
    # summation ORDER varies with the matrix layout (total-N / edge interleaving). For a FROZEN-forward read this
    # is below the answer margin, but a SPIKING-DYNAMICS read integrated over hundreds of steps (prospective_
    # memory's attractor hold; d6's slow-NMDA reverberation) AMPLIFIES a single-ULP per-step delta into a
    # 1-spike read divergence -- so a co-resident organ's slice read differs from its alone read purely because
    # the OTHER organs enlarged the shared matrix (a NON-synaptic, layout-mediated coupling, not a cross-synapse).
    # ON pins the byte-identical CSR path (sim/bridge.py:8730), making the merged-vs-coresident spiking read exact.
    # It is byte-identical to the default whenever there is no summation-order variance, so it never regresses the
    # frozen-forward organs. (Legacy keeps it OFF so the discriminator still diverges on total-N as before.)
    cfg.deterministic_transpose_matvec = not legacy
    # merge seam #3 (2026-08-27, the full-7 --keys all arc) — DEDUP the per-synapse ROUTING masks so they align
    # with cp_connections even when the plan has DUPLICATE (pre,post) edges. cp_connections is built via
    # coo->tocsr()+sum_duplicates(), which merges duplicates, but inject_explicit_wiring builds the nmda_slow /
    # gaba_b / coincidence / graded / stp / plastic masks from the un-deduped `keyed` list -> when an organ's
    # explicit_wiring_fn wires the SAME endpoints as a base RegionPathway (pmem's c2d/d2c, comprehension's
    # cue_monitor), len(keyed) > nnz and every mask entry after the first duplicate coord addresses the WRONG
    # synapse. That shift is co-residence-DEPENDENT (an organ alone may have no duplicates), so d6's nmda_slow
    # AMPA-suppression lands on different synapses merged-vs-coresident -- the full-7 --keys all NO-GO. ON collapses
    # `keyed` to one entry per unique (pre,post) so the masks are co-residence-invariant. Byte-identical to the
    # default whenever the plan has no duplicate edges (frozen-forward organs are unaffected). Legacy keeps it OFF.
    cfg.dedup_synapse_masks = not legacy
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

        # (5a) NMDA-MASK RECONCILIATION (a co-residence seam, like the per-region threshold/het seams). The engine
        #      builds a per-neuron NMDA mask the moment ANY region opts in (BrainRegion.enable_nmda=True), after
        #      which regular NMDA applies ONLY to masked neurons; with global enable_nmda=True but NO region opting
        #      in it falls back to GLOBAL NMDA (v1 back-compat). That fallback is co-residence-DEPENDENT: an organ
        #      whose regions ALL opt OUT (source_provenance, causal_whatif) gets NO NMDA when co-resident with an
        #      NMDA organ (masked out) but SPURIOUS global NMDA when ALONE on the enable_nmda=True superset config.
        #      Pin it: global-on + no-opt-in => install an ALL-ZERO mask (no neuron gets NMDA), so a no-NMDA organ's
        #      slice reads byte-identically alone vs co-resident. Organs that WANT global NMDA opt every region in
        #      via region_flags (d6), so a mask is already built and this never fires; and any pool that already
        #      has an opting-in region (self_schema present) also skips it. Init arrays are untouched.
        if getattr(cfg, "enable_nmda", False) and getattr(bridge, "cp_nmda_neuron_mask", None) is None:
            n_all = int(bridge.cp_membrane_potential_v.shape[0])
            bridge.cp_nmda_neuron_mask = xp.zeros(n_all, dtype=xp.float32)

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

    # The shipped POOL-#2 organs call `shared.metacog_idx()` / `shared.pragmatic_item_dev()` (named methods,
    # exactly like `surprise_idx_map` above) -- provide them by dispatch so `MetacogProductionOrgan` /
    # `PragmaticProductionOrgan` run UNMODIFIED against this pool (2026-08-27 fold of pool #2 into the registry).
    def metacog_idx(self):
        return self.idx("metacog")

    def pragmatic_item_dev(self):
        return self.idx("pragmatic")

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
        the full per-neuron state, let `active`'s slice evolve, restore every OTHER organ's slice at the end.

        PER-CALL scope: this is the isolation for a read that is a SINGLE drive->settle->read. It restores every
        other organ's slice on exit BUT keeps `active`'s -- so wrapping a MULTI-TURN read in ONE read_isolation
        already lets the active slice persist across the turns inside it (form -> hold -> cue). For a stateful
        organ whose read ALSO mutates per-SYNAPSE / timing state (e.g. a whole-bridge reset between sub-sequences),
        use `sequence_isolation` (below), which additionally snapshots+restores that state so nothing leaks."""
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

    # per-SYNAPSE + non-_PER_NEURON_STATE arrays a MULTI-TURN stateful read can mutate (synaptic pulse timers, the
    # nmda/gabab rise+recurrent buffers) -- snapshotted by sequence_isolation SO a whole-bridge reset between an
    # organ's own sub-sequences (e.g. pmem's _reset_dynamics) leaves NO co-resident organ perturbed at guard exit.
    _SEQ_EXTRA_STATE = (
        "cp_conductance_g_nmda_rise", "cp_conductance_g_nmda_recurrent", "cp_conductance_g_nmda_recurrent_rise",
        "cp_conductance_g_gabab_slow", "cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise",
        "cp_synapse_pulse_timers", "cp_synapse_pulse_progress",
    )

    def sequence_isolation(self):
        """SEQUENCE-scoped isolation for a MULTI-TURN STATEFUL read -- the general resolution of the per-call-
        isolation vs stateful-hold tension. read_isolation is per-CALL (snapshot at enter, restore OTHERS at exit).
        A stateful organ (self-sustaining attractor + per-neuron SFA trace) must instead hold its slice UNRESET
        across the turns of ONE read (form -> hold through arbitrary intervening turns -> cue) AND may reset the
        whole bridge between its own sub-sequences. sequence_isolation spans the WHOLE sequence: snapshot EVERY
        mutable array (per-neuron + per-synapse pulse/rise buffers) AND the runtime timing counters at enter, let
        the active organ's slice evolve freely across every turn, and at exit restore the FULL snapshot -- so no
        co-resident organ is perturbed even by a whole-bridge reset. The organ caches its numeric reads BEFORE the
        guard exits; byte-identity merged-vs-coresident is then exact (zero cross-organ synapses => the active
        slice evolves identically alone or co-resident). GENERAL: any organ whose read is multi-turn declares it
        this way -- the harness is not pmem-specific."""
        import contextlib
        import random as _random
        @contextlib.contextmanager
        def _guard():
            b = self.bridge
            names = list(self._PER_NEURON_STATE) + list(self._SEQ_EXTRA_STATE)
            snaps = [(nm, getattr(b, nm).copy() if getattr(b, nm, None) is not None else None) for nm in names]
            t_step = getattr(b.runtime_state, "current_time_step", None)
            t_ms = getattr(b.runtime_state, "current_time_ms", None)
            # ALSO snapshot the GLOBAL RNG state (np.random + Python random): a multi-turn read's calibration can
            # CONSUME the global RNG (e.g. a Python random.random() draw), which would then perturb a LATER organ's
            # read on the shared bridge -- an order-dependent leak invisible to the per-array restore (the bridge
            # state is untouched; only the RNG cursor advanced). Restoring it makes the read RNG-transparent to
            # every co-resident organ. (Caught 2026-08-27: pmem's read broke a downstream d6 read this exact way.)
            np_state = np.random.get_state()
            py_state = _random.getstate()
            try:
                yield
            finally:
                for nm, snap in snaps:
                    if snap is None:
                        continue
                    cur = getattr(b, nm, None)
                    if cur is not None and cur.shape == snap.shape:
                        cur[:] = snap
                    else:
                        setattr(b, nm, snap)
                if t_step is not None:
                    b.runtime_state.current_time_step = t_step
                if t_ms is not None:
                    b.runtime_state.current_time_ms = t_ms
                np.random.set_state(np_state)
                _random.setstate(py_state)
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
#  POOL #2 — METACOG + PRAGMATIC (the 3rd/4th bespoke pool organ, folded 2026-08-27). Mirrors pool #1's fold:
#  the bespoke `MergedSubstrate2` (`onebrain_merge_production2.py`) hand-builds ONE shared bridge for the E1
#  metacognition balance-of-evidence monitor + the D-pragmatics scalar-implicature RSA organ, with THREE
#  region-scoped seams ON (`per_region_parameter_heterogeneity`, `per_region_threshold_heterogeneity`,
#  `per_region_wiring_seed`) and an ALWAYS-ON wiring inject (base pathways + metacog's dense self-recurrent
#  assembly loops) + settle-to-rest snapshot. Both shipped production organs (`MetacogProductionOrgan`,
#  `PragmaticProductionOrgan`) ALREADY take a `shared=` kwarg (unlike most Group-A organs, which needed one
#  added) and call NAMED methods on it — `shared.metacog_idx()` / `shared.pragmatic_item_dev()` — so they run
#  UNMODIFIED against this pool via the `MergedPool.metacog_idx`/`.pragmatic_item_dev` dispatch methods added
#  above. `per_region_wiring_seed` is NOT set here: the framework's `wire=True` inject already rebuilds
#  `cp_connections` from `build_wiring_plan(seed, per_region_seed=True)` (the SAME order-invariant mechanism),
#  so the cfg flag would be redundant (clobbered by the wire=True rebuild) -- the exact reason no Group-A
#  descriptor sets it either.
#
#  NAME COLLISION (honest, not a bug): metacog's "workspace"/"workspace_fs" region names COLLIDE with
#  self_schema's (Group-A) -- the SAME collision the DESIGN doc names for affect vs metacog (seams key on
#  region NAME; a rename would break byte-identity to the standalone organ). So metacog+pragmatic are
#  registered but verified as THEIR OWN pair (`--keys metacog,pragmatic`), never in the GROUP_A/"all" batch --
#  exactly how pool #1 (surprise/worldmodel) is already excluded from "all" for the same reason (surprise's
#  "cue" collides with curiosity's).
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _pool2_metacog_specs():
    """Region/pathway specs for the METACOG organ's pool-#2 slice -- reused BY CALLING the shipped
    `MergedSubstrate2._metacog_specs` on a throwaway (never-built) instance, so there is exactly ONE
    definition of the pool-#2 geometry (no copy to drift out of sync with the bespoke class)."""
    from research.runners.onebrain_merge_production2 import MergedSubstrate2
    r, p = MergedSubstrate2()._metacog_specs()
    return list(r), list(p)


def _pool2_pragmatic_specs():
    """Region/pathway specs for the PRAGMATIC organ's pool-#2 slice -- reused BY CALLING the shipped
    `MergedSubstrate2._pragmatic_specs` (see `_pool2_metacog_specs`)."""
    from research.runners.onebrain_merge_production2 import MergedSubstrate2
    r, p = MergedSubstrate2()._pragmatic_specs()
    return list(r), list(p)


def _spec_metacog(seed):
    r, p = _pool2_metacog_specs()
    return r, p, {}


def _spec_pragmatic(seed):
    r, p = _pool2_pragmatic_specs()
    return r, p, {}


# metacog wants NMDA ON (its workspace/meta_schema slices are NMDA-capable); pragmatic's regions carry
# region-level enable_nmda=False (from `_pool2_pragmatic_specs`), so the per-neuron NMDA mask zeroes their
# current even under this global True -- byte-identical to `build_rsa_bridge`'s enable_nmda=False. Mirrors
# MergedSubstrate2.ensure_built's cfg block exactly (`has_metacog` -> True there is always True here because
# metacog's config sets it True whenever metacog is IN the pool; when pragmatic is tested ALONE as the
# coresident baseline, this key is simply absent from the union -> enable_nmda stays the CoreSimConfig default,
# matching MergedSubstrate2(organs=("pragmatic",))'s has_metacog=False -> enable_nmda=False).
# NOTE: `_NOISE_OFF` (the shared conductance/OU-noise-off dict every GROUP-A frozen organ unions) is defined
# LATER in this file (Group-A organ-read plumbing) -- inline the same three keys here rather than forward-
# referencing it, so this pool-#2 section can sit next to pool #1 (its natural place, both are the "4 pool
# organs" the migration rung names) without reordering the whole file.
_POOL2_METACOG_CONFIG = {
    "enable_nmda": True, "nmda_ratio": 0.5,
    "nmda_tau_decay": float(DEFAULT_NMDA_TAU), "nmda_recurrent_tau_decay_ms": float(DEFAULT_NMDA_TAU),
    "enable_stdp": False, "enable_reward_modulation": False, "enable_hebbian_learning": False,
    "enable_homeostasis": False, "enable_short_term_plasticity": False,
    "enable_structural_plasticity": False, "enable_ou_process": False,
    "enable_conductance_noise": False, "ou_std_current_pA": 0.0,
    "enable_parameter_heterogeneity": True,      # BOTH pool-#2 organs' graded rate codes require it (global -- no
                                                  # competing organ in this pool wants it off, so no per-region mask
                                                  # reconciliation is needed; matches MergedSubstrate2 exactly).
    # merge seam #1 (name-keyed Izhikevich param jitter) -- WITHOUT this, the engine pool's per-neuron het draw
    # falls back to the LEGACY whole-pool position-dependent draw (co-residence-DEPENDENT), diverging from the
    # shipped class's name-keyed region-scoped overwrite (caught by `--smoke2`: cp_izh_d_increment delta=111 on
    # "workspace" before this line was added). NOT auto-set here (unlike Group-A's `param_het=True` masking
    # path) because pool #2 sets it GLOBALLY, matching MergedSubstrate2 exactly (see the module comment above).
    "per_region_parameter_heterogeneity": True,
    "stdp_w_max": max(400.0, float(DEFAULT_ATTRACTOR_WEIGHT) * 4.0),
    "hebbian_max_weight": max(400.0, float(DEFAULT_ATTRACTOR_WEIGHT) * 4.0),
}
_POOL2_PRAGMATIC_CONFIG = {
    "enable_stdp": False, "enable_reward_modulation": False, "enable_hebbian_learning": False,
    "enable_homeostasis": False, "enable_short_term_plasticity": False,
    "enable_structural_plasticity": False, "enable_ou_process": False,
    "enable_conductance_noise": False, "ou_std_current_pA": 0.0,
    "enable_parameter_heterogeneity": True,      # identical value to metacog's -> no MergeConflict
    "per_region_parameter_heterogeneity": True,  # identical value to metacog's -> no MergeConflict
}


def _metacog_wiring(bridge, rm):
    """explicit_wiring_fn: metacog's K dense self-recurrent workspace assembly loops (the balance-of-evidence
    attractor), reproduced from MergedSubstrate2.ensure_built's `union[f"loop_{k}"]` union exactly (SAME
    `_build_assembly_loop_population` helper + SAME per-class member slice of the pool's "workspace" region).
    The BASE pathways (workspace<->workspace_fs, item<->item_fs) come from the engine's own
    `build_wiring_plan(seed, per_region_seed=True)` union (the wire=True inject already does this generically),
    so only the loops -- metacog's OWN topographic addition -- belong here."""
    from research.runners._gnw_rung1_ignition_curve_derisk import _build_assembly_loop_population
    ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
    union = {}
    for k in range(K_CLASSES):
        member = ws[k * ASSEMBLY_SIZE:(k + 1) * ASSEMBLY_SIZE]
        union[f"loop_{k}"] = _build_assembly_loop_population(member, float(DEFAULT_ATTRACTOR_WEIGHT))
    return union


def _metacog_post_inject(bridge):
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)   # freeze the assembly loop (balance mode reads never learn)


def _metacog_idx_fn(bridge):
    """The metacog organ's dev-index map, in the SHAPE `MetacogProductionOrgan._margin`/`nmda_norm_margin`
    consume (member_dev/meta_dev/meta_member_dev/fs_dev/confidence_read) -- computed identically to the shipped
    `MergedSubstrate2.metacog_idx` off the pool's region slices."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    rm = bridge.region_manager
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


def _pragmatic_idx_fn(bridge):
    """The pragmatic (RSA) organ's 3 item-assembly index arrays, in the shape `_rsa_recursion` consumes --
    computed identically to the shipped `MergedSubstrate2.pragmatic_item_dev`."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    rm = bridge.region_manager
    base = np.asarray(rm.indices("item"), dtype=np.int64)
    return {i: xp.asarray(base[i * RSA_ITEM_SIZE:(i + 1) * RSA_ITEM_SIZE]) for i in range(3)}


def _metacog_organ(seed, shared):
    from research.runners.metacog_production_organ import MetacogProductionOrgan
    return MetacogProductionOrgan(seed=seed, shared=shared)


def _pragmatic_organ(seed, shared):
    from research.runners.pragmatic_production_organ import PragmaticProductionOrgan
    return PragmaticProductionOrgan(seed=seed, shared=shared)


_METACOG_READ_EVIDENCE = (0.1, 0.5, 0.9)


def _metacog_reads(organ):
    """The organ's REAL calibrated confidence-margin read battery (the production `judge()` path) at a spread
    of evidence levels + the calibration numbers. Byte-identical merged-vs-coresident == the whole
    calibrate+judge balance-of-evidence pipeline is co-residence-invariant."""
    organ.ensure_built()
    out = {"threshold": float(organ.threshold),
           "calib.mean_hi": float(organ.calib["mean_hi"]), "calib.min_hi": float(organ.calib["min_hi"]),
           "calib.mean_lo": float(organ.calib["mean_lo"]), "calib.max_lo": float(organ.calib["max_lo"])}
    for i, e in enumerate(_METACOG_READ_EVIDENCE):
        out[f"margin_{i}"] = float(organ.judge(e)["balance"])
    return out


def _metacog_answer(organ):
    organ.ensure_built()
    return tuple(bool(organ.judge(e)["confident"]) for e in _METACOG_READ_EVIDENCE)


_PRAGMATIC_READ_UTTS = ("some", "all", "none")


def _pragmatic_reads(organ):
    """The organ's REAL graded RSA listener-belief read battery (the production `interpret()` path) over the
    scalar-utterance family. Byte-identical merged-vs-coresident == the whole RSA-recursion read is
    co-residence-invariant."""
    organ.ensure_built()
    out = {}
    for u in _PRAGMATIC_READ_UTTS:
        info = organ.interpret(u)
        for i, s in enumerate(info["states"]):
            out[f"{u}.belief_{s}"] = float(info["belief"][i])
        out[f"{u}.margin"] = float(info["implicature_margin"])
    return out


def _pragmatic_answer(organ):
    organ.ensure_built()
    return tuple(organ.interpret(u)["enriched_interpretation"] for u in _PRAGMATIC_READ_UTTS)


METACOG = OrganDescriptor(
    key="metacog",
    regions=("workspace", "workspace_fs", "meta_schema"),
    spec_fn=_spec_metacog,
    config=_POOL2_METACOG_CONFIG,
    explicit_wiring_fn=_metacog_wiring, post_inject_fn=_metacog_post_inject,
    idx_fn=_metacog_idx_fn,
    organ_cls=_metacog_organ, read_fn=_metacog_reads, answer_fn=_metacog_answer,
    supports_shared=True,
    scaffold_residuals=("hand-declared dense self-recurrent assembly loops (explicit_wiring_fn, "
                        "self-organize later, exactly like self_schema's)",),
)

PRAGMATIC = OrganDescriptor(
    key="pragmatic",
    regions=("item", "item_fs"),
    spec_fn=_spec_pragmatic,
    config=_POOL2_PRAGMATIC_CONFIG,
    idx_fn=_pragmatic_idx_fn,
    organ_cls=_pragmatic_organ, read_fn=_pragmatic_reads, answer_fn=_pragmatic_answer,
    supports_shared=True,
)

POOL2_KEYS = ["metacog", "pragmatic"]

# ─────────────────────────────────────────────────────────────────────────────────────────────
#  GROUP A — the declarative-NOW organs (DESIGN §5). Each is a registry ROW: a `spec_fn` that reuses the
#  organ's OWN de-risk builder (throwaway bridge, we read its BrainRegion specs) + `param_het` where the
#  organ's standalone uses parameter-heterogeneity (reconciled by the name-keyed per-region seam).
#  ORGAN-READ status (2026-08-27): FIVE of these — self_schema + d6_multiref_wm + comprehension + source_provenance
#  + causal_whatif — now take a `shared=` kwarg and RUN their real read pipeline on the wired pool byte-identically
#  (supports_shared=True; the descriptors carry organ_cls/read_fn/answer_fn + config + idx_fn/explicit_wiring_fn/
#  region_flags as needed). Three are FROZEN forward passes (comprehension's installed cue->role validities +
#  frozen gates; self_schema; d6). source_provenance + causal_whatif add a BUILD-TIME plasticity step that mutates
#  their OWN slice's weights (a Hebbian episode->prov encode; a temporal-order-STDP + phasic-DA forward-model
#  train) under a config toggle (the plasticity flags are read live per step) + a universal gain-0 freeze of every
#  non-organ edge + read_isolation, then read FROZEN — the encode/train is confined to the organ's slice and is
#  co-residence-invariant. The remaining TWO (curiosity, prospective_memory) are substrate-init GO but their read
#  still needs a seam (GROUP_A_ORGANREAD_DEFERRED names each precisely) — for those the batched gate stays the
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
    # base ProspectiveMemory (NOT the SFANmda subclass) reads the region specs WITHOUT running the multi-stage
    # homeostat/plateau calibration (the subclass's __init__ steps the substrate) -- the regions are identical
    # across the hierarchy. The rel pools are built with the POOL-GAINED recurrence (num_traits=1 delivers ~6x
    # less current per unit weight -> the rel accumulator needs the gained recurrence to ramp; only region EDGE
    # weights change, not the per-neuron INIT arrays, so substrate-init byte-identity is unaffected).
    from research.runners._pmem_intention_latch_derisk import ProspectiveMemory
    pm = ProspectiveMemory(["A", "B"], ["d0", "d1", "d2", "d3"], seed=int(seed),
                           rel_recurrent_weight=_PMEM_REL_RECURRENT)
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


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  D4 COMPREHENSION — the SpikingRoleCompetition Wong-Wang WTA read (a FROZEN forward pass). The organ's read
#  drives the SEMANTIC cue votes for a transitive's two nouns, settles the sel_agent/sel_patient WTA under
#  mutual inhibition, and reads the firing margin |agentEv_0 - agentEv_1| off cp_firing_states. Plasticity is
#  frozen (the cue->role validities are INSTALLED, the cue gates are 0), so the read never mutates a weight and is
#  a clean function of the frozen substrate slice -- exactly the closable shape self_schema/d6 have.
# The config UNIONs compatibly with self_schema/d6 (enable_nmda ON -- the sel_* regions carry per-region
# enable_nmda=True in the spec; every plasticity/noise flag OFF, matching the frozen pool). No MergeConflict: the
# keys it shares with self_schema/d6 (enable_nmda + the plasticity-OFF flags) agree exactly. Homeostasis stays OFF
# (the frozen-pool value); the read runs at the pool's dt=1.0 (the de-risk's declared merge operating point).
_COMPREHENSION_CONFIG = {
    "enable_nmda": True,
    "enable_stdp": False, "enable_hebbian_learning": False, "enable_homeostasis": False,
    "enable_short_term_plasticity": False, "enable_structural_plasticity": False,
    "enable_reward_modulation": False, **_NOISE_OFF,
}


def _comprehension_organ(seed, shared):
    from research.runners.comprehension_production_organ import ComprehensionProductionOrgan
    return ComprehensionProductionOrgan(seed=seed, shared=shared)


def _comprehension_battery(seed):
    """A small deterministic in-scope battery (well- + ill-formed transitives) for the read, from the de-risk's
    OWN battery builder so the items are guaranteed cue-covered + reproducible per seed."""
    from research.runners._spiking_comprehension_monitor_derisk import build_battery
    return build_battery(int(seed), n_per_cond=1)


def _comprehension_reads(organ):
    """The organ's REAL spiking comprehension read: the calibrated well-vs-ill threshold + the repair floor/lean,
    plus the per-item SEMANTIC sel-pool margin |agentEv_0 - agentEv_1| read off cp_firing_states for each battery
    item. Byte-identical merged-vs-coresident == the whole Wong-Wang WTA read is co-residence-invariant."""
    organ.ensure_built()
    out = {
        "threshold": float(organ.threshold),
        "role_floor": float(organ.role_floor),
        "lean_margin": float(organ.lean_margin),
        "calib.mean_well": float(organ.calib["mean_well"]),
        "calib.mean_ill": float(organ.calib["mean_ill"]),
    }
    for i, (_lab, _tag, n0, v, n1) in enumerate(_comprehension_battery(organ.seed)):
        out[f"margin_{i}"] = float(organ.read_margin(n0, v, n1))
    return out


def _comprehension_answer(organ):
    """The rendered read-out: the per-item `comprehended` decision (margin >= threshold) over the battery."""
    organ.ensure_built()
    return tuple(bool(organ.read_margin(n0, v, n1) >= organ.threshold)
                 for (_lab, _tag, n0, v, n1) in _comprehension_battery(organ.seed))


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  SOURCE-PROVENANCE (laneC #129) — a LEARNED, context-gated OPPONENT trace. Unlike the frozen-forward-pass
#  organs, its read is encode(fact,provenance) then recall(fact): the ENCODE mutates a zero-init episode->prov
#  Hebbian trace. The seam (finding 2026-08-27 organ-read-extension): move that encode to a per-region-gated
#  BUILD-TIME step under a GLOBAL-HEBBIAN TOGGLE (enable_hebbian_learning is read live each step, so the read
#  organ flips it True + sets the hebbian hyperparams ONLY around the build-time encode, then restores False;
#  every non-prov edge stays inert — the other organs' plastic edges are quiescent, the pool has zero cross-organ
#  synapses, and the prov/content edges are the only gated-plastic edges in the slice). recall() is then a clean
#  FROZEN forward pass. The whole encode + every recall runs in the pool's read_isolation("source_provenance") so
#  co-resident slices are restored -> the read is byte-identical merged-vs-coresident. enable_nmda reconciles via
#  the existing per-region mask (its regions carry enable_nmda=False -> opt OUT). param_het reconciles via the
#  name-keyed per-region seam (its standalone uses parameter heterogeneity), exactly as self_schema/d6.
# config: enable_hebbian_learning=False (union with the frozen organs; toggled True only at build-time encode);
#   every other plasticity/noise flag OFF; the hebbian hyperparams are inert while the toggle is off. enable_nmda
#   is NOT set here (unions to whatever the batch declares; the slice opts out per-region regardless).
_SOURCE_PROV_CONFIG = {
    "enable_stdp": False, "enable_hebbian_learning": False, "enable_homeostasis": False,
    "enable_short_term_plasticity": False, "enable_structural_plasticity": False,
    "enable_reward_modulation": False, **_NOISE_OFF,
}


def _source_prov_organ(seed, shared):
    return _SourceProvReadOrgan(int(seed), shared)


class _SourceProvReadOrgan:
    """Wraps `ProvenanceBrain` for the organ-read gate: a BUILD-TIME Hebbian encode of the 8-item paired battery
    (under the global-hebbian toggle + read_isolation) then a FROZEN recall read of each item's opponent
    provenance rates. shared=None -> the brain builds its own bridge and the encode runs at its own (already-on)
    enable_hebbian_learning, so the read is byte-identical to the standalone."""

    def __init__(self, seed, shared=None):
        self.seed = int(seed)
        self._shared = shared
        self.brain = None
        self.patterns = None
        self._built = False
        self._items = None

    def _guard(self):
        import contextlib
        if self._shared is not None:
            return self._shared.read_isolation("source_provenance")
        return contextlib.nullcontext()

    def ensure_built(self):
        if self._built:
            return
        from research.runners._laneC_source_provenance_opponent_derisk import (
            ProvenanceBrain, make_paired_patterns, _encode_all, HEBB_LR, HEBB_WMAX)
        self.brain = ProvenanceBrain(self.seed, shared=self._shared)
        self.patterns = make_paired_patterns(self.seed)
        with self._guard():
            if self._shared is not None:
                # GLOBAL-HEBBIAN TOGGLE — the pool config keeps enable_hebbian_learning False (frozen, unions
                # with the other organs); flip it (and the hebbian hyperparams the encode needs) True ONLY for
                # the build-time encode, then restore. The flag is read live per step, so this is exact.
                cc = self.brain._bridge.core_config
                saved = {k: getattr(cc, k) for k in (
                    "enable_hebbian_learning", "hebbian_learning_rate", "hebbian_max_weight",
                    "hebbian_min_weight", "hebbian_weight_decay", "hebbian_symmetric")}
                cc.enable_hebbian_learning = True
                cc.hebbian_learning_rate = float(HEBB_LR)
                cc.hebbian_max_weight = float(HEBB_WMAX)
                cc.hebbian_min_weight = 0.0
                cc.hebbian_weight_decay = 0.0
                cc.hebbian_symmetric = True
                # UNIVERSAL GAIN-0 FREEZE of every NON-prov edge during the encode (the finding's named seam): the
                # global enable_hebbian_learning toggle makes the whole bridge's plastic edges eligible, and a
                # co-resident organ's installed weights (e.g. comprehension's cue validities) then couple weakly
                # into the encode. Zero cp_plasticity_rate_gain everywhere first; the encode's own
                # set_plasticity_gate("prov_learn"/"content_learn", 1.0) re-opens ONLY the prov/content edges via
                # the gate->rate_gain path -> only those edges can move, deterministically + co-residence-invariant.
                b = self.brain._bridge
                saved_gain = None
                g = getattr(b, "cp_plasticity_rate_gain", None)
                if g is not None:
                    saved_gain = g.copy()
                    g[:] = 0.0
                try:
                    _encode_all(self.brain, self.patterns, learning=True)
                finally:
                    for k, v in saved.items():
                        setattr(cc, k, v)
                    if saved_gain is not None:
                        b.cp_plasticity_rate_gain[:] = saved_gain
            else:
                _encode_all(self.brain, self.patterns, learning=True)
        self._built = True

    def _recall_items(self):
        """Recall each of the 8 items from content alone (contexts silent) and read its opponent provenance rates.
        Winner + signed discriminability are computed DETERMINISTICALLY (no host tie-break rng) so the read is a
        pure function of the frozen trained substrate slice -> byte-identical merged-vs-coresident."""
        from research.runners._laneC_source_provenance_opponent_derisk import PROVENANCES, N_PAIRS
        if self._items is not None:
            return self._items
        self.ensure_built()
        items = []
        with self._guard():
            for prov in PROVENANCES:
                for i in range(N_PAIRS):
                    rec = self.brain.recall(self.patterns[prov][i])
                    rp, rg = rec["rate_perceived"], rec["rate_generated"]
                    winner = "perceived" if rp >= rg else "generated"
                    d_perc = (rp - rg) / (rp + rg + 1e-9)
                    d_true = d_perc if prov == "perceived" else -d_perc
                    items.append({"prov": prov, "pair": i, "winner": winner,
                                  "correct": bool(winner == prov), "d_true": float(d_true),
                                  "rate_perceived": float(rp), "rate_generated": float(rg),
                                  "content_rate": float(rec["content_rate"])})
        self._items = items
        return items


def _source_prov_reads(organ):
    """The organ's REAL opponent-provenance read battery: per-item perceived/generated/content rates + the signed
    discriminability, plus the aggregate min |d| and sign accuracy. Byte-identical merged-vs-coresident == the
    encode+recall provenance pipeline is co-residence-invariant."""
    import numpy as _np
    items = organ._recall_items()
    out = {}
    for k, it in enumerate(items):
        out[f"rate_perc_{k}"] = it["rate_perceived"]
        out[f"rate_gen_{k}"] = it["rate_generated"]
        out[f"content_{k}"] = it["content_rate"]
        out[f"d_true_{k}"] = it["d_true"]
    out["min_d_true"] = float(_np.min([it["d_true"] for it in items]))
    out["acc"] = float(_np.mean([it["correct"] for it in items]))
    return out


def _source_prov_answer(organ):
    """The rendered read-out: the per-item provenance verdict (which source the opponent comparator names)."""
    items = organ._recall_items()
    return tuple(it["winner"] for it in items)


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  CAUSAL WHAT-IF (T1-4) — a LEARNED, DIRECTED spiking forward model. Two seams (finding 2026-08-27):
#  (1) BUILD-TIME STDP+DA TRAIN — build_forward_model TRAINS temporal-order STDP + phasic-DA at BUILD then freezes;
#      the wire=True pool never runs that training, so the evt slice would be untrained. Seam: run the train as a
#      build-time step under a config toggle (enable_stdp/enable_reward_modulation are read live per step) + a
#      universal gain-0 freeze of every NON-evt edge (only the evt xblock edges stay plastic, so the STDP tags +
#      DA-gated three-factor updates are confined to the evt slice, co-residence-invariant), then freeze (enable_stdp
#      OFF) for the read. propagation_strength + current_time_step + the plasticity flags are SAVED and RESTORED so
#      no co-resident read on the shared bridge is perturbed.
#  (2) LIVE-COMPOSER answer — DECLARED RESIDUAL, NOT closed here. The production what_if/why ANSWER is rendered by a
#      live RFPhasorComposer (`_recalled(composer,e)` gates the emitted sentence). The organ-read gate closes the
#      SUBSTRATE forward-pass diagnostics (predicts_D / directed-ratio / cause-separation / do-intervention), which
#      the finding names as composer-INDEPENDENT + byte-identical once evt is trained; the NL rendering rides the
#      composer-grounding burn-down. The evt region carries NO RegionPathways (its xblock edges are injected
#      separately), so this organ supplies an explicit_wiring_fn to regenerate them per-region-seamed on the pool.
#  enable_nmda: evt opts OUT (region default False) -> the per-region mask + the pool's all-zero-mask reconciliation.
_CAUSAL_CONFIG = {
    "enable_stdp": False, "enable_hebbian_learning": False, "enable_homeostasis": False,
    "enable_short_term_plasticity": False, "enable_structural_plasticity": False,
    "enable_reward_modulation": False, **_NOISE_OFF,
}
# build_forward_model's build defaults (the xblock topology + the train/read operating point), reproduced here so
# the pool's evt slice trains BYTE-IDENTICALLY to a coresident evt slice (same rng, same reps, same protocol).
_CAUSAL_BLK = 30
_CAUSAL_N_EVENTS = 6


def _causal_wiring(bridge, rm):
    """explicit_wiring_fn: regenerate build_forward_model's cross-block `xblock` edges on the pool's evt slice.
    Cross-block only (i!=j), weak + plastic, NO within-block. Uses the SAME RandomState(seed+17) + loop order as
    build_forward_model, so the edge set is byte-identical (structurally) whether evt is alone or co-resident."""
    import numpy as np
    seed = int(bridge.core_config.seed)
    blk, n_events, init_w, xblock_density = _CAUSAL_BLK, _CAUSAL_N_EVENTS, 0.2, 0.6
    evt = np.asarray(rm.indices("evt"), dtype=np.int64)
    blocks = [evt[e * blk:(e + 1) * blk] for e in range(n_events)]
    rng = np.random.RandomState(seed + 17)
    pre, post, w = [], [], []
    for i in range(n_events):
        for j in range(n_events):
            if i == j:
                continue
            for a_ in blocks[i]:
                for b_ in blocks[j]:
                    if xblock_density >= 1.0 or rng.rand() < xblock_density:
                        pre.append(int(a_)); post.append(int(b_)); w.append(float(init_w))
    return {"xblock": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                       "plastic": True, "conn_type": "ff"}}


def _causal_organ(seed, shared):
    return _CausalReadOrgan(int(seed), shared)


class _CausalReadOrgan:
    """Wraps the causal forward-model for the organ-read gate: a BUILD-TIME temporal-order-STDP + phasic-DA train of
    the evt slice (under a config toggle + a gain-0 freeze of every non-evt edge + read_isolation), then the FROZEN
    substrate forward-pass reads (forward prediction, unseen-consequence rollout, DO-intervention). The composer-
    grounded NL answer is a declared residual; the substrate causal verdicts are the rendered read-out here."""

    def __init__(self, seed, shared=None):
        self.seed = int(seed)
        self._shared = shared
        self.bridge = None
        self.meta = None
        self._reads = None
        self._built = False

    def _guard(self):
        import contextlib
        if self._shared is not None:
            return self._shared.read_isolation("causal_whatif")
        return contextlib.nullcontext()

    def _freeze_non_evt(self, b, evt_arr, xp):
        """Universal gain-0 freeze of every edge that is NOT evt-internal (both endpoints in evt). tocoo() preserves
        cp_connections.data order, so the mask aligns with cp_plasticity_rate_gain (the framework's own freeze
        relies on this). Only the evt xblock edges keep gain 1.0 -> STDP+DA train the evt slice alone."""
        import numpy as np
        coo = b.cp_connections.tocoo()
        row = np.asarray(_host(coo.row)); col = np.asarray(_host(coo.col))
        row_in = np.isin(row, evt_arr); col_in = np.isin(col, evt_arr)
        both = row_in & col_in
        ng = np.zeros(row.shape[0], dtype=np.float32)
        ng[both] = 1.0
        b.cp_plasticity_rate_gain = xp.asarray(ng, dtype=xp.float32)

    def ensure_built(self):
        if self._built:
            return
        import numpy as np
        from sim.backend import get_backend
        from research.runners._causal_forward_model_derisk import (
            build_forward_model, train, OBS_EPISODES)
        xp, _ = get_backend()
        if self._shared is None:
            # standalone (for parity/testing): the de-risk's own build+train, then freeze + maturation gain.
            self.bridge, cc, self.meta = build_forward_model(self.seed)
            train(self.bridge, cc, self.meta, xp, OBS_EPISODES, obs_reps=30, interv_reps=30)
            cc.enable_stdp = False; cc.enable_reward_modulation = False
            cc.current_reward_signal = 0.0; cc.propagation_strength = 0.50
            self._reads = self._read_substrate(xp)
            self._built = True
            return

        pool = self._shared
        pool.ensure_built()
        b = pool.bridge
        cc = b.core_config
        blk, n_events = _CAUSAL_BLK, _CAUSAL_N_EVENTS
        evt = np.asarray(b.region_manager.indices("evt"), dtype=np.int64)
        blocks = [evt[e * blk:(e + 1) * blk] for e in range(n_events)]
        b._blocks = blocks
        b._blk = blk
        snap = pool.snap or {}
        b._rest_v = (np.asarray(snap["cp_membrane_potential_v"]).copy()
                     if "cp_membrane_potential_v" in snap else b.cp_membrane_potential_v.copy())
        b._rest_u = (np.asarray(snap["cp_recovery_variable_u"]).copy()
                     if "cp_recovery_variable_u" in snap else b.cp_recovery_variable_u.copy())
        self.bridge = b
        self.meta = dict(n_events=n_events, blk=blk)

        _cfg_keys = ("enable_stdp", "stdp_a_plus", "stdp_a_minus", "stdp_tau_plus_ms", "stdp_tau_minus_ms",
                     "stdp_w_max", "stdp_w_min", "enable_reward_modulation", "reward_defer_stdp_weight_update",
                     "reward_learning_rate", "reward_eligibility_tau_ms", "reward_baseline",
                     "current_reward_signal", "reward_aversive_scale", "propagation_strength")
        saved = {k: getattr(cc, k) for k in _cfg_keys}
        saved_tstep = b.runtime_state.current_time_step
        saved_tms = b.runtime_state.current_time_ms
        g0 = getattr(b, "cp_plasticity_rate_gain", None)
        saved_gain = g0.copy() if g0 is not None else None
        # The plasticity STATE arrays are allocated at BUILD only when the flags are on; the frozen pool builds with
        # them None, so toggling enable_stdp/enable_reward_modulation at train-time would silently no-op (the STDP
        # timing base + the DA eligibility trace never exist). Allocate them here (the exact shapes _initialize_
        # simulation_data uses under those flags), and restore to None after so no co-resident read is perturbed.
        saved_elig = getattr(b, "cp_eligibility_trace", None)
        saved_lst = getattr(b, "cp_last_spike_time", None)
        nnz = int(b.cp_connections.nnz)
        n_all = int(b.cp_membrane_potential_v.shape[0])
        with self._guard():
            try:
                # build_forward_model's DIRECTED-plasticity operating point (temporal-order STDP + DA three-factor)
                cc.enable_stdp = True
                cc.stdp_a_plus = 0.02; cc.stdp_a_minus = 0.010
                cc.stdp_tau_plus_ms = 12.0; cc.stdp_tau_minus_ms = 12.0
                cc.stdp_w_max = 24.0; cc.stdp_w_min = 0.0
                cc.enable_reward_modulation = True
                cc.reward_defer_stdp_weight_update = True
                cc.reward_learning_rate = 0.18; cc.reward_eligibility_tau_ms = 150.0
                cc.reward_baseline = 0.0; cc.current_reward_signal = 0.0
                cc.reward_aversive_scale = 1.0; cc.propagation_strength = 0.05
                if saved_elig is None:
                    b.cp_eligibility_trace = xp.zeros(nnz, dtype=xp.float32)
                if saved_lst is None:
                    b.cp_last_spike_time = xp.full(n_all, -1000.0, dtype=xp.float32)
                self._freeze_non_evt(b, evt, xp)
                train(b, cc, self.meta, xp, OBS_EPISODES, obs_reps=30, interv_reps=30)
                # FREEZE the learned structure + the uniform maturation gain (the de-risk read protocol).
                cc.enable_stdp = False; cc.enable_reward_modulation = False
                cc.current_reward_signal = 0.0; cc.propagation_strength = 0.50
                self._reads = self._read_substrate(xp)
            finally:
                for k, v in saved.items():
                    setattr(cc, k, v)
                b.runtime_state.current_time_step = saved_tstep
                b.runtime_state.current_time_ms = saved_tms
                if saved_gain is not None:
                    b.cp_plasticity_rate_gain[:] = saved_gain
                b.cp_eligibility_trace = saved_elig
                b.cp_last_spike_time = saved_lst
        self._built = True

    def _read_substrate(self, xp):
        """RAW (unrounded) spiking substrate diagnostics on the FROZEN trained evt slice — the composer-independent
        causal reads. All are `cp_firing_states` reads via the de-risk's held-read primitive."""
        from research.runners._causal_forward_model_derisk import (
            _held_read, _xblock_weight, CHAIN_EDGES, A, B, C, D, X, Y)
        b, blocks = self.bridge, self.bridge._blocks
        out = {}
        # 1-step forward prediction on the chain edges (raw successor rate + argmax correctness)
        n_correct = 0
        for src, tgt in CHAIN_EDGES:
            rates = _held_read(b, blocks, xp, src)
            pred = max((e for e in range(len(blocks)) if e != src), key=lambda e: rates[e])
            n_correct += int(pred == tgt)
            out[f"succ_rate_{src}_{tgt}"] = float(rates[tgt])
        out["fwd_acc"] = float(n_correct / len(CHAIN_EDGES))
        # directedness: hold B -> D fires; hold D -> B does not
        rb = _held_read(b, blocks, xp, B)
        rd = _held_read(b, blocks, xp, D)
        out["directed_fwd_BtoD"] = float(rb[D])
        out["directed_rev_DtoB"] = float(rd[B])
        # unseen 2-step consequence (roll A->B->D forward)
        ra = _held_read(b, blocks, xp, A, read_steps=20)
        out["D_rate"] = float(ra[D]); out["B_rate"] = float(ra[B])
        out["offchain_max"] = float(max(ra[C], ra[X], ra[Y]))
        # DO-intervention cause vs correlation
        out["Y_do_X"] = float(_held_read(b, blocks, xp, X)[Y])
        out["Y_do_C"] = float(_held_read(b, blocks, xp, C)[Y])
        # learned-weight probes (the DIRECT A->D must stay unlearned; the chain + genuine C->Y consolidate)
        out["w_AB"] = float(_xblock_weight(b, A, B))
        out["w_CY"] = float(_xblock_weight(b, C, Y))
        out["w_XY"] = float(_xblock_weight(b, X, Y))
        out["w_AD"] = float(_xblock_weight(b, A, D))
        return out


def _causal_reads(organ):
    """The organ's REAL directed-forward-model read battery (RAW rates + learned weights). Byte-identical
    merged-vs-coresident == the whole STDP+DA-trained forward model reads co-residence-invariantly."""
    organ.ensure_built()
    return dict(organ._reads)


def _causal_answer(organ):
    """The rendered SUBSTRATE causal verdict (composer-independent): forward prediction correct, the unseen 2-step
    consequence D predicted, and X-not-a-cause-of-Y under the DO-intervention. The composer-grounded NL rendering
    is the declared residual (rides the composer-grounding burn-down)."""
    organ.ensure_built()
    r = organ._reads
    predicts_D = bool(r["D_rate"] > max(r["offchain_max"], 1.0) * 1.5
                      and r["B_rate"] >= r["D_rate"] * 0.8 and r["w_AD"] < 1.0)
    x_not_cause = bool(r["Y_do_X"] < max(r["Y_do_C"], 1.0) * 0.5)
    return (float(r["fwd_acc"]) >= 1.0, predicts_D, x_not_cause)


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  PROSPECTIVE MEMORY (faculty-map Tier-2) — the ONE MULTI-TURN STATEFUL read. Unlike the frozen-forward-pass and
#  the build-time-plasticity-then-frozen organs, its read is a SEQUENCE of turns whose STATE (a self-sustaining
#  cortex<->dlpfc attractor holding a deferred intention + a per-neuron SFA trace on the rel cue-monitor pools)
#  must PERSIST UNRESET across the turns (form the intention -> hold through N intervening distractor turns -> the
#  cue releases it). Two seams the finding named, both now closed:
#   (1) DEEP-HIERARCHY injection — the read runs on base ProspectiveMemory -> HomeostaticProspectiveMemory ->
#       SFANmdaProspectiveMemory; each builds/steps the substrate in __init__ (a multi-STAGE homeostat + NMDA-
#       plateau CALIBRATION). Threaded with an additive `shared=None` through all three (byte-identical when None):
#       the base ADOPTS pool.bridge, the attractor + cue-monitor edges move to this descriptor's explicit_wiring_fn
#       (build-time, both arms identical), and each subclass's calibration re-homes onto the pool slice. The
#       per-seed bias/theta module caches are BYPASSED when shared (each arm calibrates independently -> the
#       byte-identity is genuine, not a cache hit).
#   (2) MULTI-TURN HOLD — the whole calibrate + form/hold/cue SEQUENCE runs inside ONE pool.sequence_isolation()
#       guard (the general per-SEQUENCE scope): the pmem slice evolves UNRESET across every turn, and the full
#       snapshot (per-neuron + per-synapse + timing) is restored at guard exit so no co-resident organ is
#       perturbed even by the whole-bridge _reset_dynamics the read calls between its sub-sequences.
#  enable_nmda: every pmem region opts IN per-region (loop_reg / rel BrainRegions carry enable_nmda=True), so the
#  per-region NMDA mask includes the pmem slice in BOTH arms; every plasticity flag OFF (config-compatible union).
#  HOST-SCAFFOLD (flagged, unchanged from the de-risk): the cue->action CONTENT binding is installed synaptically;
#  the SFA K-adaptation current + the NMDA-plateau boost are host current-injection PROXIES. The MECHANISM
#  (hold-across-turns + coincidence-gated release + homeostatic operating-point control) is brain-based.
_PMEM_CONFIG = {
    "enable_nmda": True,
    "enable_stdp": False, "enable_hebbian_learning": False, "enable_homeostasis": False,
    "enable_short_term_plasticity": False, "enable_structural_plasticity": False,
    "enable_reward_modulation": False, "fast_spike_reset": True, **_NOISE_OFF,
}
# pmem's attractor + SFA timescales are tuned at dt=0.5 (the de-risk operating point); the pool's global dt is 1.0
# (the other 5 organs' verified point). Reconciled pmem-LOCALLY: the read organ sets cc.dt_ms + the delay horizon
# to 0.5 (and rescales the cached conductance decays exactly, decay**(new/old)=exp(-new/tau)) INSIDE its
# sequence_isolation guard, restoring at exit -- so ONLY pmem's slice runs at dt=0.5, the other organs stay
# byte-identical at dt=1.0, and the read is byte-identical merged-vs-coresident (both arms set 0.5).
_PMEM_DT = 0.5
# POOL SYNAPTIC GAIN. The merge substrate uses num_traits=1 (REQUIRED for co-residence byte-identity: with
# num_traits>1 the per-neuron trait draw is a global-RNG index-order draw -> co-residence-DEPENDENT). The pmem
# de-risk was tuned at num_traits=5, whose multi-type trait draw delivers ~6x MORE effective synaptic current per
# unit weight (empirically: the cortex<->dlpfc attractor needs weight ~300 to self-sustain on the pool vs 50
# standalone). Since the homeostat's tonic bias is EXTERNAL current (pA, trait-independent), the balance between
# synaptic drive and bias is restored by scaling EVERY pmem SYNAPTIC weight (attractor, cue-monitor, rel-recurrent)
# by this one gain -> the delivered currents match the standalone's, so the whole tuned attractor+cue-monitor+
# homeostat+plateau balance transfers. Byte-identical merged-vs-coresident (both arms use the SAME gained weights).
_PMEM_POOL_GAIN = 6.0
# the pmem build defaults (must match base ProspectiveMemory.__init__ so the pool slice trains BYTE-IDENTICALLY to
# a coresident slice AND functionally reproduces the standalone edge structure), scaled by the pool gain.
_PMEM_ACTIONS = ["A", "B"]
_PMEM_DISTRACTORS = ["d0", "d1", "d2", "d3"]
_PMEM_PSIZE = 40
_PMEM_NREL = 60
_PMEM_N = 800
_PMEM_ATTRACTOR_W = 50.0 * _PMEM_POOL_GAIN
_PMEM_HOLD_W = 3.2 * _PMEM_POOL_GAIN
_PMEM_CUE_W = 4.2 * _PMEM_POOL_GAIN
_PMEM_REL_RECURRENT = 0.10 * _PMEM_POOL_GAIN
_PMEM_N_INTERVENING = 5
# SFANmdaProspectiveMemory params re-tuned for the num_traits=1 / noise-off pool operating point (the de-risk's
# defaults assume the more-excitable num_traits=5 + conductance-noise standalone). Two seam-forced changes:
#   * homeostat_bias_max > 0 -> BIDIRECTIONAL homeostasis (Turrigiano's set-point control lifts a hypo-excitable
#     pool too, not only hyperpolarizes) -- the noise-free pool's rel accumulator sits BELOW rheobase at bias 0.
#   * stronger SFA (sfa_g) + plateau (plateau_g/cap) -> the sustained-hold ramp is adapted away and the transient
#     coincidence is supralinearly amplified, so the release clears the single-input silence on the weaker pool.
# Identical for every seed (label-free); byte-identical merged-vs-coresident (both arms use the SAME params).
_PMEM_READ_PARAMS = dict(homeostat_bias_max=4000.0, homeostat_r_set=0.035,
                         sfa_g=20000.0, sfa_tau=60.0, plateau_g=40000.0, plateau_cap=30000.0)


def _pmem_wiring(bridge, rm):
    """explicit_wiring_fn: regenerate the base ProspectiveMemory's outer-product attractor loops (c2d + d2c) and
    the cue-monitor coincidence edges (act->rel + cue->rel) on the pool's cortex/dlpfc/rel slices. Uses the SAME
    np.random.default_rng(seed).permutation(n) assembly assignment + the SAME weights as the standalone __init__,
    so the edge set is byte-identical (structurally) whether the pmem slice is alone or co-resident (name-keyed
    region indices; the base 0-weight pathways + the rel-internal recurrence come from build_wiring_plan)."""
    import numpy as np
    seed = int(bridge.core_config.seed)
    cidx = np.asarray(rm.indices("cortex_ctx"), dtype=np.int64)
    didx = np.asarray(rm.indices("dlpfc_wm"), dtype=np.int64)
    n = int(cidx.size)
    ps = _PMEM_PSIZE
    attractor_concepts = list(_PMEM_ACTIONS) + list(_PMEM_DISTRACTORS)
    cue_names = [f"cue_{a}" for a in _PMEM_ACTIONS]
    all_asm = attractor_concepts + cue_names
    perm = np.random.default_rng(seed).permutation(n)
    cpat, dpat = {}, {}
    for i, name in enumerate(all_asm):
        p = perm[i * ps:(i + 1) * ps]
        cpat[name] = cidx[p]
        if name in attractor_concepts:
            dpat[name] = didx[p]
    # c2d / d2c attractor outer products (per attractor concept)
    c2d_pre, c2d_post, d2c_pre, d2c_post = [], [], [], []
    for name in attractor_concepts:
        cp_, dp_ = cpat[name], dpat[name]
        c2d_pre.append(np.repeat(cp_, ps)); c2d_post.append(np.tile(dp_, ps))
        d2c_pre.append(np.repeat(dp_, ps)); d2c_post.append(np.tile(cp_, ps))
    c2d_pre = np.concatenate(c2d_pre).astype(np.int64); c2d_post = np.concatenate(c2d_post).astype(np.int64)
    d2c_pre = np.concatenate(d2c_pre).astype(np.int64); d2c_post = np.concatenate(d2c_post).astype(np.int64)
    aw = np.full(c2d_pre.size, np.float32(_PMEM_ATTRACTOR_W), np.float32)
    # cue-monitor coincidence edges (act->rel at hold_w, cue->rel at cue_w) per action
    cm_pre, cm_post, cm_w = [], [], []
    for a in _PMEM_ACTIONS:
        relX = np.asarray(rm.indices(f"rel_{a}"), dtype=np.int64)
        actc = cpat[a]; cuec = cpat[f"cue_{a}"]
        cm_pre.append(np.repeat(actc, relX.size)); cm_post.append(np.tile(relX, actc.size))
        cm_w.append(np.full(actc.size * relX.size, np.float32(_PMEM_HOLD_W), np.float32))
        cm_pre.append(np.repeat(cuec, relX.size)); cm_post.append(np.tile(relX, cuec.size))
        cm_w.append(np.full(cuec.size * relX.size, np.float32(_PMEM_CUE_W), np.float32))
    cm_pre = np.concatenate(cm_pre).astype(np.int64); cm_post = np.concatenate(cm_post).astype(np.int64)
    cm_w = np.concatenate(cm_w).astype(np.float32)
    mk = lambda pre, post, w: {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                               "plastic": False, "conn_type": "E_TO_E", "count": int(pre.size)}
    return {"c2d": mk(c2d_pre, c2d_post, aw), "d2c": mk(d2c_pre, d2c_post, aw),
            "cue_monitor": mk(cm_pre, cm_post, cm_w)}


def _pmem_organ(seed, shared):
    return _PMemReadOrgan(int(seed), shared)


class _PMemReadOrgan:
    """Wraps the 3-class prospective-memory hierarchy for the organ-read gate: constructs SFANmdaProspectiveMemory
    against the pool (which re-homes the homeostat + NMDA-plateau CALIBRATION onto the pmem slice) and runs the
    MULTI-TURN form->hold->cue read SEQUENCE, ALL inside ONE pool.sequence_isolation() guard so the held intention
    + SFA trace persist unreset across the turns while no co-resident organ is perturbed. Numeric reads are cached
    eagerly (before the guard exits). shared=None -> the standalone build+read, byte-identical to the de-risk."""

    def __init__(self, seed, shared=None):
        self.seed = int(seed)
        self._shared = shared
        self._reads = None
        self._built = False

    def _guard(self):
        import contextlib
        if self._shared is not None:
            return self._shared.sequence_isolation()
        return contextlib.nullcontext()

    @staticmethod
    def _enter_local_dt(b, new_dt):
        """Set the pmem operating-point dt on the shared bridge WITHOUT rebuilding: pmem's attractor/SFA timescales
        are tuned at dt=0.5 while the pool runs at dt=1.0 (the other organs' point). The conductance decays are
        CACHED at build as exp(-dt/tau); rescale each EXACTLY as decay**(new_dt/old_dt)=exp(-new_dt/tau) (no tau
        needed) so g_e/g_i/g_nmda dynamics move to the new dt, and set the delay horizon. Returns a restore token."""
        cc = b.core_config
        old_dt = float(cc.dt_ms)
        saved = {"dt": old_dt, "mds": b.runtime_state.max_delay_steps, "caches": {}}
        ratio = float(new_dt) / old_dt if old_dt > 0 else 1.0
        for k in [k for k in list(vars(b)) if k.startswith("_cached_decay_")]:
            v = getattr(b, k, None)
            saved["caches"][k] = v
            if isinstance(v, (int, float)) and v > 0.0:
                setattr(b, k, float(v) ** ratio)
        cc.dt_ms = float(new_dt)
        b.runtime_state.max_delay_steps = int(cc.max_synaptic_delay_ms / new_dt) if new_dt > 0 else 200
        return saved

    @staticmethod
    def _exit_local_dt(b, saved):
        b.core_config.dt_ms = saved["dt"]
        b.runtime_state.max_delay_steps = saved["mds"]
        for k, v in saved["caches"].items():
            setattr(b, k, v)

    def ensure_built(self):
        if self._built:
            return
        from research.runners._pmem_sfa_nmda_amplifier_derisk import SFANmdaProspectiveMemory
        params = dict(_PMEM_READ_PARAMS) if self._shared is not None else {}
        with self._guard():
            restore = self._enter_local_dt(self._shared.bridge, _PMEM_DT) if self._shared is not None else None
            try:
                pm = SFANmdaProspectiveMemory(list(_PMEM_ACTIONS), list(_PMEM_DISTRACTORS),
                                              homeostat_on=True, sfa_on=True, plateau_on=True,
                                              seed=self.seed, shared=self._shared, **params)
                self._reads = self._run_multiturn(pm)
            finally:
                if restore is not None:
                    self._exit_local_dt(self._shared.bridge, restore)
        self._built = True

    def _run_multiturn(self, pm):
        """The genuine multi-turn stateful read: three form/hold/cue sub-sequences on ONE pm (the slice reset to
        rest between them via pm._reset_dynamics), exercising the cross-turn hold. Reports the RAW rel firing rates
        so byte-identity merged-vs-coresident is exact and the fire-vs-silent SEPARATION is non-degenerate:
          FIRE       : form A, HOLD N intervening distractor turns, present cue A -> rel_A must ramp (the release).
          WRONG-CUE  : form A, HOLD N turns, present cue B -> rel_A/rel_B silent (the monitor is cue-specific).
          NO-INTENT  : never form, present cue A -> rel_A silent (the fire is gated by the HELD intention, not the
                       cue alone -> the coincidence is real). held_min over the FIRE hold witnesses persistence."""
        N = _PMEM_N_INTERVENING
        dists = list(_PMEM_DISTRACTORS)
        inter = [dists[i % len(dists)] for i in range(N)]
        out = {}
        # FIRE-ON-CUE (+ persistence + no-fire-before)
        pm._reset_dynamics()
        pm.encode_intention("A")
        held_trace, before_trace = [], []
        for d in inter:
            r = pm.intervening_turn(d)
            held_trace.append(float(r["held"]["A"])); before_trace.append(float(r["rel"]["A"]))
        fire = pm.present_cue("A")
        out["fire_A_on_cueA"] = float(fire["rel"]["A"])
        out["fire_B_on_cueA"] = float(fire["rel"]["B"])
        out["held_min"] = float(min(held_trace)) if held_trace else 0.0
        out["rel_before_max"] = float(max(before_trace)) if before_trace else 0.0
        # WRONG-CUE (form A, present cue B)
        pm._reset_dynamics()
        pm.encode_intention("A")
        for d in inter:
            pm.intervening_turn(d)
        wrong = pm.present_cue("B")
        out["wrongcue_rel_A"] = float(wrong["rel"]["A"])
        out["wrongcue_rel_B"] = float(wrong["rel"]["B"])
        # NO-INTENTION (never form, present cue A)
        pm._reset_dynamics()
        for d in inter:
            pm.intervening_turn(d)
        noint = pm.present_cue("A")
        out["noint_rel_A"] = float(noint["rel"]["A"])
        out["max_silent"] = float(max(out["fire_B_on_cueA"], out["rel_before_max"], out["wrongcue_rel_A"],
                                      out["wrongcue_rel_B"], out["noint_rel_A"]))
        out["fire_min"] = float(out["fire_A_on_cueA"])
        # SAME-POOL non-degeneracy: the correct-cue release vs rel_A's OWN silence (before-cue hold ramp, wrong-cue,
        # no-intention) -- the coincidence-gated release on the TARGET pool, independent of the off-pool (rel_B)
        # lift-bias baseline. This is the clean margin (5-10x on 5/6 seeds); fire_B_on_cueA is the cross-pool
        # specificity residual (the positive homeostat bias needed to lift the hypo-excitable noise-free pool).
        out["same_pool_silent"] = float(max(out["rel_before_max"], out["wrongcue_rel_A"], out["noint_rel_A"]))
        return out


def _pmem_reads(organ):
    """The organ's REAL multi-turn prospective-memory read battery (RAW rel firing rates + the held-min
    persistence witness). Byte-identical merged-vs-coresident == the whole stateful form/hold/cue pipeline (deep-
    hierarchy calibration included) is co-residence-invariant."""
    organ.ensure_built()
    return dict(organ._reads)


def _pmem_answer(organ):
    """The rendered read-out: (released_on_correct_cue, silent_on_wrong_cue_same_pool, silent_with_no_intention) —
    the prospective-memory verdict. `released` is a coincidence-gated SAME-POOL decision (the correct-cue release
    clears 2x its own single-input/no-intention silence AND an absolute floor), NOT the de-risk's absolute
    FIRE_THR=0.20: the noise-free num_traits=1 pool operates at a lower release amplitude (~0.05-0.09 vs ~0.4), so
    the faithful read-out is the coincidence SEPARATION, not the standalone's absolute magnitude. Deterministic ->
    byte-identical merged-vs-coresident (both arms agree)."""
    organ.ensure_built()
    r = organ._reads
    fire = r["fire_A_on_cueA"]
    released = bool(fire >= max(2.0 * r["same_pool_silent"], 0.03))
    silent_wrong = bool(r["wrongcue_rel_A"] < fire and r["wrongcue_rel_A"] <= 0.03)
    silent_noint = bool(r["noint_rel_A"] < fire)
    return (released, silent_wrong, silent_noint)


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  CURIOSITY organ-read plumbing (2026-08-27) — the NEUROMODULATOR-SUBSYSTEM + OU seam, closed. The read is a
#  FROZEN forward pass of the spiking ASK-pool WANT (Hz) at a NOVEL vs FAMILIAR epistemic gap, driven by the
#  `from_novelty` -> excitability_drive `curiosity` neuromodulator (the ASK pool has NO afferent -> it fires ONLY
#  via the modulator + its OU background drive). Its ONE co-residence-dependent input was the ASK pool's OU: the
#  global cp.random.randn(n) draw is index-order, so a merged offset changed the ASK realization. CLOSED by the
#  per-neuron-seeded OU stream (cfg.per_neuron_ou_seed, sim/bridge.py) — each ASK neuron keyed on (region,
#  within-region rank), co-residence-invariant. BOTH the OU process AND the neuromodulator subsystem are built
#  LOCALLY in the read window (the pool config keeps enable_ou_process + the neuromod subsystem OFF, so it unions
#  cleanly with the OU-off frozen organs and leaves every co-resident slice untouched), inside sequence_isolation
#  (restores the full per-neuron/per-synapse state + the RNG cursor on exit) — the source_provenance/causal_whatif
#  local-toggle pattern, adapted to OU + neuromod. NON-DEGENERATE: want(novel) >> want(familiar) by design, so the
#  ASK drive genuinely tracks the gap.
_CUR_NOVEL = 0.95       # an ABSTAIN: the brain holds NO answer -> maximal epistemic gap (novel)
_CUR_FAMILIAR = 0.0     # a held concept: no gap (the calibration low anchor)
_CUR_READ_REPS = 4      # average the ASK-pool want over N reads (denoises the OU jitter; the read is drift-free)


def _curiosity_modulator_cfg():
    """The single `curiosity` neuromodulator: from_novelty -> excitability_drive on group:ask (exactly the DR-1
    fill build_curiosity_bridge registers; rebuilt here so the read owns it locally on the pool bridge)."""
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule
    return NeuromodulatorConfig(
        name="curiosity", baseline=0.0, decay_tau_ms=50.0,
        concentration_min=0.0, concentration_max=5.0,
        targets=[ModulatorTarget(target_type="excitability_drive", scope="group:ask", sensitivity=320.0)],
        production_rules=[ProductionRule(rule_type="from_novelty", sensitivity=0.10)])


class _CuriosityReadOrgan:
    """Organ-read wrapper for the spiking curiosity (crave) faculty. shared=None -> the shipped production organ's
    own-bridge read (byte-identical). shared=<MergedPool> -> the LOCAL-INIT read on the pool slice (OU per-neuron +
    neuromod built + torn down inside sequence_isolation)."""

    def __init__(self, seed, shared=None):
        self.seed = int(seed)
        self._shared = shared
        self._reads = None
        self._answer = None
        self._built = False

    def ensure_built(self):
        if self._built:
            return
        if self._shared is None:
            self._build_standalone()
        else:
            self._build_shared()
        self._built = True

    def _build_standalone(self):
        from research.runners.curiosity_production_organ import CuriosityProductionOrgan
        org = CuriosityProductionOrgan(self.seed)
        jn = org.judge(novelty=_CUR_NOVEL)
        jf = org.judge(novelty=_CUR_FAMILIAR)
        self._finalize(jn["want_hz"], jf["want_hz"])

    def _build_shared(self):
        from research.runners._curiosity_seek_learn_onbridge_derisk import (
            _settle, _snapshot_state, _restore_state, _advance, W_WANT, W_SETTLE)
        from sim.neuromodulators import NeuromodulatorManager
        pool = self._shared
        pool.ensure_built()
        b = pool.bridge
        cfg = b.core_config
        xp = pool.xp
        n = int(b.cp_membrane_potential_v.shape[0])
        idx_ask = np.asarray(sorted(int(i) for i in _region_indices(b, "ask")), dtype=np.int64)
        idx_ask_x = xp.asarray(idx_ask)
        n_ask = int(len(idx_ask))
        wn = wf = 0.0
        with pool.sequence_isolation():
            saved = {k: getattr(cfg, k, None) for k in (
                "enable_ou_process", "per_neuron_ou_seed", "ou_seed", "ou_std_current_pA",
                "ou_mean_current_pA", "ou_tau_ms", "enable_neuromodulator_subsystem",
                "neuromodulators", "reward_learning_rate", "current_novelty_signal")}
            saved_nm = b.neuromodulator_manager
            try:
                # OU (per-neuron seeded -> co-residence-invariant ASK drive), built LOCALLY on the pool slice.
                cfg.enable_ou_process = True
                cfg.per_neuron_ou_seed = True
                cfg.ou_seed = int(self.seed)
                cfg.ou_std_current_pA = 100.0
                cfg.ou_mean_current_pA = 0.0
                cfg.ou_tau_ms = 15.0
                b._initialize_ou_process_state(cfg, n)
                # neuromod subsystem (the from_novelty -> excitability_drive curiosity modulator), built LOCALLY.
                cfg.enable_neuromodulator_subsystem = True
                cfg.neuromodulators = [_curiosity_modulator_cfg()]
                b.neuromodulator_manager = NeuromodulatorManager(cfg.neuromodulators, cfg.dt_ms)
                b.neuromodulator_manager.initialize(n, xp)
                if b.region_manager is not None:
                    b.neuromodulator_manager.set_group_indices(b.region_manager.region_indices_dict())
                cfg.reward_learning_rate = 0.0
                _settle(b, W_SETTLE)
                snap0 = _snapshot_state(b)

                def read_want(novelty):
                    vals = []
                    for _ in range(_CUR_READ_REPS):
                        _restore_state(b, snap0)
                        cfg.current_novelty_signal = float(novelty)
                        cfg.reward_learning_rate = 0.0
                        spk = 0
                        for _ in range(W_WANT):
                            _advance(b)
                            spk += int(b.cp_firing_states[idx_ask_x].sum())
                        vals.append(spk / max(n_ask, 1) / (W_WANT * 1e-3))
                    _restore_state(b, snap0)
                    return float(np.mean(vals))

                wn = read_want(_CUR_NOVEL)
                wf = read_want(_CUR_FAMILIAR)
            finally:
                # tear down every locally-installed piece so no co-resident organ's read is perturbed.
                b.neuromodulator_manager = saved_nm
                b.cp_ou_current = None
                b._region_ou_streams = None
                b._ou_neuron_key_idx = None
                b._ou_neuron_keys = None
                b._ou_pn_step = 0
                for k, v in saved.items():
                    setattr(cfg, k, v)
        self._finalize(wn, wf)

    def _finalize(self, want_novel, want_familiar):
        nondegen = bool(want_novel > want_familiar + 1.0)
        threshold = 0.5 * (want_novel + want_familiar) if nondegen else float(want_familiar)
        curious = bool(want_novel >= threshold) and nondegen
        self._reads = {"want_novel_hz": float(want_novel), "want_familiar_hz": float(want_familiar),
                       "threshold_hz": float(threshold)}
        from research.runners.curiosity_production_organ import followup_question
        # the rendered answer: the honest curiosity FOLLOW-UP QUESTION when the ASK pool craves (topic fixed so the
        # answer-preservation compare isolates the spiking crave verdict, not the language scaffold).
        self._answer = followup_question("wombats") if curious else ""

    def reads(self):
        self.ensure_built()
        return dict(self._reads)

    def answer(self):
        self.ensure_built()
        return self._answer


def _curiosity_organ(seed, shared=None):
    return _CuriosityReadOrgan(int(seed), shared=shared)


def _curiosity_reads(organ):
    return organ.reads()


def _curiosity_answer(organ):
    return organ.answer()


GROUP_A = [
    OrganDescriptor(key="causal_whatif",
                    regions=("evt",),
                    spec_fn=_spec_causal_whatif,
                    config=_CAUSAL_CONFIG,
                    explicit_wiring_fn=_causal_wiring,
                    organ_cls=_causal_organ, read_fn=_causal_reads, answer_fn=_causal_answer,
                    supports_shared=True,
                    scaffold_residuals=("host-injected DA sign at train time (declared teacher signal); "
                                        "composer-grounded NL what-if answer (substrate causal verdict is closed)",)),
    OrganDescriptor(key="comprehension",
                    regions=("sel_agent", "sel_FS_agent", "sel_patient", "sel_FS_patient",
                             "cue_position_pos", "cue_position_neg", "cue_animacy_pos", "cue_animacy_neg",
                             "cue_verbfit_pos", "cue_verbfit_neg", "cue_lexbias_pos", "cue_lexbias_neg"),
                    spec_fn=_spec_comprehension,
                    config=_COMPREHENSION_CONFIG,
                    organ_cls=_comprehension_organ, read_fn=_comprehension_reads,
                    answer_fn=_comprehension_answer, supports_shared=True),
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
                    spec_fn=_spec_source_provenance, param_het=True,
                    config=_SOURCE_PROV_CONFIG,
                    organ_cls=_source_prov_organ, read_fn=_source_prov_reads,
                    answer_fn=_source_prov_answer, supports_shared=True,
                    scaffold_residuals=("caller-supplied sparse episode/content activity + innate context routing "
                                        "(the learned trace is the encode; unchanged from the de-risk)",)),
    OrganDescriptor(key="curiosity",
                    regions=("cue", "striosome_value", "reward_us", "snc", "ask"),
                    spec_fn=_spec_curiosity, param_het=True,
                    organ_cls=_curiosity_organ, read_fn=_curiosity_reads,
                    answer_fn=_curiosity_answer, supports_shared=True,
                    scaffold_residuals=("host-derived novelty scalar (the abstain = the epistemic gap, a declared "
                                        "host boundary; a graded familiarity-gate novelty is the next rung); fixed "
                                        "wh-frame follow-up language scaffold; the learning-progress SELECTOR + "
                                        "noisy-TV veto are not wired (a single-topic follow-up needs neither)",)),
    OrganDescriptor(key="prospective_memory",
                    regions=("cortex_ctx", "dlpfc_wm", "rel_A", "rel_B"),
                    spec_fn=_spec_prospective_memory,
                    config=_PMEM_CONFIG,
                    explicit_wiring_fn=_pmem_wiring,
                    organ_cls=_pmem_organ, read_fn=_pmem_reads, answer_fn=_pmem_answer,
                    supports_shared=True,
                    scaffold_residuals=("host-installed cue->action content binding (Gollwitzer one-shot Hebbian "
                                        "potentiation is the named follow-on); SFA + NMDA-plateau host current-"
                                        "injection proxies for the K-adaptation conductance + dendritic NMDA spike",)),
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

# ORGAN-READ deferrals — the Group-A organ(s) that ARE substrate-init byte-identical (migration-safe) but whose
# READ pipeline is not YET a clean function of the frozen shared substrate. Each names the concrete engine/wrapper
# seam it needs (honest boundary: substrate-init is the migration gate; organ-read is this rung, now closed for
# SIX organs — self_schema, d6_multiref_wm, comprehension [frozen forward passes] + source_provenance, causal_whatif
# [build-time-plasticity-then-frozen-read via the toggle + gain-0-freeze seam] + prospective_memory [the MULTI-TURN
# stateful read via sequence_isolation + dt-local + the pool-gain seam, 2026-08-27] — closed IN CO-RESIDENCE with
# the frozen-forward organs, 6-seed GO). Only `curiosity` remains organ-read-deferred (a further seam), and it does
# NOT block the batch. Refined 2026-08-27.
# curiosity's organ-read is now CLOSED (2026-08-27): its NEUROMODULATOR-SUBSYSTEM + OU seam is resolved by the
# per-neuron-seeded OU stream (cfg.per_neuron_ou_seed, sim/bridge.py) + a LOCAL-INIT read that builds+tears down
# BOTH the OU process and the curiosity neuromodulator on the pool slice inside sequence_isolation (organ_cls=
# _curiosity_organ). The pool config keeps enable_ou_process + the neuromod subsystem OFF, so it unions cleanly
# with the OU-off frozen organs and every co-resident slice stays byte-identical. See the descriptor + the
# 2026-08-27 per-neuron-OU finding. No Group-A organ remains organ-read-deferred.
GROUP_A_ORGANREAD_DEFERRED = {}

# prospective_memory's MULTI-TURN stateful read is now CLOSED in co-residence with the frozen-forward organs
# (organ_cls=_pmem_organ; sequence_isolation + the dt-local + pool-gain seams; 6-seed GO, read_maxerr=0 +
# answer_same, held robust 0.329..0.344, coincidence-gated release 3.2..8.5x same-pool margin). The FULL 7-organ
# strict batch is NO-GO for a SCALE reason, not a pmem defect: adding pmem (the LARGEST organ, 1720 neurons) pushes
# total-N=4968 past the point where the OTHER LONG-INTEGRATION reads (source_provenance / causal_whatif / d6) stay
# byte-identical, because the engine's non-flagged conductance-matvec paths (notably the slow-NMDA-recurrent
# `_nr_mat.T @ prev_firing`, sim/bridge.py:8973, used ONLY by d6) carry an FP summation-order variance that a
# hundreds-of-steps spiking read amplifies into a 1-spike divergence. Merge seam #2 (deterministic_transpose_matvec)
# pins the MAIN synaptic matvec (recovers pmem; the original 5 stay 5/5 GO without pmem), but closing the full batch
# needs deterministic variants of ALL matvec paths — a sim/ edit. See the 2026-08-27 multi-turn-stateful-read finding.

# causal_whatif's SUBSTRATE read (predicts_D / directed-ratio / cause-separation / DO-intervention) is now CLOSED
# via the build-time STDP+DA train seam (organ_cls=_causal_organ). Blocker (2) — the BUILD-TIME train — is solved
# (config toggle + gain-0 freeze + allocating the plasticity state arrays). Blocker (1) — the composer-grounded NL
# what_if/why ANSWER — is the DECLARED residual (the substrate causal VERDICT is the rendered read-out here); it
# rides the composer-grounding burn-down and does NOT block the organ-read migration gate. See the descriptor's
# scaffold_residuals + the 2026-08-27 organ-read-engine-seams finding.

GROUP_A_KEYS = [d.key for d in GROUP_A]

REGISTRY = {d.key: d for d in (SURPRISE, WORLDMODEL, METACOG, PRAGMATIC, *GROUP_A)}


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


def _smoke2(seed: int = 42) -> dict:
    """Build the engine pool (metacog+pragmatic via the REGISTRY, wire=True -- pool #2's wiring inject is
    ALWAYS-ON in the shipped class, unlike pool #1's post_build-only wiring) and the shipped `MergedSubstrate2`,
    and compare (1) every per-neuron INIT array over both organs' regions AND (2) the REAL shipped production
    organs' reads (`MetacogProductionOrgan.judge` / `PragmaticProductionOrgan.interpret`) run UNMODIFIED against
    EACH pool. Byte-identity on both proves the declarative descriptor path reproduces the bespoke pool #2
    exactly -- the round-trip `_smoke` proves for pool #1, extended here past init-only to the organs' actual
    production read pipeline (the stronger bar; pool #2's organs already carry `shared=` in production)."""
    from research.runners.onebrain_merge_production2 import MergedSubstrate2
    from research.runners.metacog_production_organ import MetacogProductionOrgan
    from research.runners.pragmatic_production_organ import PragmaticProductionOrgan

    pool = merge_organs([METACOG, PRAGMATIC], seed=seed, wire=True)
    shipped = MergedSubstrate2(seed=seed, organs=("metacog", "pragmatic"))
    shipped.ensure_built()

    all_regions = list(METACOG.regions) + list(PRAGMATIC.regions)
    worst, worst_where = 0.0, None
    for rname in all_regions:
        ei = _region_slice(pool.bridge, rname)
        si = _region_slice(shipped.bridge, rname)
        if ei.size != si.size:
            return {"seed": seed, "byte_identical": False, "all_go": False,
                    "reason": f"{rname} size {ei.size}!={si.size}"}
        for a in _INIT_ARRAYS:
            ea = _host(getattr(pool.bridge, a, None)); sa = _host(getattr(shipped.bridge, a, None))
            if ea is None or sa is None:
                continue
            d = float(np.max(np.abs(ea[ei].astype(np.float64) - sa[si].astype(np.float64)))) if ei.size else 0.0
            if d > worst:
                worst, worst_where = d, (rname, a)
    n_pool = int(pool.bridge.cp_membrane_potential_v.shape[0])
    n_ship = int(shipped.bridge.cp_membrane_potential_v.shape[0])
    init_ok = bool(worst == 0.0 and n_pool == n_ship)

    # ORGAN-READ round-trip: the real shipped production classes, run against EACH pool.
    m_eng, m_ship = MetacogProductionOrgan(seed=seed, shared=pool), MetacogProductionOrgan(seed=seed, shared=shipped)
    read_worst, read_worst_key = 0.0, None
    for e in _METACOG_READ_EVIDENCE:
        je, js = m_eng.judge(e), m_ship.judge(e)
        d = abs(float(je["balance"]) - float(js["balance"]))
        if d > read_worst:
            read_worst, read_worst_key = d, f"metacog.judge({e}).balance"
    d = abs(float(m_eng.threshold) - float(m_ship.threshold))
    if d > read_worst:
        read_worst, read_worst_key = d, "metacog.threshold"
    metacog_answer_same = all(bool(m_eng.judge(e)["confident"]) == bool(m_ship.judge(e)["confident"])
                              for e in _METACOG_READ_EVIDENCE)

    p_eng = PragmaticProductionOrgan(seed=seed, shared=pool)
    p_ship = PragmaticProductionOrgan(seed=seed, shared=shipped)
    for u in _PRAGMATIC_READ_UTTS:
        ie, is_ = p_eng.interpret(u), p_ship.interpret(u)
        for i in range(len(ie["states"])):
            d = abs(float(ie["belief"][i]) - float(is_["belief"][i]))
            if d > read_worst:
                read_worst, read_worst_key = d, f"pragmatic.interpret({u}).belief[{i}]"
    pragmatic_answer_same = all(p_eng.interpret(u)["enriched_interpretation"] ==
                                p_ship.interpret(u)["enriched_interpretation"] for u in _PRAGMATIC_READ_UTTS)

    read_ok = bool(read_worst == 0.0 and metacog_answer_same and pragmatic_answer_same)
    return {"seed": seed, "byte_identical": init_ok, "max_init_delta": worst, "worst_where": worst_where,
            "n_engine": n_pool, "n_shipped": n_ship, "organs": all_regions,
            "read_byte_identical": read_ok, "read_max_delta": read_worst, "read_worst_where": read_worst_key,
            "metacog_answer_preserved": metacog_answer_same, "pragmatic_answer_preserved": pragmatic_answer_same,
            "all_go": bool(init_ok and read_ok)}


_DET_HASH_ARRAYS = ("cp_membrane_potential_v", "cp_neuron_firing_thresholds", "cp_izh_a", "cp_izh_b",
                    "cp_izh_C", "cp_izh_d_increment")


def _build_hash(pool) -> list:
    """SHA1 of the per-neuron init arrays + the wired connection weights, in a stable order -- the
    build-twice-at-one-seed determinism witness (`cfg.seed` reproducibility gotcha, CLAUDE.md)."""
    import hashlib
    b = pool.bridge
    parts = [hashlib.sha1(np.ascontiguousarray(_host(getattr(b, a))).tobytes()).hexdigest()
             for a in _DET_HASH_ARRAYS]
    coo = b.cp_connections.tocoo()
    parts.append(hashlib.sha1(np.ascontiguousarray(_host(coo.data)).tobytes()).hexdigest())
    return parts


def _determinism2(seed: int = 42) -> dict:
    """Build the pool-#2 registry pool (metacog+pragmatic, wire=True) TWICE at ONE seed and hash its per-neuron
    init arrays + wired connection weights -- identical hashes == `cfg.seed` genuinely controls this pool (the
    2026-07-17 `actual_seed_used` gotcha this file's CLAUDE.md documents), not an artifact of import order."""
    p1 = merge_organs([METACOG, PRAGMATIC], seed=seed, wire=True)
    p2 = merge_organs([METACOG, PRAGMATIC], seed=seed, wire=True)
    h1, h2 = _build_hash(p1), _build_hash(p2)
    return {"seed": seed, "hash1": h1, "hash2": h2, "identical": bool(h1 == h2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="pool #1 (surprise+worldmodel) round-trip vs shipped")
    ap.add_argument("--smoke2", action="store_true", help="pool #2 (metacog+pragmatic) round-trip vs shipped")
    ap.add_argument("--determinism2", action="store_true",
                    help="pool #2 build-twice-at-one-seed hash determinism")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None, help="comma list; overrides --seed, loops + aggregates")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_onebrain_merge_framework_smoke_s42.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    if args.determinism2:
        results = [_determinism2(s) for s in seeds]
        print("=== onebrain_merge_framework DETERMINISM2 (pool #2 build-twice-at-one-seed hash) ===")
        for r in results:
            print(f"  seed={r['seed']} identical={r['identical']}")
        n_go = sum(bool(r.get("identical")) for r in results)
        print(f"  ALL-GO: {n_go}/{len(results)}")
        payload = {"mode": "onebrain_merge_framework_determinism2", "seeds": seeds, "per_seed": results,
                  "n_go": n_go, "n_seeds": len(results), "all_go": bool(n_go == len(results) and results)}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(payload, indent=2))
            print(f"  wrote {args.out}")
    elif args.smoke2:
        results = [_smoke2(s) for s in seeds]
        print("=== onebrain_merge_framework SMOKE2 (pool #2 descriptor->engine round-trip vs shipped MergedSubstrate2) ===")
        for r in results:
            print(f"  seed={r['seed']} engine_N={r.get('n_engine')} shipped_N={r.get('n_shipped')} "
                  f"init_delta={r.get('max_init_delta')} read_delta={r.get('read_max_delta')} "
                  f"worst={r.get('worst_where') or r.get('read_worst_where')}  -> "
                  f"{'PASS' if r.get('all_go') else 'FAIL'}")
        n_go = sum(bool(r.get("all_go")) for r in results)
        print(f"  ALL-GO: {n_go}/{len(results)}")
        payload = {"mode": "onebrain_merge_framework_smoke2", "seeds": seeds, "per_seed": results,
                  "n_go": n_go, "n_seeds": len(results), "all_go": bool(n_go == len(results) and results)}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(payload, indent=2))
            print(f"  wrote {args.out}")
    elif args.smoke:
        results = [_smoke(s) for s in seeds]
        print("=== onebrain_merge_framework SMOKE (descriptor->engine round-trip vs shipped MergedSubstrate) ===")
        for r in results:
            print(f"  seed={r['seed']} engine_N={r.get('n_engine')} shipped_N={r.get('n_shipped')} "
                  f"max_init_delta={r.get('max_init_delta')} worst={r.get('worst_where')}  -> "
                  f"{'PASS' if r['byte_identical'] else 'FAIL'}")
        n_go = sum(bool(r.get("byte_identical")) for r in results)
        print(f"  ALL-GO: {n_go}/{len(results)}")
        if len(seeds) == 1:
            payload = results[0]
        else:
            payload = {"mode": "onebrain_merge_framework_smoke", "seeds": seeds, "per_seed": results,
                      "n_go": n_go, "n_seeds": len(results), "all_go": bool(n_go == len(results) and results)}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(payload, indent=2))
            print(f"  wrote {args.out}")
    else:
        print("registered organs:", list(REGISTRY))
        print("run the round-trip smoke with:  SIM_BACKEND=numpy python -m "
              "research.runners.onebrain_merge_framework --smoke")
        print("run the pool-#2 round-trip smoke with:  SIM_BACKEND=numpy python -m "
              "research.runners.onebrain_merge_framework --smoke2")


if __name__ == "__main__":
    main()
