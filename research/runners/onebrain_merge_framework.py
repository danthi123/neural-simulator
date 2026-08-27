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
    organ_cls: type = None                     # the shipped *_ProductionOrgan (constructed with shared=<pool>)


class MergeConflict(ValueError):
    """Two descriptors REQUIRE a config key at different values — a genuine global-config incompatibility
    (param-het ON vs OFF; OU on vs off). Raised at BUILD so it fails loudly at registration, never silently
    corrupts a slice. The reconciliation is a per-region seam (see the twopool branch), declared per organ."""


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  2. THE MERGE ENGINE — merge_organs([descriptors], seed) -> MergedPool.
# ─────────────────────────────────────────────────────────────────────────────────────────────
# The engine-UNIVERSAL config (dt / model / profile / seeds / the always-on per-region seam + framework).
# Everything organ-family-specific (hebbian block, gabab, the disable flags) is declared PER DESCRIPTOR and
# UNIONED, so a new family is a descriptor's `config`, not an engine edit.
def _base_config(seed: int):
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
    cfg.per_region_threshold_heterogeneity = True     # merge seam #1 (name-keyed init byte-identity)
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

    def __init__(self, seed, descriptors):
        self.seed = int(seed)
        self.descriptors = list(descriptors)
        self._by_key = {d.key: d for d in descriptors}
        self.bridge = self.cfg = self.xp = None
        self.meta = {}
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
        for d in self.descriptors:
            r, p, m = d.spec_fn(self.seed)
            regions += list(r); pathways += list(p); self.meta[d.key] = m

        # (2) NAME-DISJOINTNESS — a rename is forbidden (the seams key on the name); a collision would change a
        #     slice's init. This is the exact reason affect is scoped to its own pool (DESIGN §5 GROUP C).
        names = [rg.name for rg in regions]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise MergeConflict(f"region-name collision across organs: {sorted(dup)} (rename forbidden)")

        # (3) CONFIG UNION — base + each descriptor's requirements; a key at two values is a real conflict.
        cfg = _base_config(self.seed)
        union, provenance = {}, {}
        for d in self.descriptors:
            for k, v in d.config.items():
                if k in union and union[k] != v:
                    raise MergeConflict(f"{k!r}: {provenance[k]}={union[k]!r} vs {d.key}={v!r}")
                union[k] = v; provenance[k] = d.key
        for k, v in union.items():
            setattr(cfg, k, v)

        # (4) PER-REGION FLAGS — the diffbuilder pattern: reconcile a would-be global conflict into a masked one.
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

        self.bridge, self.cfg, self.xp, self._built = bridge, cfg, xp, True

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


def merge_organs(descriptors, seed: int = 42) -> MergedPool:
    """Build ONE shared spiking bridge holding all `descriptors`' regions. The N-organ generalization of the
    bespoke per-pool MergedSubstrate (DESIGN §2). Returns a MergedPool ready for `desc.organ_cls(shared=pool)`."""
    pool = MergedPool(seed, descriptors)
    pool.ensure_built()
    return pool


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

REGISTRY = {d.key: d for d in (SURPRISE, WORLDMODEL)}


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
