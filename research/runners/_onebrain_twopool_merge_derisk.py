"""ONE-BRAIN MERGE — production POOL #1 + POOL #2 onto ONE shared spiking bridge (the TWO production pools merge).

THE GAP THIS ATTACKS. Production today runs TWO SEPARATE merged pools:
  * pool #1 (`onebrain_merge_production.MergedSubstrate`, `BRAIN_ONEBRAIN_MERGE`): D2 SURPRISE + E2 WORLD-MODEL,
    `enable_parameter_heterogeneity=False`, homeostasis-based, GABA_B inert.
  * pool #2 (`onebrain_merge_production2.MergedSubstrate2`, `BRAIN_ONEBRAIN_MERGE2`): E1 METACOG + D-PRAGMATIC,
    `enable_parameter_heterogeneity=True`, plasticity/homeostasis/OU OFF (frozen), NMDA on (metacog).
`2026-08-13-onebrain-second-pool-SCOPED.md` scoped pool #2 as a SEPARATE pool from pool #1 precisely because their
GLOBAL configs conflict (param-het ON vs OFF), so "one substrate" for the four core cortical organs is still
CO-RESIDENCY of two pools. This runner de-risks the merge of the two pools into ONE `SimulationBridge`.

THE RECONCILIATION (existing primitives, NO sim/ edit). The pool #1/pool #2 config deltas each map to a per-region
seam already on `main`:
  1. param-het ON (pool2) vs OFF (pool1) -> GLOBAL `enable_parameter_heterogeneity=False` + per-region
     `BrainRegion.enable_heterogeneity=True` ON METACOG+PRAGMATIC REGIONS ONLY. `_overwrite_region_scoped_
     parameter_heterogeneity` (bridge.py:3477) then overwrites ONLY the masked (metacog/pragmatic) slices with a
     name-keyed draw; the unmasked surprise/world-model slices keep the non-jittered preset -> byte-identical to
     pool #1's param-het-OFF organs. (`per_region_parameter_heterogeneity=True`, name-keyed = co-residence-invariant.)
  2. homeostasis ON (pool1 organs) vs OFF (pool2 frozen) -> GLOBAL `enable_homeostasis=False` + per-region
     `BrainRegion.enable_homeostasis=True` on SURPRISE+WORLD-MODEL regions (the diffbuilder pattern the parser-on-pool
     merge used, `2026-08-14-onebrain-parser-on-pool-GO.md` conflict C). Metacog/pragmatic stay frozen.
  3. NMDA on (pool2 metacog) vs off (pool1) -> GLOBAL `enable_nmda=True` + per-region `BrainRegion.enable_nmda` so
     only metacog's workspace/meta_schema carry NMDA; every other region masks it off (byte-identical to no-NMDA).
  4. GABA_B (pool1, inert) -> `enable_gabab=True, gabab_conductance_max=0.0` (inert for all) = byte-identical.
  5. wiring seam -> `per_region_wiring_seed=True` (name-keyed pathway placement = co-residence-ORDER-invariant),
     `per_region_threshold_heterogeneity=True` (name-keyed firing thresholds). Both already used by both pools.

WHAT THIS RUNNER GATES (byte-identity, SUBSTRATE-INIT level; the net is SMALL + single-process). For each of the
four organs, EVERY per-neuron init array of that organ's region slice must be byte-IDENTICAL merged-vs-CO-RESIDENT
(the organ ALONE on the SAME superset config), so a slice's substrate is invariant to its three co-residents:
  * cp_neuron_firing_thresholds, cp_membrane_potential_v, cp_recovery_variable_u,
  * cp_izh_a / cp_izh_b / cp_izh_C / cp_izh_c_reset / cp_izh_d_increment / cp_izh_vpeak / cp_izh_vt / cp_izh_vr,
  * cp_heterogeneity_neuron_mask, cp_homeostasis_neuron_mask (the two per-region gate masks).
Plus (a) DETERMINISM — build the merged pool twice at one seed, all arrays identical; (b) the param-het MASK is
LOAD-BEARING + name-keyed — metacog/pragmatic slices carry NON-trivial param-het (differ from the default preset)
that is byte-identical merged-vs-coresident, while surprise/world-model slices sit at the preset; (c) a `--legacy`
DISCRIMINATOR — with the per-region seams OFF (global param-het + global wiring), the merged-vs-coresident slices
DIVERGE (the seam is what closes it, not a vacuous all-zero compare).

SCOPE (honest). This gates the STRUCTURAL substrate merge (pool + determinism + per-organ INIT byte-identity), the
same gate `2026-08-13-one-brain-merge-CLOSED-per-region-threshold.md` used for the 2-organ pool. It does NOT run the
organ READ pipelines (SurpriseProductionOrgan.judge etc.) or the post-build topographic wiring (block-diagonal /
assembly loops) -- the full organ-read byte-identity + answer-preservation-vs-current-production is the named
follow-on rung (bigger; belongs on the pool/gpu, reuses each organ's `shared=` read path).

Run (CPU, bit-exact):
    SIM_BACKEND=numpy python -m research.runners._onebrain_twopool_merge_derisk \
        --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_twopool_merge_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from research.runners._spiking_expectation_rpe_derisk import build_expectation_circuit
from research.runners._affective_world_model_derisk import build_world_model_circuit
from research.runners._second_order_metacog_monitor_derisk import (
    ASSEMBLY_SIZE, K_CLASSES, WORKSPACE_FS_N, META_SIZE,
    WS_TO_FS_WEIGHT, FS_TO_WS_WEIGHT, DEFAULT_NMDA_TAU,
)
from research.runners._recursive_tom_rsa_derisk import (
    RSA_ITEM_SIZE, RSA_FS_N, RSA_EXC_FS_W, RSA_FS_EXC_W,
)
from sim.regions import BrainRegion, RegionPathway

# ── pool #1 organ build kwargs (must match onebrain_merge_production._SURPRISE_KW / _WORLDMODEL_KW so the merged
#    slice is the SAME organ the production pool ships). ──
_SURPRISE_KW = dict(n_trained=8, n_novel=4, blk=24, cue_blk=24, cue_to_expected_weight=0.8)
_WORLDMODEL_KW = dict(n_states=6)

_N_WS = ASSEMBLY_SIZE * K_CLASSES

# region-name sets per organ (the per-region seam masks key on these names).
SURPRISE_REGIONS = ("cue", "patient_expected", "patient_asserted", "surprise")
WORLDMODEL_REGIONS = ("state", "pred_pos", "pred_neg", "obs_pos", "obs_neg", "surprise_pos", "surprise_neg")
METACOG_REGIONS = ("workspace", "workspace_fs", "meta_schema")
PRAGMATIC_REGIONS = ("item", "item_fs")
METACOG_NMDA_REGIONS = ("workspace", "meta_schema")   # build_metacog_bridge: these carry region enable_nmda=True

ALL_ORGANS = ("surprise", "worldmodel", "metacog", "pragmatic")
ORGAN_REGIONS = {
    "surprise": SURPRISE_REGIONS,
    "worldmodel": WORLDMODEL_REGIONS,
    "metacog": METACOG_REGIONS,
    "pragmatic": PRAGMATIC_REGIONS,
}

# per-neuron INIT arrays checked for byte-identity (every array the seams could perturb).
_INIT_ARRAYS = (
    "cp_neuron_firing_thresholds", "cp_membrane_potential_v", "cp_recovery_variable_u",
    "cp_izh_a", "cp_izh_b", "cp_izh_C", "cp_izh_c_reset", "cp_izh_d_increment",
    "cp_izh_vpeak", "cp_izh_vt", "cp_izh_vr",
    "cp_heterogeneity_neuron_mask", "cp_homeostasis_neuron_mask",
)


def _metacog_specs():
    regions = [
        BrainRegion(name="workspace", n_neurons=_N_WS, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="workspace_fs", n_neurons=WORKSPACE_FS_N, exc_fraction=0.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="meta_schema", n_neurons=META_SIZE, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
    ]
    pathways = [
        RegionPathway(from_region="workspace", to_region="workspace_fs", density=0.5,
                      weight_mean=WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="workspace_fs", to_region="workspace", density=0.5,
                      weight_mean=FS_TO_WS_WEIGHT, weight_jitter=0.0, plastic=False),
    ]
    return regions, pathways


def _pragmatic_specs():
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


def _organ_specs(organ, seed):
    """Return (regions, pathways) for one organ, on the shared superset config."""
    if organ == "surprise":
        _br, cfgS, _m = build_expectation_circuit(seed, per_region_thresh=True, **_SURPRISE_KW)
        return list(cfgS.brain_regions), list(cfgS.region_pathways)
    if organ == "worldmodel":
        _br, cfgW, _m = build_world_model_circuit(seed, **_WORLDMODEL_KW)
        return list(cfgW.brain_regions), list(cfgW.region_pathways)
    if organ == "metacog":
        return _metacog_specs()
    if organ == "pragmatic":
        return _pragmatic_specs()
    raise ValueError(organ)


def build_pool(seed, organs, legacy=False, force_het_off=False):
    """Build ONE SimulationBridge holding `organs`' regions on the pool #1+#2 SUPERSET config.

    legacy=False -> the reconciled merge (per-region seams ON, per-region het/homeostasis/nmda masks).
    legacy=True  -> the DISCRIMINATOR: global param-het + global wiring, seams OFF (should DIVERGE co-resident).
    force_het_off=True -> the LOAD-BEARING control: same reconciled config but the per-region het mask is cleared,
                          so a masked organ's substrate reverts to the non-jittered preset (isolates the mask's effect).
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.enums import NeuronModel

    regions, pathways = [], []
    has = {o: (o in organs) for o in ALL_ORGANS}
    for o in ALL_ORGANS:
        if has[o]:
            r, p = _organ_specs(o, seed)
            regions += r
            pathways += p

    # per-region masks (applied by NAME so a slice is co-residence-invariant).
    het_names = set(METACOG_REGIONS) | set(PRAGMATIC_REGIONS)   # param-het ON only for the pool-2 organs
    homeo_names = set(SURPRISE_REGIONS) | set(WORLDMODEL_REGIONS)  # homeostasis ON only for the pool-1 organs
    nmda_names = set(METACOG_NMDA_REGIONS)
    if not legacy:
        for r in regions:
            nm = getattr(r, "name", None)
            r.enable_heterogeneity = (nm in het_names) and (not force_het_off)
            r.enable_homeostasis = (nm in homeo_names)
            r.enable_nmda = (nm in nmda_names)

    cfg = CoreSimConfig()
    cfg.seed = int(seed)
    cfg.heterogeneity_seed = int(seed)
    cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True

    # merge seams
    cfg.per_region_threshold_heterogeneity = not legacy
    cfg.per_region_parameter_heterogeneity = not legacy
    cfg.per_region_wiring_seed = not legacy
    cfg.per_region_homeostasis_isolation = not legacy

    # plasticity block (pool #1's surprise trains Hebbian at 45; pool #2 edges are plastic=False / frozen loops).
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = 0.06
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_max_weight = 45.0
    cfg.hebbian_weight_decay = 0.0
    cfg.hebbian_rate_window = True
    cfg.hebbian_coactivity_decay = 0.85
    cfg.hebbian_coactivity_thresh = 0.20
    cfg.hebbian_mean_subtract = 1.0
    cfg.enable_reward_modulation = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.current_reward_signal = 0.0
    cfg.reward_baseline = 0.0

    # param-het: GLOBAL off (per-region mask supplies pool #2's het); legacy -> GLOBAL on (the pre-seam behaviour).
    cfg.enable_parameter_heterogeneity = bool(legacy)

    # homeostasis: GLOBAL off, opted-in per-region on the pool-1 organs (legacy -> global on = the pre-seam behaviour).
    cfg.enable_homeostasis = bool(legacy)

    # NMDA superset (metacog needs it); per-region enable_nmda masks it off everywhere else.
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.nmda_tau_decay = float(DEFAULT_NMDA_TAU)
    cfg.nmda_recurrent_tau_decay_ms = float(DEFAULT_NMDA_TAU)

    # GABA_B superset (pool #1 sets it, inert via conductance_max=0).
    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = 150.0
    cfg.gabab_propagation_strength = 0.22
    cfg.gabab_conductance_max = 0.0

    cfg.brain_regions = regions
    cfg.region_pathways = pathways

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def _host(a):
    if a is None:
        return None
    try:
        import cupy as cp  # noqa
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def _region_indices(bridge, name):
    return np.asarray(sorted(int(i) for i in bridge.region_manager.indices(name)), dtype=np.int64)


# the two per-region GATE masks are GLOBAL bridge arrays that are None when NO region opts in. Semantically a None
# mask means "False for every neuron", so coerce None -> a full all-False array before slicing; otherwise a merged
# pool (mask present, this organ's neurons False) vs a solo pool (mask None) would spuriously read as a mismatch even
# though the organ's neurons carry the SAME (unmasked) state in both.
_MASK_ARRAYS = ("cp_heterogeneity_neuron_mask", "cp_homeostasis_neuron_mask")


def _slice_arrays(bridge, idx):
    """Extract each INIT array restricted to neuron indices `idx` (host numpy)."""
    n = int(_host(bridge.cp_membrane_potential_v).shape[0])
    out = {}
    for a in _INIT_ARRAYS:
        arr = _host(getattr(bridge, a, None))
        if arr is None and a in _MASK_ARRAYS:
            arr = np.zeros(n, dtype=np.float64)   # None mask == all-False
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
    return float(np.max(np.abs(x - y))) if x.size else 0.0, "ok"


def byte_identity(seed, legacy=False):
    """merged (all 4) vs co-resident (each organ alone), per-organ per-array max delta over the organ's regions."""
    merged, _ = build_pool(seed, ALL_ORGANS, legacy=legacy)
    per_organ = {}
    for organ in ALL_ORGANS:
        solo, _ = build_pool(seed, (organ,), legacy=legacy)
        organ_max = 0.0
        detail = {}
        for rname in ORGAN_REGIONS[organ]:
            mi = _region_indices(merged, rname)
            si = _region_indices(solo, rname)
            if mi.size != si.size:
                detail[rname] = {"maxerr": float("inf"), "note": f"size {mi.size}!={si.size}"}
                organ_max = float("inf")
                continue
            ma = _slice_arrays(merged, mi)
            sa = _slice_arrays(solo, si)
            rmax = 0.0
            worst = None
            for a in _INIT_ARRAYS:
                e, note = _maxerr(ma[a], sa[a])
                if e > rmax:
                    rmax = e; worst = (a, note)
            detail[rname] = {"maxerr": rmax, "worst": worst}
            organ_max = max(organ_max, rmax)
        per_organ[organ] = {"maxerr": organ_max, "regions": detail}
    return per_organ


def determinism(seed):
    """build merged twice at one seed -> every full array identical."""
    b1, _ = build_pool(seed, ALL_ORGANS)
    b2, _ = build_pool(seed, ALL_ORGANS)
    mx = 0.0
    worst = None
    for a in _INIT_ARRAYS:
        x = _host(getattr(b1, a, None)); y = _host(getattr(b2, a, None))
        e, _n = _maxerr(x, y)
        if e > mx:
            mx = e; worst = a
    return mx, worst


def het_loadbearing(seed):
    """The param-het MASK is DOING WORK, not a no-op. Compare each organ's Izhikevich params in the merged pool WITH
    the per-region het mask vs the SAME merged pool with the mask FORCED OFF:
      * metacog / pragmatic (masked ON) -> the params must DIFFER (delta > 0): the mask genuinely jitters them.
      * surprise / world-model (unmasked) -> the params must be IDENTICAL (delta == 0): the mask leaves them alone.
    (Raw std over a slice mixes RS vs FS neuron TYPES, so it is not a valid witness -- this delta isolates the mask.)"""
    merged, _ = build_pool(seed, ALL_ORGANS)
    het_off, _ = build_pool(seed, ALL_ORGANS, force_het_off=True)
    param_arrays = ("cp_izh_a", "cp_izh_b", "cp_izh_C", "cp_izh_d_increment")
    out = {}
    for organ in ALL_ORGANS:
        idx = np.concatenate([_region_indices(merged, r) for r in ORGAN_REGIONS[organ]])
        idx_off = np.concatenate([_region_indices(het_off, r) for r in ORGAN_REGIONS[organ]])
        delta = 0.0
        for a in param_arrays:
            x = _host(getattr(merged, a))[idx].astype(np.float64)
            y = _host(getattr(het_off, a))[idx_off].astype(np.float64)
            delta = max(delta, float(np.max(np.abs(x - y))) if x.size else 0.0)
        het_on = bool(organ in ("metacog", "pragmatic"))
        out[organ] = {"mask_effect_delta": delta, "expected_het": het_on}
    return out


def run(seeds, out_path):
    results = {"seeds": [], "per_seed": {}}
    for seed in seeds:
        bi = byte_identity(seed, legacy=False)
        det, det_worst = determinism(seed)
        het = het_loadbearing(seed)
        leg = byte_identity(seed, legacy=True)   # discriminator

        pool_merged, _ = build_pool(seed, ALL_ORGANS)
        n_total = int(_host(pool_merged.cp_membrane_potential_v).shape[0])
        one_pool = all(
            _region_indices(pool_merged, r).size > 0
            for organ in ALL_ORGANS for r in ORGAN_REGIONS[organ]
        )

        byte_ok = all(bi[o]["maxerr"] == 0.0 for o in ALL_ORGANS)
        det_ok = (det == 0.0)
        # het load-bearing: pool-2 organs have spread>0, pool-1 organs spread==0 (preset).
        het_ok = all(
            (het[o]["mask_effect_delta"] > 0.0) == het[o]["expected_het"] for o in ALL_ORGANS
        )
        # discriminator: legacy (seams off) must DIVERGE for >=1 organ (else the compare is vacuous).
        legacy_diverges = any(leg[o]["maxerr"] > 0.0 for o in ALL_ORGANS)
        go = bool(byte_ok and det_ok and het_ok and legacy_diverges and one_pool)

        results["seeds"].append(seed)
        results["per_seed"][str(seed)] = {
            "n_total": n_total,
            "one_pool": one_pool,
            "byte_identity": bi,
            "byte_ok": byte_ok,
            "determinism_maxerr": det, "determinism_worst": det_worst, "det_ok": det_ok,
            "het_loadbearing": het, "het_ok": het_ok,
            "legacy_discriminator": {o: leg[o]["maxerr"] for o in ALL_ORGANS},
            "legacy_diverges": legacy_diverges,
            "GO": go,
        }
        print(f"seed {seed}: N={n_total} one_pool={one_pool} "
              f"byte_ok={byte_ok} det={det:.3g} het_ok={het_ok} "
              f"legacy_diverges={legacy_diverges} -> {'GO' if go else 'NO-GO'}")
        for o in ALL_ORGANS:
            print(f"    {o:11s} byte_maxerr={bi[o]['maxerr']:.3g}  "
                  f"legacy_maxerr={leg[o]['maxerr']:.3g}  "
                  f"mask_effect={het[o]['mask_effect_delta']:.4g} (het_expected={het[o]['expected_het']})")

    n_go = sum(1 for s in results["seeds"] if results["per_seed"][str(s)]["GO"])
    results["summary"] = {"n_seeds": len(results["seeds"]), "n_go": n_go,
                          "verdict": "GO" if n_go == len(results["seeds"]) and results["seeds"] else "NO-GO"}
    print(f"\n=== {n_go}/{len(results['seeds'])} seeds GO -> {results['summary']['verdict']} ===")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(results, indent=2))
        print(f"wrote {out_path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]
    run(seeds, args.out)


if __name__ == "__main__":
    main()
