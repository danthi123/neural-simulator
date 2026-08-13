"""PER-REGION PARAMETER-HETEROGENEITY — the cluster-merge verify (Gate-B one-brain, 2026-08-13).

Rung 2 (`2026-08-13-merge-production-rung2-BOUNDARY.md`) mapped that the metacog / pragmatic / affect production
organs cannot join the shared spiking pool byte-identically because their graded rate code REQUIRES
`enable_parameter_heterogeneity`, whose per-neuron jitter is drawn as ONE `size=n` sample per parameter from the
GLOBAL RNG over the whole pool -> a co-resident organ's slice is position-shifted (a valid but DIFFERENT seeded
param-het than standalone). The named fix (this arc's `sim/` edit) is `cfg.per_region_parameter_heterogeneity`,
a name-keyed per-region param substream EXACTLY mirroring the landed `per_region_threshold_heterogeneity`.

WHAT THIS VERIFIES (SIM_BACKEND=numpy, so cp == numpy -> bit-exact):

 (1) ENGINE FEATURE on the REAL organ substrates. For metacog and pragmatic (RSA) -- the two cluster organs whose
     ONLY divergent global flag is param-het (homeostasis / OU / neuromod all OFF, no cross-synapse) -- build the
     organ's bridge STANDALONE vs with the OTHER cluster organs' regions PREPENDED as INERT (density-0, unwired,
     no-pathway) co-residents on ONE co-stepped pool. Those co-residents shift the organ to a NON-ZERO offset (the
     exact perturbation a shared pool introduces) while consuming NO build_wiring_plan RNG (so the organ's own
     connectivity stays byte-identical) and adding NO cross-synapse. We compare the organ's PRODUCTION READ:
        * with per_region_parameter_heterogeneity ON  -> max delta 0.0 (byte-identical) EXPECTED  (GO)
        * with it OFF                                  -> the read DIVERGES (confirms the exact bug the flag fixes)
     and confirm the FACULTY IS ALIVE on the co-resident pool (metacog: high-evidence margin > low-evidence margin;
     pragmatic: the some->not-all implicature is represented).

 (2) AFFECT BOUNDARY (mapped, measured). The affect mood-ladder organ has a SECOND + THIRD global-per-step
     position-dependence beyond param-het: its read runs with enable_ou_process=True (OU noise is a `size=n`
     per-step global draw -> position-shifted) AND drives the global neuromodulator subsystem. So affect does NOT
     become byte-identical with per-region param-het alone -- it is an honest partial (BOUNDARY), needing per-region
     OU + per-region neuromod, distinct engine features. We MEASURE its co-resident divergence to make the boundary
     concrete rather than asserted.

HONEST SCOPE. The co-resident organs' regions are present + co-stepped but NOT simultaneously independently
wired-and-read (that adds cross-synapse-free pathway-sampling ORDER dependence in build_wiring_plan's shared RNG --
a THIRD position source solvable by per-organ-plan-remap, the mapped next rung). This rung proves the param-het
engine feature makes each organ's read INVARIANT to co-residence on one pool -- the load-bearing dependency rung 2
named -- for the two organs whose only blocker was param-het.

NO new `sim/` behavior beyond the guarded flag; additive default-preserving kwargs on two de-risk builders
(`build_metacog_bridge`, `build_rsa_bridge`: `coresident_regions`/`per_region_param_het`/`per_region_thresh`,
defaults None/False -> byte-identical). Process backend (cupy in production, numpy in tests).

Reproduce:
    SIM_BACKEND=numpy python -m research.runners._per_region_param_het_cluster_verify \
        --seeds 42,43,44,100,101,102 --out research/findings/raw/_per_region_param_het_cluster_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sim.regions import BrainRegion


# ── inert (density-0, unwired, exc-only so no inhibitory tagging) co-resident region specs per organ ──────────
def _rsa_coresident():
    from research.runners._recursive_tom_rsa_derisk import RSA_ITEM_SIZE, RSA_FS_N
    return [BrainRegion(name="cx_item", n_neurons=RSA_ITEM_SIZE * 3, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="cx_item_fs", n_neurons=RSA_FS_N, exc_fraction=1.0, internal_density=0.0)]


def _metacog_coresident():
    from research.runners._second_order_metacog_monitor_derisk import (
        ASSEMBLY_SIZE, K_CLASSES, WORKSPACE_FS_N, META_SIZE)
    return [BrainRegion(name="cx_workspace", n_neurons=ASSEMBLY_SIZE * K_CLASSES, exc_fraction=1.0,
                        internal_density=0.0),
            BrainRegion(name="cx_ws_fs", n_neurons=WORKSPACE_FS_N, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="cx_meta", n_neurons=META_SIZE, exc_fraction=1.0, internal_density=0.0)]


def _affect_coresident(n_neurons: int):
    return [BrainRegion(name="cx_affect", n_neurons=int(max(1, n_neurons)), exc_fraction=1.0,
                        internal_density=0.0)]


def _max_delta(a: dict, b: dict):
    keys = sorted(set(a) & set(b))
    worst_k, worst = None, 0.0
    for k in keys:
        d = abs(float(a[k]) - float(b[k]))
        if d >= worst:
            worst, worst_k = d, k
    return worst, worst_k, sorted(set(a) ^ set(b))


# ── metacog production read (balance-of-evidence margin), replicating MetacogProductionOrgan._margin ──────────
def _metacog_reads(seed, coresident, per_region):
    from research.runners._second_order_metacog_monitor_derisk import build_metacog_bridge, _run_trial
    from research.runners.metacog_production_organ import (
        BASE_PA, SIG_LO, SIG_HI, READ_REPS, READ_JITTER_PA, READ_SEED)
    bridge, xp, idx, snap = build_metacog_bridge(
        seed=seed, confidence_read="balance",
        coresident_regions=coresident, per_region_param_het=per_region, per_region_thresh=per_region)

    def margin(evidence):
        sig = SIG_LO + float(np.clip(evidence, 0.0, 1.0)) * (SIG_HI - SIG_LO)
        rng = np.random.default_rng(READ_SEED)
        vals = []
        for _ in range(READ_REPS):
            j = float(rng.normal(0.0, READ_JITTER_PA)) if READ_JITTER_PA > 0 else 0.0
            dp = [max(0.0, BASE_PA + sig + j), max(0.0, BASE_PA + j)]
            vals.append(float(_run_trial(bridge, xp, idx, snap, dp)["meta"]))
        return float(np.mean(vals))

    evid = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
    out = {f"margin[{e:.2f}]": margin(e) for e in evid}
    hi = float(np.mean([out[f"margin[{e:.2f}]"] for e in (0.75, 0.9, 1.0)]))
    lo = float(np.mean([out[f"margin[{e:.2f}]"] for e in (0.0, 0.15, 0.3)]))
    out["_alive"] = 1.0 if hi > lo else 0.0                       # graded confidence separates high vs low evidence
    return out


# ── pragmatic production read (graded RSA L1 belief), replicating graded_belief_sources ──────────────────────
def _pragmatic_reads(seed, coresident, per_region):
    from research.runners._recursive_tom_rsa_derisk import (
        build_rsa_bridge, _rsa_recursion, TRUTH, STATES, UTTS)
    b, xp, item_dev, snap = build_rsa_bridge(
        seed, normalize=True, coresident_regions=coresident,
        per_region_param_het=per_region, per_region_thresh=per_region)
    _L0, S1, _L1 = _rsa_recursion(b, xp, item_dev, snap, TRUTH, 25)
    belief = {}
    for j, u in enumerate(UTTS):
        v = np.asarray(S1[j], dtype=np.float64).copy()
        if v.sum() <= 1e-9:
            v = np.array([TRUTH[u][s] for s in STATES], dtype=np.float64)
        belief[u] = v / v.sum()
    out = {}
    for u in UTTS:
        for t, s in enumerate(STATES):
            out[f"belief[{u}][{s}]"] = float(belief[u][t])
    some = belief["some"]
    margin = float(some[STATES.index("SBNA")] - some[STATES.index("all")])
    out["_alive"] = 1.0 if margin > 1e-3 else 0.0                 # the some->not-all implicature is represented
    return out


def run_seed(seed: int, verbose=True) -> dict:
    # ---- metacog: co-resident with pragmatic(RSA)+affect-sized regions ----
    mc_cores = _rsa_coresident() + _affect_coresident(240)
    mc_solo_on = _metacog_reads(seed, None, True)
    mc_merge_on = _metacog_reads(seed, mc_cores, True)
    mc_solo_off = _metacog_reads(seed, None, False)
    mc_merge_off = _metacog_reads(seed, mc_cores, False)
    mc_d_on, mc_k_on, _ = _max_delta({k: v for k, v in mc_solo_on.items() if k != "_alive"},
                                     {k: v for k, v in mc_merge_on.items() if k != "_alive"})
    mc_d_off, _, _ = _max_delta({k: v for k, v in mc_solo_off.items() if k != "_alive"},
                                {k: v for k, v in mc_merge_off.items() if k != "_alive"})
    mc_byte_id = bool(mc_d_on == 0.0)
    mc_alive = bool(mc_merge_on["_alive"] == 1.0)
    mc_go = bool(mc_byte_id and mc_alive)

    # ---- pragmatic: co-resident with metacog+affect-sized regions ----
    pr_cores = _metacog_coresident() + _affect_coresident(240)
    pr_solo_on = _pragmatic_reads(seed, None, True)
    pr_merge_on = _pragmatic_reads(seed, pr_cores, True)
    pr_solo_off = _pragmatic_reads(seed, None, False)
    pr_merge_off = _pragmatic_reads(seed, pr_cores, False)
    pr_d_on, pr_k_on, _ = _max_delta({k: v for k, v in pr_solo_on.items() if k != "_alive"},
                                     {k: v for k, v in pr_merge_on.items() if k != "_alive"})
    pr_d_off, _, _ = _max_delta({k: v for k, v in pr_solo_off.items() if k != "_alive"},
                                {k: v for k, v in pr_merge_off.items() if k != "_alive"})
    pr_byte_id = bool(pr_d_on == 0.0)
    pr_alive = bool(pr_merge_on["_alive"] == 1.0)
    pr_go = bool(pr_byte_id and pr_alive)

    res = {
        "seed": seed,
        "metacog_maxdelta_on": mc_d_on, "metacog_worst_on": mc_k_on, "metacog_maxdelta_off": mc_d_off,
        "metacog_byte_identical": mc_byte_id, "metacog_alive": mc_alive, "metacog_go": mc_go,
        "pragmatic_maxdelta_on": pr_d_on, "pragmatic_worst_on": pr_k_on, "pragmatic_maxdelta_off": pr_d_off,
        "pragmatic_byte_identical": pr_byte_id, "pragmatic_alive": pr_alive, "pragmatic_go": pr_go,
    }
    if verbose:
        print(f"  [seed {seed}] METACOG on-delta={mc_d_on:.2e}(off={mc_d_off:.2e}) alive={mc_alive} -> GO={mc_go} | "
              f"PRAGMATIC on-delta={pr_d_on:.2e}(off={pr_d_off:.2e}) alive={pr_alive} -> GO={pr_go}", flush=True)
    return res


def _ou_region_trajectory(seed, regions_spec, ou_on, steps=40):
    """Build a co-stepped multi-region bridge (per-region param-het ON) and return region 'R''s cp_ou_current
    slice after `steps`. With OU ON, the per-step noise is a size-n GLOBAL draw (bridge.py cp.random.randn(n)),
    so R's slice depends on the pool size + R's offset; with OU OFF it stays at the constant OU mean."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.enums import NeuronModel
    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True
    cfg.per_region_parameter_heterogeneity = True          # the init seam IS closed by this flag
    cfg.enable_ou_process = bool(ou_on)                    # the OU seam is NOT
    cfg.brain_regions = [BrainRegion(name=nm, n_neurons=nn, exc_fraction=1.0, internal_density=0.0)
                         for (nm, nn) in regions_spec]
    cfg.region_pathways = []
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.actual_seed_used = int(seed)
    b._initialize_simulation_data(called_from_playback_init=False)
    for _ in range(steps):
        b._run_one_simulation_step()
    idx = np.asarray(sorted(int(i) for i in b.region_manager.indices("R")), dtype=np.int64)
    cur = getattr(b, "cp_ou_current", None)
    if cur is None:                                       # OU off -> read membrane instead (also co-step-driven)
        cur = b.cp_membrane_potential_v
    return np.asarray(cur, dtype=np.float64)[idx].copy()


def affect_boundary(seed: int, verbose=True) -> dict:
    """MEASURE the OU-noise position-dependence that underlies affect's boundary. Affect's read runs with
    enable_ou_process=True + drives the global neuromodulator subsystem. Even with per_region_parameter_heterogeneity
    ON, region 'R''s OU-driven trajectory STILL differs alone vs co-resident (behind a spacer) -- because OU noise
    is a size-n per-step GLOBAL draw, an OPEN seam param-het does not close. The OU-OFF control at the same offset
    is byte-identical (delta 0.0), isolating OU as the cause. So affect cannot join byte-identically on param-het
    alone -> BOUNDARY (needs per-region OU + per-region neuromod, distinct engine features)."""
    # OU ON: R alone (offset 0) vs R behind a 30-neuron spacer (offset 30).
    r_solo_ou = _ou_region_trajectory(seed, [("R", 20)], ou_on=True)
    r_cores_ou = _ou_region_trajectory(seed, [("X", 30), ("R", 20)], ou_on=True)
    ou_delta = float(np.max(np.abs(r_solo_ou - r_cores_ou)))
    # OU OFF control: same comparison, param-het ON -> byte-identical (delta 0.0).
    r_solo_no = _ou_region_trajectory(seed, [("R", 20)], ou_on=False)
    r_cores_no = _ou_region_trajectory(seed, [("X", 30), ("R", 20)], ou_on=False)
    noou_delta = float(np.max(np.abs(r_solo_no - r_cores_no)))
    # affect organ structurally uses the global neuromodulator subsystem (a second open seam)
    has_nm = None
    try:
        from research.runners import affect_production_organ as AP
        a = AP.AffectProductionOrgan(seed=seed); a.ensure_built()
        has_nm = a.bridge.neuromodulator_manager is not None
    except Exception:
        has_nm = None
    boundary = bool(ou_delta > 0.0 and noou_delta == 0.0)
    if verbose:
        print(f"  [seed {seed}] AFFECT OU-seam: ou_on_delta={ou_delta:.3e} (>0 -> position-dependent)  "
              f"ou_off_control_delta={noou_delta:.3e} (==0 -> param-het closes init)  neuromod={has_nm} "
              f"-> BOUNDARY={boundary}", flush=True)
    return {"seed": seed, "affect_ou_on_delta": ou_delta, "affect_ou_off_control_delta": noou_delta,
            "affect_has_neuromod": has_nm, "affect_boundary_confirmed": boundary,
            "affect_verdict": "BOUNDARY",
            "affect_reason": "OU size-n per-step global draw (open seam) + global neuromodulator subsystem"}


def _gate(n_go, n):
    return "GO" if ((n >= 6 and n_go >= 5) or (n < 6 and n_go == n)) else "BOUNDARY"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--skip-affect", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=== PER-REGION PARAM-HET — CLUSTER MERGE VERIFY ===")
    print("(1) metacog + pragmatic byte-identical on a co-resident pool (param-het ON) + faculty alive")
    results = [run_seed(s) for s in seeds]
    n = len(results)
    n_mc = sum(r["metacog_go"] for r in results)
    n_pr = sum(r["pragmatic_go"] for r in results)
    n_mc_bi = sum(r["metacog_byte_identical"] for r in results)
    n_pr_bi = sum(r["pragmatic_byte_identical"] for r in results)

    affect = []
    if not args.skip_affect:
        print("\n--- (2) affect BOUNDARY (OU + neuromod global-per-step) ---")
        affect = [affect_boundary(s) for s in seeds]

    print("\n=== VERDICT (cluster) ===")
    print(f"  metacog   byte-identical: {n_mc_bi}/{n}   FACULTY GO (byte-id + alive): {n_mc}/{n} -> {_gate(n_mc, n)}")
    print(f"  pragmatic byte-identical: {n_pr_bi}/{n}   FACULTY GO (byte-id + alive): {n_pr}/{n} -> {_gate(n_pr, n)}")
    if affect:
        print(f"  affect: BOUNDARY (mapped) — OU size-n per-step + global neuromodulator subsystem, "
              f"not fixed by per-region param-het alone")

    # off-path divergence (the flag is load-bearing): the WORST off-delta across seeds/organs must exceed 0.
    worst_off = 0.0
    for r in results:
        worst_off = max(worst_off, float(r["metacog_maxdelta_off"]), float(r["pragmatic_maxdelta_off"]))
    from tools.verdict import Verdict
    v = Verdict("per_region_param_het cluster merge (metacog + pragmatic)")
    v.require("metacog_read_byte_identical_on", n_mc_bi, expect=n,
              note="metacog production read max delta 0.0 merged-vs-co-resident, flag ON, all seeds")
    v.require("pragmatic_read_byte_identical_on", n_pr_bi, expect=n,
              note="pragmatic production read max delta 0.0 merged-vs-co-resident, flag ON, all seeds")
    v.control("flag_is_load_bearing", treatment=worst_off, control=0.0, min_separation=0.0,
              note="with the flag OFF the reads DIVERGE (position-shifted param-het) -> not a no-op")
    decided = v.decide(go=(n_mc_bi == n and n_pr_bi == n and worst_off > 0.0), verbose=False)
    payload = {"mode": "per_region_param_het_cluster", "n_seeds": n, "results": results,
               "n_metacog_byte_identical": n_mc_bi, "n_pragmatic_byte_identical": n_pr_bi,
               "n_metacog_go": n_mc, "n_pragmatic_go": n_pr,
               "metacog_verdict": _gate(n_mc, n), "pragmatic_verdict": _gate(n_pr, n),
               "verdict": decided["status"], "preconditions": decided["preconditions"],
               "undefined_reasons": decided["undefined_reasons"],
               "affect_boundary": affect,
               "note": ("per-region param-het makes metacog + pragmatic reads invariant to co-residence on one "
                        "co-stepped pool (byte-identical); affect needs per-region OU + neuromod (mapped boundary)")}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
