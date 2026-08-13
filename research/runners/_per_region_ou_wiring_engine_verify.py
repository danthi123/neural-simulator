"""PER-REGION OU-NOISE + WIRING SEED — the ENGINE-feature verify (the two `sim/` guards' substrate proof).

The one-substrate merge's rung-2b finding (`2026-08-13-per-region-param-het-cluster-GO.md`) named the next two
byte-identity seams blocking further organs from joining the shared spiking pool:

  (A) enable_ou_process — the OU background drive draws its per-step white noise as ONE `size=n` global
      cp.random.randn(n) sample, so a region's noise slice is indexed by absolute pool position -> a merged organ's
      OU realization diverges (measured: OU-on co-resident delta ~1.5e2 vs OU-off 0.0).
  (B) build_wiring_plan — samples every region's internal connectivity THEN every pathway from ONE shared
      random.Random(seed) in region-then-pathway ORDER, so a region's / pathway's synapse placement depends on how
      much RNG the entries BEFORE it consumed (co-residence ORDER dependence).

This runner exercises the two guards DIRECTLY on the substrate:

  cfg.per_region_ou_seed:
   (1) POSITION-INVARIANCE (ON). Region R's cp_ou_current trajectory (OU on) is BYTE-IDENTICAL whether R is built
       ALONE (offset 0) or co-resident BEHIND a spacer X (offset 30). 6/6 True EXPECTED.
   (2) POSITION-DEPENDENCE (OFF) — the bug the flag fixes. SAME comparison, flag OFF -> DIFFERS. 6/6 False EXPECTED.
   (3) DETERMINISM (ON). Build+step the co-resident pool twice at one seed -> byte-identical. 6/6 True EXPECTED.

  cfg.per_region_wiring_seed (via RegionManager.build_wiring_plan(per_region_seed=...)):
   (1) REGION-INTERNAL ORDER-INVARIANCE (ON). Region R's internal-connectivity LOCAL pattern (pre/post relative to
       R's base + weights) is BYTE-IDENTICAL whether R is built ALONE or AFTER a spacer X (which consumes shared
       RNG first). 6/6 True EXPECTED.
   (2) PATHWAY ORDER-INVARIANCE (ON). Pathway A->B's placement is BYTE-IDENTICAL whether or not an extra pathway
       B->A is drawn BEFORE it. 6/6 True EXPECTED.
   (3) ORDER-DEPENDENCE (OFF) — the bug. SAME comparisons, flag OFF -> DIFFER. 6/6 False EXPECTED.
   (4) DETERMINISM (ON). Build the plan twice -> byte-identical. 6/6 True EXPECTED.

 (OFF-PATH HASH, --mode off). Prints a hash of (a) a co-resident OU bridge's full substrate + cp_ou_current and
 (b) a wiring plan, both with the flags OFF, for a git-stash byte-identity comparison to HEAD (default-off must be
 bit-for-bit today):
     SIM_BACKEND=numpy python -m research.runners._per_region_ou_wiring_engine_verify --mode off
     git stash push -- sim/bridge.py sim/config.py sim/regions.py
     SIM_BACKEND=numpy python -m research.runners._per_region_ou_wiring_engine_verify --mode off
     git stash pop
     # the two OFF_HASHES lines must be identical.

Process backend (numpy for the bit-exact checks; cp == numpy so region-scoped host draws are backend-neutral).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────────────── OU seam ────────────
def _ou_bridge(seed, regions_spec, per_region_ou, ou_on=True):
    """Build a co-stepped multi-region bridge with enable_ou_process=`ou_on` and cfg.per_region_ou_seed=
    `per_region_ou`; regions are inert (density-0, unwired) so R's cp_ou_current depends only on its noise stream."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.enums import NeuronModel
    from sim.regions import BrainRegion
    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_parameter_heterogeneity",
              "enable_conductance_noise"):
        setattr(cfg, f, False)
    cfg.enable_ou_process = bool(ou_on)
    if per_region_ou:
        setattr(cfg, "per_region_ou_seed", True)   # setattr keeps this runnable at HEAD too
    cfg.brain_regions = [BrainRegion(name=nm, n_neurons=nn, exc_fraction=1.0, internal_density=0.0)
                         for (nm, nn) in regions_spec]
    cfg.region_pathways = []
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.actual_seed_used = int(seed)
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def _ou_traj(seed, regions_spec, per_region_ou, steps=40):
    b = _ou_bridge(seed, regions_spec, per_region_ou, ou_on=True)
    for _ in range(steps):
        b._run_one_simulation_step()
    idx = np.asarray(sorted(int(i) for i in b.region_manager.indices("R")), dtype=np.int64)
    return np.asarray(b.cp_ou_current, dtype=np.float64)[idx].copy()


def ou_checks(seed):
    a_on = _ou_traj(seed, [("R", 20)], True)
    c_on = _ou_traj(seed, [("X", 30), ("R", 20)], True)
    inv_on = bool(np.array_equal(a_on, c_on))
    a_off = _ou_traj(seed, [("R", 20)], False)
    c_off = _ou_traj(seed, [("X", 30), ("R", 20)], False)
    inv_off = bool(np.array_equal(a_off, c_off))
    d1 = _ou_traj(seed, [("X", 30), ("R", 20)], True)
    d2 = _ou_traj(seed, [("X", 30), ("R", 20)], True)
    det = bool(np.array_equal(d1, d2))
    off_delta = float(np.max(np.abs(a_off - c_off))) if a_off.size else 0.0
    return inv_on, inv_off, det, off_delta


# ─────────────────────────────────────────────────────────────────────────────────── wiring seam ────────────
def _mgr(regions_spec, pathways_spec, seed):
    from sim.regions import RegionManager, BrainRegion, RegionPathway
    regions = [BrainRegion(name=nm, n_neurons=nn, exc_fraction=1.0, internal_density=dens)
               for (nm, nn, dens) in regions_spec]
    pathways = [RegionPathway(from_region=fr, to_region=to, density=dn, weight_mean=1.0, weight_jitter=0.1,
                              plastic=False) for (fr, to, dn) in pathways_spec]
    mgr = RegionManager(regions, pathways)
    mgr.initialize(seed=int(seed))
    return mgr


def _region_internal_local(mgr, plan, region):
    """R's internal entry normalized to LOCAL indices (subtract R's base) + weights -> an order-invariant signature."""
    entry = plan.get(f"{region}_internal")
    if entry is None:
        return None
    base = min(int(i) for i in mgr.indices(region))
    pre = [int(p) - base for p in entry["pre_indices"]]
    post = [int(p) - base for p in entry["post_indices"]]
    w = [float(x) for x in entry["initial_weights"]]
    return (tuple(pre), tuple(post), tuple(w))


def _pathway_sig(plan, name):
    entry = plan.get(name)
    if entry is None:
        return None
    return (tuple(int(p) for p in entry["pre_indices"]), tuple(int(p) for p in entry["post_indices"]),
            tuple(float(x) for x in entry["initial_weights"]))


def wiring_checks(seed):
    D = 0.35
    # region-internal order (R alone vs R after spacer X): R's LOCAL pattern must be order-invariant when ON.
    def r_local(regions_spec, per_region):
        m = _mgr(regions_spec, [], seed)
        plan = m.build_wiring_plan(seed=int(seed), per_region_seed=per_region)
        return _region_internal_local(m, plan, "R")
    ri_on = bool(r_local([("R", 22, D)], True) == r_local([("X", 30, D), ("R", 22, D)], True))
    ri_off = bool(r_local([("R", 22, D)], False) == r_local([("X", 30, D), ("R", 22, D)], False))

    # pathway order (A->B alone vs after an extra B->A pathway): A->B's placement must be order-invariant when ON.
    def ab_sig(pathways_spec, per_region):
        m = _mgr([("A", 18, 0.0), ("B", 18, 0.0)], pathways_spec, seed)
        plan = m.build_wiring_plan(seed=int(seed), per_region_seed=per_region)
        return _pathway_sig(plan, "pathway_A_to_B")
    pw_on = bool(ab_sig([("A", "B", D)], True) == ab_sig([("B", "A", D), ("A", "B", D)], True))
    pw_off = bool(ab_sig([("A", "B", D)], False) == ab_sig([("B", "A", D), ("A", "B", D)], False))

    # determinism (ON): build the same plan twice -> identical.
    m = _mgr([("X", 30, D), ("R", 22, D)], [("R", "X", D)], seed)
    p1 = m.build_wiring_plan(seed=int(seed), per_region_seed=True)
    p2 = m.build_wiring_plan(seed=int(seed), per_region_seed=True)
    det = bool(json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True))

    inv_on = bool(ri_on and pw_on)
    inv_off = bool(ri_off and pw_off)   # want False (order-dependent) for BOTH under the flag OFF
    return inv_on, inv_off, det, (ri_on, pw_on, ri_off, pw_off)


# ─────────────────────────────────────────────────────────────────────────────── off-path byte-identity ─────
def _off_hashes(seeds):
    out = {}
    for s in seeds:
        h = hashlib.sha256()
        b = _ou_bridge(s, [("X", 30), ("R", 20)], per_region_ou=False, ou_on=True)
        for _ in range(20):
            b._run_one_simulation_step()
        for nm in ("cp_membrane_potential_v", "cp_recovery_variable_u", "cp_ou_current",
                   "cp_neuron_firing_thresholds"):
            a = getattr(b, nm, None)
            h.update(nm.encode() + (b":none" if a is None else np.asarray(a, dtype=np.float32).tobytes()))
        m = _mgr([("X", 30, 0.35), ("R", 22, 0.35)], [("R", "X", 0.35)], s)
        plan = m.build_wiring_plan(seed=int(s))   # NO per_region_seed kwarg -> default (runs at HEAD too)
        h.update(json.dumps(plan, sort_keys=True).encode())
        out[str(s)] = h.hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--mode", choices=["all", "off"], default="all")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.mode == "off":
        print("OFF_HASHES " + json.dumps(_off_hashes(seeds)))
        return

    print("=== PER-REGION OU-NOISE + WIRING SEED — ENGINE VERIFY ===")
    ou_inv, ou_offdep, ou_det = [], [], []
    wr_inv, wr_offdep, wr_det = [], [], []
    worst_ou_off = 0.0
    for s in seeds:
        oi, oo, od, oud = ou_checks(s)
        ou_inv.append(oi); ou_offdep.append(not oo); ou_det.append(od); worst_ou_off = max(worst_ou_off, oud)
        wi, wo, wd, wdet = wiring_checks(s)
        wr_inv.append(wi); wr_offdep.append(not wo); wr_det.append(wd)
        print(f"  [seed {s}] OU: inv_ON={oi} inv_OFF={oo}(want F,delta={oud:.2e}) det={od} | "
              f"WIRE: inv_ON={wi} inv_OFF={wo}(want F) det={wd} {wdet}", flush=True)

    n = len(seeds)
    n_ou_inv, n_ou_off, n_ou_det = sum(ou_inv), sum(ou_offdep), sum(ou_det)
    n_wr_inv, n_wr_off, n_wr_det = sum(wr_inv), sum(wr_offdep), sum(wr_det)
    print("\n=== VERDICT (engine) ===")
    print(f"  OU   position-INVARIANT (ON):  {n_ou_inv}/{n}  -> {'GO' if n_ou_inv==n else 'FAIL'}")
    print(f"  OU   position-DEPENDENT (OFF): {n_ou_off}/{n} (confirms the bug; worst off-delta={worst_ou_off:.2e})")
    print(f"  OU   determinism (ON):         {n_ou_det}/{n}  -> {'GO' if n_ou_det==n else 'FAIL'}")
    print(f"  WIRE order-INVARIANT (ON):     {n_wr_inv}/{n}  -> {'GO' if n_wr_inv==n else 'FAIL'}")
    print(f"  WIRE order-DEPENDENT (OFF):    {n_wr_off}/{n} (confirms the bug)")
    print(f"  WIRE determinism (ON):         {n_wr_det}/{n}  -> {'GO' if n_wr_det==n else 'FAIL'}")

    from tools.verdict import Verdict
    v = Verdict("per_region_ou_seed + per_region_wiring_seed engine features")
    v.require("ou_position_invariant_on", n_ou_inv, expect=n,
              note="R's cp_ou_current trajectory byte-identical alone-vs-co-resident, per_region_ou_seed ON, all seeds")
    v.require("ou_position_dependent_off_control", n_ou_off, expect=n,
              note="the SAME OU trajectory DIFFERS with the flag OFF (load-bearing, not a no-op)")
    v.require("ou_determinism_on", n_ou_det, expect=n, note="co-resident OU pool built twice at one seed is identical")
    v.require("wiring_order_invariant_on", n_wr_inv, expect=n,
              note="R-internal + pathway A->B placement byte-identical regardless of co-residence order, flag ON")
    v.require("wiring_order_dependent_off_control", n_wr_off, expect=n,
              note="the SAME placements DIFFER with the flag OFF (order-dependent shared stream)")
    v.require("wiring_determinism_on", n_wr_det, expect=n, note="the wiring plan built twice at one seed is identical")
    go = (n_ou_inv == n and n_ou_off == n and n_ou_det == n
          and n_wr_inv == n and n_wr_off == n and n_wr_det == n)
    decided = v.decide(go=go, verbose=False)
    payload = {"mode": "per_region_ou_wiring_engine", "n_seeds": n,
               "n_ou_position_invariant_on": n_ou_inv, "n_ou_position_dependent_off": n_ou_off,
               "n_ou_determinism_on": n_ou_det, "worst_ou_off_delta": worst_ou_off,
               "n_wiring_order_invariant_on": n_wr_inv, "n_wiring_order_dependent_off": n_wr_off,
               "n_wiring_determinism_on": n_wr_det,
               "verdict": decided["status"], "preconditions": decided["preconditions"],
               "undefined_reasons": decided["undefined_reasons"]}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
