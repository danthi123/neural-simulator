"""PER-REGION PARAMETER-HETEROGENEITY — the ENGINE-feature verify (the `sim/` edit's substrate-level proof).

Complements `_per_region_param_het_cluster_verify.py` (the real-organ read proof). This runner exercises the
`cfg.per_region_parameter_heterogeneity` guard DIRECTLY on the substrate arrays:

 (1) POSITION-INVARIANCE (flag ON). Region 'R''s per-neuron Izhikevich param-het slice (cp_izh_a/b/d/C) is
     BYTE-IDENTICAL whether R is built ALONE (offset 0) or co-resident BEHIND a spacer region X (offset 30) --
     the exact perturbation a shared pool introduces. 6/6 True EXPECTED.

 (2) POSITION-DEPENDENCE (flag OFF) -- the bug the flag fixes. The SAME comparison with the flag OFF DIFFERS
     (the global size-n draw hands R a position-shifted slice). 6/6 False EXPECTED (i.e. NOT invariant).

 (3) DETERMINISM (flag ON). Build the co-resident pool TWICE at one seed -> byte-identical. 6/6 True EXPECTED.

 (4) OFF-PATH HASH (--mode off). Prints a hash of the full param + threshold arrays with the flag OFF, for a
     git-stash byte-identity comparison to HEAD (default-off must be bit-for-bit today):
        # working tree:
        SIM_BACKEND=numpy python -m research.runners._per_region_param_het_engine_verify --mode off
        git stash push -- sim/bridge.py sim/config.py
        SIM_BACKEND=numpy python -m research.runners._per_region_param_het_engine_verify --mode off
        git stash pop
        # the two OFF_HASHES lines must be identical (proven this arc: identical for all 6 seeds).

NO `sim/` edit here (this is the guard's verify). Process backend (numpy for the bit-exact checks).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

_PARAMS = ("cp_izh_a", "cp_izh_b", "cp_izh_d_increment", "cp_izh_C")


def _build(seed, regions_spec, per_region_param):
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
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_reward_modulation", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process",
              "enable_conductance_noise"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True
    if per_region_param:
        setattr(cfg, "per_region_parameter_heterogeneity", True)   # setattr keeps this runnable at HEAD too
    cfg.brain_regions = [BrainRegion(name=nm, n_neurons=nn, exc_fraction=1.0, internal_density=0.0)
                         for (nm, nn) in regions_spec]
    cfg.region_pathways = []
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.actual_seed_used = int(seed)
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def _arr(b, name):
    a = getattr(b, name, None)
    return None if a is None else np.asarray(a, dtype=np.float32)


def _region_slice_hash(b, region):
    idx = np.asarray(sorted(int(i) for i in b.region_manager.indices(region)), dtype=np.int64)
    h = hashlib.sha256()
    for nm in _PARAMS:
        h.update(nm.encode()); h.update(_arr(b, nm)[idx].tobytes())
    return h.hexdigest()


def _full_hash(b):
    h = hashlib.sha256()
    for nm in _PARAMS + ("cp_neuron_firing_thresholds",):
        a = _arr(b, nm)
        h.update(nm.encode() + (b":none" if a is None else a.tobytes()))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--mode", choices=["all", "off"], default="all")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.mode == "off":
        offh = {str(s): _full_hash(_build(s, [("X", 30), ("R", 20)], per_region_param=False)) for s in seeds}
        print("OFF_HASHES " + json.dumps(offh))
        return

    print("=== PER-REGION PARAM-HET — ENGINE VERIFY ===")
    inv_on, inv_off, det = [], [], []
    for s in seeds:
        h_alone = _region_slice_hash(_build(s, [("R", 20)], True), "R")
        h_cores = _region_slice_hash(_build(s, [("X", 30), ("R", 20)], True), "R")
        inv_on.append(h_alone == h_cores)
        o_alone = _region_slice_hash(_build(s, [("R", 20)], False), "R")
        o_cores = _region_slice_hash(_build(s, [("X", 30), ("R", 20)], False), "R")
        inv_off.append(o_alone == o_cores)
        d1 = _full_hash(_build(s, [("X", 30), ("R", 20)], True))
        d2 = _full_hash(_build(s, [("X", 30), ("R", 20)], True))
        det.append(d1 == d2)
        print(f"  [seed {s}] invariant_ON={inv_on[-1]}  invariant_OFF={inv_off[-1]} (want False)  "
              f"determinism_ON={det[-1]}", flush=True)
    n = len(seeds)
    n_inv = sum(inv_on); n_offdep = sum(1 for x in inv_off if not x); n_det = sum(det)
    print("\n=== VERDICT (engine) ===")
    print(f"  position-INVARIANT (flag ON):    {n_inv}/{n}  -> {'GO' if n_inv == n else 'FAIL'}")
    print(f"  position-DEPENDENT (flag OFF):   {n_offdep}/{n} (confirms the bug the flag fixes)")
    print(f"  determinism (flag ON, 2x build): {n_det}/{n}  -> {'GO' if n_det == n else 'FAIL'}")
    from tools.verdict import Verdict
    v = Verdict("per_region_param_het engine feature")
    v.require("position_invariant_on", n_inv, expect=n,
              note="R's param-het slice byte-identical alone-vs-co-resident with the flag ON, all seeds")
    v.require("position_dependent_off_control", n_offdep, expect=n,
              note="the SAME slice DIFFERS with the flag OFF (the flag is load-bearing, not a no-op)")
    v.require("determinism_on", n_det, expect=n,
              note="the co-resident pool built twice at one seed is byte-identical, all seeds")
    decided = v.decide(go=(n_inv == n and n_det == n and n_offdep == n), verbose=False)
    payload = {"mode": "per_region_param_het_engine", "n_seeds": n,
               "n_position_invariant_on": n_inv, "n_position_dependent_off": n_offdep,
               "n_determinism_on": n_det,
               "verdict": decided["status"], "preconditions": decided["preconditions"],
               "undefined_reasons": decided["undefined_reasons"]}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
