"""Stage-1 parser-suppression A/B (diag #3): WHY is parse_role silent on gen-ON though conj fires + weights present?

Diag #2 localized: gen-ON => conj fires (rate 20), conj->role weights present (mean 6.96), yet parse_role rate = 0,
not rescued by OU or a hard membrane reset => STRUCTURAL suppression. This probe builds gen-OFF AND gen-ON and, under
the SAME conj drive, measures parse_role's actual membrane + the excitatory/inhibitory conductance it RECEIVES, plus
the static count of excitatory vs inhibitory synapses INTO parse_role. The A/B pinpoints the mechanism:

  - gen-ON V low + g_i HIGH (+ more inh edges into parse_role)   => inhibitory wiring differs (RNG/order shift)
  - gen-ON V low + g_i HIGH (+ SAME inh edges)                   => same wiring, inhibitory sources fire more (dynamic)
  - gen-ON V low + g_e LOW (conj current not delivered)          => transmission/gate issue on the conj->role edges
  - gen-ON V high but no spike                                   => depolarization block / threshold issue

Run: SIM_BACKEND=cupy python -m research.runners._unified_stage1_parser_diag3 --seed 42
"""
import argparse
import json
import time

import numpy as np

from sim.backend import get_backend, to_host
from research.runners.nav_conv_merged_bridge import (
    build_merged_nav_conv_bridge, _step_reset, ROLES,
)

DRIVE = 2500.0


def _all_inhibitory_set(rm):
    s = set()
    for region in rm.regions():
        s.update(int(i) for i in rm.inhibitory_indices(region.name))
    return s


def parse_role_input_wiring(bridge, rm, xp):
    """Static: edges INTO parse_role, split by whether the PRE neuron is inhibitory. Returns counts + weight sums."""
    csr = bridge.cp_connections
    indptr_h = to_host(csr.indptr)
    post_h = to_host(csr.indices).astype(np.int64)
    data_h = np.asarray(to_host(csr.data)).astype(np.float64)
    nnz = int(post_h.shape[0])
    pre_h = np.zeros(nnz, dtype=np.int64)
    for r in range(int(csr.shape[0])):
        pre_h[int(indptr_h[r]):int(indptr_h[r + 1])] = r
    role_idx = np.asarray(list(rm.indices("parse_role")), dtype=np.int64)
    into_role = np.isin(post_h, role_idx)
    inh_set = _all_inhibitory_set(rm)
    pre_is_inh = np.array([int(p) in inh_set for p in pre_h], dtype=bool)
    exc_mask = into_role & (~pre_is_inh)
    inh_mask = into_role & pre_is_inh
    return {
        "n_into_role": int(into_role.sum()),
        "n_exc_edges": int(exc_mask.sum()), "n_inh_edges": int(inh_mask.sum()),
        "sum_w_exc": float(data_h[exc_mask].sum()), "sum_w_inh": float(data_h[inh_mask].sum()),
    }


def driven_parse_role_state(bridge, conj_arr, role_arr, position, xp, test_steps=80, reset=60):
    """Drive (position, active) conjunction; report parse_role's membrane + g_e/g_i accumulated over the read."""
    n = bridge.core_config.num_neurons
    k = position * 2
    _step_reset(bridge, reset)
    cur = xp.zeros(n, dtype=xp.float32)
    cur[conj_arr[k]] = DRIVE
    bridge.cp_external_input_current[:] = cur
    role_all = xp.concatenate([role_arr[r] for r in role_arr])
    v = bridge.cp_membrane_potential_v
    ge = getattr(bridge, "cp_conductance_g_e", None)
    gi = getattr(bridge, "cp_conductance_g_i", None)
    acc = {"v_mean": 0.0, "v_max": -1e9, "ge_mean": 0.0, "gi_mean": 0.0, "role_fire": 0.0, "conj_fire": 0.0}
    for _ in range(test_steps):
        bridge._run_one_simulation_step()
        vh = to_host(v[role_all].astype(xp.float64))
        acc["v_mean"] += float(vh.mean())
        acc["v_max"] = max(acc["v_max"], float(vh.max()))
        if ge is not None:
            acc["ge_mean"] += float(to_host(ge[role_all].astype(xp.float64).mean()))
        if gi is not None:
            acc["gi_mean"] += float(to_host(gi[role_all].astype(xp.float64).mean()))
        acc["role_fire"] += float(to_host(bridge.cp_firing_states[role_all].astype(xp.float64).mean()))
        acc["conj_fire"] += float(to_host(bridge.cp_firing_states[conj_arr[k]].astype(xp.float64).mean()))
    bridge.cp_external_input_current[:] = 0.0
    acc["v_mean"] /= test_steps
    acc["ge_mean"] /= test_steps
    acc["gi_mean"] /= test_steps
    return {kk: round(vv, 4) for kk, vv in acc.items()}


def run_one(seed, gen_flag):
    xp, _ = get_backend()
    print(f"\n[diag3] build gen={'ON' if gen_flag else 'OFF'}...", flush=True)
    t0 = time.time()
    bridge, h = build_merged_nav_conv_bridge(
        seed=seed, co_resident_rf=True, co_resident_perception=True, enable_spiking_wta_readout=True,
        co_resident_generalization=gen_flag)
    rm = bridge.region_manager
    conj_arr, role_arr = h["conj_arr"], h["role_arr"]
    print(f"[diag3] built in {time.time()-t0:.0f}s | num_neurons={bridge.core_config.num_neurons}", flush=True)
    wiring = parse_role_input_wiring(bridge, rm, xp)
    state = driven_parse_role_state(bridge, conj_arr, role_arr, 1, xp)   # pos1 = action
    print(f"[diag3] gen={'ON ' if gen_flag else 'OFF'} parse_role INPUT WIRING: {json.dumps(wiring)}", flush=True)
    print(f"[diag3] gen={'ON ' if gen_flag else 'OFF'} parse_role DRIVEN STATE: {json.dumps(state)}", flush=True)
    del bridge
    return {"gen": gen_flag, "wiring": wiring, "driven_state": state}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="research/findings/raw/_unified_stage1_parser_diag3.json")
    args = ap.parse_args()
    xp, backend = get_backend()
    print(f"[diag3] backend={backend} seed={args.seed}", flush=True)
    off = run_one(args.seed, False)
    on = run_one(args.seed, True)

    w_off, w_on = off["wiring"], on["wiring"]
    s_off, s_on = off["driven_state"], on["driven_state"]
    wiring_same = (w_off == w_on)
    diag = {
        "wiring_identical": wiring_same,
        "inh_edges_off": w_off["n_inh_edges"], "inh_edges_on": w_on["n_inh_edges"],
        "sum_w_inh_off": w_off["sum_w_inh"], "sum_w_inh_on": w_on["sum_w_inh"],
        "role_v_mean_off": s_off["v_mean"], "role_v_mean_on": s_on["v_mean"],
        "role_gi_off": s_off["gi_mean"], "role_gi_on": s_on["gi_mean"],
        "role_ge_off": s_off["ge_mean"], "role_ge_on": s_on["ge_mean"],
        "role_fire_off": s_off["role_fire"], "role_fire_on": s_on["role_fire"],
    }
    if not wiring_same:
        diag["mechanism"] = "WIRING DIFFERS into parse_role (gen presence shifts parser wiring/RNG order)"
    elif s_on["gi_mean"] > s_off["gi_mean"] * 1.5 and s_on["ge_mean"] >= s_off["ge_mean"] * 0.5:
        diag["mechanism"] = "SAME wiring but higher inhibition on gen-ON (dynamic: inhibitory sources fire more)"
    elif s_on["ge_mean"] < s_off["ge_mean"] * 0.5:
        diag["mechanism"] = "conj->role excitatory current NOT delivered on gen-ON (transmission/gate)"
    else:
        diag["mechanism"] = "same wiring + same g_e/g_i but role silent -> deeper (threshold/global term)"
    print(f"\n[diag3] SUMMARY {json.dumps(diag, indent=2)}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"diag": diag, "off": off, "on": on}, f, indent=2)
    print(f"[diag3] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
