"""Stage-1 parser-SILENCING localizer (follow-up to _unified_stage1_parser_diag).

Diag #1 proved: on the gen-ON merged bridge the parser role ensembles fire EXACTLY 0.00 (deterministic, settle-
invariant), while gen-OFF parses both sentences perfectly. Weights are provably unchanged by gen training (gain 0,
no CSR re-sort) and OU state is identical (off post-build) — so the silencing is structural/dynamic, not weights.
This probe builds gen-ON ONCE and localizes WHERE the silence comes from:

  (1) reproduce the 0.00 read;
  (2) free-run 30 steps (no external drive) + report per-region mean firing — is some region a self-sustaining
      attractor (e.g. the NMDA gen_concept driven hard during convergence) that could globally suppress the parser?
  (3) the trained conj->role weight magnitude from cp_connections — did the parser train actually grow weights on
      the gen-ON bridge? (If ~0 the train FAILED; if substantial the weights are present and the read is suppressed.)
  (4) drive the action conjunction and read BOTH the conjunction firing AND the role rates — does the drive reach
      parse_conj at all? (conj fires + role silent => downstream suppression; conj silent => the drive/conj is the
      problem.)
  (5) read with OU forced ON (the parser-train condition) — does noise rescue the WTA?
  (6) HARD reset (v<-rest, u/firing zeroed, 80 settle) then read — recovers => dynamic-state; persists => structural.

Run: SIM_BACKEND=cupy python -m research.runners._unified_stage1_parser_diag2 --seed 42
"""
import argparse
import json
import time

import numpy as np

from sim.backend import get_backend, to_host
from research.runners.nav_conv_merged_bridge import (
    build_merged_nav_conv_bridge, _step_reset, ROLES,
)
from research.runners._unified_stage1_parser_diag import role_of_with_rates

DRIVE = 2500.0
V_REST = -65.0
ROI = ["parse_conj", "parse_role", "gen_perception", "gen_concept", "gen_fact",
       "dlpfc_wm", "cortex_ctx", "motor_N", "snc", "rf"]


def region_mean_firing(bridge, rm, name, xp):
    idx = rm.indices(name)
    return float(to_host(bridge.cp_firing_states[idx].astype(xp.float64).mean()))


def conj_and_role_read(bridge, conj_arr, role_arr, position, voice, test_steps, reset):
    """Drive (position,voice) conjunction; accumulate BOTH the conjunction's own firing AND each role rate."""
    xp, _ = get_backend()
    n = bridge.core_config.num_neurons
    k = position * 2 + (0 if voice in (0, "active") else 1)
    _step_reset(bridge, reset)
    cur = xp.zeros(n, dtype=xp.float32)
    cur[conj_arr[k]] = DRIVE
    bridge.cp_external_input_current[:] = cur
    conj_fire = 0.0
    rates = {r: 0.0 for r in ROLES}
    for _ in range(test_steps):
        bridge._run_one_simulation_step()
        conj_fire += float(to_host(bridge.cp_firing_states[conj_arr[k]].astype(xp.float64).mean()))
        for r in ROLES:
            rates[r] += float(to_host(bridge.cp_firing_states[role_arr[r]].astype(xp.float64).mean()))
    bridge.cp_external_input_current[:] = 0.0
    return conj_fire, rates


def parser_weight_mag(bridge, conj_arr, role_arr, xp):
    """Sum |weight| of conj->role synapses from the FINAL CSR (data-aligned via indptr/indices), the
    finalize_conv_for_nav_gate pattern. conj_arr/role_arr are dicts of index arrays; flatten to all-conj/all-role."""
    csr = bridge.cp_connections
    indptr_h = to_host(csr.indptr)
    post_h = to_host(csr.indices).astype(np.int64)
    data_h = np.asarray(to_host(csr.data)).astype(np.float64)
    nnz = int(post_h.shape[0])
    pre_h = np.zeros(nnz, dtype=np.int64)
    for r in range(int(csr.shape[0])):
        pre_h[int(indptr_h[r]):int(indptr_h[r + 1])] = r
    all_conj = np.asarray(to_host(conj_arr), dtype=np.int64)                 # the 6 conjunction indices (flat array)
    all_role = np.concatenate([np.asarray(to_host(role_arr[r]), dtype=np.int64) for r in role_arr])
    mask = np.isin(pre_h, all_conj) & np.isin(post_h, all_role)
    return {"n_edges": int(mask.sum()), "sum_abs": float(np.abs(data_h[mask]).sum()),
            "mean_abs": float(np.abs(data_h[mask]).mean()) if mask.sum() else 0.0,
            "max_abs": float(np.abs(data_h[mask]).max()) if mask.sum() else 0.0}


def hard_reset(bridge, xp, settle=80):
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_membrane_potential_v[:] = V_REST
    try:
        bridge.cp_recovery_variable_u[:] = 0.0
    except Exception:
        pass
    try:
        bridge.cp_firing_states[:] = 0
    except Exception:
        pass
    for _ in range(settle):
        bridge._run_one_simulation_step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="research/findings/raw/_unified_stage1_parser_diag2.json")
    args = ap.parse_args()
    xp, backend = get_backend()
    print(f"[diag2] backend={backend} seed={args.seed}", flush=True)

    t0 = time.time()
    bridge, h = build_merged_nav_conv_bridge(
        seed=args.seed, co_resident_rf=True, co_resident_perception=True, enable_spiking_wta_readout=True,
        co_resident_generalization=True)
    rm = bridge.region_manager
    conj_arr, role_arr = h["conj_arr"], h["role_arr"]
    print(f"[diag2] built gen-ON in {time.time()-t0:.0f}s | num_neurons={bridge.core_config.num_neurons}", flush=True)
    out = {"seed": args.seed, "backend": backend}

    # (1) reproduce the 0.00 read
    rep = []
    for pos in range(3):
        win, rates = role_of_with_rates(bridge, conj_arr, role_arr, pos, "active", 80, DRIVE, 60)
        rep.append({"pos": pos, "win": win, "rates": {r: round(rates[r], 3) for r in ROLES}})
    print(f"[diag2] (1) reproduce: {[ (r['pos'], r['win'], r['rates']) for r in rep ]}", flush=True)
    out["reproduce"] = rep

    # (2) free-run per-region firing (no external drive) — hunt for a self-sustaining attractor
    _step_reset(bridge, 5)
    roi = [r for r in ROI if r in rm.region_indices_dict()]
    acc = {r: 0.0 for r in roi}
    FR = 30
    for _ in range(FR):
        bridge._run_one_simulation_step()
        for r in roi:
            acc[r] += region_mean_firing(bridge, rm, r, xp)
    freerun = {r: round(acc[r] / FR, 4) for r in roi}
    print(f"[diag2] (2) free-run mean firing/region/step: {json.dumps(freerun)}", flush=True)
    out["freerun_firing"] = freerun

    # (3) trained conj->role weight magnitude
    wmag = parser_weight_mag(bridge, conj_arr, role_arr, xp)
    print(f"[diag2] (3) conj->role weights: {json.dumps(wmag)}", flush=True)
    out["parser_weights"] = wmag

    # (4) drive the action conjunction (pos 1) and read conj firing + role rates
    conj_fire, rates4 = conj_and_role_read(bridge, conj_arr, role_arr, 1, "active", 80, 60)
    print(f"[diag2] (4) drive pos1: conj_fire={conj_fire:.3f}  role_rates={ {r: round(rates4[r],3) for r in ROLES} }",
          flush=True)
    out["driven_pos1"] = {"conj_fire": round(conj_fire, 4), "rates": {r: round(rates4[r], 3) for r in ROLES}}

    # (5) OU forced ON
    cc = bridge.core_config
    saved_ou = cc.enable_ou_process
    cc.enable_ou_process = True
    ou_rep = []
    for pos in range(3):
        win, rates = role_of_with_rates(bridge, conj_arr, role_arr, pos, "active", 80, DRIVE, 60)
        ou_rep.append({"pos": pos, "win": win, "rates": {r: round(rates[r], 3) for r in ROLES}})
    cc.enable_ou_process = saved_ou
    print(f"[diag2] (5) OU-ON read: {[ (r['pos'], r['win'], r['rates']) for r in ou_rep ]}", flush=True)
    out["ou_on_read"] = ou_rep

    # (6) hard reset then read
    hard_reset(bridge, xp, settle=80)
    hr_rep = []
    for pos in range(3):
        win, rates = role_of_with_rates(bridge, conj_arr, role_arr, pos, "active", 80, DRIVE, 60)
        hr_rep.append({"pos": pos, "win": win, "rates": {r: round(rates[r], 3) for r in ROLES}})
    print(f"[diag2] (6) HARD-RESET read: {[ (r['pos'], r['win'], r['rates']) for r in hr_rep ]}", flush=True)
    out["hard_reset_read"] = hr_rep

    # interpretation
    role_silent = all(all(v == 0.0 for v in r["rates"].values()) for r in rep)
    conj_fires = out["driven_pos1"]["conj_fire"] > 0.0
    weights_present = wmag["sum_abs"] > 1e-3
    ou_rescues = any(any(v > 0.0 for v in r["rates"].values()) for r in ou_rep)
    hardreset_rescues = any(any(v > 0.0 for v in r["rates"].values()) for r in hr_rep)
    out["interpretation"] = {
        "role_silent_default": role_silent, "conj_fires_under_drive": conj_fires,
        "trained_weights_present": weights_present, "ou_rescues": ou_rescues,
        "hard_reset_rescues": hardreset_rescues,
        "diagnosis": (
            "weights MISSING (parser train failed on gen-ON)" if not weights_present else
            "drive does NOT reach parse_conj (conj silent)" if not conj_fires else
            "dynamic-state suppression (hard reset rescues)" if hardreset_rescues else
            "noise-dependent WTA (OU rescues)" if ou_rescues else
            "structural suppression: weights present + conj fires but role silent even after reset"),
    }
    print(f"\n[diag2] SUMMARY {json.dumps(out['interpretation'], indent=2)}", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[diag2] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
