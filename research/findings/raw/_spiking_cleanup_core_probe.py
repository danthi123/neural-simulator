"""(spine item 2, cheap-first) Spiking cleanup ON the core SimulationBridge.

The composer's cleanup is numpy: `words[argmax(concepts[w] @ est)]` -- a dot-product match + argmax. This probe
replaces BOTH with spiking dynamics on the core bridge: concept codes become synaptic RECEPTIVE FIELDS (the match
becomes synaptic propagation) and lateral inhibition selects the winner (the argmax becomes WTA). Operates on the
composer's ON/OFF channels so it handles signed codes: concept c's drive = code_c_on . input_ON + code_c_off .
input_OFF (a non-negative agreement match), driven by the cue's (e_on, e_off).

Cheap-first FALSIFICATION (pre-registered):
  - RECOVERY: the spiking cleanup picks the correct concept from a moderately-corrupted cue (>= 0.90), at the
    target regime cos-0.80 D=2048 (and the easy cos-0 first).
  - GRACEFUL DEGRADATION (anti-cheat): recovery falls toward chance as noise -> full randomization (not magically
    always-right).
  - W_INH>0 sharpens the match into a single-winner WTA (the fully-spiking selection); W_INH=0 tests the match alone
    (argmax readout).

  python -m research.findings.raw._spiking_cleanup_core_probe --m 32 --d 512 --rho 0.0
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host
from research.runners.core_sim_composition import _center, onoff, _scale_to_current
from research.findings.raw._core_composer_v320_capacity_probe import make_codes

RESET_STEPS = 20
INPUT_DRIVE = 2500.0


def build_cleanup_bridge(seed, codes, w_match, w_inh):
    """codes: (M, D) centered+normalized concept codes -> input_ON(D)+input_OFF(D)+concept(M) matched-filter+WTA."""
    M, D = codes.shape
    cfg = CoreSimConfig()
    cfg.num_neurons = 2 * D + M
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed); cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0; cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = 20.0

    in_on = np.arange(0, D); in_off = np.arange(D, 2 * D); concept = np.arange(2 * D, 2 * D + M)
    code_on = np.maximum(codes, 0.0); code_off = np.maximum(-codes, 0.0)
    mpre, mpost, mw = [], [], []                      # matched filter (E_TO_E): codes as synaptic receptive fields
    for c in range(M):
        for i in range(D):
            if code_on[c, i] > 0:
                mpre.append(int(in_on[i])); mpost.append(int(concept[c])); mw.append(float(code_on[c, i] * w_match))
            if code_off[c, i] > 0:
                mpre.append(int(in_off[i])); mpost.append(int(concept[c])); mw.append(float(code_off[c, i] * w_match))
    plan = {"match": {"pre_indices": mpre, "post_indices": mpost,
                      "initial_weights": np.array(mw, dtype=np.float32), "plastic": False,
                      "conn_type": "E_TO_E", "count": len(mpre)}}
    if w_inh > 0:                                     # WTA (I_TO_E): each concept inhibits the others
        ipre, ipost, iw = [], [], []
        for a in range(M):
            for b in range(M):
                if a != b:
                    ipre.append(int(concept[a])); ipost.append(int(concept[b])); iw.append(float(w_inh))
        plan["wta"] = {"pre_indices": ipre, "post_indices": ipost,
                       "initial_weights": np.array(iw, dtype=np.float32), "plastic": False,
                       "conn_type": "I_TO_E", "count": len(ipre)}
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    bridge.inject_explicit_wiring(plan)
    xp, _ = get_backend()
    idx = {"in_on": xp.asarray(in_on, dtype=xp.int64), "in_off": xp.asarray(in_off, dtype=xp.int64),
           "concept": xp.asarray(concept, dtype=xp.int64)}
    return bridge, idx


def cleanup_spiking(bridge, idx, D, M, est, concept_bias, run_steps=80):
    xp, _ = get_backend()
    e_on, e_off = onoff(est)
    on_cur, off_cur = _scale_to_current(e_on, e_off, INPUT_DRIVE)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge._run_one_simulation_step()
    cur = xp.zeros(2 * D + M, dtype=xp.float32)
    cur[idx["in_on"]] = xp.asarray(on_cur.astype(np.float32))
    cur[idx["in_off"]] = xp.asarray(off_cur.astype(np.float32))
    cur[idx["concept"]] = concept_bias
    bridge.cp_external_input_current[:] = cur
    acc = xp.zeros(M, dtype=xp.float64)
    for _ in range(run_steps):
        bridge._run_one_simulation_step()
        acc += bridge.cp_firing_states[idx["concept"]].astype(xp.float64)
    bridge.cp_external_input_current[:] = 0.0
    return to_host(acc) / run_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--w-match", type=float, default=40.0)
    ap.add_argument("--w-inh", type=float, default=0.0)
    ap.add_argument("--concept-bias", type=float, default=-150.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    codes = make_codes(args.m, args.d, args.seed, rho=args.rho)
    codes = np.stack([_center(c) for c in codes])
    bridge, idx = build_cleanup_bridge(args.seed, codes, args.w_match, args.w_inh)

    rng = np.random.default_rng(args.seed + 1)
    sigmas = [0.0, 0.02, 0.04, 0.06, 0.10]            # scaled to the code magnitude: cue-code cos ~1.0 -> ~0.4
    rows = {}
    for sigma in sigmas:
        ok = ok_np = tot = 0
        coss = []
        for c in range(args.m):
            for _ in range(args.trials):
                est = codes[c] + rng.normal(0, sigma, size=args.d)
                coss.append(float(codes[c] @ est / (np.linalg.norm(est) + 1e-12)))
                rates = cleanup_spiking(bridge, idx, args.d, args.m, est, args.concept_bias)
                ok += int(int(np.argmax(rates)) == c)
                ok_np += int(int(np.argmax(codes @ est)) == c)        # numpy-argmax baseline (what the composer does)
                tot += 1
        rows[f"{sigma}"] = {"spiking": ok / tot, "numpy": ok_np / tot, "cue_cos": float(np.mean(coss))}
        print(f"[clean] M={args.m} D={args.d} rho={args.rho} w_inh={args.w_inh} sigma={sigma} "
              f"cue_cos={np.mean(coss):.3f}  spiking={ok / tot:.3f}  numpy={ok_np / tot:.3f}", flush=True)

    res = {"m": args.m, "d": args.d, "rho": args.rho, "w_match": args.w_match, "w_inh": args.w_inh,
           "concept_bias": args.concept_bias, "recovery_by_sigma": rows}
    print("[clean] " + json.dumps(res), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
