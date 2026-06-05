"""(de-risk A, literature-grounded) NEF thresholded cleanup (Stewart-Tang-Eliasmith 2011, the Spaun cleanup).
The fix per the synthesis (2026-06-05-spiking-cleanup-memory-literature-synthesis.md): a rate readout is a LINEAR
reconstructor, not argmax -> off-target codes leak. The cure is THRESHOLD PLACEMENT: each concept neuron fires
ONLY when code_w . est > theta, with theta BETWEEN the off-target similarity (~0) and the true cue-cos (~0.31).
Then the true concept's neurons fire and every off-target neuron stays SILENT -> the per-concept readout is clean.

Architecture (feed-forward, NO recurrent WTA):
  - INPUT normalization (a spiking inhibitory-trait FS pool shunting the est ON/OFF input population) so the
    matched-filter drive is ~scale-invariant = cosine across seeds (the Betteti input-driven idea: cue clamped
    throughout). This is why our input-norm helped 0.844->0.911; here it makes the THRESHOLD seed-invariant.
  - MATCHED filter: n_per neurons per concept, encoders = the stored code (codes as ON/OFF receptive fields).
  - THRESHOLD: a negative concept bias places the firing intercept so off-target -> 0 spikes, true -> fires.
  - READOUT: per-concept summed firing -> argmax (clean because off-target is silent). No output divisive norm.

GOAL: a FIXED operating point whose worst-case recovery across seeds 42/43/44 reaches numpy parity (>=~0.95).

  python -u -m research.findings.raw._spiking_cleanup_nef --out research/findings/raw/_nef.json
NO sim/ edits; reuse-by-import.
"""
from __future__ import annotations
import argparse
import json
import itertools

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host
from research.runners.core_sim_composition import onoff, _scale_to_current
from research.findings.raw._spiking_cleanup_divnorm_probe import capture_real_est

RESET_STEPS = 20
INPUT_DRIVE = 2500.0
INH_TRAIT = 1


def build_nef_bridge(seed, codes, n_per, w_match, w_in_cfs, w_in_fs, n_in_fs, einh, ou_std=20.0):
    """in_on[0,D) in_off[D,2D) concept[2D, 2D+M*n_per) input_FS[base_ifs, +n_in_fs).
      input norm: in -> input_FS (E_TO_I), input_FS -> in (I_TO_E, inhibitory trait, scale-invariance).
      matched filter: in_on/in_off -> concept[w,k] with code_w weights (n_per neurons share concept w's encoder)."""
    M, D = codes.shape
    n_concept = M * n_per
    base_c = 2 * D
    base_ifs = base_c + n_concept
    N = base_ifs + n_in_fs
    cfg = CoreSimConfig()
    cfg.num_neurons = N
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed); cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0; cfg.num_traits = 2
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = float(ou_std)
    cfg.enable_inhibitory_neurons = True
    cfg.inhibitory_trait_indices = [INH_TRAIT]
    cfg.syn_reversal_potential_i = float(einh)

    in_on = np.arange(0, D); in_off = np.arange(D, 2 * D)
    concept = np.arange(base_c, base_c + n_concept)        # concept[w*n_per + k]
    ifs = np.arange(base_ifs, base_ifs + n_in_fs)
    inp = np.concatenate([in_on, in_off])
    code_on = np.maximum(codes, 0.0); code_off = np.maximum(-codes, 0.0)

    pre, post, w, _ct = [], [], [], []
    # input normalization (scale-invariance)
    for p in inp:
        for q in ifs:
            pre.append(int(p)); post.append(int(q)); w.append(float(w_in_cfs))
    for q in ifs:
        for p in inp:
            pre.append(int(q)); post.append(int(p)); w.append(float(w_in_fs))
    norm_plan = {"norm": {"pre_indices": pre, "post_indices": post,
                          "initial_weights": np.array(w, dtype=np.float32), "plastic": False,
                          "conn_type": "E_TO_E", "count": len(pre)}}
    # matched filter: n_per neurons per concept share code_w as encoder
    mpre, mpost, mw = [], [], []
    for c in range(M):
        nz_on = np.where(code_on[c] > 0)[0]; nz_off = np.where(code_off[c] > 0)[0]
        for k in range(n_per):
            cn = int(concept[c * n_per + k])
            for i in nz_on:
                mpre.append(int(in_on[i])); mpost.append(cn); mw.append(float(code_on[c, i] * w_match))
            for i in nz_off:
                mpre.append(int(in_off[i])); mpost.append(cn); mw.append(float(code_off[c, i] * w_match))
    match_plan = {"match": {"pre_indices": mpre, "post_indices": mpost,
                            "initial_weights": np.array(mw, dtype=np.float32), "plastic": False,
                            "conn_type": "E_TO_E", "count": len(mpre)}}

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    xp, _ = get_backend()
    tr = bridge.cp_traits
    tr[:] = 0
    tr[xp.asarray(ifs, dtype=tr.dtype)] = INH_TRAIT
    bridge.cp_traits = tr
    bridge._cached_inhibitory_mask = None
    bridge.inject_explicit_wiring({**norm_plan, **match_plan})
    idx = {"in_on": xp.asarray(in_on, dtype=xp.int64), "in_off": xp.asarray(in_off, dtype=xp.int64),
           "concept": xp.asarray(concept, dtype=xp.int64)}
    return bridge, idx, M, n_per


def cleanup(bridge, idx, M, n_per, est, concept_bias, run_steps, settle_frac=0.4, input_drive=INPUT_DRIVE):
    """Drive the est (clamped throughout = input-driven), read per-concept summed firing over the settled window."""
    xp, _ = get_backend()
    e_on, e_off = onoff(est)
    on_cur, off_cur = _scale_to_current(e_on, e_off, input_drive)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge._run_one_simulation_step()
    cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
    cur[idx["in_on"]] = xp.asarray(on_cur.astype(np.float32))
    cur[idx["in_off"]] = xp.asarray(off_cur.astype(np.float32))
    cur[idx["concept"]] = concept_bias                      # the NEF threshold (negative bias)
    bridge.cp_external_input_current[:] = cur
    acc = xp.zeros(M * n_per, dtype=xp.float64)
    settle = int(run_steps * settle_frac)
    for t in range(run_steps):
        bridge._run_one_simulation_step()
        if t >= settle:
            acc += bridge.cp_firing_states[idx["concept"]].astype(xp.float64)
    bridge.cp_external_input_current[:] = 0.0
    per_concept = to_host(acc).reshape(M, n_per).sum(axis=1)  # clean: off-target silent
    return per_concept


def eval_op(captured, op, run_steps, input_drive):
    recs = {}
    for seed, (items, code_mat, widx, words) in captured.items():
        bridge, idx, M, n_per = build_nef_bridge(seed, code_mat, op["n_per"], op["w_match"], op["w_in_cfs"],
                                                 op["w_in_fs"], op["n_in_fs"], op["einh"])
        ok = 0
        for est, true, _ in items:
            pc = cleanup(bridge, idx, M, n_per, est, op["bias"], run_steps, input_drive=input_drive)
            ok += int(words[int(np.argmax(pc))] == true)
        recs[seed] = ok / len(items)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--vocab", type=int, default=320)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-flat", type=int, default=10)
    ap.add_argument("--n-attr", type=int, default=5)
    ap.add_argument("--n-per", type=int, default=6)
    ap.add_argument("--run-steps", type=int, default=400)
    ap.add_argument("--input-drive", type=float, default=2500.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    captured = {}
    numpy_rec = {}
    for s in args.seeds:
        items, code_mat, widx, words = capture_real_est(s, args.vocab, args.proj_dim, args.n_flat, args.n_attr)
        captured[s] = (items, code_mat, widx, words)
        numpy_rec[s] = sum(int(words[int(np.argmax(code_mat @ est))] == t) for est, t, _ in items) / len(items)
        print(f"[nef] captured seed {s}: {len(items)} items, numpy={numpy_rec[s]:.3f}", flush=True)

    # MULTI-SEED test at the seed-42 winning region (bias=-400/w_match=60, bias=-700/-1000 w_match=120 each hit
    # 1.000 on seed 42). Decisive: does a FIXED op reach numpy parity across seeds 42/43/44 (the threshold on the
    # input-normalized = cosine similarity should be scale-invariant)?
    grid = []
    for bias in [-625, -700, -775]:                       # finer threshold sweep around the winner (-700, w_match 120)
        grid.append({"bias": bias, "w_match": 120, "n_per": args.n_per, "w_in_cfs": 1.0,
                     "w_in_fs": 10.0, "n_in_fs": 60, "einh": -80})

    results = []
    for op in grid:
        recs = eval_op(captured, op, args.run_steps, args.input_drive)
        mn = min(recs.values()); mean = sum(recs.values()) / len(recs)
        results.append({"op": op, "per_seed": {s: round(r, 3) for s, r in recs.items()}, "min": mn, "mean": mean})
        print(f"[nef] bias={op['bias']} w_match={op['w_match']} w_in_cfs={op['w_in_cfs']} -> "
              f"per_seed={ {s: round(r,3) for s,r in recs.items()} } min={mn:.3f}", flush=True)

    results.sort(key=lambda r: (r["min"], r["mean"]), reverse=True)
    best = results[0]
    np_min = min(numpy_rec.values())
    verdict = "GO" if best["min"] >= 0.95 else "NEGATIVE"
    print(f"\n[ROBUST BEST] min={best['min']:.3f} mean={best['mean']:.3f} per_seed={best['per_seed']} op={best['op']}")
    print(f"[VERDICT] NEF cleanup robust worst-case {best['min']:.3f} vs numpy {np_min:.3f} -> {verdict}")
    if args.out:
        json.dump({"numpy_rec": numpy_rec, "robust_best": best, "all": results, "verdict": verdict},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
