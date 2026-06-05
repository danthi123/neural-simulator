"""(de-risk A, deeper fix) TWO-STAGE spiking normalization cleanup: INPUT-layer divisive normalization
(a spiking FS pool that normalizes the est's ON/OFF input population firing) BEFORE the matched filter,
PLUS the concept-layer (output) divisive normalization. The NEGATIVE finding
(`2026-06-04-composer-cleanup-divisive-norm-NEGATIVE.md`) showed output-only norm is not SEED-ROBUST: the
absolute threshold is scale-variant because the est magnitude differs seed-to-seed. Input normalization
standardizes the input drive so the matched-filter drive is ~scale-invariant -> a single fixed operating
point should reach numpy parity across seeds.

GOAL: a FIXED operating point whose worst-case recovery across seeds 42/43/44 reaches numpy parity (>=~0.95).
Self-contained: captures the real est for 3 seeds ONCE, sweeps a focused input-norm grid, aggregates
min-across-seeds, prints the robust-best + verdict in ONE run.

  python -u -m research.findings.raw._spiking_cleanup_2stage --out research/findings/raw/_2stage.json

NO sim/ edits; reuse-by-import (capture_real_est + onoff + _scale_to_current from the divnorm probe).
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


def build_2stage_bridge(seed, codes, w_match, w_in_cfs, w_in_fs, w_cfs, w_fs, n_in_fs, n_fs, einh,
                        ou_std=20.0):
    """Layout: in_on[0,D) in_off[D,2D) concept[2D,2D+M) concept_FS[.. +n_fs) input_FS[.. +n_in_fs).
      Stage 1 (INPUT norm): in_on/in_off -> input_FS (E_TO_I pooling); input_FS -> in_on/in_off (I_TO_E
        shunting). Normalizes the est input population firing so the matched-filter drive is scale-invariant.
      Stage 2 (matched filter): in_on/in_off -> concept (codes as receptive fields).
      Stage 3 (OUTPUT norm): concept -> concept_FS (E_TO_I); concept_FS -> concept (I_TO_E shunting).
    BOTH FS pools carry the inhibitory trait (g_i routing keys on the presynaptic trait, not conn_type)."""
    M, D = codes.shape
    base_cfs = 2 * D + M
    base_ifs = base_cfs + n_fs
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
    concept = np.arange(2 * D, 2 * D + M)
    cfs = np.arange(base_cfs, base_cfs + n_fs)
    ifs = np.arange(base_ifs, base_ifs + n_in_fs)
    inp = np.concatenate([in_on, in_off])

    pre, post, w, ct = [], [], [], []

    def add(ps, qs, val, c):
        for p in ps:
            for q in qs:
                pre.append(int(p)); post.append(int(q)); w.append(float(val)); ct.append(c)

    # Stage 1: input divisive normalization
    add(inp, ifs, w_in_cfs, "E_TO_I")          # input -> input_FS (pool total input activity)
    add(ifs, inp, w_in_fs, "I_TO_E")           # input_FS -> input (divisive shunt)
    # Stage 2: matched filter (codes as receptive fields, ON/OFF)
    code_on = np.maximum(codes, 0.0); code_off = np.maximum(-codes, 0.0)
    mpre, mpost, mw = [], [], []
    for c in range(M):
        for i in range(D):
            if code_on[c, i] > 0:
                mpre.append(int(in_on[i])); mpost.append(int(concept[c])); mw.append(float(code_on[c, i] * w_match))
            if code_off[c, i] > 0:
                mpre.append(int(in_off[i])); mpost.append(int(concept[c])); mw.append(float(code_off[c, i] * w_match))
    # Stage 3: output divisive normalization
    add(concept, cfs, w_cfs, "E_TO_I")
    add(cfs, concept, w_fs, "I_TO_E")

    plan = {
        "match": {"pre_indices": mpre, "post_indices": mpost,
                  "initial_weights": np.array(mw, dtype=np.float32), "plastic": False,
                  "conn_type": "E_TO_E", "count": len(mpre)},
        "norm": {"pre_indices": pre, "post_indices": post,
                 "initial_weights": np.array(w, dtype=np.float32), "plastic": False,
                 "conn_type": "E_TO_E", "count": len(pre)},
    }
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    xp, _ = get_backend()
    tr = bridge.cp_traits
    tr[:] = 0
    fs_all = np.concatenate([cfs, ifs])
    tr[xp.asarray(fs_all, dtype=tr.dtype)] = INH_TRAIT   # BOTH FS pools inhibitory (before first step)
    bridge.cp_traits = tr
    bridge._cached_inhibitory_mask = None
    bridge.inject_explicit_wiring(plan)
    idx = {"in_on": xp.asarray(in_on, dtype=xp.int64), "in_off": xp.asarray(in_off, dtype=xp.int64),
           "concept": xp.asarray(concept, dtype=xp.int64)}
    return bridge, idx


def cleanup(bridge, idx, M, est, concept_bias, run_steps, input_drive=INPUT_DRIVE):
    xp, _ = get_backend()
    e_on, e_off = onoff(est)
    on_cur, off_cur = _scale_to_current(e_on, e_off, input_drive)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge._run_one_simulation_step()
    cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
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


def eval_op(captured, op, run_steps, input_drive):
    """Evaluate one operating point across all captured seeds -> dict seed->recovery + min/mean."""
    recs = {}
    for seed, (items, code_mat, widx, words) in captured.items():
        M = len(words)
        bridge, idx = build_2stage_bridge(seed, code_mat, op["w_match"], op["w_in_cfs"], op["w_in_fs"],
                                          op["w_cfs"], op["w_fs"], op["n_in_fs"], op["n_fs"], op["einh"])
        ok = 0
        for est, true, _ in items:
            rates = cleanup(bridge, idx, M, est, op["bias"], run_steps, input_drive)
            ok += int(words[int(np.argmax(rates))] == true)
        recs[seed] = ok / len(items)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--vocab", type=int, default=320)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-flat", type=int, default=10)
    ap.add_argument("--n-attr", type=int, default=5)
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
        print(f"[2stage] captured seed {s}: {len(items)} items, numpy={numpy_rec[s]:.3f}", flush=True)

    # Focused first-pass grid: input-norm strengths (the new lever) at a fixed sensible output-norm + match/bias.
    grid = []
    for w_in_cfs, w_in_fs in itertools.product([100, 200], [25, 50]):
        for w_match, bias in [(60, -500), (100, -700)]:
            grid.append({"w_match": w_match, "bias": bias, "w_in_cfs": w_in_cfs, "w_in_fs": w_in_fs,
                         "w_cfs": 15, "w_fs": 8, "n_in_fs": 60, "n_fs": 40, "einh": -80})

    results = []
    for op in grid:
        recs = eval_op(captured, op, args.run_steps, args.input_drive)
        mn = min(recs.values()); mean = sum(recs.values()) / len(recs)
        results.append({"op": op, "per_seed": {s: round(r, 3) for s, r in recs.items()},
                        "min": mn, "mean": mean})
        print(f"[2stage] w_in_cfs={op['w_in_cfs']} w_in_fs={op['w_in_fs']} w_match={op['w_match']} "
              f"bias={op['bias']} -> per_seed={ {s: round(r,3) for s,r in recs.items()} } min={mn:.3f}", flush=True)

    results.sort(key=lambda r: (r["min"], r["mean"]), reverse=True)
    best = results[0]
    np_min = min(numpy_rec.values())
    verdict = "GO" if best["min"] >= 0.95 else "NEGATIVE"
    print(f"\n[ROBUST BEST] min={best['min']:.3f} mean={best['mean']:.3f} per_seed={best['per_seed']} op={best['op']}")
    print(f"[numpy parity bar] min numpy={np_min:.3f}")
    print(f"[VERDICT] two-stage robust worst-case {best['min']:.3f} vs numpy {np_min:.3f} -> {verdict}")
    if args.out:
        json.dump({"numpy_rec": numpy_rec, "robust_best": best, "all": results, "verdict": verdict},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
