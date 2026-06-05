"""(de-risk A, deeper-deeper) ATTRACTOR / winner-take-all spiking cleanup. The divisive-normalization approach
(rate readout) plateaus ~0.91 and is op-fragile because a rate code can't match numpy argmax's infinite
precision + exact scale-invariance. An ATTRACTOR settles to a clean 1-HOT state based on which concept is most
driven (RELATIVE drive), so it is inherently scale-invariant (no input-norm) + exact (converges to one stored
concept regardless of input magnitude) -- the dynamical analogue of argmax.

Mechanism (NO sim/ edits): matched filter (concept codes as ON/OFF receptive fields) + concept self-excitation
(the attractor latch) + a STRONG WTA inhibitory FS pool (concept->FS E_TO_I, FS->concept I_TO_E, inhibitory
trait => conductance shunting). The sustained est drive biases which concept gets ahead; self-excitation latches
the winner; global inhibition (driven by total activity) suppresses losers below threshold -> 1-hot.

GOAL: a FIXED operating point whose worst-case recovery across seeds 42/43/44 reaches numpy parity (>=~0.95).

  python -u -m research.findings.raw._spiking_cleanup_attractor --out research/findings/raw/_attractor.json
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


def build_attractor_bridge(seed, codes, w_match, w_self, w_cfs, w_fs, n_fs, einh, ou_std=20.0):
    """in_on[0,D) in_off[D,2D) concept[2D,2D+M) wta_FS[base,base+n_fs).
      matched filter: in_on/in_off -> concept (codes as RFs, E_TO_E).
      self-excitation: concept_i -> concept_i (E_TO_E self-loop, the attractor latch).
      WTA: concept -> wta_FS (E_TO_I pool), wta_FS -> concept (I_TO_E shunt, inhibitory trait)."""
    M, D = codes.shape
    base = 2 * D + M
    N = base + n_fs
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
    concept = np.arange(2 * D, 2 * D + M); fs = np.arange(base, base + n_fs)
    code_on = np.maximum(codes, 0.0); code_off = np.maximum(-codes, 0.0)

    mpre, mpost, mw = [], [], []
    for c in range(M):
        for i in range(D):
            if code_on[c, i] > 0:
                mpre.append(int(in_on[i])); mpost.append(int(concept[c])); mw.append(float(code_on[c, i] * w_match))
            if code_off[c, i] > 0:
                mpre.append(int(in_off[i])); mpost.append(int(concept[c])); mw.append(float(code_off[c, i] * w_match))
    # self-excitation (attractor latch)
    if w_self > 0:
        for c in range(M):
            mpre.append(int(concept[c])); mpost.append(int(concept[c])); mw.append(float(w_self))
    plan = {"match": {"pre_indices": mpre, "post_indices": mpost,
                      "initial_weights": np.array(mw, dtype=np.float32), "plastic": False,
                      "conn_type": "E_TO_E", "count": len(mpre)}}
    # WTA: concept -> FS (pool), FS -> concept (shunt)
    cpre, cpost, cw = [], [], []
    for c in range(M):
        for j in range(n_fs):
            cpre.append(int(concept[c])); cpost.append(int(fs[j])); cw.append(float(w_cfs))
    ipre, ipost, iw = [], [], []
    for j in range(n_fs):
        for c in range(M):
            ipre.append(int(fs[j])); ipost.append(int(concept[c])); iw.append(float(w_fs))
    plan["pool"] = {"pre_indices": cpre, "post_indices": cpost,
                    "initial_weights": np.array(cw, dtype=np.float32), "plastic": False,
                    "conn_type": "E_TO_I", "count": len(cpre)}
    plan["shunt"] = {"pre_indices": ipre, "post_indices": ipost,
                     "initial_weights": np.array(iw, dtype=np.float32), "plastic": False,
                     "conn_type": "I_TO_E", "count": len(ipre)}

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    xp, _ = get_backend()
    tr = bridge.cp_traits
    tr[:] = 0
    tr[xp.asarray(fs, dtype=tr.dtype)] = INH_TRAIT
    bridge.cp_traits = tr
    bridge._cached_inhibitory_mask = None
    bridge.inject_explicit_wiring(plan)
    idx = {"in_on": xp.asarray(in_on, dtype=xp.int64), "in_off": xp.asarray(in_off, dtype=xp.int64),
           "concept": xp.asarray(concept, dtype=xp.int64)}
    return bridge, idx


def cleanup(bridge, idx, M, est, concept_bias, run_steps, settle_frac=0.5, input_drive=INPUT_DRIVE):
    """Drive the est, let the attractor settle, read concept rates over the LATER (settled) part of the window."""
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
    settle = int(run_steps * settle_frac)
    for t in range(run_steps):
        bridge._run_one_simulation_step()
        if t >= settle:                                 # read only the settled part of the attractor
            acc += bridge.cp_firing_states[idx["concept"]].astype(xp.float64)
    bridge.cp_external_input_current[:] = 0.0
    return to_host(acc) / max(1, run_steps - settle)


def eval_op(captured, op, run_steps, input_drive):
    recs = {}
    for seed, (items, code_mat, widx, words) in captured.items():
        M = len(words)
        bridge, idx = build_attractor_bridge(seed, code_mat, op["w_match"], op["w_self"], op["w_cfs"],
                                             op["w_fs"], op["n_fs"], op["einh"])
        ok = 0
        for est, true, _ in items:
            rates = cleanup(bridge, idx, M, est, op["bias"], run_steps, input_drive=input_drive)
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
    ap.add_argument("--run-steps", type=int, default=500)
    ap.add_argument("--input-drive", type=float, default=2500.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    captured = {}
    numpy_rec = {}
    for s in args.seeds:
        items, code_mat, widx, words = capture_real_est(s, args.vocab, args.proj_dim, args.n_flat, args.n_attr)
        captured[s] = (items, code_mat, widx, words)
        numpy_rec[s] = sum(int(words[int(np.argmax(code_mat @ est))] == t) for est, t, _ in items) / len(items)
        print(f"[attr] captured seed {s}: {len(items)} items, numpy={numpy_rec[s]:.3f}", flush=True)

    # WTA attractor: self-excitation (latch) x inhibition strength (competition) x match gain. concept_bias 0
    # (the attractor decides; no absolute threshold needed -> scale-invariant).
    grid = []
    for w_self, w_fs in itertools.product([5, 15, 40], [20, 50, 100]):
        for w_match in [40, 80]:
            grid.append({"w_match": w_match, "bias": 0.0, "w_self": w_self, "w_cfs": 8.0, "w_fs": w_fs,
                         "n_fs": 40, "einh": -80})

    results = []
    for op in grid:
        recs = eval_op(captured, op, args.run_steps, args.input_drive)
        mn = min(recs.values()); mean = sum(recs.values()) / len(recs)
        results.append({"op": op, "per_seed": {s: round(r, 3) for s, r in recs.items()}, "min": mn, "mean": mean})
        print(f"[attr] w_self={op['w_self']} w_fs={op['w_fs']} w_match={op['w_match']} -> "
              f"per_seed={ {s: round(r,3) for s,r in recs.items()} } min={mn:.3f}", flush=True)

    results.sort(key=lambda r: (r["min"], r["mean"]), reverse=True)
    best = results[0]
    np_min = min(numpy_rec.values())
    verdict = "GO" if best["min"] >= 0.95 else "NEGATIVE"
    print(f"\n[ROBUST BEST] min={best['min']:.3f} mean={best['mean']:.3f} per_seed={best['per_seed']} op={best['op']}")
    print(f"[VERDICT] attractor robust worst-case {best['min']:.3f} vs numpy {np_min:.3f} -> {verdict}")
    if args.out:
        json.dump({"numpy_rec": numpy_rec, "robust_best": best, "all": results, "verdict": verdict},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
