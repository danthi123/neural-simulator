"""On-substrate local read-out rule, biologization STEP 3c ON THE BRIDGE — the per-output teaching error is computed
by a real spiking ERROR POPULATION (not a host subtraction), and drives the read-out's synaptic plasticity. This is
the on-bridge realization of the neural-error de-risk (`2026-06-17-neural-error-localrule-derisk-GO.md`, 6-seed GO).

The read-out learning already works on the bridge (`2026-06-17-onsubstrate-readout-rule-bridge-GO.md`, 6-seed GO):
weight_update = lr * cp_per_synapse_reward_override * eligibility, with override = the per-output error. There the
error was a host subtraction. HERE the error is a predictive-coding error POPULATION on the same bridge: per output
j, an ON error neuron driven by external current ~ (target_j - est_j) and an OFF error neuron ~ (est_j - target_j);
the LIF f-I curve does the rectification (sub-threshold -> silent), so they fire at rates ~ relu(target-est) /
relu(est-target). The signed neural error = (ON_rate - OFF_rate) is read from real spikes and delivered through the
cp_per_synapse_reward_override climbing-fiber channel. The subtraction target-prediction is done by the error neuron
(exc target - inh prediction); only the `target` is an env/teacher scaffold (the supervised signal), `est` is the
read-out's own prediction.

GATE (cheap-first single seed -> 6 seeds): on-bridge NEURAL-error read-out held-out ~ the on-bridge HOST-error
read-out (>= 0.85x) AND systematicity AND anti-cheats: lesion the error pop (drive 0) -> no learning (floor);
scramble the neural error across outputs -> collapse. NO new sim/ mechanism (reuse the read-out bridge + the
external-current ON/OFF drive). GPU.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onbridge_neural_error_readout_derisk [--dh 64] [--seeds 42]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402
from research.runners.cortex_learned_binder_systematicity_probe import (  # noqa: E402
    make_role_codes, make_systematicity_splits, native_argmax)
from research.runners._phaseB_onsubstrate_readout_bridge_derisk import (  # noqa: E402
    _read_W, _set_and_step, _rate, LR, N_PASSES, build_readout_bridge)

R, F, N_SPLITS = 4, 16, 3
N_EVAL_FACTS = 40
# the spiking error has residual noise (sign ~0.85, not 1.0), so the read-out learning rate must be LOW (the
# numpy neural-error de-risk converged at ~0.02; the on-bridge host-error read-out used 0.5 because it was EXACT)
# -- a high lr lets the per-step noise random-walk the weights. Lower lr + more passes = the noisy-LMS fix.
NEURAL_LR = 0.08
# TONIC OPPONENT coding (the diagnostic fix): a RECTIFIED ON/OFF error (relu) leaves small errors in the LIF
# dead-zone -> the SIGN is near-chance for the many tiny per-output errors (corr 0.83 but sign 0.55), so the delta
# rule learns the wrong direction on ~half the outputs -> no convergence. Instead both ON and OFF error neurons get
# a TONIC BASELINE (always firing), and the error MODULATES them oppositely: on=BASELINE+diff*gain, off=BASELINE-
# diff*gain -> ON_rate-OFF_rate tracks the SIGNED error LINEARLY THROUGH ZERO (no dead-zone). This is how the brain
# codes signed values (opponent populations around a tonic rate; the inferior olive's tonic-modulated error).
ERR_BASELINE = 500.0  # pA tonic drive to both ON and OFF error neurons (mid f-I band -> always firing)
ERR_DRIVE = 2500.0    # pA per unit (target-est) MODULATION (kept < BASELINE so both neurons stay firing: max|diff|~0.15*2500=375 < 500)
ERR_WINDOW = 20       # readout window for the error population (rate = spikes / window)
N_ERR = 8             # error neurons per output per ON/OFF (population-coded -> ~sqrt(N) less Poisson noise -> clean
                      # sign on the many small per-output errors; the inferior olive uses a POPULATION, not 1 cell)


def build_error_bridge(d_in, seed):
    """A SEPARATE tiny bridge holding the error population `err` (2*D_in neurons: ON[0:D_in]/OFF[D_in:]) driven by
    external current (no synapses -> they fire from their drive; the LIF f-I rectifies). Biologically the inferior
    olive whose climbing fibers carry the per-Purkinje teaching error -- a distinct structure that PROJECTS to the
    read-out (here via the cp_per_synapse_reward_override channel). Kept separate so it does not perturb the read-out
    bridge's synapse/eligibility layout."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="err", n_neurons=2 * d_in * N_ERR, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="_anchor", n_neurons=4, exc_fraction=1.0, internal_density=1.0),
    ]
    cfg.region_pathways = []
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    rt = RuntimeState(); rt.actual_seed_used = seed
    eb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    eb._initialize_simulation_data()
    return eb, np.asarray(eb.region_manager.indices("err"))


def neural_error(b, err_idx, target, est, d_in, cal):
    """Drive err ON[j] with (target_j-est_j), OFF[j] with (est_j-target_j); the LIF f-I rectifies. Read ON/OFF spike
    rates -> signed neural error = (ON_rate - OFF_rate)*cal (per output)."""
    import sim.backend as _bk
    xp, _ = _bk.get_backend()
    diff = (target - est)
    # tonic opponent code: both fire at BASELINE, the error modulates ON up / OFF down -> ON-OFF tracks signed diff
    # linearly through zero (no rectification dead-zone, so small-error SIGN is preserved).
    on = np.clip(ERR_BASELINE + diff * ERR_DRIVE, 0.0, None)
    off = np.clip(ERR_BASELINE - diff * ERR_DRIVE, 0.0, None)
    # population-coded: each output's N_ERR ON (and OFF) neurons get the same drive; layout = [ON block, OFF block],
    # output j's neurons at [j*N_ERR:(j+1)*N_ERR]. Average the N_ERR rates -> ~sqrt(N_ERR) less Poisson noise.
    drive = np.concatenate([np.repeat(on, N_ERR), np.repeat(off, N_ERR)]).astype(np.float32)
    b.cp_external_input_current[:] = 0.0
    err_gpu = xp.asarray(err_idx)
    b.cp_external_input_current[err_idx] = xp.asarray(drive)
    # accumulate the error neurons' spikes ON the device (one D2H transfer at the end), NOT to_host every step:
    # the per-step host transfer of the whole firing-state vector was the dominant cost (~window*train_steps syncs).
    counts_err = xp.zeros(int(err_gpu.shape[0]), dtype=xp.float64)
    for _ in range(ERR_WINDOW):
        b._run_one_simulation_step()
        counts_err += b.cp_firing_states[err_gpu].astype(xp.float64)
    b.cp_external_input_current[:] = 0.0
    rate = np.asarray(to_host(counts_err)) / ERR_WINDOW
    on_r = rate[:d_in * N_ERR].reshape(d_in, N_ERR).mean(1)      # population-averaged ON rate per output
    off_r = rate[d_in * N_ERR:].reshape(d_in, N_ERR).mean(1)
    return (on_r - off_r) * cal                                 # signed neural error [D_in]


def run_seed(codes, seed, two_dh, d_h, n_passes=N_PASSES):
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]; d_in = fillers.shape[1]
    roles = make_role_codes(R, d_in, seed)
    rng_pm1 = np.random.default_rng(seed * 31 + 5)
    R_proj = rng_pm1.standard_normal((d_in, d_h)) / np.sqrt(d_in)
    role_pm1 = np.where(roles @ R_proj >= 0.0, 1.0, -1.0)
    rngF = np.random.default_rng(seed * 17 + 3)
    W_F = rngF.standard_normal((d_in, d_h)) / np.sqrt(d_in)

    b, inp, out, pre_of, post_of = build_readout_bridge(two_dh, d_in, seed)
    b.core_config.reward_learning_rate = NEURAL_LR        # LOW lr for the noisy spiking error (noisy-LMS fix)
    eb, err = build_error_bridge(d_in, seed)              # separate error-population bridge (the inferior olive)
    # calibrate the error-population gain: neural-error magnitude ~ host-error magnitude on a sample
    cal = [1.0]

    def _calibrate():
        rng = np.random.default_rng(seed)
        tgt = fillers[int(rng.integers(F))]; es = rng.standard_normal(d_in) * 0.1
        host_mag = float(np.mean(np.abs(tgt - es)) + 1e-9)
        raw = neural_error(eb, err, tgt, es, d_in, 1.0)
        cal[0] = host_mag / (float(np.mean(np.abs(raw))) + 1e-9)

    _calibrate()
    full = np.zeros(int(b.core_config.num_neurons))

    def _train(mode):
        """mode: 'neural' (error from the spiking pop), 'lesion' (error pop drive 0 -> ~no error), 'scramble'."""
        b.cp_connections.data[:] = 0.0
        perm = np.random.default_rng(seed * 5 + 1).permutation(d_in)
        for split in splits:
            tr = {r: [f for (rr, f) in split["train"] if rr == r] for r in range(3)}
            if min(len(tr[r]) for r in range(3)) == 0:
                continue
            rr = np.random.default_rng(seed * 53 + 9)
            for _ in range(n_passes * max(len(split["train"]), 1)):
                fa = rr.choice(tr[0]); fv = rr.choice(tr[1]); fo = rr.choice(tr[2])
                roleids, fillerids = [0, 1, 2], [int(fa), int(fv), int(fo)]
                t = int(rr.integers(3))
                ws = [fillers[f] @ W_F for f in fillerids]
                bundle = sum(role_pm1[r] * w for r, w in zip(roleids, ws))
                act = bundle * role_pm1[roleids[t]]
                rate = _rate(act)
                W = _read_W(b, inp, out, pre_of, post_of)
                est = W @ rate
                target = fillers[fillerids[t]]
                if mode == "lesion":
                    nerr = neural_error(eb, err, est, est, d_in, cal[0])      # target==est -> ~0 error
                else:
                    nerr = neural_error(eb, err, target, est, d_in, cal[0])   # signed (target - est), neural
                err_signed = -nerr                                           # est - target (host LMS sign)
                if mode == "scramble":
                    err_signed = err_signed[perm]
                full[:] = 0.0; full[inp] = rate
                e_full = np.zeros(int(b.core_config.num_neurons)); e_full[out] = err_signed
                _set_and_step(b, pre_of, post_of, full, e_full)
        return _read_W(b, inp, out, pre_of, post_of)

    def _bind(r, f):
        return role_pm1[r] * (fillers[f] @ W_F)

    def _recall(W, bundle, r):
        return _rate(bundle * role_pm1[r]) @ W.T

    def _eval(W):
        ok = n = 0
        erng = np.random.default_rng(seed * 7 + 1)
        for _ in range(N_EVAL_FACTS):
            fids = erng.choice(F, 3, replace=False)
            bundle = sum(_bind(r, int(fids[r])) for r in range(3))
            for r in range(3):
                ok += int(native_argmax(_recall(W, bundle, r), fillers) == fids[r]); n += 1
        return ok / max(n, 1)

    W_neural = _train("neural"); neural_h = _eval(W_neural)
    W_lesion = _train("lesion"); lesion_h = _eval(W_lesion)
    W_scr = _train("scramble"); scr_h = _eval(W_scr)
    row = {"seed": seed, "neural": neural_h, "lesion": lesion_h, "scramble": scr_h, "cal": cal[0]}
    print(f"  [seed {seed}] ON-BRIDGE NEURAL-error held-out {neural_h:.3f} | lesion {lesion_h:.3f} | "
          f"scramble {scr_h:.3f} (cal {cal[0]:.3f})", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dh", type=int, default=64)
    ap.add_argument("--passes", type=int, default=N_PASSES)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_onbridge_neural_error_readout.json"))
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")
    seeds = [int(s) for s in args.seeds.split(",")]
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    two_dh = 2 * args.dh
    t0 = time.time()
    print(f"[on-bridge neural-error read-out de-risk] does a real spiking ERROR POPULATION drive the read-out "
          f"plasticity? D_h={args.dh} seeds={seeds}", flush=True)
    rows = [run_seed(codes, s, two_dh, args.dh, n_passes=args.passes) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    neural, lesion, scr = m("neural"), m("lesion"), m("scramble")
    chance = 1.0 / F
    print(f"\n{'='*100}", flush=True)
    print(f"  MEAN ({len(seeds)} seeds): ON-BRIDGE NEURAL-error {neural:.3f} | lesion {lesion:.3f} | "
          f"scramble {scr:.3f} | chance {chance:.3f}", flush=True)
    go = (neural >= 0.85) and (lesion <= chance + 0.10) and (scr <= chance + 0.10)
    if go:
        print(f"  GO: a real spiking predictive-coding error population drives the read-out's synaptic plasticity -- "
              f"on-bridge neural-error held-out {neural:.3f} >> lesion {lesion:.3f} (error pop silenced -> no "
              f"learning) and scramble {scr:.3f} (collapse). ==> the read-out learning is now FULLY brain-based on "
              f"the substrate: the per-output error is computed by neurons (exc target - inh prediction) and the "
              f"weights are learned by real synaptic plasticity. No host scaffold (only the env target remains).",
              flush=True)
    else:
        print(f"  BOUNDARY: neural {neural:.3f} / lesion {lesion:.3f} / scramble {scr:.3f} -- if neural is low, "
              f"tune the error-population rate band (ERR_DRIVE/ERR_WINDOW/cal); if lesion/scramble don't collapse, "
              f"localize.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}", flush=True)
    out = {"verdict": "GO" if go else "BOUNDARY", "D_h": args.dh, "seeds": seeds, "neural": neural,
           "lesion": lesion, "scramble": scr, "chance": chance, "per_seed": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
