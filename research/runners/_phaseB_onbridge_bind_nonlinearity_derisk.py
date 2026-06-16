"""CYCLE 102 — on-substrate build STEP 1 (minimal): realize the bind NONLINEARITY as REAL LIF spikes on the
SimulationBridge. Does the binding survive real spiking dynamics (threshold, refractory, temporal, finite
spike count) -- vs the numpy rate approximation?

The de-risked ON/OFF binder's bound = [relu(h), relu(-h)] where h = role@W_R + filler@W_F (numpy: a clean
analog value). This step replaces that analog ON/OFF with the firing RATES of two real LIF populations on the
bridge: drive `bind_pos` with relu(h) and `bind_neg` with relu(-h) (as external current), run the bridge, read
their per-neuron spike RATES = the spiking ON/OFF bound, then numpy-unbind + cleanup. If held-out recall ~ the
numpy binder, the real LIF f-I (in the calibrated rate band) preserves the bind -> the spiking forward is
sound (the synaptic projections are the known-easy part: population codes carry graded values at ~94%,
CYCLE 91). If it collapses, the spiking nonlinearity is the issue (localize the rate-band calibration).

This isolates the spiking nonlinearity (1-2 populations, external-current drive, rate read) -- the smallest
meaningful on-bridge step, low intricacy, before the full synaptic ON/OFF net + the local-rule training.

GATE (3 seeds): on-bridge (LIF-rate ON/OFF) held-out recall ~ the numpy binder's held-out (>> mem-floor).
Reuse-by-import (OnOffRateBinder for the trained weights + the systematicity protocol). GPU (tiny bridge).
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onbridge_bind_nonlinearity_derisk
"""
from __future__ import annotations

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
from research.runners._phaseB_spiking_bind_onoff_derisk import OnOffRateBinder  # noqa: E402

R, F, N_SPLITS, N_EPOCHS, D_H, LR = 4, 16, 2, 500, 64, 0.005   # 2 splits (GPU bridge per combo is the cost)
N_PER = 16                # neurons per bind dim (population code; the CYCLE-91 lift so SNR isn't the limiter)
DRIVE_SCALE = 400.0       # pA per unit relu(h); calibrated so rates span the ~linear LIF band
RUN_STEPS = 60            # readout window (rate = spikes / RUN_STEPS)


def build_bind_bridge(d_h, seed):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [                                # two driven populations (N_PER neurons per bind dim)
        BrainRegion(name="bind_pos", n_neurons=d_h * N_PER, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="bind_neg", n_neurons=d_h * N_PER, exc_fraction=1.0, internal_density=0.0),
        # inert anchor: NEVER driven -> stays silent -> zero influence on the ON/OFF channels. Its only job is
        # to make the wiring plan NON-EMPTY (an all-zero-synapse plan hits a latent bridge init-fallback bug).
        BrainRegion(name="_anchor", n_neurons=4, exc_fraction=1.0, internal_density=1.0),
    ]
    cfg.region_pathways = []
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    return b, np.asarray(b.region_manager.indices("bind_pos")), np.asarray(b.region_manager.indices("bind_neg"))


def lif_onoff(bridge, pos_idx, neg_idx, h, scale):
    """Drive bind_pos with relu(h), bind_neg with relu(-h); read per-neuron spike RATES = the spiking ON/OFF."""
    xp = bridge._cp if hasattr(bridge, "_cp") else None       # the proven CYCLE-95 pattern (per-region slice set)
    on = (np.repeat(np.maximum(h, 0.0) * scale, N_PER)).astype(np.float32)   # each dim's N_PER neurons get its drive
    off = (np.repeat(np.maximum(-h, 0.0) * scale, N_PER)).astype(np.float32)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[pos_idx] = xp.asarray(on) if xp is not None else on
    bridge.cp_external_input_current[neg_idx] = xp.asarray(off) if xp is not None else off
    counts = np.zeros(int(bridge.core_config.num_neurons), np.float64)
    for _ in range(RUN_STEPS):
        bridge._run_one_simulation_step()
        counts += np.asarray(to_host(bridge.cp_firing_states)).astype(np.float64)
    bridge.cp_external_input_current[:] = 0.0
    rate = counts / RUN_STEPS
    on_rate = rate[pos_idx].reshape(-1, N_PER).mean(1)        # per-dim population-averaged rate
    off_rate = rate[neg_idx].reshape(-1, N_PER).mean(1)
    return np.concatenate([on_rate, off_rate])                # [2*D_h] spiking ON/OFF bound


def run_seed(codes, seed):
    splits = make_systematicity_splits(R, F, N_SPLITS, seed)
    fillers = codes[:F]; D_in = fillers.shape[1]
    roles = make_role_codes(R, D_in, seed)
    bridge, pos_idx, neg_idx = build_bind_bridge(D_H, seed)
    onbridge_held, numpy_held, memf = [], [], []
    for split in splits:
        binder = OnOffRateBinder(D_in=D_in, D_h=D_H, lr=LR, lam=1e-4, seed=seed, read_noise=0.0)
        binder.train(split["train"], roles, fillers, n_epochs=N_EPOCHS, batch_size=max(1, len(split["train"]) // 4))
        # calibrate: scale the LIF rates so their magnitude ~ the numpy bound (relu(h)) the unbind was trained on
        h_demo = roles[split["train"][0][0]] @ binder.W_R + fillers[split["train"][0][1]] @ binder.W_F + binder.b_bind
        np_mag = float(np.mean(np.abs(np.concatenate([np.maximum(h_demo, 0), np.maximum(-h_demo, 0)]))) + 1e-9)
        lif_mag = float(np.mean(lif_onoff(bridge, pos_idx, neg_idx, h_demo, DRIVE_SCALE)) + 1e-9)
        cal = np_mag / lif_mag
        ob_ok = nn_ok = 0
        for r, f in split["held_out"]:
            h = roles[r] @ binder.W_R + fillers[f] @ binder.W_F + binder.b_bind
            bound_lif = lif_onoff(bridge, pos_idx, neg_idx, h, DRIVE_SCALE) * cal      # spiking ON/OFF (calibrated)
            est = binder._unbind(bound_lif, roles[r])
            ob_ok += int(native_argmax(est, fillers) == f)
            bound_np = np.concatenate([np.maximum(h, 0), np.maximum(-h, 0)])           # numpy reference
            nn_ok += int(native_argmax(binder._unbind(bound_np, roles[r]), fillers) == f)
        n = max(len(split["held_out"]), 1)
        onbridge_held.append(ob_ok / n); numpy_held.append(nn_ok / n)
        from research.runners.cortex_learned_binder_systematicity_probe import score_memorization_floor
        memf.append(score_memorization_floor(split["train"], split["held_out"], fillers)["held_out_acc"])
    oh, nh, mf = float(np.mean(onbridge_held)), float(np.mean(numpy_held)), float(np.mean(memf))
    print(f"  [seed {seed}] ON-BRIDGE (LIF-rate ON/OFF) held-out {oh:.3f} | numpy reference {nh:.3f} | "
          f"mem-floor {mf:.3f}", flush=True)
    return {"seed": seed, "onbridge": oh, "numpy": nh, "mem_floor": mf}


def main():
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    print(f"[on-bridge bind-nonlinearity de-risk] does the REAL LIF spiking ON/OFF (vs numpy relu) preserve the "
          f"learned bind on the substrate?", flush=True)
    rows = [run_seed(codes, s) for s in (42, 43, 44)]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    ob, nn, mf = m("onbridge"), m("numpy"), m("mem_floor")
    print(f"\n{'='*94}\n  MEAN (3 seeds): ON-BRIDGE held-out {ob:.3f} | numpy reference {nn:.3f} | mem-floor {mf:.3f}"
          f" | chance {1.0/F:.3f}", flush=True)
    print(f"{'='*94}", flush=True)
    if ob >= mf + 0.25 and ob >= 0.75 * nn:
        print(f"  GO: the real LIF spiking ON/OFF preserves the bind -- on-bridge held-out {ob:.3f} = "
              f"{ob/max(nn,1e-9):.0%} of the numpy reference ({nn:.3f}), >> mem-floor {mf:.3f}. The spiking "
              f"nonlinearity (real threshold/refractory/finite-count dynamics) carries the binding. ==> proceed "
              f"to the full synaptic ON/OFF net (projections via synapses) + the local-rule training.", flush=True)
    elif ob >= mf + 0.10:
        print(f"  PARTIAL: the LIF rates partly preserve it ({ob:.3f} vs numpy {nn:.3f}) -- tune the rate-band "
              f"calibration (DRIVE_SCALE / RUN_STEPS / a larger population per dim for SNR).", flush=True)
    else:
        print(f"  NEGATIVE: the LIF spiking nonlinearity does not preserve the bind ({ob:.3f} vs floor {mf:.3f}) -- "
              f"the rate code at D_h={D_H} (1 neuron/dim) hits the rate-code wall; needs a population per dim "
              f"(the CYCLE-91 lift).", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"onbridge": ob, "numpy": nn, "mem_floor": mf, "chance": 1.0 / F, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_onbridge_bind_nonlinearity.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
