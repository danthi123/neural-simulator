"""Direction I Stage 1 DIRECT INJECT variant: bypass lang_input
routing; directly drive dlpfc_wm with strong current to isolate the
NMDA bistability question from input-routing failures.

Per stress sweep: dlpfc_wm barely fires (~0.0003 stim rate) regardless
of lang_input drive/weight/density. The bottleneck is the sparse +
random-init pathway not reaching dlpfc_wm with enough drive.

Direct test:
1. Build dlpfc_wm-only substrate (no lang_input routing)
2. Inject strong current directly on dlpfc_wm during stim window
3. Stop injection; measure delay-period firing
4. Use exact g11_bg_runner config: enable_pfc_nmda=True ->
   cfg.nmda_ratio=0.5 (Wang 2002 calibration), pathway-style NMDA

If dlpfc_wm CAN sustain persistent activity when DIRECTLY driven,
the substrate-level NMDA bistability works -- the problem is just
routing. If it CANNOT even with direct injection, NMDA bistability
genuinely doesn't engage at this substrate scale.

NUMPY/GPU; ~2 min wall.
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion
from sim.enums import NeuronType
from sim.bridge import SimulationBridge
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(_HERE, "direction_I_stage1_direct_inject.json")
SEED = 42
N_DLPFC_WM = 60
BASELINE_STEPS = 50
STIM_STEPS = 100
DELAY_STEPS = 200  # longer delay to look for fade-out
INJECTION_PA_VALUES = [500, 1000, 2000, 5000]


def build_pfc_only(seed):
    """Just dlpfc_wm region with Wang 2002 NMDA settings."""
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [BrainRegion(
        name="dlpfc_wm", n_neurons=N_DLPFC_WM,
        exc_fraction=0.8,
        internal_density=0.3,
        exc_weight_mean=2.0, inh_weight_mean=4.0,
        weight_jitter=0.2,
        plastic_internal=True,
        izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        enable_nmda=True,
    )]
    cfg.region_pathways = []
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5  # Wang 2002 PFC calibration (was missing!)
    cfg.fast_spike_reset = True

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def measure(bridge, dlpfc_arr, n_steps, inject_pA=0.0):
    cp, _ = get_backend()
    n_total = bridge.cp_external_input_current.shape[0]
    ext = cp.zeros(n_total, dtype=cp.float32)
    spikes = cp.zeros(len(dlpfc_arr), dtype=cp.float32)
    for _ in range(n_steps):
        ext.fill(0)
        if inject_pA > 0:
            ext[dlpfc_arr] = inject_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[dlpfc_arr]
        spikes = spikes + fired.astype(cp.float32)
    return float(cp.asnumpy(spikes).sum() / (len(dlpfc_arr) * n_steps))


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction I Stage 1 DIRECT INJECT ===", flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Tests if dlpfc_wm NMDA bistability holds when DIRECTLY",
          flush=True)
    print(f"  driven (bypasses input-routing failure).", flush=True)
    print(f"  Wang 2002 PFC calibration: nmda_ratio=0.5 (was missing)",
          flush=True)

    t0 = time.time()
    bridge = build_pfc_only(SEED)
    cp, _ = get_backend()
    rm = bridge.region_manager
    dlpfc_arr = cp.asarray(list(rm.indices("dlpfc_wm")),
                              dtype=cp.int64)
    print(f"  built dlpfc_wm-only substrate "
          f"({len(dlpfc_arr)} neurons; nmda_ratio=0.5)", flush=True)

    results = []
    for inj_pA in INJECTION_PA_VALUES:
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(30):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        baseline = measure(bridge, dlpfc_arr, BASELINE_STEPS, inject_pA=0)
        stim = measure(bridge, dlpfc_arr, STIM_STEPS, inject_pA=inj_pA)
        delay = measure(bridge, dlpfc_arr, DELAY_STEPS, inject_pA=0)
        d_b = delay / (baseline + 1e-9)
        d_s = delay / (stim + 1e-9)
        passes = (delay > 0.005 and d_b > 2.0 and d_s > 0.5)
        results.append({
            "inject_pA": inj_pA, "baseline": baseline,
            "stim": stim, "delay": delay,
            "d_b": d_b, "d_s": d_s, "passes": passes,
        })
        mark = "PASS" if passes else "----"
        print(f"  {mark} inject={inj_pA:>5}: baseline={baseline:.4f}"
              f" stim={stim:.4f} delay={delay:.4f} "
              f"d/b={d_b:.2f} d/s={d_s:.2f}", flush=True)

    total_min = (time.time() - t0) / 60
    print(f"\nWall: {total_min:.1f} min", flush=True)

    passing = [r for r in results if r["passes"]]
    print(f"\n=== VERDICT ===", flush=True)
    if passing:
        best = max(passing, key=lambda r: r["delay"])
        verdict = "PFC_BISTABILITY_HOLDS_VIA_DIRECT_INJECTION"
        print(f"  PFC NMDA bistability HOLDS via direct injection"
              f" (best inject={best['inject_pA']}, delay="
              f"{best['delay']:.4f}). The substrate-level dynamics"
              f" support persistent activity. Original smoke failed"
              f" because of INPUT ROUTING (lang_input -> dlpfc_wm"
              f" pathway sparse + random init; doesn't deliver enough"
              f" drive). Stage 2 must include trained input pathway.",
              flush=True)
    else:
        # Find best delay/stim ratio
        max_d = max(r["delay"] for r in results)
        max_d_s = max(r["d_s"] for r in results)
        if max_d > 0.005:
            verdict = "PFC_PARTIAL_BISTABILITY_NEEDS_PARAM_REFINEMENT"
            print(f"  Partial bistability (max delay={max_d:.4f},"
                  f" d/s={max_d_s:.2f}). NMDA fires during delay"
                  f" but not at clean bistable level. Try stronger"
                  f" recurrent density (0.5+) or different neuron"
                  f" model.", flush=True)
        else:
            verdict = "PFC_BISTABILITY_GENUINELY_FAILS_PIVOT"
            print(f"  PFC NMDA bistability genuinely doesn't engage"
                  f" even with direct injection up to 5000 pA. The"
                  f" substrate's IZH2007_HIPPO_PYRAMIDAL + nmda_ratio"
                  f"=0.5 + 60 neurons + density 0.3 doesn't produce"
                  f" the Wang 2002 persistent attractor at this"
                  f" scale. Direction I closed; pivot to N or O.",
                  flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seed": SEED,
        "n_dlpfc_wm": N_DLPFC_WM, "nmda_ratio": 0.5,
        "injection_values": INJECTION_PA_VALUES,
        "results": results, "verdict": verdict,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
