"""Direction I Stage 1 HH variant: try HH_PFC_PYRAMIDAL (full
Hodgkin-Huxley biophysics) instead of Izhikevich approximation.

Per direct-inject test: Izhikevich (IZH2007_HIPPO_PYRAMIDAL) +
nmda_ratio=0.5 + 60 neurons + density 0.3 produces NO persistent
activity. HH biophysics has FULL NMDA gating dynamics (voltage-
dependent Mg2+ block + NR2A/NR2B subtype-specific tau); Wang 2002's
attractor was originally formulated in HH-class models.

If HH PASSes: NMDA bistability works with HH biophysics. Stage 2 with
HH neurons.
If HH ALSO fails: NMDA bistability genuinely doesn't engage at this
substrate scale (60 neurons) regardless of biophysics. Direction I
closed; pivot.

NUMPY/GPU; ~2-3 min wall.

WARNING: HH model requires dt=0.05ms (vs Izh dt=0.5ms); 10x more sim
steps per second of biological time. The bridge auto-adjusts dt per
the CLAUDE.md note: "When switching to Hodgkin-Huxley model, dt is
automatically reduced to 0.05ms for numerical stability."
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
from sim.enums import NeuronType, NeuronModel
from sim.bridge import SimulationBridge
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(_HERE, "direction_I_stage1_hh_inject.json")
SEED = 42
N_DLPFC_WM = 60
# Account for 10x more steps per ms with HH dt=0.05
BASELINE_STEPS = 500  # 25 ms
STIM_STEPS = 1000     # 50 ms
DELAY_STEPS = 2000    # 100 ms
INJECTION_PA_VALUES = [500, 1000, 2000, 5000]


def build_pfc_hh(seed):
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [BrainRegion(
        name="dlpfc_wm", n_neurons=N_DLPFC_WM,
        exc_fraction=0.8,
        internal_density=0.3,
        exc_weight_mean=2.0, inh_weight_mean=4.0,
        weight_jitter=0.2,
        plastic_internal=True,
        # HH PFC pyramidal — full biophysics
        izh_neuron_type=None,
        enable_nmda=True,
    )]
    cfg.region_pathways = []
    cfg.neuron_model_type = NeuronModel.HODGKIN_HUXLEY.name
    cfg.default_neuron_type_hh = NeuronType.HH_PFC_PYRAMIDAL.name
    cfg.dt_ms = 0.05  # HH stability
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.fast_spike_reset = False  # HH doesn't use fast_spike_reset

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
    print(f"=== Direction I Stage 1 HH (Hodgkin-Huxley) inject ===",
          flush=True)
    print(f"  HH_PFC_PYRAMIDAL biophysics + nmda_ratio=0.5",
          flush=True)
    print(f"  dt=0.05ms (HH stability)", flush=True)

    t0 = time.time()
    try:
        bridge = build_pfc_hh(SEED)
    except Exception as e:
        print(f"  [FATAL] HH build failed: {e}", flush=True)
        return 1

    cp, _ = get_backend()
    rm = bridge.region_manager
    dlpfc_arr = cp.asarray(list(rm.indices("dlpfc_wm")),
                              dtype=cp.int64)
    print(f"  built HH dlpfc_wm-only substrate "
          f"({len(dlpfc_arr)} neurons)", flush=True)

    results = []
    for inj_pA in INJECTION_PA_VALUES:
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(300):  # settle (15ms)
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        baseline = measure(bridge, dlpfc_arr, BASELINE_STEPS, inject_pA=0)
        stim = measure(bridge, dlpfc_arr, STIM_STEPS, inject_pA=inj_pA)
        delay = measure(bridge, dlpfc_arr, DELAY_STEPS, inject_pA=0)
        d_b = delay / (baseline + 1e-9)
        d_s = delay / (stim + 1e-9)
        passes = (delay > 0.0005 and d_b > 2.0 and d_s > 0.5)
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
        verdict = "HH_PFC_BISTABILITY_HOLDS_USE_HH_FOR_STAGE2"
        print(f"  HH PFC NMDA bistability HOLDS (best inject="
              f"{best['inject_pA']}, delay={best['delay']:.4f}). "
              f"Stage 2 with HH neurons + lang_input routing.",
              flush=True)
    else:
        max_d = max(r["delay"] for r in results)
        if max_d > 0.001:
            verdict = "HH_PFC_PARTIAL_NOT_BISTABLE"
            print(f"  HH PFC fires during stim but doesn't sustain"
                  f" (max delay={max_d:.4f}). Persistent activity"
                  f" partial; not full attractor. Pivot.", flush=True)
        else:
            verdict = "HH_PFC_BISTABILITY_GENUINELY_FAILS_DIRECTION_I_CLOSED"
            print(f"  HH PFC NMDA bistability genuinely fails at"
                  f" this substrate scale (60 neurons). Both Izh"
                  f" and HH neuron models fail. DIRECTION I CLOSED;"
                  f" pivot to N or O.", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seed": SEED,
        "n_dlpfc_wm": N_DLPFC_WM, "nmda_ratio": 0.5,
        "neuron_type": "HH_PFC_PYRAMIDAL",
        "dt_ms": 0.05,
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
