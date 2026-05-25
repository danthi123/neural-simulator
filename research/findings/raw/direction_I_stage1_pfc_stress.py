"""Direction I Stage 1 STRESS variant: sweep drive + pathway weight
to find the regime where dlpfc_wm NMDA bistability actually engages.

Per Stage 1 smoke (PFC_BISTABILITY_FAILS at default DRIVE_PA=200,
pathway weight=3.0, density=0.20): dlpfc_wm firing during stim was
~0.0012 (almost silent); NMDA never reached threshold; delay decayed
to baseline.

Hypothesis: dlpfc_wm needs stronger drive to cross NMDA threshold
during stim. The pillar n=98 work showed dlpfc_wm "pulls cortical
drive 3.09x sparser" -- it's INTRINSICALLY DAMPENED. To get NMDA
bistability working, need:
- Stronger lang_input -> dlpfc_wm drive
- Higher pathway density
- Higher pathway weight

Sweep: DRIVE_PA in {200, 500, 1000, 2000} x pathway_weight in
{3, 5, 10} x pathway_density in {0.2, 0.4, 0.6}. Find regime where
delay/baseline > 2 AND delay/stim > 0.5.

If found: Stage 2 with those parameters. If not found across
reasonable ranges: NMDA bistability genuinely doesn't engage at
this substrate scale; pivot.

NUMPY/GPU-light; ~3-5 min wall (12 cells, each ~15s sim).
"""
from __future__ import annotations
import json
import os
import sys
import time
import itertools

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion, RegionPathway
from sim.enums import NeuronType
from sim.bridge import SimulationBridge
from sim.text_embeddings import orthogonal_drive_pattern
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(_HERE, "direction_I_stage1_pfc_stress.json")
SEED = 42

N_LANG_INPUT = 256
N_DLPFC_WM = 60
STIM_STEPS = 100
DELAY_STEPS = 100
BASELINE_STEPS = 50

DRIVE_VALUES = [200, 500, 1000, 2000]
WEIGHT_VALUES = [3.0, 5.0, 10.0]
DENSITY_VALUES = [0.2, 0.4, 0.6]


def build_substrate(seed, weight_mean, density):
    regions = []
    pathways = []
    regions.append(BrainRegion(
        name="language_input", n_neurons=N_LANG_INPUT,
        exc_fraction=1.0, internal_density=0.0,
        exc_weight_mean=0.0, inh_weight_mean=0.0,
        weight_jitter=0.0, plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
    ))
    regions.append(BrainRegion(
        name="dlpfc_wm", n_neurons=N_DLPFC_WM,
        exc_fraction=0.8,
        internal_density=0.3,
        exc_weight_mean=2.0, inh_weight_mean=4.0,
        weight_jitter=0.2, plastic_internal=True,
        izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        enable_nmda=True,
    ))
    pathways.append(RegionPathway(
        from_region="language_input", to_region="dlpfc_wm",
        density=density, weight_mean=weight_mean, weight_jitter=0.5,
        plastic=False, plasticity_gate=None,
    ))
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.nmda_tau_decay = 100.0
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


def measure(bridge, dlpfc_arr, n_steps, drive=None, lang_arr=None):
    cp, _ = get_backend()
    n_total = bridge.cp_external_input_current.shape[0]
    ext = cp.zeros(n_total, dtype=cp.float32)
    spikes = cp.zeros(len(dlpfc_arr), dtype=cp.float32)
    for _ in range(n_steps):
        ext.fill(0)
        if drive is not None and lang_arr is not None:
            ext[lang_arr] = cp.asarray(drive, dtype=cp.float32)
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[dlpfc_arr]
        spikes = spikes + fired.astype(cp.float32)
    return float(cp.asnumpy(spikes).sum() / (len(dlpfc_arr) * n_steps))


def test_cell(drive_pA, weight_mean, density):
    cp, _ = get_backend()
    bridge = build_substrate(SEED, weight_mean, density)
    rm = bridge.region_manager
    lang_arr = cp.asarray(list(rm.indices("language_input")),
                            dtype=cp.int64)
    dlpfc_arr = cp.asarray(list(rm.indices("dlpfc_wm")),
                              dtype=cp.int64)
    drive = orthogonal_drive_pattern(
        cue_idx=0, n_cues=4, n_neurons=N_LANG_INPUT,
        drive_max_pA=drive_pA, sparsity=0.10)
    # settle
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    baseline = measure(bridge, dlpfc_arr, BASELINE_STEPS)
    stim = measure(bridge, dlpfc_arr, STIM_STEPS, drive, lang_arr)
    delay = measure(bridge, dlpfc_arr, DELAY_STEPS)
    return baseline, stim, delay


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction I Stage 1 PFC STRESS sweep ===", flush=True)
    t0 = time.time()

    results = []
    for drive, weight, density in itertools.product(
            DRIVE_VALUES, WEIGHT_VALUES, DENSITY_VALUES):
        try:
            baseline, stim, delay = test_cell(drive, weight, density)
            d_b = delay / (baseline + 1e-9)
            d_s = delay / (stim + 1e-9)
            passes = (d_b > 2.0 and d_s > 0.5 and delay > 0.005)
            results.append({
                "drive_pA": drive, "weight": weight,
                "density": density, "baseline": baseline,
                "stim": stim, "delay": delay,
                "delay_baseline": d_b, "delay_stim": d_s,
                "passes": passes,
            })
            mark = "PASS" if passes else "----"
            print(f"  {mark} drive={drive:>4} w={weight:>4.1f} d="
                  f"{density:>3.1f}: baseline={baseline:.4f} "
                  f"stim={stim:.4f} delay={delay:.4f} "
                  f"d/b={d_b:.2f} d/s={d_s:.2f}", flush=True)
        except Exception as e:
            print(f"  ERROR drive={drive} w={weight} d={density}: "
                  f"{e}", flush=True)
            results.append({
                "drive_pA": drive, "weight": weight,
                "density": density, "error": str(e),
            })

    total_min = (time.time() - t0) / 60
    print(f"\nWall: {total_min:.1f} min", flush=True)

    passing = [r for r in results if r.get("passes", False)]
    print(f"\n=== VERDICT ===", flush=True)
    if passing:
        # Find best cell (highest delay rate)
        best = max(passing, key=lambda r: r["delay"])
        verdict = "PFC_BISTABILITY_FOUND_IN_PARAMETER_REGIME"
        print(f"  {len(passing)} cells PASS; best cell: drive="
              f"{best['drive_pA']} weight={best['weight']} density="
              f"{best['density']} (delay={best['delay']:.4f} "
              f"d/b={best['delay_baseline']:.2f}x). Stage 2 with these"
              f" parameters JUSTIFIED.", flush=True)
    else:
        # Check for ANY persistent activity
        best_d_b = max((r.get("delay_baseline", 0) for r in results
                          if "error" not in r), default=0)
        if best_d_b > 1.5:
            verdict = "PFC_PARTIAL_PERSISTENCE_NEEDS_TUNING"
            print(f"  No cells fully PASS but best delay/baseline = "
                  f"{best_d_b:.2f}; PFC persistence partially exists"
                  f" at higher drive. Try even stronger params OR"
                  f" different neuron model (HH_PFC_PYRAMIDAL).",
                  flush=True)
        else:
            verdict = "PFC_BISTABILITY_GENUINELY_FAILS_PIVOT"
            print(f"  No regime found where PFC bistability holds "
                  f"persistent activity (best delay/baseline = "
                  f"{best_d_b:.2f}x). Substrate dynamics don't "
                  f"support NMDA-mediated frame holding at this "
                  f"scale. Pivot to Direction N (vocab scaling) OR"
                  f" Direction O (UI). Direction I closed.",
                  flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seed": SEED,
        "n_dlpfc_wm": N_DLPFC_WM,
        "drive_values": DRIVE_VALUES, "weight_values": WEIGHT_VALUES,
        "density_values": DENSITY_VALUES,
        "n_total_cells": len(DRIVE_VALUES) * len(WEIGHT_VALUES) *
                         len(DENSITY_VALUES),
        "n_passing": len(passing),
        "results": results,
        "verdict": verdict, "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
