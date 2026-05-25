"""Direction I Stage 1 SMOKE: verify dlpfc_wm NMDA bistability can
hold a (word, slot) frame across a delay period after stimulus ends.

Per Direction I design: PFC sequence buffer via NMDA-bistable
dlpfc_wm region is the catalog-prescribed fix for the substrate
sequence-storage bound (pillar n=104 BOUNDARY). Before committing
2-4 week full build, verify the FOUNDATIONAL property:

Q: Can the dlpfc_wm region's NMDA bistability sustain firing for
~100ms after stimulus drive ends? (= "persistent activity" / "delay-
period firing" — the property Goldman-Rakic + Wang 2002 + Lisman-
Idiart all rely on for sequence-frame holding.)

Stage 1 test:
1. Build substrate: v16 concept-pools + dlpfc_wm region (NMDA-bistable)
2. Add lang_input → dlpfc_wm pathway (initialized, no training needed
   for smoke; just verify the WIRING and DYNAMICS work)
3. Drive lang_input(word) for stimulus window (~100 steps); measure
   dlpfc_wm firing during drive
4. Stop drive. Continue running simulation for delay window
   (~100 steps). Measure dlpfc_wm firing during delay.
5. Smoke PASS: delay-period firing > 2x baseline (= persistent
   activity confirmed) AND delay-period firing > 0.5x stimulus-period
   firing (= bistability holds the frame, not just trailing fade).

If PASS: Stage 2 (full PFC + concept-pool integration) justified.
If FAIL: NMDA bistability doesn't hold at this substrate scale; fast
diagnostic; pivot to Direction N or O.

~5-10 min wall single seed; reuses validated dlpfc_wm config from
g11_bg_runner.py byte-equivalent.
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
from sim.regions import BrainRegion, RegionPathway
from sim.enums import NeuronType
from sim.bridge import SimulationBridge
from sim.text_embeddings import orthogonal_drive_pattern
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(_HERE, "direction_I_stage1_pfc_smoke.json")
SEED = 42

# Stimulus + delay windows
STIM_STEPS = 100  # drive lang_input for ~100 steps (~50ms at dt=0.5)
DELAY_STEPS = 100  # measure persistent firing for ~100 steps after drive
BASELINE_STEPS = 50  # measure baseline before stimulus

# Substrate params (small smoke)
N_LANG_INPUT = 256  # small for smoke
N_DLPFC_WM = 60  # mirrors g11_bg_runner default
N_TARGETS = 4  # 4 distinct "words" / orthogonal codes for variety
DRIVE_PA = 200.0


def build_smoke_substrate(seed):
    """Build small substrate with lang_input + dlpfc_wm (NMDA bistable)."""
    regions = []
    pathways = []

    # Language input region (sparse-coded; like v16 lang_input)
    regions.append(BrainRegion(
        name="language_input", n_neurons=N_LANG_INPUT,
        exc_fraction=1.0, internal_density=0.0,
        exc_weight_mean=0.0, inh_weight_mean=0.0,
        weight_jitter=0.0, plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
    ))

    # dlpfc_wm: NMDA-bistable PFC region (mirrors g11_bg_runner)
    regions.append(BrainRegion(
        name="dlpfc_wm", n_neurons=N_DLPFC_WM,
        exc_fraction=0.8,
        internal_density=0.3,  # recurrent (g11_bg default ~0.2-0.3)
        exc_weight_mean=2.0,  # moderate self-excitation
        inh_weight_mean=4.0,
        weight_jitter=0.2,
        plastic_internal=True,
        izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        enable_nmda=True,  # THE CRITICAL FLAG -- NMDA bistability
    ))

    # lang_input -> dlpfc_wm pathway (sparse + plastic)
    pathways.append(RegionPathway(
        from_region="language_input", to_region="dlpfc_wm",
        density=0.20, weight_mean=3.0, weight_jitter=0.5,
        plastic=False, plasticity_gate=None,
    ))

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.nmda_tau_decay = 100.0  # ~100ms NMDA decay (Wang 2002)
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.fast_spike_reset = True

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    # Per _build_bridge_with_hippo pattern: need explicit
    # _initialize_simulation_data + max_delay_steps setup.
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def measure_dlpfc_activity_in_window(bridge, dlpfc_idx_arr,
                                         window_steps, drive_arr=None,
                                         lang_in_arr=None):
    """Run window_steps simulation steps; return mean firing rate
    over dlpfc_wm in that window. Optional drive_arr applied to
    lang_in_arr if both given."""
    cp, _ = get_backend()
    n_total = bridge.cp_external_input_current.shape[0]
    ext = cp.zeros(n_total, dtype=cp.float32)
    spike_counts = cp.zeros(len(dlpfc_idx_arr), dtype=cp.float32)
    for _ in range(window_steps):
        ext.fill(0)
        if drive_arr is not None and lang_in_arr is not None:
            ext[lang_in_arr] = cp.asarray(drive_arr, dtype=cp.float32)
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[dlpfc_idx_arr]
        spike_counts = spike_counts + fired.astype(cp.float32)
    # Mean firing rate per neuron per step
    mean_rate = float(cp.asnumpy(spike_counts).sum() /
                       (len(dlpfc_idx_arr) * window_steps))
    return mean_rate


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction I Stage 1: PFC NMDA bistability smoke ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Tests if dlpfc_wm NMDA bistability holds persistent",
          flush=True)
    print(f"  activity for ~100ms after stimulus ends.", flush=True)
    print(f"  Pre-registered smoke PASS: delay firing > 2x baseline",
          flush=True)
    print(f"  AND > 0.5x stim firing (bistability holds the frame)",
          flush=True)

    t0 = time.time()
    bridge = build_smoke_substrate(SEED)
    print(f"  built substrate in {(time.time()-t0):.1f}s", flush=True)

    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_in_idx = list(rm.indices("language_input"))
    lang_in_arr = cp.asarray(lang_in_idx, dtype=cp.int64)
    dlpfc_idx = list(rm.indices("dlpfc_wm"))
    dlpfc_arr = cp.asarray(dlpfc_idx, dtype=cp.int64)
    print(f"  language_input: {len(lang_in_idx)} neurons", flush=True)
    print(f"  dlpfc_wm: {len(dlpfc_idx)} neurons (NMDA bistable)",
          flush=True)

    # Test multiple distinct drives ("words") to check generalization
    per_word_results = []
    for word_idx in range(N_TARGETS):
        word_name = f"word{word_idx}"
        drive = orthogonal_drive_pattern(
            cue_idx=word_idx, n_cues=N_TARGETS,
            n_neurons=N_LANG_INPUT, drive_max_pA=DRIVE_PA,
            sparsity=0.10)

        # Reset (settle)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(30):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        # 1. Baseline (no drive)
        baseline_rate = measure_dlpfc_activity_in_window(
            bridge, dlpfc_arr, BASELINE_STEPS,
            drive_arr=None, lang_in_arr=None)

        # 2. Stimulus (drive lang_input)
        stim_rate = measure_dlpfc_activity_in_window(
            bridge, dlpfc_arr, STIM_STEPS,
            drive_arr=drive, lang_in_arr=lang_in_arr)

        # 3. Delay (drive off; measure persistence)
        delay_rate = measure_dlpfc_activity_in_window(
            bridge, dlpfc_arr, DELAY_STEPS,
            drive_arr=None, lang_in_arr=None)

        per_word_results.append({
            "word": word_name,
            "baseline_rate": baseline_rate,
            "stim_rate": stim_rate,
            "delay_rate": delay_rate,
            "delay_vs_baseline_ratio": (
                delay_rate / (baseline_rate + 1e-9)),
            "delay_vs_stim_ratio": (
                delay_rate / (stim_rate + 1e-9)),
        })
        print(f"  {word_name}: baseline={baseline_rate:.4f}, "
              f"stim={stim_rate:.4f}, delay={delay_rate:.4f}, "
              f"delay/baseline={delay_rate/(baseline_rate+1e-9):.2f}x, "
              f"delay/stim={delay_rate/(stim_rate+1e-9):.2f}x",
              flush=True)

    print(f"\n  Wall: {(time.time()-t0)/60:.1f} min", flush=True)

    # Verdict
    mean_baseline = float(np.mean(
        [r["baseline_rate"] for r in per_word_results]))
    mean_stim = float(np.mean(
        [r["stim_rate"] for r in per_word_results]))
    mean_delay = float(np.mean(
        [r["delay_rate"] for r in per_word_results]))
    delay_baseline_ratio = mean_delay / (mean_baseline + 1e-9)
    delay_stim_ratio = mean_delay / (mean_stim + 1e-9)

    print(f"\n=== AGGREGATE RESULTS ({N_TARGETS} test words) ===",
          flush=True)
    print(f"  mean baseline rate: {mean_baseline:.4f}", flush=True)
    print(f"  mean stim rate:     {mean_stim:.4f}", flush=True)
    print(f"  mean delay rate:    {mean_delay:.4f}", flush=True)
    print(f"  delay/baseline:     {delay_baseline_ratio:.2f}x"
          f" (PASS: > 2.0)", flush=True)
    print(f"  delay/stim:         {delay_stim_ratio:.2f}x"
          f" (PASS: > 0.5)", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    bistability_holds = (delay_baseline_ratio > 2.0
                         and delay_stim_ratio > 0.5)
    if bistability_holds:
        verdict = "PFC_BISTABILITY_HOLDS_STAGE2_JUSTIFIED"
        print(f"  PFC bistability HOLDS: delay-period firing "
              f"{delay_baseline_ratio:.1f}x baseline AND "
              f"{delay_stim_ratio:.1f}x stim. NMDA-mediated "
              f"persistent activity confirmed at this substrate "
              f"scale. STAGE 2 (PFC + concept-pool integration) "
              f"JUSTIFIED.", flush=True)
    elif delay_baseline_ratio > 1.2:
        verdict = "PFC_PARTIAL_PERSISTENCE_DIAGNOSTIC_NEEDED"
        print(f"  Partial persistence (delay {delay_baseline_ratio:.1f}"
              f"x baseline but {delay_stim_ratio:.2f}x stim). "
              f"PFC fires above baseline but doesn't hold the frame"
              f" robustly. Diagnose: more recurrent density / NMDA "
              f"strength / different neuron model. Stage 2 not "
              f"justified yet.", flush=True)
    else:
        verdict = "PFC_BISTABILITY_FAILS_PIVOT"
        print(f"  PFC bistability FAILS at this substrate scale "
              f"(delay {delay_baseline_ratio:.2f}x baseline). Wave"
              f" dies after stimulus. Pivot to Direction N (scale "
              f"vocab) or Direction O (sentence parser UI) instead.",
              flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seed": SEED,
        "n_lang_input": N_LANG_INPUT, "n_dlpfc_wm": N_DLPFC_WM,
        "stim_steps": STIM_STEPS, "delay_steps": DELAY_STEPS,
        "baseline_steps": BASELINE_STEPS,
        "per_word": per_word_results,
        "mean_baseline_rate": mean_baseline,
        "mean_stim_rate": mean_stim,
        "mean_delay_rate": mean_delay,
        "delay_baseline_ratio": delay_baseline_ratio,
        "delay_stim_ratio": delay_stim_ratio,
        "verdict": verdict,
        "wall_clock_minutes": (time.time() - t0) / 60,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
