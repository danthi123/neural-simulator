"""Diagnostic: does the BG cascade have a default cortex_N bias even
with NO training and SYMMETRIC inputs?

If yes: language_input → cortex_X training fails because the cascade
itself is biased; we need stronger language drive or pre-balance.
If no: training failure is in STDP / regime, not cascade.
"""
from __future__ import annotations

import sys

import numpy as np


def main():
    import cupy as cp
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from research.runners.g11_bg_runner import build_bg_brain_regions
    from sim.text_embeddings import vocab_to_drive_pattern

    regions, pathways = build_bg_brain_regions(
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_pfc=True, pfc_enable_nmda=True,
        enable_visual_cortex=True, enable_text_io=True,
    )
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = 0.5
    cfg.seed = 42
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.enable_structural_plasticity = False

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    cortex_idx = {
        a: cp.asarray(list(bridge.region_manager.indices(f"cortex_{a}")),
                      dtype=cp.int64)
        for a in ["N", "E", "S", "W"]
    }
    lang_input_idx = cp.asarray(
        list(bridge.region_manager.indices("language_input")), dtype=cp.int64,
    )

    # Test 1: No input at all — what's the baseline cortex_X firing?
    print("=" * 60)
    print("TEST 1: SPONTANEOUS (no input) — does cascade have a default bias?")
    print("=" * 60)
    bridge.cp_external_input_current[:] = 0.0
    bridge.core_config.current_reward_signal = 0.0
    counts = {a: 0 for a in ["N", "E", "S", "W"]}
    for s in range(400):  # 200ms warmup + readout
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        if s >= 200:
            firing = bridge.cp_firing_states
            for a in counts:
                counts[a] += int(firing[cortex_idx[a]].sum().get())
    print(f"  Spontaneous cortex spikes (last 100ms): {counts}")

    # Test 2: Symmetric drive on all 4 cortex pools
    print()
    print("=" * 60)
    print("TEST 2: EQUAL drive to all 4 cortex_X — should fire equally if symmetric")
    print("=" * 60)
    bridge.cp_external_input_current[:] = 0.0
    for a, idx in cortex_idx.items():
        bridge.cp_external_input_current[idx] = cp.float32(100.0)
    counts = {a: 0 for a in ["N", "E", "S", "W"]}
    for s in range(400):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        if s >= 200:
            firing = bridge.cp_firing_states
            for a in counts:
                counts[a] += int(firing[cortex_idx[a]].sum().get())
    print(f"  Equal-drive cortex spikes (last 100ms): {counts}")
    total = sum(counts.values())
    if total > 0:
        for a in ["N", "E", "S", "W"]:
            print(f"  {a}: {100*counts[a]/total:.1f}% of total")

    # Test 3: Drive language_input with each token in turn
    print()
    print("=" * 60)
    print("TEST 3: language_input drive (untrained) — what does cascade output?")
    print("=" * 60)
    for word in ["north", "east", "south", "west"]:
        bridge.cp_external_input_current[:] = 0.0
        drive = vocab_to_drive_pattern(word, n_neurons=int(lang_input_idx.size),
                                         drive_max_pA=200.0, sparsity=0.1)
        bridge.cp_external_input_current[lang_input_idx] = cp.asarray(drive, dtype=cp.float32)
        counts = {a: 0 for a in ["N", "E", "S", "W"]}
        for s in range(400):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            if s >= 200:
                firing = bridge.cp_firing_states
                for a in counts:
                    counts[a] += int(firing[cortex_idx[a]].sum().get())
        winner = max(counts, key=lambda x: counts[x])
        print(f"  word='{word}'  cortex spikes: {counts}  winner={winner}")


if __name__ == "__main__":
    main()
