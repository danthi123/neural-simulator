"""Mirror the static probe EXACTLY, but in a stand-alone script.

Static probe in g11_bg_runner.py runs 500 steps of cortex_W drive and
sustains motor_W = 7 Hz throughout. The trial-structure probe dies
after ~100 steps. The probes differ in two ways:
  1. Cortex pool driven (W vs N)
  2. Drive pattern (set once vs zero-and-reset every step)

This probe replicates the static probe pattern (set baselines once,
only update cortex drive each step) but uses cortex_N. If it
sustains, the trial-structure probe's zero-and-reset is the bug.
If it dies, there's an asymmetry in cortex_N pool wiring.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cupy as cp


def main():
    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
    )
    from sim.enums import NeuronModel
    from research.runners.g11_bg_runner import build_bg_brain_regions, ACTION_NAMES

    regions, pathways = build_bg_brain_regions(n_cortex=400)
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    cfg.seed = 42
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.enable_parameter_heterogeneity = False

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    region_indices_cp = {}
    for r in regions:
        idx = list(bridge.region_manager.indices(r.name))
        if idx:
            region_indices_cp[r.name] = cp.asarray(idx, dtype=cp.int64)

    # === EXACTLY MATCH STATIC PROBE: baselines set ONCE, never re-zeroed ===
    bridge.cp_external_input_current[:] = 0.0
    for rn in [f"gpe_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(150.0)
    for rn in [f"gpi_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(110.0)
    for rn in ["stn", "dopamine"]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(150.0)
    for rn in [f"thal_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(300.0)

    target = "N"
    cortex_cp = region_indices_cp[f"cortex_{target}"]
    motor_idx = {a: region_indices_cp[f"motor_{a}"] for a in ACTION_NAMES}

    bridge.runtime_state.current_time_step = 0
    bridge.runtime_state.current_time_ms = 0.0

    print(f"\nStatic-pattern probe: drive cortex_{target}=800 pA + tonic baselines, no zeroing.\n")
    bin_size = 100
    n_bins = 20
    total_motor_counts = {a: 0 for a in ACTION_NAMES}
    for bin_idx in range(n_bins):
        motor_counts = {a: 0 for a in ACTION_NAMES}
        for s in range(bin_size):
            # ONLY set cortex drive — leave baselines as set above
            bridge.cp_external_input_current[cortex_cp] = cp.float32(800.0)
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            firing = bridge.cp_firing_states.get().astype(bool)
            for a in ACTION_NAMES:
                motor_counts[a] += int(firing[motor_idx[a].get()].sum())
                total_motor_counts[a] += motor_counts[a] if s == bin_size - 1 else 0
        print(f"Bin {bin_idx:2d}: motor {[motor_counts[a] for a in ACTION_NAMES]}")


if __name__ == "__main__":
    main()
