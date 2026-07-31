"""Test if BG cascade survives trial-loop structure with constant input.

Strategy: replicate the moving-goal trial structure (110ms stim + 30-100ms
readout) but DON'T move the agent or change goals. Drive cortex_N + cortex_E
constantly across all trials. If motor activity decays across trials,
the issue is the trial structure itself (timing, state accumulation),
not the moving-goal specifics.
"""
import sys
import time
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

    # Set baselines + drive cortex_N and cortex_E
    bridge.cp_external_input_current[:] = 0.0
    for rn in [f"gpe_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(150.0)
    for rn in [f"gpi_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(110.0)
    for rn in ["stn", "dopamine"]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(150.0)
    for rn in [f"thal_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(300.0)
    bridge.cp_external_input_current[region_indices_cp["cortex_N"]] = cp.float32(800.0)
    bridge.cp_external_input_current[region_indices_cp["cortex_E"]] = cp.float32(800.0)

    motor_idx = {a: region_indices_cp[f"motor_{a}"] for a in ACTION_NAMES}

    print("\nContinuous run with RE-SET inputs every sim step (matching probe pattern).\n")
    bin_size = 100
    n_bins = 20
    for bin_idx in range(n_bins):
        motor_counts = {a: 0 for a in ACTION_NAMES}
        for s in range(bin_size):
            # Re-set drives every sim step
            bridge.cp_external_input_current[:] = 0.0
            for rn in [f"gpe_{a}" for a in ACTION_NAMES]:
                bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(150.0)
            for rn in [f"gpi_{a}" for a in ACTION_NAMES]:
                bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(110.0)
            for rn in ["stn", "dopamine"]:
                bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(150.0)
            for rn in [f"thal_{a}" for a in ACTION_NAMES]:
                bridge.cp_external_input_current[region_indices_cp[rn]] = cp.float32(300.0)
            bridge.cp_external_input_current[region_indices_cp["cortex_N"]] = cp.float32(800.0)

            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            firing = bridge.cp_firing_states.get().astype(bool)
            for a in ACTION_NAMES:
                motor_counts[a] += int(firing[motor_idx[a].get()].sum())
        print(f"Bin {bin_idx:2d}: motor {[motor_counts[a] for a in ACTION_NAMES]}")


if __name__ == "__main__":
    main()
