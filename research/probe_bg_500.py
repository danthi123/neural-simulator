"""Replicate runner static probe EXACTLY but bin per 50ms.

Runner's static probe runs 500 steps and reports cumulative rate. If
spikes happen in only first 50ms then die, runner's average will hide
that. Bin per 50ms to catch decay.
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

    regions, pathways = build_bg_brain_regions()  # default n_cortex=100, matching runner
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
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    region_indices = {r.name: list(bridge.region_manager.indices(r.name)) for r in regions if list(bridge.region_manager.indices(r.name))}

    # WARMUP: 50 steps no input (lets BG output nuclei reach tonic firing state)
    bridge.cp_external_input_current[:] = 0.0
    print("Warmup: 50 steps no input...")
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    bridge.cp_external_input_current[:] = 0.0
    for rn in [f"gpe_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[cp.asarray(region_indices[rn], dtype=cp.int64)] = cp.float32(150.0)
    for rn in [f"gpi_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[cp.asarray(region_indices[rn], dtype=cp.int64)] = cp.float32(110.0)
    for rn in ["stn", "dopamine"]:
        bridge.cp_external_input_current[cp.asarray(region_indices[rn], dtype=cp.int64)] = cp.float32(150.0)
    for rn in [f"thal_{a}" for a in ACTION_NAMES]:
        bridge.cp_external_input_current[cp.asarray(region_indices[rn], dtype=cp.int64)] = cp.float32(300.0)

    target = "N"
    cortex_cp = cp.asarray(region_indices[f"cortex_{target}"], dtype=cp.int64)
    motor_arrs = {a: np.array(region_indices[f"motor_{a}"]) for a in ACTION_NAMES}
    str_d1_arrs = {a: np.array(region_indices[f"str_D1_{a}"]) for a in ACTION_NAMES}
    gpi_arrs = {a: np.array(region_indices[f"gpi_{a}"]) for a in ACTION_NAMES}
    thal_arrs = {a: np.array(region_indices[f"thal_{a}"]) for a in ACTION_NAMES}

    bridge.runtime_state.current_time_step = 0
    bridge.runtime_state.current_time_ms = 0.0

    print(f"\nReplicate static probe pattern, drive cortex_{target}, bin per 50 steps for 500 steps.\n")
    bin_size = 50
    n_bins = 10
    for bin_idx in range(n_bins):
        motor_counts = {a: 0 for a in ACTION_NAMES}
        d1_counts = {a: 0 for a in ACTION_NAMES}
        gpi_counts = {a: 0 for a in ACTION_NAMES}
        thal_counts = {a: 0 for a in ACTION_NAMES}
        for s in range(bin_size):
            bridge.cp_external_input_current[cortex_cp] = cp.float32(800.0)
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            firing = bridge.cp_firing_states.get().astype(bool)
            for a in ACTION_NAMES:
                motor_counts[a] += int(firing[motor_arrs[a]].sum())
                d1_counts[a] += int(firing[str_d1_arrs[a]].sum())
                gpi_counts[a] += int(firing[gpi_arrs[a]].sum())
                thal_counts[a] += int(firing[thal_arrs[a]].sum())
        m_n = motor_counts['N']
        d1_n = d1_counts['N']
        gpi_n = gpi_counts['N']
        thal_n = thal_counts['N']
        print(f"Bin {bin_idx:2d} (steps {bin_idx*bin_size:3d}-{(bin_idx+1)*bin_size:3d}): "
              f"d1_N={d1_n:3d} gpi_N={gpi_n:3d} thal_N={thal_n:3d} motor_N={m_n:3d} "
              f"all_motors={[motor_counts[a] for a in ACTION_NAMES]}")


if __name__ == "__main__":
    main()
