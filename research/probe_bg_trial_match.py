"""Probe that matches the runner trial structure EXACTLY.

Trial: zero input, set baselines + cortex drives, run 100 steps, count
motor spikes during steps 30-100. Repeat 20 trials.
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

    regions, pathways = build_bg_brain_regions(n_cortex=100)  # match runner
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
    cfg.enable_stdp = True
    cfg.enable_reward_modulation = True
    cfg.stdp_w_max = 30.0  # match runner — prevents STDP collapse of cortex→D1 weights
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

    region_idx = {r.name: list(bridge.region_manager.indices(r.name)) for r in regions if list(bridge.region_manager.indices(r.name))}
    region_idx_cp = {k: cp.asarray(v, dtype=cp.int64) for k, v in region_idx.items()}
    motor_arr = {a: np.array(region_idx[f"motor_{a}"]) for a in ACTION_NAMES}
    d1_arr = {a: np.array(region_idx[f"str_D1_{a}"]) for a in ACTION_NAMES}
    gpi_arr = {a: np.array(region_idx[f"gpi_{a}"]) for a in ACTION_NAMES}
    thal_arr = {a: np.array(region_idx[f"thal_{a}"]) for a in ACTION_NAMES}

    n_stim = 100
    readout_start = 0   # broaden to catch initial transient
    readout_end = 100
    n_trials = 20

    print("\nTrial-match probe: 100ms stim/trial, 30-100ms readout, drive cortex_N (single pool, like static probe).\n")

    for trial in range(n_trials):
        # === Match runner: zero everything, reset baselines + cortex drive ===
        bridge.cp_external_input_current[:] = 0.0
        for rn in [f"gpe_{a}" for a in ACTION_NAMES]:
            bridge.cp_external_input_current[region_idx_cp[rn]] = cp.float32(150.0)
        for rn in [f"gpi_{a}" for a in ACTION_NAMES]:
            bridge.cp_external_input_current[region_idx_cp[rn]] = cp.float32(110.0)
        for rn in ["stn", "dopamine"]:
            bridge.cp_external_input_current[region_idx_cp[rn]] = cp.float32(150.0)
        for rn in [f"thal_{a}" for a in ACTION_NAMES]:
            bridge.cp_external_input_current[region_idx_cp[rn]] = cp.float32(300.0)
        # Drive only cortex_N (single pool)
        bridge.cp_external_input_current[region_idx_cp["cortex_N"]] = cp.float32(800.0)

        motor_counts = {a: 0 for a in ACTION_NAMES}
        d1_counts = {a: 0 for a in ACTION_NAMES}
        gpi_counts = {a: 0 for a in ACTION_NAMES}
        thal_counts = {a: 0 for a in ACTION_NAMES}
        bridge.core_config.current_reward_signal = 0.0
        for s in range(n_stim):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            if readout_start <= s < readout_end:
                firing = bridge.cp_firing_states.get().astype(bool)
                for a in ACTION_NAMES:
                    motor_counts[a] += int(firing[motor_arr[a]].sum())
                    d1_counts[a] += int(firing[d1_arr[a]].sum())
                    gpi_counts[a] += int(firing[gpi_arr[a]].sum())
                    thal_counts[a] += int(firing[thal_arr[a]].sum())
        print(f"Trial {trial:2d}: motor[N,E,S,W]={[motor_counts[a] for a in ACTION_NAMES]} "
              f"d1[N]={d1_counts['N']:>3d} gpi[N]={gpi_counts['N']:>3d} thal[N]={thal_counts['N']:>3d}", flush=True)

        # No reward-hold step (since no agent / reward)


if __name__ == "__main__":
    main()
