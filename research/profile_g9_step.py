"""Quick profile of g9_runner inner loop to identify optimization targets.

Measures time-per-step broken down by:
  - bridge._run_one_simulation_step() — the GPU work
  - per-step CPU sync for firing readout (motor, hidden, first_spike)
  - Python loop + runtime state updates

This informs Route B: which syncs to batch first.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def main():
    import cupy as cp
    from research.runners.g9_runner import _build_g9_plan
    from sim import (SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig)
    from sim.config import (ExperimentConfig, ExperimentPhase, StimulusChannel,
                            StimulusPattern, NeuronGroup, ReadoutConfig)
    from sim.enums import (StimulusPatternType, ExperimentPhaseType,
                           NeuronGroupRole, NeuronModel)
    from experiment import ExperimentEngine

    core_cfg, plan = _build_g9_plan(seed=42)
    bridge = SimulationBridge(core_config=core_cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    layout = plan["layout"]
    new_traits = np.zeros(core_cfg.num_neurons, dtype=np.int32)
    for i in layout["hidden_inh_idx"]:
        new_traits[i] = 1
    bridge.cp_traits = cp.asarray(new_traits)
    bridge._cached_inhibitory_mask = None
    bridge.inject_explicit_wiring(plan, output_inhibitory_indices=None)
    if bridge.cp_external_input_current is not None:
        bridge.cp_external_input_current[:] = 0.0

    # Lightweight experiment setup
    engine = ExperimentEngine(core_cfg.num_neurons, core_cfg.dt_ms)
    exp_cfg = ExperimentConfig()
    exp_cfg.neuron_groups = [
        NeuronGroup(name="input", role=NeuronGroupRole.INPUT.name,
                    neuron_indices=layout["input_idx"]),
    ]
    exp_cfg.readout = ReadoutConfig(rate_window_ms=100.0,
                                     spike_count_window_ms=100.0,
                                     rate_group_names=["input"])
    exp_cfg.phases = [ExperimentPhase(name="profile",
                                       phase_type=ExperimentPhaseType.TRAINING.name,
                                       duration_ms=1e9)]
    engine.load_experiment(exp_cfg)
    engine.initialize(cp_traits=bridge.cp_traits, cp_module=cp)
    engine.is_experiment_running = True
    bridge.experiment_engine = engine

    motor_idx_cp = cp.asarray(layout["motor_idx"], dtype=cp.int32)
    hidden_idx_cp = cp.asarray(layout["hidden_idx"], dtype=cp.int32)
    n_motor = len(layout["motor_idx"])
    n_hidden = len(layout["hidden_idx"])

    # Warm up
    for _ in range(20):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    print("\n== Baseline: current serial-sync pattern ==")
    t0 = time.time()
    motor_counts = np.zeros(n_motor, dtype=np.int32)
    hidden_counts = np.zeros(n_hidden, dtype=np.int32)
    first_spike_step = np.full(n_motor, -1, dtype=np.int32)
    for s in range(150):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step
        if 50 <= s < 150:
            fired = bridge.cp_firing_states[motor_idx_cp].get().astype(bool)
            motor_counts += fired.astype(np.int32)
            new_firing = fired & (first_spike_step == -1)
            first_spike_step[new_firing] = s
            hidden_counts += bridge.cp_firing_states[hidden_idx_cp].get().astype(np.int32)
    t_serial = time.time() - t0
    print(f"  150 steps with CPU sync per step: {t_serial*1000:.1f} ms "
          f"({t_serial*1000/150:.3f} ms/step)")

    # Alternative: accumulate on GPU, sync only at end
    print("\n== Route B: GPU-side accumulation, single sync at trial end ==")
    t0 = time.time()
    motor_counts_gpu = cp.zeros(n_motor, dtype=cp.int32)
    hidden_counts_gpu = cp.zeros(n_hidden, dtype=cp.int32)
    first_spike_step_gpu = cp.full(n_motor, -1, dtype=cp.int32)
    for s in range(150):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step
        if 50 <= s < 150:
            fired = bridge.cp_firing_states[motor_idx_cp].astype(cp.int32)
            motor_counts_gpu += fired
            # first-spike update, GPU side
            fired_bool = fired > 0
            is_first = (first_spike_step_gpu == -1) & fired_bool
            first_spike_step_gpu = cp.where(is_first, s, first_spike_step_gpu)
            hidden_counts_gpu += bridge.cp_firing_states[hidden_idx_cp].astype(cp.int32)
    motor_counts_b = motor_counts_gpu.get()
    hidden_counts_b = hidden_counts_gpu.get()
    first_spike_step_b = first_spike_step_gpu.get()
    t_batched = time.time() - t0
    print(f"  150 steps with GPU accumulation + end sync: {t_batched*1000:.1f} ms "
          f"({t_batched*1000/150:.3f} ms/step)")

    print(f"\n  Speedup: {t_serial/t_batched:.2f}x")
    print(f"  Trial savings: {(t_serial - t_batched)*1000:.1f} ms per 150-step trial")
    # 600 trials per episode -> total savings
    print(f"  At 600 trials/episode: {(t_serial - t_batched)*600:.1f} s per episode")
    # Sanity: results should match
    if (motor_counts == motor_counts_b).all() and (hidden_counts == hidden_counts_b).all():
        print("  Result consistency: motor/hidden counts match serial baseline")
    else:
        print("  WARNING: motor/hidden counts differ — nondeterminism OR a bug")

    bridge.clear_simulation_state_and_gpu_memory()


if __name__ == "__main__":
    main()
