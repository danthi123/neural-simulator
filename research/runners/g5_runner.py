"""G5: Sensorimotor closed-loop gridworld.

A single long episode where the brain's motor output drives the environment,
which produces the next sensory input. No reward, no training. We want to
see whether the reservoir + motor head produces distinguishable trajectories
across seeds.

Topology (266 neurons):
    64 input neurons  (sensor, trait 0)  — Poisson-driven by position encoding
    160 hidden exc     (trait 0, reservoir)
    40 hidden inh      (trait 1, reservoir)
    2 motor neurons    (trait 0, read by argmax)  — indices 264 (left), 265 (right)

Connectivity (all fixed, no plasticity):
    input -> hidden (50% density, same as v3)
    hidden recurrent (10% density, same as v3)
    hidden -> motor (25% density)

Episode: 200 timesteps. At each step we encode current position as 64-d rate
vector (Gaussian tuning), present it for 150 ms, read motor spike counts in
[100, 150] ms, apply argmax action, update position (±1 on 16-cell line,
clipped), log. No reset between steps.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import (ExperimentConfig, ExperimentPhase, StimulusChannel,
                        StimulusPattern, NeuronGroup, ReadoutConfig,
                        CoreSimConfig)
from sim.enums import (StimulusPatternType, ExperimentPhaseType,
                       NeuronGroupRole, NeuronModel)
from experiment import ExperimentEngine


STIMULUS_MS = 150.0
READOUT_START_MS = 100.0
READOUT_END_MS = 150.0


def _position_to_rates(x, n_positions=16, n_input=64,
                       rate_peak=30.0, rate_floor=1.0, sigma=1.5):
    """Encode 1D position as per-neuron Poisson rate vector.

    Each input neuron has a Gaussian receptive field over positions.
    Neuron tuned to position p_i fires at rate ~ rate_peak * exp(-((x-p_i)/sigma)^2)
    + rate_floor.
    """
    rates = np.zeros(n_input, dtype=np.float32)
    for i in range(n_input):
        # Assign each input neuron a preferred position in [0, n_positions-1]
        p_i = (i / (n_input - 1)) * (n_positions - 1)
        gauss = rate_peak * np.exp(-((x - p_i) / sigma) ** 2)
        rates[i] = gauss + rate_floor
    return rates


def _build_g5_plan(
    seed,
    n_input=64,
    n_hidden_exc=160,
    n_hidden_inh=40,
    n_motor=2,
    input_to_hidden_density=0.5,
    hidden_to_hidden_density=0.1,
    hidden_to_motor_density=0.25,
    input_to_hidden_weight=1.5,
    hidden_exc_weight=0.3,
    hidden_inh_weight=0.8,
    hidden_to_motor_weight=1.5,
):
    n_total = n_input + n_hidden_exc + n_hidden_inh + n_motor
    input_idx = list(range(0, n_input))
    hidden_exc_idx = list(range(n_input, n_input + n_hidden_exc))
    hidden_inh_idx = list(range(n_input + n_hidden_exc,
                                n_input + n_hidden_exc + n_hidden_inh))
    hidden_idx = hidden_exc_idx + hidden_inh_idx
    motor_idx = list(range(n_input + n_hidden_exc + n_hidden_inh, n_total))

    core_cfg = CoreSimConfig()
    core_cfg.num_neurons = n_total
    core_cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    core_cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    core_cfg.seed = int(seed)
    core_cfg.dt_ms = 1.0
    core_cfg.num_traits = 2
    core_cfg.inhibitory_trait_indices = [1]
    core_cfg.connections_per_neuron = 0

    core_cfg.enable_stdp = False
    core_cfg.enable_hebbian_learning = False
    core_cfg.enable_short_term_plasticity = False
    core_cfg.enable_homeostasis = False
    core_cfg.enable_reward_modulation = False
    core_cfg.enable_structural_plasticity = False
    core_cfg.enable_watts_strogatz = False

    core_cfg.propagation_strength = 1.0
    core_cfg.inhibitory_propagation_strength = 1.0
    core_cfg.ou_std_current_pA = 60.0

    rng = np.random.default_rng(seed)

    # Input -> hidden
    pre_ih, post_ih = [], []
    for i in input_idx:
        n_conn = max(1, int(len(hidden_idx) * input_to_hidden_density))
        targets = rng.choice(hidden_idx, size=n_conn, replace=False)
        for t in targets:
            pre_ih.append(i); post_ih.append(int(t))
    w_ih = np.clip(rng.normal(input_to_hidden_weight,
                              input_to_hidden_weight * 0.2, size=len(pre_ih)),
                   0.01, None).astype(np.float32)

    # Hidden recurrent
    pre_hh, post_hh = [], []
    w_hh_list = []
    for i in hidden_idx:
        is_inh = i in hidden_inh_idx
        candidates = [j for j in hidden_idx if j != i]
        n_conn = max(1, int(len(candidates) * hidden_to_hidden_density))
        targets = rng.choice(candidates, size=n_conn, replace=False)
        base_w = hidden_inh_weight if is_inh else hidden_exc_weight
        for t in targets:
            pre_hh.append(i); post_hh.append(int(t))
            w_hh_list.append(base_w + rng.normal(0, base_w * 0.2))
    w_hh = np.clip(np.asarray(w_hh_list, dtype=np.float32), 0.01, None)

    # Hidden -> motor
    pre_hm, post_hm = [], []
    for i in hidden_idx:
        is_inh = i in hidden_inh_idx
        n_conn = max(1, int(len(motor_idx) * hidden_to_motor_density))
        # All motor targets are reachable
        targets = rng.choice(motor_idx, size=n_conn, replace=False)
        for t in targets:
            pre_hm.append(i); post_hm.append(int(t))
    # Fixed motor input weight scaled down for excitatory, up for inhibitory
    # (sim handles signs via pre-neuron trait).
    w_hm_list = []
    for pre in pre_hm:
        is_inh = pre in hidden_inh_idx
        base_w = hidden_inh_weight if is_inh else hidden_to_motor_weight
        w_hm_list.append(base_w + rng.normal(0, base_w * 0.2))
    w_hm = np.clip(np.asarray(w_hm_list, dtype=np.float32), 0.01, None)

    plan = {
        "input_to_hidden": {
            "pre_indices": pre_ih, "post_indices": post_ih,
            "initial_weights": w_ih, "plastic": False,
            "conn_type": "E_TO_MIX", "count": len(pre_ih),
        },
        "hidden_recurrent": {
            "pre_indices": pre_hh, "post_indices": post_hh,
            "initial_weights": w_hh, "plastic": False,
            "conn_type": "MIXED", "count": len(pre_hh),
        },
        "hidden_to_motor": {
            "pre_indices": pre_hm, "post_indices": post_hm,
            "initial_weights": w_hm, "plastic": False,
            "conn_type": "MIXED", "count": len(pre_hm),
        },
        "layout": {
            "input_idx": input_idx, "hidden_exc_idx": hidden_exc_idx,
            "hidden_inh_idx": hidden_inh_idx, "hidden_idx": hidden_idx,
            "motor_idx": motor_idx,
        },
    }
    return core_cfg, plan


def run_g5_episode(
    out_path,
    seed,
    n_steps=200,
    n_positions=16,
    start_position=8,
    verbose=True,
):
    import cupy as cp

    core_cfg, plan = _build_g5_plan(seed=seed)
    bridge = SimulationBridge(
        core_config=core_cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    layout = plan["layout"]
    # Assign traits: inhibitory only for hidden_inh_idx.
    new_traits = np.zeros(core_cfg.num_neurons, dtype=np.int32)
    for i in layout["hidden_inh_idx"]:
        new_traits[i] = 1
    bridge.cp_traits = cp.asarray(new_traits)
    bridge._cached_inhibitory_mask = None

    bridge.inject_explicit_wiring(plan, output_inhibitory_indices=None)

    if bridge.cp_external_input_current is not None:
        bridge.cp_external_input_current[:] = 0.0

    engine = ExperimentEngine(core_cfg.num_neurons, core_cfg.dt_ms)
    exp_cfg = ExperimentConfig()
    exp_cfg.neuron_groups = [
        NeuronGroup(name="input", role=NeuronGroupRole.INPUT.name,
                    neuron_indices=layout["input_idx"]),
        NeuronGroup(name="hidden", role=NeuronGroupRole.HIDDEN.name,
                    neuron_indices=layout["hidden_idx"]),
        NeuronGroup(name="motor", role=NeuronGroupRole.OUTPUT.name,
                    neuron_indices=layout["motor_idx"]),
    ]
    exp_cfg.readout = ReadoutConfig(
        rate_window_ms=50.0, spike_count_window_ms=50.0,
        rate_group_names=["input", "hidden", "motor"],
    )
    exp_cfg.phases = [ExperimentPhase(
        name="g5", phase_type=ExperimentPhaseType.TRAINING.name,
        duration_ms=1e9,
    )]
    engine.load_experiment(exp_cfg)
    engine.initialize(cp_traits=bridge.cp_traits, cp_module=cp)
    engine.is_experiment_running = True
    bridge.experiment_engine = engine

    dt = core_cfg.dt_ms
    n_stim_steps = int(STIMULUS_MS / dt)
    readout_start_step = int(READOUT_START_MS / dt)
    readout_end_step = int(READOUT_END_MS / dt)

    motor_idx_cp = cp.asarray(layout["motor_idx"], dtype=cp.int32)

    x = int(start_position)
    trajectory = [x]
    motor_counts_log = []

    t0 = time.time()
    for step in range(n_steps):
        # Encode current position as rate vector
        rates = _position_to_rates(x, n_positions=n_positions,
                                   n_input=len(layout["input_idx"]))

        pat = StimulusPattern(
            pattern_type=StimulusPatternType.RATE_VECTOR_POISSON.name,
            spike_current_pA=1000.0, spike_duration_ms=2.0,
            rate_vector_hz=[float(r) for r in rates],
        )
        ch = StimulusChannel(
            name="sensor", pattern=pat,
            target_neuron_indices=layout["input_idx"],
            onset_ms=0.0, duration_ms=STIMULUS_MS, enabled=True,
        )
        engine.stimulus_manager.cleanup()
        engine.stimulus_manager.initialize([ch], engine.group_manager, cp)
        engine.phase_start_ms = bridge.runtime_state.current_time_ms

        counts = np.zeros(len(layout["motor_idx"]), dtype=np.int32)
        for s in range(n_stim_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * dt
            if readout_start_step <= s < readout_end_step:
                fired = bridge.cp_firing_states[motor_idx_cp].get().astype(np.int32)
                counts += fired

        motor_counts_log.append(counts.tolist())
        action = int(np.argmax(counts)) if counts.sum() > 0 else 0  # idle → left
        # 0 = left (−1), 1 = right (+1)
        new_x = x + (1 if action == 1 else -1)
        new_x = int(np.clip(new_x, 0, n_positions - 1))
        x = new_x
        trajectory.append(x)

        if verbose and step % 20 == 19:
            print(f"[g5 seed={seed}] step {step+1}/{n_steps}  x={x}  "
                  f"last_motor={counts.tolist()}", flush=True)

    elapsed = time.time() - t0
    # Summary statistics
    traj_arr = np.asarray(trajectory)
    motor_arr = np.asarray(motor_counts_log)

    results = {
        "seed": seed, "n_steps": n_steps, "n_positions": n_positions,
        "start_position": start_position,
        "trajectory": trajectory,
        "motor_counts": motor_counts_log,
        "trajectory_mean": float(traj_arr.mean()),
        "trajectory_std": float(traj_arr.std()),
        "trajectory_min": int(traj_arr.min()),
        "trajectory_max": int(traj_arr.max()),
        "n_distinct_positions_visited": int(len(np.unique(traj_arr))),
        "n_left_actions": int((motor_arr[:, 0] >= motor_arr[:, 1]).sum()),
        "n_right_actions": int((motor_arr[:, 1] > motor_arr[:, 0]).sum()),
        "motor_total_spikes": int(motor_arr.sum()),
        "motor_silent_steps": int((motor_arr.sum(axis=1) == 0).sum()),
        "elapsed_seconds": elapsed,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"[g5 seed={seed}] done: mean_x={results['trajectory_mean']:.1f}  "
              f"range=[{results['trajectory_min']},{results['trajectory_max']}]  "
              f"visited={results['n_distinct_positions_visited']}  "
              f"silent_steps={results['motor_silent_steps']}  {elapsed:.1f}s",
              flush=True)
    return results
