"""G5.v2: Goal-seeking sensorimotor loop with reward modulation.

Extends G5's closed loop with a task-level reward signal and plastic
hidden->motor synapses. Per step:
    1. Encode position, present 150 ms stimulus.
    2. Read motor spike counts -> action (0=left, 1=right).
    3. Update position.
    4. Compute reward = -|x - goal| / n_positions  (in [-1, 0])
       — positive signal (0) when at goal, negative when far.
    5. Run a short "reward delivery window" (50 ms) during which
       `core_config.current_reward_signal` is set, letting the sim's
       reward-modulation update plastic (hidden->motor) weights via
       eligibility × reward.
    6. Clear reward signal, loop.

Success metric: mean distance to goal in the *second half* of the episode
is lower than the *first half*, averaged across seeds. Stronger: monotone
improvement across episode quarters.
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

from research.runners.g5_runner import _position_to_rates


STIMULUS_MS = 150.0
READOUT_START_MS = 100.0
READOUT_END_MS = 150.0
REWARD_WINDOW_MS = 50.0


def _build_g5v2_plan(
    seed,
    n_input=64,
    n_hidden_exc=160,
    n_hidden_inh=40,
    n_motor=2,
    input_to_hidden_density=0.5,
    hidden_to_hidden_density=0.1,
    hidden_to_motor_density=0.5,
    input_to_hidden_weight=1.5,
    hidden_exc_weight=0.3,
    hidden_inh_weight=0.8,
    hidden_to_motor_weight=1.0,
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

    # STDP + reward modulation on. Only hidden->motor is plastic (via mask).
    core_cfg.enable_stdp = True
    core_cfg.stdp_a_plus = 0.010
    core_cfg.stdp_a_minus = 0.010
    core_cfg.stdp_tau_plus_ms = 20.0
    core_cfg.stdp_tau_minus_ms = 20.0
    core_cfg.stdp_w_min = 0.0
    core_cfg.stdp_w_max = 3.0

    core_cfg.enable_reward_modulation = True
    core_cfg.reward_learning_rate = 0.05
    core_cfg.reward_eligibility_tau_ms = 500.0  # faster decay for tight loop
    core_cfg.reward_baseline = 0.0
    core_cfg.current_reward_signal = 0.0

    # Other plasticity off
    core_cfg.enable_hebbian_learning = False
    core_cfg.enable_short_term_plasticity = False
    core_cfg.enable_homeostasis = False
    core_cfg.enable_structural_plasticity = False
    core_cfg.enable_watts_strogatz = False

    core_cfg.propagation_strength = 1.0
    core_cfg.inhibitory_propagation_strength = 1.0
    core_cfg.ou_std_current_pA = 60.0

    rng = np.random.default_rng(seed)

    # Input -> hidden (FROZEN)
    pre_ih, post_ih = [], []
    for i in input_idx:
        n_conn = max(1, int(len(hidden_idx) * input_to_hidden_density))
        targets = rng.choice(hidden_idx, size=n_conn, replace=False)
        for t in targets:
            pre_ih.append(i); post_ih.append(int(t))
    w_ih = np.clip(rng.normal(input_to_hidden_weight,
                              input_to_hidden_weight * 0.2, size=len(pre_ih)),
                   0.01, None).astype(np.float32)

    # Hidden recurrent (FROZEN)
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

    # Hidden -> motor (PLASTIC — this is what the reward signal shapes)
    pre_hm, post_hm = [], []
    w_hm_list = []
    for i in hidden_idx:
        is_inh = i in hidden_inh_idx
        n_conn = max(1, int(len(motor_idx) * hidden_to_motor_density))
        targets = rng.choice(motor_idx, size=n_conn, replace=False)
        base_w = hidden_inh_weight if is_inh else hidden_to_motor_weight
        for t in targets:
            pre_hm.append(i); post_hm.append(int(t))
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
            "initial_weights": w_hm, "plastic": True,    # <-- THE plastic layer
            "conn_type": "MIXED", "count": len(pre_hm),
        },
        "layout": {
            "input_idx": input_idx, "hidden_exc_idx": hidden_exc_idx,
            "hidden_inh_idx": hidden_inh_idx, "hidden_idx": hidden_idx,
            "motor_idx": motor_idx,
        },
    }
    return core_cfg, plan


def run_g5_v2_episode(
    out_path,
    seed,
    n_steps=400,
    n_positions=16,
    start_position=8,
    goal_position=12,
    verbose=True,
):
    import cupy as cp

    core_cfg, plan = _build_g5v2_plan(seed=seed)
    bridge = SimulationBridge(
        core_config=core_cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
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
        rate_group_names=["motor"],
    )
    exp_cfg.phases = [ExperimentPhase(
        name="g5v2", phase_type=ExperimentPhaseType.TRAINING.name,
        duration_ms=1e9,
    )]
    engine.load_experiment(exp_cfg)
    engine.initialize(cp_traits=bridge.cp_traits, cp_module=cp)
    engine.is_experiment_running = True
    bridge.experiment_engine = engine

    dt = core_cfg.dt_ms
    n_stim_steps = int(STIMULUS_MS / dt)
    n_reward_steps = int(REWARD_WINDOW_MS / dt)
    readout_start_step = int(READOUT_START_MS / dt)
    readout_end_step = int(READOUT_END_MS / dt)

    motor_idx_cp = cp.asarray(layout["motor_idx"], dtype=cp.int32)

    x = int(start_position)
    trajectory = [x]
    motor_counts_log = []
    reward_log = []
    distance_log = [abs(x - goal_position)]

    t0 = time.time()
    for step in range(n_steps):
        # Encode current position
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

        # Stimulus window (reward = 0, eligibility accumulates naturally)
        counts = np.zeros(len(layout["motor_idx"]), dtype=np.int32)
        core_cfg.current_reward_signal = 0.0
        for s in range(n_stim_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * dt
            if readout_start_step <= s < readout_end_step:
                fired = bridge.cp_firing_states[motor_idx_cp].get().astype(np.int32)
                counts += fired

        # Action & environment update
        motor_counts_log.append(counts.tolist())
        action = int(np.argmax(counts)) if counts.sum() > 0 else 0
        new_x = int(np.clip(x + (1 if action == 1 else -1), 0, n_positions - 1))
        x = new_x
        trajectory.append(x)
        dist = abs(x - goal_position)
        distance_log.append(dist)

        # Reward: in [-1, 0], 0 when at goal. Use a centered signal so
        # reward_prediction_error = reward - baseline can push or pull.
        # baseline = -avg_distance / max_distance roughly; keep simple.
        max_dist = n_positions - 1
        reward = 1.0 - (dist / max_dist) * 2.0  # maps dist=0 -> +1, dist=max -> -1
        reward_log.append(reward)

        # Reward delivery window: continue sim with reward signal active,
        # stimulus off (gap). During this the eligibility decays and
        # reward_modulation applies delta_w = lr * reward * eligibility.
        engine.stimulus_manager.cleanup()
        engine.stimulus_manager.initialize([], engine.group_manager, cp)
        core_cfg.current_reward_signal = float(reward)
        for s in range(n_reward_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * dt
        core_cfg.current_reward_signal = 0.0

        if verbose and (step + 1) % 50 == 0:
            recent_dist = np.mean(distance_log[-50:])
            w = cp.asnumpy(bridge.cp_connections.data)
            mask = cp.asnumpy(bridge.cp_synapse_plastic_mask)
            w_plastic = w[mask]
            print(f"[g5v2 seed={seed}] step {step+1}/{n_steps}  x={x}  "
                  f"recent_dist={recent_dist:.2f}  W plastic range=[{w_plastic.min():.2f},{w_plastic.max():.2f}] mean={w_plastic.mean():.2f}",
                  flush=True)

    elapsed = time.time() - t0
    traj_arr = np.asarray(trajectory[1:])  # exclude start
    dist_arr = np.asarray(distance_log[1:])

    # Quarter-averaged distance: trained brain should show decrease.
    q = len(dist_arr) // 4
    quarters = [float(dist_arr[i*q:(i+1)*q].mean()) for i in range(4)]

    results = {
        "seed": seed, "n_steps": n_steps, "n_positions": n_positions,
        "start_position": start_position, "goal_position": goal_position,
        "trajectory": trajectory,
        "motor_counts": motor_counts_log,
        "reward_log": reward_log,
        "distance_log": distance_log,
        "trajectory_mean": float(traj_arr.mean()),
        "trajectory_std": float(traj_arr.std()),
        "mean_distance_overall": float(dist_arr.mean()),
        "mean_distance_quarters": quarters,
        "distance_quarter_improvement": float(quarters[0] - quarters[-1]),
        "n_steps_at_goal": int((dist_arr == 0).sum()),
        "elapsed_seconds": elapsed,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"[g5v2 seed={seed}] done: mean_dist={results['mean_distance_overall']:.2f}  "
              f"quarters={quarters}  at_goal={results['n_steps_at_goal']}  {elapsed:.1f}s",
              flush=True)
    return results
