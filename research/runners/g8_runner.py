"""G8: goal-context channel probe on 2D gridworld.

Extends G6/G7 with an explicit "goal-context" input channel. Biology analogue:
PFC top-down persistent-activity signal that represents the current goal and
projects to motor-preparing circuits. Same encoding scheme as the sensory
channel (Gaussian tuning curves over goal x/y position).

Purpose: diagnostic. G7 (runner-side perceptron, no goal-context) was NO-GO
on moving-goal readaptation — the agent couldn't switch its policy when the
goal moved. Two hypotheses:

  (H1) Missing-information: the hidden layer never sees the goal, so the
       perceptron can only learn one fixed input->motor mapping. Adding a
       goal-context channel should enable context-dependent policies.
  (H2) Architectural-limit: the cliff-edged argmax + per-step credit rule
       can't readapt regardless of what information is available — the
       already-specialized weights form a local minimum.

G8 distinguishes H1 from H2. If G8 readapts cleanly on the moving-goal
task, the issue was (H1) and we have a biologically-plausible fix (PFC-like
context inputs). If G8 still can't readapt, it's (H2) and we need to move
learning into the sim with sim-native R-STDP + soft action selection
(first-spike WTA).

Biology constraint: the goal-context encoding is a stand-in for PFC
persistent activity. In a full biological model this would be a recurrent
PFC circuit that holds the goal representation via reverberating activity;
here we simulate the *output* of that circuit (steady Poisson rates) to
keep the probe quick. If G8 passes, the follow-up is to add a real
recurrent PFC submodule in the sim.
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
READOUT_START_MS = 50.0
READOUT_END_MS = 150.0

# Cardinal movements, in (dx, dy). Order matches motor-neuron index:
# 0=N, 1=E, 2=S, 3=W.
ACTION_DELTAS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def _gaussian_tuned_rates(value, n_neurons, n_positions, sigma, rate_peak, rate_floor):
    """Gaussian-tuned rates across n_neurons covering [0, n_positions-1]."""
    rates = np.zeros(n_neurons, dtype=np.float32)
    for i in range(n_neurons):
        p_i = (i / (n_neurons - 1)) * (n_positions - 1) if n_neurons > 1 else 0.0
        rates[i] = rate_peak * np.exp(-((value - p_i) / sigma) ** 2) + rate_floor
    return rates


def _position_to_rates_2d(x, y, n_sensor_half, n_positions,
                          rate_peak=30.0, rate_floor=1.0, sigma=1.5):
    """Sensory channel: current agent position."""
    rx = _gaussian_tuned_rates(x, n_sensor_half, n_positions, sigma, rate_peak, rate_floor)
    ry = _gaussian_tuned_rates(y, n_sensor_half, n_positions, sigma, rate_peak, rate_floor)
    return np.concatenate([rx, ry])


def _goal_to_context_rates(gx, gy, n_context_half, n_positions,
                            rate_peak=30.0, rate_floor=1.0, sigma=1.5,
                            enabled=True):
    """Goal-context channel: current goal location (PFC-like top-down signal).

    If enabled=False, returns rate_floor everywhere (ablation control).
    """
    if not enabled:
        return np.full(2 * n_context_half, rate_floor, dtype=np.float32)
    gcx = _gaussian_tuned_rates(gx, n_context_half, n_positions, sigma, rate_peak, rate_floor)
    gcy = _gaussian_tuned_rates(gy, n_context_half, n_positions, sigma, rate_peak, rate_floor)
    return np.concatenate([gcx, gcy])


def _build_g8_plan(
    seed,
    n_sensor=64,                # 32 x + 32 y (current position)
    n_goal_context=64,          # 32 gx + 32 gy (goal position)
    n_hidden_exc=160,
    n_hidden_inh=40,
    n_motor=4,                  # N, E, S, W
    input_to_hidden_density=0.5,
    hidden_to_hidden_density=0.1,
    hidden_to_motor_density=0.5,
    input_to_hidden_weight=1.5,
    hidden_exc_weight=0.3,
    hidden_inh_weight=0.8,
    hidden_to_motor_weight=1.0,
):
    """Build G8 wiring plan.

    Layout:
        [0 .. n_sensor-1]                            : sensory input (position)
        [n_sensor .. n_sensor+n_goal_context-1]       : goal-context input (PFC-like)
        [n_sensor+n_goal_context ..]                  : hidden exc
        [ .. ]                                        : hidden inh
        [ .. n_total-1]                               : motor

    Sensory and goal-context both project to the hidden layer with the same
    density and weight. The hidden -> motor layer is the only plastic layer.
    """
    n_input = n_sensor + n_goal_context
    n_total = n_input + n_hidden_exc + n_hidden_inh + n_motor
    sensor_idx = list(range(0, n_sensor))
    goal_context_idx = list(range(n_sensor, n_sensor + n_goal_context))
    input_idx = sensor_idx + goal_context_idx
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

    # All sim plasticity off (runner-side learning).
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

    # Input -> hidden (sensory + goal-context both project the same way)
    pre_ih, post_ih = [], []
    for i in input_idx:
        n_conn = max(1, int(len(hidden_idx) * input_to_hidden_density))
        targets = rng.choice(hidden_idx, size=n_conn, replace=False)
        for t in targets:
            pre_ih.append(i)
            post_ih.append(int(t))
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
            pre_hh.append(i)
            post_hh.append(int(t))
            w_hh_list.append(base_w + rng.normal(0, base_w * 0.2))
    w_hh = np.clip(np.asarray(w_hh_list, dtype=np.float32), 0.01, None)

    # Hidden -> motor (plastic)
    pre_hm, post_hm = [], []
    w_hm_list = []
    for i in hidden_idx:
        is_inh = i in hidden_inh_idx
        n_conn = max(1, int(len(motor_idx) * hidden_to_motor_density))
        targets = rng.choice(motor_idx, size=n_conn, replace=False)
        base_w = hidden_inh_weight if is_inh else hidden_to_motor_weight
        for t in targets:
            pre_hm.append(i)
            post_hm.append(int(t))
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
            "initial_weights": w_hm, "plastic": True,
            "conn_type": "MIXED", "count": len(pre_hm),
        },
        "layout": {
            "sensor_idx": sensor_idx,
            "goal_context_idx": goal_context_idx,
            "input_idx": input_idx,
            "hidden_exc_idx": hidden_exc_idx,
            "hidden_inh_idx": hidden_inh_idx,
            "hidden_idx": hidden_idx,
            "motor_idx": motor_idx,
        },
    }
    return core_cfg, plan


def run_g8_episode(
    out_path,
    seed,
    n_steps=600,
    grid_size=8,
    start_pos=(1, 1),
    goal_pos=(6, 6),
    goal_schedule=None,
    negative_reward_rule="B",
    learning_rate=0.01,
    w_max=3.0,
    lr_schedule="decay_after_goal",
    lr_decay_factor=0.25,
    epsilon_start=0.1,
    epsilon_end=0.0,
    epsilon_decay_steps=150,
    reset_epsilon_on_goal_change=True,
    goal_context_enabled=True,   # If False, goal-context is zero-ed out
                                 # (ablation control to replicate G6/G7 behavior)
    verbose=True,
):
    """Run a G8 episode with optional goal-context channel.

    Set goal_context_enabled=False to ablate the PFC-like signal and
    replicate G6/G7 behavior for comparison.
    """
    import cupy as cp

    core_cfg, plan = _build_g8_plan(seed=seed)
    bridge = SimulationBridge(
        core_config=core_cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    bridge.strict_step_errors = True

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
        NeuronGroup(name="sensor", role=NeuronGroupRole.INPUT.name,
                    neuron_indices=layout["sensor_idx"]),
        NeuronGroup(name="goal_context", role=NeuronGroupRole.INPUT.name,
                    neuron_indices=layout["goal_context_idx"]),
        NeuronGroup(name="hidden", role=NeuronGroupRole.HIDDEN.name,
                    neuron_indices=layout["hidden_idx"]),
        NeuronGroup(name="motor", role=NeuronGroupRole.OUTPUT.name,
                    neuron_indices=layout["motor_idx"]),
    ]
    exp_cfg.readout = ReadoutConfig(
        rate_window_ms=100.0, spike_count_window_ms=100.0,
        rate_group_names=["motor"],
    )
    exp_cfg.phases = [ExperimentPhase(
        name="g8", phase_type=ExperimentPhaseType.TRAINING.name,
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
    hidden_idx_cp = cp.asarray(layout["hidden_idx"], dtype=cp.int32)
    n_hidden = len(layout["hidden_idx"])
    n_motor = len(layout["motor_idx"])

    # Locate hidden->motor synapses.
    coo = bridge.cp_connections.tocoo(copy=False)
    pre_h = cp.asnumpy(coo.row)
    post_h = cp.asnumpy(coo.col)
    hidden_set = set(layout["hidden_idx"])
    motor_set = set(layout["motor_idx"])
    motor_base = layout["motor_idx"][0]
    i2m_mask = np.array(
        [(int(p) in hidden_set) and (int(q) in motor_set)
         for p, q in zip(pre_h, post_h)],
        dtype=np.bool_,
    )
    i2m_flat_indices = np.where(i2m_mask)[0].astype(np.int64)
    i2m_pre_global = pre_h[i2m_mask].astype(np.int64)
    hidden_global_to_local = {g: i for i, g in enumerate(layout["hidden_idx"])}
    i2m_pre_local_np = np.array(
        [hidden_global_to_local[int(p)] for p in i2m_pre_global],
        dtype=np.int64,
    )
    i2m_post_local_np = (post_h[i2m_mask] - motor_base).astype(np.int64)

    i2m_flat_indices_cp = cp.asarray(i2m_flat_indices)
    n_plastic = int(i2m_flat_indices.size)
    if verbose:
        print(f"[g8 seed={seed}] {n_plastic} hidden->motor synapses  "
              f"goal_context_enabled={goal_context_enabled}", flush=True)

    initial_data = cp.asnumpy(bridge.cp_connections.data).copy()

    x, y = start_pos
    if goal_schedule is None:
        goal_schedule_sorted = [(0, tuple(goal_pos))]
    else:
        goal_schedule_sorted = sorted(
            [(int(s), tuple(g)) for s, g in goal_schedule], key=lambda t: t[0]
        )
    current_schedule_idx = 0
    gx, gy = goal_schedule_sorted[0][1]
    goal_change_steps = []
    epsilon_schedule_origin = 0

    def manhattan(px, py, goal_x=None, goal_y=None):
        gxi = gx if goal_x is None else goal_x
        gyi = gy if goal_y is None else goal_y
        return abs(px - gxi) + abs(py - gyi)

    trajectory = [(x, y)]
    goal_log = [(gx, gy)]
    motor_counts_log = []
    action_log = []
    reward_log = []
    distance_log = [manhattan(x, y)]
    first_goal_step = None
    n_sensor_half = len(layout["sensor_idx"]) // 2
    n_goal_half = len(layout["goal_context_idx"]) // 2

    t0 = time.time()
    for step in range(n_steps):
        # Advance goal schedule
        while (current_schedule_idx + 1 < len(goal_schedule_sorted)
               and step >= goal_schedule_sorted[current_schedule_idx + 1][0]):
            current_schedule_idx += 1
            gx, gy = goal_schedule_sorted[current_schedule_idx][1]
            goal_change_steps.append(step)
            first_goal_step = None
            if reset_epsilon_on_goal_change:
                epsilon_schedule_origin = step
            if verbose:
                print(f"[g8 seed={seed}] step {step}: GOAL CHANGED to ({gx}, {gy})",
                      flush=True)

        # LR schedule
        if lr_schedule == "decay_after_goal" and first_goal_step is not None:
            effective_lr = learning_rate * lr_decay_factor
        elif lr_schedule == "inverse_sqrt":
            effective_lr = learning_rate / max(1.0, float(max(step - 100, 1)) ** 0.5)
        else:
            effective_lr = learning_rate

        dist_before = manhattan(x, y)

        # Build dual-channel input rates: [sensor_rates ; goal_context_rates]
        sensor_rates = _position_to_rates_2d(x, y, n_sensor_half, grid_size)
        goal_rates = _goal_to_context_rates(
            gx, gy, n_goal_half, grid_size, enabled=goal_context_enabled,
        )
        all_rates = np.concatenate([sensor_rates, goal_rates])
        pat = StimulusPattern(
            pattern_type=StimulusPatternType.RATE_VECTOR_POISSON.name,
            spike_current_pA=1000.0, spike_duration_ms=2.0,
            rate_vector_hz=[float(r) for r in all_rates],
        )
        ch = StimulusChannel(
            name="dual_input", pattern=pat,
            target_neuron_indices=layout["input_idx"],
            onset_ms=0.0, duration_ms=STIMULUS_MS, enabled=True,
        )
        engine.stimulus_manager.cleanup()
        engine.stimulus_manager.initialize([ch], engine.group_manager, cp)
        engine.phase_start_ms = bridge.runtime_state.current_time_ms

        motor_counts = np.zeros(n_motor, dtype=np.int32)
        hidden_counts = np.zeros(n_hidden, dtype=np.int32)

        for s in range(n_stim_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * dt
            if readout_start_step <= s < readout_end_step:
                motor_counts += bridge.cp_firing_states[motor_idx_cp].get().astype(np.int32)
                hidden_counts += bridge.cp_firing_states[hidden_idx_cp].get().astype(np.int32)

        motor_counts_log.append(motor_counts.tolist())

        # Epsilon-greedy action selection
        steps_since_epsilon_reset = step - epsilon_schedule_origin
        if epsilon_decay_steps > 0:
            frac = min(1.0, steps_since_epsilon_reset / epsilon_decay_steps)
            epsilon = epsilon_start * (1.0 - frac) + epsilon_end * frac
        else:
            epsilon = epsilon_end
        explore_rng = np.random.default_rng(seed * 7919 + step * 2)
        if explore_rng.random() < epsilon:
            action = int(explore_rng.integers(0, n_motor))
            explore_flag = True
        elif motor_counts.sum() > 0:
            action = int(np.argmax(motor_counts))
            explore_flag = False
        else:
            action = int(np.random.default_rng(seed * 10_000 + step).integers(0, n_motor))
            explore_flag = True

        action_log.append(action)
        dx, dy = ACTION_DELTAS[action]
        new_x = int(np.clip(x + dx, 0, grid_size - 1))
        new_y = int(np.clip(y + dy, 0, grid_size - 1))
        dist_after = manhattan(new_x, new_y)
        x, y = new_x, new_y
        trajectory.append((x, y))
        goal_log.append((gx, gy))
        distance_log.append(dist_after)

        if dist_after < dist_before:
            reward = 1
        elif dist_after > dist_before:
            reward = -1
        else:
            reward = 0
        reward_log.append(reward)

        if dist_after == 0 and first_goal_step is None:
            first_goal_step = step

        # Perceptron delta (4-motor, same as G6)
        if reward != 0 and hidden_counts.sum() > 0:
            h_act = hidden_counts.astype(np.float32) / max(hidden_counts.max(), 1)
            if reward > 0:
                direction_per_syn = np.where(
                    i2m_post_local_np == action, 1.0, -1.0
                ).astype(np.float32)
                delta_np = effective_lr * h_act[i2m_pre_local_np] * direction_per_syn
            elif negative_reward_rule == "C":
                want_n = 1 if gy > y else 0
                want_s = 1 if gy < y else 0
                want_e = 1 if gx > x else 0
                want_w = 1 if gx < x else 0
                correct_motor_flags = np.array([want_n, want_e, want_s, want_w],
                                               dtype=np.float32)
                n_correct = correct_motor_flags.sum()
                if n_correct == 0:
                    direction_per_syn = None
                else:
                    n_wrong = n_motor - n_correct
                    per_motor = np.where(
                        correct_motor_flags > 0, 1.0 / n_correct, -1.0 / n_wrong
                    ).astype(np.float32)
                    direction_per_syn = per_motor[i2m_post_local_np]
                    delta_np = effective_lr * h_act[i2m_pre_local_np] * direction_per_syn
            else:
                direction_per_syn = np.where(
                    i2m_post_local_np == action, -1.0, 1.0 / (n_motor - 1)
                ).astype(np.float32)
                delta_np = effective_lr * h_act[i2m_pre_local_np] * direction_per_syn

            if direction_per_syn is not None:
                delta_cp = cp.asarray(delta_np)
                bridge.cp_connections.data[i2m_flat_indices_cp] += delta_cp
                w_slice = bridge.cp_connections.data[i2m_flat_indices_cp]
                cp.clip(w_slice, 0.0, w_max, out=w_slice)
                bridge.cp_connections.data[i2m_flat_indices_cp] = w_slice

        if verbose and (step + 1) % 100 == 0:
            recent_dist = float(np.mean(distance_log[-100:]))
            w_all = cp.asnumpy(bridge.cp_connections.data)
            w_plastic = w_all[i2m_flat_indices]
            print(
                f"[g8 seed={seed}] step {step+1}/{n_steps}  pos=({x},{y})  "
                f"goal=({gx},{gy})  recent_dist={recent_dist:.2f}  "
                f"W=[{w_plastic.min():.2f},{w_plastic.max():.2f}] "
                f"mean={w_plastic.mean():.2f}  lr={effective_lr:.4f}",
                flush=True,
            )

    elapsed = time.time() - t0

    final_data = cp.asnumpy(bridge.cp_connections.data)
    non_plastic_mask = ~i2m_mask
    reservoir_drift = float(
        np.abs(final_data[non_plastic_mask] - initial_data[non_plastic_mask]).max()
    )

    dist_arr = np.asarray(distance_log[1:])
    q = len(dist_arr) // 4
    quarters = [float(dist_arr[i*q:(i+1)*q].mean()) for i in range(4)]

    phase_stats = []
    phase_boundaries = [0] + goal_change_steps + [n_steps]
    for phase_idx in range(len(phase_boundaries) - 1):
        p_start = phase_boundaries[phase_idx]
        p_end = phase_boundaries[phase_idx + 1]
        p_dist = dist_arr[p_start:p_end]
        p_actions = action_log[p_start:p_end]
        if len(p_dist) == 0:
            continue
        p_goal = goal_log[p_start + 1] if p_start + 1 < len(goal_log) else goal_log[-1]
        phase_stats.append({
            "phase": phase_idx,
            "step_start": p_start,
            "step_end": p_end,
            "goal": list(p_goal),
            "mean_distance": float(p_dist.mean()),
            "final_quarter_mean_distance": float(
                p_dist[len(p_dist) * 3 // 4:].mean()
            ) if len(p_dist) >= 4 else float(p_dist.mean()),
            "n_steps_at_goal": int((p_dist == 0).sum()),
            "n_steps": len(p_dist),
            "action_counts": [int((np.asarray(p_actions) == a).sum()) for a in range(n_motor)],
        })

    results = {
        "seed": seed, "n_steps": n_steps, "grid_size": grid_size,
        "start_pos": list(start_pos), "goal_pos": list(goal_pos),
        "goal_schedule": [[s, list(g)] for s, g in goal_schedule_sorted],
        "goal_change_steps": goal_change_steps,
        "goal_context_enabled": goal_context_enabled,
        "phase_stats": phase_stats,
        "learning_rate": learning_rate, "lr_schedule": lr_schedule,
        "lr_decay_factor": lr_decay_factor,
        "epsilon_start": epsilon_start, "epsilon_end": epsilon_end,
        "epsilon_decay_steps": epsilon_decay_steps,
        "reset_epsilon_on_goal_change": reset_epsilon_on_goal_change,
        "negative_reward_rule": negative_reward_rule,
        "first_goal_step": first_goal_step,
        "w_max": w_max, "n_plastic_synapses": n_plastic,
        "reservoir_weight_drift_max": reservoir_drift,
        "trajectory": trajectory,
        "goal_log": goal_log,
        "motor_counts": motor_counts_log,
        "action_log": action_log,
        "reward_log": reward_log,
        "distance_log": distance_log,
        "mean_distance_overall": float(dist_arr.mean()),
        "mean_distance_quarters": quarters,
        "distance_quarter_improvement": float(quarters[0] - quarters[-1]),
        "n_steps_at_goal": int((dist_arr == 0).sum()),
        "action_counts": [int((np.asarray(action_log) == a).sum()) for a in range(n_motor)],
        "elapsed_seconds": elapsed,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(
            f"[g8 seed={seed}] done: mean_dist={results['mean_distance_overall']:.2f}  "
            f"quarters={[round(q, 2) for q in quarters]}  "
            f"at_goal={results['n_steps_at_goal']}  actions={results['action_counts']}  "
            f"reservoir_drift={reservoir_drift:.2e}  {elapsed:.1f}s",
            flush=True,
        )
    return results
