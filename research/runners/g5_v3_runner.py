"""G5.v3: Signed-perceptron sensorimotor learning, reward-driven.

G5.v2 hit the sim's unsigned-eligibility ceiling: negative reward uniformly
depressed all recently-plastic synapses, so the agent got stuck. v3 bypasses
sim plasticity entirely — sim is a pure forward pass, runner applies a
signed perceptron delta to hidden->motor weights per step, directly on
`cp_connections.data`.

Per step:
  1. Encode position → Poisson rates.
  2. Present 150 ms stimulus; read hidden spike counts + motor spike counts.
  3. action = argmax(motor counts).
  4. Update world: x := clip(x ± 1, 0, n_positions-1).
  5. reward = sign(dist_before - dist_after).
  6. If reward != 0:
        target = chosen if reward>0 else other
        ΔW[hidden_i → motor_target]  += lr * hidden_active[i]
        ΔW[hidden_i → motor_!target] -= lr * hidden_active[i]
        clip weights to [0, w_max].
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
READOUT_START_MS = 50.0
READOUT_END_MS = 150.0


def _build_g5v3_plan(
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
    """Same topology as G5.v2, but all sim plasticity is OFF.

    The plastic flag on hidden->motor is kept True as documentation — if a
    later iteration wants to layer STDP on top, the mask is already correct.
    Since `enable_stdp=False` here, the flag has no effect on sim behaviour.
    """
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

    # All sim plasticity OFF. Learning is entirely external.
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

    # Hidden -> motor (the LEARNED layer — updated externally by the runner)
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
            "initial_weights": w_hm, "plastic": True,
            "conn_type": "MIXED", "count": len(pre_hm),
        },
        "layout": {
            "input_idx": input_idx, "hidden_exc_idx": hidden_exc_idx,
            "hidden_inh_idx": hidden_inh_idx, "hidden_idx": hidden_idx,
            "motor_idx": motor_idx,
        },
    }
    return core_cfg, plan


def run_g5_v3_episode(
    out_path,
    seed,
    n_steps=400,
    n_positions=16,
    start_position=8,
    goal_position=12,
    learning_rate=0.01,
    w_max=3.0,
    lr_schedule="constant",    # "constant" | "inverse_sqrt" | "decay_after_goal"
    lr_schedule_warmup=100,    # for inverse_sqrt: lr / sqrt(max(step-warmup, 1))
    lr_decay_factor=0.25,      # for decay_after_goal: multiplier applied after first goal reached
    verbose=True,
):
    import cupy as cp
    import cupyx.scipy.sparse as csp

    core_cfg, plan = _build_g5v3_plan(seed=seed)
    bridge = SimulationBridge(
        core_config=core_cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    # Loud failure on any step-level error — this is a research run, not the UI.
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
        NeuronGroup(name="input", role=NeuronGroupRole.INPUT.name,
                    neuron_indices=layout["input_idx"]),
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
        name="g5v3", phase_type=ExperimentPhaseType.TRAINING.name,
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

    # Locate hidden->motor synapses in cp_connections.data.
    # tocoo preserves CSR internal order (row-major by pre).
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
    i2m_pre_np = pre_h[i2m_mask].astype(np.int64)  # global neuron index
    # Map global pre index -> local hidden index (0..n_hidden-1)
    hidden_global_to_local = {g: i for i, g in enumerate(layout["hidden_idx"])}
    i2m_pre_local_np = np.array(
        [hidden_global_to_local[int(p)] for p in i2m_pre_np],
        dtype=np.int64,
    )
    i2m_post_local_np = (post_h[i2m_mask] - motor_base).astype(np.int64)

    i2m_flat_indices_cp = cp.asarray(i2m_flat_indices)
    i2m_pre_local_cp = cp.asarray(i2m_pre_local_np)
    i2m_post_local_cp = cp.asarray(i2m_post_local_np)
    n_plastic = int(i2m_flat_indices.size)
    if verbose:
        print(f"[g5v3 seed={seed}] {n_plastic} hidden->motor synapses identified",
              flush=True)

    # Snapshot initial reservoir weights to verify they don't change.
    initial_data = cp.asnumpy(bridge.cp_connections.data).copy()

    x = int(start_position)
    trajectory = [x]
    motor_counts_log = []
    reward_log = []
    distance_log = [abs(x - goal_position)]
    first_goal_step = None   # for decay_after_goal schedule

    t0 = time.time()
    for step in range(n_steps):
        dist_before = abs(x - goal_position)

        # Compute effective learning rate for this step.
        if lr_schedule == "inverse_sqrt":
            effective_lr = learning_rate / max(
                1.0, float(max(step - lr_schedule_warmup, 1)) ** 0.5
            )
        elif lr_schedule == "decay_after_goal" and first_goal_step is not None:
            effective_lr = learning_rate * lr_decay_factor
        else:
            effective_lr = learning_rate

        # Present stimulus
        rates = _position_to_rates(
            x, n_positions=n_positions, n_input=len(layout["input_idx"])
        )
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

        motor_counts = np.zeros(n_motor, dtype=np.int32)
        hidden_counts = np.zeros(n_hidden, dtype=np.int32)

        for s in range(n_stim_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = (
                bridge.runtime_state.current_time_step * dt
            )
            if readout_start_step <= s < readout_end_step:
                fired_motor = bridge.cp_firing_states[motor_idx_cp].get().astype(np.int32)
                motor_counts += fired_motor
                fired_hidden = bridge.cp_firing_states[hidden_idx_cp].get().astype(np.int32)
                hidden_counts += fired_hidden

        motor_counts_log.append(motor_counts.tolist())

        # Action + environment update
        if motor_counts.sum() > 0:
            action = int(np.argmax(motor_counts))
        else:
            # Silent — random choice so we don't always go left
            # (RNG seeded from main seed; deterministic).
            action = int(np.random.default_rng(seed * 10_000 + step).integers(0, 2))
        new_x = int(np.clip(x + (1 if action == 1 else -1), 0, n_positions - 1))
        dist_after = abs(new_x - goal_position)
        x = new_x
        trajectory.append(x)
        distance_log.append(dist_after)

        # Signed reward
        if dist_after < dist_before:
            reward = 1
        elif dist_after > dist_before:
            reward = -1
        else:
            reward = 0
        reward_log.append(reward)

        # Track first-goal hit for decay_after_goal schedule.
        if dist_after == 0 and first_goal_step is None:
            first_goal_step = step

        # Perceptron delta on hidden->motor
        if reward != 0 and hidden_counts.sum() > 0:
            target = action if reward > 0 else (1 - action)
            # Normalise hidden activity so lr is scale-stable per step.
            h_act = hidden_counts.astype(np.float32) / max(hidden_counts.max(), 1)
            # +lr*h_act on synapses whose post matches target; -lr*h_act elsewhere.
            delta_np = effective_lr * h_act[i2m_pre_local_np] * np.where(
                i2m_post_local_np == target, 1.0, -1.0
            ).astype(np.float32)
            delta_cp = cp.asarray(delta_np)
            bridge.cp_connections.data[i2m_flat_indices_cp] += delta_cp
            # Clip only the plastic slice (cheap) — reservoir weights untouched.
            w_slice = bridge.cp_connections.data[i2m_flat_indices_cp]
            cp.clip(w_slice, 0.0, w_max, out=w_slice)
            bridge.cp_connections.data[i2m_flat_indices_cp] = w_slice

        if verbose and (step + 1) % 50 == 0:
            recent_dist = float(np.mean(distance_log[-50:]))
            w_all = cp.asnumpy(bridge.cp_connections.data)
            w_plastic = w_all[i2m_flat_indices]
            print(
                f"[g5v3 seed={seed}] step {step+1}/{n_steps}  x={x}  "
                f"recent_dist={recent_dist:.2f}  "
                f"W plastic range=[{w_plastic.min():.2f},{w_plastic.max():.2f}] "
                f"mean={w_plastic.mean():.2f}",
                flush=True,
            )

    elapsed = time.time() - t0

    # Verify reservoir weights untouched
    final_data = cp.asnumpy(bridge.cp_connections.data)
    non_plastic_mask = ~i2m_mask
    reservoir_drift = float(np.abs(final_data[non_plastic_mask] - initial_data[non_plastic_mask]).max())

    traj_arr = np.asarray(trajectory[1:])
    dist_arr = np.asarray(distance_log[1:])
    q = len(dist_arr) // 4
    quarters = [float(dist_arr[i*q:(i+1)*q].mean()) for i in range(4)]

    results = {
        "seed": seed, "n_steps": n_steps, "n_positions": n_positions,
        "start_position": start_position, "goal_position": goal_position,
        "learning_rate": learning_rate, "lr_schedule": lr_schedule,
        "lr_schedule_warmup": lr_schedule_warmup,
        "lr_decay_factor": lr_decay_factor,
        "first_goal_step": first_goal_step,
        "w_max": w_max,
        "n_plastic_synapses": n_plastic,
        "reservoir_weight_drift_max": reservoir_drift,
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
        print(
            f"[g5v3 seed={seed}] done: mean_dist={results['mean_distance_overall']:.2f}  "
            f"quarters={[round(q, 2) for q in quarters]}  at_goal={results['n_steps_at_goal']}  "
            f"reservoir_drift={reservoir_drift:.2e}  {elapsed:.1f}s",
            flush=True,
        )
    return results
