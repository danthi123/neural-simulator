"""G9: sim-native R-STDP sensorimotor learning.

This is Session B.2's response to the G8 probe finding that H2 (architectural
limit of runner-side cliff-edged argmax + per-step specialization) is the
binding constraint on moving-goal readaptation.

Changes from G6/G8:

  1. Sim-native learning. STDP + reward modulation enabled in the bridge.
     The runner sets `current_reward_signal` per step based on Manhattan
     distance change and lets the sim's own eligibility trace (tau=500 ms)
     handle credit assignment. Plastic-mask keeps the reservoir frozen;
     only hidden->motor synapses participate in STDP + reward.

  2. Action selection. `action_selection="argmax"` (G6/G8-compatible) or
     `"first_spike"` (biology-canonical WTA). First-spike reads the earliest
     firing time across motor neurons during the readout window. The motor
     that spikes first wins (others would be suppressed by lateral
     inhibition in a real circuit; here we just pick via min-time).

  3. No runner-side weight update. The sim's three-factor learning
     (Fremaux & Gerstner 2016) handles all plasticity. The runner only
     (a) delivers the stimulus, (b) reads out firing counts, (c) picks
     the action, (d) sets the reward signal for the next step.

Biology notes:

  - Three-factor learning: STDP locally tags synapses with eligibility
    (pre-post co-firing), and a global third factor (here, a scalar reward
    proxy for phasic dopamine) gates whether the tagged synapses potentiate
    or depress. This is the canonical cortico-striatal / cortico-cortical
    reinforcement mechanism (Schultz 1998; Reynolds & Wickens 2002).

  - First-spike WTA: motor cortex / basal ganglia action selection typically
    takes ~20-50 ms. Lateral inhibition (GPi -> thalamus disinhibition, or
    local M1 interneurons) silences the losers before they can fire. Picking
    the earliest-firing motor neuron approximates this winner-take-all
    dynamics without modeling the inhibitory circuit explicitly.

  - Eligibility tau: default 1000 ms, we tighten to 500 ms to match
    dopamine kinetics (phasic DA bursts last ~100-300 ms in vivo, but the
    downstream plasticity window extends to ~500 ms; Yagishita et al. 2014).

If this runner succeeds on the G7 moving-goal task (seeds 42/43/44 Phase 2
finalQ < 3.0 in >= 2/3 seeds), the architectural limit identified in G7/G8
is dissolved and the sim has full biology-compliant sensorimotor learning.
If it fails, Session C (neuromodulatory synaptic-gain branch) is the next
investigation.
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

# Cardinal movements, in (dx, dy). Motor-neuron index: 0=N, 1=E, 2=S, 3=W.
ACTION_DELTAS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def _gaussian_tuned_rates(value, n_neurons, n_positions, sigma, rate_peak, rate_floor):
    rates = np.zeros(n_neurons, dtype=np.float32)
    for i in range(n_neurons):
        p_i = (i / (n_neurons - 1)) * (n_positions - 1) if n_neurons > 1 else 0.0
        rates[i] = rate_peak * np.exp(-((value - p_i) / sigma) ** 2) + rate_floor
    return rates


def _position_to_rates_2d(x, y, n_sensor_half, n_positions,
                          rate_peak=30.0, rate_floor=1.0, sigma=1.5):
    rx = _gaussian_tuned_rates(x, n_sensor_half, n_positions, sigma, rate_peak, rate_floor)
    ry = _gaussian_tuned_rates(y, n_sensor_half, n_positions, sigma, rate_peak, rate_floor)
    return np.concatenate([rx, ry])


def _build_g9_plan(
    seed,
    n_input=64,
    n_hidden_exc=160,
    n_hidden_inh=40,
    n_motor=4,
    input_to_hidden_density=0.5,
    hidden_to_hidden_density=0.1,
    hidden_to_motor_density=0.5,
    input_to_hidden_weight=1.5,
    hidden_exc_weight=0.3,
    hidden_inh_weight=0.8,
    hidden_to_motor_weight=1.0,
    stdp_w_min=0.0,
    stdp_w_max=3.0,
    reward_learning_rate=0.01,
    reward_eligibility_tau_ms=500.0,
    enable_neuromod_gating=False,          # Session C: fast gain modulation (DEPRECATED)
    nm_configs=None,                       # Session E.1: list[NeuromodulatorConfig] — preferred path
    neuromod_tau_ms=100.0,
    neuromod_strength=0.5,
):
    """Build G9 wiring plan with sim-native STDP + reward modulation enabled.

    The plastic mask (from inject_explicit_wiring honoring plastic=True/False
    flags in each group) limits STDP events to the hidden->motor layer. The
    reservoir (input->hidden, hidden->hidden) stays frozen.
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

    # Sim-native learning: STDP + reward modulation. Plastic mask (set by
    # inject_explicit_wiring) limits STDP to hidden->motor.
    core_cfg.enable_stdp = True
    core_cfg.enable_reward_modulation = True
    core_cfg.enable_hebbian_learning = False
    core_cfg.enable_short_term_plasticity = False
    core_cfg.enable_homeostasis = False
    core_cfg.enable_structural_plasticity = False
    core_cfg.enable_watts_strogatz = False

    # STDP parameters — defaults are Bi & Poo 1998.
    core_cfg.stdp_a_plus = 0.012
    core_cfg.stdp_a_minus = 0.01
    core_cfg.stdp_tau_plus_ms = 20.0
    core_cfg.stdp_tau_minus_ms = 20.0
    core_cfg.stdp_w_min = stdp_w_min
    core_cfg.stdp_w_max = stdp_w_max

    # Reward modulation parameters
    core_cfg.reward_learning_rate = reward_learning_rate
    core_cfg.reward_eligibility_tau_ms = reward_eligibility_tau_ms
    core_cfg.reward_baseline = 0.0
    core_cfg.current_reward_signal = 0.0

    # Session C: neuromodulatory gain gating (opt-in, DEPRECATED — use nm_configs)
    core_cfg.enable_neuromod_gating = enable_neuromod_gating
    core_cfg.neuromod_tau_ms = neuromod_tau_ms
    core_cfg.neuromod_strength = neuromod_strength

    # Session E.1: neuromodulator subsystem (preferred path)
    if nm_configs:
        core_cfg.enable_neuromodulator_subsystem = True
        core_cfg.neuromodulators = list(nm_configs)
    else:
        core_cfg.enable_neuromodulator_subsystem = False
        core_cfg.neuromodulators = []

    core_cfg.propagation_strength = 1.0
    core_cfg.inhibitory_propagation_strength = 1.0
    core_cfg.ou_std_current_pA = 60.0

    rng = np.random.default_rng(seed)

    # Input -> hidden (frozen reservoir)
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

    # Hidden recurrent (frozen reservoir)
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

    # Hidden -> motor (PLASTIC: subject to STDP + reward modulation)
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
            "initial_weights": w_ih, "plastic": False,  # Reservoir frozen
            "conn_type": "E_TO_MIX", "count": len(pre_ih),
        },
        "hidden_recurrent": {
            "pre_indices": pre_hh, "post_indices": post_hh,
            "initial_weights": w_hh, "plastic": False,  # Reservoir frozen
            "conn_type": "MIXED", "count": len(pre_hh),
        },
        "hidden_to_motor": {
            "pre_indices": pre_hm, "post_indices": post_hm,
            "initial_weights": w_hm, "plastic": True,   # Plastic via STDP+reward
            "conn_type": "MIXED", "count": len(pre_hm),
        },
        "layout": {
            "input_idx": input_idx,
            "hidden_exc_idx": hidden_exc_idx,
            "hidden_inh_idx": hidden_inh_idx,
            "hidden_idx": hidden_idx,
            "motor_idx": motor_idx,
        },
    }
    return core_cfg, plan


def _select_action_first_spike(bridge, motor_idx_cp, n_motor, n_stim_steps,
                               readout_start_step, readout_end_step, dt, cp):
    """Run the trial and return (action, motor_counts, hidden_counts, first_spike_times).

    Action is the motor neuron that fires FIRST within the readout window,
    after stimulus onset. Ties broken by argmax over total counts.
    """
    # We have to delegate the stepping here because we want to record first-
    # spike times as they happen. Returns all the useful state.
    motor_counts = np.zeros(n_motor, dtype=np.int32)
    first_spike_step = np.full(n_motor, -1, dtype=np.int32)
    return motor_counts, first_spike_step


def run_g9_episode(
    out_path,
    seed,
    n_steps=600,
    grid_size=8,
    start_pos=(1, 1),
    goal_pos=(6, 6),
    goal_schedule=None,
    learning_rate=0.01,                 # reward_learning_rate for sim
    reward_eligibility_tau_ms=500.0,
    reward_hold_steps=10,               # hold reward signal for N steps after action
    stdp_w_max=3.0,
    action_selection="argmax",          # "argmax", "first_spike", or
                                        # "proportional" (Session G v5:
                                        # sample motor with probability
                                        # proportional to its readout-window
                                        # spike count + 1; biologically
                                        # equivalent to rate-coded WTA where
                                        # firing rate maps to selection
                                        # probability via softmax-like rule).
    enable_neuromod_gating=False,       # Session C: fast neuromod gain (DEPRECATED)
    neuromod_tau_ms=100.0,
    neuromod_strength=0.5,
    nm_configs=None,                    # Session E.1: list[NeuromodulatorConfig]
                                        # Preferred replacement for enable_neuromod_gating.
                                        # When non-empty, the bridge runs the
                                        # full neuromodulator subsystem.
    n_hidden_exc=160,                   # Route C (E.2.5): reservoir size knobs.
    n_hidden_inh=40,                    # Default 200 hidden total (G9 baseline).
    hidden_to_hidden_density=0.1,       # Tighten when scaling to 5000+ neurons
    input_to_hidden_density=0.5,        # so synapse count stays tractable.
    eval_random_starts=0,               # Session D.A.3: number of random-start
                                        # eval episodes to run AFTER training.
                                        # 0 = no eval (backward compat).
    eval_steps_per_start=30,            # steps per eval episode
    motor_exploration_rate_hz=0.0,      # Session G: Poisson rate at which each
                                        # motor neuron receives spurious spike
                                        # input during the stimulus window.
                                        # Forces all motors to fire occasionally
                                        # so silent motors can acquire eligibility
                                        # traces. 0 disables (backward compat);
                                        # 5-15 Hz typical (corresponds to ~0.5-1
                                        # spurious spikes/motor/100ms readout).
    motor_exploration_current_pA=1000.0,  # Per-spike current amplitude
    motor_exploration_spike_ms=2.0,       # Duration of each spurious spike pulse
    positive_only_reward=False,         # Session G v3: when True, dist_after >
                                        # dist_before emits reward=0 (no
                                        # punishment) instead of -1. Avoids the
                                        # action-blind eligibility-depression
                                        # problem where E winning + going wrong
                                        # way would depress W's noise-driven
                                        # eligibility traces.
    action_attribution_eligibility=False,  # Session G v4: when True, after
                                        # picking action `a`, zero eligibility
                                        # for hidden->motor synapses targeting
                                        # any motor != a, before the
                                        # reward-hold steps. Reward then
                                        # selectively updates only the chosen
                                        # motor's synapses — a runner-side
                                        # implementation of selective DA /
                                        # lateral-inhibition action attribution.
    verbose=True,
):
    """Run a G9 episode with sim-native R-STDP learning.

    The sim handles all weight updates via its STDP + three-factor reward
    modulation path. Runner only sets the reward signal based on distance
    change and picks actions.

    action_selection:
        "argmax"      — picks motor with highest spike count in [50, 150] ms
                        (same as G6/G8, for direct comparison)
        "first_spike" — picks motor whose first spike during the readout
                        window happened earliest (biology-canonical WTA)
    """
    import cupy as cp

    assert action_selection in ("argmax", "first_spike", "proportional"), (
        f"action_selection must be 'argmax', 'first_spike', or 'proportional', "
        f"got {action_selection}"
    )

    core_cfg, plan = _build_g9_plan(
        seed=seed,
        reward_learning_rate=learning_rate,
        reward_eligibility_tau_ms=reward_eligibility_tau_ms,
        stdp_w_max=stdp_w_max,
        enable_neuromod_gating=enable_neuromod_gating,
        neuromod_tau_ms=neuromod_tau_ms,
        neuromod_strength=neuromod_strength,
        nm_configs=nm_configs,
        n_hidden_exc=n_hidden_exc,
        n_hidden_inh=n_hidden_inh,
        hidden_to_hidden_density=hidden_to_hidden_density,
        input_to_hidden_density=input_to_hidden_density,
    )
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

    # Verify plastic mask is set (should be, since we have plastic=False groups)
    assert bridge.cp_synapse_plastic_mask is not None, (
        "Plastic mask not set — inject_explicit_wiring should have set it "
        "since hidden_recurrent and input_to_hidden are plastic=False."
    )

    # Verify eligibility trace exists (created by _initialize_synaptic_dynamics
    # when enable_reward_modulation=True)
    assert bridge.cp_eligibility_trace is not None, (
        "Eligibility trace not allocated. Check enable_reward_modulation."
    )

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
        name="g9", phase_type=ExperimentPhaseType.TRAINING.name,
        duration_ms=1e9,
    )]
    engine.load_experiment(exp_cfg)
    engine.initialize(cp_traits=bridge.cp_traits, cp_module=cp)
    engine.is_experiment_running = True
    bridge.experiment_engine = engine

    # Register neuron groups with the neuromodulator manager so
    # `scope="group:NAME"` targets work (e.g. NE excitability_drive on motor).
    if bridge.neuromodulator_manager is not None:
        bridge.neuromodulator_manager.set_group_indices({
            "input": layout["input_idx"],
            "hidden": layout["hidden_idx"],
            "hidden_exc": layout["hidden_exc_idx"],
            "hidden_inh": layout["hidden_inh_idx"],
            "motor": layout["motor_idx"],
        })

    dt = core_cfg.dt_ms
    n_stim_steps = int(STIMULUS_MS / dt)
    readout_start_step = int(READOUT_START_MS / dt)
    readout_end_step = int(READOUT_END_MS / dt)

    motor_idx_cp = cp.asarray(layout["motor_idx"], dtype=cp.int32)
    hidden_idx_cp = cp.asarray(layout["hidden_idx"], dtype=cp.int32)
    n_hidden = len(layout["hidden_idx"])
    n_motor = len(layout["motor_idx"])
    n_sensor_half = len(layout["input_idx"]) // 2

    # For diagnostics: track hidden->motor weights over time
    coo = bridge.cp_connections.tocoo(copy=False)
    pre_h = cp.asnumpy(coo.row)
    post_h = cp.asnumpy(coo.col)
    hidden_set = set(layout["hidden_idx"])
    motor_set = set(layout["motor_idx"])
    i2m_mask = np.array(
        [(int(p) in hidden_set) and (int(q) in motor_set)
         for p, q in zip(pre_h, post_h)],
        dtype=np.bool_,
    )
    i2m_flat_indices = np.where(i2m_mask)[0].astype(np.int64)
    n_plastic = int(i2m_flat_indices.size)
    initial_data = cp.asnumpy(bridge.cp_connections.data).copy()

    # Per-motor synapse index arrays for action_attribution_eligibility.
    # i2m_per_motor[a] = synapse indices targeting motor[a] (cupy int64).
    i2m_per_motor = []
    for m_neuron in layout["motor_idx"]:
        mask_m = i2m_mask & (post_h == m_neuron)
        i2m_per_motor.append(
            cp.asarray(np.where(mask_m)[0], dtype=cp.int64)
        )

    if verbose:
        explore_str = (f"  motor_explore={motor_exploration_rate_hz}Hz"
                       if motor_exploration_rate_hz > 0 else "")
        print(f"[g9 seed={seed}] {n_plastic} plastic hidden->motor synapses  "
              f"action_selection={action_selection}  lr={learning_rate}  "
              f"tau_elig={reward_eligibility_tau_ms}ms{explore_str}", flush=True)

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

    def manhattan(px, py):
        return abs(px - gx) + abs(py - gy)

    trajectory = [(x, y)]
    goal_log = [(gx, gy)]
    motor_counts_log = []
    action_log = []
    reward_log = []
    distance_log = [manhattan(x, y)]
    first_goal_step = None

    t0 = time.time()
    for step in range(n_steps):
        # Advance goal schedule
        while (current_schedule_idx + 1 < len(goal_schedule_sorted)
               and step >= goal_schedule_sorted[current_schedule_idx + 1][0]):
            current_schedule_idx += 1
            gx, gy = goal_schedule_sorted[current_schedule_idx][1]
            goal_change_steps.append(step)
            first_goal_step = None
            if verbose:
                print(f"[g9 seed={seed}] step {step}: GOAL CHANGED to ({gx}, {gy})",
                      flush=True)

        dist_before = manhattan(x, y)

        # Build sensory input
        rates = _position_to_rates_2d(x, y, n_sensor_half, grid_size)
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
        channels_this_step = [ch]
        if motor_exploration_rate_hz > 0.0:
            explore_pat = StimulusPattern(
                pattern_type=StimulusPatternType.POISSON_SPIKE_TRAIN.name,
                poisson_rate_hz=float(motor_exploration_rate_hz),
                spike_current_pA=float(motor_exploration_current_pA),
                spike_duration_ms=float(motor_exploration_spike_ms),
            )
            explore_ch = StimulusChannel(
                name="motor_explore", pattern=explore_pat,
                target_neuron_indices=layout["motor_idx"],
                onset_ms=0.0, duration_ms=STIMULUS_MS, enabled=True,
            )
            channels_this_step.append(explore_ch)
        engine.stimulus_manager.cleanup()
        engine.stimulus_manager.initialize(channels_this_step, engine.group_manager, cp)
        engine.phase_start_ms = bridge.runtime_state.current_time_ms

        # Ensure reward is zero during stimulus integration (so eligibility
        # accumulates cleanly; reward is applied AFTER action resolved).
        bridge.core_config.current_reward_signal = 0.0

        motor_counts = np.zeros(n_motor, dtype=np.int32)
        hidden_counts = np.zeros(n_hidden, dtype=np.int32)
        first_spike_step = np.full(n_motor, -1, dtype=np.int32)

        for s in range(n_stim_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * dt
            if readout_start_step <= s < readout_end_step:
                fired = bridge.cp_firing_states[motor_idx_cp].get().astype(bool)
                motor_counts += fired.astype(np.int32)
                # Track first-spike step for WTA
                new_firing = fired & (first_spike_step == -1)
                first_spike_step[new_firing] = s
                hidden_counts += bridge.cp_firing_states[hidden_idx_cp].get().astype(np.int32)

        motor_counts_log.append(motor_counts.tolist())

        # Action selection
        if action_selection == "first_spike":
            valid = first_spike_step >= 0
            if valid.any():
                masked_times = np.where(valid, first_spike_step,
                                        n_stim_steps + 1)
                action = int(np.argmin(masked_times))
            else:
                action = int(np.random.default_rng(seed * 10_000 + step).integers(0, n_motor))
        elif action_selection == "proportional":
            # Sample motor with probability ∝ (spike_count + 1). The +1 ensures
            # silent motors still have positive selection probability;
            # otherwise zero-count motors would never be chosen and the
            # silent-motor trap re-forms exactly. With +1, a 0-count motor
            # still has 1/(total+n_motor) chance of being selected, giving
            # silent motors a path to acquire reward feedback.
            probs = (motor_counts.astype(np.float64) + 1.0)
            probs = probs / probs.sum()
            rng_step = np.random.default_rng(seed * 10_000 + step)
            action = int(rng_step.choice(n_motor, p=probs))
        else:  # argmax
            if motor_counts.sum() > 0:
                action = int(np.argmax(motor_counts))
            else:
                action = int(np.random.default_rng(seed * 10_000 + step).integers(0, n_motor))

        action_log.append(action)

        # Session G v4: action attribution. Zero eligibility for hidden->motor
        # synapses targeting any non-chosen motor before the reward signal
        # applies. This makes reward selectively update only the chosen
        # motor's synapses — runner-side credit assignment.
        if action_attribution_eligibility and bridge.cp_eligibility_trace is not None:
            for m_idx in range(n_motor):
                if m_idx == action:
                    continue
                bridge.cp_eligibility_trace[i2m_per_motor[m_idx]] = 0.0

        dx, dy = ACTION_DELTAS[action]
        new_x = int(np.clip(x + dx, 0, grid_size - 1))
        new_y = int(np.clip(y + dy, 0, grid_size - 1))
        dist_after = manhattan(new_x, new_y)
        x, y = new_x, new_y
        trajectory.append((x, y))
        goal_log.append((gx, gy))
        distance_log.append(dist_after)

        if dist_after < dist_before:
            reward = 1.0
        elif dist_after > dist_before:
            reward = 0.0 if positive_only_reward else -1.0
        else:
            reward = 0.0
        reward_log.append(float(reward))

        if dist_after == 0 and first_goal_step is None:
            first_goal_step = step

        # Apply reward: hold current_reward_signal for `reward_hold_steps`
        # simulation steps. This gives the sim's reward modulation path
        # (bridge.py:3862-3890) multiple opportunities to apply delta-w from
        # recently-accumulated eligibility.
        if abs(reward) > 0:
            bridge.core_config.current_reward_signal = float(reward)
            for _ in range(reward_hold_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                bridge.runtime_state.current_time_ms = (
                    bridge.runtime_state.current_time_step * dt
                )
            bridge.core_config.current_reward_signal = 0.0

        if verbose and (step + 1) % 100 == 0:
            recent_dist = float(np.mean(distance_log[-100:]))
            w_all = cp.asnumpy(bridge.cp_connections.data)
            w_plastic = w_all[i2m_flat_indices]
            elig_norm = float(cp.linalg.norm(bridge.cp_eligibility_trace).get())
            print(
                f"[g9 seed={seed}] step {step+1}/{n_steps}  pos=({x},{y})  "
                f"goal=({gx},{gy})  recent_dist={recent_dist:.2f}  "
                f"W=[{w_plastic.min():.2f},{w_plastic.max():.2f}] "
                f"mean={w_plastic.mean():.2f}  |elig|={elig_norm:.2f}",
                flush=True,
            )

    elapsed = time.time() - t0

    # -------------------- Session D.A.3: RSG post-training eval --------------------
    # Freeze plastic weights and reward signal; run `eval_random_starts` short
    # episodes, each from a random start position, and record mean-distance
    # over the final 1/3 of each episode. This tests whether the trained
    # hidden->motor mapping is a *controller* (generalizes) or a *trajectory*
    # (memorized one path from start_pos).
    rsg_results = None
    if eval_random_starts > 0:
        from numpy.random import default_rng as _drng
        eval_rng = _drng(seed * 13 + 7)
        # Disable further plasticity: zero reward signal, zero reward lr.
        # (Easier than flipping enable_stdp because weights were already
        # written via sim-native three-factor learning.)
        saved_reward_lr = bridge.core_config.reward_learning_rate
        bridge.core_config.reward_learning_rate = 0.0
        bridge.core_config.current_reward_signal = 0.0

        # Also suppress further STDP weight writes by clearing eligibility
        # and zeroing the STDP amplitudes for the eval phase.
        saved_a_plus = bridge.core_config.stdp_a_plus
        saved_a_minus = bridge.core_config.stdp_a_minus
        bridge.core_config.stdp_a_plus = 0.0
        bridge.core_config.stdp_a_minus = 0.0
        if bridge.cp_eligibility_trace is not None:
            bridge.cp_eligibility_trace.fill(0.0)

        eval_episodes = []
        # Use the final goal in effect at end of training for fair eval
        eval_gx, eval_gy = gx, gy
        for ep in range(eval_random_starts):
            # Pick a random start cell not exactly at the goal
            while True:
                sx = int(eval_rng.integers(0, grid_size))
                sy = int(eval_rng.integers(0, grid_size))
                if (sx, sy) != (eval_gx, eval_gy):
                    break
            ex, ey = sx, sy
            ep_dist = [abs(ex - eval_gx) + abs(ey - eval_gy)]
            for _ in range(eval_steps_per_start):
                rates = _position_to_rates_2d(ex, ey, n_sensor_half, grid_size)
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

                motor_counts_eval = np.zeros(n_motor, dtype=np.int32)
                first_spike_eval = np.full(n_motor, -1, dtype=np.int32)
                for s in range(n_stim_steps):
                    bridge._run_one_simulation_step()
                    bridge.runtime_state.current_time_step += 1
                    bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * dt
                    if readout_start_step <= s < readout_end_step:
                        fired = bridge.cp_firing_states[motor_idx_cp].get().astype(bool)
                        motor_counts_eval += fired.astype(np.int32)
                        new_firing = fired & (first_spike_eval == -1)
                        first_spike_eval[new_firing] = s
                # pick action same way as training
                if action_selection == "first_spike":
                    valid = first_spike_eval >= 0
                    if valid.any():
                        masked_times = np.where(valid, first_spike_eval, n_stim_steps + 1)
                        action = int(np.argmin(masked_times))
                    else:
                        action = int(eval_rng.integers(0, n_motor))
                else:
                    action = int(np.argmax(motor_counts_eval)) if motor_counts_eval.sum() > 0 else int(eval_rng.integers(0, n_motor))
                dxe, dye = ACTION_DELTAS[action]
                ex = int(np.clip(ex + dxe, 0, grid_size - 1))
                ey = int(np.clip(ey + dye, 0, grid_size - 1))
                ep_dist.append(abs(ex - eval_gx) + abs(ey - eval_gy))
            tail = ep_dist[-(max(1, eval_steps_per_start // 3)):]
            eval_episodes.append({
                "start": [sx, sy],
                "goal": [eval_gx, eval_gy],
                "initial_dist": ep_dist[0],
                "final_dist": ep_dist[-1],
                "tail_mean_dist": float(np.mean(tail)),
                "full_traj_mean_dist": float(np.mean(ep_dist)),
            })

        # Restore training config (in case someone re-runs downstream)
        bridge.core_config.reward_learning_rate = saved_reward_lr
        bridge.core_config.stdp_a_plus = saved_a_plus
        bridge.core_config.stdp_a_minus = saved_a_minus

        tail_dists = [e["tail_mean_dist"] for e in eval_episodes]
        # Random-walk baseline approximation: expected Manhattan distance
        # after T uniform steps on an 8x8 grid, starting far from goal.
        # Empirically on this grid, E[dist | random walk] converges to ~6-7.
        rsg_results = {
            "n_random_starts": eval_random_starts,
            "steps_per_start": eval_steps_per_start,
            "final_goal": [eval_gx, eval_gy],
            "episodes": eval_episodes,
            "tail_mean_dist_aggregate": float(np.mean(tail_dists)),
            "tail_std_dist_aggregate": float(np.std(tail_dists)),
            "fraction_near_goal": float(np.mean([t <= 2 for t in tail_dists])),
        }
        if verbose:
            print(
                f"[g9 seed={seed}] RSG: {eval_random_starts} random-start evals, "
                f"tail_mean_dist={rsg_results['tail_mean_dist_aggregate']:.2f}+/-"
                f"{rsg_results['tail_std_dist_aggregate']:.2f}  "
                f"frac_near_goal={rsg_results['fraction_near_goal']:.2f}",
                flush=True,
            )

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
        "phase_stats": phase_stats,
        "action_selection": action_selection,
        "reward_learning_rate": learning_rate,
        "reward_eligibility_tau_ms": reward_eligibility_tau_ms,
        "reward_hold_steps": reward_hold_steps,
        "stdp_w_max": stdp_w_max,
        "enable_neuromod_gating": enable_neuromod_gating,
        "neuromod_tau_ms": neuromod_tau_ms if enable_neuromod_gating else None,
        "neuromod_strength": neuromod_strength if enable_neuromod_gating else None,
        "first_goal_step": first_goal_step,
        "n_plastic_synapses": n_plastic,
        "motor_exploration_rate_hz": motor_exploration_rate_hz,
        "positive_only_reward": positive_only_reward,
        "action_attribution_eligibility": action_attribution_eligibility,
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
        "plastic_weight_final_min": float(final_data[i2m_flat_indices].min()),
        "plastic_weight_final_max": float(final_data[i2m_flat_indices].max()),
        "plastic_weight_final_mean": float(final_data[i2m_flat_indices].mean()),
        "plastic_weight_final_std": float(final_data[i2m_flat_indices].std()),
        "rsg": rsg_results,
        "neuromodulator_concentrations": (
            {
                name: bridge.neuromodulator_manager.get_concentration(name)
                for name in bridge.neuromodulator_manager.modulator_names()
            }
            if bridge.neuromodulator_manager is not None
            else None
        ),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(
            f"[g9 seed={seed}] done: mean_dist={results['mean_distance_overall']:.2f}  "
            f"quarters={[round(q, 2) for q in quarters]}  "
            f"at_goal={results['n_steps_at_goal']}  actions={results['action_counts']}  "
            f"reservoir_drift={reservoir_drift:.2e}  {elapsed:.1f}s",
            flush=True,
        )
    return results
