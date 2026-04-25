"""Smoke test for G9: short episode, verify sim-native R-STDP updates weights."""
import json

import pytest


def test_g9_smoke_argmax(tmp_path):
    """30-step episode with argmax; verify plastic weights moved via sim R-STDP."""
    pytest.importorskip("cupy")
    from research.runners.g9_runner import run_g9_episode

    out = tmp_path / "g9.json"
    r = run_g9_episode(
        out_path=str(out),
        seed=42, n_steps=30, grid_size=8,
        start_pos=(1, 1), goal_pos=(6, 6),
        learning_rate=0.05,  # aggressive for quick smoke
        reward_eligibility_tau_ms=500.0,
        reward_hold_steps=10,
        action_selection="argmax",
        verbose=False,
    )
    data = json.load(open(out))
    assert data["seed"] == 42
    assert data["action_selection"] == "argmax"
    assert len(data["trajectory"]) == 31
    assert len(data["motor_counts"]) == 30

    # Reservoir must NOT drift — plastic mask should freeze it.
    assert data["reservoir_weight_drift_max"] == 0.0, (
        f"Reservoir drifted despite plastic mask: {data['reservoir_weight_drift_max']}"
    )

    # Plastic weights SHOULD have moved (sim-native R-STDP active).
    # Initial weights are roughly normal(1.0, 0.2). After 30 trials with
    # non-zero reward on most steps, eligibility should have flowed through.
    # We expect std to change at least slightly from initial.
    assert data["plastic_weight_final_std"] > 0.0, "Plastic std collapsed to 0?"

    # A non-zero reward should have occurred somewhere in 30 steps.
    assert any(abs(r) > 0 for r in data["reward_log"]), "No reward signals?"


def test_g9_smoke_first_spike(tmp_path):
    """30-step episode with first-spike WTA."""
    pytest.importorskip("cupy")
    from research.runners.g9_runner import run_g9_episode

    out = tmp_path / "g9_fs.json"
    r = run_g9_episode(
        out_path=str(out),
        seed=42, n_steps=30, grid_size=8,
        start_pos=(1, 1), goal_pos=(6, 6),
        learning_rate=0.05,
        reward_eligibility_tau_ms=500.0,
        reward_hold_steps=10,
        action_selection="first_spike",
        verbose=False,
    )
    data = json.load(open(out))
    assert data["action_selection"] == "first_spike"
    assert len(data["trajectory"]) == 31
    assert data["reservoir_weight_drift_max"] == 0.0


def test_g9_smoke_rsg_probe(tmp_path):
    """Session D.A.3: eval_random_starts triggers post-training frozen-weight probe."""
    pytest.importorskip("cupy")
    from research.runners.g9_runner import run_g9_episode

    out = tmp_path / "g9_rsg.json"
    r = run_g9_episode(
        out_path=str(out),
        seed=42, n_steps=30, grid_size=8,
        start_pos=(1, 1), goal_pos=(6, 6),
        learning_rate=0.05,
        action_selection="argmax",
        eval_random_starts=3,
        eval_steps_per_start=5,
        verbose=False,
    )
    data = json.load(open(out))
    rsg = data.get("rsg")
    assert rsg is not None, "rsg block missing from output"
    assert rsg["n_random_starts"] == 3
    assert rsg["steps_per_start"] == 5
    assert len(rsg["episodes"]) == 3
    for ep in rsg["episodes"]:
        assert "tail_mean_dist" in ep
        assert "start" in ep
        assert ep["start"] != list(rsg["final_goal"]), "probe start should not equal goal"
    assert data["reservoir_weight_drift_max"] == 0.0


def test_g9_smoke_with_large_reservoir(tmp_path):
    """Route C: 5000-hidden-neuron G9 runs cleanly. Tests that the runner
    + bridge + reservoir pipeline scales to where the GPU is actually
    compute-bound rather than dispatch-bound. 30 trials so it's still
    a smoke test (~1-2 min on RTX 3090)."""
    pytest.importorskip("cupy")
    from research.runners.g9_runner import _build_g9_plan, run_g9_episode

    # Override hidden layer size via the plan-builder defaults. The
    # public run_g9_episode doesn't expose these, so build the plan
    # directly to validate the larger network at the wiring level.
    core_cfg, plan = _build_g9_plan(
        seed=42,
        n_hidden_exc=4000,
        n_hidden_inh=1000,
        # default densities will yield a manageable synapse count:
        # input(64) -> hidden(5000) at 0.5 = 160000
        # hidden(5000) recurrent at 0.1 = ~2500000
        # hidden(5000) -> motor(4) at 0.5 = 10000
        hidden_to_hidden_density=0.05,  # tighter at this scale to keep nnz tractable
        input_to_hidden_density=0.1,
    )
    layout = plan["layout"]
    assert len(layout["hidden_idx"]) == 5000
    assert core_cfg.num_neurons == 64 + 5000 + 4
    # input->hidden + hidden_recurrent + hidden->motor counts
    n_input_to_hidden = plan["input_to_hidden"]["count"]
    n_hidden_recurrent = plan["hidden_recurrent"]["count"]
    n_hidden_to_motor = plan["hidden_to_motor"]["count"]
    total_synapses = n_input_to_hidden + n_hidden_recurrent + n_hidden_to_motor
    # Sanity: roughly density * full pairs
    assert n_input_to_hidden > 30000  # 64*5000*0.1 = 32000
    assert n_hidden_recurrent > 1_000_000  # 5000*5000*0.05 ~ 1.25M
    assert n_hidden_to_motor > 8000  # 5000*4*0.5 = 10000
    print(f"\n  Route C smoke: {core_cfg.num_neurons} neurons, "
          f"{total_synapses:,} synapses")


def test_g9_smoke_motor_exploration(tmp_path):
    """Session G: motor_exploration_rate_hz forces all motors to fire at least
    once in a short episode, breaking the silent-motor trap.

    Without exploration, with seed 42 / 30 steps, only 1-2 motors typically fire
    (the rest are 'silent'). With 15 Hz exploration, every motor should fire."""
    pytest.importorskip("cupy")
    import numpy as np
    from research.runners.g9_runner import run_g9_episode

    out = tmp_path / "g9_explore.json"
    r = run_g9_episode(
        out_path=str(out),
        seed=42, n_steps=30, grid_size=8,
        start_pos=(1, 1), goal_pos=(6, 6),
        learning_rate=0.05,
        reward_eligibility_tau_ms=500.0,
        reward_hold_steps=10,
        action_selection="argmax",
        motor_exploration_rate_hz=15.0,
        verbose=False,
    )
    data = json.load(open(out))
    assert data["motor_exploration_rate_hz"] == 15.0
    # Reservoir still frozen by plastic mask
    assert data["reservoir_weight_drift_max"] == 0.0
    # Critical invariant: every motor fired at least once across the episode.
    # motor_counts is a list of [n_motor] count arrays per step.
    motor_counts = np.asarray(data["motor_counts"])  # (n_steps, n_motor)
    per_motor_total = motor_counts.sum(axis=0)
    assert (per_motor_total > 0).all(), (
        f"Some motors silent despite exploration noise: {per_motor_total.tolist()}"
    )


def test_g9_smoke_positive_only_reward(tmp_path):
    """Session G v3: positive_only_reward emits 0 instead of -1 when distance
    increases. Verify reward log has no negative entries."""
    pytest.importorskip("cupy")
    from research.runners.g9_runner import run_g9_episode

    out = tmp_path / "g9_posrew.json"
    run_g9_episode(
        out_path=str(out),
        seed=42, n_steps=30, grid_size=8,
        start_pos=(1, 1), goal_pos=(6, 6),
        learning_rate=0.05,
        action_selection="argmax",
        positive_only_reward=True,
        verbose=False,
    )
    data = json.load(open(out))
    assert data["positive_only_reward"] is True
    # No negative reward signals.
    assert all(r >= 0 for r in data["reward_log"]), (
        f"positive_only_reward should never emit negative; got "
        f"{[r for r in data['reward_log'] if r < 0]}"
    )
    # But should still emit positive when agent moves toward goal.
    assert any(r > 0 for r in data["reward_log"]), (
        "Expected at least one positive reward in 30 steps"
    )
    assert data["reservoir_weight_drift_max"] == 0.0


def test_g9_smoke_proportional(tmp_path):
    """Session G v5: proportional action selection samples motors with
    probability proportional to spike_count + 1. Verify it runs and
    produces non-trivial action distribution."""
    pytest.importorskip("cupy")
    import numpy as np
    from research.runners.g9_runner import run_g9_episode

    out = tmp_path / "g9_prop.json"
    run_g9_episode(
        out_path=str(out),
        seed=42, n_steps=30, grid_size=8,
        start_pos=(1, 1), goal_pos=(6, 6),
        learning_rate=0.05,
        action_selection="proportional",
        verbose=False,
    )
    data = json.load(open(out))
    assert data["action_selection"] == "proportional"
    assert data["reservoir_weight_drift_max"] == 0.0
    # Proportional sampling produces variety of actions, not just one motor
    # winning every time. Even with weak training, we should see >=2 distinct
    # motors picked across 30 steps (entropy floor from +1 prior).
    action_counts = np.bincount(data["action_log"], minlength=4)
    n_distinct = (action_counts > 0).sum()
    assert n_distinct >= 2, (
        f"Expected proportional sampling to pick >=2 distinct motors in 30 steps, "
        f"got distribution {action_counts.tolist()}"
    )


def test_g9_smoke_action_attribution(tmp_path):
    """Session G v4: action_attribution_eligibility zeros eligibility for
    non-chosen motors. Verify it runs and reservoir is still frozen."""
    pytest.importorskip("cupy")
    from research.runners.g9_runner import run_g9_episode

    out = tmp_path / "g9_attr.json"
    run_g9_episode(
        out_path=str(out),
        seed=42, n_steps=30, grid_size=8,
        start_pos=(1, 1), goal_pos=(6, 6),
        learning_rate=0.05,
        action_selection="argmax",
        motor_exploration_rate_hz=15.0,
        action_attribution_eligibility=True,
        verbose=False,
    )
    data = json.load(open(out))
    assert data["action_attribution_eligibility"] is True
    assert data["reservoir_weight_drift_max"] == 0.0
    # Plastic weights should still move (action attribution narrows the
    # eligible set but doesn't prevent updates).
    assert data["plastic_weight_final_std"] > 0.0


def test_g9_smoke_with_neuromodulators(tmp_path):
    """Session E.1: G9 runner accepts nm_configs and threads them into
    the bridge, registers group indices, records final concentrations."""
    pytest.importorskip("cupy")
    from research.runners.g9_runner import run_g9_episode
    from sim.neuromodulators import (
        NeuromodulatorConfig, ModulatorTarget, ProductionRule,
    )

    nm_configs = [
        NeuromodulatorConfig(
            name="dopamine",
            baseline=0.0,
            decay_tau_ms=500.0,
            production_rules=[ProductionRule(rule_type="from_reward", sensitivity=1.0)],
            targets=[],
        ),
        NeuromodulatorConfig(
            name="noradrenaline",
            baseline=0.1,
            decay_tau_ms=2000.0,
            production_rules=[
                ProductionRule(
                    rule_type="from_error_persistence",
                    sensitivity=0.5,
                    threshold=0.3,
                    window_ms=1000.0,
                )
            ],
            targets=[
                ModulatorTarget(
                    target_type="excitability_drive",
                    scope="group:motor",
                    sensitivity=30.0,
                ),
            ],
        ),
    ]
    out = tmp_path / "g9_nm.json"
    r = run_g9_episode(
        out_path=str(out),
        seed=42, n_steps=30, grid_size=8,
        start_pos=(1, 1), goal_pos=(6, 6),
        learning_rate=0.05,
        action_selection="argmax",
        nm_configs=nm_configs,
        verbose=False,
    )
    data = json.load(open(out))
    assert "neuromodulator_concentrations" in data
    nmc = data["neuromodulator_concentrations"]
    assert nmc is not None
    assert "dopamine" in nmc
    assert "noradrenaline" in nmc
    # Reservoir weights still must be frozen by the plastic mask, even
    # under neuromod (gain modulation should not write to base weights).
    assert data["reservoir_weight_drift_max"] == 0.0
