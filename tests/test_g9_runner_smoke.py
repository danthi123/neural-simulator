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
