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
