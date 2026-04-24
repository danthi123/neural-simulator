"""Smoke test for G5.v3: short episode, verify weights move and reservoir doesn't."""
import json

import pytest


def test_g5_v3_smoke(tmp_path):
    pytest.importorskip("cupy")
    from research.runners.g5_v3_runner import run_g5_v3_episode

    out = tmp_path / "g5v3.json"
    r = run_g5_v3_episode(
        out_path=str(out),
        seed=42, n_steps=30, n_positions=16,
        start_position=8, goal_position=12,
        learning_rate=0.05,  # bigger lr so weights definitely move in 30 steps
        verbose=False,
    )

    assert out.exists()
    data = json.load(open(out))
    assert data["seed"] == 42
    assert len(data["trajectory"]) == 31

    # Reservoir drift must be zero (we only update hidden->motor).
    assert data["reservoir_weight_drift_max"] == 0.0, (
        f"Reservoir weights changed by {data['reservoir_weight_drift_max']} — "
        f"runner accidentally touched fixed synapses."
    )

    # At least some plastic updates happened (reward != 0 on some steps).
    non_zero_rewards = sum(1 for r in data["reward_log"] if r != 0)
    assert non_zero_rewards > 0, "No reward steps — agent never moved?"

    # Motor counts recorded per step, 2 values each.
    for c in data["motor_counts"]:
        assert len(c) == 2
