"""Smoke test for G6 2D gridworld runner."""
import json

import pytest


def test_g6_smoke(tmp_path):
    pytest.importorskip("cupy")
    from research.runners.g6_runner import run_g6_episode

    out = tmp_path / "g6.json"
    r = run_g6_episode(
        out_path=str(out),
        seed=42, n_steps=30, grid_size=8,
        start_pos=(1, 1), goal_pos=(6, 6),
        learning_rate=0.05,  # big enough to see weight moves in 30 steps
        lr_schedule="constant",
        verbose=False,
    )

    assert out.exists()
    data = json.load(open(out))
    assert data["seed"] == 42
    assert len(data["trajectory"]) == 31
    # Positions must stay in bounds.
    for (px, py) in data["trajectory"]:
        assert 0 <= px <= 7
        assert 0 <= py <= 7
    # Reservoir weights untouched.
    assert data["reservoir_weight_drift_max"] == 0.0
    # Action counts exist for all 4 motors.
    assert len(data["action_counts"]) == 4
    # Non-zero rewards somewhere.
    assert any(r != 0 for r in data["reward_log"])
