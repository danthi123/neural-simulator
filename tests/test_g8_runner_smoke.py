"""Smoke test for G8: short episode, verify dual-channel input works."""
import json

import pytest


def test_g8_smoke_with_goal_context(tmp_path):
    pytest.importorskip("cupy")
    from research.runners.g8_runner import run_g8_episode

    out = tmp_path / "g8.json"
    r = run_g8_episode(
        out_path=str(out),
        seed=42, n_steps=30, grid_size=8,
        start_pos=(1, 1), goal_pos=(6, 6),
        learning_rate=0.05,
        goal_context_enabled=True,
        verbose=False,
    )

    assert out.exists()
    data = json.load(open(out))
    assert data["seed"] == 42
    assert data["goal_context_enabled"] is True
    assert len(data["trajectory"]) == 31
    assert len(data["motor_counts"]) == 30
    for c in data["motor_counts"]:
        assert len(c) == 4  # 4 motor neurons

    # Reservoir must not drift — only hidden->motor is plastic.
    assert data["reservoir_weight_drift_max"] == 0.0, (
        f"Reservoir weights changed by {data['reservoir_weight_drift_max']} — "
        f"runner accidentally touched fixed synapses."
    )


def test_g8_smoke_goal_context_disabled(tmp_path):
    """Ablation: goal_context_enabled=False should still run without errors."""
    pytest.importorskip("cupy")
    from research.runners.g8_runner import run_g8_episode

    out = tmp_path / "g8_ablate.json"
    r = run_g8_episode(
        out_path=str(out),
        seed=42, n_steps=30, grid_size=8,
        start_pos=(1, 1), goal_pos=(6, 6),
        learning_rate=0.05,
        goal_context_enabled=False,
        verbose=False,
    )
    data = json.load(open(out))
    assert data["goal_context_enabled"] is False
    assert data["reservoir_weight_drift_max"] == 0.0
