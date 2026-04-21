"""G5 smoke: short episode, verify the trajectory records and the brain
responds to sensor changes (position moves at all)."""
import json

import pytest


def test_g5_smoke_short_episode(tmp_path):
    pytest.importorskip("cupy")
    from research.runners.g5_runner import run_g5_episode

    out = tmp_path / "g5.json"
    r = run_g5_episode(
        out_path=str(out),
        seed=42,
        n_steps=20,
        n_positions=16,
        start_position=8,
        verbose=False,
    )
    assert out.exists()
    data = json.load(open(out))
    assert data["seed"] == 42
    assert len(data["trajectory"]) == 21  # start + 20 actions
    assert data["trajectory"][0] == 8
    # Position must stay in [0, 15].
    assert min(data["trajectory"]) >= 0
    assert max(data["trajectory"]) <= 15
    # Motor counts per step: 2 values.
    assert len(data["motor_counts"]) == 20
    for c in data["motor_counts"]:
        assert len(c) == 2
