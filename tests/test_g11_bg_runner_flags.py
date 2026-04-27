"""Smoke tests for g11_bg_runner opt-in flags.

Each test runs a tiny moving-goal episode (50-100 steps, no learning load)
with one or more flags enabled. Verifies the runner doesn't crash and
produces structurally valid output. Does NOT test learning quality —
that's covered by the acid-test runs documented in
research/findings/2026-04-25/26-*.md.

These guards are intended to catch regressions when the runner is edited.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def tmp_out_path(tmp_path):
    return str(tmp_path / "g11_smoke.json")


def _run_one(out_path, **kwargs):
    """Run one moving-goal episode with given kwargs, return parsed result."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode

    n_steps = kwargs.pop("n_steps", 50)
    run_moving_goal_episode(
        out_path=out_path,
        seed=42,
        n_steps=n_steps,
        verbose=False,
        **kwargs,
    )

    assert os.path.exists(out_path), f"runner did not produce {out_path}"
    with open(out_path) as f:
        result = json.load(f)
    assert "motor_counts" in result
    assert "phase_stats" in result
    assert len(result["motor_counts"]) == n_steps
    return result


def test_baseline_no_flags(tmp_out_path):
    """Default behavior: no opt-in flags, just Phase B baseline."""
    _run_one(tmp_out_path)


def test_motor_lateral_inhibition(tmp_out_path):
    """WTA microcircuit (FS interneurons + motor cross-pool inhibition)."""
    _run_one(tmp_out_path, enable_motor_lateral_inhibition=True)


def test_per_action_da_hard(tmp_out_path):
    """Hard per-action DA targeting (always-on eligibility gating)."""
    _run_one(tmp_out_path, enable_per_action_da_targeting=True)


def test_adaptive_per_action_da(tmp_out_path):
    """Symmetric adaptive DA (reward-EMA-gated eligibility)."""
    _run_one(tmp_out_path, enable_adaptive_per_action_da=True)


def test_asymmetric_adaptive_da(tmp_out_path):
    """Asymmetric adaptive DA (slow positive, fast negative — recommended for slow-change)."""
    _run_one(
        tmp_out_path,
        enable_adaptive_per_action_da=True,
        adaptive_da_ema_decay=0.9,
        adaptive_da_ema_decay_negative=0.7,
    )


def test_da_gated_wta(tmp_out_path):
    """DA-gated WTA: motor FS->motor weights scaled by gating_strength."""
    _run_one(
        tmp_out_path,
        enable_motor_lateral_inhibition=True,
        enable_adaptive_per_action_da=True,
        enable_da_gated_wta=True,
    )


def test_learned_perception(tmp_out_path):
    """Sensory layer + plastic sensory->cortex mapping (replaces heuristic)."""
    _run_one(tmp_out_path, enable_learned_perception=True)


def test_rpe_scaled_reward(tmp_out_path):
    """RPE-scaled reward (delivered = reward + alpha * RPE)."""
    _run_one(tmp_out_path, enable_rpe_scaled_reward=True)


def test_surprise_lr_boost(tmp_out_path):
    """Surprise-boosted learning rate (most robust across task types)."""
    _run_one(tmp_out_path, enable_surprise_lr_boost=True)


def test_multi_goal_schedule(tmp_out_path):
    """4-corner goal schedule (validates phase counting through multiple goal changes)."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode

    # 100 steps, 2 goal changes (compressed for speed)
    run_moving_goal_episode(
        out_path=tmp_out_path,
        seed=42,
        n_steps=80,
        verbose=False,
        goal_schedule=[(0, (6, 6)), (30, (1, 6)), (60, (1, 1))],
    )
    with open(tmp_out_path) as f:
        result = json.load(f)
    # 3 phases expected from the goal_schedule
    assert len(result["phase_stats"]) == 3, (
        f"expected 3 phases, got {len(result['phase_stats'])}"
    )


def test_combo_lr_boost_plus_asym_da(tmp_out_path):
    """Combination flag: surprise LR + asymmetric adaptive DA. Should not crash even if not optimal."""
    _run_one(
        tmp_out_path,
        enable_adaptive_per_action_da=True,
        adaptive_da_ema_decay_negative=0.7,
        enable_surprise_lr_boost=True,
    )


def test_motor_counts_structure(tmp_out_path):
    """Verify motor_counts log is per-trial, length-4 list per entry."""
    result = _run_one(tmp_out_path, n_steps=20)
    for trial_counts in result["motor_counts"]:
        assert len(trial_counts) == 4, f"expected 4 actions, got {len(trial_counts)}"
        for c in trial_counts:
            assert isinstance(c, int)
            assert c >= 0


# ───────────────────────── 2026-04-27 additions ─────────────────────────


def test_hippocampus_with_curriculum(tmp_out_path):
    """Hippocampus + curriculum (Phase C breakthrough recipe)."""
    _run_one(
        tmp_out_path,
        enable_hippocampus=True,
        enable_adaptive_per_action_da=True,
        adaptive_da_ema_decay_negative=0.7,
        enable_curriculum=True,
        curriculum_warmup_steps=20,
    )


def test_pfc_region_builds(tmp_out_path):
    """PFC region (Item 3): recurrent prefrontal cortex for working memory."""
    result = _run_one(
        tmp_out_path,
        enable_hippocampus=True,
        enable_pfc=True,
        n_pfc=30,  # smaller for speed
    )
    # Should produce output with phase_stats; PFC region builds cleanly
    assert "phase_stats" in result
    assert len(result["phase_stats"]) >= 1


def test_sensory_plus_pfc_plus_curriculum(tmp_out_path):
    """Best-config recipe: sensory + hippo + PFC + curriculum (recommended)."""
    _run_one(
        tmp_out_path,
        enable_hippocampus=True,
        enable_learned_perception=True,
        enable_pfc=True,
        n_pfc=30,
        enable_adaptive_per_action_da=True,
        adaptive_da_ema_decay_negative=0.7,
        enable_curriculum=True,
        curriculum_warmup_steps=20,
    )


def test_grid_size_scaling(tmp_out_path):
    """Grid size + n_hippocampus_per_layer scaling (Item 2)."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode

    run_moving_goal_episode(
        out_path=tmp_out_path,
        seed=42,
        n_steps=50,
        verbose=False,
        grid_size=12,  # non-default
        n_hippocampus_per_layer=144,  # 12² for one cell per position
        enable_hippocampus=True,
        goal_schedule=[(0, (10, 10)), (25, (1, 10))],
    )
    with open(tmp_out_path) as f:
        result = json.load(f)
    assert result["grid_size"] == 12


def test_sleep_replay_smoke(tmp_out_path):
    """Sleep-replay infrastructure: agent freezes, gates flip during sleep."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode

    run_moving_goal_episode(
        out_path=tmp_out_path,
        seed=42,
        n_steps=80,
        verbose=False,
        enable_hippocampus=True,
        enable_curriculum=True,
        curriculum_warmup_steps=20,
        sleep_replay_after_step=50,
        sleep_replay_steps=20,
    )
    with open(tmp_out_path) as f:
        result = json.load(f)
    # Distance during sleep should be flat (agent doesn't move)
    distances = result["distance_log"]
    sleep_distances = distances[50:70]
    assert len(set(sleep_distances)) <= 2, (
        "agent should not move during sleep (distance should be near-constant)"
    )


def test_goal_silence_smoke(tmp_out_path):
    """PFC Stage 2 delayed-response: goal_silence flag drives goal/heuristic to 0."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode

    # Smoke test only: just verify it runs without crashing
    run_moving_goal_episode(
        out_path=tmp_out_path,
        seed=42,
        n_steps=80,
        verbose=False,
        enable_hippocampus=True,
        enable_pfc=True,
        n_pfc=30,
        enable_curriculum=True,
        curriculum_warmup_steps=20,
        goal_silence_after_step=50,
        goal_silence_duration=20,
    )
    with open(tmp_out_path) as f:
        result = json.load(f)
    # Just verify the run completed
    assert "phase_stats" in result
