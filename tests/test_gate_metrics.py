"""Unit tests for the redesigned gate metrics (Session D)."""
import json

import numpy as np
import pytest


def test_ttp_monotone_learning():
    """A distance log that monotonically decreases to 0 should have TTP < n."""
    from research.gate_metrics import time_to_proficiency

    # Fast learner: reaches dist<=2 by step 10, stays there.
    dist = list(range(10, 0, -1)) + [0] * 100
    ttp = time_to_proficiency(dist, proficiency_dist=2, window_size=20, threshold=0.5)
    # Window ends at 19; steps 0-19 include indices 0..19. dist[0..7] > 2,
    # dist[8..19] <= 2. That's 12 out of 20 -> 0.60 >= 0.5. TTP = 19.
    assert ttp is not None
    assert ttp <= 40, f"TTP too late for monotone learner: {ttp}"


def test_ttp_never_for_random_walk():
    """Random-walk distance should have TTP = None (never reaches threshold)."""
    from research.gate_metrics import time_to_proficiency

    rng = np.random.default_rng(42)
    # Uniform over 0..14 (Manhattan diameter of 8x8 grid)
    dist = rng.integers(0, 15, size=600).tolist()
    ttp = time_to_proficiency(dist, proficiency_dist=2, window_size=50, threshold=0.5)
    # Random baseline PF ~ 3/15 = 0.20, far below 0.5 threshold
    assert ttp is None, f"Random walk should not acquire, got TTP={ttp}"


def test_pf_equals_baseline_for_random():
    """PF on uniform random distances should be near the analytical baseline."""
    from research.gate_metrics import proficiency_fraction, random_baseline_proficiency

    rng = np.random.default_rng(42)
    # For 8x8 grid with goal at (6,6) and proficiency_dist=2:
    # Analytical random baseline != uniform-over-dist; but for a uniform-random
    # POSITION, the fraction within Manhattan <= 2 of (6,6) is what we expect.
    # Here we just test the plumbing with a uniform 0..14 distance array and
    # verify the PF is close to fraction of integers in [0, 2] within [0, 14].
    dist = rng.integers(0, 15, size=100000)
    pf = proficiency_fraction(dist.tolist(), proficiency_dist=2)
    # Expected: P(dist <= 2) = 3/15 = 0.20
    assert abs(pf - 0.20) < 0.01, f"PF for uniform random: {pf}"


def test_pf_full_coverage():
    """All-zero distance -> PF = 1.0."""
    from research.gate_metrics import proficiency_fraction

    assert proficiency_fraction([0] * 100, proficiency_dist=2) == 1.0


def test_random_baseline_proficiency_8x8():
    """8x8 grid, goal=(6,6), D=2: cells within dist<=2 of (6,6)."""
    from research.gate_metrics import random_baseline_proficiency

    # Cells: diamond of radius 2 around (6,6)
    # Manually: |dx|+|dy| <= 2, clipped to [0,7] on both axes
    # (4,6), (5,5-7), (6,4-8 clipped to 4-7), (7,5-7), (8 clipped)
    # Enumerate:
    expected = 0
    for x in range(8):
        for y in range(8):
            if abs(x - 6) + abs(y - 6) <= 2:
                expected += 1
    rb = random_baseline_proficiency(8, proficiency_dist=2, goal=(6, 6))
    assert abs(rb - expected / 64.0) < 1e-9


def test_summarize_on_existing_g9_file(tmp_path):
    """End-to-end: summarize a synthesized G9-like JSON and verify shape."""
    from research.gate_metrics import summarize_g_run

    # Synthesize a fast-learning fixed-goal log
    run = {
        "seed": 42,
        "n_steps": 300,
        "grid_size": 8,
        "start_pos": [1, 1],
        "goal_pos": [6, 6],
        "goal_schedule": [[0, [6, 6]]],
        "phase_stats": [
            {
                "phase": 0,
                "step_start": 0,
                "step_end": 300,
                "goal": [6, 6],
                "mean_distance": 1.5,
                "final_quarter_mean_distance": 1.0,
                "n_steps_at_goal": 50,
                "n_steps": 300,
                "action_counts": [100, 100, 50, 50],
            }
        ],
        "distance_log": list(range(10, 0, -1))
        + [0] * 290
        + [0],  # monotone then stays
    }
    p = tmp_path / "synthetic_g.json"
    p.write_text(json.dumps(run))
    s = summarize_g_run(str(p))
    # Single-phase run -> treated as fixed-goal (correct semantics)
    assert "fixed_goal" in s
    fg = s["fixed_goal"]
    assert fg["acquired"] is True
    assert fg["PF_overall"] > 0.9  # mostly at goal
    assert fg["TTP"] is not None


def test_summarize_multi_phase_moving_goal(tmp_path):
    """Multi-phase run -> treated as moving-goal with per-phase metrics."""
    from research.gate_metrics import summarize_g_run

    # Synthesize a moving-goal run with two phases
    # Phase 0: monotone learning
    # Phase 1: never reaches goal (simulates failed readaptation)
    phase0_log = list(range(10, 0, -1)) + [0] * 290
    phase1_log = [6] * 300  # flat, never near goal
    full_log = phase0_log + phase1_log
    assert len(full_log) == 600

    run = {
        "seed": 42,
        "n_steps": 600,
        "grid_size": 8,
        "start_pos": [1, 1],
        "goal_pos": [6, 6],
        "goal_schedule": [[0, [6, 6]], [300, [1, 6]]],
        "phase_stats": [
            {"phase": 0, "step_start": 0, "step_end": 300, "goal": [6, 6],
             "mean_distance": 1.5, "final_quarter_mean_distance": 1.0,
             "n_steps_at_goal": 50, "n_steps": 300,
             "action_counts": [100, 100, 50, 50]},
            {"phase": 1, "step_start": 300, "step_end": 600, "goal": [1, 6],
             "mean_distance": 6.0, "final_quarter_mean_distance": 6.0,
             "n_steps_at_goal": 0, "n_steps": 300,
             "action_counts": [100, 100, 50, 50]},
        ],
        "distance_log": full_log,
    }
    p = tmp_path / "synthetic_moving.json"
    p.write_text(json.dumps(run))
    s = summarize_g_run(str(p))
    assert "moving_goal_phases" in s
    assert len(s["moving_goal_phases"]) == 2
    assert s["moving_goal_phases"][0]["acquired"] is True
    assert s["moving_goal_phases"][1]["acquired"] is False
    assert s["n_phases_acquired"] == 1
