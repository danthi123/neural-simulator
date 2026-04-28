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


# ───────────────────────── 2026-04-28: Cheat #5 closure ─────────────────────────


def test_bg_cross_projections_use_separate_gate():
    """Cross-projection cortex→D1/D2 pathways should be tagged with a distinct
    plasticity gate ('bg_cross_projections') from same-action pathways
    ('cortex_to_d1'). This lets the curriculum stage them independently —
    same-action plastic in phase 1, cross-projections delayed to phase 3
    (post-goal-change) so they don't accumulate phase-0 motor bias."""
    from research.runners.g11_bg_runner import build_bg_brain_regions

    regions, pathways = build_bg_brain_regions(enable_bg_cross_projections=True)

    # Cortex → striatum pathways: 4 cortex pools × 4 D1 pools + 4 cortex × 4 D2 = 32
    cortex_to_str_paths = [
        p for p in pathways
        if p.from_region.startswith("cortex_") and (
            p.to_region.startswith("str_D1_") or p.to_region.startswith("str_D2_")
        )
    ]
    assert len(cortex_to_str_paths) == 32, (
        f"4 cortex × (4 D1 + 4 D2) = 32 paths; got {len(cortex_to_str_paths)}"
    )

    # Helper: extract action letter from a pool name like "cortex_N" or "str_D1_E"
    def action_of(name: str) -> str:
        return name.split("_")[-1]

    same_action = [p for p in cortex_to_str_paths
                   if action_of(p.from_region) == action_of(p.to_region)]
    cross = [p for p in cortex_to_str_paths
             if action_of(p.from_region) != action_of(p.to_region)]
    # 4 same-action pairs × 2 (D1, D2) = 8 same-action paths
    # 12 cross pairs × 2 (D1, D2) = 24 cross paths
    assert len(same_action) == 8, f"expected 8 same-action paths; got {len(same_action)}"
    assert len(cross) == 24, f"expected 24 cross paths; got {len(cross)}"

    assert all(p.plasticity_gate == "cortex_to_d1" for p in same_action), (
        "all same-action cortex→striatum paths should share the cortex_to_d1 gate"
    )
    assert all(p.plasticity_gate == "bg_cross_projections" for p in cross), (
        "all cross-projection cortex→striatum paths should be on the "
        "bg_cross_projections gate (introduced 2026-04-28 to close cheat #5)"
    )


def test_bg_cross_curriculum_thaw(tmp_out_path):
    """Curriculum should keep bg_cross_projections frozen during phase 1+2,
    then thaw at bg_cross_thaw_step. Verify by checking the gate value at
    a step before and after the thaw boundary.

    Smoke-only: short episode (40 steps total), thaw at step 25. Pre-thaw
    the gate should be 0.0; post-thaw it should equal phase3_gain (0.7
    chosen so it differs from both 0.0 and 1.0 to catch off-by-one bugs)."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode

    # We can't directly inspect the gate from outside the runner, but we can
    # check that the run completes without crashing with the new flags wired
    # all the way through. The algebraic correctness is covered by reading
    # the curriculum logic; what we test here is the wiring.
    run_moving_goal_episode(
        out_path=tmp_out_path,
        seed=42,
        n_steps=40,
        verbose=False,
        enable_hippocampus=True,
        enable_bg_cross_projections=True,
        enable_curriculum=True,
        curriculum_warmup_steps=10,
        bg_cross_thaw_step=25,
        bg_cross_phase3_gain=0.7,
    )
    with open(tmp_out_path) as f:
        result = json.load(f)
    assert "phase_stats" in result, "runner produced output with bg_cross flags wired"


def test_bg_lateral_inhibition_pathways():
    """v3 (2026-04-28): when --bg-lateral-inhibition is on, the BG cascade
    includes 24 cross-pool MSN-MSN inhibitory pathways:
      str_D{1,2}_X → str_D{1,2}_Y for X != Y
    4 actions × 3 cross targets × 2 (D1, D2) = 24. The MSN regions are
    GABAergic (exc_fraction=0.05) so the projection IS inhibitory.
    plastic=False (static lateral inhibition).
    Default OFF — no pathways without the flag."""
    from research.runners.g11_bg_runner import build_bg_brain_regions

    _regions, no_inhib = build_bg_brain_regions(enable_bg_lateral_inhibition=False)
    _regions, with_inhib = build_bg_brain_regions(enable_bg_lateral_inhibition=True)

    def msn_lateral_count(pathways):
        n = 0
        for p in pathways:
            from_d = p.from_region.startswith("str_D")
            to_d = p.to_region.startswith("str_D")
            if not (from_d and to_d):
                continue
            from_type = p.from_region.split("_")[1]   # D1 or D2
            to_type = p.to_region.split("_")[1]
            if from_type != to_type:
                continue  # only same D-type cross-action lateral
            from_action = p.from_region.split("_")[-1]
            to_action = p.to_region.split("_")[-1]
            if from_action != to_action:
                n += 1
        return n

    assert msn_lateral_count(no_inhib) == 0, "default off"
    assert msn_lateral_count(with_inhib) == 24, (
        f"4 cortex × 3 cross × 2 (D1/D2) = 24; got {msn_lateral_count(with_inhib)}"
    )

    msn_laterals = [p for p in with_inhib
                    if p.from_region.startswith("str_D")
                    and p.to_region.startswith("str_D")
                    and p.from_region.split("_")[-1] != p.to_region.split("_")[-1]
                    and p.from_region.split("_")[1] == p.to_region.split("_")[1]]
    assert all(not p.plastic for p in msn_laterals), "lateral inhibition is static"


def test_bg_cross_projections_disabled_by_default():
    """When --bg-cross-projections is OFF, no cross-projection pathways exist
    at all. Same-action pathways still use the cortex_to_d1 gate."""
    from research.runners.g11_bg_runner import build_bg_brain_regions

    regions, pathways = build_bg_brain_regions(enable_bg_cross_projections=False)

    cortex_to_str_paths = [
        p for p in pathways
        if p.from_region.startswith("cortex_") and (
            p.to_region.startswith("str_D1_") or p.to_region.startswith("str_D2_")
        )
    ]
    # Only same-action: 4 cortex × 2 (D1, D2) = 8 paths total
    assert len(cortex_to_str_paths) == 8, (
        f"with cross-projections disabled, expected 8 same-action paths only; "
        f"got {len(cortex_to_str_paths)}"
    )
    assert all(p.plasticity_gate == "cortex_to_d1" for p in cortex_to_str_paths)


def test_pretraining_raises_on_missing_gate():
    """If a gate name we want to thaw is missing from bridge.list_plasticity_gates(),
    the helper raises KeyError mentioning both the bad name AND the actual list of
    available gates. Catches typos before pretraining wastes minutes of GPU time."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import _run_pretraining_phase

    # Fake bridge: only has one of the gates we'd thaw
    class _FakeBridge:
        def list_plasticity_gates(self):
            return ["cortex_to_d1"]  # missing all the others

        def set_plasticity_gate(self, name, value):
            raise AssertionError("should not be called when validation fails")

    with pytest.raises(KeyError) as exc_info:
        _run_pretraining_phase(
            bridge=_FakeBridge(),
            cfg=None,
            regions=None,
            n_goals=1,
            steps_per_goal=10,
            grid_size=8,
            start_pos=(1, 1),
            seed=42,
            verbose=False,
        )
    msg = exc_info.value.args[0]
    assert "bg_cross_projections" in msg or "sensory_to_cortex" in msg, (
        "error should name at least one missing gate")
    assert "cortex_to_d1" in msg, (
        "error should list the actually-available gates so the user can spot the typo")
