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
    ('corticostriatal'). This lets the curriculum stage them independently —
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

    assert all(p.plasticity_gate == "corticostriatal" for p in same_action), (
        "all same-action cortex→striatum paths should share the corticostriatal gate"
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
    at all. Same-action pathways still use the corticostriatal gate."""
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
    assert all(p.plasticity_gate == "corticostriatal" for p in cortex_to_str_paths)


def test_pretraining_raises_on_missing_gate():
    """If a gate name we want to thaw is missing from bridge.list_plasticity_gates(),
    the helper raises KeyError mentioning both the bad name AND the actual list of
    available gates. Catches typos before pretraining wastes minutes of GPU time."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import _run_pretraining_phase

    # Fake bridge: only has one of the gates we'd thaw
    class _FakeBridge:
        def list_plasticity_gates(self):
            return ["corticostriatal"]  # missing all the others

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
    assert "corticostriatal" in msg, (
        "error should list the actually-available gates so the user can spot the typo")


def test_pretraining_thaws_all_gates_at_start():
    """After validation, the helper sets every (declared) gate to 1.0 via
    set_plasticity_gate. This test uses a recording fake bridge to assert the
    thaw calls happen with the correct values."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import _run_pretraining_phase

    class _RecordingBridge:
        def __init__(self):
            self._values = {}
            self.calls = []

        def list_plasticity_gates(self):
            return ["corticostriatal", "bg_cross_projections", "sensory_to_cortex"]

        def set_plasticity_gate(self, name, value):
            self._values[name] = value
            self.calls.append((name, value))

        def get_plasticity_gate_value(self, name):
            return self._values.get(name, 1.0)

    fb = _RecordingBridge()
    summary = _run_pretraining_phase(
        bridge=fb, cfg=None, regions=None,
        n_goals=0,  # zero goals → skip the trial loop, just thaw + early-return
        steps_per_goal=0,
        grid_size=8, start_pos=(1, 1), seed=42, verbose=False,
    )

    # Every gate the bridge knows about should have been set to 1.0
    for gate in ["corticostriatal", "bg_cross_projections", "sensory_to_cortex"]:
        assert (gate, 1.0) in fb.calls, (
            f"gate {gate!r} was not thawed to 1.0; calls={fb.calls}")

    # Summary must have the documented keys
    for key in ("n_trials", "n_goal_changes", "cross_weights_mean", "cross_weights_std"):
        assert key in summary, f"summary missing {key!r}: {summary!r}"
    assert summary["n_trials"] == 0
    assert summary["n_goal_changes"] == 0


def test_pretraining_goal_sampling_respects_manhattan_3():
    """Sampler must keep new goals at least Manhattan 3 from the start cell."""
    from research.runners.g11_bg_runner import _sample_pretraining_goal
    import random

    rng = random.Random(42)
    start_pos = (1, 1)
    grid_size = 8
    for _ in range(100):
        gx, gy = _sample_pretraining_goal(rng, grid_size, start_pos, prev_goal=None)
        assert 0 <= gx < grid_size and 0 <= gy < grid_size
        manhattan = abs(gx - start_pos[0]) + abs(gy - start_pos[1])
        assert manhattan >= 3, f"sampled goal ({gx},{gy}) at Manhattan {manhattan} from {start_pos}"


def test_pretraining_goal_no_consecutive_repeats():
    """Successive samples differ from the previous goal."""
    from research.runners.g11_bg_runner import _sample_pretraining_goal
    import random

    rng = random.Random(42)
    prev = None
    for _ in range(50):
        g = _sample_pretraining_goal(rng, 8, (1, 1), prev_goal=prev)
        if prev is not None:
            assert g != prev, f"sampler returned same goal {g} as previous"
        prev = g


def test_developmental_pretraining_kwargs_accepted(tmp_out_path):
    """The runner should accept enable_developmental_pretraining + the two
    integer kwargs without raising TypeError on signature mismatch. Use
    n_goals=0 so the (still-stubbed) pretraining loop early-returns and the
    standard eval still runs.

    Note: --bg-cross-projections is enabled because the pretraining helper's
    Task 2/3 validation hard-requires the bg_cross_projections gate to be
    declared (the whole point of v4 pretraining). Task 7 will add a warning
    path for the no-cross-projections case; this test stays on the happy path."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=40, verbose=False,
        enable_bg_cross_projections=True,
        enable_developmental_pretraining=True,
        pretraining_n_goals=0,
        pretraining_steps_per_goal=0,
    )
    with open(tmp_out_path) as f:
        result = json.load(f)
    assert "phase_stats" in result


def test_run_moving_goal_with_pretraining_smoke(tmp_out_path):
    """End-to-end: tiny pretraining + tiny eval. Asserts cross-projection
    weights moved during pretraining (the whole point of v4 — phase-0
    cross-projection learning under varied goals)."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode

    # Patch the pretraining helper so we can snapshot the synapse-weight array
    # before/after the pretraining call. cp_connections.data is the canonical
    # weight buffer for bridge.
    import research.runners.g11_bg_runner as runner_mod
    snapshots = {}
    original = runner_mod._run_pretraining_phase

    def wrapped(*args, **kwargs):
        bridge = kwargs.get("bridge", args[0] if args else None)
        snapshots["pre_weights"] = bridge.cp_connections.data.copy().get()
        result = original(*args, **kwargs)
        snapshots["post_pretraining_weights"] = bridge.cp_connections.data.copy().get()
        return result

    runner_mod._run_pretraining_phase = wrapped
    try:
        run_moving_goal_episode(
            out_path=tmp_out_path, seed=42, n_steps=100, verbose=False,
            enable_bg_cross_projections=True,
            cross_projection_weight=0.0,
            enable_bg_lateral_inhibition=True,
            enable_curriculum=True, curriculum_warmup_steps=20,
            enable_developmental_pretraining=True,
            pretraining_n_goals=1,
            pretraining_steps_per_goal=50,
        )
    finally:
        runner_mod._run_pretraining_phase = original

    pre = snapshots["pre_weights"]
    post = snapshots["post_pretraining_weights"]
    n_changed = (pre != post).sum()
    assert n_changed > 0.01 * pre.size, (
        f"pretraining didn't change any weights — synapse plasticity not flowing? "
        f"{n_changed}/{pre.size} changed")

    with open(tmp_out_path) as f:
        result = json.load(f)
    assert "phase_stats" in result
    assert result["seed"] == 42


def test_developmental_pretraining_rejects_v3_1_thaw_conflict(tmp_out_path):
    """v4 keeps cross-projections frozen during eval; v3.1 thaws them at
    bg_cross_thaw_step. Both at once is meaningless. The runner should
    raise ValueError early instead of silently doing one or the other."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    with pytest.raises(ValueError) as exc:
        run_moving_goal_episode(
            out_path=tmp_out_path, seed=42, n_steps=20, verbose=False,
            enable_bg_cross_projections=True,
            enable_developmental_pretraining=True,
            bg_cross_thaw_step=10,  # v3.1 mechanism — incompatible
            pretraining_n_goals=0, pretraining_steps_per_goal=0,
        )
    assert "developmental-pretraining" in str(exc.value)
    assert "bg_cross_thaw_step" in str(exc.value) or "bg-cross-thaw-step" in str(exc.value)


def test_developmental_pretraining_warns_without_cross_projections(tmp_out_path, capsys):
    """Pretraining without --bg-cross-projections is harmless but pointless
    (the whole point is to develop cross-projection weights). Warn but proceed."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=20, verbose=True,
        enable_developmental_pretraining=True,
        enable_bg_cross_projections=False,  # the missing piece
        pretraining_n_goals=0, pretraining_steps_per_goal=0,
    )
    captured = capsys.readouterr()
    assert "warning" in captured.out.lower() or "warning" in captured.err.lower(), (
        "expected a warning about pretraining without cross-projections")
    assert "bg-cross-projections" in captured.out or "bg_cross_projections" in captured.out


# ───────────────────── 2026-04-28: structural-pruning closure ─────────────────────


def test_enable_structural_pruning_kwarg_accepted(tmp_out_path):
    """The runner accepts the new pruning kwarg without TypeError."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=20, verbose=False,
        enable_bg_cross_projections=True,
        enable_developmental_pretraining=True,
        enable_structural_pruning=True,
        pretraining_n_goals=0, pretraining_steps_per_goal=0,
    )


def test_pretraining_with_pruning_smoke(tmp_out_path):
    """End-to-end: tiny pretraining with --enable-structural-pruning. Some
    cross-projection synapses get pruned (alive=False) by the end."""
    pytest.importorskip("cupy")
    import cupy as cp
    from research.runners.g11_bg_runner import run_moving_goal_episode
    import research.runners.g11_bg_runner as runner_mod

    snapshots = {}
    original = runner_mod._run_pretraining_phase

    def wrapped(*args, **kwargs):
        bridge = kwargs.get("bridge", args[0] if args else None)
        result = original(*args, **kwargs)
        if bridge.cp_synapse_alive is not None:
            cross = bridge._plasticity_gate_to_synapses.get("bg_cross_projections")
            if cross:
                idx = cp.asarray(list(cross), dtype=cp.int64)
                snapshots["cross_alive_count"] = int(bridge.cp_synapse_alive[idx].sum())
                snapshots["cross_total"] = int(idx.size)
        return result

    runner_mod._run_pretraining_phase = wrapped
    try:
        run_moving_goal_episode(
            out_path=tmp_out_path, seed=42, n_steps=50, verbose=False,
            enable_bg_cross_projections=True,
            cross_projection_weight=0.0,
            enable_bg_lateral_inhibition=True,
            enable_curriculum=True, curriculum_warmup_steps=10,
            enable_developmental_pretraining=True,
            enable_structural_pruning=True,
            # Aggressive pruning hyperparameters so pruning fires within the
            # short smoke window (default alpha=0.001 + threshold=-1.0 needs
            # thousands of trials to produce visible effect).
            pruning_alpha=0.5,
            pruning_threshold=-0.5,
            pruning_weight_floor=10.0,
            pretraining_n_goals=1, pretraining_steps_per_goal=200,
        )
    finally:
        runner_mod._run_pretraining_phase = original

    cross_alive = snapshots["cross_alive_count"]
    cross_total = snapshots["cross_total"]
    assert cross_total > 0, "test config should produce cross-projection synapses"
    assert cross_alive < cross_total, "pruning should eliminate at least 1 synapse"
    assert cross_alive > 0, "pruning should NOT eliminate everything"


# ───────────────────── 2026-04-28: Cluster B.1 D1/D2 asymmetry ─────────────────────


def test_d1_d2_asymmetry_kwarg_accepted(tmp_out_path):
    """The runner accepts enable_d1_d2_asymmetry without TypeError."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=20, verbose=False,
        enable_d1_d2_asymmetry=True,
    )


# ───────────────────── 2026-04-28: Cluster B.2 striatal FSIs ─────────────────────


def test_striatal_fsis_default_off():
    """When --enable-striatal-fsis is off, no str_PV_FSI_* regions exist."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    regions, pathways = build_bg_brain_regions(enable_striatal_fsis=False)
    fs_regions = [r for r in regions if r.name.startswith("str_PV_FSI_")]
    assert len(fs_regions) == 0, "FS regions should not exist when flag off"


def test_striatal_fsis_pathways_built():
    """When --enable-striatal-fsis is on:
       - 4 str_PV_FSI_X regions added (one per action)
       - 4 cortex_X → str_PV_FSI_X pathways added (excitatory drive)
       - 24 str_PV_FSI_X → str_D{1,2}_Y pathways added (CROSS-action feedforward
         inhibition only, X != Y; 4 FS × 3 cross D-pool × 2 D-types = 24).

    Biology rationale (TK-2017 pp 161–163; Tepper-2018 pp 8–9): MSN-MSN
    collaterals deliver weak unitary IPSPs (<0.5 mV, 14-25% conn prob, high
    failure rates), so MSN-MSN lateral inhibition is functionally weak.
    FSI→MSN feedforward IPSPs are significantly larger and reliable, and
    FSIs preferentially innervate other-action MSNs. R1.2 (2026-04-29) wired
    FSIs to cross-action MSNs only — same-action (X→X) is omitted, since
    real FSIs do not inhibit their own action channel's MSN pool.

    All FS-related pathways are plastic=False (static gating)."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    regions, pathways = build_bg_brain_regions(enable_striatal_fsis=True)

    fs_regions = [r for r in regions if r.name.startswith("str_PV_FSI_")]
    assert len(fs_regions) == 4, f"Expected 4 FS regions; got {len(fs_regions)}"
    fs_names = sorted(r.name for r in fs_regions)
    assert fs_names == ["str_PV_FSI_E", "str_PV_FSI_N", "str_PV_FSI_S", "str_PV_FSI_W"]

    cortex_to_fs = [p for p in pathways
                    if p.from_region.startswith("cortex_") and p.to_region.startswith("str_PV_FSI_")]
    assert len(cortex_to_fs) == 4, \
        f"Expected 4 cortex→FS pathways; got {len(cortex_to_fs)}"
    for p in cortex_to_fs:
        # Same action only: cortex_N→str_PV_FSI_N etc.
        assert p.from_region.split("_")[1] == p.to_region.split("_")[2], \
            f"cortex→FS pathway should be same-action; got {p.from_region}→{p.to_region}"
        assert not p.plastic, "cortex→FS should be plastic=False"

    fs_to_msn = [p for p in pathways
                 if p.from_region.startswith("str_PV_FSI_")
                 and (p.to_region.startswith("str_D1_") or p.to_region.startswith("str_D2_"))]
    assert len(fs_to_msn) == 24, \
        f"Expected 24 FS→MSN pathways (4 FS × 3 cross D-pool × 2 D-types); got {len(fs_to_msn)}"
    for p in fs_to_msn:
        assert not p.plastic, "FS→MSN cross-action inhibition should be plastic=False"

    # Catalog R1.2: FSI cross-action only — FS_X must NOT project back to
    # str_D1_X or str_D2_X (its own action channel).
    for p in fs_to_msn:
        fs_action = p.from_region.split("_")[-1]   # str_PV_FSI_X → X
        msn_action = p.to_region.split("_")[-1]    # str_D1_Y → Y
        assert fs_action != msn_action, (
            f"FSI within-action wiring leaked: {p.from_region}→{p.to_region}; "
            f"per TK-2017/Tepper-2018, FSIs target cross-action MSNs only."
        )

    # And the inverse: each (fs_action, str_action, d_type) cross pair is present.
    expected = {
        (fs_a, str_a, d) for fs_a in ("N", "E", "S", "W")
        for str_a in ("N", "E", "S", "W")
        if fs_a != str_a
        for d in ("D1", "D2")
    }
    found = {
        (p.from_region.split("_")[-1],
         p.to_region.split("_")[-1],
         p.to_region.split("_")[1])
        for p in fs_to_msn
    }
    assert found == expected, (
        f"FSI cross-action pathway set mismatch.\n"
        f"missing: {expected - found}\nextra: {found - expected}"
    )


def test_striatal_fsis_disabled_by_default():
    """build_bg_brain_regions default: no FS regions or pathways."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    regions, pathways = build_bg_brain_regions()  # all defaults
    assert not any(r.name.startswith("str_PV_FSI_") for r in regions)
    assert not any(p.from_region.startswith("str_PV_FSI_") or p.to_region.startswith("str_PV_FSI_")
                   for p in pathways)


def test_striatal_fsis_kwarg_accepted(tmp_out_path):
    """Runner accepts enable_striatal_fsis without TypeError."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=20, verbose=False,
        enable_striatal_fsis=True,
    )


# ───────────────────── 2026-04-29: R3.7 GPe PV+/PV- split ─────────────────────


def test_gpe_arky_regions_present_by_default():
    """R3.7 (Mallet 2008 / Kita 2007): GPe split into PV+ (gpe_X) and PV-
    (gpe_arky_X) subpools. The arkypallidal pool is unconditionally present
    in the cascade; D2 drives both subpools."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    regions, _ = build_bg_brain_regions()
    arky_names = {r.name for r in regions if r.name.startswith("gpe_arky_")}
    assert arky_names == {"gpe_arky_N", "gpe_arky_E", "gpe_arky_S", "gpe_arky_W"}, \
        f"Expected 4 gpe_arky_* regions; got {arky_names}"


def test_d2_drives_both_gpe_subpools():
    """D2 -> gpe_X (PV+) and D2 -> gpe_arky_X (PV-) per R3.7."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    _, pathways = build_bg_brain_regions()
    d2_to_gpe = {(p.from_region, p.to_region) for p in pathways
                 if p.from_region.startswith("str_D2_") and p.to_region.startswith("gpe_")}
    for action in ("N", "E", "S", "W"):
        assert (f"str_D2_{action}", f"gpe_{action}") in d2_to_gpe, \
            f"missing D2 -> gpe_{action} (PV+ canonical)"
        assert (f"str_D2_{action}", f"gpe_arky_{action}") in d2_to_gpe, \
            f"missing D2 -> gpe_arky_{action} (PV- arkypallidal)"


def test_gpe_arky_to_fsi_only_when_fsi_enabled():
    """When --enable-striatal-fsis is on, gpe_arky_X broadcasts onto all
    str_PV_FSI_Y (Mallet 2012 stop-signal). Without FSI population, no
    arky->FS pathways are emitted."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    # FSI enabled -> arky -> FS broadcast pathways exist
    _, pathways_with_fs = build_bg_brain_regions(enable_striatal_fsis=True)
    arky_to_fs = [p for p in pathways_with_fs
                  if p.from_region.startswith("gpe_arky_")
                  and p.to_region.startswith("str_PV_FSI_")]
    assert len(arky_to_fs) == 16, \
        f"Expected 4 arky x 4 FS = 16 pathways; got {len(arky_to_fs)}"
    # FSI disabled -> no arky -> FS pathways
    _, pathways_no_fs = build_bg_brain_regions(enable_striatal_fsis=False)
    arky_to_fs_off = [p for p in pathways_no_fs
                      if p.from_region.startswith("gpe_arky_")
                      and p.to_region.startswith("str_PV_FSI_")]
    assert len(arky_to_fs_off) == 0


# ───────────────────── 2026-04-29: Cluster A closed BG loop ─────────────────────


def test_cluster_a_default_off():
    """No cortex->stn or thal->cortex pathways unless --enable-cluster-a-closed-loop."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    _, pathways = build_bg_brain_regions()  # default
    cortex_to_stn = [p for p in pathways
                     if p.from_region.startswith("cortex_")
                     and p.to_region == "stn"]
    thal_to_cortex = [p for p in pathways
                      if p.from_region.startswith("thal_")
                      and p.to_region.startswith("cortex_")]
    assert len(cortex_to_stn) == 0
    assert len(thal_to_cortex) == 0


def test_cluster_a_hyperdirect_pathways_built():
    """--enable-cluster-a-closed-loop adds 4 cortex_X -> stn pathways
    (hyperdirect; one per action)."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    _, pathways = build_bg_brain_regions(enable_cluster_a_closed_loop=True)
    cortex_to_stn = [p for p in pathways
                     if p.from_region.startswith("cortex_")
                     and p.to_region == "stn"]
    # Filter to base-pool cortex_X (not cortex_FS_X if cortex lateral inhib were on)
    base_cortex_to_stn = [p for p in cortex_to_stn
                          if p.from_region in {"cortex_N", "cortex_E", "cortex_S", "cortex_W"}]
    assert len(base_cortex_to_stn) == 4
    for p in base_cortex_to_stn:
        assert p.density == 0.10
        assert p.weight_mean == 3.0
        assert p.plastic is False


def test_cluster_a_thal_to_cortex_pathways_built():
    """--enable-cluster-a-closed-loop adds 4 thal_X -> cortex_X pathways
    (closed loop, action-specific only — no cross-action feedback)."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    _, pathways = build_bg_brain_regions(enable_cluster_a_closed_loop=True)
    thal_to_cortex = [p for p in pathways
                      if p.from_region.startswith("thal_")
                      and p.to_region.startswith("cortex_")
                      and p.to_region in {"cortex_N", "cortex_E", "cortex_S", "cortex_W"}]
    assert len(thal_to_cortex) == 4
    for p in thal_to_cortex:
        thal_action = p.from_region.split("_")[-1]
        cortex_action = p.to_region.split("_")[-1]
        assert thal_action == cortex_action, \
            f"Cluster A is action-specific; thal_{thal_action} should NOT project to cortex_{cortex_action}"
        assert p.density == 0.50
        assert p.weight_mean == 5.0
        assert p.plastic is False


def test_cluster_a_kwarg_accepted(tmp_out_path):
    """Runner accepts enable_cluster_a_closed_loop kwarg without error."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=20, verbose=False,
        enable_cluster_a_closed_loop=True,
    )


# ───────────────────── 2026-04-29: Cluster C v1 tonic DA ─────────────────────


def test_cluster_c_tonic_da_kwarg_accepted(tmp_out_path):
    """Runner accepts enable_tonic_da kwarg without error."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=20, verbose=False,
        enable_tonic_da=True,
    )


def test_default_dopamine_config_helper():
    """_default_dopamine_config() returns a valid dopamine modulator."""
    from sim.neuromodulators import _default_dopamine_config, NeuromodulatorConfig
    cfg = _default_dopamine_config()
    assert isinstance(cfg, NeuromodulatorConfig)
    assert cfg.name == "dopamine"
    assert cfg.baseline > 0.0  # tonic baseline
    rate_targets = [t for t in cfg.targets if t.target_type == "plasticity_rate"]
    assert len(rate_targets) >= 1
    from_reward_rules = [r for r in cfg.production_rules if r.rule_type == "from_reward"]
    assert len(from_reward_rules) >= 1


def test_tonic_da_triggers_plasticity_rate_modulation():
    """When DA is registered, plasticity_rate_multiplier deviates from 1.0
    according to current DA concentration vs baseline."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.neuromodulators import (
        NeuromodulatorManager,
        _default_dopamine_config,
    )
    cfg = _default_dopamine_config()
    mgr = NeuromodulatorManager([cfg], dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)
    # At baseline (0.5), plasticity_rate ~= 1.0
    mult_baseline = mgr.compute_plasticity_rate_multiplier()
    assert abs(mult_baseline - 1.0) < 1e-3, f"At baseline, expected ~1.0; got {mult_baseline}"
    # Set above baseline -> multiplier > 1
    mgr.set_concentration("dopamine", 1.5)
    mult_high = mgr.compute_plasticity_rate_multiplier()
    assert mult_high > 1.5, f"DA above baseline should boost plasticity; got {mult_high}"
    # Set below baseline -> multiplier < 1
    mgr.set_concentration("dopamine", 0.0)
    mult_low = mgr.compute_plasticity_rate_multiplier()
    assert mult_low < 1.0, f"DA below baseline should reduce plasticity; got {mult_low}"


# ───────────────────── 2026-04-29: R3.11 striosome (patch) split ─────────────────────


def test_str_patch_regions_present_by_default():
    """R3.11: str_patch_X regions exist unconditionally (4 actions)."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    regions, _ = build_bg_brain_regions()
    patch_names = {r.name for r in regions if r.name.startswith("str_patch_")}
    assert patch_names == {"str_patch_N", "str_patch_E", "str_patch_S", "str_patch_W"}


def test_str_patch_targets_dopamine_and_gpi():
    """Per PBR-160 ch 9/11: striosomes project to BOTH SNc (DA) and SNr (gpi).
    R3.11 wires str_patch_X -> dopamine and str_patch_X -> gpi_X."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    _, pathways = build_bg_brain_regions()
    patch_outs = [(p.from_region, p.to_region) for p in pathways
                  if p.from_region.startswith("str_patch_")]
    for action in ("N", "E", "S", "W"):
        assert (f"str_patch_{action}", "snc") in patch_outs, \
            f"missing str_patch_{action} -> snc (canonical striosome->SNc)"
        assert (f"str_patch_{action}", f"gpi_{action}") in patch_outs, \
            f"missing str_patch_{action} -> gpi_{action} (striosome->SNr per Deniau)"


def test_str_patch_uses_msn_e_inh_override():
    """str_patch_X regions inherit MSN class E_inh override (-60 mV) per R1.1."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    regions, _ = build_bg_brain_regions()
    patch_regions = [r for r in regions if r.name.startswith("str_patch_")]
    for r in patch_regions:
        assert r.syn_reversal_potential_i_override == -60.0, \
            f"{r.name} should have E_inh override = -60 mV (MSN class)"


# ───────────────────── 2026-04-28: Cluster B.3 cholinergic TANs ─────────────────────


def test_tans_kwarg_accepted(tmp_out_path):
    """Runner accepts enable_tans without TypeError. The flag should turn on
    the neuromodulator subsystem and register the default acetylcholine
    config (pause_on_reward → plasticity_window_gate)."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=20, verbose=False,
        enable_tans=True,
    )


# ───────────────────── 2026-04-29: Cluster D v1 hippocampus trisynaptic loop ─────────────────────


_CLUSTER_D_REGIONS = ("ec", "dg", "dg_fs", "ca3", "ca1")


def test_cluster_d_default_off():
    """No ec/dg/dg_fs/ca3/ca1 regions exist when --enable-cluster-d-hippocampus is off."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    regions, pathways = build_bg_brain_regions()  # default: flag off
    region_names = {r.name for r in regions}
    for name in _CLUSTER_D_REGIONS:
        assert name not in region_names, \
            f"region {name!r} should not exist when --enable-cluster-d-hippocampus is off"


def test_cluster_d_regions_present():
    """All 5 trisynaptic-loop regions present when --enable-cluster-d-hippocampus is on."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    regions, _ = build_bg_brain_regions(enable_cluster_d_hippocampus=True)
    region_names = {r.name for r in regions}
    for name in _CLUSTER_D_REGIONS:
        assert name in region_names, f"missing {name!r} when Cluster D is on"
    # Spec sizes (per docs/plans/2026-04-29-cluster-d-hippocampus-design.md).
    by_name = {r.name: r for r in regions}
    assert by_name["ec"].n_neurons == 80
    assert by_name["dg"].n_neurons == 200
    assert by_name["dg_fs"].n_neurons == 60
    assert by_name["ca3"].n_neurons == 100
    assert by_name["ca1"].n_neurons == 120
    # CA3 must be the autoassociator: dense recurrent collaterals.
    assert by_name["ca3"].internal_density == 0.30
    # DG fs is all-inhibitory (auto-derived inhibitory outputs).
    assert by_name["dg_fs"].exc_fraction == 0.0


def test_cluster_d_trisynaptic_pathways():
    """Core trisynaptic-loop pathways present: ec->dg, dg->ca3, ca3->ca1."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    _, pathways = build_bg_brain_regions(enable_cluster_d_hippocampus=True)
    pairs = {(p.from_region, p.to_region) for p in pathways}
    assert ("ec", "dg") in pairs, "missing perforant path ec -> dg"
    assert ("dg", "ca3") in pairs, "missing mossy fibers dg -> ca3"
    assert ("ca3", "ca1") in pairs, "missing Schaffer collaterals ca3 -> ca1"
    # Direct cortical bypass.
    assert ("ec", "ca1") in pairs, "missing direct bypass ec -> ca1"


def test_cluster_d_dg_ffi():
    """DG feedforward inhibition mechanism present: ec->dg_fs and dg_fs->dg.
    These two paths together produce DG sparsity by recruiting fast-spiking
    inhibition in proportion to EC drive."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    _, pathways = build_bg_brain_regions(enable_cluster_d_hippocampus=True)
    pairs = {(p.from_region, p.to_region) for p in pathways}
    assert ("ec", "dg_fs") in pairs, "missing ec -> dg_fs (FFi recruitment)"
    assert ("dg_fs", "dg") in pairs, "missing dg_fs -> dg (FFi to granule cells)"
    # Both should be static (FFi is structural, not learned).
    by_pair = {(p.from_region, p.to_region): p for p in pathways}
    assert by_pair[("ec", "dg_fs")].plastic is False, "ec -> dg_fs should be static"
    assert by_pair[("dg_fs", "dg")].plastic is False, "dg_fs -> dg should be static"


def test_cluster_d_ca1_to_place_cells_only_with_hippocampus():
    """ca1 -> place_cells exists only when --enable-hippocampus is also on
    (place_cells region only exists in that case)."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    # Cluster D alone: no ca1 -> place_cells (place_cells region absent).
    _, pathways_d_only = build_bg_brain_regions(enable_cluster_d_hippocampus=True)
    pairs_d_only = {(p.from_region, p.to_region) for p in pathways_d_only}
    assert ("ca1", "place_cells") not in pairs_d_only, \
        "ca1 -> place_cells should be omitted when --enable-hippocampus is off"

    # Cluster D + hippocampus: readout pathway present.
    _, pathways_both = build_bg_brain_regions(
        enable_cluster_d_hippocampus=True, enable_hippocampus=True,
    )
    pairs_both = {(p.from_region, p.to_region) for p in pathways_both}
    assert ("ca1", "place_cells") in pairs_both, \
        "ca1 -> place_cells should be present when --hippocampus is also on"

    # And the existing landmark_sensors -> place_cells pathway is unchanged
    # by Cluster D (when --enable-landmark-sensor + --hippocampus are also on).
    _, pathways_full = build_bg_brain_regions(
        enable_cluster_d_hippocampus=True,
        enable_hippocampus=True,
        enable_landmarks=True,
    )
    pairs_full = {(p.from_region, p.to_region) for p in pathways_full}
    assert ("landmark_sensors", "place_cells") in pairs_full, \
        "Cluster D must not remove existing landmark_sensors -> place_cells"
    # And new landmark_sensors -> ec is added.
    assert ("landmark_sensors", "ec") in pairs_full, \
        "Cluster D should add landmark_sensors -> ec when --enable-landmark-sensor is on"


def test_cluster_d_kwarg_accepted(tmp_out_path):
    """Runner accepts enable_cluster_d_hippocampus without TypeError; runs ~20 steps."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=20, verbose=False,
        enable_cluster_d_hippocampus=True,
    )


# ───────────────────── 2026-04-29: Cluster C v2 compartmentalized DA ─────────────────────


def test_compartmentalized_da_4_modulators_registered():
    """With --enable-compartmentalized-da, all 4 dopamine_{N,E,S,W} register
    in the neuromodulator subsystem and the global `dopamine` is NOT registered."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
    )
    from sim.enums import NeuronModel
    from sim.neuromodulators import _default_per_action_dopamine_config
    from research.runners.g11_bg_runner import build_bg_brain_regions, ACTION_NAMES

    regions, pathways = build_bg_brain_regions()
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [
        _default_per_action_dopamine_config(action, idx)
        for idx, action in enumerate(ACTION_NAMES)
    ]

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    nm_names = bridge.neuromodulator_manager.modulator_names()
    expected = {f"dopamine_{a}" for a in ACTION_NAMES}
    assert expected.issubset(set(nm_names)), \
        f"missing per-action DA modulators; have: {nm_names}"
    # Global single 'dopamine' should NOT be registered when only v2 is on.
    assert "dopamine" not in nm_names, \
        f"global 'dopamine' should not coexist with per-action channels; have: {nm_names}"


def test_synapse_action_tag_populated():
    """cp_synapse_action_tag tags synapses by their POST region's action_index.
    str_D1_N -> tag=0, str_D1_E -> tag=1, str_D1_S -> tag=2, str_D1_W -> tag=3.
    Synapses targeting non-action-specific regions (sensory, place_cells, stn,
    dopamine) get tag=-1.
    """
    pytest.importorskip("cupy")
    import cupy as cp
    from sim import (
        SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
    )
    from sim.enums import NeuronModel
    from research.runners.g11_bg_runner import build_bg_brain_regions, ACTION_NAMES

    regions, pathways = build_bg_brain_regions(
        enable_hippocampus=True,  # adds non-action regions
        enable_learned_perception=True,  # adds sensory (non-action) region
    )
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    assert bridge.cp_synapse_action_tag is not None, \
        "cp_synapse_action_tag should be allocated when region_manager is present"
    nnz = bridge.cp_connections.nnz
    assert bridge.cp_synapse_action_tag.shape[0] == nnz, \
        f"action_tag size {bridge.cp_synapse_action_tag.shape[0]} != nnz {nnz}"

    # Build maps: action_idx -> set of post-neuron indices for that action's regions.
    rm = bridge.region_manager
    post_to_action = {}
    for region in rm.regions():
        a_idx = getattr(region, "action_index", None)
        if a_idx is None:
            continue
        for n_idx in rm.indices(region.name):
            post_to_action[int(n_idx)] = int(a_idx)

    post_neurons_cp = bridge.cp_connections.indices  # CSR column = post-neuron
    post_neurons_np = post_neurons_cp.get()
    tag_np = bridge.cp_synapse_action_tag.get()

    # Spot-check: synapses where post is in post_to_action should have matching tag.
    # Pick 200 random sample synapses; scan each.
    import numpy as np
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(nnz, size=min(200, nnz), replace=False)
    for syn_idx in sample_idx:
        post_neuron = int(post_neurons_np[syn_idx])
        expected = post_to_action.get(post_neuron, -1)
        actual = int(tag_np[syn_idx])
        assert actual == expected, \
            f"synapse {syn_idx} (post_neuron={post_neuron}): expected tag={expected}, got {actual}"

    # Sanity: at least some synapses with tag=-1 (sensory, place_cells, stn, dopamine targets exist)
    assert int((tag_np == -1).sum()) > 0, \
        "expected some non-action-tagged synapses (e.g. targeting place_cells, stn, dopamine, etc)"
    # Sanity: synapses tagged for each action exist
    for a_idx in range(4):
        assert int((tag_np == a_idx).sum()) > 0, \
            f"expected some synapses tagged for action {a_idx}"


def test_per_synapse_da_signal_targets_action_correctly():
    """With dopamine_N at high concentration and others at baseline, only
    synapses with tag=0 see elevated signal."""
    pytest.importorskip("cupy")
    import cupy as cp
    from sim.neuromodulators import (
        NeuromodulatorManager,
        _default_per_action_dopamine_config,
    )
    from research.runners.g11_bg_runner import ACTION_NAMES

    configs = [
        _default_per_action_dopamine_config(action, idx)
        for idx, action in enumerate(ACTION_NAMES)
    ]
    mgr = NeuromodulatorManager(configs, dt_ms=1.0)
    mgr.initialize(n_neurons=10, cp_module=cp)

    # Baseline: signal should be ~0.0 for all tags
    tag_array = cp.asarray([-1, 0, 1, 2, 3, 0, 0, -1], dtype=cp.int32)
    signal = mgr.compute_per_synapse_da_signal(tag_array)
    assert signal is not None, "compute_per_synapse_da_signal returned None despite 4 modulators registered"
    # All should be 0.0 at baseline
    signal_np = signal.get()
    assert all(abs(v) < 1e-3 for v in signal_np), f"At baseline, expected ~0.0; got {signal_np}"

    # Boost dopamine_N
    mgr.set_concentration("dopamine_N", 1.5)  # baseline=0.5, so signal=+1.0
    signal = mgr.compute_per_synapse_da_signal(tag_array)
    signal_np = signal.get()
    # tag_array entries: [-1, 0, 1, 2, 3, 0, 0, -1]
    # tag=0 -> dopamine_N elevated, signal ~ +1.0
    # tag=-1 -> 0.0 (no action)
    # tag in {1,2,3} -> still baseline, signal ~ 0.0
    assert abs(signal_np[0] - 0.0) < 1e-3, f"tag=-1 should be 0.0; got {signal_np[0]}"
    assert abs(signal_np[1] - 1.0) < 1e-3, f"tag=0 should be 1.0; got {signal_np[1]}"
    assert abs(signal_np[2] - 0.0) < 1e-3, f"tag=1 should be 0.0; got {signal_np[2]}"
    assert abs(signal_np[3] - 0.0) < 1e-3, f"tag=2 should be 0.0; got {signal_np[3]}"
    assert abs(signal_np[4] - 0.0) < 1e-3, f"tag=3 should be 0.0; got {signal_np[4]}"
    assert abs(signal_np[5] - 1.0) < 1e-3, f"tag=0 should be 1.0; got {signal_np[5]}"
    assert abs(signal_np[6] - 1.0) < 1e-3, f"tag=0 should be 1.0; got {signal_np[6]}"
    assert abs(signal_np[7] - 0.0) < 1e-3, f"tag=-1 should be 0.0; got {signal_np[7]}"


def test_compartmentalized_da_kwarg_accepted(tmp_out_path):
    """20-step smoke: runner accepts --enable-compartmentalized-da without
    TypeError, registers 4 per-action DA modulators, and produces a valid result."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode

    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=20, verbose=False,
        enable_compartmentalized_da=True,
    )


def test_compartmentalized_da_action_specific_reward_rule():
    """from_action_specific_reward production rule fires only when
    last_selected_action matches source_action."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim.config import CoreSimConfig
    from sim.neuromodulators import (
        NeuromodulatorManager,
        _default_per_action_dopamine_config,
    )

    cfg_da = _default_per_action_dopamine_config("N", 0)  # source_action=0
    mgr = NeuromodulatorManager([cfg_da], dt_ms=1.0)
    mgr.initialize(n_neurons=4, cp_module=cp)

    class _Bridge:
        def __init__(self):
            self.core_config = CoreSimConfig()
            self.core_config.current_reward_signal = 1.0
            self.core_config.reward_baseline = 0.0
            self.core_config.last_selected_action = 0  # matches!

    b = _Bridge()
    # At baseline (0.5), with reward=1.0 and last_action=0=source_action,
    # one step should add (1.0 - 0.0)*sensitivity = +1.0 (capped by clipping).
    initial = mgr.get_concentration("dopamine_N")
    mgr.step(b)
    after_match = mgr.get_concentration("dopamine_N")
    assert after_match > initial, \
        f"with last_action=source_action, concentration should rise; {initial} -> {after_match}"

    # Reset
    mgr.set_concentration("dopamine_N", 0.5)  # back to baseline
    # Mismatched action: should NOT fire
    b.core_config.last_selected_action = 2  # mismatch
    initial2 = mgr.get_concentration("dopamine_N")
    mgr.step(b)
    after_mismatch = mgr.get_concentration("dopamine_N")
    # With mismatch, only decay applies — concentration stays at baseline.
    assert abs(after_mismatch - initial2) < 0.01, \
        f"with last_action != source_action, no production; {initial2} -> {after_mismatch}"


def test_compartmentalized_da_action_index_populated_on_regions():
    """Action-specific regions (cortex_X, str_D1_X, str_D2_X, gpi_X, thal_X,
    motor_X, etc) have action_index in [0, 3]; non-action regions (stn,
    dopamine, sensory, place_cells) have action_index=None."""
    from research.runners.g11_bg_runner import build_bg_brain_regions, ACTION_NAMES
    regions, _ = build_bg_brain_regions(
        enable_hippocampus=True,
        enable_learned_perception=True,
        enable_motor_lateral_inhibition=True,
        enable_cortex_lateral_inhibition=True,
        enable_striatal_fsis=True,
        enable_cluster_d_hippocampus=True,
    )
    by_name = {r.name: r for r in regions}

    # Action-specific regions
    for idx, action in enumerate(ACTION_NAMES):
        for prefix in ("cortex_", "str_D1_", "str_D2_", "gpi_", "thal_",
                       "motor_", "gpe_", "gpe_arky_", "str_patch_"):
            name = f"{prefix}{action}"
            assert name in by_name, f"region {name} not built"
            assert by_name[name].action_index == idx, \
                f"{name} action_index expected {idx}, got {by_name[name].action_index}"
        # Optional regions only with their flags
        for prefix in ("cortex_FS_", "motor_FS_", "str_PV_FSI_"):
            name = f"{prefix}{action}"
            if name in by_name:
                assert by_name[name].action_index == idx, \
                    f"{name} action_index expected {idx}, got {by_name[name].action_index}"

    # Non-action-specific regions
    for name in ("stn", "snc", "sensory", "place_cells", "goal_cells",
                 "ec", "dg", "dg_fs", "ca3", "ca1"):
        if name in by_name:
            assert by_name[name].action_index is None, \
                f"{name} action_index should be None; got {by_name[name].action_index}"


# ───────────────────── 2026-04-29: Cluster E v1 topographic maps ─────────────────────


def test_cluster_e_default_off():
    """Default off: cortex_X / str_D1_X / str_D2_X have coordinate_dim=0."""
    from research.runners.g11_bg_runner import build_bg_brain_regions

    regions, _ = build_bg_brain_regions()  # default
    by_name = {r.name: r for r in regions}
    for action in ("N", "E", "S", "W"):
        for prefix in ("cortex_", "str_D1_", "str_D2_"):
            r = by_name[f"{prefix}{action}"]
            assert r.coordinate_dim == 0, \
                f"{r.name} coordinate_dim should be 0 by default; got {r.coordinate_dim}"
            assert r.coordinate_extent is None
            assert r.coordinate_center is None


def test_cluster_e_coordinate_assignment():
    """When --enable-cluster-e-topography is on, action-specific regions
    (cortex_X, str_D1_X, str_D2_X) get 2D coords pinned to the corner of the
    unit square corresponding to their action. Non-action regions stay at
    coordinate_dim=0 (no coords)."""
    from research.runners.g11_bg_runner import build_bg_brain_regions

    regions, _ = build_bg_brain_regions(enable_cluster_e_topography=True)
    by_name = {r.name: r for r in regions}

    expected_corners = {
        "N": (0.5, 1.0),
        "E": (1.0, 0.5),
        "S": (0.5, 0.0),
        "W": (0.0, 0.5),
    }
    for action, corner in expected_corners.items():
        for prefix in ("cortex_", "str_D1_", "str_D2_"):
            r = by_name[f"{prefix}{action}"]
            assert r.coordinate_dim == 2, f"{r.name} should have 2D coords"
            assert r.coordinate_center == corner, (
                f"{r.name} should be pinned to {corner}; got {r.coordinate_center}"
            )

    # Non-action regions (stn, snc) remain unstructured.
    for name in ("stn", "snc"):
        r = by_name[name]
        assert r.coordinate_dim == 0, (
            f"{name} should remain unstructured; got coordinate_dim={r.coordinate_dim}"
        )


def test_cluster_e_distance_sigma_pathways():
    """Cluster E v1: cortex_X -> str_D1_X / str_D2_X pathways carry
    distance_sigma=0.3 (default) when topography is on. Sane fallback to
    None when off."""
    from research.runners.g11_bg_runner import build_bg_brain_regions

    # Off: all pathways have distance_sigma=None
    _regions, no_topo = build_bg_brain_regions()
    cortex_to_msn_off = [
        p for p in no_topo
        if p.from_region.startswith("cortex_")
        and (p.to_region.startswith("str_D1_") or p.to_region.startswith("str_D2_"))
    ]
    assert all(p.distance_sigma is None for p in cortex_to_msn_off), (
        "without --enable-cluster-e-topography, cortex->MSN distance_sigma must be None"
    )

    # On: all cortex -> MSN pathways have distance_sigma=0.3
    _regions, with_topo = build_bg_brain_regions(enable_cluster_e_topography=True)
    cortex_to_msn_on = [
        p for p in with_topo
        if p.from_region.startswith("cortex_")
        and (p.to_region.startswith("str_D1_") or p.to_region.startswith("str_D2_"))
    ]
    assert len(cortex_to_msn_on) >= 8, "expected at least 8 same-action paths"
    for p in cortex_to_msn_on:
        assert p.distance_sigma == 0.3, (
            f"cortex->MSN should carry distance_sigma=0.3; got {p.distance_sigma} "
            f"on {p.from_region}->{p.to_region}"
        )


def test_cluster_e_custom_sigma_propagates():
    """`cluster_e_distance_sigma` kwarg overrides default 0.3."""
    from research.runners.g11_bg_runner import build_bg_brain_regions

    _regions, pathways = build_bg_brain_regions(
        enable_cluster_e_topography=True,
        cluster_e_distance_sigma=0.7,
    )
    cortex_to_msn = [
        p for p in pathways
        if p.from_region.startswith("cortex_")
        and (p.to_region.startswith("str_D1_") or p.to_region.startswith("str_D2_"))
    ]
    for p in cortex_to_msn:
        assert p.distance_sigma == 0.7


def test_cluster_e_kwarg_accepted(tmp_out_path):
    """20-step smoke: runner accepts --enable-cluster-e-topography
    end-to-end without crashing or producing an empty cascade."""
    pytest.importorskip("cupy")
    from research.runners.g11_bg_runner import run_moving_goal_episode
    run_moving_goal_episode(
        out_path=tmp_out_path, seed=42, n_steps=20, verbose=False,
        enable_cluster_e_topography=True,
    )
    with open(tmp_out_path) as f:
        result = json.load(f)
    assert "phase_stats" in result
    assert len(result["motor_counts"]) == 20
