"""Smoke tests for P3.1 concept replay (run_concept_replay_phase).

Catalog D.19 (SWRs) + D.14 (engram cells). Roadmap T1.B.

These tests verify the replay function's mechanics on a small bridge:
- The function dispatches stimulate_tag for each tag in the order
- per_tag_replay_count adds up to n_replays_per_tag × len(tag_names)
- Missing tags are silently skipped (KeyError caught)
- Zero-size tags are skipped

Full integration test (consolidation actually transfers patterns to
cortex) is a future runner-style test, not a unit test.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def tagged_bridge():
    """Build a small bridge + create 2 engram tags."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion
    from sim.backend import get_backend
    cp, _ = get_backend()
    import numpy as np

    regions = [
        BrainRegion(name="a", n_neurons=32, exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False),
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = []
    cfg.dt_ms = 1.0
    cfg.seed = 42
    cfg.fast_spike_reset = True
    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # Manually create two tags
    bridge._init_engram_tagging()
    bridge._engram_tags["tag_a"] = cp.asarray(
        np.array([0, 1, 2, 3, 4], dtype=np.int64))
    bridge._engram_tags["tag_b"] = cp.asarray(
        np.array([10, 11, 12, 13, 14], dtype=np.int64))
    bridge._engram_tags["empty_tag"] = cp.asarray(
        np.array([], dtype=np.int64))
    return bridge


def test_concept_replay_dispatches_each_tag(tagged_bridge):
    """run_concept_replay_phase calls stimulate_tag for each tag in
    the order, n_replays_per_tag times."""
    from research.runners.consolidation_trainer import (
        run_concept_replay_phase,
    )
    result = run_concept_replay_phase(
        tagged_bridge,
        tag_names=["tag_a", "tag_b"],
        n_replays_per_tag=3,
        burst_duration_ms=5,  # Keep test fast
        inter_burst_ms=2,
        randomize_order=False,
    )
    assert result["n_replays"] == 6  # 2 tags × 3 replays
    assert result["per_tag_replay_count"]["tag_a"] == 3
    assert result["per_tag_replay_count"]["tag_b"] == 3
    assert result["tags_replayed"] == ["tag_a", "tag_b"]


def test_concept_replay_skips_empty_tags(tagged_bridge):
    """Tags with 0 neurons are silently skipped (stimulate_tag returns 0)."""
    from research.runners.consolidation_trainer import (
        run_concept_replay_phase,
    )
    result = run_concept_replay_phase(
        tagged_bridge,
        tag_names=["tag_a", "empty_tag"],
        n_replays_per_tag=2,
        burst_duration_ms=5,
        inter_burst_ms=2,
        randomize_order=False,
    )
    # tag_a fires 2 times; empty_tag should be skipped (0 firings)
    assert result["per_tag_replay_count"]["tag_a"] == 2
    assert result["per_tag_replay_count"]["empty_tag"] == 0
    assert result["n_replays"] == 2


def test_concept_replay_skips_missing_tags(tagged_bridge):
    """Tags not committed are silently skipped."""
    from research.runners.consolidation_trainer import (
        run_concept_replay_phase,
    )
    result = run_concept_replay_phase(
        tagged_bridge,
        tag_names=["tag_a", "nonexistent_tag"],
        n_replays_per_tag=2,
        burst_duration_ms=5,
        inter_burst_ms=2,
        randomize_order=False,
    )
    # tag_a fires 2 times; nonexistent skipped (count stays 0)
    assert result["per_tag_replay_count"]["tag_a"] == 2
    assert result["per_tag_replay_count"]["nonexistent_tag"] == 0
    assert result["n_replays"] == 2


def test_concept_replay_advances_sim_time(tagged_bridge):
    """Replay steps advance the simulation timestep counter."""
    from research.runners.consolidation_trainer import (
        run_concept_replay_phase,
    )
    initial_step = tagged_bridge.runtime_state.current_time_step
    result = run_concept_replay_phase(
        tagged_bridge,
        tag_names=["tag_a"],
        n_replays_per_tag=2,
        burst_duration_ms=5,
        inter_burst_ms=3,
        randomize_order=False,
    )
    final_step = tagged_bridge.runtime_state.current_time_step
    # 2 replays × (5 burst + 3 quiet) = 16 steps
    expected = initial_step + 16
    assert final_step == expected, (
        f"expected {expected} steps, got {final_step}"
    )


def test_concept_replay_randomize_order_changes_order(tagged_bridge):
    """randomize_order=True can shuffle. Just verify it doesn't crash
    and produces the right counts."""
    from research.runners.consolidation_trainer import (
        run_concept_replay_phase,
    )
    import numpy as np
    rng = np.random.default_rng(7)
    result = run_concept_replay_phase(
        tagged_bridge,
        tag_names=["tag_a", "tag_b"],
        n_replays_per_tag=4,
        burst_duration_ms=3,
        inter_burst_ms=1,
        randomize_order=True,
        rng=rng,
    )
    # Each tag should still get exactly 4 replays
    assert result["per_tag_replay_count"]["tag_a"] == 4
    assert result["per_tag_replay_count"]["tag_b"] == 4
    assert result["n_replays"] == 8
