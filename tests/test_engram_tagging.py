"""Tests for the engram-tagging API (P2 / catalog D.14 / roadmap T1.C).

The API provides Tonegawa-style ensemble tagging:
- start_engram_recording(name): begin accumulating spike counts
- commit_engram_tag(name, threshold_hz | top_k, region_filter):
  finalize the tag
- stimulate_tag(name, drive_pA): drive the tagged ensemble
- clear_tag_drive, list_engram_tags, get_engram_tag_indices,
  delete_engram_tag

Uses a small NumPy-backend bridge to keep tests fast (~5-10s each).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def small_bridge():
    """Build a tiny brain-region bridge once per module (~3s)."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway

    regions = [
        BrainRegion(name="input_a", n_neurons=64, exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="output_b", n_neurons=128, exc_fraction=0.8,
                    internal_density=0.0, exc_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False),
    ]
    pathways = [
        RegionPathway(
            from_region="input_a", to_region="output_b",
            density=0.2, weight_mean=3.0, weight_jitter=0.1,
            plastic=False,
        ),
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
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
    return bridge


# ──────────────────────────────────────────────────────────────────────
# API surface
# ──────────────────────────────────────────────────────────────────────


def test_engram_methods_present(small_bridge):
    """All 7 user-facing engram-tagging methods exist."""
    for m in ("start_engram_recording", "commit_engram_tag",
              "stimulate_tag", "clear_tag_drive", "list_engram_tags",
              "get_engram_tag_indices", "delete_engram_tag"):
        assert hasattr(small_bridge, m), f"missing method: {m}"


def test_empty_state_initially(small_bridge):
    """Before any recording, list_engram_tags returns []."""
    # Note: small_bridge is module-scoped; if other tests ran first
    # the dict may have entries. Force reset for this test.
    small_bridge._engram_tags = {}
    small_bridge._engram_recordings = {}
    assert small_bridge.list_engram_tags() == []


# ──────────────────────────────────────────────────────────────────────
# Recording lifecycle
# ──────────────────────────────────────────────────────────────────────


def test_start_then_commit_top_k(small_bridge):
    """start_engram_recording + steps + commit_engram_tag(top_k) tags
    the most-active K neurons."""
    from sim.backend import get_backend
    cp, _ = get_backend()

    # Reset state
    small_bridge._engram_tags = {}
    small_bridge._engram_recordings = {}

    # Drive a known subset of input_a so it fires consistently
    rm = small_bridge.region_manager
    input_indices = list(rm.indices("input_a"))
    # Pick first 8 to drive (we want these to be the engram)
    target = input_indices[:8]
    target_arr = cp.asarray(target, dtype=cp.int64)

    small_bridge.start_engram_recording("test_tag")
    # Drive + step
    small_bridge.cp_external_input_current[:] = 0.0
    small_bridge.cp_external_input_current[target_arr] = 300.0
    for _ in range(20):
        small_bridge._run_one_simulation_step()
        small_bridge.runtime_state.current_time_step += 1
    small_bridge.cp_external_input_current[:] = 0.0

    stats = small_bridge.commit_engram_tag("test_tag", top_k=8)
    assert stats["name"] == "test_tag"
    assert stats["n_recorded_steps"] == 20
    assert stats["window_ms"] == 20.0  # 20 steps * 1 ms
    assert stats["n_tagged"] > 0
    # The driven neurons should be in the top by spike count
    tagged = small_bridge.get_engram_tag_indices("test_tag")
    # Convert to host for set comparison
    from sim.backend import to_host
    tagged_host = to_host(tagged)
    # At least some of the driven targets should be tagged
    overlap = set(int(x) for x in tagged_host) & set(int(x) for x in target)
    assert len(overlap) >= 1, (
        f"Expected driven neurons in top-K tag; got tagged={tagged_host}, "
        f"target={target}"
    )


def test_commit_without_recording_raises(small_bridge):
    """commit_engram_tag without start_engram_recording raises KeyError."""
    small_bridge._engram_recordings = {}
    with pytest.raises(KeyError):
        small_bridge.commit_engram_tag("never_started")


def test_threshold_hz_selection(small_bridge):
    """commit_engram_tag with threshold_hz tags neurons above the
    threshold firing rate."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    small_bridge._engram_tags = {}
    small_bridge._engram_recordings = {}

    rm = small_bridge.region_manager
    input_indices = list(rm.indices("input_a"))
    target = input_indices[:4]
    target_arr = cp.asarray(target, dtype=cp.int64)

    small_bridge.start_engram_recording("hz_test")
    small_bridge.cp_external_input_current[:] = 0.0
    small_bridge.cp_external_input_current[target_arr] = 500.0
    for _ in range(50):
        small_bridge._run_one_simulation_step()
        small_bridge.runtime_state.current_time_step += 1
    small_bridge.cp_external_input_current[:] = 0.0

    # 50 steps at dt=1ms = 50ms = 0.05s window
    # Threshold 100 Hz means 100 * 0.05 = 5 spikes minimum
    stats = small_bridge.commit_engram_tag("hz_test", threshold_hz=20.0)
    assert stats["n_recorded_steps"] == 50
    assert stats["window_ms"] == 50.0
    # At least 1 neuron should be tagged (the driven ones)
    assert stats["n_tagged"] >= 1


# ──────────────────────────────────────────────────────────────────────
# Stimulate / clear
# ──────────────────────────────────────────────────────────────────────


def test_stimulate_tag_sets_current(small_bridge):
    """stimulate_tag sets cp_external_input_current at tagged indices."""
    from sim.backend import to_host
    small_bridge._engram_tags = {}
    small_bridge._engram_recordings = {}

    # Manually create a small tag (avoid the recording dance)
    import numpy as np
    fake_tag_indices = np.array([0, 5, 10, 15, 20], dtype=np.int64)
    from sim.backend import get_backend
    cp, _ = get_backend()
    small_bridge._engram_tags["manual"] = cp.asarray(fake_tag_indices)

    # Zero current
    small_bridge.cp_external_input_current[:] = 0.0

    n = small_bridge.stimulate_tag("manual", drive_pA=123.0)
    assert n == 5

    cur_host = to_host(small_bridge.cp_external_input_current)
    for i in fake_tag_indices:
        assert abs(cur_host[i] - 123.0) < 0.01, f"index {i} not driven"
    # Non-tagged indices should be 0
    assert cur_host[1] == 0.0
    assert cur_host[7] == 0.0


def test_clear_tag_drive_zeros_targeted_indices(small_bridge):
    """clear_tag_drive(name) zeros current at tagged indices only."""
    from sim.backend import to_host, get_backend
    cp, _ = get_backend()
    small_bridge._engram_tags = {}
    small_bridge._engram_recordings = {}
    import numpy as np
    small_bridge._engram_tags["t"] = cp.asarray(
        np.array([2, 4, 6], dtype=np.int64))

    # Set current globally
    small_bridge.cp_external_input_current[:] = 50.0
    # Drive the tag specifically (overwrites)
    small_bridge.stimulate_tag("t", drive_pA=200.0)

    # Clear just the tag
    small_bridge.clear_tag_drive("t")
    cur = to_host(small_bridge.cp_external_input_current)
    # Tagged should be 0
    for i in (2, 4, 6):
        assert cur[i] == 0.0
    # Untagged should still be 50
    assert cur[0] == 50.0
    assert cur[1] == 50.0


def test_clear_tag_drive_all(small_bridge):
    """clear_tag_drive() with no name zeros the entire current array."""
    from sim.backend import to_host
    small_bridge.cp_external_input_current[:] = 99.0
    small_bridge.clear_tag_drive()
    cur = to_host(small_bridge.cp_external_input_current)
    assert (cur == 0).all()


def test_stimulate_unknown_tag_raises(small_bridge):
    """stimulate_tag on unknown tag raises KeyError."""
    small_bridge._engram_tags = {}
    with pytest.raises(KeyError):
        small_bridge.stimulate_tag("never_committed", drive_pA=100.0)


# ──────────────────────────────────────────────────────────────────────
# Region filter
# ──────────────────────────────────────────────────────────────────────


def test_region_filter_restricts_tag(small_bridge):
    """commit_engram_tag with region_filter only tags neurons from
    those regions."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    small_bridge._engram_tags = {}
    small_bridge._engram_recordings = {}

    rm = small_bridge.region_manager
    output_b_indices = set(rm.indices("output_b"))
    input_a_indices = set(rm.indices("input_a"))

    # Drive input_a strongly so output_b also fires
    target = list(input_a_indices)[:16]
    target_arr = cp.asarray(target, dtype=cp.int64)

    small_bridge.start_engram_recording("filtered")
    small_bridge.cp_external_input_current[:] = 0.0
    small_bridge.cp_external_input_current[target_arr] = 400.0
    for _ in range(30):
        small_bridge._run_one_simulation_step()
        small_bridge.runtime_state.current_time_step += 1
    small_bridge.cp_external_input_current[:] = 0.0

    # Tag only output_b neurons
    stats = small_bridge.commit_engram_tag(
        "filtered", top_k=20, region_filter=["output_b"],
    )
    from sim.backend import to_host
    tagged_host = to_host(small_bridge.get_engram_tag_indices("filtered"))
    # All tagged neurons should be in output_b
    for idx in tagged_host:
        assert int(idx) in output_b_indices, (
            f"tagged neuron {int(idx)} not in output_b region"
        )


# ──────────────────────────────────────────────────────────────────────
# Lifecycle utilities
# ──────────────────────────────────────────────────────────────────────


def test_delete_engram_tag(small_bridge):
    """delete_engram_tag removes the tag."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    import numpy as np
    small_bridge._engram_tags["to_delete"] = cp.asarray(
        np.array([0, 1, 2], dtype=np.int64))
    assert small_bridge.delete_engram_tag("to_delete") is True
    assert "to_delete" not in [t["name"] for t in small_bridge.list_engram_tags()]
    # Deleting twice returns False
    assert small_bridge.delete_engram_tag("to_delete") is False


def test_list_engram_tags_shows_sizes(small_bridge):
    """list_engram_tags returns dicts with name + n_neurons."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    import numpy as np
    small_bridge._engram_tags = {
        "a": cp.asarray(np.array([0, 1, 2], dtype=np.int64)),
        "b": cp.asarray(np.array([5, 6, 7, 8, 9], dtype=np.int64)),
    }
    listing = sorted(
        small_bridge.list_engram_tags(), key=lambda d: d["name"])
    assert listing == [
        {"name": "a", "n_neurons": 3},
        {"name": "b", "n_neurons": 5},
    ]
