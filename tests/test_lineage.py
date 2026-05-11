"""Tests for sim.lineage Bridge Lineage Manager.

Per user (2026-05-10): persistent continuous-learning state for the sim.
These tests verify lineage save/load/fork/rollback work correctly without
needing GPU.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.lineage import (
    BridgeLineage, LineageMetadata, GrowthEvent, AccuracyDatapoint,
    METADATA_SCHEMA_VERSION,
)


class _MockBridge:
    """Lightweight stand-in for SimulationBridge to test save/load
    without needing GPU."""
    def __init__(self, content: str = "fake-bridge-state"):
        self.content = content

    def save_checkpoint(self, path: str):
        Path(path).write_text(self.content, encoding="utf-8")

    def load_checkpoint(self, path: str):
        self.content = Path(path).read_text(encoding="utf-8")


def _mock_save(bridge, path: str):
    bridge.save_checkpoint(path)


def _mock_load(path: str, mode: str, seed: int) -> _MockBridge:
    b = _MockBridge()
    b.load_checkpoint(path)
    return b


# ──────────────────────────────────────────────────────────────────────
# LineageMetadata tests
# ──────────────────────────────────────────────────────────────────────


def test_metadata_default_construction():
    """A fresh metadata has expected defaults."""
    m = LineageMetadata(lineage_name="main")
    assert m.lineage_name == "main"
    assert m.schema_version == METADATA_SCHEMA_VERSION
    assert m.cumulative_training_events == 0
    assert m.parent_lineage is None
    assert m.accuracy_history == []
    assert m.growth_events == []


def test_metadata_round_trip_via_dict():
    """to_dict + from_dict preserves all fields."""
    m = LineageMetadata(
        lineage_name="experiment_x",
        current_tier="8-word",
        vocab=["north", "east", "south", "west"],
        cumulative_training_events=1000,
    )
    d = m.to_dict()
    m2 = LineageMetadata.from_dict(d)
    assert m2.lineage_name == "experiment_x"
    assert m2.current_tier == "8-word"
    assert m2.vocab == ["north", "east", "south", "west"]
    assert m2.cumulative_training_events == 1000


def test_metadata_from_dict_ignores_unknown_fields():
    """Schema evolution-friendly: old metadata with extra fields loads cleanly."""
    d = {
        "lineage_name": "main",
        "future_field_we_dont_have_yet": "blah",
        "current_tier": "8-word",
    }
    m = LineageMetadata.from_dict(d)
    assert m.lineage_name == "main"
    assert m.current_tier == "8-word"


def test_add_growth_event_appends_to_list():
    m = LineageMetadata(lineage_name="main")
    m.add_growth_event(kind="init", description="From scratch", source="test")
    assert len(m.growth_events) == 1
    assert m.growth_events[0]["kind"] == "init"
    assert m.growth_events[0]["description"] == "From scratch"
    assert m.growth_events[0]["metadata"]["source"] == "test"


def test_add_accuracy_appends_with_timestamp():
    m = LineageMetadata(lineage_name="main")
    m.add_accuracy(metric="A2W any", value=0.75, context="post-training")
    assert len(m.accuracy_history) == 1
    point = m.accuracy_history[0]
    assert point["metric"] == "A2W any"
    assert point["value"] == 0.75
    assert point["context"] == "post-training"
    assert point["at"]  # non-empty timestamp


# ──────────────────────────────────────────────────────────────────────
# BridgeLineage tests
# ──────────────────────────────────────────────────────────────────────


def test_does_not_exist_on_fresh_directory(tmp_path):
    lineage = BridgeLineage("main", root=tmp_path)
    assert lineage.exists() is False


def test_save_creates_files(tmp_path):
    lineage = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("initial-state")
    lineage.save(bridge, save_fn=_mock_save, tier="4-word",
                  arch={"n_lang_input": 2048, "n_motor_per_action": 500})
    assert lineage.current_path.exists()
    assert lineage.metadata_path.exists()
    # Verify content
    assert lineage.current_path.read_text(encoding="utf-8") == "initial-state"
    # Verify metadata
    meta = lineage.read_metadata()
    assert meta.current_tier == "4-word"
    assert meta.arch["n_lang_input"] == 2048


def test_load_returns_path_when_no_loader(tmp_path):
    lineage = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("loaded-content")
    lineage.save(bridge, save_fn=_mock_save)
    # Load without bridge_loader returns the checkpoint path
    path = lineage.load()
    assert path == str(lineage.current_path)


def test_load_with_loader_returns_bridge(tmp_path):
    lineage = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("loaded-content")
    lineage.save(bridge, save_fn=_mock_save)
    # Load WITH bridge_loader
    loaded = lineage.load(bridge_loader=_mock_load)
    assert loaded.content == "loaded-content"


def test_load_raises_if_no_state(tmp_path):
    lineage = BridgeLineage("main", root=tmp_path)
    with pytest.raises(FileNotFoundError):
        lineage.load(bridge_loader=_mock_load)


def test_save_appends_to_history(tmp_path):
    lineage = BridgeLineage("main", root=tmp_path)
    # First save: no prior state, no history entry
    bridge = _MockBridge("v1")
    lineage.save(bridge, save_fn=_mock_save)
    assert len(lineage.list_history()) == 0
    # Second save: should snapshot v1 to history
    bridge.content = "v2"
    lineage.save(bridge, save_fn=_mock_save)
    snapshots = lineage.list_history()
    assert len(snapshots) == 1
    # The snapshot should be "v1" (the previous state)
    assert snapshots[0].read_text(encoding="utf-8") == "v1"


def test_save_can_skip_snapshot(tmp_path):
    lineage = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("v1")
    lineage.save(bridge, save_fn=_mock_save)
    bridge.content = "v2"
    lineage.save(bridge, save_fn=_mock_save, snapshot=False)
    assert len(lineage.list_history()) == 0


def test_rollback_restores_previous_state(tmp_path):
    lineage = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("v1")
    lineage.save(bridge, save_fn=_mock_save)
    bridge.content = "v2"
    lineage.save(bridge, save_fn=_mock_save)
    # Now history has one entry (v1); current is v2
    snapshots = lineage.list_history()
    assert len(snapshots) == 1
    # Extract the snapshot ID from filename: <ts>-checkpoint.simstate.h5
    snap_id = snapshots[0].name.replace("-checkpoint.simstate.h5", "")
    # Roll back
    lineage.rollback_to(snap_id)
    # Current is now v1
    assert lineage.current_path.read_text(encoding="utf-8") == "v1"


def test_rollback_missing_snapshot_raises(tmp_path):
    lineage = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("v1")
    lineage.save(bridge, save_fn=_mock_save)
    with pytest.raises(FileNotFoundError):
        lineage.rollback_to("2099-01-01T00-00-00")


def test_fork_creates_new_lineage(tmp_path):
    main = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("main-state")
    main.save(bridge, save_fn=_mock_save, tier="4-word",
              arch={"n_lang_input": 2048})
    forked = main.fork("experiment_x")
    assert forked.exists()
    assert forked.current_path.read_text(encoding="utf-8") == "main-state"
    fmeta = forked.read_metadata()
    assert fmeta.parent_lineage == "main"
    assert fmeta.current_tier == "4-word"
    assert fmeta.arch["n_lang_input"] == 2048
    # Fork records growth event
    assert any(e["kind"] == "fork" for e in fmeta.growth_events)


def test_fork_into_existing_raises(tmp_path):
    main = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("main-state")
    main.save(bridge, save_fn=_mock_save)
    forked = main.fork("experiment_x")
    bridge.content = "still-main"
    main.save(bridge, save_fn=_mock_save)
    with pytest.raises(FileExistsError):
        main.fork("experiment_x")


def test_list_all(tmp_path):
    main = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("main-state")
    main.save(bridge, save_fn=_mock_save)
    main.fork("experiment_x")
    main.fork("experiment_y")
    all_lineages = BridgeLineage.list_all(root=tmp_path)
    names = sorted([l.name for l in all_lineages])
    assert names == ["experiment_x", "experiment_y", "main"]


def test_metadata_persists_across_saves(tmp_path):
    lineage = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("v1")
    lineage.save(bridge, save_fn=_mock_save)
    # Update metadata
    meta = lineage.read_metadata()
    meta.cumulative_training_events = 100
    meta.add_growth_event(kind="init", description="Test")
    meta.add_accuracy(metric="A2W any", value=0.5)
    lineage.write_metadata(meta)
    # Save again and confirm metadata not lost
    bridge.content = "v2"
    lineage.save(bridge, save_fn=_mock_save)
    meta2 = lineage.read_metadata()
    assert meta2.cumulative_training_events == 100
    assert len(meta2.growth_events) == 1
    assert len(meta2.accuracy_history) == 1


def test_atomic_save_temp_files_removed(tmp_path):
    """After successful save, no leftover .new files."""
    lineage = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("v1")
    lineage.save(bridge, save_fn=_mock_save)
    # No .new artifacts should remain
    artifacts = list(lineage.root.glob("*.new"))
    assert artifacts == []


def test_prune_history_keeps_recent(tmp_path):
    """prune_history keeps the most-recent N snapshots."""
    lineage = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("v0")
    lineage.save(bridge, save_fn=_mock_save)
    # Save many times to build history
    for i in range(10):
        bridge.content = f"v{i+1}"
        lineage.save(bridge, save_fn=_mock_save)
    snapshots = lineage.list_history()
    assert len(snapshots) == 10
    # Prune to last 3
    lineage.prune_history(keep_last=3)
    remaining = lineage.list_history()
    assert len(remaining) == 3
    # Should keep the most recent
    contents = sorted([p.read_text(encoding="utf-8") for p in remaining])
    assert contents == ["v7", "v8", "v9"]


def test_fork_preserves_history_count(tmp_path):
    """Fork copies metadata growth_events but starts with empty history dir."""
    main = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("v0")
    main.save(bridge, save_fn=_mock_save)
    # Build some history
    for i in range(3):
        bridge.content = f"v{i+1}"
        main.save(bridge, save_fn=_mock_save)
    forked = main.fork("experiment_x")
    # Forked lineage starts with empty history dir (history is per-lineage)
    assert forked.list_history() == []
    # But growth_events ARE copied
    fmeta = forked.read_metadata()
    assert len(fmeta.growth_events) >= 1  # at least the fork event


# ──────────────────────────────────────────────────────────────────────
# Growth log markdown rendering (added 2026-05-11; completes Phase 1)
# ──────────────────────────────────────────────────────────────────────


def test_render_growth_log_basic_structure(tmp_path):
    """render_growth_log produces a valid markdown header + sections."""
    lineage = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("v1")
    lineage.save(bridge, save_fn=_mock_save, tier="4-word",
                  arch={"n_lang_input": 2048, "mode": "tier1"})
    meta = lineage.read_metadata()
    meta.cumulative_training_events = 200
    meta.add_growth_event(kind="init", description="Initial train")
    meta.add_accuracy(metric="A2W any", value=0.92, context="post-train")
    lineage.write_metadata(meta)
    md = lineage.render_growth_log()
    assert "# Growth log" in md
    assert "lineage `main`" in md
    assert "Current tier:" in md and "4-word" in md
    assert "Cumulative training events:** 200" in md
    assert "## Growth events" in md
    assert "`init`" in md and "Initial train" in md
    assert "## Accuracy history" in md
    assert "A2W any" in md
    assert "0.920" in md or "0.92" in md


def test_render_growth_log_no_events(tmp_path):
    """Empty growth_events renders the placeholder message."""
    lineage = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("v0")
    lineage.save(bridge, save_fn=_mock_save)
    md = lineage.render_growth_log()
    assert "No growth events recorded yet" in md


def test_render_growth_log_includes_event_metadata(tmp_path):
    """Per-event metadata is rendered as nested list items."""
    lineage = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("v1")
    lineage.save(bridge, save_fn=_mock_save)
    meta = lineage.read_metadata()
    meta.add_growth_event(
        kind="tier_promotion",
        description="Promoted 4-word -> 8-word",
        from_tier=4, to_tier=8,
    )
    lineage.write_metadata(meta)
    md = lineage.render_growth_log()
    assert "`tier_promotion`" in md
    assert "Promoted 4-word -> 8-word" in md
    assert "`from_tier`: 4" in md
    assert "`to_tier`: 8" in md


def test_write_growth_log_creates_file(tmp_path):
    """write_growth_log writes _growth_log.md atomically."""
    lineage = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("v1")
    lineage.save(bridge, save_fn=_mock_save)
    meta = lineage.read_metadata()
    meta.add_growth_event(kind="init", description="Test")
    lineage.write_metadata(meta)
    path = lineage.write_growth_log()
    assert path.exists()
    assert path.name == "_growth_log.md"
    content = path.read_text(encoding="utf-8")
    assert "# Growth log" in content
    # No leftover .new
    assert not (path.with_suffix(".md.new").exists())


def test_write_growth_log_idempotent(tmp_path):
    """Writing the growth log twice overwrites cleanly."""
    lineage = BridgeLineage("main", root=tmp_path)
    bridge = _MockBridge("v1")
    lineage.save(bridge, save_fn=_mock_save)
    meta = lineage.read_metadata()
    meta.add_growth_event(kind="init", description="First")
    lineage.write_metadata(meta)
    lineage.write_growth_log()
    # Now add another event + rewrite
    meta2 = lineage.read_metadata()
    meta2.add_growth_event(kind="tier_promotion", description="Second")
    lineage.write_metadata(meta2)
    lineage.write_growth_log()
    content = lineage.growth_log_path.read_text(encoding="utf-8")
    assert "First" in content
    assert "Second" in content


# ──────────────────────────────────────────────────────────────────────
# Per-pathway shard export (Strategy C of tiering Phase 3 part 2)
# Added 2026-05-11
# ──────────────────────────────────────────────────────────────────────


def test_export_shards_requires_region_manager(tmp_path):
    """export_shards raises when bridge has no region_manager."""
    lineage = BridgeLineage("rm_none", root=tmp_path)
    class _FakeBridge:
        region_manager = None
    bridge = _FakeBridge()
    with pytest.raises(RuntimeError, match="region_manager"):
        lineage.export_shards(bridge)


def test_list_shards_empty_when_not_exported(tmp_path):
    """list_shards returns [] when shards directory is absent."""
    lineage = BridgeLineage("no_shards", root=tmp_path)
    assert lineage.list_shards() == []


def test_export_shards_with_mock_bridge_round_trip(tmp_path):
    """export_shards writes per-pathway shards; list_shards returns them.

    Uses a minimal mock bridge providing region_manager + extract_per_pathway_csrs.
    """
    import scipy.sparse as sp
    import numpy as np

    class _MockRegionManager:
        def pathways(self):
            return [type("Pw", (), {
                "from_region": "lang", "to_region": "motor"
            })()]
        def indices(self, name):
            return list(range(5)) if name == "lang" else list(range(5, 10))

    class _MockBridge:
        region_manager = _MockRegionManager()
        def extract_per_pathway_csrs(self):
            # Build a small CSR for the lang->motor pathway
            data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            indices = np.array([0, 1, 2], dtype=np.int32)
            indptr = np.array([0, 1, 2, 3, 3, 3], dtype=np.int64)
            csr = sp.csr_matrix((data, indices, indptr), shape=(5, 5))
            return {"lang_to_motor": csr}

    lineage = BridgeLineage("shard_test", root=tmp_path)
    bridge = _MockBridge()
    n = lineage.export_shards(bridge)
    assert n == 1

    # Shard file exists where we expect
    shard_path = tmp_path / "shard_test" / "shards" / "lang_to_motor.npz"
    assert shard_path.exists()

    # list_shards reports the pathway
    names = lineage.list_shards()
    assert names == ["lang_to_motor"]


def test_export_shards_custom_root(tmp_path):
    """export_shards respects a custom shard_root path."""
    import scipy.sparse as sp
    import numpy as np

    class _MockRM:
        def pathways(self):
            return [type("Pw", (), {"from_region": "a", "to_region": "b"})()]
        def indices(self, name):
            return list(range(3)) if name == "a" else list(range(3, 6))

    class _MockB:
        region_manager = _MockRM()
        def extract_per_pathway_csrs(self):
            csr = sp.csr_matrix(np.eye(3, dtype=np.float32))
            return {"a_to_b": csr}

    lineage = BridgeLineage("custom_root", root=tmp_path)
    custom_path = tmp_path / "elsewhere" / "shards"
    n = lineage.export_shards(_MockB(), shard_root=custom_path)
    assert n == 1
    assert (custom_path / "a_to_b.npz").exists()
