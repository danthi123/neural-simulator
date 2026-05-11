"""Tests for sim.bridge_memory — Phase 3.1 scaffold.

Tests the API surface + lineage integration. The actual bridge
binding/recall is stubbed in Phase 3.1; tests focus on:
- Object construction
- Lineage integration (growth events recorded per call)
- stats() schema
- list_keys() returns the vocab for the mode
- Phase 3.2 placeholders documented as stubs

All CPU-only. Mock lineage; no real bridge construction (that's tested
in the chat_repl test suite).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────────────
# Mock helpers — we don't want each test to load a real bridge
# ──────────────────────────────────────────────────────────────────────


class _MockBridge:
    """Stand-in for SimulationBridge with the attrs BridgeMemory reads."""
    class _Cfg:
        num_neurons = 6336
    core_config = _Cfg()
    actual_total_connections_n = 3218125

    def save_checkpoint(self, path: str):
        from pathlib import Path
        Path(path).write_text("fake-state", encoding="utf-8")


@pytest.fixture
def mock_memory(tmp_path, monkeypatch):
    """Create a BridgeMemory backed by a mock bridge + temp lineage."""
    from sim.bridge_memory import BridgeMemory
    from sim.lineage import BridgeLineage

    # Pre-construct a lineage so _ensure_loaded picks it up
    lineage = BridgeLineage("test_memory", root=tmp_path)
    bridge = _MockBridge()
    lineage.save(bridge, tier="synonym", arch={"mode": "synonym"})

    mem = BridgeMemory(
        lineage_name="test_memory",
        mode="synonym",
        bridge=bridge,
        auto_save=True,
        verbose=False,
    )
    # Pre-set the lineage to the tmp_path one (skip auto-load logic)
    mem._lineage = lineage
    yield mem


# ──────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────


def test_bridge_memory_construction_defaults():
    """BridgeMemory has expected defaults."""
    from sim.bridge_memory import BridgeMemory
    mem = BridgeMemory()
    assert mem.lineage_name == "main"
    assert mem.mode == "synonym"
    assert mem.auto_save is True
    assert mem.bridge is None
    assert mem._vocab_size_estimate == 0


def test_bridge_memory_explicit_args():
    """Explicit args override defaults."""
    from sim.bridge_memory import BridgeMemory
    mem = BridgeMemory(
        lineage_name="custom", mode="tier1",
        auto_save=False, verbose=True,
    )
    assert mem.lineage_name == "custom"
    assert mem.mode == "tier1"
    assert mem.auto_save is False
    assert mem.verbose is True


# ──────────────────────────────────────────────────────────────────────
# store() — Phase 3.1 stub
# ──────────────────────────────────────────────────────────────────────


def test_store_returns_stub_schema(mock_memory):
    """store() returns the expected stub schema."""
    result = mock_memory.store("user_name", "alice")
    assert "key" in result and result["key"] == "user_name"
    assert "value" in result and result["value"] == "alice"
    assert "confidence" in result
    assert "bound_correctly" in result
    assert "n_events_run" in result
    assert "elapsed_seconds" in result
    assert "stub_note" in result  # Phase 3.1 placeholder
    assert "Phase 3.1" in result["stub_note"]


def test_store_records_growth_event(mock_memory):
    """store() records a memory_bind growth event."""
    mock_memory.store("favorite_color", "blue")
    meta = mock_memory._lineage.read_metadata()
    bind_events = [e for e in meta.growth_events
                    if e["kind"] == "memory_bind"]
    assert len(bind_events) >= 1
    assert "favorite_color" in bind_events[-1]["description"]


def test_store_bumps_vocab_estimate(mock_memory):
    """Each store() increments _vocab_size_estimate."""
    assert mock_memory._vocab_size_estimate == 0
    mock_memory.store("k1", "v1")
    mock_memory.store("k2", "v2")
    mock_memory.store("k3", "v3")
    assert mock_memory._vocab_size_estimate == 3


# ──────────────────────────────────────────────────────────────────────
# recall() — Phase 3.1 stub
# ──────────────────────────────────────────────────────────────────────


def test_recall_returns_empty_list_in_stub(mock_memory):
    """Phase 3.1 stub: recall() returns []."""
    result = mock_memory.recall("user_name")
    assert result == []


def test_recall_does_not_record_growth_event(mock_memory):
    """recall() is read-only; no growth event."""
    initial_meta = mock_memory._lineage.read_metadata()
    initial_count = len(initial_meta.growth_events)
    mock_memory.recall("test")
    after_meta = mock_memory._lineage.read_metadata()
    assert len(after_meta.growth_events) == initial_count


# ──────────────────────────────────────────────────────────────────────
# forget() — Phase 3.1 stub
# ──────────────────────────────────────────────────────────────────────


def test_forget_returns_stub_schema(mock_memory):
    """forget() returns stub schema with n_synapses_decayed=0."""
    result = mock_memory.forget("test_key", decay_rate=0.5)
    assert result["key"] == "test_key"
    assert result["decay_rate"] == 0.5
    assert result["n_synapses_decayed"] == 0
    assert result["estimated_retention"] == 1.0


def test_forget_records_growth_event(mock_memory):
    """forget() records a memory_forget growth event."""
    mock_memory.forget("user_name")
    meta = mock_memory._lineage.read_metadata()
    forget_events = [e for e in meta.growth_events
                       if e["kind"] == "memory_forget"]
    assert len(forget_events) >= 1
    assert "user_name" in forget_events[-1]["description"]


# ──────────────────────────────────────────────────────────────────────
# consolidate() — Phase 3.1 stub
# ──────────────────────────────────────────────────────────────────────


def test_consolidate_returns_stub_schema(mock_memory):
    """consolidate() returns expected stub schema."""
    result = mock_memory.consolidate(n_sleep_cycles=3)
    assert "pre_silence_acc" in result
    assert "hippo_off_acc" in result
    assert "retention_ratio" in result
    assert "n_sleep_cycles_run" in result
    assert "stub_note" in result


def test_consolidate_records_growth_event(mock_memory):
    """consolidate() records a memory_consolidate growth event."""
    mock_memory.consolidate(n_sleep_cycles=5)
    meta = mock_memory._lineage.read_metadata()
    cons_events = [e for e in meta.growth_events
                     if e["kind"] == "memory_consolidate"]
    assert len(cons_events) >= 1
    assert cons_events[-1]["metadata"]["n_sleep_cycles"] == 5


def test_consolidate_caches_last_result(mock_memory):
    """consolidate() caches its result in _last_consolidation."""
    assert mock_memory._last_consolidation is None
    result = mock_memory.consolidate(n_sleep_cycles=2)
    assert mock_memory._last_consolidation is result


# ──────────────────────────────────────────────────────────────────────
# stats()
# ──────────────────────────────────────────────────────────────────────


def test_stats_returns_expected_schema(mock_memory):
    """stats() returns dict with required fields."""
    mock_memory.store("k", "v")
    s = mock_memory.stats()
    assert s["lineage_name"] == "test_memory"
    assert s["mode"] == "synonym"
    assert s["n_bindings_estimate"] == 1
    assert "cumulative_training_events" in s
    assert "vocab_size" in s
    assert "bridge_synapses" in s
    assert "bridge_neurons" in s
    assert s["stub_phase"] == "3.1"


def test_stats_includes_consolidation_when_run(mock_memory):
    """stats()['last_consolidation'] reflects most recent consolidate()."""
    s_before = mock_memory.stats()
    assert s_before["last_consolidation"] is None

    mock_memory.consolidate(n_sleep_cycles=3)
    s_after = mock_memory.stats()
    assert s_after["last_consolidation"] is not None
    assert "retention_ratio" in s_after["last_consolidation"]


# ──────────────────────────────────────────────────────────────────────
# list_keys()
# ──────────────────────────────────────────────────────────────────────


def test_list_keys_returns_mode_vocab(mock_memory):
    """list_keys() returns the static vocab for the mode."""
    keys = mock_memory.list_keys()
    assert isinstance(keys, list)
    # synonym mode has 8 words (4 primary + 4 synonyms)
    assert len(keys) == 8
    # primary direction words are always in there
    assert "north" in keys
    assert "east" in keys
    assert "south" in keys
    assert "west" in keys


# ──────────────────────────────────────────────────────────────────────
# save()
# ──────────────────────────────────────────────────────────────────────


def test_save_records_growth_event(mock_memory):
    """save() records a manual_save growth event."""
    path = mock_memory.save(growth_kind="manual_save", description="test save")
    assert path == mock_memory._lineage.current_path
    meta = mock_memory._lineage.read_metadata()
    saves = [e for e in meta.growth_events
              if e["kind"] == "manual_save"]
    assert len(saves) >= 1


# ──────────────────────────────────────────────────────────────────────
# End-to-end conversation flow
# ──────────────────────────────────────────────────────────────────────


def test_conversation_flow_records_all_events(mock_memory):
    """Simulate a multi-turn LLM interaction; verify lineage records."""
    # Turn 1: user introduces self
    mock_memory.store("user_name", "alice")
    mock_memory.store("user_role", "engineer")
    # Turn 2: LLM queries something it hasn't been told
    result = mock_memory.recall("favorite_color")
    assert result == []  # stub
    # Turn 3: user provides the answer
    mock_memory.store("favorite_color", "blue")
    # Session end: consolidate
    mock_memory.consolidate(n_sleep_cycles=3)

    meta = mock_memory._lineage.read_metadata()
    bind_count = sum(1 for e in meta.growth_events
                     if e["kind"] == "memory_bind")
    cons_count = sum(1 for e in meta.growth_events
                     if e["kind"] == "memory_consolidate")
    assert bind_count == 3
    assert cons_count == 1
