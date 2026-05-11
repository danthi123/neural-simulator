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


def test_value_to_action_letter_passthrough(mock_memory):
    """value='N' / 'E' / 'S' / 'W' maps to itself."""
    assert mock_memory._value_to_action("N") == "N"
    assert mock_memory._value_to_action("e") == "E"
    assert mock_memory._value_to_action("South") == "S"
    assert mock_memory._value_to_action("w") == "W"


def test_value_to_action_primary_words(mock_memory):
    """value='north' etc. maps via vocab table."""
    assert mock_memory._value_to_action("north") == "N"
    assert mock_memory._value_to_action("east") == "E"
    assert mock_memory._value_to_action("South") == "S"
    assert mock_memory._value_to_action("WEST") == "W"


def test_value_to_action_synonyms(mock_memory):
    """In synonym mode, 'up'/'right'/'down'/'left' map to N/E/S/W."""
    # mock_memory is mode="synonym"
    assert mock_memory._value_to_action("up") == "N"
    assert mock_memory._value_to_action("right") == "E"
    assert mock_memory._value_to_action("down") == "S"
    assert mock_memory._value_to_action("left") == "W"


def test_value_to_action_rejects_unknown(mock_memory):
    """Unknown values raise ValueError."""
    with pytest.raises(ValueError, match="doesn't map to N/E/S/W"):
        mock_memory._value_to_action("blueberry")


# The store/recall tests that exercise the real bridge are in
# tests/test_numpy_backend_integration.py — they need a real
# SimulationBridge built under SIM_BACKEND=numpy. Here we test the
# scaffolding around them.


# ──────────────────────────────────────────────────────────────────────
# recall() — Phase 3.1 stub
# ──────────────────────────────────────────────────────────────────────


def test_recall_does_not_record_growth_event(mock_memory):
    """recall() is read-only; no growth event recorded.

    Note: recall against a mock bridge can't return real results (it
    would call chat_inference on a non-functional bridge). We just
    verify it doesn't add growth events.
    """
    initial_meta = mock_memory._lineage.read_metadata()
    initial_count = len(initial_meta.growth_events)
    try:
        mock_memory.recall("test")
    except Exception:
        # Mock bridge can't actually run chat_inference; that's fine
        pass
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


def test_stats_returns_expected_schema(mock_memory, monkeypatch):
    """stats() returns dict with required fields."""
    # Patch learn_word_pairing + chat_inference so store() works on the
    # mock bridge without GPU
    import research.runners.chat_repl as cr
    monkeypatch.setattr(cr, "learn_word_pairing",
                          lambda b, **kw: {"n_events_run": 50})
    monkeypatch.setattr(cr, "chat_inference",
                          lambda b, w: {"predicted_action": "N",
                                          "confidence_ratio": 1.5})
    mock_memory.store("alice", "north")
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


def test_conversation_flow_records_all_events(mock_memory, monkeypatch):
    """Simulate a multi-turn LLM interaction; verify lineage records.

    Patches learn_word_pairing + chat_inference so store() works
    against the mock bridge. Values are direction words (today's
    bridge has 4 motor pools).
    """
    import research.runners.chat_repl as cr
    monkeypatch.setattr(cr, "learn_word_pairing",
                          lambda b, **kw: {"n_events_run": 50})
    monkeypatch.setattr(cr, "chat_inference",
                          lambda b, w: {"predicted_action": "N",
                                          "confidence_ratio": 1.5,
                                          "delta_counts": {"N": 50, "E": 10,
                                                            "S": 5, "W": 2}})

    # Turn 1: bind 3 cues to direction values
    mock_memory.store("alice", "north")
    mock_memory.store("engineer", "east")
    # Turn 2: LLM queries (recall path doesn't need patching here
    # because it returns based on delta_counts from chat_inference)
    result = mock_memory.recall("alice")
    assert isinstance(result, list)
    # Turn 3: another bind
    mock_memory.store("blue_color", "south")
    # Session end: consolidate
    mock_memory.consolidate(n_sleep_cycles=3)

    meta = mock_memory._lineage.read_metadata()
    bind_count = sum(1 for e in meta.growth_events
                     if e["kind"] == "memory_bind")
    cons_count = sum(1 for e in meta.growth_events
                     if e["kind"] == "memory_consolidate")
    assert bind_count == 3
    assert cons_count == 1
