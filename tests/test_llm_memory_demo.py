"""Smoke test for research.runners.llm_memory_demo.

Tests the end-to-end Path 3 Phase 3.2 demo: MockLLM → orchestrator →
BridgeMemory → real SimulationBridge → BridgeLineage. Uses a single-turn
script to keep wall time reasonable (~3-5s on CPU).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.slow
def test_llm_memory_demo_end_to_end():
    """End-to-end: MockLLM stores a fact through the real bridge stack."""
    # Force NumPy backend for CI portability
    os.environ.setdefault("SIM_BACKEND", "numpy")

    from research.runners.llm_memory_demo import run_llm_demo

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "demo.json"
        lineage_root = Path(tmpdir) / "lineages"

        result = run_llm_demo(
            seed=42,
            lineage_name="smoke_test",
            lineage_root=lineage_root,
            out_path=out_path,
            script=["Remember that my favorite is north."],
            verbose=False,
        )

        # Result schema
        assert result["seed"] == 42
        assert result["lineage_name"] == "smoke_test"
        assert result["bridge_neurons"] > 0
        assert result["bridge_synapses"] > 0
        assert result["build_seconds"] > 0
        assert result["n_turns"] == 1
        # 1 user + 1 tool + 1 assistant = 3 messages
        assert result["n_messages"] == 3
        # 1 store call expected
        assert result["tool_call_counts"]["memory_store"] == 1
        assert result["tool_call_counts"]["memory_recall"] == 0
        assert result["tool_call_counts"]["memory_speak"] == 0

        # Per-turn payload
        turn = result["turns"][0]
        assert "favorite" in turn["user"].lower()
        assert turn["tools_called"] == ["memory_store"]
        assert "remember" in turn["assistant"].lower() or \
                "got it" in turn["assistant"].lower()

        # JSON output exists + parses
        assert out_path.exists()
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["seed"] == 42
        assert loaded["tool_call_counts"]["memory_store"] == 1


@pytest.mark.slow
def test_llm_memory_demo_multi_turn():
    """Multi-turn: store then recall in the same session."""
    os.environ.setdefault("SIM_BACKEND", "numpy")

    from research.runners.llm_memory_demo import run_llm_demo

    with tempfile.TemporaryDirectory() as tmpdir:
        lineage_root = Path(tmpdir) / "lineages"

        result = run_llm_demo(
            seed=42,
            lineage_name="smoke_multi",
            lineage_root=lineage_root,
            script=[
                "Remember that my favorite is north.",
                "What's my favorite?",
            ],
            verbose=False,
        )

        assert result["n_turns"] == 2
        # 2× (user+tool+assistant) = 6 messages
        assert result["n_messages"] == 6
        assert result["tool_call_counts"]["memory_store"] == 1
        assert result["tool_call_counts"]["memory_recall"] == 1

        # Turn 1: store
        assert result["turns"][0]["tools_called"] == ["memory_store"]
        # Turn 2: recall
        assert result["turns"][1]["tools_called"] == ["memory_recall"]
