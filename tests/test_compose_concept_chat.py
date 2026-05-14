"""Smoke tests for compose_concept_chat.py REPL commands.

These tests verify the chat REPL's command parsing + bridge interaction
end-to-end. Heavy: each test loads a v16 bridge (~5s) and runs encoding
(~5s/pair). Run selectively in CI.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
V16_SEED_44_BRIDGE = REPO_ROOT / "research/findings/raw/g11_bg/concept_pool_demo/seed44_v16.simstate.h5"


def _run_chat(scripted: str, load_bridge: Path, pairs: str = "",
                timeout_s: int = 120) -> str:
    """Run compose_concept_chat with --scripted; return stdout."""
    cmd = [
        sys.executable, "-m", "research.runners.compose_concept_chat",
        "--load-bridge", str(load_bridge),
        "--seed", "44",
        "--n-lang-input", "2048",
        "--n-per-pool", "200",
        "--n-fs-per-pool", "24",
        "--n-words-for-orthogonal", "16",
        "--encoding-steps", "500",
        "--sparsity", "0.05",
        "--balanced-teacher-pA", "500.0",
        "--pairs", pairs,
        "--scripted", scripted,
    ]
    result = subprocess.run(
        cmd, cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=timeout_s,
        encoding="utf-8", errors="replace",
    )
    return result.stdout


@pytest.mark.skipif(not V16_SEED_44_BRIDGE.exists(),
                     reason="seed44_v16 bridge not present")
class TestChatReplCommands:
    """Smoke tests for chat REPL command parsing + execution.

    Each test runs the actual REPL via subprocess on a v16 bridge.
    Validates that commands produce the expected output structure.
    """

    def test_vocab_command(self):
        """vocab lists 12 concept words."""
        out = _run_chat("vocab", V16_SEED_44_BRIDGE)
        assert "vocab:" in out
        for word in ["apple", "dog", "cat", "big", "small", "hot", "cold"]:
            assert word in out, f"missing word {word} in vocab output"

    def test_tags_empty(self):
        """tags returns empty list when no pairs preloaded."""
        out = _run_chat("tags", V16_SEED_44_BRIDGE)
        assert "tags: []" in out or "tags: []" in out

    def test_remember_and_query(self):
        """remember encodes a pair, what is retrieves it."""
        out = _run_chat(
            "remember apple is big,tags,what is apple,is apple big",
            V16_SEED_44_BRIDGE
        )
        assert "[remembered: apple_big]" in out
        assert "apple_big" in out
        assert "matched 1 tag" in out
        assert "YES" in out  # 'is apple big' returns YES

    def test_is_a_b_negative(self):
        """is a b returns NO for untrained pair."""
        out = _run_chat("is apple cold", V16_SEED_44_BRIDGE)
        assert "NO" in out

    def test_describe_command(self):
        """describe synthesizes natural-language response."""
        out = _run_chat(
            "remember apple is big,remember apple is hot,describe apple",
            V16_SEED_44_BRIDGE
        )
        # Should say "apple is X and Y" or "apple is X, Y, and Z"
        assert "apple is" in out

    def test_intersection_query(self):
        """what is a and b returns shared associates."""
        out = _run_chat(
            "remember apple is big,remember cat is big,what is apple and cat",
            V16_SEED_44_BRIDGE
        )
        assert "intersection" in out

    def test_forget_command(self):
        """forget removes a tag."""
        out = _run_chat(
            "remember apple is big,tags,forget apple_big,tags",
            V16_SEED_44_BRIDGE
        )
        assert "[forgot: apple_big]" in out

    def test_unknown_word_handled(self):
        """remember with unknown word gives clear error."""
        out = _run_chat("remember banana is yellow", V16_SEED_44_BRIDGE)
        assert "unknown word" in out

    def test_describe_unknown(self):
        """describe of an unknown word gives clear response."""
        out = _run_chat("describe river", V16_SEED_44_BRIDGE)
        # river not encoded → "I don't know anything about 'river'."
        assert "don't know" in out
