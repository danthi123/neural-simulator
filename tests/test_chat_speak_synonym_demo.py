"""Smoke tests for chat_speak_synonym_demo (CPU-only, no GPU).

Verifies:
1. Module imports cleanly (catches structural bugs before GPU run)
2. Tier 2.1 8-word vocab + synonym helpers re-exported correctly
3. SYNONYM_GROUPS / ACTION_TO_PRIMARY / WORD_TO_ACTION consistency

Anything that requires a GPU/bridge is left to the runner's smoke
(launch via webapp + verify .json schema).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_module_imports():
    """Catches structural bugs (missing imports, broken composition) early."""
    from research.runners import chat_speak_synonym_demo as m
    # Public entry points
    assert hasattr(m, "run_chat_speak_synonym_demo")
    assert hasattr(m, "evaluate_a_to_w_synonym")
    assert hasattr(m, "evaluate_w_to_a_baseline_synonym")
    assert hasattr(m, "main")


def test_reexports_tier21_vocab():
    """Verify vocab + helpers are pulled from chat_synonym_demo."""
    from research.runners.chat_speak_synonym_demo import (
        SYNONYM_GROUPS, ALL_WORDS, WORD_TO_ACTION, ACTION_TO_PRIMARY,
    )
    assert len(SYNONYM_GROUPS) == 4  # N/E/S/W
    assert set(SYNONYM_GROUPS.keys()) == {"N", "E", "S", "W"}
    assert len(ALL_WORDS) == 8  # 4 actions × 2 synonyms
    assert WORD_TO_ACTION["north"] == "N"
    assert WORD_TO_ACTION["up"] == "N"
    assert WORD_TO_ACTION["right"] == "E"
    assert ACTION_TO_PRIMARY["N"] == "north"
    assert ACTION_TO_PRIMARY["E"] == "east"
    assert ACTION_TO_PRIMARY["S"] == "south"
    assert ACTION_TO_PRIMARY["W"] == "west"


def test_synonym_groups_two_words_per_action():
    """Each action has exactly 2 synonyms (primary + synonym)."""
    from research.runners.chat_speak_synonym_demo import SYNONYM_GROUPS
    for action, syns in SYNONYM_GROUPS.items():
        assert len(syns) == 2, f"action {action} has {len(syns)} synonyms"


def test_word_to_action_round_trip():
    """Every ALL_WORDS entry maps back to a valid action."""
    from research.runners.chat_speak_synonym_demo import (
        ALL_WORDS, WORD_TO_ACTION, SYNONYM_GROUPS,
    )
    for word in ALL_WORDS:
        action = WORD_TO_ACTION[word]
        assert word in SYNONYM_GROUPS[action]


def test_evaluate_a_to_w_synonym_signature():
    """evaluate_a_to_w_synonym takes (bridge, verbose=...)."""
    import inspect
    from research.runners.chat_speak_synonym_demo import evaluate_a_to_w_synonym
    sig = inspect.signature(evaluate_a_to_w_synonym)
    params = list(sig.parameters.keys())
    assert params[0] == "bridge"
    assert "verbose" in params


def test_run_chat_speak_synonym_demo_signature():
    """run_chat_speak_synonym_demo accepts the Tier 2.1 v4 scale-up args."""
    import inspect
    from research.runners.chat_speak_synonym_demo import run_chat_speak_synonym_demo
    sig = inspect.signature(run_chat_speak_synonym_demo)
    params = list(sig.parameters.keys())
    # Required: seed; optional: arch + training knobs
    assert "seed" in params
    assert "n_train_events" in params
    assert "n_lang_input" in params
    assert "n_motor_per_action" in params
    assert "n_motor_fs_per_action" in params

    # Default values match Tier 2.1 v4 scale-up
    assert sig.parameters["n_lang_input"].default == 4096
    assert sig.parameters["n_motor_per_action"].default == 1000
    assert sig.parameters["n_motor_fs_per_action"].default == 120
    assert sig.parameters["n_train_events"].default == 400
