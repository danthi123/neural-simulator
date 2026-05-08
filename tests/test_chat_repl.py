"""Smoke tests for chat_repl helper functions (CPU-only, no GPU).

Tests vocab/mode mappings and CLI arg validation. The actual REPL loop +
training are GPU-bound and tested by smoke runs.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.runners.chat_repl import (
    VOCAB_TIER1, VOCAB_SYNONYM, VOCAB_SYNONYM_12, VOCAB_SYNONYM_16,
    WORD_TO_ACTION_SYNONYM, WORD_TO_ACTION_SYNONYM_12, WORD_TO_ACTION_SYNONYM_16,
    _vocab_for_mode,
)


def test_vocab_sizes_match_tier_levels():
    """Each vocab size matches the tier it represents."""
    assert len(VOCAB_TIER1) == 4
    assert len(VOCAB_SYNONYM) == 8
    assert len(VOCAB_SYNONYM_12) == 12
    assert len(VOCAB_SYNONYM_16) == 16


def test_synonym_vocab_supersets():
    """Larger vocabs strictly contain smaller ones."""
    assert VOCAB_TIER1 <= VOCAB_SYNONYM
    assert VOCAB_SYNONYM <= VOCAB_SYNONYM_12
    assert VOCAB_SYNONYM_12 <= VOCAB_SYNONYM_16


def test_word_to_action_consistency():
    """Each word maps to one of N/E/S/W; primaries map correctly."""
    valid_actions = {"N", "E", "S", "W"}
    primaries = {"north": "N", "east": "E", "south": "S", "west": "W"}

    for mapping in [WORD_TO_ACTION_SYNONYM, WORD_TO_ACTION_SYNONYM_12,
                     WORD_TO_ACTION_SYNONYM_16]:
        for word, action in mapping.items():
            assert action in valid_actions, f"{word} -> {action} invalid"
        for primary, expected in primaries.items():
            assert mapping[primary] == expected


def test_synonym_groups_4_per_action_in_16():
    """vocab=16 has exactly 4 synonyms per action."""
    counts = {"N": 0, "E": 0, "S": 0, "W": 0}
    for action in WORD_TO_ACTION_SYNONYM_16.values():
        counts[action] += 1
    assert counts == {"N": 4, "E": 4, "S": 4, "W": 4}


def test_unicode_arrows_in_16_word():
    """Unicode arrows are present in the 16-word vocab."""
    assert "↑" in VOCAB_SYNONYM_16
    assert "→" in VOCAB_SYNONYM_16
    assert "↓" in VOCAB_SYNONYM_16
    assert "←" in VOCAB_SYNONYM_16
    assert WORD_TO_ACTION_SYNONYM_16["↑"] == "N"
    assert WORD_TO_ACTION_SYNONYM_16["→"] == "E"


def test_vocab_for_mode_returns_correct_pair():
    """_vocab_for_mode returns the right (vocab, word_to_action) for each mode."""
    v_t1, m_t1 = _vocab_for_mode("tier1")
    assert v_t1 == VOCAB_TIER1
    assert m_t1["north"] == "N"

    v_s, m_s = _vocab_for_mode("synonym")
    assert v_s == VOCAB_SYNONYM
    assert m_s == WORD_TO_ACTION_SYNONYM

    v_12, m_12 = _vocab_for_mode("synonym12")
    assert v_12 == VOCAB_SYNONYM_12
    assert m_12 == WORD_TO_ACTION_SYNONYM_12

    v_16, m_16 = _vocab_for_mode("synonym16")
    assert v_16 == VOCAB_SYNONYM_16
    assert m_16 == WORD_TO_ACTION_SYNONYM_16


def test_vocab_for_mode_rejects_unknown():
    with pytest.raises(ValueError, match="unknown mode"):
        _vocab_for_mode("not_a_mode")


def test_synonym_groups_match_canonical_lookup():
    """chat_repl vocab must match the canonical lookup table."""
    import importlib
    te = importlib.import_module("research.runners.text_eval")
    # 8-word
    grp_8 = te.get_synonym_groups(8)
    expected_words_8 = {w for words in grp_8.values() for w in words}
    assert VOCAB_SYNONYM == expected_words_8

    # 12-word
    grp_12 = te.get_synonym_groups(12)
    expected_words_12 = {w for words in grp_12.values() for w in words}
    assert VOCAB_SYNONYM_12 == expected_words_12

    # 16-word
    grp_16 = te.get_synonym_groups(16)
    expected_words_16 = {w for words in grp_16.values() for w in words}
    assert VOCAB_SYNONYM_16 == expected_words_16

    # Mappings match
    eta_16 = te.get_extended_word_to_action(16)
    for word, action in WORD_TO_ACTION_SYNONYM_16.items():
        assert eta_16[word] == action
