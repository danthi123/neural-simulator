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


# ─────────────────────────────────────────────────────────────────────────
# _parse_learn_command — Track 3 online vocab learning (2026-05-09)
# ─────────────────────────────────────────────────────────────────────────


def test_parse_learn_command_basic():
    """The basic 'learn <word> <N|E|S|W>' form works."""
    from research.runners.chat_repl import _parse_learn_command
    assert _parse_learn_command("learn ahead N") == ("ahead", "N")
    assert _parse_learn_command("learn back S") == ("back", "S")
    assert _parse_learn_command("learn rightward E") == ("rightward", "E")


def test_parse_learn_command_word_aliases():
    """Action can be a direction word, a synonym, or a Unicode arrow."""
    from research.runners.chat_repl import _parse_learn_command
    # Full direction names
    assert _parse_learn_command("learn ahead north") == ("ahead", "N")
    assert _parse_learn_command("learn back south") == ("back", "S")
    # Synonyms (matches existing Tier 2.1 mapping)
    assert _parse_learn_command("learn forward up") == ("forward", "N")
    assert _parse_learn_command("learn lefty left") == ("lefty", "W")
    # Unicode arrows (matches Tier 2.3 16-word mapping)
    assert _parse_learn_command("learn whatever ↑") == ("whatever", "N")


def test_parse_learn_command_case_and_whitespace():
    """Word lowercases, action handles mixed case + extra whitespace."""
    from research.runners.chat_repl import _parse_learn_command
    assert _parse_learn_command("LEARN HELLO E") == ("hello", "E")
    assert _parse_learn_command("learn  TheWord  W") == ("theword", "W")
    assert _parse_learn_command("  learn nice n  ") == ("nice", "N")


def test_parse_learn_command_rejects_bad_input():
    """Invalid inputs return None — caller can show usage hint."""
    from research.runners.chat_repl import _parse_learn_command
    # Missing args
    assert _parse_learn_command("learn") is None
    assert _parse_learn_command("learn ahead") is None
    # Wrong command verb
    assert _parse_learn_command("teach ahead N") is None
    # Bad action
    assert _parse_learn_command("learn ahead nope") is None
    assert _parse_learn_command("learn ahead 5") is None
    # Empty word (in practice unreachable through split, but defensive)
    assert _parse_learn_command("learn  N") is None or \
           _parse_learn_command("learn  N") == ("n", "N")  # split eats blanks


def test_parse_learn_command_does_not_match_chat_inputs():
    """A user typing 'north' or 'east' shouldn't accidentally invoke learn."""
    from research.runners.chat_repl import _parse_learn_command
    assert _parse_learn_command("north") is None
    assert _parse_learn_command("east up") is None  # only 2 tokens
    assert _parse_learn_command("learnsomething weird") is None  # not 'learn '


# ─────────────────────────────────────────────────────────────────────────
# Dialog state commands (Track 3 layer 3, 2026-05-09)
# ─────────────────────────────────────────────────────────────────────────


def test_parse_dialog_command_recognized_verbs():
    """Each : -prefixed verb returns a structured dialog command."""
    from research.runners.chat_repl import _parse_dialog_command
    assert _parse_dialog_command(":again") == {"verb": "again"}
    assert _parse_dialog_command(":opposite") == {"verb": "opposite"}
    assert _parse_dialog_command(":history") == {"verb": "history", "n": 5}
    assert _parse_dialog_command(":forget") == {"verb": "forget"}


def test_parse_dialog_command_history_with_n():
    """':history N' returns the requested count clamped to a sane range."""
    from research.runners.chat_repl import _parse_dialog_command
    assert _parse_dialog_command(":history 10") == {"verb": "history", "n": 10}
    assert _parse_dialog_command(":history 1")  == {"verb": "history", "n": 1}
    # Clamped to [1, 50] so we don't print megabytes
    assert _parse_dialog_command(":history 0")["n"] == 1
    assert _parse_dialog_command(":history 999")["n"] == 50
    # Junk N falls back to default 5
    assert _parse_dialog_command(":history nope")["n"] == 5


def test_parse_dialog_command_rejects_non_dialog():
    """Plain words and other inputs return None."""
    from research.runners.chat_repl import _parse_dialog_command
    assert _parse_dialog_command("again") is None       # missing colon
    assert _parse_dialog_command("north") is None       # vocab word
    assert _parse_dialog_command(":") is None           # bare colon
    assert _parse_dialog_command(":unknownverb") is None  # unrecognized verb
    assert _parse_dialog_command("learn ahead N") is None  # learn is unprefixed


def test_parse_dialog_command_case_and_whitespace():
    """Whitespace + case insensitive on the verb."""
    from research.runners.chat_repl import _parse_dialog_command
    assert _parse_dialog_command("  :AGAIN  ") == {"verb": "again"}
    assert _parse_dialog_command(":Opposite") == {"verb": "opposite"}
    assert _parse_dialog_command(":HISTORY 3") == {"verb": "history", "n": 3}


def test_action_inverse_table():
    """Each NESW action maps to its opposite."""
    from research.runners.chat_repl import ACTION_OPPOSITE
    assert ACTION_OPPOSITE["N"] == "S"
    assert ACTION_OPPOSITE["S"] == "N"
    assert ACTION_OPPOSITE["E"] == "W"
    assert ACTION_OPPOSITE["W"] == "E"
    # Closed under inverse
    for a in ("N", "E", "S", "W"):
        assert ACTION_OPPOSITE[ACTION_OPPOSITE[a]] == a


def test_action_to_canonical_word():
    """Each action maps to a canonical primary direction word for echo."""
    from research.runners.chat_repl import ACTION_TO_PRIMARY_WORD
    assert ACTION_TO_PRIMARY_WORD["N"] == "north"
    assert ACTION_TO_PRIMARY_WORD["E"] == "east"
    assert ACTION_TO_PRIMARY_WORD["S"] == "south"
    assert ACTION_TO_PRIMARY_WORD["W"] == "west"
