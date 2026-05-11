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


# ─────────────────────────────────────────────────────────────────────────
# Generative decoder — Track 3 layer 4 (action → word, A→W direction)
# ─────────────────────────────────────────────────────────────────────────


def test_parse_speak_command_basic():
    """':speak <action>' parses cleanly with action aliases."""
    from research.runners.chat_repl import _parse_speak_command
    assert _parse_speak_command(":speak N") == "N"
    assert _parse_speak_command(":speak E") == "E"
    # Word aliases (same as :learn action-aliases for consistency)
    assert _parse_speak_command(":speak north") == "N"
    assert _parse_speak_command(":speak east") == "E"
    assert _parse_speak_command(":speak up") == "N"  # synonym alias
    assert _parse_speak_command(":speak right") == "E"
    assert _parse_speak_command(":speak down") == "S"
    assert _parse_speak_command(":speak left") == "W"
    # Unicode arrows
    assert _parse_speak_command(":speak ↑") == "N"


def test_parse_speak_command_case_and_whitespace():
    """Whitespace + case insensitive on action."""
    from research.runners.chat_repl import _parse_speak_command
    assert _parse_speak_command("  :SPEAK N  ") == "N"
    assert _parse_speak_command(":Speak East") == "E"


def test_parse_speak_command_rejects_bad_input():
    """Bad inputs return None — caller can show usage."""
    from research.runners.chat_repl import _parse_speak_command
    assert _parse_speak_command(":speak") is None       # missing action
    assert _parse_speak_command("speak N") is None      # missing colon
    assert _parse_speak_command(":speak nope") is None  # bad action
    assert _parse_speak_command(":speak 5") is None     # bad action
    assert _parse_speak_command(":otherverb N") is None  # not :speak


def test_cosine_similarity_basic():
    """_cosine_similarity is correct on simple vectors."""
    import numpy as np
    from research.runners.chat_repl import _cosine_similarity
    # Identical vectors → 1.0
    a = np.array([1.0, 2.0, 3.0])
    assert abs(_cosine_similarity(a, a) - 1.0) < 1e-6
    # Orthogonal → 0.0
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    assert abs(_cosine_similarity(b, c)) < 1e-6
    # Opposite → -1.0
    d = np.array([1.0, 1.0])
    assert abs(_cosine_similarity(d, -d) - (-1.0)) < 1e-6
    # Zero vector → 0.0 (defined that way to avoid div by zero)
    z = np.array([0.0, 0.0, 0.0])
    assert _cosine_similarity(z, a) == 0.0
    assert _cosine_similarity(a, z) == 0.0


def test_cosine_similarity_nonnegative_drive_patterns():
    """Realistic case: drive patterns are nonneg sparse vectors."""
    import numpy as np
    from research.runners.chat_repl import _cosine_similarity
    # Two sparse patterns with partial overlap
    p1 = np.zeros(100)
    p1[:10] = 1.0  # active in first 10 neurons
    p2 = np.zeros(100)
    p2[5:15] = 1.0  # active in 5-15
    sim = _cosine_similarity(p1, p2)
    # Overlap = 5 active neurons; ||p1|| = sqrt(10), ||p2|| = sqrt(10)
    # sim = 5 / (sqrt(10) * sqrt(10)) = 0.5
    assert abs(sim - 0.5) < 1e-6


def test_rank_words_by_similarity():
    """_rank_words_by_similarity produces a sorted list with best match first."""
    import numpy as np
    from research.runners.chat_repl import _rank_words_by_similarity
    # Spike pattern matches "north" perfectly, partially "up"
    spike = np.zeros(100)
    spike[:10] = 1.0
    word_patterns = {
        "north": spike.copy(),       # identical → sim 1.0
        "up": np.concatenate([spike[:5], np.zeros(95)]),  # partial
        "south": np.zeros(100),      # zero — sim 0.0
    }
    rankings = _rank_words_by_similarity(spike, word_patterns)
    # Top match is "north", "up" second, "south" last
    assert rankings[0][0] == "north"
    assert abs(rankings[0][1] - 1.0) < 1e-6
    assert rankings[1][0] == "up"
    assert rankings[2][0] == "south"

# ─────────────────────────────────────────────────────────────────────────
# Lineage integration (wiring + CLI flags) — added 2026-05-10 23:50
# ─────────────────────────────────────────────────────────────────────────


def test_lineage_save_helper_records_growth_event(tmp_path):
    """_lineage_save updates metadata: tier, arch, cumulative events, growth"""
    from research.runners.chat_repl import _lineage_save
    from sim.lineage import BridgeLineage

    class _FakeBridge:
        """Mock that satisfies _lineage_save's interface."""
        class _Cfg:
            num_neurons = 42288
        core_sim_config = _Cfg()
        actual_total_connections_n = 111929115

        def save_checkpoint(self, path):
            from pathlib import Path
            Path(path).write_text("fake-state", encoding="utf-8")

    bridge = _FakeBridge()
    lineage = BridgeLineage("unit_test", root=tmp_path)
    _lineage_save(
        lineage, bridge, mode="synonym", seed=42, n_train_events=400,
        kind="init", description="unit test",
        accuracy_metric="REPL in-vocab", accuracy_value=0.9,
        accuracy_context="session_n_turns=10",
    )
    assert lineage.exists()
    meta = lineage.read_metadata()
    assert meta.current_tier == "8-word"
    assert meta.arch["mode"] == "synonym"
    assert meta.arch["n_neurons"] == 42288
    assert meta.cumulative_training_events == 400
    assert any(e["kind"] == "init" for e in meta.growth_events)
    assert len(meta.accuracy_history) == 1
    assert meta.accuracy_history[0]["metric"] == "REPL in-vocab"
    assert abs(meta.accuracy_history[0]["value"] - 0.9) < 1e-9


def test_lineage_save_helper_uses_correct_tier_per_mode(tmp_path):
    """Each mode -> a distinct tier label in metadata."""
    from research.runners.chat_repl import _lineage_save
    from sim.lineage import BridgeLineage

    class _FakeBridge:
        class _Cfg:
            num_neurons = 0
        core_sim_config = _Cfg()
        def save_checkpoint(self, path):
            from pathlib import Path
            Path(path).write_text("fake", encoding="utf-8")

    cases = [
        ("tier1", "4-word"),
        ("synonym", "8-word"),
        ("synonym12", "12-word"),
        ("synonym16", "16-word"),
    ]
    for mode, expected_tier in cases:
        lineage = BridgeLineage(f"unit_{mode}", root=tmp_path)
        _lineage_save(lineage, _FakeBridge(), mode=mode, seed=42,
                       n_train_events=100, kind="init")
        meta = lineage.read_metadata()
        assert meta.current_tier == expected_tier, (
            f"mode={mode!r} should -> tier={expected_tier!r}, got {meta.current_tier!r}"
        )


def test_chat_repl_help_mentions_lineage_flags():
    """--lineage / --from-scratch / --fork-lineage are advertised."""
    import subprocess, sys as _sys, os as _os
    p = subprocess.run(
        [_sys.executable, "-m", "research.runners.chat_repl", "--help"],
        capture_output=True, text=True, timeout=30,
        cwd=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    )
    assert p.returncode == 0, p.stderr
    assert "--lineage" in p.stdout
    assert "--from-scratch" in p.stdout
    assert "--fork-lineage" in p.stdout

