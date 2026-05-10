"""Smoke tests for text_eval vocab tier dispatcher.

Verifies the find-the-ceiling vocab support: 8/12/16/24/32/48/64/96/128/256.
Each tier should produce vocab_size words, all unique, mapping correctly
to N/E/S/W actions.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.runners.text_eval import (
    SYNONYM_GROUPS,
    SYNONYM_GROUPS_12,
    SYNONYM_GROUPS_16,
    SYNONYM_GROUPS_24,
    SYNONYM_GROUPS_32,
    SYNONYM_GROUPS_48,
    SYNONYM_GROUPS_64,
    SYNONYM_GROUPS_96,
    SYNONYM_GROUPS_128,
    SYNONYM_GROUPS_256,
    get_synonym_groups,
    get_extended_word_to_action,
)


ALL_TIERS = [8, 12, 16, 24, 32, 48, 64, 96, 128, 256]


@pytest.mark.parametrize("vocab_size", ALL_TIERS)
def test_dispatcher_returns_correct_size(vocab_size):
    """get_synonym_groups(N) returns exactly N words across N/E/S/W."""
    groups = get_synonym_groups(vocab_size)
    n_words = sum(len(v) for v in groups.values())
    assert n_words == vocab_size, (
        f"vocab_size={vocab_size} dispatched to dict with {n_words} words"
    )


@pytest.mark.parametrize("vocab_size", ALL_TIERS)
def test_all_words_unique(vocab_size):
    """No duplicate words within a vocab tier."""
    groups = get_synonym_groups(vocab_size)
    all_words = [w for syns in groups.values() for w in syns]
    assert len(set(all_words)) == len(all_words), (
        f"vocab_size={vocab_size} has duplicates: "
        f"{[w for w in all_words if all_words.count(w) > 1][:5]}"
    )


@pytest.mark.parametrize("vocab_size", ALL_TIERS)
def test_balanced_actions(vocab_size):
    """Each of N/E/S/W has exactly vocab_size/4 synonyms."""
    groups = get_synonym_groups(vocab_size)
    expected = vocab_size // 4
    for action in ("N", "E", "S", "W"):
        assert len(groups[action]) == expected, (
            f"vocab_size={vocab_size} action {action} has "
            f"{len(groups[action])} synonyms (expected {expected})"
        )


@pytest.mark.parametrize("vocab_size", ALL_TIERS)
def test_extended_word_to_action_consistent(vocab_size):
    """get_extended_word_to_action matches get_synonym_groups."""
    groups = get_synonym_groups(vocab_size)
    word_map = get_extended_word_to_action(vocab_size)
    for action, words in groups.items():
        for word in words:
            assert word_map[word] == action, (
                f"vocab_size={vocab_size}: word '{word}' maps to "
                f"{word_map.get(word)} but is in {action}'s synonym list"
            )


def test_primary_words_present_in_all_tiers():
    """north/east/south/west are always the first synonym in each action group."""
    primaries = {"N": "north", "E": "east", "S": "south", "W": "west"}
    for vocab_size in ALL_TIERS:
        groups = get_synonym_groups(vocab_size)
        for action, expected_primary in primaries.items():
            assert groups[action][0] == expected_primary, (
                f"vocab_size={vocab_size} action {action} primary is "
                f"'{groups[action][0]}', expected '{expected_primary}'"
            )


def test_smaller_tiers_are_prefix_of_larger():
    """Each smaller tier is a prefix of the next larger up to 64.

    Beyond 64, the numbered-variant generator may produce different
    orderings, so the prefix property is only guaranteed for the
    hand-curated tiers (8/12/16/24/32/48/64).
    """
    # Hand-curated tiers up to 64 should be strict supersets
    curated_tiers = [8, 12, 16, 24, 32, 48, 64]
    for i, smaller in enumerate(curated_tiers[:-1]):
        larger = curated_tiers[i + 1]
        small_groups = get_synonym_groups(smaller)
        large_groups = get_synonym_groups(larger)
        for action in ("N", "E", "S", "W"):
            small_words = small_groups[action]
            large_words = large_groups[action]
            for j, w in enumerate(small_words):
                assert large_words[j] == w, (
                    f"vocab_size={smaller}->{larger}: action {action} "
                    f"position {j}: smaller '{w}' vs larger '{large_words[j]}'"
                )


def test_numbered_variants_in_higher_tiers():
    """96/128/256 use numbered variants like 'north_05'."""
    for vocab_size in [96, 128, 256]:
        groups = get_synonym_groups(vocab_size)
        # At least one action has a numbered variant
        has_numbered = any(
            any("_" in w and any(c.isdigit() for c in w) for w in syns)
            for syns in groups.values()
        )
        assert has_numbered, (
            f"vocab_size={vocab_size}: expected numbered variants like "
            f"'north_05', none found"
        )


def test_unicode_arrows_in_16_and_above():
    """Tiers 16+ include Unicode arrows."""
    for vocab_size in [16, 24, 32, 48, 64, 96, 128, 256]:
        word_map = get_extended_word_to_action(vocab_size)
        assert "↑" in word_map, f"vocab_size={vocab_size} missing ↑"
        assert "→" in word_map, f"vocab_size={vocab_size} missing →"
        assert "↓" in word_map, f"vocab_size={vocab_size} missing ↓"
        assert "←" in word_map, f"vocab_size={vocab_size} missing ←"
        assert word_map["↑"] == "N"
        assert word_map["→"] == "E"
        assert word_map["↓"] == "S"
        assert word_map["←"] == "W"


def test_unsupported_vocab_size_falls_back_to_8():
    """Unsupported sizes (e.g. 100) fall back to 8-word base."""
    groups = get_synonym_groups(100)
    n_words = sum(len(v) for v in groups.values())
    assert n_words == 8, (
        f"vocab_size=100 should fall back to 8-word, got {n_words}"
    )
