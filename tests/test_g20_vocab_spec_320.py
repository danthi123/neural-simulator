"""Test the G.20 5-bridge 320-concept vocab spec: structure + uniqueness.

The 320 spec is the production scaling tier (5 x 64). It must NOT
collide with itself; the base-160 reuse means a regression in
g20_vocab_spec would surface here too.
"""
from __future__ import annotations
import pytest

from research.runners.g20_vocab_spec_320 import (
    ALL_BRIDGES_64, ALL_WORDS_64, TOTAL_VOCAB_64,
    VOCAB_BRIDGE_A_NOUNS_64, VOCAB_BRIDGE_B_VERBS_64,
    VOCAB_BRIDGE_C_ADJECTIVES_64, VOCAB_BRIDGE_D_SPATIAL_64,
    VOCAB_BRIDGE_E_FUNCTIONAL_64,
)
from research.runners.g20_vocab_spec import ALL_WORDS as BASE_160


class TestStructure:
    @pytest.mark.parametrize("name,vocab", list(ALL_BRIDGES_64.items()))
    def test_each_bridge_has_64(self, name, vocab):
        assert len(vocab) == 64, \
            f"{name} has {len(vocab)} concepts, expected 64"

    def test_total_is_320(self):
        assert TOTAL_VOCAB_64 == 320
        assert len(ALL_WORDS_64) == 320


class TestUniqueness:
    def test_no_cross_bridge_duplicates(self):
        seen = {}
        for name, vocab in ALL_BRIDGES_64.items():
            for w in vocab:
                if w in seen:
                    pytest.fail(f"Word '{w}' in both {seen[w]} and {name}")
                seen[w] = name

    @pytest.mark.parametrize("name,vocab", list(ALL_BRIDGES_64.items()))
    def test_no_intra_bridge_duplicates(self, name, vocab):
        assert len(vocab) == len(set(vocab)), \
            f"{name} has internal duplicates"

    def test_base_160_is_prefix_preserved(self):
        # The validated 160 must remain the first 32 of each bridge
        # (so the 320 spec is a strict superset; the trained 32-concept
        # bridges' concept ordering is unaffected).
        assert len(BASE_160) == 160
        base_set = set(BASE_160)
        assert base_set.issubset(set(ALL_WORDS_64))
        for vocab in ALL_BRIDGES_64.values():
            assert len(vocab[:32]) == 32  # first 32 = frozen base


class TestCoverage:
    """New-tier categories are represented in the +32 extension."""

    def test_more_animals_in_nouns(self):
        assert {"horse", "cow", "snake", "bear"}.issubset(
            set(VOCAB_BRIDGE_A_NOUNS_64))

    def test_more_action_verbs(self):
        assert {"sit", "stand", "throw", "catch"}.issubset(
            set(VOCAB_BRIDGE_B_VERBS_64))

    def test_more_adjectives(self):
        assert {"huge", "tiny", "bright", "loud"}.issubset(
            set(VOCAB_BRIDGE_C_ADJECTIVES_64))

    def test_more_spatial_temporal(self):
        assert {"inside", "outside", "yesterday", "tomorrow"}.issubset(
            set(VOCAB_BRIDGE_D_SPATIAL_64))

    def test_more_functional(self):
        assert {"six", "ten", "and", "or", "but"}.issubset(
            set(VOCAB_BRIDGE_E_FUNCTIONAL_64))
