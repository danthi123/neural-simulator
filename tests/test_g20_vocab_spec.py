"""Test the G.20 5-bridge vocab spec: structure + uniqueness."""
from __future__ import annotations
import pytest

from research.runners.g20_vocab_spec import (
    ALL_BRIDGES, ALL_WORDS, TOTAL_VOCAB,
    VOCAB_BRIDGE_A_NOUNS, VOCAB_BRIDGE_B_VERBS,
    VOCAB_BRIDGE_C_ADJECTIVES, VOCAB_BRIDGE_D_SPATIAL,
    VOCAB_BRIDGE_E_FUNCTIONAL,
)


class TestStructure:
    """Each bridge has exactly 32 concepts."""

    @pytest.mark.parametrize("name,vocab", list(ALL_BRIDGES.items()))
    def test_each_bridge_has_32(self, name, vocab):
        assert len(vocab) == 32, \
            f"{name} has {len(vocab)} concepts, expected 32"

    def test_total_is_160(self):
        assert TOTAL_VOCAB == 160
        assert len(ALL_WORDS) == 160


class TestUniqueness:
    """No duplicates across or within bridges."""

    def test_no_cross_bridge_duplicates(self):
        seen = {}
        for name, vocab in ALL_BRIDGES.items():
            for w in vocab:
                if w in seen:
                    pytest.fail(
                        f"Word '{w}' in both {seen[w]} and {name}")
                seen[w] = name

    @pytest.mark.parametrize("name,vocab", list(ALL_BRIDGES.items()))
    def test_no_intra_bridge_duplicates(self, name, vocab):
        assert len(vocab) == len(set(vocab)), \
            f"{name} has internal duplicates"


class TestCoverage:
    """Spot-check that key conversational categories are represented."""

    def test_animals_in_nouns(self):
        animals = {"dog", "cat", "bird", "fish", "mouse"}
        assert animals.issubset(set(VOCAB_BRIDGE_A_NOUNS))

    def test_body_parts_in_nouns(self):
        body = {"hand", "foot", "head", "eye"}
        assert body.issubset(set(VOCAB_BRIDGE_A_NOUNS))

    def test_motion_in_verbs(self):
        motion = {"go", "come", "run", "walk"}
        assert motion.issubset(set(VOCAB_BRIDGE_B_VERBS))

    def test_perception_in_verbs(self):
        perception = {"look", "see", "hear", "listen"}
        assert perception.issubset(set(VOCAB_BRIDGE_B_VERBS))

    def test_colors_in_adjectives(self):
        colors = {"red", "blue", "green", "yellow"}
        assert colors.issubset(set(VOCAB_BRIDGE_C_ADJECTIVES))

    def test_size_in_adjectives(self):
        size = {"big", "small", "tall", "short"}
        assert size.issubset(set(VOCAB_BRIDGE_C_ADJECTIVES))

    def test_cardinal_directions_in_spatial(self):
        cardinal = {"north", "south", "east", "west"}
        assert cardinal.issubset(set(VOCAB_BRIDGE_D_SPATIAL))

    def test_relative_directions_in_spatial(self):
        relative = {"up", "down", "left", "right"}
        assert relative.issubset(set(VOCAB_BRIDGE_D_SPATIAL))

    def test_question_words_in_functional(self):
        qw = {"what", "where", "when", "who", "why", "how"}
        assert qw.issubset(set(VOCAB_BRIDGE_E_FUNCTIONAL))

    def test_yes_no_in_functional(self):
        yn = {"yes", "no"}
        assert yn.issubset(set(VOCAB_BRIDGE_E_FUNCTIONAL))

    def test_numbers_in_functional(self):
        numbers = {"one", "two", "three", "four", "five"}
        assert numbers.issubset(set(VOCAB_BRIDGE_E_FUNCTIONAL))
