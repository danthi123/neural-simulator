"""Unit tests for path 2 subword/morpheme tokenizer.

Tests morphological decomposition: prefix, suffix, irregular forms,
spelling repairs.
"""
from __future__ import annotations
import pytest

from research.runners.subword_tokenizer import (
    tokenize_word, tokenize_sentence,
    DEFAULT_ROOTS_60, IRREGULAR_PAST, IRREGULAR_PLURAL,
    get_combinatorial_vocab_estimate,
)


class TestBareRoots:
    """Roots in the dictionary tokenize to themselves."""

    @pytest.mark.parametrize("root", [
        "apple", "dog", "go", "big", "tree", "walk", "red",
        "house", "fire", "give", "tall",
        "person", "open", "happy", "food", "speak", "new",
    ])
    def test_root_is_singleton(self, root):
        assert tokenize_word(root, DEFAULT_ROOTS_60) == [root]


class TestRegularSuffixes:
    """Regular suffix decomposition (-s, -ing, -er)."""

    def test_plural_s(self):
        # 'dogs' -> [PLURAL, dog] via irregular table
        assert tokenize_word("dogs", DEFAULT_ROOTS_60) == ["PLURAL", "dog"]

    def test_continuous_ing(self):
        assert tokenize_word("running", DEFAULT_ROOTS_60) == ["run", "ing"]

    def test_continuous_doubled_consonant(self):
        """running -> run+ning -> run+ing (drop doubled n)"""
        assert tokenize_word("running", DEFAULT_ROOTS_60) == ["run", "ing"]
        assert tokenize_word("stopping", DEFAULT_ROOTS_60) == ["stop", "ing"]

    def test_continuous_simple(self):
        assert tokenize_word("reading", DEFAULT_ROOTS_60) == ["read", "ing"]
        assert tokenize_word("walking", DEFAULT_ROOTS_60) == ["walk", "ing"]
        assert tokenize_word("sleeping", DEFAULT_ROOTS_60) == ["sleep", "ing"]

    def test_comparative_er(self):
        assert tokenize_word("bigger", DEFAULT_ROOTS_60) == ["big", "er"]
        assert tokenize_word("smaller", DEFAULT_ROOTS_60) == ["small", "er"]
        assert tokenize_word("colder", DEFAULT_ROOTS_60) == ["cold", "er"]
        assert tokenize_word("hotter", DEFAULT_ROOTS_60) == ["hot", "er"]


class TestIrregularForms:
    """Irregular past + plural forms via lookup tables."""

    def test_irregular_past_ate(self):
        assert tokenize_word("ate", DEFAULT_ROOTS_60) == ["PAST", "eat"]

    def test_irregular_past_ran(self):
        assert tokenize_word("ran", DEFAULT_ROOTS_60) == ["PAST", "run"]

    def test_irregular_past_drank(self):
        assert tokenize_word("drank", DEFAULT_ROOTS_60) == ["PAST", "drink"]

    def test_irregular_past_went(self):
        assert tokenize_word("went", DEFAULT_ROOTS_60) == ["PAST", "go"]

    def test_irregular_plural_babies(self):
        assert tokenize_word("babies", DEFAULT_ROOTS_60) == ["PLURAL", "baby"]

    def test_irregular_plural_feet(self):
        assert tokenize_word("feet", DEFAULT_ROOTS_60) == ["PLURAL", "foot"]

    def test_irregular_plural_people(self):
        assert tokenize_word("people", DEFAULT_ROOTS_60) == ["PLURAL", "person"]


class TestPrefixDecomposition:
    """Prefix splits (un-X, re-X)."""

    def test_un_happy(self):
        assert tokenize_word("unhappy", DEFAULT_ROOTS_60) == ["un", "happy"]

    def test_un_clean(self):
        assert tokenize_word("unclean", DEFAULT_ROOTS_60) == ["un", "clean"]

    def test_no_false_prefix(self):
        """'red' should NOT decompose as [re, d]"""
        assert tokenize_word("red", DEFAULT_ROOTS_60) == ["red"]

    def test_no_false_prefix_2(self):
        """'reading' should be [read, ing] not [re, ading]"""
        assert tokenize_word("reading", DEFAULT_ROOTS_60) == ["read", "ing"]


class TestSentenceTokenization:
    """End-to-end sentence tokenization."""

    def test_simple_past_sentence(self):
        tokens = tokenize_sentence("the dog ran fast", DEFAULT_ROOTS_60)
        assert tokens == ["the", "dog", "PAST", "run", "fast"]

    def test_compound_sentence(self):
        tokens = tokenize_sentence("the dogs are running", DEFAULT_ROOTS_60)
        assert tokens == ["the", "PLURAL", "dog", "are", "run", "ing"]

    def test_comparative_sentence(self):
        tokens = tokenize_sentence(
            "trees are bigger than houses", DEFAULT_ROOTS_60)
        assert tokens == ["PLURAL", "tree", "are", "big", "er",
                           "than", "PLURAL", "house"]

    def test_negation_sentence(self):
        tokens = tokenize_sentence(
            "the unhappy person is sleeping", DEFAULT_ROOTS_60)
        assert tokens == ["the", "un", "happy", "person", "is",
                           "sleep", "ing"]

    def test_empty_sentence(self):
        assert tokenize_sentence("", DEFAULT_ROOTS_60) == []

    def test_unknown_word_passthrough(self):
        """Unknown surface words pass through unchanged."""
        tokens = tokenize_sentence("xyzzy plover", DEFAULT_ROOTS_60)
        assert tokens == ["xyzzy", "plover"]


class TestCombinatorialEstimate:
    """Vocab estimate calculations."""

    def test_estimate_keys(self):
        est = get_combinatorial_vocab_estimate(DEFAULT_ROOTS_60)
        assert "n_roots" in est
        assert "with_plural" in est
        assert "with_past_tense" in est
        assert "combined_max" in est

    def test_estimate_scaling(self):
        est = get_combinatorial_vocab_estimate(DEFAULT_ROOTS_60)
        # 64 roots × 6 morphological variations = 384
        assert est["combined_max"] >= est["n_roots"] * 4
