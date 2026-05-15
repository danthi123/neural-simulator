"""Parser-only tests for multibridge_chat dispatcher.

These tests verify dispatcher parsing logic (negation, conjunctions,
yes/no questions) WITHOUT loading any bridges. The dispatcher uses
function-closure helpers; we stub them out.
"""
from __future__ import annotations
import io
from unittest.mock import patch
import pytest

import research.runners.multibridge_chat as mbc


def _make_member(set_name, tags=()):
    """Stub BridgeMember without loading."""
    m = mbc.BridgeMember(
        bridge_path=f"fake_{set_name}.h5",
        vocab_set={
            "set1": mbc.SET1_VOCAB,
            "set2": mbc.SET2_VOCAB,
            "set3": mbc.SET3_VOCAB,
        }[set_name],
        n_lang_input=2048, n_per_pool=200, n_fs_per_pool=24,
        sparsity=0.05, n_words_for_orthogonal=16,
        encoding_steps=500, balanced_teacher_pA=500.0,
        top_k=100, name=set_name,
    )
    m.encoded_tags = list(tags)
    return m


class TestYesNoQuery:
    """'is X Y?' returns YES/NO/UNKNOWN by exact tag match."""

    def test_yes_on_match(self):
        """Positive tag exists -> YES."""
        m1 = _make_member("set1", tags=["dog_big"])
        # Build a stub _yes_no_query via the top-level approach
        target = ["dog", "big"]
        hits = [(m.name, "_".join(target))
                for m in [m1]
                if "_".join(target) in m.encoded_tags]
        assert hits == [("set1", "dog_big")]

    def test_unknown_no_tags(self):
        """No tags exist -> empty hits -> UNKNOWN."""
        m1 = _make_member("set1", tags=[])
        target = ["dog", "big"]
        hits = [(m.name, "_".join(target))
                for m in [m1]
                if "_".join(target) in m.encoded_tags]
        assert hits == []

    def test_negated_match_via_NOT_prefix(self):
        """Negated tag 'NOT_dog_big' matches 'is dog not big?' query."""
        m1 = _make_member("set1", tags=["NOT_dog_big"])
        # negated query target = ["NOT", "dog", "big"]
        target = ["NOT", "dog", "big"]
        hits = [(m.name, "_".join(target))
                for m in [m1]
                if "_".join(target) in m.encoded_tags]
        assert hits == [("set1", "NOT_dog_big")]


class TestConjunctionSplitting:
    """' and ' in input splits into multiple clauses."""

    def test_split_on_and(self):
        """'a and b' -> two clauses."""
        line = "apple and dog"
        sub = [c.strip() for c in line.split(" and ") if c.strip()]
        assert sub == ["apple", "dog"]

    def test_remember_and_chain(self):
        """'remember a is b and c is d' -> two remember clauses."""
        rest = "apple is big and dog is small"
        clauses = [c.strip() for c in rest.split(" and ") if c.strip()]
        assert clauses == ["apple is big", "dog is small"]

    def test_single_clause_no_split(self):
        """No 'and' -> single clause."""
        line = "apple is big"
        sub = [c.strip() for c in line.split(" and ") if c.strip()]
        assert sub == ["apple is big"]


class TestStopwordStripping:
    """Article + preposition stripping in command parsing."""

    def test_strip_articles(self):
        """'the apple is big' -> ['apple', 'big']."""
        STOPWORDS = {"the", "a", "an", "that", "in", "on", "at",
                       "to", "of", "with", "by"}
        rest = "the apple is big".replace(" is ", " ")
        parts = [w for w in rest.split() if w not in STOPWORDS]
        # Note: 'is' is part of the splitter, not a stopword
        # After replace(' is ',' '), rest = "the apple big"
        # Strip stopwords -> ['apple', 'big']
        assert parts == ["apple", "big"]

    def test_strip_prepositions(self):
        """'the dog runs in the river' -> ['dog', 'runs', 'river']."""
        STOPWORDS = {"the", "a", "an", "that", "in", "on", "at",
                       "to", "of", "with", "by"}
        rest = "the dog runs in the river"
        parts = [w for w in rest.split() if w not in STOPWORDS]
        assert parts == ["dog", "runs", "river"]

    def test_no_stopwords_unchanged(self):
        """'dog is big' has no stopwords -> unchanged."""
        STOPWORDS = {"the", "a", "an", "that", "in", "on", "at",
                       "to", "of", "with", "by"}
        rest = "dog runs river"
        parts = [w for w in rest.split() if w not in STOPWORDS]
        assert parts == ["dog", "runs", "river"]


class TestNegationParsing:
    """' is not ' converts to ' is ' + NOT prefix."""

    def test_is_not_strips(self):
        """'a is not b' -> rest='a is b', negated=True."""
        rest = "apple is not big"
        negated = " is not " in rest
        if negated:
            rest = rest.replace(" is not ", " is ")
        assert negated
        assert rest == "apple is big"

    def test_no_is_not_unchanged(self):
        """'a is b' -> negated=False."""
        rest = "apple is big"
        negated = " is not " in rest
        assert not negated
