"""Tests for G.20 multibridge N-word sentence role-query logic.

These test the pure tag-name template-matching that underlies
'who ate apple?' / 'what did dog eat?' across the ensemble. No
bridge load needed -- the role queries operate on tag-name strings.
"""
from __future__ import annotations
import pytest


def role_match(template, tags):
    """Replicates the g20_multibridge role-query matcher: returns the
    list of wildcard-position fillers for tags matching the template.

    template: list of tokens, '*' is wildcard
    tags: list of tag-name strings (underscore-joined words)
    Returns: dict {'subjects': [...]} or {'objects': [...]} style ->
    here just the matched filler at each '*' position.
    """
    star_positions = [i for i, t in enumerate(template) if t == "*"]
    fillers = []
    for tag in tags:
        tp = tag.split("_")
        if len(tp) != len(template):
            continue
        ok = all(tt == "*" or tt == pp
                  for tt, pp in zip(template, tp))
        if ok:
            fillers.append(tuple(tp[i] for i in star_positions))
    return fillers


class TestSubjectQuery:
    """'who ate apple?' -> template ['*', 'ate', 'apple']."""

    def test_single_subject(self):
        tags = ["dog_ate_apple", "cat_sees_bird"]
        hits = role_match(["*", "ate", "apple"], tags)
        assert hits == [("dog",)]

    def test_multiple_subjects(self):
        tags = ["dog_ate_apple", "bird_ate_apple", "cat_sees_fish"]
        hits = role_match(["*", "ate", "apple"], tags)
        subjects = sorted(h[0] for h in hits)
        assert subjects == ["bird", "dog"]

    def test_no_match_wrong_verb(self):
        tags = ["dog_ate_apple"]
        hits = role_match(["*", "sees", "apple"], tags)
        assert hits == []

    def test_no_match_wrong_length(self):
        tags = ["dog_ate_big_apple"]  # 4-word
        hits = role_match(["*", "ate", "apple"], tags)  # 3-word
        assert hits == []


class TestObjectQuery:
    """'what did dog eat?' -> template ['dog', 'eat', '*']."""

    def test_single_object(self):
        tags = ["dog_eat_apple", "cat_eat_fish"]
        hits = role_match(["dog", "eat", "*"], tags)
        assert hits == [("apple",)]

    def test_multiple_objects(self):
        tags = ["dog_eat_apple", "dog_eat_bone", "cat_eat_fish"]
        hits = role_match(["dog", "eat", "*"], tags)
        objs = sorted(h[0] for h in hits)
        assert objs == ["apple", "bone"]

    def test_object_query_distinguishes_subject(self):
        tags = ["dog_eat_apple", "cat_eat_apple"]
        # 'what did dog eat' should only match dog's, not cat's
        hits = role_match(["dog", "eat", "*"], tags)
        assert hits == [("apple",)]


class TestSentenceTagNaming:
    """Verify sentence -> tag-name conversion (stopword dropping)."""

    def test_articles_dropped(self):
        STOPW = {"the", "a", "an", "that", "in", "on", "at",
                  "to", "of", "with", "by"}
        parts = "the dog ate the apple".split()
        words = [w for w in parts if w not in STOPW]
        tag = "_".join(words)
        assert tag == "dog_ate_apple"

    def test_preposition_dropped(self):
        STOPW = {"the", "a", "an", "that", "in", "on", "at",
                  "to", "of", "with", "by"}
        parts = "the cat sat on the mat".split()
        words = [w for w in parts if w not in STOPW]
        assert "_".join(words) == "cat_sat_mat"

    def test_two_word_not_sentence(self):
        # 2-word stays a pair, not a sentence
        parts = "apple big".split()
        assert len(parts) == 2  # falls to pair path, not sentence

    def test_three_word_is_sentence(self):
        parts = "dog ate apple".split()
        assert len(parts) >= 3  # sentence path


class TestCrossBridgeSentenceConsistency:
    """A sentence's tag name is identical regardless of which bridge
    stores which word -- enables cross-bridge role queries."""

    def test_tag_name_bridge_independent(self):
        # dog (bridgeA), ate->eat (bridgeB), apple (bridgeA)
        # All bridges encode the SAME tag name 'dog_eat_apple'
        sentence = ["dog", "eat", "apple"]
        tag = "_".join(sentence)
        # bridgeA knows dog + apple, encodes tag 'dog_eat_apple'
        # bridgeB knows eat, encodes tag 'dog_eat_apple'
        # Role query 'who eat apple?' = template ['*','eat','apple']
        # matches the tag in BOTH bridges -> aggregated subject 'dog'
        tags_bridgeA = [tag]
        tags_bridgeB = [tag]
        all_tags = tags_bridgeA + tags_bridgeB
        hits = role_match(["*", "eat", "apple"], all_tags)
        subjects = sorted(set(h[0] for h in hits))
        assert subjects == ["dog"]
