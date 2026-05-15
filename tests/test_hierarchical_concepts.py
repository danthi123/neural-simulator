"""Unit tests for path 3 hierarchical concept trees (Patterson 2007)."""
from __future__ import annotations
import pytest

from research.runners.hierarchical_concepts import (
    get_ancestors, get_descendants, is_a, common_ancestor,
    category_summary, DEFAULT_HIERARCHY,
)


class TestAncestors:
    """get_ancestors returns the path from concept to root."""

    def test_dog_ancestors(self):
        anc = get_ancestors("dog")
        assert "mammal" in anc
        assert "animal" in anc
        assert "living_thing" in anc
        assert "thing" in anc
        assert anc.index("mammal") < anc.index("animal")
        assert anc.index("animal") < anc.index("living_thing")

    def test_red_ancestors(self):
        anc = get_ancestors("red")
        assert anc == ["color", "property", "attribute"]

    def test_run_ancestors(self):
        anc = get_ancestors("run")
        assert anc == ["motion", "action", "event"]

    def test_unknown_concept_no_ancestors(self):
        assert get_ancestors("xyzzy") == []

    def test_root_concept_no_ancestors(self):
        # 'thing' is a root, has no parent
        assert get_ancestors("thing") == []


class TestDescendants:
    """get_descendants returns all concepts under a category."""

    def test_mammal_descendants(self):
        desc = set(get_descendants("mammal"))
        assert desc == {"dog", "cat", "person", "baby"}

    def test_color_descendants(self):
        desc = set(get_descendants("color"))
        assert desc == {"red", "blue"}

    def test_motion_descendants(self):
        desc = set(get_descendants("motion"))
        assert "go" in desc
        assert "come" in desc
        assert "walk" in desc
        assert "run" in desc
        assert "push" in desc
        assert "pull" in desc

    def test_thing_descendants_includes_subtree(self):
        """thing should descend to all object/animal/substance subtypes."""
        desc = set(get_descendants("thing"))
        assert "dog" in desc
        assert "apple" in desc
        assert "water" in desc
        assert "house" in desc


class TestIsA:
    """is_a transitively traverses parent links."""

    def test_dog_is_animal(self):
        assert is_a("dog", "animal")

    def test_dog_is_living_thing(self):
        assert is_a("dog", "living_thing")

    def test_dog_is_dog(self):
        # Reflexive: every concept is itself
        assert is_a("dog", "dog")

    def test_red_is_property(self):
        assert is_a("red", "property")

    def test_dog_not_action(self):
        assert not is_a("dog", "action")

    def test_red_not_animal(self):
        assert not is_a("red", "animal")


class TestCommonAncestor:
    """common_ancestor returns the nearest common parent."""

    def test_dog_cat_share_mammal(self):
        assert common_ancestor("dog", "cat") == "mammal"

    def test_red_blue_share_color(self):
        assert common_ancestor("red", "blue") == "color"

    def test_dog_apple_share_thing(self):
        # Both are descendants of 'thing'
        # dog -> mammal -> animal -> living_thing -> thing
        # apple -> food -> substance -> thing
        assert common_ancestor("dog", "apple") == "thing"

    def test_dog_run_no_common(self):
        # Different trees: dog (thing) vs run (event)
        assert common_ancestor("dog", "run") == ""

    def test_run_walk_share_motion(self):
        assert common_ancestor("run", "walk") == "motion"

    def test_self_is_common(self):
        assert common_ancestor("dog", "dog") == "dog"


class TestSummary:
    """category_summary statistics."""

    def test_summary_keys(self):
        s = category_summary()
        assert "n_concepts" in s
        assert "n_roots" in s
        assert "roots" in s
        assert "max_depth" in s

    def test_summary_has_three_roots(self):
        """Hierarchy organized around 3 roots: thing, event, attribute."""
        s = category_summary()
        roots = set(s["roots"])
        assert "thing" in roots
        assert "event" in roots
        assert "attribute" in roots

    def test_summary_concepts_count(self):
        """Should have ~80-95 concepts (60 vocab + categories)."""
        s = category_summary()
        assert s["n_concepts"] >= 80
        assert s["max_depth"] >= 4
