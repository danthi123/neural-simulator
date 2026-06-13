"""Test the G.20 32-cluster 2048-concept vocab spec: structure + uniqueness.

The 2048 spec is the production sharding tier for the dual/CLS learned-graded
cortex (32 spiking bridges x 64 concepts). Unlike the 320 spec (which shards
by part-of-speech), this spec shards by SEMANTIC super-cluster: each bridge's
64 concepts must be mutually similar (a taxonomic cluster) so within-bridge
generalization works. The import-time global-uniqueness assert is the
correctness net; this test pins structure, uniqueness, total, base reuse, and
the lowercase/no-whitespace surface-form invariant.
"""
from __future__ import annotations
import pytest

from research.runners.g20_vocab_spec_2048 import (
    ALL_CLUSTERS_2048, ALL_WORDS_2048, TOTAL_VOCAB_2048,
)
from research.runners.g20_vocab_spec_320 import ALL_WORDS_64 as BASE_320


EXPECTED_CLUSTER_NAMES = [
    "mammals",
    "birds",
    "fish_reptiles",
    "insects",
    "fruits",
    "vegetables",
    "prepared_foods",
    "drinks",
    "land_vehicles",
    "air_water_vehicles",
    "hand_tools",
    "machines",
    "clothing",
    "furniture",
    "buildings",
    "body_parts",
    "plants_trees",
    "weather_nature",
    "kinship_people",
    "motion_verbs",
    "perception_verbs",
    "communication_verbs",
    "manipulation_verbs",
    "emotion_states",
    "size_shape_adj",
    "color_adj",
    "texture_material_adj",
    "time_words",
    "spatial_words",
    "quantity_number_words",
    "question_discourse",
    "abstract_relations",
]


class TestStructure:
    def test_exactly_32_clusters(self):
        assert len(ALL_CLUSTERS_2048) == 32, \
            f"expected 32 clusters, got {len(ALL_CLUSTERS_2048)}"

    def test_expected_cluster_names_present(self):
        keys = list(ALL_CLUSTERS_2048.keys())
        assert keys == EXPECTED_CLUSTER_NAMES, (
            "cluster keys must match the 32 expected names in order; "
            f"got {keys}"
        )

    @pytest.mark.parametrize("name", EXPECTED_CLUSTER_NAMES)
    def test_each_cluster_has_64(self, name):
        vocab = ALL_CLUSTERS_2048[name]
        assert len(vocab) == 64, \
            f"{name} has {len(vocab)} concepts, expected 64"

    def test_total_is_2048(self):
        assert TOTAL_VOCAB_2048 == 2048
        assert len(ALL_WORDS_2048) == 2048


class TestUniqueness:
    @pytest.mark.parametrize("name", EXPECTED_CLUSTER_NAMES)
    def test_no_intra_cluster_duplicates(self, name):
        vocab = ALL_CLUSTERS_2048[name]
        assert len(vocab) == len(set(vocab)), (
            f"{name} has internal duplicates: "
            f"{sorted(w for w in vocab if vocab.count(w) > 1)}"
        )

    def test_no_cross_cluster_duplicates(self):
        seen = {}
        for name, vocab in ALL_CLUSTERS_2048.items():
            for w in vocab:
                if w in seen:
                    pytest.fail(
                        f"Word '{w}' in both {seen[w]} and {name}")
                seen[w] = name

    def test_global_uniqueness(self):
        assert len(ALL_WORDS_2048) == len(set(ALL_WORDS_2048)), (
            "duplicate words across clusters: "
            f"{sorted(w for w in ALL_WORDS_2048 if ALL_WORDS_2048.count(w) > 1)}"
        )


class TestReuse:
    def test_at_least_315_base_words_reused(self):
        base_set = set(BASE_320)
        flat_set = set(ALL_WORDS_2048)
        reused = base_set & flat_set
        missing = sorted(base_set - flat_set)
        assert len(reused) >= 315, (
            f"only {len(reused)}/320 base words reused; "
            f"missing: {missing}"
        )


class TestSurfaceForm:
    def test_all_words_lowercase_nonempty_no_whitespace(self):
        for name, vocab in ALL_CLUSTERS_2048.items():
            for w in vocab:
                assert isinstance(w, str), f"{name}: {w!r} not a str"
                assert w, f"{name}: empty word"
                assert w == w.lower(), f"{name}: {w!r} not lowercase"
                assert not any(c.isspace() for c in w), \
                    f"{name}: {w!r} contains whitespace"
