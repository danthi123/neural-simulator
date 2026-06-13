"""Tests for the frozen G.20 2048-concept within-cluster sub-taxonomy.

The sub-taxonomy (research/runners/g20_subtaxonomy_2048.py) partitions
EACH of the 32 semantic clusters from g20_vocab_spec_2048.ALL_CLUSTERS_2048
into 8 within-cluster semantic sub-groups of exactly 8 words each (8x8=64,
a complete partition per cluster).

The critical assertion is the EXACT partition: each cluster's 8 sub-groups
contain exactly the same 64 words as that cluster in the vocab spec, with no
additions, drops, or duplicates. A hand-curation slip fails here, never
silently.
"""
from __future__ import annotations

from research.runners.g20_subtaxonomy_2048 import (
    SUBTAXONOMY_2048,
    cluster_sublabels,
)
from research.runners.g20_vocab_spec_2048 import ALL_CLUSTERS_2048


def test_thirty_two_clusters_present_in_spec_order():
    """All 32 clusters present, in the exact order of ALL_CLUSTERS_2048."""
    assert list(SUBTAXONOMY_2048.keys()) == list(ALL_CLUSTERS_2048.keys())
    assert len(SUBTAXONOMY_2048) == 32


def test_each_cluster_has_exactly_eight_subgroups():
    for cname, subgroups in SUBTAXONOMY_2048.items():
        assert len(subgroups) == 8, (
            f"{cname} has {len(subgroups)} sub-groups, expected 8"
        )


def test_each_subgroup_has_exactly_eight_words():
    for cname, subgroups in SUBTAXONOMY_2048.items():
        for sgname, words in subgroups.items():
            assert len(words) == 8, (
                f"{cname}/{sgname} has {len(words)} words, expected 8"
            )


def test_subgroup_names_unique_within_cluster():
    for cname, subgroups in SUBTAXONOMY_2048.items():
        names = list(subgroups.keys())
        assert len(names) == len(set(names)), (
            f"{cname} has duplicate sub-group names: "
            f"{sorted(n for n in names if names.count(n) > 1)}"
        )


def test_exact_partition_of_vocab_spec_cluster():
    """THE critical net: each cluster's 8 sub-groups are an EXACT partition
    of that cluster's 64 words in g20_vocab_spec_2048 -- same set, no
    extras, no missing, no duplicates."""
    for cname, subgroups in SUBTAXONOMY_2048.items():
        spec_words = ALL_CLUSTERS_2048[cname]
        spec_set = set(spec_words)

        flat: list[str] = []
        for words in subgroups.values():
            flat.extend(words)

        # No duplicates within the cluster's partition.
        assert len(flat) == len(set(flat)), (
            f"{cname} sub-taxonomy has duplicate words: "
            f"{sorted(w for w in flat if flat.count(w) > 1)}"
        )
        # Exactly 64 words.
        assert len(flat) == 64, (
            f"{cname} sub-taxonomy has {len(flat)} words, expected 64"
        )
        # Same SET as the vocab-spec cluster.
        missing = spec_set - set(flat)
        extra = set(flat) - spec_set
        assert not missing, f"{cname} sub-taxonomy missing: {sorted(missing)}"
        assert not extra, f"{cname} sub-taxonomy has extras: {sorted(extra)}"


def test_global_total_is_2048():
    total = sum(
        len(words)
        for subgroups in SUBTAXONOMY_2048.values()
        for words in subgroups.values()
    )
    assert total == 2048


def test_all_words_lowercase_nonempty():
    for cname, subgroups in SUBTAXONOMY_2048.items():
        for sgname, words in subgroups.items():
            assert sgname and sgname.strip(), f"{cname} has empty sub-group name"
            for w in words:
                assert w and w.strip(), f"{cname}/{sgname} has an empty word"
                assert w == w.lower(), (
                    f"{cname}/{sgname} word not lowercase: {w!r}"
                )


def test_cluster_sublabels_shape_and_balance():
    """cluster_sublabels returns the 64 words + 64 sub-group ids in 0..7,
    with exactly 8 words per id (the within-cluster similarity block)."""
    for cname in SUBTAXONOMY_2048:
        words, sublabels = cluster_sublabels(cname)
        assert len(words) == 64, f"{cname}: {len(words)} words, expected 64"
        assert len(sublabels) == 64, (
            f"{cname}: {len(sublabels)} sublabels, expected 64"
        )
        assert set(sublabels) == set(range(8)), (
            f"{cname}: sublabels not 0..7: {sorted(set(sublabels))}"
        )
        for sid in range(8):
            cnt = sublabels.count(sid)
            assert cnt == 8, (
                f"{cname}: sub-group id {sid} has {cnt} members, expected 8"
            )
        # The returned words are exactly the cluster's 64 (as a set).
        assert set(words) == set(ALL_CLUSTERS_2048[cname])


def test_cluster_sublabels_words_match_concatenated_subgroups():
    """The word order from cluster_sublabels is the concatenation of the
    8 sub-groups in declaration order, and each word's sublabel is its
    sub-group index."""
    for cname, subgroups in SUBTAXONOMY_2048.items():
        words, sublabels = cluster_sublabels(cname)
        expected_words: list[str] = []
        expected_labels: list[int] = []
        for sid, sg_words in enumerate(subgroups.values()):
            expected_words.extend(sg_words)
            expected_labels.extend([sid] * len(sg_words))
        assert words == expected_words, f"{cname}: word order mismatch"
        assert sublabels == expected_labels, f"{cname}: label order mismatch"


def test_cluster_sublabels_rejects_unknown_cluster():
    import pytest

    with pytest.raises(KeyError):
        cluster_sublabels("not_a_real_cluster")
