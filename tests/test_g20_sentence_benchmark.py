"""Tests for the 3-way sentence-benchmark triple sampler (pure, CPU)."""
from __future__ import annotations

from research.runners.g20_sentence_benchmark import sample_sentences

NAMES = ["bridgeA_nouns", "bridgeB_verbs", "bridgeC_adj"]
VOCABS = [
    [f"n{i}" for i in range(30)],
    [f"v{i}" for i in range(30)],
    [f"a{i}" for i in range(30)],
]


class TestSentenceSampler:
    def test_count_and_origin(self):
        s = sample_sentences(VOCABS, NAMES, 15, 42,
                              "bridgeA_nouns", "bridgeB_verbs",
                              "bridgeC_adj")
        assert len(s) == 15
        for subj, verb, obj in s:
            assert subj in VOCABS[0]
            assert verb in VOCABS[1]
            assert obj in VOCABS[2]
            assert len({subj, verb, obj}) == 3

    def test_deterministic(self):
        a = sample_sentences(VOCABS, NAMES, 10, 7, "bridgeA_nouns",
                              "bridgeB_verbs", "bridgeC_adj")
        b = sample_sentences(VOCABS, NAMES, 10, 7, "bridgeA_nouns",
                              "bridgeB_verbs", "bridgeC_adj")
        assert a == b

    def test_seed_varies(self):
        a = sample_sentences(VOCABS, NAMES, 10, 7, "bridgeA_nouns",
                              "bridgeB_verbs", "bridgeC_adj")
        b = sample_sentences(VOCABS, NAMES, 10, 8, "bridgeA_nouns",
                              "bridgeB_verbs", "bridgeC_adj")
        assert a != b

    def test_exclude_idx(self):
        s = sample_sentences(VOCABS, NAMES, 30, 1, "bridgeA_nouns",
                              "bridgeB_verbs", "bridgeC_adj",
                              exclude_idx=12)
        for subj, verb, obj in s:
            assert subj != VOCABS[0][12]
            assert verb != VOCABS[1][12]
            assert obj != VOCABS[2][12]

    def test_no_duplicate_triples(self):
        s = sample_sentences(VOCABS, NAMES, 25, 99, "bridgeA_nouns",
                              "bridgeB_verbs", "bridgeC_adj")
        assert len(s) == len(set(s))
