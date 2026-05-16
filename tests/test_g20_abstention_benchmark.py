"""Tests for the abstention-benchmark split (pure, CPU)."""
from __future__ import annotations

from research.runners.g20_abstention_benchmark import split_encoded_control

PAIRS = [(0, f"a{i}", 1, f"b{i}") for i in range(20)]


class TestSplit:
    def test_partitions_disjoint_and_cover(self):
        enc, ctrl = split_encoded_control(PAIRS, seed=42)
        assert len(enc) + len(ctrl) == len(PAIRS)
        assert set(map(tuple, enc)).isdisjoint(set(map(tuple, ctrl)))
        # union == all
        assert set(map(tuple, enc)) | set(map(tuple, ctrl)) == \
            set(map(tuple, PAIRS))

    def test_roughly_half(self):
        enc, ctrl = split_encoded_control(PAIRS, seed=42)
        assert abs(len(enc) - len(ctrl)) <= 1

    def test_deterministic(self):
        a = split_encoded_control(PAIRS, seed=7)
        b = split_encoded_control(PAIRS, seed=7)
        assert a == b

    def test_seed_varies_partition(self):
        a, _ = split_encoded_control(PAIRS, seed=7)
        b, _ = split_encoded_control(PAIRS, seed=8)
        assert a != b

    def test_handles_odd_count(self):
        odd = PAIRS[:15]
        enc, ctrl = split_encoded_control(odd, seed=1)
        assert len(enc) + len(ctrl) == 15
