"""Tests for the cross-bridge benchmark pair sampler (pure, CPU)."""
from __future__ import annotations

from research.runners.g20_xbridge_benchmark import sample_xbridge_pairs


VOCABS = [
    [f"a{i}" for i in range(20)],   # bridge 0
    [f"b{i}" for i in range(20)],   # bridge 1
    [f"c{i}" for i in range(20)],   # bridge 2
]


class TestSampler:
    def test_count_and_cross_bridge(self):
        pairs = sample_xbridge_pairs(VOCABS, 15, seed=42)
        assert len(pairs) == 15
        for ba, wa, bb, wb in pairs:
            assert ba != bb, "pairs must be cross-bridge"
            assert wa in VOCABS[ba] and wb in VOCABS[bb]
            assert wa != wb

    def test_deterministic(self):
        a = sample_xbridge_pairs(VOCABS, 12, seed=7)
        b = sample_xbridge_pairs(VOCABS, 12, seed=7)
        assert a == b

    def test_seed_varies(self):
        a = sample_xbridge_pairs(VOCABS, 12, seed=7)
        b = sample_xbridge_pairs(VOCABS, 12, seed=8)
        assert a != b

    def test_exclude_idx_drops_position(self):
        pairs = sample_xbridge_pairs(VOCABS, 30, seed=1, exclude_idx=12)
        for ba, wa, bb, wb in pairs:
            assert wa != VOCABS[ba][12]
            assert wb != VOCABS[bb][12]

    def test_exclude_none_keeps_all(self):
        pairs = sample_xbridge_pairs(VOCABS, 30, seed=1, exclude_idx=None)
        # idx-12 words are now eligible (probabilistically present over
        # 30 pairs across 60 positions); at minimum it must not crash
        # and must still be cross-bridge.
        for ba, wa, bb, wb in pairs:
            assert ba != bb

    def test_no_duplicate_pairs(self):
        pairs = sample_xbridge_pairs(VOCABS, 25, seed=99)
        assert len(pairs) == len(set(pairs))
