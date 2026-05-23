"""Unit tests for the multi-bridge helper.

The helper is a pure function but it is load-bearing: any drift in
vocab order, pattern shape, or per-bridge seed independence would
silently change every bridge's symbols. These tests pin: vocab match
against the existing g20_vocab_spec exactly, 32 sparse K-of-N
patterns per bridge with correct shape, determinism in seed,
per-bridge pattern independence, unknown-bridge rejection.
"""
import numpy as np
import pytest

from research.findings.raw.vocabulary_scaling_160ensemble_helpers import (
    bridge_vocab_and_patterns, BRIDGE_NAMES,
)
from research.runners.g20_vocab_spec import ALL_BRIDGES


def test_bridge_names_match_spec():
    assert sorted(BRIDGE_NAMES) == sorted(ALL_BRIDGES.keys())
    assert len(BRIDGE_NAMES) == 5


def test_returns_vocab_matching_spec_exactly():
    """The runner does NOT regenerate or re-order vocabs; the helper
    returns the spec's vocab verbatim, in spec order."""
    for name in BRIDGE_NAMES:
        vocab, _ = bridge_vocab_and_patterns(
            name, seed=42, n_pool=2000, k=100)
        assert vocab == list(ALL_BRIDGES[name])
        assert len(vocab) == 32


def test_returns_32_sparse_patterns_of_k_neurons():
    vocab, pats = bridge_vocab_and_patterns(
        "bridgeA_nouns", seed=42, n_pool=2000, k=100)
    assert len(pats) == 32
    for p in pats:
        assert len(p) == 100
        assert all(0 <= int(i) < 2000 for i in p)
        # No duplicate indices within a pattern.
        assert len(set(int(i) for i in p)) == 100


def test_deterministic_in_seed():
    v1, p1 = bridge_vocab_and_patterns(
        "bridgeA_nouns", seed=42, n_pool=2000, k=100)
    v2, p2 = bridge_vocab_and_patterns(
        "bridgeA_nouns", seed=42, n_pool=2000, k=100)
    assert v1 == v2
    assert [list(x) for x in p1] == [list(x) for x in p2]


def test_per_bridge_patterns_differ():
    """Each bridge's patterns are seeded with a per-bridge derivative
    of the base seed so the 5 bridges' pattern sets are decorrelated.
    bridgeA's patterns must not equal bridgeB's at the same base
    seed."""
    _, pA = bridge_vocab_and_patterns(
        "bridgeA_nouns", seed=42, n_pool=2000, k=100)
    _, pB = bridge_vocab_and_patterns(
        "bridgeB_verbs", seed=42, n_pool=2000, k=100)
    same = all(list(pA[i]) == list(pB[i]) for i in range(32))
    assert not same


def test_unknown_bridge_raises():
    with pytest.raises(ValueError):
        bridge_vocab_and_patterns(
            "not_a_bridge", seed=42, n_pool=2000, k=100)
