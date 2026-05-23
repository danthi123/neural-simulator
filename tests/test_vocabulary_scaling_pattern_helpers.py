"""Unit tests for the pattern-grounded helpers.

`pattern_vector(pattern, n_pool)` is a tiny pure function but it is a
load-bearing one: the pattern-grounded runner's symbol-derivation step
passes its output through the same fixed deriver the activity-grounded
path uses, so any drift here would silently change every concept's
symbol. These tests pin shape, dtype, values, range-checking, and
determinism.
"""
import numpy as np
import pytest

from research.findings.raw.vocabulary_scaling_pattern_helpers import (
    pattern_vector,
)


def test_pattern_vector_basic_shape_and_values():
    v = pattern_vector([1, 3, 5], n_pool=8)
    assert v.shape == (8,)
    assert v.dtype == np.float64
    assert v.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]


def test_pattern_vector_full_size():
    """The realistic case: a 64-concept G.20 K-of-N pattern at K=100
    in N=2000 must produce a binary vector with exactly 100 ones."""
    pat = list(range(100))
    v = pattern_vector(pat, n_pool=2000)
    assert v.shape == (2000,)
    assert int(v.sum()) == 100
    ones = set(np.where(v > 0)[0].tolist())
    assert ones == set(pat)


def test_pattern_vector_rejects_out_of_range():
    with pytest.raises(ValueError):
        pattern_vector([0, 1, 2000], n_pool=2000)
    with pytest.raises(ValueError):
        pattern_vector([-1, 0, 1], n_pool=2000)


def test_pattern_vector_deterministic():
    v1 = pattern_vector([7, 13, 22], n_pool=64)
    v2 = pattern_vector([7, 13, 22], n_pool=64)
    assert np.array_equal(v1, v2)
