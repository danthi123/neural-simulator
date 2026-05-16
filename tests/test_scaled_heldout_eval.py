"""Pure-logic tests for the Inc-3 held-out evaluator's window
selection. The held-out gate is decision-relevant and anti-cheat:
the held-out windows MUST be provably disjoint (zero character
overlap) from the 2000 windows the scaled net trained on, else the
"held-out" loss is contaminated by memorized windows and the gate
is a cheat. These pin that invariant.
"""
import numpy as np
import pytest

from research.runners.scaled_heldout_eval import (
    reconstruct_train_starts, select_heldout_starts,
)


def test_reconstruct_train_starts_is_deterministic_and_matches_sampler():
    # make_seq_dataset draws one rng.integers(0, n_chars-seq_len-1)
    # per sample. Reconstruction must reproduce that exact stream.
    n_chars, seq_len, n = 100_000, 96, 500
    a = reconstruct_train_starts(n_chars, seq_len, n, seed=42)
    b = reconstruct_train_starts(n_chars, seq_len, n, seed=42)
    assert a == b
    # Independent direct replay of the documented sampler call.
    rng = np.random.default_rng(42)
    expected = [int(rng.integers(0, n_chars - seq_len - 1))
                for _ in range(n)]
    assert a == expected


def test_heldout_starts_zero_overlap_with_train():
    n_chars, seq_len = 200_000, 96
    train = reconstruct_train_starts(n_chars, seq_len, 2000, seed=42)
    train_set = set(train)
    ho = select_heldout_starts(
        n_chars, seq_len, train, n_heldout=1000,
        rng=np.random.default_rng(12345))
    assert len(ho) == 1000
    train_arr = np.array(sorted(train_set))
    for s in ho:
        assert 0 <= s <= n_chars - seq_len - 1
        # zero CHARACTER overlap: nearest train start is > seq_len away
        nearest = np.min(np.abs(train_arr - s))
        assert nearest > seq_len, (
            f"held-out start {s} overlaps a training window "
            f"(nearest train start {nearest} <= seq_len {seq_len})")


def test_heldout_starts_deterministic():
    n_chars, seq_len = 200_000, 96
    train = reconstruct_train_starts(n_chars, seq_len, 2000, seed=42)
    ho1 = select_heldout_starts(n_chars, seq_len, train, 800,
                                np.random.default_rng(7))
    ho2 = select_heldout_starts(n_chars, seq_len, train, 800,
                                np.random.default_rng(7))
    assert ho1 == ho2


def test_heldout_raises_if_insufficient_clean_space():
    # Tiny corpus fully covered by train exclusion -> cannot find
    # disjoint held-out; must raise, never silently return overlap.
    n_chars, seq_len = 400, 96
    train = list(range(0, n_chars - seq_len - 1, 5))  # dense
    with pytest.raises(ValueError):
        select_heldout_starts(n_chars, seq_len, train, 100,
                               np.random.default_rng(0))
