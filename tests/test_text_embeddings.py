"""Tests for sim.text_embeddings — token I/O for language regions."""
from __future__ import annotations

import numpy as np
import pytest


def test_embed_deterministic():
    """Same token must always produce the same vector across calls."""
    from sim.text_embeddings import embed

    a = embed("north")
    b = embed("north")
    assert np.allclose(a, b)


def test_embed_different_tokens_different_vectors():
    """Different tokens must produce distinct embeddings (non-collision)."""
    from sim.text_embeddings import embed

    n = embed("north")
    e = embed("east")
    assert not np.allclose(n, e)
    # And cosine similarity should be near zero (orthogonal in expectation)
    cos = float(np.dot(n, e))  # both L2-normalized
    assert abs(cos) < 0.5, f"north vs east cosine={cos:.3f} too similar"


def test_embed_shape_and_norm():
    """Embeddings are 256-dim and L2-normalized to unit length."""
    from sim.text_embeddings import embed

    v = embed("goal")
    assert v.shape == (256,)
    assert v.dtype == np.float32
    norm = float(np.linalg.norm(v))
    assert abs(norm - 1.0) < 1e-5, f"norm={norm} not unit"


def test_embed_empty_string():
    """Empty string gets zero vector (avoids hash issues)."""
    from sim.text_embeddings import embed

    v = embed("")
    assert np.allclose(v, np.zeros(256))


def test_nearest_token_finds_self():
    """Embedding a token, then asking for nearest token to that vector,
    must return the original token."""
    from sim.text_embeddings import embed, nearest_token

    for token in ["north", "east", "goal", "agent"]:
        v = embed(token)
        got = nearest_token(v, k=1)
        assert got[0] == token, f"nearest({token}) = {got[0]} (expected {token})"


def test_nearest_token_top_k():
    """Top-k must return k items, ranked by similarity."""
    from sim.text_embeddings import embed, nearest_token, DEFAULT_VOCAB

    v = embed("north")
    top3 = nearest_token(v, k=3)
    assert len(top3) == 3
    assert top3[0] == "north"  # first must be exact match
    # The other two should also be in the vocabulary
    for t in top3:
        assert t in DEFAULT_VOCAB


def test_nearest_token_zero_activity_returns_some_tokens():
    """When activity is all-zeros, return arbitrary top-k (no crash)."""
    from sim.text_embeddings import nearest_token

    got = nearest_token(np.zeros(256, dtype=np.float32), k=2)
    assert len(got) == 2


def test_nearest_token_wrong_shape_raises():
    """Mismatched activity dim raises ValueError."""
    from sim.text_embeddings import nearest_token

    with pytest.raises(ValueError, match="shape"):
        nearest_token(np.zeros(128, dtype=np.float32))


def test_vocab_to_drive_pattern():
    """Drive pattern is sparse (~sparsity fraction active) and consistent
    across calls."""
    from sim.text_embeddings import vocab_to_drive_pattern

    drive = vocab_to_drive_pattern("north", n_neurons=256, sparsity=0.1)
    assert drive.shape == (256,)
    assert drive.dtype == np.float32
    n_active = int(np.sum(drive > 0))
    assert 20 <= n_active <= 30, f"n_active={n_active} not ~10% of 256"

    # Determinism: same token, same pattern
    drive2 = vocab_to_drive_pattern("north", n_neurons=256, sparsity=0.1)
    assert np.allclose(drive, drive2)


def test_vocab_to_drive_pattern_different_tokens_differ():
    """Different tokens produce different drive patterns (different
    active neuron sets in expectation)."""
    from sim.text_embeddings import vocab_to_drive_pattern

    d_north = vocab_to_drive_pattern("north")
    d_east = vocab_to_drive_pattern("east")
    # Active sets should be mostly disjoint (independent random patterns)
    overlap = int(np.sum((d_north > 0) & (d_east > 0)))
    n_active = int(np.sum(d_north > 0))
    # In expectation, overlap = sparsity^2 * n = 0.01 * 256 ≈ 2-3
    assert overlap < n_active // 2, (
        f"overlap={overlap} suggests too-similar drive patterns"
    )
