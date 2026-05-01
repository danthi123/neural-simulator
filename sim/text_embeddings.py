"""Text embedding helpers for the brain-region framework's language regions.

For v1 we use a simple deterministic-random embedding scheme: each known
token gets a fixed Gaussian-random 256-dim vector keyed by hash(token),
so embeddings are deterministic across runs and don't require any
external corpus download.

This is intentionally a placeholder. v2 should swap to GloVe / word2vec /
Claude API embeddings, but the rest of the language pipeline (regions,
training, decoding) doesn't care about the embedding source — it just
needs `embed(token) -> np.ndarray[D]` and `nearest_token(activity, k)
-> List[str]`.

Design source: docs/plans/2026-05-01-text-interaction-design.md.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence

import numpy as np


# Default vocabulary for v1: cardinal directions + simple gridworld objects.
# Order is significant — used for fixed indices when embeddings are
# index-based (e.g., one-hot at vocab_size <= 256).
DEFAULT_VOCAB: List[str] = [
    # Cardinal direction action words
    "north", "east", "south", "west",
    # Synonyms for direction (so the agent can map multiple words to one action)
    "up", "right", "down", "left",
    "n", "e", "s", "w",
    # Object / scene words
    "goal", "agent", "wall", "empty", "landmark",
    # Quality / status words
    "yes", "no", "stop", "go", "near", "far",
    # Self-referential / meta
    "see", "what", "where", "how",
]


def _seed_for_token(token: str) -> int:
    """Hash-based deterministic seed for a token's embedding vector."""
    h = hashlib.sha256(token.encode("utf-8")).digest()
    # Use first 4 bytes as little-endian uint32 (modulo 2^31 for signed safety)
    return int.from_bytes(h[:4], "little") % (2**31)


def embed(token: str, dim: int = 256) -> np.ndarray:
    """Return a deterministic Gaussian-random embedding for `token`.

    Same `token` always produces same vector. Different tokens produce
    near-orthogonal vectors in expectation (since they're independent
    Gaussian draws).

    Args:
        token: lowercased word.
        dim: embedding dimensionality (must match language_input neuron count).

    Returns:
        np.ndarray of shape (dim,), L2-normalized to unit magnitude.
    """
    if not token:
        return np.zeros(dim, dtype=np.float32)
    rng = np.random.default_rng(_seed_for_token(token.lower()))
    v = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 1e-8:
        v = v / norm
    return v


def embed_batch(tokens: Iterable[str], dim: int = 256) -> np.ndarray:
    """Stack embed(token) for each token. Returns shape (N, dim)."""
    tokens = list(tokens)
    out = np.zeros((len(tokens), dim), dtype=np.float32)
    for i, t in enumerate(tokens):
        out[i] = embed(t, dim=dim)
    return out


def nearest_token(
    activity: np.ndarray,
    vocab: Sequence[str] = None,
    k: int = 1,
    dim: int = 256,
) -> List[str]:
    """Find the `k` tokens whose embeddings are most cosine-similar to
    the given activity vector.

    Args:
        activity: shape (dim,) — agent's language_output mean firing rate
            or any (dim,) feature vector.
        vocab: list of candidate tokens. Defaults to DEFAULT_VOCAB.
        k: number of top tokens to return.
        dim: embedding dimensionality.

    Returns:
        List of `k` tokens, ranked by cosine similarity (highest first).
    """
    if vocab is None:
        vocab = DEFAULT_VOCAB
    vocab = list(vocab)
    if activity.shape != (dim,):
        raise ValueError(
            f"nearest_token: activity shape {activity.shape} != ({dim},)"
        )
    # Normalize activity for cosine similarity
    a_norm = float(np.linalg.norm(activity))
    if a_norm < 1e-8:
        # All-zero activity — return top-k arbitrarily
        return list(vocab[:k])
    a = (activity / a_norm).astype(np.float32)

    embeddings = embed_batch(vocab, dim=dim)  # (V, dim), already L2-normalized
    sims = embeddings @ a  # (V,) — cosine similarity (since both L2-normalized)
    top_idx = np.argsort(-sims)[:k]
    return [vocab[int(i)] for i in top_idx]


def vocab_to_drive_pattern(
    token: str,
    n_neurons: int = 256,
    drive_max_pA: float = 200.0,
    sparsity: float = 0.1,
) -> np.ndarray:
    """Convert a token to an (n_neurons,) input current vector for the
    language_input region.

    Strategy: embed -> threshold the top `sparsity * n_neurons` components
    -> set those to drive_max_pA, rest to 0. Models a sparse population
    code where each token activates a small set of language_input neurons
    (rather than spreading drive across all 256).

    Sparse coding is biologically motivated (real cortical word
    representations are sparse) and avoids saturating the language_input
    region.
    """
    e = embed(token, dim=n_neurons)
    n_active = max(1, int(round(sparsity * n_neurons)))
    # Use top |e_i| (most-positive components) — Gaussian random, so this
    # picks ~10% of neurons quasi-randomly per token, but consistently
    # the SAME ~10% per token.
    top_idx = np.argsort(-e)[:n_active]
    drive = np.zeros(n_neurons, dtype=np.float32)
    drive[top_idx] = drive_max_pA
    return drive
