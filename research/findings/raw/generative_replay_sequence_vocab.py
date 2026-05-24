"""Sequence vocabulary helper for the generative-replay loop (Task 1 of
`docs/plans/2026-05-24-generative-replay-implementation.md`).

Exposes one function:

    generate_k_stored_sequences(seed, k, n_words, slot_count)

which produces K ordered sequences of length `slot_count`, each drawn
WITHOUT REPLACEMENT from the V-word vocabulary. The default vocabulary
is the v16 16-word concept-pool vocab (DIRECTION + NOUN + VERB +
ADJECTIVE) from `research.runners.concept_pool_demo` -- reused by
import per the standing reuse-by-import discipline.

Discipline guarantees:

- Determinism: same `(seed, k, n_words, slot_count)` produces a
  byte-identical list of tuples across runs. The RNG seeding uses
  numpy.random.default_rng(seed), which is reproducible across
  numpy versions per the documented PCG64 stream.
- No within-sequence repeats: each sequence's items are pairwise
  distinct (enforced by sampling WITHOUT REPLACEMENT via
  rng.choice(replace=False)).
- Inter-sequence diversity: no two returned sequences are equal as
  ordered tuples. Enforced by rejection sampling -- candidate
  sequences whose ordered tuple has already been emitted are
  discarded and re-sampled. A hard cap on rejection attempts raises
  RuntimeError so a pathological caller can't loop forever.
- Vocab consistency: the default 16-word vocabulary is the
  concatenation of the v16 pools in declaration order:
    DIRECTION ("north","east","south","west")
    NOUN ("apple","river","dog","cat")
    VERB ("go","come","stop","look")
    ADJECTIVE ("big","small","hot","cold")
  This matches the validated v14/v16 16-pool architecture.

Edge cases (all raise ValueError for caller-correctness; or
RuntimeError for the rejection-sampling cap):

- slot_count < 1 -> ValueError
- k < 1 -> ValueError
- n_words < slot_count -> ValueError (can't draw slot_count items
  without replacement from fewer than slot_count words)
- k > number of ordered tuples = perm(n_words, slot_count)
  -> ValueError (rejection sampling can never satisfy)
- rejection cap exceeded -> RuntimeError (defensive)

Plain ASCII only. No autograd, no protected/frozen/moat module
modified, no torch.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

# Reuse-by-import: the v16 vocabulary is canonical in
# concept_pool_demo. We import the per-kind dicts and concatenate
# in declaration order so the default 16-word vocab is exactly the
# v16 production set.
from research.runners.concept_pool_demo import (
    ADJECTIVE_VOCAB,
    DIRECTION_VOCAB,
    NOUN_VOCAB,
    VERB_VOCAB,
)


# ---------------------------------------------------------------------
# Canonical default 16-word vocabulary (v16 production set).
# ---------------------------------------------------------------------

V16_DEFAULT_VOCAB: Tuple[str, ...] = (
    *DIRECTION_VOCAB.keys(),  # ("north","east","south","west")
    *NOUN_VOCAB.keys(),       # ("apple","river","dog","cat")
    *VERB_VOCAB.keys(),       # ("go","come","stop","look")
    *ADJECTIVE_VOCAB.keys(),  # ("big","small","hot","cold")
)
assert len(V16_DEFAULT_VOCAB) == 16, (
    "V16_DEFAULT_VOCAB drifted from 16 words; check that the v16 pool "
    "constants in concept_pool_demo still total 4+4+4+4 = 16")
assert len(set(V16_DEFAULT_VOCAB)) == 16, (
    "V16_DEFAULT_VOCAB has duplicate words; v16 vocab should be all "
    "unique")


# ---------------------------------------------------------------------
# Helper.
# ---------------------------------------------------------------------

def _resolve_vocab(n_words: int,
                   vocab: Sequence[str] | None) -> Tuple[str, ...]:
    """Pick the working vocabulary.

    If `vocab` is provided, use exactly that (and require
    n_words == len(vocab) for self-consistency). Otherwise default
    to the first `n_words` of V16_DEFAULT_VOCAB (which must therefore
    satisfy n_words <= 16). This lets a caller specify either a
    custom word list OR a prefix of the canonical v16 vocab.
    """
    if vocab is not None:
        if len(vocab) != n_words:
            raise ValueError(
                f"vocab length {len(vocab)} does not match n_words "
                f"{n_words}; pass the matching count or omit n_words")
        if len(set(vocab)) != len(vocab):
            raise ValueError(
                "vocab contains duplicate words; pass an all-unique "
                "list")
        return tuple(vocab)
    # Default path: prefix of v16 vocab.
    if n_words > len(V16_DEFAULT_VOCAB):
        raise ValueError(
            f"n_words={n_words} exceeds the V16_DEFAULT_VOCAB size "
            f"({len(V16_DEFAULT_VOCAB)}); pass an explicit `vocab` "
            f"argument if a larger vocabulary is needed")
    return tuple(V16_DEFAULT_VOCAB[:n_words])


def generate_k_stored_sequences(seed: int,
                                  k: int,
                                  n_words: int = 16,
                                  slot_count: int = 3,
                                  vocab: Sequence[str] | None = None
                                  ) -> List[Tuple[str, ...]]:
    """Generate K stored sequences for the generative-replay loop.

    Each sequence is a length-`slot_count` ordered tuple of distinct
    words drawn from the V-word vocabulary. Inter-sequence diversity
    is enforced: no two returned sequences are equal as ordered
    tuples.

    Determinism: same (seed, k, n_words, slot_count, vocab) yields
    byte-identical output across runs (and across numpy versions
    that share the PCG64 stream).

    Args:
        seed: Integer RNG seed. Maps to numpy.random.default_rng.
        k: Number of stored sequences to generate. Must be >= 1.
        n_words: Vocabulary size. Defaults to 16 (the v16 production
            vocabulary). If `vocab` is supplied, must equal
            `len(vocab)`.
        slot_count: Items per sequence. Must be 1 <= slot_count
            <= n_words.
        vocab: Optional explicit word list. If None, uses the first
            `n_words` of V16_DEFAULT_VOCAB. Must contain unique
            entries when supplied.

    Returns:
        list of K tuples, each of length `slot_count`, where each
        tuple's items are pairwise distinct and no two tuples in the
        list are equal.

    Raises:
        ValueError: bad caller inputs (k<1, slot_count<1,
            slot_count>n_words, k>perm(n_words,slot_count), vocab
            length mismatch, duplicate vocab, etc.).
        RuntimeError: rejection-sampling defensive cap exceeded
            (should be unreachable under valid inputs).
    """
    # ---- Input validation ----
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if slot_count < 1:
        raise ValueError(f"slot_count must be >= 1; got {slot_count}")
    if n_words < 1:
        raise ValueError(f"n_words must be >= 1; got {n_words}")
    if slot_count > n_words:
        raise ValueError(
            f"slot_count={slot_count} exceeds n_words={n_words}; "
            f"cannot draw slot_count items without replacement from "
            f"a smaller vocabulary")

    # The total number of distinct ordered tuples = perm(n_words,
    # slot_count) = n_words! / (n_words - slot_count)!. If the caller
    # asks for more sequences than that, no rejection-sampling pass
    # can ever satisfy the diversity constraint -- raise immediately
    # rather than spin forever.
    n_ordered_tuples = math.perm(n_words, slot_count)
    if k > n_ordered_tuples:
        raise ValueError(
            f"k={k} exceeds the number of distinct ordered tuples "
            f"perm({n_words}, {slot_count}) = {n_ordered_tuples}; "
            f"cannot guarantee inter-sequence diversity")

    words = _resolve_vocab(n_words, vocab)

    # ---- Rejection-sampling main loop ----
    rng = np.random.default_rng(seed)
    seen: set[Tuple[str, ...]] = set()
    sequences: List[Tuple[str, ...]] = []
    # Defensive cap: at most this many rejection attempts. Even when
    # k == n_ordered_tuples (the worst case), the expected number of
    # draws is the n_ordered_tuples * H_{n_ordered_tuples} (coupon-
    # collector). For our typical (k=16, perm=3360) this cap is
    # generous by orders of magnitude.
    max_attempts = max(100 * k, k + 10_000)
    attempts = 0
    while len(sequences) < k:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"generate_k_stored_sequences: rejection-sampling "
                f"defensive cap hit ({max_attempts} attempts) trying "
                f"to produce k={k} distinct sequences of length "
                f"{slot_count} from a {n_words}-word vocab")
        # rng.choice with replace=False guarantees within-sequence
        # uniqueness. The result is a numpy array of object/str
        # dtype; convert to a plain tuple for hashability and stable
        # representation.
        picks = rng.choice(words, size=slot_count, replace=False)
        candidate = tuple(str(x) for x in picks)
        if candidate in seen:
            continue
        seen.add(candidate)
        sequences.append(candidate)
    return sequences


# ---------------------------------------------------------------------
# Module self-check footer: no autograd, no torch, plain ASCII.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke print for direct invocation -- not load-bearing.
    demo = generate_k_stored_sequences(seed=42, k=4,
                                          n_words=16, slot_count=3)
    print("V16_DEFAULT_VOCAB:", V16_DEFAULT_VOCAB)
    for i, seq in enumerate(demo):
        print(f"  seq[{i}]: {seq}")
