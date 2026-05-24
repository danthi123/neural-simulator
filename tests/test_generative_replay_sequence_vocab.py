"""Unit tests for the sequence vocabulary helper (Task 1 of
`docs/plans/2026-05-24-generative-replay-implementation.md`).

Four load-bearing tests:

  1. Determinism across runs given seed -- same (seed, k, n_words,
     slot_count) returns byte-identical output.
  2. No within-sequence repeats -- every sequence's items are
     pairwise distinct.
  3. Inter-sequence diversity -- no two sequences in the K-list are
     equal as ordered tuples.
  4. Vocab consistency -- the default 16-word vocab is exactly
     DIRECTION + NOUN + VERB + ADJECTIVE from concept_pool_demo.

Each test is written to actually CATCH a buggy implementation:
  - test 1 compares between two independent invocations (a buggy
    helper that secretly mutated global RNG state would fail this).
  - test 2 walks every sequence and asserts set(seq) == len(seq).
  - test 3 asserts len(set(sequences)) == k -- a buggy helper that
    emitted the same tuple twice would fail.
  - test 4 directly imports the concept_pool_demo dicts and verifies
    the helper's default vocab is their concatenation in order.

Plain ASCII only. No protected/frozen/moat module imported or
modified. No autograd.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------
# Test 1: determinism.
# ---------------------------------------------------------------------

def test_determinism_across_runs_given_seed():
    """Two independent calls with the same (seed, k, n_words,
    slot_count) must return byte-identical lists of tuples."""
    from research.findings.raw.generative_replay_sequence_vocab import (
        generate_k_stored_sequences,
    )
    a = generate_k_stored_sequences(seed=42, k=8, n_words=16,
                                      slot_count=3)
    b = generate_k_stored_sequences(seed=42, k=8, n_words=16,
                                      slot_count=3)
    assert a == b, (
        "determinism violated: same (seed, k, n_words, slot_count) "
        "produced different output across two invocations.\n"
        f"  a = {a}\n  b = {b}")
    # Cross-seed difference sanity: changing the seed MUST change
    # at least one sequence (otherwise the helper is ignoring the
    # seed entirely, which would also pass the equality above).
    c = generate_k_stored_sequences(seed=43, k=8, n_words=16,
                                      slot_count=3)
    assert a != c, (
        "helper appears to ignore the seed: seed=42 and seed=43 "
        "produced identical output; expected at least one "
        "differing sequence")


# ---------------------------------------------------------------------
# Test 2: no within-sequence repeats.
# ---------------------------------------------------------------------

def test_no_within_sequence_repeats():
    """Every sequence's items must be pairwise distinct (sampling
    WITHOUT REPLACEMENT)."""
    from research.findings.raw.generative_replay_sequence_vocab import (
        generate_k_stored_sequences,
    )
    # Use a fairly large K and the full 16-word vocab so the test is
    # exercised on enough material to surface any repeats.
    seqs = generate_k_stored_sequences(seed=42, k=16, n_words=16,
                                          slot_count=3)
    assert len(seqs) == 16, (
        f"helper returned {len(seqs)} sequences; expected 16")
    for i, seq in enumerate(seqs):
        assert len(seq) == 3, (
            f"sequence {i} has length {len(seq)}; expected 3")
        assert len(set(seq)) == len(seq), (
            f"sequence {i} has within-sequence repeats: {seq}")


# ---------------------------------------------------------------------
# Test 3: inter-sequence diversity.
# ---------------------------------------------------------------------

def test_inter_sequence_diversity():
    """No two returned sequences may be equal as ordered tuples."""
    from research.findings.raw.generative_replay_sequence_vocab import (
        generate_k_stored_sequences,
    )
    # Pick a K large enough that a buggy implementation (no
    # diversity check) would frequently emit collisions, but small
    # enough that the rejection-sampling search is trivial relative
    # to perm(16, 3) = 3360.
    k = 32
    seqs = generate_k_stored_sequences(seed=42, k=k, n_words=16,
                                          slot_count=3)
    assert len(seqs) == k, (
        f"helper returned {len(seqs)} sequences; expected {k}")
    # Set-of-tuples must equal K (no duplicates).
    unique = set(seqs)
    assert len(unique) == k, (
        f"inter-sequence diversity violated: returned {k} "
        f"sequences but only {len(unique)} are unique. Duplicates "
        f"present.")


# ---------------------------------------------------------------------
# Test 4: default vocab is the v16 16-word vocab from
# concept_pool_demo (reuse-by-import discipline).
# ---------------------------------------------------------------------

def test_vocab_consistency_v16():
    """The default 16-word vocab must be exactly the concatenation
    of DIRECTION + NOUN + VERB + ADJECTIVE from concept_pool_demo,
    in declaration order. This pins reuse-by-import of the validated
    v16 production vocabulary."""
    from research.findings.raw.generative_replay_sequence_vocab import (
        V16_DEFAULT_VOCAB,
        generate_k_stored_sequences,
    )
    from research.runners.concept_pool_demo import (
        ADJECTIVE_VOCAB,
        DIRECTION_VOCAB,
        NOUN_VOCAB,
        VERB_VOCAB,
    )
    expected = tuple(
        list(DIRECTION_VOCAB.keys())
        + list(NOUN_VOCAB.keys())
        + list(VERB_VOCAB.keys())
        + list(ADJECTIVE_VOCAB.keys())
    )
    assert V16_DEFAULT_VOCAB == expected, (
        "V16_DEFAULT_VOCAB drifted from the v16 production "
        "vocabulary.\n"
        f"  got:      {V16_DEFAULT_VOCAB}\n"
        f"  expected: {expected}")
    assert len(V16_DEFAULT_VOCAB) == 16, (
        f"V16_DEFAULT_VOCAB has {len(V16_DEFAULT_VOCAB)} words; "
        f"expected 16 (v16 production vocab)")

    # Verify the helper's default path actually pulls from this
    # vocabulary -- every word in every returned sequence must be a
    # member of V16_DEFAULT_VOCAB when n_words defaults.
    seqs = generate_k_stored_sequences(seed=42, k=8)
    vocab_set = set(V16_DEFAULT_VOCAB)
    for i, seq in enumerate(seqs):
        for w in seq:
            assert w in vocab_set, (
                f"sequence {i} contains word `{w}` not in the "
                f"v16 default vocabulary; helper may be using a "
                f"wrong default")


# ---------------------------------------------------------------------
# Discipline footer: this test module imports only stdlib + pytest +
# the helper-under-test + the v16 vocab dicts from concept_pool_demo
# (reuse-by-import). No protected/frozen/moat module is touched.
# ---------------------------------------------------------------------
