"""Tests for the G.20 multibridge --sparse mode.

The sparse-distributed bridges store each concept as a scattered K-of-N
random pattern in a 2000-neuron pool (Kanerva SDM form), NOT contiguous
slices. The end-to-end demo MUST regenerate those patterns byte-identically
to what training used -- a drift in the RNG seed formula or param order
would make the demo stimulate/read the WRONG neurons and silently produce
garbage rankings. These tests pin that reproducibility invariant + the
SharedPoolMember sparse wiring. They are CPU-only (no bridge build).
"""
from __future__ import annotations
import inspect

from research.runners.concept_pool_sparse_distributed import (
    generate_sparse_patterns,
)
from research.runners.g20_multibridge import SharedPoolMember
from research.runners import shared_pool_chat


# Exact params the 5-bridge chain trained with (g20_sparse_5bridge_chain.ps1)
TRAIN_SEED = 42
TRAIN_NCONCEPTS = 32
TRAIN_POOL = 2000
TRAIN_PATTERN = 100


class TestPatternRegenDeterminism:
    """generate_sparse_patterns must be a pure function of its args."""

    def test_two_calls_identical(self):
        a = generate_sparse_patterns(TRAIN_NCONCEPTS, TRAIN_POOL,
                                      TRAIN_PATTERN, TRAIN_SEED)
        b = generate_sparse_patterns(TRAIN_NCONCEPTS, TRAIN_POOL,
                                      TRAIN_PATTERN, TRAIN_SEED)
        assert a == b

    def test_shape_and_bounds(self):
        pats = generate_sparse_patterns(TRAIN_NCONCEPTS, TRAIN_POOL,
                                         TRAIN_PATTERN, TRAIN_SEED)
        assert len(pats) == TRAIN_NCONCEPTS
        for p in pats:
            assert len(p) == TRAIN_PATTERN
            assert len(set(p)) == TRAIN_PATTERN  # all unique
            assert all(0 <= idx < TRAIN_POOL for idx in p)

    def test_distinct_concepts_distinct_patterns(self):
        pats = generate_sparse_patterns(TRAIN_NCONCEPTS, TRAIN_POOL,
                                         TRAIN_PATTERN, TRAIN_SEED)
        # Expected overlap between two random K=100 of N=2000 is ~5;
        # no two concept patterns should be identical.
        for i in range(TRAIN_NCONCEPTS):
            for j in range(i + 1, TRAIN_NCONCEPTS):
                assert pats[i] != pats[j]
                overlap = len(set(pats[i]) & set(pats[j]))
                assert overlap < TRAIN_PATTERN // 2  # nowhere near identical

    def test_seed_changes_patterns(self):
        a = generate_sparse_patterns(TRAIN_NCONCEPTS, TRAIN_POOL,
                                      TRAIN_PATTERN, 42)
        b = generate_sparse_patterns(TRAIN_NCONCEPTS, TRAIN_POOL,
                                      TRAIN_PATTERN, 43)
        assert a != b


class TestSharedPoolMemberSparseWiring:
    """SharedPoolMember(sparse=True) must regenerate the SAME patterns the
    sparse runner used at training time."""

    def _member(self, vocab):
        return SharedPoolMember(
            bridge_path="dummy.h5", vocab=vocab, name="bridgeX",
            n_lang_input=8192, n_shared_pool=TRAIN_POOL,
            sparse=True, pattern_size=TRAIN_PATTERN,
        )

    def test_sparse_flag_recorded(self):
        m = self._member([f"w{i}" for i in range(TRAIN_NCONCEPTS)])
        assert m.sparse is True
        assert m.pattern_size == TRAIN_PATTERN
        assert m.n_shared_pool == TRAIN_POOL

    def test_regen_matches_training(self):
        vocab = [f"w{i}" for i in range(TRAIN_NCONCEPTS)]
        m = self._member(vocab)
        regen = m.regen_sparse_patterns(TRAIN_SEED)
        expected = generate_sparse_patterns(
            len(vocab), TRAIN_POOL, TRAIN_PATTERN, TRAIN_SEED)
        assert regen == expected

    def test_regen_uses_vocab_length_as_n_concepts(self):
        vocab = [f"w{i}" for i in range(TRAIN_NCONCEPTS)]
        m = self._member(vocab)
        regen = m.regen_sparse_patterns(TRAIN_SEED)
        assert len(regen) == len(vocab) == TRAIN_NCONCEPTS

    def test_contiguous_member_has_no_sparse(self):
        m = SharedPoolMember(bridge_path="d.h5",
                              vocab=["a", "b"], name="c")
        assert m.sparse is False


class TestSparseHelpersExist:
    """The sparse recall + engram-capture helpers must be defined
    alongside their contiguous-slice siblings in shared_pool_chat."""

    def test_stim_recall_sparse_rates_signature(self):
        fn = getattr(shared_pool_chat, "stim_recall_sparse_rates", None)
        assert fn is not None, "stim_recall_sparse_rates missing"
        params = list(inspect.signature(fn).parameters)
        assert params[:3] == ["bridge", "tag_name", "sparse_patterns"]

    def test_encode_partial_pair_engram_sparse_signature(self):
        fn = getattr(shared_pool_chat,
                     "encode_partial_pair_engram_sparse", None)
        assert fn is not None, "encode_partial_pair_engram_sparse missing"
        params = list(inspect.signature(fn).parameters)
        assert params[0] == "bridge"
        assert "sparse_patterns" in params
        assert "tag_name" in params
