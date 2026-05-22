"""Tests for the 64-concept G.20 sparse bridge builder (Task 1 of
docs/plans/2026-05-22-vocabulary-scaling-implementation.md).

The vocabulary-scaling arc needs a 64-concept G.20 sparse-distributed
substrate -- the project's validated large-vocabulary substrate
(scattered K-of-N concept codes, Kanerva SDM form). Task 1 is a thin
wrapper, `build_64_concept_sparse_bridge(seed)`, that reuses the
validated G.20 sparse builder (`build_sparse_pool_bridge` /
`generate_sparse_patterns` from `concept_pool_sparse_distributed`)
BYTE-UNCHANGED to construct one such bridge for a fixed 64-distinct-word
vocabulary.

These tests pin:
  - the wrapper exposes the pre-registered 64-concept vocabulary
    (distinct words);
  - the per-concept sparse-pool structure (64 patterns, each a
    K-of-N subset of the shared pool, distinct, byte-identical to the
    G.20 builder's own `generate_sparse_patterns` for the same seed);
  - a real reduced-scale bridge build produces the expected G.20 sparse
    architecture (4 brain regions incl. `shared_concept_pool`), so the
    same code path supports the full 64-concept GPU build for Task 4.

The structural pattern checks are CPU-only and fast (pure function).
The one real-bridge smoke build runs at a reduced pool size on the
NumPy backend so it stays fast in CI; the wrapper itself supports the
full-scale 64-concept build.
"""
from __future__ import annotations

import os

# Force the CPU NumPy backend so this test is portable / CI-safe
# (see CLAUDE.md "Pluggable backend"). Set before importing sim.
os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.findings.raw.vocabulary_scaling_substrate import (
    build_64_concept_sparse_bridge,
    sixty_four_concept_vocabulary,
    sixty_four_concept_sparse_patterns,
)
from research.runners.concept_pool_sparse_distributed import (
    generate_sparse_patterns,
)


N_CONCEPTS = 64


class TestVocabulary:
    """The 64-concept vocabulary is a fixed list of distinct words."""

    def test_size_is_64(self):
        vocab = sixty_four_concept_vocabulary()
        assert len(vocab) == N_CONCEPTS

    def test_all_words_distinct(self):
        vocab = sixty_four_concept_vocabulary()
        assert len(set(vocab)) == N_CONCEPTS, (
            "the 64-concept vocabulary must have no duplicate words")

    def test_words_are_nonempty_strings(self):
        vocab = sixty_four_concept_vocabulary()
        for w in vocab:
            assert isinstance(w, str) and w.strip(), (
                f"vocabulary entry {w!r} is not a non-empty string")

    def test_vocabulary_is_stable(self):
        # Fixed list -- two calls return the identical ordering.
        assert sixty_four_concept_vocabulary() == \
            sixty_four_concept_vocabulary()


class TestSparsePatternStructure:
    """Per-concept sparse patterns: 64 distinct K-of-N subsets, and
    byte-identical to the reused G.20 builder for the same seed."""

    SEED = 42
    POOL = 2000
    PATTERN = 100

    def test_pattern_count_matches_vocab(self):
        pats = sixty_four_concept_sparse_patterns(
            self.SEED, n_shared_pool=self.POOL, pattern_size=self.PATTERN)
        assert len(pats) == N_CONCEPTS

    def test_each_pattern_is_k_of_n(self):
        pats = sixty_four_concept_sparse_patterns(
            self.SEED, n_shared_pool=self.POOL, pattern_size=self.PATTERN)
        for p in pats:
            assert len(p) == self.PATTERN
            assert len(set(p)) == self.PATTERN  # all unique within pattern
            assert all(0 <= idx < self.POOL for idx in p)

    def test_patterns_distinct_across_concepts(self):
        pats = sixty_four_concept_sparse_patterns(
            self.SEED, n_shared_pool=self.POOL, pattern_size=self.PATTERN)
        for i in range(N_CONCEPTS):
            for j in range(i + 1, N_CONCEPTS):
                assert pats[i] != pats[j], (
                    f"concepts {i},{j} got identical sparse patterns")

    def test_patterns_byte_identical_to_g20_builder(self):
        # The wrapper MUST reuse the validated G.20 builder unchanged --
        # a drift here would read the wrong neurons (see
        # test_g20_sparse_multibridge.py reproducibility invariant).
        ours = sixty_four_concept_sparse_patterns(
            self.SEED, n_shared_pool=self.POOL, pattern_size=self.PATTERN)
        canonical = generate_sparse_patterns(
            N_CONCEPTS, self.POOL, self.PATTERN, self.SEED)
        assert ours == canonical

    def test_seed_changes_patterns(self):
        a = sixty_four_concept_sparse_patterns(
            42, n_shared_pool=self.POOL, pattern_size=self.PATTERN)
        b = sixty_four_concept_sparse_patterns(
            43, n_shared_pool=self.POOL, pattern_size=self.PATTERN)
        assert a != b


class TestBuild64ConceptSparseBridge:
    """`build_64_concept_sparse_bridge` returns (bridge, words) with the
    expected G.20 sparse architecture. A reduced-scale build keeps the
    test fast; the wrapper supports the full 64-concept build for the
    decisive GPU run."""

    def test_returns_bridge_and_64_distinct_words(self):
        bridge, words = build_64_concept_sparse_bridge(
            seed=42, n_lang_input=256, n_shared_pool=256,
            n_shared_fs=40, verbose=False)
        assert bridge is not None
        assert len(words) == N_CONCEPTS
        assert len(set(words)) == N_CONCEPTS

    def test_words_match_vocabulary(self):
        _, words = build_64_concept_sparse_bridge(
            seed=42, n_lang_input=256, n_shared_pool=256,
            n_shared_fs=40, verbose=False)
        assert list(words) == sixty_four_concept_vocabulary()

    def test_bridge_has_g20_sparse_regions(self):
        bridge, _ = build_64_concept_sparse_bridge(
            seed=42, n_lang_input=256, n_shared_pool=256,
            n_shared_fs=40, verbose=False)
        rm = bridge.region_manager
        assert rm is not None, "bridge must use the brain-region framework"
        for region in ("language_input", "shared_concept_pool",
                        "shared_FS", "language_output"):
            idx = list(rm.indices(region))
            assert len(idx) > 0, f"region {region} missing from bridge"

    def test_shared_pool_size_honoured(self):
        n_pool = 256
        bridge, _ = build_64_concept_sparse_bridge(
            seed=42, n_lang_input=256, n_shared_pool=n_pool,
            n_shared_fs=40, verbose=False)
        rm = bridge.region_manager
        assert len(list(rm.indices("shared_concept_pool"))) == n_pool

    def test_per_concept_sparse_structure_available(self):
        # The build also yields the per-concept sparse-pool structure --
        # 64 patterns over the shared pool, fitting within it.
        n_pool = 256
        pattern_size = 30
        bridge, words = build_64_concept_sparse_bridge(
            seed=42, n_lang_input=256, n_shared_pool=n_pool,
            n_shared_fs=40, pattern_size=pattern_size, verbose=False)
        pats = sixty_four_concept_sparse_patterns(
            42, n_shared_pool=n_pool, pattern_size=pattern_size)
        assert len(pats) == len(words) == N_CONCEPTS
        for p in pats:
            assert len(p) == pattern_size
            assert all(0 <= idx < n_pool for idx in p)
