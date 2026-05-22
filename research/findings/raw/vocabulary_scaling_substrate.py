"""64-concept G.20 sparse bridge builder (Task 1 of
docs/plans/2026-05-22-vocabulary-scaling-implementation.md).

The vocabulary-scaling arc asks whether the biologized grounded-
composition pipeline still clears the frozen 0.80 compositional bar at a
64-concept vocabulary. The right substrate (per the design doc) is the
project's validated large-vocabulary substrate: the catalog G.20
sparse-distributed ensemble -- each concept a scattered K-of-N random
pattern in a shared pool (Kanerva SDM form), validated at 100% per-bridge
discrimination for 64 concepts.

This module is a THIN WRAPPER. It reuses the validated G.20 sparse
builder BYTE-UNCHANGED:

  - `build_sparse_pool_bridge`  -- builds one shared-pool bridge
  - `generate_sparse_patterns`  -- the per-concept K-of-N sparse codes

both imported from `research.runners.concept_pool_sparse_distributed`
(the `g20_multibridge --sparse` path's own builder). Nothing in the G.20
builder or any sim module is modified; this wrapper only composes them
and pins a fixed 64-distinct-word vocabulary.

The 64-word vocabulary is taken from the project's existing G.20 vocab
specification (`research.runners.g20_vocab_spec`): the 32 noun concepts
of Bridge A plus the 32 verb concepts of Bridge B. That spec already
asserts global word-uniqueness across all five bridges, so the 64-word
union is guaranteed distinct.

Public API:
  - `sixty_four_concept_vocabulary()`        -> list[str], 64 distinct
  - `sixty_four_concept_sparse_patterns(seed, ...)` -> list[list[int]]
  - `build_64_concept_sparse_bridge(seed, ...)` -> (bridge, words)

The default build parameters are the validated G.20 64-concept tier
(8192 lang_input neurons, a 2000-neuron shared pool, K=100 sparse
patterns) so the same code path serves the decisive GPU run. Tests may
pass reduced sizes for a fast structural smoke; the patterns stay a pure
function of (n_concepts, n_pool, pattern_size, seed).
"""
from __future__ import annotations

from typing import List, Tuple

# Reuse-by-import only: the validated G.20 sparse builder. NOT modified.
from research.runners.concept_pool_sparse_distributed import (
    build_sparse_pool_bridge,
    generate_sparse_patterns,
)
from research.runners.g20_vocab_spec import (
    VOCAB_BRIDGE_A_NOUNS,
    VOCAB_BRIDGE_B_VERBS,
)

# Pre-registered vocabulary size for the scaling test.
N_CONCEPTS = 64

# Default G.20 64-concept-tier substrate parameters (validated 100%
# per-bridge discrimination at this scale; see CLAUDE.md G.20 section
# and concept_pool_sparse_distributed's own defaults).
DEFAULT_N_LANG_INPUT = 8192
DEFAULT_N_SHARED_POOL = 2000
DEFAULT_N_SHARED_FS = 300
DEFAULT_PATTERN_SIZE = 100


def sixty_four_concept_vocabulary() -> List[str]:
    """Return the fixed list of 64 distinct concept words.

    Sourced from the project's existing G.20 vocab spec: Bridge A's 32
    noun concepts followed by Bridge B's 32 verb concepts. `g20_vocab_spec`
    asserts global uniqueness across all five bridges, so this 64-word
    union has no duplicates.
    """
    vocab = list(VOCAB_BRIDGE_A_NOUNS) + list(VOCAB_BRIDGE_B_VERBS)
    if len(vocab) != N_CONCEPTS:
        raise AssertionError(
            f"expected {N_CONCEPTS} concept words, got {len(vocab)}")
    if len(set(vocab)) != N_CONCEPTS:
        dupes = sorted({w for w in vocab if vocab.count(w) > 1})
        raise AssertionError(
            f"64-concept vocabulary has duplicate words: {dupes}")
    return vocab


def sixty_four_concept_sparse_patterns(
    seed: int,
    n_shared_pool: int = DEFAULT_N_SHARED_POOL,
    pattern_size: int = DEFAULT_PATTERN_SIZE,
) -> List[List[int]]:
    """Return the 64 per-concept sparse K-of-N patterns for `seed`.

    A direct call into the validated G.20 builder's
    `generate_sparse_patterns` (reused unchanged) for exactly
    `N_CONCEPTS` concepts. Pure function of its arguments -- identical
    to what `g20_multibridge --sparse` regenerates at the same seed, so
    the wrapper reads the same neurons the G.20 substrate trained.
    """
    return generate_sparse_patterns(
        n_concepts=N_CONCEPTS,
        n_pool=n_shared_pool,
        pattern_size=pattern_size,
        seed=seed,
    )


def build_64_concept_sparse_bridge(
    seed: int,
    n_lang_input: int = DEFAULT_N_LANG_INPUT,
    n_shared_pool: int = DEFAULT_N_SHARED_POOL,
    n_shared_fs: int = DEFAULT_N_SHARED_FS,
    pattern_size: int = DEFAULT_PATTERN_SIZE,
    verbose: bool = True,
) -> Tuple[object, List[str]]:
    """Build one 64-concept G.20 sparse-distributed bridge.

    Reuses the validated G.20 sparse builder `build_sparse_pool_bridge`
    byte-unchanged. Returns ``(bridge, words)`` where ``words`` is the
    fixed 64-distinct-word vocabulary and ``bridge`` is a freshly
    initialised SimulationBridge with the G.20 sparse architecture
    (``language_input`` / ``shared_concept_pool`` / ``shared_FS`` /
    ``language_output`` brain regions).

    The bridge's per-concept sparse-pool structure -- 64 scattered
    K-of-N patterns over ``shared_concept_pool`` -- is obtained
    deterministically from the same ``seed`` via
    :func:`sixty_four_concept_sparse_patterns` (use that function to
    retrieve it; it is a pure function so the caller regenerates it
    rather than the builder stashing per-bridge state, matching the
    `g20_multibridge` reproducibility invariant).

    Default sizes are the validated G.20 64-concept tier; tests may pass
    reduced sizes for a fast structural smoke. ``pattern_size`` is
    accepted so a reduced-pool smoke can keep K within the pool, and is
    validated against ``n_shared_pool`` here.
    """
    words = sixty_four_concept_vocabulary()

    if pattern_size > n_shared_pool:
        raise ValueError(
            f"pattern_size ({pattern_size}) cannot exceed n_shared_pool "
            f"({n_shared_pool}) -- a K-of-N sparse pattern needs K <= N")

    bridge = build_sparse_pool_bridge(
        seed=seed,
        n_lang_input=n_lang_input,
        n_shared_pool=n_shared_pool,
        n_shared_fs=n_shared_fs,
        n_lang_output=n_lang_input,
        verbose=verbose,
    )

    if verbose:
        print(
            f"[64-concept sparse bridge] seed={seed}, "
            f"{N_CONCEPTS} concepts, shared_pool={n_shared_pool}, "
            f"pattern_size={pattern_size}, lang_input={n_lang_input}",
            flush=True,
        )
    return bridge, words


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Build a 64-concept G.20 sparse-distributed bridge.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=DEFAULT_N_LANG_INPUT)
    p.add_argument("--n-shared-pool", type=int, default=DEFAULT_N_SHARED_POOL)
    p.add_argument("--n-shared-fs", type=int, default=DEFAULT_N_SHARED_FS)
    p.add_argument("--pattern-size", type=int, default=DEFAULT_PATTERN_SIZE)
    args = p.parse_args()

    bridge, words = build_64_concept_sparse_bridge(
        seed=args.seed,
        n_lang_input=args.n_lang_input,
        n_shared_pool=args.n_shared_pool,
        n_shared_fs=args.n_shared_fs,
        pattern_size=args.pattern_size,
        verbose=True,
    )
    patterns = sixty_four_concept_sparse_patterns(
        args.seed,
        n_shared_pool=args.n_shared_pool,
        pattern_size=args.pattern_size,
    )
    print(f"built bridge with {len(words)} distinct concepts")
    print(f"first 8 words: {words[:8]}")
    print(f"sparse patterns: {len(patterns)} x K={len(patterns[0])} "
          f"in pool of {args.n_shared_pool}")
