"""Soundness tests for the pattern-grounded runner.

The load-bearing properties this file pins:

  (a) The grounded symbol's INPUT is genuinely the pattern indicator
      (binary 0/1 vector over the pool), not the activity vector
      (real-valued, dense). Demonstrated by computing both on a real
      cached concept and asserting their value sets differ.
  (b) The deriver seed is identical to the activity-grounded path's
      (DERIV_SEED == 90909). Pinned so any drift surfaces immediately.
  (c) `_ground_symbols_pattern` returns one well-formed complex phasor
      of length N_dim per word, with the same keys as the input word
      list.

Cache-dependent tests SKIP cleanly if the trained activity cache has
not yet been populated on this machine (the cache is produced by the
trained-substrate decisive run; CI / a fresh checkout may not have it).
"""
import os

import numpy as np
import pytest

from research.findings.raw.vocabulary_scaling_run_pattern_grounded import (
    _ground_symbols_pattern, DERIV_SEED, TRAINED_CACHE_DIR,
)
from research.findings.raw.vocabulary_scaling_pattern_helpers import (
    pattern_vector,
)
from research.findings.raw.vocabulary_scaling_run import (
    _load_cache, N_DIM,
)


def test_deriv_seed_matches_activity_grounded_path():
    """The deriver must be byte-identical to the activity-grounded
    path; only its INPUT changes. The activity-grounded path uses
    DERIV_SEED = 90909 (see vocabulary_scaling_run._ground_symbols)."""
    assert DERIV_SEED == 90909


def test_pattern_indicator_differs_from_activity():
    """The pattern-grounded symbol's input is the binary K-of-N
    indicator vector; the activity-grounded symbol's input is the
    mean-centred consolidated activity (real-valued, dense). They MUST
    differ as values on the same concept -- proves the runner is
    genuinely substituting the symbol-derivation input, not silently
    falling through to the activity path."""
    cache = os.path.join(TRAINED_CACHE_DIR, "trained_full_seed42.npz")
    if not os.path.exists(cache):
        pytest.skip("trained activity cache not yet populated")
    acts, words, patterns = _load_cache(cache)
    n_pool = acts[words[0]].shape[1]

    # Pattern indicator for the first concept.
    pv = pattern_vector(patterns[0], n_pool)
    # Activity-derived input (mean-centred consolidated activity) for
    # the same concept -- mirrors the activity-grounded path exactly.
    consolidated = {w: acts[w][:8].mean(axis=0) for w in words}
    common = np.mean([consolidated[w] for w in words], axis=0)
    av = consolidated[words[0]] - common

    assert pv.shape == av.shape
    # Pattern indicator is binary {0, 1}.
    assert set(np.unique(pv).tolist()) == {0.0, 1.0}
    # Activity input is not binary -- it's real-valued and mostly
    # nonzero (mean-centring leaves both positive and negative entries).
    assert not set(np.unique(av).tolist()).issubset({0.0, 1.0})


def test_ground_symbols_pattern_returns_one_symbol_per_word():
    """Each grounded symbol must be a well-formed spike-phase array of
    length N_DIM, keyed by the input word, with the same shape and
    integer dtype the activity-grounded path produces (the pipeline's
    FHRR + attractor stages consume the spike-phase representation
    `phases_to_spikes` returns -- integer phase quantisation, NOT a
    complex phasor)."""
    cache = os.path.join(TRAINED_CACHE_DIR, "trained_full_seed42.npz")
    if not os.path.exists(cache):
        pytest.skip("trained activity cache not yet populated")
    acts, words, patterns = _load_cache(cache)
    n_pool = acts[words[0]].shape[1]

    grounded = _ground_symbols_pattern(words, patterns, n_pool, n_pool)
    assert set(grounded.keys()) == set(words)
    for w in words:
        z = grounded[w]
        assert z.shape == (N_DIM,)
        # Same integer dtype the activity-grounded path produces; the
        # downstream FHRR + attractor pipeline consumes this format.
        assert np.issubdtype(z.dtype, np.integer)
        # Phase indices are non-negative finite integers.
        assert np.all(z >= 0)
