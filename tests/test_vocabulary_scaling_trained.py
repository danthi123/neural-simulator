"""Soundness tests for the trained-substrate vocabulary-scaling re-run
(`research/findings/raw/vocabulary_scaling_run_trained.py`).

The trained-substrate re-run corrects the diagnosed setup gap behind the
64-concept vocabulary-scaling NEGATIVE: the decisive run captured from a
freshly-built, UNTRAINED G.20 sparse bridge, so the language_input ->
shared_concept_pool pathway was random and non-selective and the
captured activity was near-silent. The corrected runner inserts the
validated G.20 encoding (topographic prior + per-concept training)
before the activity capture.

The LOAD-BEARING property these tests pin: the training stage must
GENUINELY exercise the substrate. If `train_substrate` were a silent
no-op (a misrouted gate, a swallowed exception, a zero-length loop) the
re-run would capture from an effectively-untrained bridge and reproduce
the NEGATIVE for the WRONG reason -- a misleading honest-looking
negative. So one test pins that training substantially modifies the
substrate connectivity and that the train -> capture handoff is clean,
and one pins the validated-encoding constants so they cannot drift.

`train_substrate` calls the validated `apply_sparse_topographic_prior`,
which is CuPy-only (`cp.asnumpy`); the decisive run is GPU. The
exercise test therefore requires the CuPy/GPU backend and skips
cleanly on a CPU-only box.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

from research.findings.raw.vocabulary_scaling_run_trained import (
    train_substrate,
    N_TRAIN_EVENTS,
    TOPOGRAPHIC_FACTOR,
    OFF_TARGET_FACTOR,
    TRAIN_TEACHER_PA,
)
from research.findings.raw.vocabulary_scaling_run import (
    N_CONCEPTS,
    BAR,
    LOADS,
    SPARSITY,
    capture_concept_activity,
)
from research.findings.raw.vocabulary_scaling_substrate import (
    build_64_concept_sparse_bridge,
    sixty_four_concept_sparse_patterns,
)


def test_pinned_validated_encoding_constants():
    """The trained-substrate runner uses the validated G.20 encoding
    defaults verbatim, and imports the frozen compositional bar
    unchanged. Pinned so neither can silently drift."""
    assert N_TRAIN_EVENTS == 400          # validated G.20 default
    assert TOPOGRAPHIC_FACTOR == 10.0     # validated G.20 default
    assert OFF_TARGET_FACTOR == 0.1       # validated G.20 default
    assert TRAIN_TEACHER_PA == 500.0      # validated G.20 default
    # The frozen compositional bar + task shape, imported unchanged.
    assert BAR == 0.80
    assert N_CONCEPTS == 64
    assert LOADS == [2, 3, 5]


def test_training_stage_genuinely_exercises_the_substrate():
    """`train_substrate` must GENUINELY modify the substrate: the
    topographic prior reshapes the language_input -> shared_concept_pool
    weights and the per-concept training STDP-updates them. If
    `train_substrate` were a silent no-op (a misrouted gate, a swallowed
    exception, a zero-length loop) the connectivity would be
    byte-unchanged and the re-run would capture from an effectively-
    untrained bridge -- reproducing the NEGATIVE for the wrong reason.

    This test pins that the substrate connectivity changes substantially
    and that the train -> capture handoff produces well-formed activity.
    """
    from sim.backend import get_backend, to_host, is_gpu_backend
    get_backend()  # initialize the backend
    if not is_gpu_backend():
        pytest.skip("train_substrate's topographic prior is CuPy-only; "
                    "the decisive run is GPU -- this test requires the "
                    "CuPy/GPU backend")
    cp, _ = get_backend()

    n_lang, n_pool, n_fs, k = 512, 256, 30, 16
    n_concepts_test = 6
    seed = 42

    bridge, words = build_64_concept_sparse_bridge(
        seed=seed, n_lang_input=n_lang, n_shared_pool=n_pool,
        n_shared_fs=n_fs, pattern_size=k, verbose=False)
    patterns = sixty_four_concept_sparse_patterns(
        seed, n_shared_pool=n_pool, pattern_size=k)

    # Snapshot the substrate connectivity before training.
    weights_before = np.array(to_host(bridge.cp_connections.data),
                              dtype=np.float64, copy=True)

    train_substrate(
        bridge, patterns, n_lang_input=n_lang,
        n_concepts=n_concepts_test, seed=seed, n_train_events=8,
        sparsity=SPARSITY, n_words_for_orthogonal=N_CONCEPTS,
        verbose=False)

    weights_after = np.array(to_host(bridge.cp_connections.data),
                             dtype=np.float64, copy=True)

    # train_substrate must substantially reshape the connectivity. A
    # no-op would leave it byte-identical.
    assert weights_after.shape == weights_before.shape
    frac_changed = float(np.mean(
        np.abs(weights_after - weights_before) > 1e-6))
    assert frac_changed > 0.01, (
        f"train_substrate barely changed the substrate connectivity "
        f"(frac_changed={frac_changed:.5f}) -- likely a silent no-op")

    # The train -> capture handoff must produce well-formed activity.
    test_words = list(words[:n_concepts_test])
    test_pats = [patterns[i] for i in range(n_concepts_test)]
    acts = capture_concept_activity(
        bridge, test_words, test_pats, m_obs=2, n_lang_input=n_lang,
        n_words_for_orthogonal=N_CONCEPTS, stim_steps=30, verbose=False)
    for w in test_words:
        a = acts[w]
        assert a.shape == (2, n_pool)
        assert np.all(np.isfinite(a))
        assert np.all(a >= 0.0)             # firing rates are non-negative
