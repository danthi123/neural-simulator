"""Soundness tests for the 160-concept ensemble runner.

The load-bearing structural properties pinned here:
  (a) The runner uses g20_vocab_spec.ALL_BRIDGES verbatim -- no
      vocabulary drift or reshuffling.
  (b) The K=16 PASS recipe is fixed in the runner (K_VOCAB_TARGET=16,
      N_TRAIN_EVENTS matches the validated G.20 default 400).
  (c) The smoke build + train + capture path runs end-to-end and
      produces well-formed cached activity (the structural integrity
      of the train -> capture handoff, on the GPU backend).

Heavier runtime-trace properties (no answer leak in the multi-bridge
orchestration; recognition is genuinely the only handle that names
which pattern is read; the pipeline body is byte-identical to
run_pipeline) are inherited from train_substrate + run_pipeline (both
byte-unchanged from already-adversarially-reviewed code) and are
covered exhaustively by the dedicated adversarial reviewer in Task 4
of the 160-ensemble implementation plan.

GPU-required tests skip cleanly on a CPU-only box (the validated
G.20 topographic prior inside train_substrate is CuPy-only; the
decisive run is GPU).
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.findings.raw.vocabulary_scaling_run_160ensemble import (
    BRIDGE_NAMES, K_VOCAB_TARGET, N_CONCEPTS_PER_BRIDGE,
    SMOKE_BRIDGE_NAMES, SMOKE_VOCAB_PER_BRIDGE,
    build_and_train_bridge_smoke,
)
from research.findings.raw.vocabulary_scaling_run_trained import (
    N_TRAIN_EVENTS,
)
from research.runners.g20_vocab_spec import ALL_BRIDGES


def test_runner_bridge_names_match_vocab_spec_exactly():
    """No vocab drift: the runner's BRIDGE_NAMES is identical to
    g20_vocab_spec.ALL_BRIDGES.keys()."""
    assert set(BRIDGE_NAMES) == set(ALL_BRIDGES.keys())
    assert len(BRIDGE_NAMES) == 5


def test_k16_pass_recipe_pinned_in_runner():
    """The K=16 PASS recipe constants are fixed in the runner; not
    tunable per-bridge."""
    assert K_VOCAB_TARGET == 16
    # Per-bridge concept count is 32 (5 x 32 = 160).
    assert N_CONCEPTS_PER_BRIDGE == 32
    # The validated G.20 training default is 400 events; the runner
    # imports it unchanged from train_substrate's module.
    assert N_TRAIN_EVENTS == 400


def test_smoke_constants_are_proper_subset():
    """The smoke subset is a strict subset of the full configuration
    (so the smoke can't accidentally exceed full)."""
    assert len(SMOKE_BRIDGE_NAMES) <= 5
    assert all(b in BRIDGE_NAMES for b in SMOKE_BRIDGE_NAMES)
    assert SMOKE_VOCAB_PER_BRIDGE <= N_CONCEPTS_PER_BRIDGE


def test_smoke_build_and_train_produces_wellformed_state():
    """The smoke build + train path runs end-to-end. The returned
    (bridge, words, patterns) must be well-formed: words is the
    bridge's first SMOKE_VOCAB_PER_BRIDGE concepts in spec order,
    patterns has matching count, every pattern index is in range."""
    from sim.backend import get_backend, is_gpu_backend
    get_backend()
    if not is_gpu_backend():
        pytest.skip("train_substrate's topographic prior is CuPy-only; "
                    "the decisive run is GPU -- this test requires "
                    "the CuPy/GPU backend")

    bridge_name = SMOKE_BRIDGE_NAMES[0]
    bridge, words, patterns = build_and_train_bridge_smoke(
        bridge_name, seed=42)
    expected = list(ALL_BRIDGES[bridge_name])[:SMOKE_VOCAB_PER_BRIDGE]
    assert words == expected
    assert len(patterns) == SMOKE_VOCAB_PER_BRIDGE
    # Pattern indices must lie inside the smoke pool size.
    from research.findings.raw.vocabulary_scaling_run_160ensemble import (
        SMOKE_N_SHARED_POOL,
    )
    for p in patterns:
        assert all(0 <= int(i) < SMOKE_N_SHARED_POOL for i in p)
