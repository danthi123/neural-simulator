"""Tests for the vocabulary-scaling runner (Task 2 of
docs/plans/2026-05-22-vocabulary-scaling-implementation.md).

The runner (`research/findings/raw/vocabulary_scaling_run.py`) (a) builds
a 64-concept G.20 sparse-distributed bridge (Task 1's wrapper), (b)
captures per-neuron concept-population activity of the G.20
`shared_concept_pool` for each concept by driving its sparse pattern --
mirroring the activity-capture pattern in
`research/findings/raw/activity_level_integration.py` -- and (c) runs the
biologized grounded-composition pipeline (reused by import from
`research/findings/raw/biologized_grounded_composition.py`) generalised
from its v14/v16 16-pool concept taxonomy to a 64-concept layout, with
cue/filler roles assigned by a fixed partition of the 64 concepts.

These tests pin:
  - the pre-registered constants (N_CONCEPTS == 64, BAR == 0.80,
    LOADS == [2, 3, 5]) -- the same constants Task 0's pin test asserts;
  - `capture_concept_activity` returns, per concept word, an
    (M, n_neurons) per-neuron activity array;
  - `run_one_seed(seed, smoke=True)` returns a sane result dict carrying
    per-load integrated + composition-only accuracies;
  - the fixed cue/filler partition splits the 64 concepts in two.

All tests run at SMOKE scale on the NumPy backend so they stay fast /
CI-portable (see CLAUDE.md "Pluggable backend"); the runner itself
supports the full-scale 64-concept GPU run.
"""
from __future__ import annotations

import os

# Force the CPU NumPy backend before importing sim (CI-safe / portable).
os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np
import pytest

from research.findings.raw import vocabulary_scaling_run as vsr


# ---------------------------------------------------------------------
# Pre-registered constants (mirror Task 0's pin -- frozen in advance).
# ---------------------------------------------------------------------
class TestPinnedConstants:
    def test_n_concepts_is_64(self):
        assert vsr.N_CONCEPTS == 64

    def test_bar_is_frozen_080(self):
        assert vsr.BAR == 0.80

    def test_loads_are_2_3_5(self):
        assert list(vsr.LOADS) == [2, 3, 5]


# ---------------------------------------------------------------------
# Fixed cue/filler partition of the 64 concepts.
# ---------------------------------------------------------------------
class TestCueFillerPartition:
    def test_partition_splits_words_in_two_disjoint_groups(self):
        words = [f"w{i}" for i in range(vsr.N_CONCEPTS)]
        cues, fillers = vsr.partition_cue_filler(words)
        assert len(cues) > 0 and len(fillers) > 0
        # Disjoint + exhaustive.
        assert set(cues).isdisjoint(set(fillers))
        assert set(cues) | set(fillers) == set(words)
        assert len(cues) + len(fillers) == vsr.N_CONCEPTS

    def test_partition_is_a_fixed_first_half_last_half_split(self):
        # The plan/design require a FIXED partition (not pool-name
        # prefixes). First-32 cues / last-32 fillers is the simplest
        # fixed split; pin the determinism + halving.
        words = [f"w{i}" for i in range(vsr.N_CONCEPTS)]
        cues, fillers = vsr.partition_cue_filler(words)
        assert cues == words[:32]
        assert fillers == words[32:]

    def test_partition_works_on_a_smoke_subset(self):
        words = [f"w{i}" for i in range(8)]
        cues, fillers = vsr.partition_cue_filler(words)
        assert len(cues) + len(fillers) == 8
        assert set(cues).isdisjoint(set(fillers))
        assert len(cues) > 0 and len(fillers) > 0


# ---------------------------------------------------------------------
# Per-neuron activity capture.
# ---------------------------------------------------------------------
class TestCaptureConceptActivity:
    """`capture_concept_activity` drives each concept's sparse pattern and
    records the per-neuron firing-rate vector over the shared pool."""

    def test_returns_one_observation_matrix_per_word(self):
        bridge, words = vsr.build_smoke_bridge(seed=42)
        m_obs = 3
        acts = vsr.capture_concept_activity(
            bridge, words, vsr.smoke_sparse_patterns(seed=42),
            m_obs=m_obs)
        # One (M, n_neurons) array per captured concept word.
        assert set(acts.keys()) == set(words)
        for w in words:
            arr = acts[w]
            assert arr.ndim == 2
            assert arr.shape[0] == m_obs

    def test_activity_vector_spans_the_shared_pool(self):
        bridge, words = vsr.build_smoke_bridge(seed=42)
        acts = vsr.capture_concept_activity(
            bridge, words, vsr.smoke_sparse_patterns(seed=42), m_obs=2)
        rm = bridge.region_manager
        n_pool = len(list(rm.indices("shared_concept_pool")))
        for w in words:
            # Per-neuron capture over the whole shared concept pool.
            assert acts[w].shape[1] == n_pool

    def test_activity_is_nonnegative_rate(self):
        bridge, words = vsr.build_smoke_bridge(seed=42)
        acts = vsr.capture_concept_activity(
            bridge, words, vsr.smoke_sparse_patterns(seed=42), m_obs=2)
        for w in words:
            arr = acts[w]
            assert np.all(np.isfinite(arr))
            assert np.all(arr >= 0.0)  # firing-rate fractions


# ---------------------------------------------------------------------
# run_one_seed -- smoke scale.
# ---------------------------------------------------------------------
class TestRunOneSeedSmoke:
    """`run_one_seed(seed, smoke=True)` runs the whole capture +
    biologized grounded-composition pipeline on a tiny smoke vocabulary
    and returns a sane result dict."""

    @pytest.fixture(scope="class")
    def result(self):
        return vsr.run_one_seed(42, smoke=True)

    def test_result_is_dict_with_seed(self, result):
        assert isinstance(result, dict)
        assert result["seed"] == 42

    def test_result_has_per_load_entries(self, result):
        assert "per_load" in result
        for load in vsr.LOADS:
            assert load in result["per_load"], (
                f"missing per-load entry for load {load}")

    def test_each_load_reports_integrated_and_composition_only(self, result):
        for load in vsr.LOADS:
            entry = result["per_load"][load]
            assert "integrated_accuracy" in entry
            assert "composition_only_accuracy" in entry
            ia = entry["integrated_accuracy"]
            assert 0.0 <= ia <= 1.0, f"integrated acc {ia} out of [0,1]"
            ca = entry["composition_only_accuracy"]
            # composition-only may be NaN if no clean trials at smoke scale.
            assert (ca != ca) or (0.0 <= ca <= 1.0)

    def test_result_reports_recognition_separately(self, result):
        # The design doc requires recognition reported separately and
        # honestly -- it is NOT folded into the composition accuracy.
        assert "recognition_accuracy" in result
        ra = result["recognition_accuracy"]
        assert (ra != ra) or (0.0 <= ra <= 1.0)

    def test_smoke_flag_recorded(self, result):
        assert result.get("smoke") is True
