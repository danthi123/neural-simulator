"""CPU tests for EMERGE-86 -- the fully-spiking RANK-3 WM buffer (OrderedPositionWM + spiking mirror-pair coincidence).

Structural tests (the buffer wraps the validated spiking RF ordered-WM; the mirror-pair match is a phase-coincidence) +
a slow single-seed smoke asserting the spiking WM surpasses depth 1 and its recall is load-bearing. CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import pytest

import research.runners._emerge86_spiking_wm_buffer_recursion_derisk as m86


def test_buffer_wraps_the_spiking_rf_ordered_wm():
    from research.runners.ordered_position_wm import OrderedPositionWM
    buf = m86.SpikingWMBuffer(seed=42)
    assert isinstance(buf.wm, OrderedPositionWM)                  # the validated spiking RF Lisman-Idiart ordered-WM
    assert buf.n_slots == m86._N_SLOTS and set(buf.wm.words) == set(m86._NUMS)


def test_mirror_pair_coincidence_is_a_phase_cosine():
    buf = m86.SpikingWMBuffer(seed=42)
    a = np.zeros(m86._D); b = np.zeros(m86._D)
    assert buf._coincidence(a, b) == pytest.approx(1.0)          # identical phasors -> full coincidence
    c = np.full(m86._D, 0.5)                                     # antiphase
    assert buf._coincidence(a, c) < 0.0


def test_feature_reads_only_number_markers_and_is_capacity_bounded():
    buf = m86.SpikingWMBuffer(seed=42)
    # a depth-3 sentence has 8 numbers = exactly n_slots; the feature encodes them (no overflow error)
    rng = np.random.default_rng(0)
    toks, _y = m86.m84._make(3, True, rng, ["dog"], ["run"])
    f = buf.feature(toks)
    assert f.shape == (buf.dim + 2,)
    # a depth-4 sentence has 10 numbers > n_slots -> the buffer truncates (bounded stack), no error
    toks4, _y4 = m86.m84._make(4, True, rng, ["dog"], ["run"])
    f4 = buf.feature(toks4)
    assert f4.shape == (buf.dim + 2,)


@pytest.mark.slow
def test_seed42_spiking_wm_surpasses_depth1_and_recall_is_load_bearing():
    d = m86._one(42)
    bd = d["by_depth"]
    assert bd[1]["spiking_wm"] >= 0.90                            # shallow nested matching works on spikes
    assert bd[1]["count_baseline"] <= 0.65                       # not the count shortcut
    # the unbind (spiking slot recall) is load-bearing: lesioning it collapses the match
    assert bd[1]["unbind_lesion"] <= bd[1]["spiking_wm"] - 0.15
    # the ordered slots (stack structure) are load-bearing
    assert bd[1]["slot_scramble"] <= bd[1]["spiking_wm"] - 0.10
