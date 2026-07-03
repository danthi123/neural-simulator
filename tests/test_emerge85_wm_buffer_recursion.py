"""CPU tests for EMERGE-85 -- the theta-gamma WM buffer + stack-match surpassing the reservoir's recursion boundary.

Structural tests of the mirror-pair (stack) match + a slow single-seed smoke asserting the buffer surpasses the reservoir
in recursion depth then boundaries at its capacity. CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import pytest

import research.runners._emerge85_wm_buffer_recursion_derisk as m85


def test_wm_buffer_stack_match_is_mirror_pairing():
    buf = m85.WMBuffer(capacity=8)
    # a grammatical depth-1 chain: subjects [sng, plu], verbs (reversed) [plu, sng] -> mirror pairs (0,3)=(sng,sng),
    # (1,2)=(plu,plu) both agree
    toks = ["sng", "d", "that", "plu", "c", "plu", "v2", "sng", "v1"]
    f = buf.feature(toks)
    assert f[0] == 1.0 and f[1] == 1.0                            # both mirror pairs agree -> grammatical


def test_wm_buffer_detects_a_mismatch():
    buf = m85.WMBuffer(capacity=8)
    # swap the verbs -> [sng, v2, plu, v1] : mirror pairs (sng vs plu) disagree
    toks = ["sng", "d", "that", "plu", "c", "sng", "v2", "plu", "v1"]
    f = buf.feature(toks)
    assert not (f[0] == 1.0 and f[1] == 1.0)                      # a mirror pair disagrees -> ungrammatical


def test_slot_scramble_destroys_the_mirror_structure():
    buf = m85.WMBuffer(capacity=8)
    toks = ["sng", "d", "that", "plu", "c", "plu", "v2", "sng", "v1"]
    f_ok = buf.feature(toks)
    f_scr = buf.feature(toks, slot_scramble_rng=np.random.default_rng(0))
    assert not np.allclose(f_ok, f_scr)                          # scrambling the slots changes the mirror agreements


def test_capacity_is_bounded():
    assert m85._CAPACITY <= 12                                   # a BOUNDED WM buffer (theta-gamma ~7+-2), not unbounded


@pytest.mark.slow
def test_seed42_buffer_surpasses_reservoir_then_boundaries():
    d = m85._one(42)
    assert d["markers_ok"] is True
    bd = d["by_depth"]
    # the buffer surpasses the reservoir at depth 3 (where the reservoir degrades)
    assert bd[3]["wm_buffer"] >= 0.90 and bd[3]["reservoir"] < 0.90
    # the buffer boundaries at its capacity (depth 4 = 10 numbers > 8 slots)
    assert bd[4]["wm_buffer"] <= 0.65
    # the ordered slots (stack structure) are load-bearing
    assert bd[3]["buffer_slot_scramble"] <= 0.65
