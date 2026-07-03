"""CPU tests for EMERGE-83 -- the reservoir recursion-depth (agreement-across-center-embedding) characterization.

Light structural tests + a slow single-seed smoke asserting the reservoir tracks the MATRIX subject's number past the
agreement-attraction baseline. CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import pytest

import research.runners._emerge83_reservoir_recursion_depth_derisk as m83


def test_center_embedding_answer_is_the_matrix_subject_number():
    rng = np.random.default_rng(0)
    toks, ans = m83._make(2, rng, ["dog", "cat", "fox"])
    # depth-2 chain: <n1> s1 that <n2> s2 that <n3> s3 ; the answer is the FIRST (matrix) number
    assert toks[0] in m83._NUMS and ans == toks[0]
    assert toks.count(m83._THAT) == 2                              # two embeddings


def test_nearest_number_baseline_is_attraction():
    # the nearest-number baseline predicts the LAST number -> wrong on a depth>=1 chain whose matrix != innermost
    sents = [(["sng", "dog", "that", "plu", "cat"], "sng")]
    assert m83._nearest_number_acc(sents) == 0.0                  # predicts 'plu' (nearest) != 'sng' (matrix)


def test_depths_and_markers():
    assert m83._TEST_DEPTHS[0] == 0 and m83._TEST_DEPTHS[-1] >= 3
    assert set(m83._NUMS) == {"sng", "plu"} and m83._THAT == "that"


@pytest.mark.slow
def test_seed42_reservoir_resists_attraction():
    d = m83._one(42)
    assert d["markers_ok"] is True
    # the reservoir tracks the matrix number at depth 2 while the nearest-number baseline is at chance (attraction)
    assert d["by_depth"][2]["reservoir"] >= 0.90
    assert d["by_depth"][2]["nearest_number"] <= 0.65
    assert (d["by_depth"][2]["reservoir"] - d["matrix_lesion_acc"]) >= 0.20   # genuinely the matrix number
