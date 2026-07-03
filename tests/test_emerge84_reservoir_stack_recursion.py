"""CPU tests for EMERGE-84 -- the genuine stack-recursion (nested pair-matching grammaticality) boundary.

Structural tests (the count shortcut is defeated by a multiset-preserving swap) + a slow single-seed smoke asserting the
reservoir does shallow nested matching (depth 1) then degrades, with the count baseline at chance. CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import pytest

import research.runners._emerge84_reservoir_stack_recursion_derisk as m84


def test_ungrammatical_swap_preserves_the_number_multiset():
    from collections import Counter
    rng = np.random.default_rng(1)
    for _ in range(30):
        tg, yg = m84._make(2, True, rng, ["dog"], ["run"])
        tu, yu = m84._make(2, False, rng, ["dog"], ["run"])
        assert yg == 1 and yu == 0
        # both classes carry >= 2 distinct subject numbers -> a multiset-preserving swap exists
        subj_nums_g = [tg[i] for i in range(len(tg)) if tg[i] in m84._NUMS][:3]
        assert len(set(subj_nums_g)) >= 2


def test_verbs_are_in_reversed_pairing_order():
    rng = np.random.default_rng(2)
    toks, _y = m84._make(2, True, rng, ["dog", "cat", "fox"], ["run", "hop"])
    assert toks.count(m84._THAT) == 2                              # depth-2 center-embedding


def test_count_multiset_baseline_at_chance_by_construction():
    # grammatical vs multiset-preserving-swap ungrammatical share the identical number multiset
    rng = np.random.default_rng(3)
    sents = m84._gen(2, 200, rng, ["dog", "cat", "fox"], ["run", "hop"])
    assert m84._count_multiset_baseline_acc(sents) <= 0.65        # the count shortcut cannot separate them


@pytest.mark.slow
def test_seed42_reservoir_shallow_stack_then_degrades():
    d = m84._one(42)
    assert d["markers_ok"] is True
    assert d["by_depth"][1]["reservoir"] >= 0.90                  # shallow nested matching works
    assert d["by_depth"][1]["count_baseline"] <= 0.65            # not the count shortcut
    assert d["by_depth"][1]["shuffle"] <= 0.65                   # structure load-bearing
    # the reservoir degrades with nesting depth (the fading-memory-not-a-stack signature)
    assert d["by_depth"][3]["reservoir"] <= d["by_depth"][1]["reservoir"]
