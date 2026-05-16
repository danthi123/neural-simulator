import numpy as np
from research.runners.song_g1_core import (
    score_order, permuted_order_controls, compose_reward,
)
from research.runners.song_g1_core import g1_verdict

def test_score_order_identity_max_scrambled_lower():
    assert score_order([1, 2, 3], [1, 2, 3]) == 1.0
    assert score_order([3, 2, 1], [1, 2, 3]) < 1.0
    # right concepts, wrong order scores strictly below identity
    assert score_order([2, 1, 3], [1, 2, 3]) < 1.0
    # wrong concepts entirely -> low
    assert score_order([9, 9, 9], [1, 2, 3]) < score_order(
        [2, 1, 3], [1, 2, 3])

def test_permuted_order_controls_same_multiset_diff_order():
    ctrls = permuted_order_controls([1, 2, 3],
                                    np.random.default_rng(0), n=5)
    for c in ctrls:
        assert sorted(c) == [1, 2, 3]      # same concepts
        assert c != [1, 2, 3]              # order scrambled
    # deterministic given rng
    assert permuted_order_controls(
        [1, 2, 3], np.random.default_rng(0), n=5) == ctrls

def test_compose_reward_zero_when_gate_failed():
    # any slot below gate -> reward 0 (no-confabulation moat)
    assert compose_reward([1, 2, 3], [1, 2, 3],
                           gate_cleared=False) == 0.0
    assert compose_reward([1, 2, 3], [1, 2, 3],
                           gate_cleared=True) == 1.0
    assert 0.0 <= compose_reward([2, 1, 3], [1, 2, 3],
                                 gate_cleared=True) < 1.0

def test_g1_verdict_pass_requires_gate_floor_contrast_and_margin():
    # clears gate, ts>=0.5 floor, real contrast, >=10% over permuted
    v = g1_verdict(true_score=0.90, best_perm_score=0.50,
                   gate_cleared=True)
    assert v["GATE"] == "PASS" and v["pct_over_permuted"] >= 10.0
    # gate not cleared -> FAIL regardless
    assert g1_verdict(0.99, 0.10, gate_cleared=False)["GATE"] == "FAIL"
    # < 10% over permuted -> FAIL (concepts, not learned order)
    assert g1_verdict(0.62, 0.60, gate_cleared=True)["GATE"] == "FAIL"

def test_g1_verdict_zero_permuted_is_fail_not_pass():
    # C1: no permuted-ORDER contrast => NO order-learning evidence.
    # Must FAIL even with a positive (even perfect) true score.
    assert g1_verdict(1.0, 0.0, gate_cleared=True)["GATE"] == "FAIL"
    assert g1_verdict(0.34, 0.0, gate_cleared=True)["GATE"] == "FAIL"
    assert g1_verdict(0.34, 0.0, gate_cleared=True)[
        "pct_over_permuted"] == 0.0

def test_g1_verdict_absolute_floor_blocks_weak_relative_edge():
    # both terrible but true technically +>=10% over permuted:
    # below the 0.5 majority floor -> FAIL (not a real generative claim)
    assert g1_verdict(0.11, 0.10, gate_cleared=True)["GATE"] == "FAIL"
    # exactly at the +10% bar AND above floor -> PASS (">=" honored)
    assert g1_verdict(0.55, 0.50, gate_cleared=True)["GATE"] == "PASS"

def test_score_order_penalizes_trailing_confabulation():
    # C2: right triple THEN hallucinated concepts must NOT score 1.0
    assert score_order([1, 2, 3], [1, 2, 3]) == 1.0
    assert score_order([1, 2, 3, 9, 9, 9], [1, 2, 3]) < 1.0
    # a clean terminal stop (-1 sentinel) is NOT confabulation
    assert score_order([1, 2, 3, -1], [1, 2, 3]) == 1.0
    assert score_order([1, 2, 3, -1, -1], [1, 2, 3]) == 1.0
    # confabulated reward is likewise < 1.0 (feeds Task-9 training)
    assert compose_reward([1, 2, 3, 9, 9], [1, 2, 3],
                          gate_cleared=True) < 1.0

def test_permuted_order_controls_degenerate_multiset_is_empty():
    # I3: all-same multiset has no non-identity permutation -> []
    # (Task-10 caller must treat "no controls" as FAIL; g1_verdict's
    #  best_perm_score<=0 guard makes that composition safe.)
    assert permuted_order_controls(
        [7, 7, 7], np.random.default_rng(0), n=5) == []
    assert permuted_order_controls(
        [], np.random.default_rng(0), n=5) == []
