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

def test_g1_verdict_pass_requires_gate_and_10pct_over_permuted():
    # true-order score must clear abstention AND beat best permuted
    # control by >= 10% (relative). Bar is FIXED here.
    v = g1_verdict(true_score=0.90, best_perm_score=0.50,
                   gate_cleared=True)
    assert v["GATE"] == "PASS" and v["pct_over_permuted"] >= 10.0
    # gate not cleared -> FAIL regardless of score gap
    assert g1_verdict(0.99, 0.10, gate_cleared=False)["GATE"] == "FAIL"
    # < 10% over permuted -> FAIL (not order-learning, just concepts)
    assert g1_verdict(0.52, 0.50, gate_cleared=True)["GATE"] == "FAIL"
