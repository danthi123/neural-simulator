import numpy as np
from research.runners.song_g1_core import (
    score_order, permuted_order_controls, compose_reward,
)

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
