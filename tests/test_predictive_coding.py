import numpy as np
from sim.predictive_coding import PredictiveCoder

def test_pc_state_resets_and_updates_deterministically():
    pc = PredictiveCoder(n_concepts=8, state_dim=16, seed=42)
    pc.reset(intention=[3, 5])
    assert pc.state.shape == (16,)
    assert np.allclose(pc.state, 0.0)            # reset -> zero prefix
    pc.update_state(3)
    s1 = pc.state.copy()
    assert not np.allclose(s1, 0.0)               # prefix changed state
    pc.update_state(5)
    s2 = pc.state.copy()
    assert not np.allclose(s2, s1)                # order-dependent
    # deterministic given seed + same concept stream
    pc2 = PredictiveCoder(n_concepts=8, state_dim=16, seed=42)
    pc2.reset(intention=[3, 5]); pc2.update_state(3); pc2.update_state(5)
    assert np.allclose(pc2.state, s2)
    # ORDER matters: 3->5 != 5->3
    pc3 = PredictiveCoder(n_concepts=8, state_dim=16, seed=42)
    pc3.reset(intention=[3, 5]); pc3.update_state(5); pc3.update_state(3)
    assert not np.allclose(pc3.state, s2)


def test_predict_next_logits_shape_and_determinism():
    pc = PredictiveCoder(n_concepts=8, state_dim=16, seed=1)
    pc.reset(intention=[2, 4]); pc.update_state(2)
    logits = pc.predict_next()
    assert logits.shape == (8,)
    assert np.all(np.isfinite(logits))
    assert np.allclose(pc.predict_next(), logits)   # pure/deterministic
    # different prefix -> different prediction
    pc.update_state(4)
    assert not np.allclose(pc.predict_next(), logits)


def test_prediction_error_is_softmax_minus_onehot():
    from sim.bptt_snn import softmax_grad_np
    pc = PredictiveCoder(n_concepts=6, state_dim=12, seed=2)
    pc.reset(intention=[1, 3]); pc.update_state(1)
    logits = pc.predict_next()
    err = pc.prediction_error(realized_next_idx=3)
    assert err.shape == (6,)
    assert np.all(np.isfinite(err))
    # Rao-Ballard residual == stabilized softmax CE gradient (DRY:
    # reuses sim.bptt_snn.softmax_grad_np, the log-sum-exp-stable one)
    expected = softmax_grad_np(logits.reshape(1, -1), 3)[0]
    assert np.allclose(err, expected, atol=1e-6)


def test_learn_reduces_ce_on_a_fixed_prefix_target():
    from sim.bptt_snn import cross_entropy_loss_np
    pc = PredictiveCoder(n_concepts=6, state_dim=12, seed=3)
    prefix, target = [1, 4], 2
    def ce():
        pc.reset(intention=prefix + [target])
        for c in prefix: pc.update_state(c)
        return cross_entropy_loss_np(
            pc.predict_next().reshape(1, -1), target)
    before = ce()
    for _ in range(200):
        pc.learn(prefix=prefix, target_next_idx=target, lr=0.05)
    after = ce()
    assert after < before * 0.5      # self-supervised CE drops
    # learning is confined to P weights; shapes unchanged
    assert pc.W_pred.shape == (12, 6) and pc.W_in.shape == (6, 12)


def test_select_next_picks_the_learned_continuation():
    pc = PredictiveCoder(n_concepts=6, state_dim=12, seed=4)
    prefix, target = [0, 5], 3
    for _ in range(300):
        pc.learn(prefix=prefix, target_next_idx=target, lr=0.05)
    pc.reset(intention=prefix + [target])
    for c in prefix: pc.update_state(c)
    # active inference: emit the concept the generative model most
    # predicts given the prefix (argmax predicted prob)
    assert pc.select_next(candidates=list(range(6))) == target
    # restricting candidates still returns the best AVAILABLE one
    alt = pc.select_next(candidates=[1, 3, 4])
    assert alt == 3


def test_rollout_reproduces_a_learned_two_concept_proposition():
    pc = PredictiveCoder(n_concepts=8, state_dim=24, seed=5)
    intended = [2, 6]                       # ordered proposition
    # self-supervised on each prefix->next of the intended order
    for _ in range(400):
        pc.learn(prefix=[], target_next_idx=2, lr=0.05)
        pc.learn(prefix=[2], target_next_idx=6, lr=0.05)
    produced = pc.rollout(intention=intended, length=2,
                          candidates=list(range(8)))
    assert produced == intended            # order-correct generation
