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
