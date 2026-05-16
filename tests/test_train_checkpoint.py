import os, numpy as np, pytest
from sim.train_checkpoint import save_checkpoint, load_checkpoint, resume_epoch

def _weights():
    return [np.arange(6, dtype=np.float32).reshape(2, 3),
            np.ones((3, 2), dtype=np.float32)]

def test_roundtrip(tmp_path):
    p = str(tmp_path / "c.npz")
    w = _weights(); rng = np.random.default_rng(7).bit_generator.state
    save_checkpoint(p, epoch=5, weights=w, rng_state=rng,
                    loss_history=[3.2, 2.1])
    ck = load_checkpoint(p)
    assert ck["epoch"] == 5
    assert ck["loss_history"] == [3.2, 2.1]
    for a, b in zip(ck["weights"], w):
        assert np.array_equal(a, b)
    assert ck["rng_state"] == rng

def test_missing_returns_none(tmp_path):
    assert load_checkpoint(str(tmp_path / "nope.npz")) is None

def test_resume_epoch(tmp_path):
    p = str(tmp_path / "c.npz")
    save_checkpoint(p, epoch=11, weights=_weights(),
                    rng_state=np.random.default_rng(1).bit_generator.state,
                    loss_history=[])
    assert resume_epoch(load_checkpoint(p)) == 12
    assert resume_epoch(None) == 0

def test_atomic_no_partial_on_overwrite(tmp_path):
    p = str(tmp_path / "c.npz")
    save_checkpoint(p, epoch=1, weights=_weights(),
                    rng_state=np.random.default_rng(1).bit_generator.state,
                    loss_history=[1.0])
    # second save must fully replace, no .tmp left behind
    save_checkpoint(p, epoch=2, weights=_weights(),
                    rng_state=np.random.default_rng(2).bit_generator.state,
                    loss_history=[1.0, 0.5])
    assert load_checkpoint(p)["epoch"] == 2
    assert not os.path.exists(p + ".tmp")

def test_rng_state_enables_deterministic_continuation(tmp_path):
    p = str(tmp_path / "c.npz")
    g = np.random.default_rng(42)
    _ = g.random(10)
    st = g.bit_generator.state
    save_checkpoint(p, epoch=0, weights=_weights(), rng_state=st,
                    loss_history=[])
    nxt_a = g.random(5)
    g2 = np.random.default_rng()
    g2.bit_generator.state = load_checkpoint(p)["rng_state"]
    nxt_b = g2.random(5)
    assert np.array_equal(nxt_a, nxt_b)
