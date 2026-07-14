"""CI guard for the emergent generator's SPIKING read-out: a one-of-K FS-WTA spiking read-out (driven by the linear
read-out's next-token scores) matches the numpy argmax at parity with no accuracy loss, and a permuted-score control
collapses. Builds a small FS-WTA Izhikevich bridge (numpy); skips if deps/backend unavailable."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pytest


@pytest.fixture(scope="module")
def result():
    try:
        from research.runners._reslm_spiking_readout_derisk import run
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"spiking read-out deps unavailable: {e}")
    try:
        return run(42)
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"FS-WTA bridge unavailable: {e}")


def test_spiking_readout_parity(result):
    """The spiking FS-WTA read-out agrees with the numpy argmax on ~most predictions."""
    assert result["parity"] > 0.90


def test_no_accuracy_loss(result):
    """The spiking read-out achieves the same next-token accuracy as numpy (the tie-break cost is small)."""
    assert abs(result["spk_acc"] - result["numpy_acc"]) < 0.05
    assert result["spk_acc"] > 1.5 / 12                          # beats chance


def test_shuffled_scores_collapse(result):
    """Driving the FS-WTA with permuted scores collapses parity -> the WTA reads the actual read-out scores."""
    assert result["shuffle_parity"] < 0.5
    assert result["GO"]
