"""CI guard for the past-reservoir Rung-1 result: explicit Sigma-Pi PRODUCT (conjunction) features on a fixed reservoir's
local read-out solve a 2nd-order (conjunction-requiring) task the linear read-out is bounded on, and it is the CONJUNCTION
structure (product > a param-matched random-nonlinear control) not mere capacity; a permuted-product control collapses.
Offline numpy; fast."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pytest


@pytest.fixture(scope="module")
def result():
    try:
        from research.runners._reslm_conjunction_readout_derisk import run
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"conjunction-readout deps unavailable: {e}")
    return run(42)


def test_product_beats_linear(result):
    """Conjunction features help: product beats the linear read-out over the same reservoir."""
    assert result["product"] > result["linear"] + 0.05


def test_it_is_the_conjunction_not_capacity(result):
    """The load-bearing control: product beats a PARAM-MATCHED random-nonlinear feature set -> it's the conjunction
    STRUCTURE, not the added feature count / nonlinear capacity."""
    assert result["product"] > result["randnl"] + 0.05


def test_permuted_products_collapse_and_beat_bigram(result):
    """Shuffling the products destroys the signal; product decisively beats the memoryless bigram floor."""
    assert result["permprod"] < result["product"] - 0.05
    assert result["product"] > result["bigram"] + 0.10
    assert result["GO"]
