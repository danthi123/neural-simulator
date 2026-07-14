"""CI guard for past-reservoir Rung 3 (REAL text): the eligibility-trained per-neuron SELECTIVE diagonal SSM beats the
fixed reservoir, an untrained-gate control, a random-token-gate control, AND the bigram at DEEP context depth on
TinyStories next-token CE. Needs the corpus (data/corpus/tinystories.txt); skips gracefully if absent. Offline numpy;
one seed (real-corpus training is a few seconds at this scale)."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pytest

_CORPUS = "data/corpus/tinystories.txt"


@pytest.fixture(scope="module")
def result():
    if not os.path.exists(_CORPUS):                              # pragma: no cover
        pytest.skip("tinystories corpus absent")
    try:
        from research.runners._reslm_rung3_selective_ssm_realtext_derisk import run
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"rung3 realtext deps unavailable: {e}")
    return run(42, _CORPUS, 3000, 200)                           # smaller n_sent for CI speed


def test_selective_beats_reservoir_and_bigram(result):
    """LOWER CE is better: the learned selective gate captures more deep-context than the fixed reservoir + the bigram."""
    dp = result["deep"]
    assert dp["selective"] < dp["fixed_res"] - 0.05
    assert dp["selective"] < dp["bigram"] - 0.05


def test_learning_and_current_token_are_load_bearing(result):
    """selective beats the untrained-gate (detached) AND the random-token-gate (randgate) controls -> LEARNING the gate
    AND conditioning on the CURRENT token are both load-bearing."""
    dp = result["deep"]
    assert dp["selective"] < dp["detached"] - 0.05
    assert dp["selective"] < dp["randgate"] - 0.05
    assert result["GO"]
