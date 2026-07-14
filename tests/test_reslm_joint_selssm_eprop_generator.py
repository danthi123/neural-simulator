"""CI guard for the JOINT-training de-risk: with the reservoir CO-TRAINED by e-prop (a stronger baseline than the frozen
coupling), adding a co-trained input-dependent SELECTIVE channel still lowers deep-context CE, and the input-dependent
selectivity beats a co-trained FIXED accumulator (the selectivity is load-bearing, not just an extra co-trained channel).
Fully transport-free (random feedback, no BPTT). Needs the TinyStories corpus; skips if absent. Offline numpy; one seed at
a small scale (the joint e-prop is O(n^2) -> keep it small)."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("SIM_BACKEND", "numpy")
import math
import pytest

_CORPUS = "data/corpus/tinystories.txt"


@pytest.fixture(scope="module")
def result():
    if not os.path.exists(_CORPUS):                              # pragma: no cover
        pytest.skip("tinystories corpus absent")
    try:
        from research.runners._reslm_joint_selssm_eprop_generator_derisk import run
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"joint-selssm deps unavailable: {e}")
    return run(42, _CORPUS, 3000, 180)                           # small n_sent/vocab for CI speed (matches the couple CI scale)


def test_all_arms_finite(result):
    dp = result["deep"]
    for a in ("joint_eprop", "joint_eprop_sel", "joint_eprop_fix", "bigram"):
        assert a in dp and math.isfinite(dp[a])


def test_cotrained_selectivity_lowers_deep_ce(result):
    """The co-trained selective channel lowers the co-trained reservoir generator's deep-context CE."""
    assert result["sel_gain"] > 0.0
    assert result["deep"]["joint_eprop_sel"] < result["deep"]["joint_eprop"]


def test_selectivity_beats_cotrained_fixed_accumulator(result):
    """The gain needs the INPUT-DEPENDENT selectivity: it beats a co-trained FIXED (input-independent) accumulator ->
    the trained reservoir does NOT absorb the selective function."""
    assert result["sel_gain"] > result["fix_gain"]
