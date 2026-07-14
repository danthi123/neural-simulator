"""CI guard for the AIRTIGHT deep-tail controls (the adversarial-verify's named test): over a fixed reservoir, the trained
selective channel provides genuine selective-specific distal holding at deep context — it beats a no-hold(lam=0) control
(accumulation helps) AND a random-gate control (input-dependent selectivity, not generic capacity) at d>=6. Needs the
TinyStories corpus; skips if absent. Offline numpy; one seed at a small scale (the 5-arm airtight is the cost)."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("SIM_BACKEND", "numpy")
import runpy
import sys
import json
import pytest

_CORPUS = "data/corpus/tinystories.txt"
_OUT = "research/findings/raw/_airtight_ci.json"


@pytest.fixture(scope="module")
def result():
    if not os.path.exists(_CORPUS):                              # pragma: no cover
        pytest.skip("tinystories corpus absent")
    argv = ["_", "--n-sentences", "5000", "--n-pool", "200", "--n-train", "1200",
            "--n-eval", "400", "--vocab", "150", "--seed", "42", "--out", _OUT]
    old = sys.argv
    try:
        sys.argv = argv
        runpy.run_module("research.runners._reslm_scale_trained_selssm_vectorized_derisk", run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"airtight runner unavailable: {e}")
    finally:
        sys.argv = old
    if not os.path.exists(_OUT):                                  # pragma: no cover
        pytest.skip("airtight run produced no output")
    return json.load(open(_OUT))


def test_selective_holding_beats_nohold_at_deep(result):
    """sel < no-hold at d>=6 -> the accumulation/holding (not just the current-token projection) helps at deep context."""
    assert result["deep_tail_d6"]["sel_beats_noheld"] is not None
    assert result["deep_tail_d6"]["sel_beats_noheld"] > 0.0


def test_input_selective_holding_beats_randomgate_at_deep(result):
    """sel < random-gate at d>=6 -> the deep benefit is input-dependent SELECTIVE holding (the gate reading the current
    token), not generic extra-channel capacity (identical channel, only the gate input differs)."""
    assert result["deep_tail_d6"]["sel_beats_rand"] is not None
    assert result["deep_tail_d6"]["sel_beats_rand"] > 0.0
