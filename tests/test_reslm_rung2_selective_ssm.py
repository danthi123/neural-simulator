"""CI guard for past-reservoir Rung 2: a per-neuron SELECTIVE diagonal SSM trained by an exact forward-mode eligibility
trace (no BPTT, no weight transport) beats (a) a fixed-lambda leaky reservoir, (b) the same architecture with the gate
NOT trained, and (c) the gate trained on a permuted input, on a long-range gated-conjunction task. Offline numpy."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pytest


@pytest.fixture(scope="module")
def result():
    try:
        from research.runners._reslm_rung2_selective_ssm_derisk import run
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"rung2 selective-SSM deps unavailable: {e}")
    return run(42)


def test_selective_beats_fixed_reservoir(result):
    """Input-DEPENDENT (selective) gating beats a fixed-lambda leaky reservoir -> the selectivity is load-bearing."""
    assert result["selective"] > result["fixed_res"] + 0.08


def test_learning_the_gate_matters(result):
    """LEARNING the gate (eligibility trace) beats the SAME input-dependent architecture with a fixed random gate."""
    assert result["selective"] > result["detached"] + 0.05


def test_gate_needs_real_input(result):
    """The gate must read the REAL input (permuted-input gate collapses); and selective beats chance decisively."""
    assert result["selective"] > result["permgate"] + 0.05
    assert result["selective"] > 1.0 / 6 + 0.15
    assert result["GO"]
