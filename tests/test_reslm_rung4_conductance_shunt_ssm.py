"""CI guard for past-reservoir Rung 4a: the selective SSM's input-modulated leak, realized with a BIOLOGICAL
conductance-based shunt (an input-driven shunting conductance = the input-modulated time constant, trained by the exact
forward-mode eligibility trace), still beats the fixed-leak reservoir + untrained-gate + random-input-gate controls on the
long-range gated-conjunction task. Offline numpy."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pytest


@pytest.fixture(scope="module")
def result():
    try:
        from research.runners._reslm_rung4_conductance_shunt_ssm_derisk import run
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"rung4 conductance-shunt deps unavailable: {e}")
    return run(42)


def test_conductance_shunt_selective_beats_controls(result):
    """The conductance-shunt input-modulated leak realizes the selective advantage: selective beats the fixed-leak
    reservoir, the untrained-gate, and the random-input-gate controls."""
    assert result["selective"] > result["fixed_res"] + 0.08
    assert result["selective"] > result["detached"] + 0.05
    assert result["selective"] > result["randgate"] + 0.05
    assert result["selective"] > 1.0 / 6 + 0.12
    assert result["GO"]
