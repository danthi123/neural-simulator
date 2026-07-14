"""CI guard for the additive `enable_selective_ssm_state` sim/ mechanism (past-reservoir Rung 4b-ii): a per-neuron SLOW
graded leaky-integrator state whose leak is set by an input-driven shunt (s = lam_eff*s + (1-lam_eff)*inject, lam_eff =
1 - ssm_k_leak*(1+shunt)) realizes the input-modulated HOLD/RELEASE on the bridge; and it is byte-identical when the flag
is OFF (arrays None, per-step block skipped). Offline numpy."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pytest


def test_ssm_state_hold_release():
    """The slow state HOLDS an injected value under low shunt and RELEASES it (->0) under high shunt."""
    try:
        from research.runners._reslm_rung4b_ii_onbridge_slow_ssm_state_derisk import run
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"rung4b-ii deps unavailable: {e}")
    r = run(42)
    assert r["hold"] - r["release"] > 0.30
    assert r["hold"] > 0.30 and r["release"] < 0.15
    assert r["GO"]


def test_byte_identical_when_flag_off():
    """Flag OFF -> cp_ssm_state/inject/shunt are None, the per-step block is unreached, the step runs unchanged."""
    try:
        from research.runners._emerge14_stageC_onbridge_learning_derisk import build_pool_bridge
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"bridge deps unavailable: {e}")
    b, _ci, _r, _c = build_pool_bridge(8, 8, 42)                 # default cfg: enable_selective_ssm_state = False
    assert getattr(b.core_config, "enable_selective_ssm_state", None) is False
    assert b.cp_ssm_state is None and b.cp_ssm_inject is None and b.cp_ssm_shunt is None
    b._run_one_simulation_step()                                 # steps fine; the SSM block is skipped
    assert b.cp_ssm_state is None


def test_onbridge_state_equivalent_to_numpy():
    """The on-bridge cp_ssm_state update IS the numpy selective-SSM update to float32 precision (Rung 4b-iii-a) ->
    the validated numpy ladder (Rung 2/3/4a) transfers to the bridge exactly."""
    try:
        from research.runners._reslm_rung4b_iii_onbridge_ssm_equivalence_derisk import run
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"rung4b-iii-eq deps unavailable: {e}")
    r = run(42)
    assert r["max_abs_diff"] < 1e-5
    assert r["GO"]
