"""tools.lab.membrane_bound_check: an integration overshoot must fail loudly (FAILURE_LOG 2026-09-29).

Failing direction first: the -377 mV commit pools of the local model's N=5 basal-ganglia build must raise.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from tools.lab import LeverError, membrane_bound_check


def test_overshoot_raises():
    v = np.full(30, -65.0)
    v[3] = -377.4  # the observed N=5 commit-pool minimum
    with pytest.raises(LeverError):
        membrane_bound_check("commit pools", v)


def test_non_finite_raises():
    v = np.array([-65.0, np.nan, -70.0])
    with pytest.raises(LeverError):
        membrane_bound_check("nan", v)


def test_non_strict_returns_false():
    assert membrane_bound_check("warn", np.array([-200.0]), strict=False) is False


def test_physical_range_passes():
    assert membrane_bound_check("ok", np.array([-90.0, -65.0, 30.0])) is True


def test_gate_a_v2_selector_stays_physical():
    """The two-channel Gate-A v2 selector (the topology the need-reversal seam reuses) never overshoots."""
    from research.runners._vocal_action_selector_gate import (
        DIRECT_PATH_GATE, _run_trial, _set_equal_tonic_current, _step, build_selector_bridge, selector_config,
    )
    cfg = selector_config("v2")
    b = build_selector_bridge(7, cfg)
    b.set_transmission_gate(DIRECT_PATH_GATE, 1.0)
    _set_equal_tonic_current(b, cfg)
    _step(b, cfg.warmup_steps)
    _run_trial(b, cfg)
    assert membrane_bound_check("gate-A v2 after one trial", b.cp_membrane_potential_v) is True
