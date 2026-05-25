"""Tests for direction_Q_protocol - Wang 2002 stim + delay protocol functions."""
import os
import sys
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_run_baseline_returns_rate():
    from research.findings.raw.direction_Q_bridge_builder import (
        build_q_test_bridge,
    )
    from research.findings.raw.direction_Q_protocol import (
        run_baseline_period,
    )
    bridge = build_q_test_bridge(
        seed=42, n_dlpfc=200, dlpfc_density=0.10,
        enable_nmda=True, verbose=False,
    )
    rate = run_baseline_period(bridge, duration_ms=200.0)
    assert isinstance(rate, float)
    assert rate >= 0.0


def test_apply_cue_stimulates_dlpfc():
    from research.findings.raw.direction_Q_bridge_builder import (
        build_q_test_bridge,
    )
    from research.findings.raw.direction_Q_protocol import (
        apply_cue_stimulus,
    )
    bridge = build_q_test_bridge(
        seed=42, n_dlpfc=200, dlpfc_density=0.10,
        enable_nmda=True, verbose=False,
    )
    rate = apply_cue_stimulus(
        bridge, cue_amplitude_pA=1500.0, duration_ms=200.0,
        cue_fraction=0.5,
    )
    assert rate > 0.0  # stim should drive at least some firing


def test_measure_delay_period_returns_rate_trajectory():
    from research.findings.raw.direction_Q_bridge_builder import (
        build_q_test_bridge,
    )
    from research.findings.raw.direction_Q_protocol import (
        measure_delay_period,
    )
    bridge = build_q_test_bridge(
        seed=42, n_dlpfc=200, dlpfc_density=0.10,
        enable_nmda=True, verbose=False,
    )
    rates = measure_delay_period(
        bridge, duration_ms=500.0, bin_ms=50.0,
    )
    assert len(rates) == 10  # 500 / 50 = 10 bins
    assert all(r >= 0.0 for r in rates)
