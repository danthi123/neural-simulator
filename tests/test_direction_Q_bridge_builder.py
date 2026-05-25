# tests/test_direction_Q_bridge_builder.py
"""Tests for direction_Q_bridge_builder - the standalone Q test bridge."""
import os
import sys
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_build_q_test_bridge_returns_bridge():
    from research.findings.raw.direction_Q_bridge_builder import (
        build_q_test_bridge,
    )
    bridge = build_q_test_bridge(
        seed=42, n_dlpfc=1000, dlpfc_density=0.10,
        enable_nmda=True, verbose=False,
    )
    assert bridge is not None
    # dlpfc_wm region exists
    rm = bridge.region_manager
    idx = rm.indices("dlpfc_wm")
    assert len(idx) == 1000


def test_build_q_test_bridge_has_stim_region():
    from research.findings.raw.direction_Q_bridge_builder import (
        build_q_test_bridge,
    )
    bridge = build_q_test_bridge(
        seed=42, n_dlpfc=1000, dlpfc_density=0.10,
        enable_nmda=True, verbose=False,
    )
    rm = bridge.region_manager
    stim_idx = rm.indices("q_stim_input")
    assert len(stim_idx) >= 100  # at least 100 stim neurons


def test_build_q_test_bridge_nmda_off_control():
    from research.findings.raw.direction_Q_bridge_builder import (
        build_q_test_bridge,
    )
    bridge = build_q_test_bridge(
        seed=42, n_dlpfc=1000, dlpfc_density=0.10,
        enable_nmda=False, verbose=False,
    )
    # Should construct successfully with NMDA off
    assert bridge is not None
