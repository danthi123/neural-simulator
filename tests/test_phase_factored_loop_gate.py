"""Tests for the phase-factored two-phase spiking loop controller
(Task 2 of docs/plans/2026-05-30-phase-factored-integrated-loop-
implementation.md).

The controller `research/runners/phase_factored_loop_gate.py` runs
compositional memory in TWO PHASES and scores it, reusing four
already-validated subsystems unchanged:
  Phase 1 (ONLINE, theta-ordered): present a length-N concept sequence
    in order; bind it order-preservingly via the engram-tagging API
    (gamma sub-cycle k binds item k).
  Phase 2 (OFFLINE, shuffled): replay via the validated SWR / Phase-1.3
    consolidation to build concept selectivity in cortex, in SHUFFLED
    order.
  Readout 1 (wm): "is concept X in the buffer?" from cortical concept-
    pool activity (selectivity built offline).
  Readout 2 (ep): "what came after X?" from the gamma-slot order of the
    index (built online).
  Shared theta-gamma rhythm: reuse the parked loop's controller;
    lesioning it must collapse BOTH readouts.

These tests run at --tiny-synth scale with SIM_BACKEND=numpy so they
execute on CPU without a GPU. The heavy end-to-end run_rung is a
subprocess smoke (CPU/numpy); the structural / faithfulness / no-autograd
pins are pure source greps + light in-process probes that do NOT build a
bridge (so they stay fast and deterministic).

Plain ASCII. No autograd anywhere in the shipped path.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
_CONTROLLER_PATH = REPO_ROOT / "research" / "runners" / \
    "phase_factored_loop_gate.py"

# The 7 frozen lesion names (mirror integrated_loop_core's frozen
# partition; pinned here so a drift in the controller is caught).
_SHARED = ("no_binding", "no_shared_clock", "no_hippo_store")
_HELPER_WM = ("no_bg_gate",)
_HELPER_EP = ("no_sequencing", "no_cls_replay")
_HELPER_BOTH = ("no_neuromod_timing",)
_ALL_LESIONS = _SHARED + _HELPER_WM + _HELPER_EP + _HELPER_BOTH


def _import_controller():
    """Import the controller module under SIM_BACKEND=numpy with a
    benign argv (no flags). Returns the module or skips if absent."""
    if not _CONTROLLER_PATH.exists():
        pytest.skip("phase_factored_loop_gate.py not landed yet")
    os.environ.setdefault("SIM_BACKEND", "numpy")
    import importlib
    mod = importlib.import_module(
        "research.runners.phase_factored_loop_gate")
    return mod


# ---------------------------------------------------------------------------
# Increment 1: rung shape pin.
# ---------------------------------------------------------------------------
def test_run_rung_returns_exact_rung_shape():
    """run_rung(N, seed, tiny_synth=True) must return EXACTLY the rung
    dict shape the frozen integrated_loop_core.integrated_loop_verdict
    consumes: {"N", "n_seeds", "v1":{wm,ep}, "full":{wm,ep},
    "lesions":{<7 names>:{wm,ep}}}."""
    mod = _import_controller()
    rung = mod.run_rung(2, 42, tiny_synth=True)
    assert isinstance(rung, dict)
    # Top-level keys.
    assert rung["N"] == 2
    assert rung["n_seeds"] == 1
    for key in ("v1", "full"):
        assert key in rung, "rung missing %r" % key
        pair = rung[key]
        assert set(pair.keys()) >= {"wm", "ep"}, (
            "%s pair missing wm/ep: %r" % (key, pair))
        assert isinstance(pair["wm"], float)
        assert isinstance(pair["ep"], float)
    # All 7 lesion keys present with wm/ep pairs.
    assert "lesions" in rung
    les = rung["lesions"]
    assert set(les.keys()) == set(_ALL_LESIONS), (
        "lesion keys %r != frozen 7 %r"
        % (sorted(les.keys()), sorted(_ALL_LESIONS)))
    for name in _ALL_LESIONS:
        lp = les[name]
        assert set(lp.keys()) >= {"wm", "ep"}, (
            "lesion %s pair missing wm/ep: %r" % (name, lp))
        assert isinstance(lp["wm"], float)
        assert isinstance(lp["ep"], float)


def test_run_rung_values_are_finite_in_unit_interval():
    """Every wm/ep readout in the rung is a finite float in [0,1]
    (accuracies). Placeholder-or-real, the contract holds."""
    import math
    mod = _import_controller()
    rung = mod.run_rung(2, 42, tiny_synth=True)
    pairs = [rung["v1"], rung["full"]] + list(rung["lesions"].values())
    for p in pairs:
        for fld in ("wm", "ep"):
            v = p[fld]
            assert math.isfinite(v), "%r not finite" % v
            assert 0.0 <= v <= 1.0, "%r out of [0,1]" % v
