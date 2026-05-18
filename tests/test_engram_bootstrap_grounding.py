"""Task-0 grounding pin for the Option B engram-bootstrap gate.

Intentionally RED until Task 2 ships research/runners/engram_bootstrap_gate.py.
This pins the reused-interface contract so a drift is caught loudly.
"""
import importlib


def test_compose_bridge_core_frozen_bars_unchanged():
    core = importlib.import_module("research.runners.compose_bridge_core")
    # REUSED byte-UNMODIFIED — these frozen bars are INHERITED, never moved.
    assert core._CBR_V1_ACC_MIN == 0.80
    assert core._CBR_SCI_ACC_MIN == 0.80
    assert core._CBR_CTRL_ACC_MAX == 0.35
    assert core._CBR_MIN_SEEDS == 3
    assert core._CONTROLS == ("hebbian_no_trace", "permuted", "wrongsign")


def test_engram_bridge_api_present():
    from sim.bridge import SimulationBridge
    for m in ("start_engram_recording", "commit_engram_tag",
              "stimulate_tag", "clear_tag_drive"):
        assert callable(getattr(SimulationBridge, m, None)), m


def test_engram_bootstrap_gate_exists_and_reuses_core():
    g = importlib.import_module("research.runners.engram_bootstrap_gate")
    # Net-new gate REUSES cbr_verdict byte-UNMODIFIED (no new movable bar).
    from research.runners.compose_bridge_core import cbr_verdict
    assert g.cbr_verdict is cbr_verdict
    # Frozen scale ladder + tol pre-registered in the gate (never tuned).
    assert g._SCALE_LADDER == (4, 8, 16)
    assert g._SCALE_TOL == 0.05
    # Scale-confidence aggregator is a pure callable.
    assert callable(g.scale_confidence)
    # NO autograd anywhere in the shipped module source.
    import inspect
    src = inspect.getsource(g)
    assert "autograd" not in src and "torch" not in src
