"""Phase-factored integrated loop grounding pin (Task 0).

This is a GROUNDING PIN: it fixes the contracts of the phase-factored
integrated-loop arc UP FRONT so later tasks (Task 1: cheap probe;
Task 2: spiking controller) cannot drift silently from the design.

The grounding-pin pattern is the same disciplined pattern used by prior
project arcs (Direction 4 / Direction 7 grounding pins; the
generative-replay Task 0). The contracts are codified BEFORE the
implementation lands so the implementation must conform to the pin,
not the other way around.

Status: intentionally green-with-skips at Task 0.
- (a) cheap probe `run_probe` + `probe_verdict` exposure  -> SKIP until Task 1.
- (b) cheap probe frozen bar `_PROBE_BAR == 0.90`         -> SKIP until Task 1.
- (c) spiking controller reuses 4 subsystems by import +
      imports parked `integrated_loop_core.integrated_loop_verdict`
                                                            -> SKIP until Task 2.
- (d) parked verdict `integrated_loop_core.py` frozen bars
      (_IL_V1_MIN, _IL_SCI_MIN, _IL_LESION_MAX, _IL_LADDER)
                                                            -> RUNS NOW.

Frozen bars pinned here mirror the pre-registered constants in
research/runners/integrated_loop_core.py lines 52-57:
  _IL_LADDER     == (2, 4, 8)
  _IL_V1_MIN     == 0.90
  _IL_SCI_MIN    == 0.80
  _IL_LESION_MAX == 0.40
  _IL_MIN_SEEDS  == 3
The cheap probe's own bar is frozen separately at _PROBE_BAR == 0.90.
"""
from __future__ import annotations
import importlib.util
import os
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pre-registered absolute target paths for each task's deliverable.
_CHEAP_PROBE_PATH = os.path.join(
    REPO_ROOT,
    "research/findings/raw/phase_factored_cheap_probe.py",
)
_CONTROLLER_PATH = os.path.join(
    REPO_ROOT,
    "research/runners/phase_factored_loop_gate.py",
)
_PARKED_VERDICT_PATH = os.path.join(
    REPO_ROOT,
    "research/runners/integrated_loop_core.py",
)


def _load_module(mod_name, path):
    """Load a module by absolute path (same importlib pattern the repo
    uses in test_direction_7_grounding.py)."""
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# (a) Cheap probe module exists and exposes run_probe + probe_verdict.
#     SKIP until Task 1 lands the file.
# ---------------------------------------------------------------------------
def test_a_cheap_probe_module_exists_and_exposes_api():
    """Task 1: cheap probe exposes run_probe + probe_verdict."""
    if not os.path.exists(_CHEAP_PROBE_PATH):
        pytest.skip("Task 1 not landed yet (cheap probe module absent)")
    mod = _load_module("phase_factored_cheap_probe", _CHEAP_PROBE_PATH)
    assert hasattr(mod, "run_probe"), (
        "cheap probe module missing run_probe()"
    )
    assert hasattr(mod, "probe_verdict"), (
        "cheap probe module missing probe_verdict()"
    )


# ---------------------------------------------------------------------------
# (b) Cheap probe's frozen bar constant _PROBE_BAR is exactly 0.90 and is a
#     module-level constant. SKIP until Task 1 lands the file.
# ---------------------------------------------------------------------------
def test_b_cheap_probe_bar_frozen_at_0_90():
    """Task 1: cheap probe frozen bar _PROBE_BAR == 0.90 (module-level)."""
    if not os.path.exists(_CHEAP_PROBE_PATH):
        pytest.skip("Task 1 not landed yet (cheap probe module absent)")
    mod = _load_module("phase_factored_cheap_probe", _CHEAP_PROBE_PATH)
    assert hasattr(mod, "_PROBE_BAR"), (
        "cheap probe module missing module-level _PROBE_BAR constant"
    )
    assert mod._PROBE_BAR == 0.90, (
        "_PROBE_BAR tampered: design fixes the cheap-probe bar at 0.90"
    )


# ---------------------------------------------------------------------------
# (c) Spiking controller module exists, reuses the four validated subsystems
#     by import, and imports the parked frozen verdict
#     integrated_loop_core.integrated_loop_verdict.
#     SKIP until Task 2 lands the file.
# ---------------------------------------------------------------------------
def test_c_controller_module_exists():
    """Task 2: spiking controller module exists."""
    if not os.path.exists(_CONTROLLER_PATH):
        pytest.skip("Task 2 not landed yet (spiking controller absent)")
    # Presence is the only assertion here; reuse contracts checked below.
    assert os.path.exists(_CONTROLLER_PATH)


def test_c_controller_reuses_four_subsystems_by_import():
    """Task 2: spiking controller must REUSE the four validated subsystems
    by import (grep the source for the import references), NOT reimplement
    them. The four subsystems are:
      1. engram-tag API usage (start_engram_recording / stimulate_tag /
         commit_engram_tag),
      2. consolidation_trainer (hippocampal CLS replay),
      3. concept_pool_demo (concept-pool architecture builder),
      4. abstention_gate (the calibrated abstention/decision gate).
    """
    if not os.path.exists(_CONTROLLER_PATH):
        pytest.skip("Task 2 not landed yet (spiking controller absent)")
    with open(_CONTROLLER_PATH, "r", encoding="utf-8") as f:
        src = f.read()
    # 1. engram-tag API reuse (any of the canonical tag-API calls present).
    assert (
        "start_engram_recording" in src
        or "commit_engram_tag" in src
        or "stimulate_tag" in src
        or "engram" in src
    ), (
        "controller must reuse the validated engram-tag API "
        "(start_engram_recording / commit_engram_tag / stimulate_tag)"
    )
    # 2. consolidation_trainer reuse.
    assert "consolidation_trainer" in src, (
        "controller must reuse consolidation_trainer (hippocampal CLS replay)"
    )
    # 3. concept_pool_demo reuse.
    assert "concept_pool_demo" in src, (
        "controller must reuse concept_pool_demo (concept-pool builder)"
    )
    # 4. abstention_gate reuse.
    assert "abstention_gate" in src, (
        "controller must reuse abstention_gate (calibrated decision gate)"
    )


def test_c_controller_imports_parked_frozen_verdict():
    """Task 2: spiking controller must import the parked frozen verdict
    (integrated_loop_core / integrated_loop_verdict) rather than redefine
    the integrated-loop bars locally."""
    if not os.path.exists(_CONTROLLER_PATH):
        pytest.skip("Task 2 not landed yet (spiking controller absent)")
    with open(_CONTROLLER_PATH, "r", encoding="utf-8") as f:
        src = f.read()
    assert "integrated_loop_core" in src, (
        "controller must import the parked integrated_loop_core module"
    )
    assert "integrated_loop_verdict" in src, (
        "controller must import the parked integrated_loop_verdict callable "
        "(the frozen integrated-loop verdict)"
    )


# ---------------------------------------------------------------------------
# (d) Parked verdict module present with frozen bars intact. RUNS NOW.
#     The parked module imports only stdlib (math, typing) so it loads
#     cleanly with no GPU / backend dependency.
# ---------------------------------------------------------------------------
def test_d_parked_verdict_module_present():
    """The parked integrated_loop_core.py module is present (Task 0 lives
    on top of a module that already exists)."""
    assert os.path.exists(_PARKED_VERDICT_PATH), (
        "parked verdict module missing: " + _PARKED_VERDICT_PATH
    )


def test_d_parked_verdict_frozen_bars_intact():
    """Pin the four frozen integrated-loop bars in integrated_loop_core.py.
    These are pre-registered BEFORE any full-model run and must never be
    tuned to an outcome. Values mirror lines 52-57 of the parked module.
    This test MUST pass now.
    """
    mod = _load_module("integrated_loop_core", _PARKED_VERDICT_PATH)
    # V1 (cheap-probe-aligned) operational bar.
    assert hasattr(mod, "_IL_V1_MIN"), (
        "integrated_loop_core missing _IL_V1_MIN"
    )
    assert mod._IL_V1_MIN == 0.90, (
        "_IL_V1_MIN tampered: design fixes this at 0.90"
    )
    # Science (multi-seed) bar.
    assert hasattr(mod, "_IL_SCI_MIN"), (
        "integrated_loop_core missing _IL_SCI_MIN"
    )
    assert mod._IL_SCI_MIN == 0.80, (
        "_IL_SCI_MIN tampered: design fixes this at 0.80"
    )
    # Lesion ceiling: a proper lesion must collapse BELOW this.
    assert hasattr(mod, "_IL_LESION_MAX"), (
        "integrated_loop_core missing _IL_LESION_MAX"
    )
    assert mod._IL_LESION_MAX == 0.40, (
        "_IL_LESION_MAX tampered: design fixes this at 0.40"
    )
    # Compositional load ladder.
    assert hasattr(mod, "_IL_LADDER"), (
        "integrated_loop_core missing _IL_LADDER"
    )
    assert tuple(mod._IL_LADDER) == (2, 4, 8), (
        "_IL_LADDER tampered: design fixes this at (2, 4, 8)"
    )


def test_d_parked_verdict_min_seeds_frozen():
    """Pin the multi-seed floor _IL_MIN_SEEDS == 3 (below three seeds a
    multi-seed claim is not supportable). RUNS NOW."""
    mod = _load_module("integrated_loop_core", _PARKED_VERDICT_PATH)
    assert hasattr(mod, "_IL_MIN_SEEDS"), (
        "integrated_loop_core missing _IL_MIN_SEEDS"
    )
    assert mod._IL_MIN_SEEDS == 3, (
        "_IL_MIN_SEEDS tampered: design fixes this at 3"
    )


def test_d_parked_verdict_exposes_integrated_loop_verdict():
    """The parked module must expose integrated_loop_verdict (the callable
    the Task 2 controller imports). RUNS NOW."""
    mod = _load_module("integrated_loop_core", _PARKED_VERDICT_PATH)
    assert hasattr(mod, "integrated_loop_verdict"), (
        "integrated_loop_core missing integrated_loop_verdict() callable"
    )
