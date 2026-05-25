"""Direction Q grounding pin - intentionally RED until later tasks land.

These tests pin the contracts the Direction Q standalone test bridge
runner MUST satisfy. They are RED on Task 0 commit; turn GREEN as
Tasks 1-5 land per docs/plans/2026-05-25-direction-Q-dlpfc-scale-up-
implementation.md. Final tests keep the contract permanent.

The grounding-pin pattern is the same disciplined pattern used by
prior project arcs (Task 0 of the (c) generative-replay TDD plan etc.):
the contracts are codified UP FRONT so the implementation cannot drift
silently from the design doc bar.
"""
from __future__ import annotations
import importlib.util
import os
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_direction_Q_runner_module_exists():
    """Task 4: the standalone test bridge runner module exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_Q_dlpfc_scale_up_standalone.py",
    )
    assert os.path.exists(path), (
        "Task 4 not yet landed: " + path + " does not exist"
    )


def test_direction_Q_verdict_module_exists():
    """Task 3: the verdict module exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_Q_verdict.py",
    )
    assert os.path.exists(path), (
        "Task 3 not yet landed: " + path + " does not exist"
    )


def test_direction_Q_bridge_builder_module_exists():
    """Task 1: the standalone test bridge builder exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_Q_bridge_builder.py",
    )
    assert os.path.exists(path), (
        "Task 1 not yet landed: " + path + " does not exist"
    )


def test_direction_Q_protocol_module_exists():
    """Task 2: the stim + delay protocol module exists."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_Q_protocol.py",
    )
    assert os.path.exists(path), (
        "Task 2 not yet landed: " + path + " does not exist"
    )


def test_direction_Q_verdict_thresholds_frozen():
    """Task 3: pre-registered thresholds are present and match design
    doc values, not modifiable by results."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_Q_verdict.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 3 not landed yet")
    spec = importlib.util.spec_from_file_location(
        "direction_Q_verdict", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Frozen thresholds (must be present)
    assert hasattr(mod, "_Q_RATE_RATIO_MIN"), (
        "verdict module missing _Q_RATE_RATIO_MIN"
    )
    assert hasattr(mod, "_Q_DELAY_MIN_SEC"), (
        "verdict module missing _Q_DELAY_MIN_SEC"
    )
    assert hasattr(mod, "_Q_MIN_SEEDS_PASS"), (
        "verdict module missing _Q_MIN_SEEDS_PASS"
    )
    # Pre-registered values (must equal design doc; tampering would
    # be caught here)
    assert mod._Q_RATE_RATIO_MIN == 2.0, (
        "_Q_RATE_RATIO_MIN tampered: design doc fixes this at 2.0"
    )
    assert mod._Q_DELAY_MIN_SEC == 3.0, (
        "_Q_DELAY_MIN_SEC tampered: design doc fixes this at 3.0"
    )
    assert mod._Q_MIN_SEEDS_PASS == 3, (
        "_Q_MIN_SEEDS_PASS tampered: design doc fixes this at 3"
    )


def test_direction_Q_runner_has_nmda_off_control():
    """Task 5: runner integrates NMDA-off control."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_Q_dlpfc_scale_up_standalone.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 4 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert (
        "enable_nmda=False" in src
        or "ENABLE_NMDA_OFF_CONTROL" in src
        or "nmda_off" in src.lower()
    ), (
        "Task 5 control runner not yet integrated -- runner must "
        "include an enable_nmda=False (AMPA-only) control sweep"
    )


def test_direction_Q_verdict_void_branch_exists():
    """Task 3: verdict module must distinguish VOID (control-also-pass)
    from PASS. Without this, a substrate bug that drives persistence
    without NMDA would silently inflate the result."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_Q_verdict.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 3 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert (
        "VOID" in src or "void" in src
    ), (
        "verdict module must include VOID branch for "
        "control-also-passed case"
    )
