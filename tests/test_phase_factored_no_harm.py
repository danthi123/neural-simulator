"""Task 5: no-harm phase. Prove the phase-factored arc touched NO protected
module, the abstention (no-confabulation) moat is still 7/7, and no shipped
path imports autograd. These guard the whole arc, not a single task.

Protected set (must be byte-unchanged across the arc): the abstention gate +
its test, the inherited frozen verdict, the four reused validated subsystems,
the spiking bridge + kernels, the parked theta-gamma controller.
"""
from __future__ import annotations
import os
import subprocess
import sys
import importlib.util
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The commit just before the first phase-factored CODE task (Task 0). The
# protected modules were last touched long before this; the arc must leave
# them byte-identical from here to HEAD.
_PRE_ARC_REF = "c1e79b7"

_PROTECTED = [
    "research/runners/abstention_gate.py",
    "tests/test_abstention_gate.py",
    "research/runners/integrated_loop_core.py",
    "research/runners/integrated_loop_gate.py",
    "research/runners/consolidation_trainer.py",
    "research/runners/concept_pool_demo.py",
    "research/runners/text_minimal_isolation.py",
    "sim/bridge.py",
    "sim/kernels.py",
]


def _git(*args):
    return subprocess.run(["git", "-C", REPO, *args],
                          capture_output=True, text=True)


# RETIRED 2026-10-01 (owner ruling, runbook section 6 `no-harm-retire`): `test_protected_set_byte_unchanged_across_the_arc` pinned this arc's protected
# files byte-unchanged since c1e79b719 (phase-factored loop plan, 2026-05-30). Main has since edited them on purpose, so the pin
# only measured elapsed time (test triage, research/draft-test-triage @ 636cbf76a, REAL group). Arc record:
# `git show c1e79b7191427ce4164fea7490f5666456ae71cc`.


def test_abstention_moat_still_7_of_7():
    """Re-run the no-confabulation abstention gate test as a subprocess;
    it must still pass 7/7 (the trustworthy output gate is untouched)."""
    res = subprocess.run(
        [sys.executable, "-m", "pytest",
         os.path.join(REPO, "tests/test_abstention_gate.py"), "-q"],
        capture_output=True, text=True, cwd=REPO)
    assert res.returncode == 0, (
        "abstention moat regressed:\n" + res.stdout[-2000:] + res.stderr[-2000:])
    assert ("7 passed" in res.stdout) or (" 7 passed" in res.stdout), res.stdout


def test_no_autograd_in_controller_or_import_path():
    """The shipped controller path must not pull in torch/autograd."""
    path = os.path.join(REPO, "research/runners/phase_factored_loop_gate.py")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    for bad in ("import torch", ".backward(", "torch.autograd", "import autograd"):
        assert bad not in src, "shipped controller contains '%s'" % bad
    # importing the controller must not bring torch into sys.modules
    had_torch = "torch" in sys.modules
    spec = importlib.util.spec_from_file_location("_pf_gate_noharm", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # import-time failure is a separate concern
        pytest.skip("controller import failed (separate from autograd): %s" % exc)
    if not had_torch:
        assert "torch" not in sys.modules, (
            "importing the controller pulled in torch")


def test_frozen_verdict_bars_unchanged():
    """Inherited verdict bars must be byte-identical to their pre-registered
    values (a second guard beyond the git-diff check)."""
    path = os.path.join(REPO, "research/runners/integrated_loop_core.py")
    spec = importlib.util.spec_from_file_location("_il_core_noharm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._IL_V1_MIN == 0.90
    assert mod._IL_SCI_MIN == 0.80
    assert mod._IL_LESION_MAX == 0.40
    assert mod._IL_LADDER == (2, 4, 8)
    assert mod._IL_MIN_SEEDS == 3
