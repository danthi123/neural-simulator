import subprocess, sys, importlib, inspect


# RETIRED 2026-10-01 (owner ruling, runbook section 6 `no-harm-retire`): `test_protected_byte_empty` pinned this arc's protected
# files byte-unchanged since a1035cff5 (Q2R design, 2026-05-18). Main has since edited them on purpose, so the pin
# only measured elapsed time (test triage, research/draft-test-triage @ 636cbf76a, REAL group). Arc record:
# `git show a1035cff5759fd912a586c3d2bb6c3fc2d3f2acc`.


def test_no_confab_moat_7_of_7():
    r = subprocess.run([sys.executable,"-m","pytest",
                        "tests/test_abstention_gate.py","-q"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]


def test_q2r_core_pure_and_imports_no_core():
    c = importlib.import_module("research.runners.q2r_core")
    src = inspect.getsource(c)
    assert "import torch" not in src and "backward(" not in src
    assert "constrained_decode_core" not in src


def test_q2r_gate_reuses_byte_unmodified_and_no_new_training():
    g = importlib.import_module("research.runners.q2r_gate")
    from research.runners.constrained_decode_gate import _GroundedConstrainedLM
    from research.runners.constrained_decode_core import cdc_verdict
    assert g._GroundedConstrainedLM is _GroundedConstrainedLM
    assert g.cdc_verdict is cdc_verdict
    src = inspect.getsource(g)
    assert "backward(" not in src and ".step()" not in src
    assert "optimizer" not in src.lower() and "loss" not in src.lower()
