"""Phase B no-harm: the net-new gate harms NOTHING protected."""
import subprocess, sys, importlib, inspect


# RETIRED 2026-10-01 (owner ruling, runbook section 6 `no-harm-retire`): `test_protected_set_byte_empty_diff` pinned this arc's protected
# files byte-unchanged since bda6e462c (engram-bootstrap arc base, 2026-05-18). Main has since edited them on purpose, so the pin
# only measured elapsed time (test triage, research/draft-test-triage @ 636cbf76a, REAL group). Arc record:
# `git show bda6e462cf8d012a70966e93921a35aa5201182d`.


def test_no_confab_moat_still_7_of_7():
    r = subprocess.run([sys.executable, "-m", "pytest",
                        "tests/test_abstention_gate.py", "-q"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]


def test_no_autograd_in_shipped_path():
    g = importlib.import_module("research.runners.engram_bootstrap_gate")
    src = inspect.getsource(g)
    assert "autograd" not in src and "import torch" not in src


def test_cbr_verdict_reused_byte_identical():
    from research.runners.compose_bridge_core import cbr_verdict
    g = importlib.import_module("research.runners.engram_bootstrap_gate")
    assert g.cbr_verdict is cbr_verdict
