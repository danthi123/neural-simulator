import subprocess, sys, importlib, inspect


# RETIRED 2026-10-01 (owner ruling, runbook section 6 `no-harm-retire`): `test_protected_byte_empty` pinned this arc's protected
# files byte-unchanged since 02addfa63 (Q2 falsify-first probe, 2026-05-18). Main has since edited them on purpose, so the pin
# only measured elapsed time (test triage, research/draft-test-triage @ 636cbf76a, REAL group). Arc record:
# `git show 02addfa63c6b7261e8319f5eedb807d4c3c967be`.


def test_no_confab_moat_7_of_7():
    r = subprocess.run([sys.executable,"-m","pytest",
                        "tests/test_abstention_gate.py","-q"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]


def test_core_is_pure_no_torch():
    core = importlib.import_module(
        "research.runners.constrained_decode_core")
    src = inspect.getsource(core)
    assert "import torch" not in src and "backward(" not in src


def test_net_new_gate_no_new_training():
    g = importlib.import_module(
        "research.runners.constrained_decode_gate")
    src = inspect.getsource(g)
    assert "backward(" not in src and ".step()" not in src
