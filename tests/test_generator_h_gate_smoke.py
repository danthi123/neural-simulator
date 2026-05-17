"""Import/signature + <3-seeds->exit2 smoke. The end-to-end pipeline-
turns assertion is Task 0's grounding pin (goes green here)."""
import subprocess
import sys


def test_module_imports_and_has_main():
    import research.runners.generator_h_gate as m
    assert hasattr(m, "main") and callable(m.main)
    assert hasattr(m, "_TinyGPTLM")
    assert isinstance(m._GROUNDED, dict) and len(m._GROUNDED) >= 3
    assert isinstance(m._UNGROUNDED, list) and len(m._UNGROUNDED) >= 3


def test_fewer_than_three_seeds_exits_2():
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.generator_h_gate",
         "--seeds", "42,43", "--tiny"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 2
    assert "NOT RUNNABLE" in r.stdout
