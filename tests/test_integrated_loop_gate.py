import json
import subprocess
import sys
from pathlib import Path


def test_integration_runner_tiny_smoke_produces_verdict(tmp_path):
    """Grounding pin: the integration runner exists, runs a fast
    --tiny-synth smoke end-to-end on the CPU backend, and writes a
    verdict JSON whose classification is the explicitly-not-propagated
    TINY marker (never a real PASS/FAIL/VOID at toy scale)."""
    out = tmp_path / "tiny.json"
    proc = subprocess.run(
        [sys.executable, "-m", "research.runners.integrated_loop_gate",
         "--tiny-synth", "--seeds", "42", "43", "44",
         "--out", str(out)],
        capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        "runner failed: %s\n%s" % (proc.stdout, proc.stderr))
    assert out.exists(), "runner did not write the verdict JSON"
    v = json.loads(out.read_text())
    assert "GATE" in v, "verdict has no GATE field"
    assert "TINY" in json.dumps(v), (
        "tiny-synth verdict must be marked TINY / NOT propagated")
