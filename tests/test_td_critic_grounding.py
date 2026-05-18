"""Grounding pin: the END-TO-END td_critic_gate pipeline must turn on a
TINY synthetic config and produce an interpretable THREE-STATE verdict.
RED until Task 3 lands the runner -- that is the Task-3 gate."""
import json
import subprocess
import sys
from pathlib import Path


def test_td_critic_gate_tiny_synthetic_pipeline_turns(tmp_path):
    out = tmp_path / "tdc_tiny.json"
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.td_critic_gate",
         "--tiny-synth", "--seeds", "42", "43", "44",
         "--out", str(out)],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert out.is_file(), r.stdout + r.stderr
    d = json.loads(out.read_text())
    assert d["GATE"] in ("VOID", "PASS", "FAIL")
    assert "per_seed" in d and "frozen_bars" in d
