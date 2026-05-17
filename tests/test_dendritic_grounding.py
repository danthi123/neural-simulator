"""Grounding pin: the dendritic_wa_gate pipeline TURNS end-to-end at a
tiny config and produces an interpretable verdict. (The PRINCIPLE was
already grounded by the 2026-05-17 rate/XOR falsify-cheaply probe;
this pin is the spiking-substrate end-to-end turn.) Green after Task 4."""
import subprocess
import sys
import json
import pytest


def test_dendritic_wa_gate_pipeline_turns(tmp_path):
    out = str(tmp_path / "d.json")
    ck = str(tmp_path / "d.ckpt")
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.dendritic_wa_gate",
         "--seeds", "42,43,44", "--tiny", "--out", out, "--ckpt", ck],
        capture_output=True, text=True, timeout=600)
    if r.returncode == 2 and "NOT RUNNABLE" in r.stdout:
        pytest.skip("dependency absent in this env")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = json.loads(open(out, encoding="utf-8").read())
    assert d["n_seeds"] == 3 and "aggregate_verdict" in d
    for s in d["per_seed"]:
        assert "verdict" in s and "grad_cosine" in s
