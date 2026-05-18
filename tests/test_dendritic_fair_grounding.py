"""Grounding pin: the dendritic_fair_gate pipeline TURNS end-to-end on
a TINY synthetic zero-network config and produces an interpretable
THREE-STATE (VOID/PASS/FAIL) verdict. Green after Task 3."""
import subprocess
import sys
import json
import pytest


def test_dendritic_fair_gate_pipeline_turns(tmp_path):
    out = str(tmp_path / "d.json")
    ck = str(tmp_path / "d.ckpt")
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.dendritic_fair_gate",
         "--seeds", "42,43,44", "--tiny-synth", "--out", out,
         "--ckpt-dir", ck],
        capture_output=True, text=True, timeout=900)
    if r.returncode == 2 and "NOT RUNNABLE" in r.stdout:
        pytest.skip("dependency/dataset absent in this env")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = json.loads(open(out, encoding="utf-8").read())
    assert d["n_seeds"] == 3 and d["GATE"] in ("VOID", "PASS", "FAIL")
    for s in d["per_seed"]:
        assert s["verdict"]["GATE"] in ("VOID", "PASS", "FAIL")
