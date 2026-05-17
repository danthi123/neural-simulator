"""Grounding: the generator_g_gate pipeline TURNS end-to-end at a
tiny zero-network config and produces an interpretable verdict.
(Faithfulness-is-hard already grounded by the conditioning probe.)
Green after Task 3."""
import subprocess
import sys
import json
import pytest


def test_generator_g_gate_pipeline_turns(tmp_path):
    out = str(tmp_path / "g.json")
    ck = str(tmp_path / "g.ckpt")
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.generator_g_gate",
         "--seeds", "42,43,44", "--tiny", "--out", out,
         "--ckpt", ck],
        capture_output=True, text=True, timeout=600)
    if r.returncode == 2 and "NOT RUNNABLE" in r.stdout:
        pytest.skip("trained Generator-F ckpt absent in this env")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = json.loads(open(out, encoding="utf-8").read())
    assert d["n_seeds"] == 3 and "aggregate_verdict" in d
    for s in d["per_seed"]:
        assert "verdict" in s and "transcripts" in s
