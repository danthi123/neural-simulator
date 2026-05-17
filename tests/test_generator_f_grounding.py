"""Grounding: the generator_f_gate pipeline TURNS end-to-end on local
shakespeare (zero network) at a TINY config and produces an
interpretable verdict. Competence+coherence already grounded by probe
bzzzmy1se. Green after Task 3."""
import os
import subprocess
import sys
import json
import pytest


def test_generator_f_gate_pipeline_turns_local(tmp_path):
    if not os.path.exists("data/tinyshakespeare.txt"):
        pytest.skip("local grounding corpus absent")
    out = str(tmp_path / "f.json")
    ck = str(tmp_path / "f.ckpt")
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.generator_f_gate",
         "--seeds", "42,43,44", "--corpus", "data/tinyshakespeare.txt",
         "--vocab-size", "96", "--d-model", "32", "--n-layer", "1",
         "--n-head", "2", "--block-size", "16", "--steps", "20",
         "--batch-size", "8", "--gen-tokens", "30",
         "--eval-positions", "40", "--device", "cpu",
         "--out", out, "--ckpt", ck],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = json.loads(open(out, encoding="utf-8").read())
    assert d["n_seeds"] == 3 and "aggregate_verdict" in d
    assert all("verdict" in s for s in d["per_seed"])
    for s in d["per_seed"]:
        assert s["uniform_ppl"] == d["config"]["vocab_size"]
        assert "gen_sample" in s
