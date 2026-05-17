"""Grounding: the generator_e_gate pipeline TURNS end-to-end on local
shakespeare (zero network) and produces an interpretable verdict.
(N-gram ppl competence is already grounded by probe ba1jyepwf ~14-15;
this pin is the END-TO-END pipeline gate.) Green after Task 3."""
import os
import subprocess
import sys
import json
import pytest


def test_generator_e_gate_pipeline_turns_local(tmp_path):
    if not os.path.exists("data/tinyshakespeare.txt"):
        pytest.skip("local grounding corpus absent")
    out = str(tmp_path / "e.json")
    ck = str(tmp_path / "e.ckpt")
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.generator_e_gate",
         "--seeds", "42,43,44", "--corpus", "data/tinyshakespeare.txt",
         "--vocab-size", "96", "--gen-tokens", "40",
         "--eval-positions", "60", "--out", out, "--ckpt", ck],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    d = json.loads(open(out, encoding="utf-8").read())
    assert d["n_seeds"] == 3 and "aggregate_verdict" in d
    assert all("verdict" in s for s in d["per_seed"])
    for s in d["per_seed"]:
        assert s["uniform_ppl"] == d["config"]["vocab_size"]
