"""--tiny-synth smoke: the gate RUNS end-to-end in-bridge, emits the
cbr_verdict-shaped per-rung structure with n_rewarded>0 (proves the
engram bootstrap dissolved the compose-bridge n_rewarded=0 cause), and
the toy verdict is explicitly NOT propagated."""
import json
import subprocess
import sys
import tempfile
import os


def test_tiny_synth_smoke_runs_and_bootstraps(tmp_path):
    out = tmp_path / "smoke.json"
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.engram_bootstrap_gate",
         "--tiny-synth", "--seeds", "42", "43", "44",
         "--out", str(out)],
        capture_output=True, text=True, timeout=1800,
        env={**os.environ, "SIM_BACKEND": "numpy"})
    assert r.returncode == 0, r.stderr[-3000:]
    d = json.loads(out.read_text())
    assert d["note"].startswith("TINY-SYNTH")          # NOT propagated
    assert d["tiny_synth"] is True
    # Single shrunk rung in tiny mode; cbr_verdict-shaped.
    assert len(d["ladder"]) == 1
    rung = d["ladder"][0]
    assert rung["verdict"]["GATE"] in ("PASS", "FAIL", "VOID")
    # The decisive bootstrap evidence: n_rewarded>0 for the td condition
    # on at least one seed (the compose-bridge VOID had n_rewarded==0).
    nrew = [s.get("n_rewarded_td", 0)
            for s in rung["verdict"]["per_seed"].values()]
    assert max(nrew) > 0, "engram bootstrap failed to produce a "\
                          "rewarded episode (n_rewarded still 0)"
    assert "scale_confident" in d and d["scale_confident"] in (True, False)
