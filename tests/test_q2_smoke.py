"""--tiny smoke: gate runs end-to-end (CPU OK for smoke; DECISIVE run is
GPU/controller-only), cdc_verdict-shaped, toy verdict NOT propagated,
no-confab preserved on ungrounded (LM never touched)."""
import json, subprocess, sys, os


def test_tiny_smoke_runs(tmp_path):
    out = tmp_path / "q2_smoke.json"
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.constrained_decode_gate",
         "--tiny", "--seeds", "42", "43", "44", "--out", str(out)],
        capture_output=True, text=True, timeout=3600, env={**os.environ})
    assert r.returncode == 0, r.stderr[-3000:]
    d = json.loads(out.read_text())
    assert d["tiny"] is True and d["note"].startswith("TINY")
    assert len(d["ladder"]) == 1
    ps = d["ladder"][0]["verdict"]["per_seed"]
    assert all(v["abstain_on_ungrounded_rate"]
               >= v["bare_moat_abstain_rate"] - 1e-9 for v in ps.values())
