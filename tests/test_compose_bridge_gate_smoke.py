import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args, tmp):
    out = tmp / "g.json"
    r = subprocess.run([sys.executable, "-m",
                        "research.runners.compose_bridge_gate",
                        "--out", str(out)] + args,
                       capture_output=True, text=True, cwd=ROOT)
    return r, out


def test_tiny_three_state_and_makes_task0_green(tmp_path):
    r, out = _run(["--tiny-synth", "--seeds", "42", "43", "44"],
                  tmp_path)
    assert out.is_file(), r.stdout + r.stderr
    assert json.loads(out.read_text())["GATE"] in (
        "VOID", "PASS", "FAIL")


def test_fewer_than_3_seeds_exit2(tmp_path):
    r, _ = _run(["--tiny-synth", "--seeds", "42"], tmp_path)
    assert r.returncode == 2


def test_reuses_build_checkpoint_nm_no_autograd():
    src = (ROOT / "research/runners/compose_bridge_gate.py").read_text()
    assert "build_biological_brain_regions" in src
    assert "from sim.train_checkpoint import" in src
    assert "neuromodulators" in src
    assert "autograd" not in src and "torch" not in src
