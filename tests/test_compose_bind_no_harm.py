"""LOAD-BEARING no-harm: protected/validated modules byte-UNTOUCHED
across the whole compose-bind range (plan-base 2fde0ed..HEAD); NO
shipped path imports autograd/torch. Self-contained (base SHA is a
module default; no shared conftest change)."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BASE = "2fde0ed"
PROTECTED = [
    "research/runners/abstention_gate.py", "tests/test_abstention_gate.py",
    "sim/td_value_critic.py", "sim/kernels.py", "sim/neuromodulators.py",
    "sim/train_checkpoint.py", "sim/backend.py",
    "sim/dendritic_plasticity.py", "research/runners/td_critic_core.py",
    "research/runners/dendritic_fair_core.py"]


def test_protected_byte_untouched_across_range():
    diff = subprocess.run(
        ["git", "diff", "--name-only", "%s..HEAD" % _BASE, "--"]
        + PROTECTED, capture_output=True, text=True, cwd=ROOT)
    assert diff.stdout.strip() == "", "PROTECTED MODIFIED:\n" + diff.stdout


def test_no_autograd_in_shipped_path():
    for p in ("sim/compose_temporal_bind.py",
              "research/runners/compose_bind_core.py",
              "research/runners/compose_bind_gate.py"):
        s = (ROOT / p).read_text()
        assert "autograd" not in s and "torch" not in s, p
