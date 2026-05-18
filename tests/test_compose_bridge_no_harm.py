"""LOAD-BEARING no-harm: protected/validated modules byte-UNTOUCHED
across the whole compose-bridge range (e8a99a2..HEAD); NO shipped path
imports autograd/torch."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BASE = "e8a99a2"
PROTECTED = [
    "research/runners/abstention_gate.py", "tests/test_abstention_gate.py",
    "sim/td_value_critic.py", "sim/compose_temporal_bind.py",
    "sim/kernels.py", "sim/bridge.py", "sim/neuromodulators.py",
    "sim/train_checkpoint.py", "sim/backend.py",
    "sim/dendritic_plasticity.py",
    "research/runners/text_minimal_isolation.py",
    "research/runners/compose_bind_core.py",
    "research/runners/td_critic_core.py",
    "research/runners/dendritic_fair_core.py"]


def test_protected_byte_untouched_across_range():
    diff = subprocess.run(
        ["git", "diff", "--name-only", "%s..HEAD" % _BASE, "--"]
        + PROTECTED, capture_output=True, text=True, cwd=ROOT)
    assert diff.stdout.strip() == "", "PROTECTED MODIFIED:\n" + diff.stdout


def test_no_autograd_in_shipped_path():
    for p in ("research/runners/compose_bridge_core.py",
              "research/runners/compose_bridge_gate.py"):
        s = (ROOT / p).read_text()
        assert "autograd" not in s and "torch" not in s, p
