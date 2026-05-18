"""LOAD-BEARING no-harm: the protected/validated modules are
byte-UNTOUCHED across the whole TD-critic commit range (plan-base
0150e5b..HEAD); NO shipped TD path imports autograd/torch. Self-
contained (base SHA is a module default; no shared conftest change)."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TD_BASE = "0150e5b"  # the plan commit; build started after this
PROTECTED = [
    "research/runners/abstention_gate.py", "tests/test_abstention_gate.py",
    "sim/neuromodulators.py", "sim/kernels.py", "sim/bridge.py",
    "research/runners/g11_bg_runner.py", "sim/train_checkpoint.py",
    "sim/backend.py", "sim/dendritic_plasticity.py",
    "research/runners/dendritic_fair_core.py"]


def test_protected_byte_untouched_across_td_critic_range():
    diff = subprocess.run(
        ["git", "diff", "--name-only", "%s..HEAD" % _TD_BASE, "--"]
        + PROTECTED, capture_output=True, text=True, cwd=ROOT)
    assert diff.stdout.strip() == "", (
        "PROTECTED MODIFIED:\n" + diff.stdout + diff.stderr)


def test_no_autograd_in_shipped_td_path():
    for p in ("sim/td_value_critic.py",
              "research/runners/td_critic_core.py",
              "research/runners/td_critic_gate.py"):
        s = (ROOT / p).read_text()
        assert "autograd" not in s and "torch" not in s, p
