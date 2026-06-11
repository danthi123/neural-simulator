"""LOAD-BEARING no-harm: protected/validated modules have NO UNREVIEWED
change since the last owner-byte-reviewed protected edit; NO shipped path
imports autograd/torch.

The base SHA tracks the most recent byte-approved protected edit (advanced
from e8a99a2 to ed880244 — the approved N9 determinism-matvec cleanup —
after the GABA_B/GIRK + determinism edits legitimately landed). Bump it
whenever a new protected sim/ edit is byte-approved."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BASE = "ed880244"  # last byte-reviewed protected edit (N9 determinism cleanup)
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
