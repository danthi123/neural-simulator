"""LOAD-BEARING no-harm: the protected/validated modules have NO
UNREVIEWED change since the last byte-reviewed protected edit; NO shipped
TD path imports autograd/torch. Self-contained (base SHA is a module
default; no shared conftest change).

The base SHA tracks the most recent OWNER-BYTE-REVIEWED protected edit, so
the guard means "no protected module changed since the last approval."
Bump it whenever a new protected sim/ edit is byte-approved. (Was the plan
commit 0150e5b; advanced to ed880244 — the approved N9 determinism-matvec
cleanup — after the GABA_B/GIRK + determinism edits legitimately landed.)"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TD_BASE = "c1d1a3d2"  # last byte-reviewed protected edit (N9 TD cue-shift A-CSC conductance-derivative re-applied; byte-proof COMBO e728d7f1, owner-directive auto-approved)
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
