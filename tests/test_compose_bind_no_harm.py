"""LOAD-BEARING no-harm: protected/validated modules byte-UNTOUCHED
across the whole compose-bind range (plan-base 2fde0ed..HEAD); NO
shipped path imports autograd/torch. Self-contained (base SHA is a
module default; no shared conftest change)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BASE = "2fde0ed"
PROTECTED = [
    "research/runners/abstention_gate.py", "tests/test_abstention_gate.py",
    "sim/td_value_critic.py", "sim/kernels.py", "sim/neuromodulators.py",
    "sim/train_checkpoint.py", "sim/backend.py",
    "sim/dendritic_plasticity.py", "research/runners/td_critic_core.py",
    "research/runners/dendritic_fair_core.py"]


# RETIRED 2026-10-01 (owner ruling, runbook section 6 `no-harm-retire`): `test_protected_byte_untouched_across_range` pinned this arc's protected
# files byte-unchanged since 2fde0ed24 (compose-bind plan base, 2026-05-18). Main has since edited them on purpose, so the pin
# only measured elapsed time (test triage, research/draft-test-triage @ 636cbf76a, REAL group). Arc record:
# `git show 2fde0ed246ea93d59ab9b5e06d8e49466ca81fbe`.


def test_no_autograd_in_shipped_path():
    for p in ("sim/compose_temporal_bind.py",
              "research/runners/compose_bind_core.py",
              "research/runners/compose_bind_gate.py"):
        s = (ROOT / p).read_text()
        assert "autograd" not in s and "torch" not in s, p
