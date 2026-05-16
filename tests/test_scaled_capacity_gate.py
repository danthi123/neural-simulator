"""Pure-CPU TDD for the scaled-generator capacity-scan verdict.

`verdict` is a pure function (no file IO) so it is unit-testable on CPU
without any GPU training. The honest gate: the scaled (REAL-corpus)
student must beat its own PERMUTED-corpus control by >= 10% lower
held-out loss. PASS => capacity was the bottleneck. FAIL => even a
maxed local char-SNN cannot learn robust structure.
"""
from research.runners.scaled_capacity_gate import verdict


def test_pass_when_real_beats_permuted_10pct():
    v = verdict([9, 3.0], [9, 4.0])           # 3.0 <= 0.9*4.0=3.6 -> PASS
    assert v["GATE"] == "PASS" and v["real_end"] == 3.0


def test_fail_when_margin_too_small():
    v = verdict([9, 3.7], [9, 4.0])           # 3.7 > 3.6 -> FAIL
    assert v["GATE"] == "FAIL"


def test_fail_when_real_worse():
    assert verdict([9, 4.2], [9, 4.0])["GATE"] == "FAIL"


def test_baseline_pct_reported():
    v = verdict([9, 3.0], [9, 4.0], baseline_end=4.18)
    assert v["vs_baseline_pct"] > 0   # 3.0 is 28% below 4.18
