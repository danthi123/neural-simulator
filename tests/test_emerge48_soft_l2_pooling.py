"""CI guard for EMERGE-48 — SOFT / UNION L2 POOLING (HTM temporal pooler / HMAX soft-max) to surpass the EMERGE-46
fully-spiking-stacked-pooler boundary. These tests pin the MECHANISM FACTS (fast, CPU/numpy): (1) softening the L2
winner-inactive depression rate RECOVERS held-out generalization from the over-selective failing regime, with within-super
overlap EXCEEDING cross-super (generalization, NOT indiscriminate collision); (2) the over-selective L2 regime is the
faithful FAILING regime (near-zero held-out within-super overlap, reproducing the on-substrate boundary); (3) a low L2
ld_wi does NOT raise cross-super overlap (so the anti-cheat holds). The DECISIVE on-substrate port (the GO) is a slow
bridge-build run and is validated by the runner + finding, not pinned here. Skip if deps missing.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pytest

pytest.importorskip("sim.bridge")
_mod = pytest.importorskip("research.runners._emerge48_soft_l2_pooling_derisk")

SoftL2NumpyProbe = _mod.SoftL2NumpyProbe
POOL_LD_STRONG = _mod.POOL_LD_STRONG


def test_over_selective_l2_is_the_failing_regime():
    """At the OVER-SELECTIVE L2 winner-inactive depression rate (POOL_LD_STRONG=0.15, the on-substrate-faithful regime),
    the held-out within-super L2 overlap is near-zero and super-acc is near chance -- this reproduces the EMERGE-46
    boundary (the pooler tunes tightly to seen members' discriminative features; a held-out sub-category does not share
    the L2 columns)."""
    seeds = [42, 43, 44]
    wi, acc = [], []
    for s in seeds:
        p = SoftL2NumpyProbe(seed=s, epochs=40, l2_ld=POOL_LD_STRONG, normalize=False)
        w, _ = p.held_out_within_cross_overlap()
        wi.append(w); acc.append(float(p.held_out_super_acc()))
    wi, acc = float(np.mean(wi)), float(np.mean(acc))
    assert wi < 0.03, f"over-selective L2 must be in the failing regime (held-out within-super overlap {wi:.3f})"
    assert acc < 0.40, f"over-selective L2 super-acc must be near chance/collapse (got {acc:.2f})"


def test_soft_l2_recovers_generalization_without_collision():
    """The core EMERGE-48 claim (numpy): SOFTENING the L2 depression (low ld_wi) RECOVERS held-out generalization to GO,
    and it is GENERALIZATION not indiscriminate collision -- within-super held-out overlap EXCEEDS cross-super, and
    super-acc jumps from the over-selective failing regime to >= 0.80. This is the dominant lever EMERGE-47 identified."""
    seeds = [42, 43, 44]
    strong_acc, soft_acc, soft_wi, soft_cr = [], [], [], []
    for s in seeds:
        ps = SoftL2NumpyProbe(seed=s, epochs=40, l2_ld=POOL_LD_STRONG, normalize=False)
        strong_acc.append(float(ps.held_out_super_acc()))
        pf = SoftL2NumpyProbe(seed=s, epochs=40, l2_ld=0.01, normalize=False)
        soft_acc.append(float(pf.held_out_super_acc()))
        w, c = pf.held_out_within_cross_overlap(); soft_wi.append(w); soft_cr.append(c)
    strong_acc, soft_acc = float(np.mean(strong_acc)), float(np.mean(soft_acc))
    soft_wi, soft_cr = float(np.mean(soft_wi)), float(np.mean(soft_cr))
    assert soft_acc >= 0.80, f"soft/union L2 must recover held-out super-acc to GO (got {soft_acc:.2f})"
    assert soft_acc > strong_acc + 0.25, f"soft must beat the over-selective regime ({strong_acc:.2f} -> {soft_acc:.2f})"
    # the shortcut guard: within-super must EXCEED cross-super (generalization, not indiscriminate collision)
    assert soft_wi > soft_cr + 0.05, f"within-super {soft_wi:.3f} must exceed cross-super {soft_cr:.3f} (generalization not collision)"


def test_soft_l2_does_not_raise_cross_super_overlap():
    """Softening the L2 depression must NOT raise the CROSS-super held-out overlap vs the over-selective regime -- else
    the recovery would be indiscriminate collision (which breaks the anti-cheat: permuted would also route). Cross-super
    stays at/near zero across the soft regime, so the recovery comes from SHARED same-super columns, not collision."""
    seeds = [42, 43, 44]
    for ld in (0.005, 0.01, 0.02):
        cr = []
        for s in seeds:
            p = SoftL2NumpyProbe(seed=s, epochs=40, l2_ld=ld, normalize=False)
            _, c = p.held_out_within_cross_overlap(); cr.append(c)
        cr = float(np.mean(cr))
        assert cr <= 0.03, f"soft ld_wi={ld} must keep cross-super overlap near zero (got {cr:.3f})"
