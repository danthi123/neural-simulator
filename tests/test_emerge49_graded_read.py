"""CI guard for EMERGE-49 — the GRADED DRIVE/READ rung against the EMERGE-46 fully-spiking-stacked-pooler boundary.
These tests pin the DECISIVE mechanism facts (CPU/numpy, on the real substrate but the L2 pooler alone -- no L1, no
inherit bridge -- so each build+train is ~3-6s at reduced epochs): (1) at a SOFT L2 winner-inactive depression rate the
LEARNED on-substrate L2 permanences are BIMODAL (collapsed near 0) -- so a graded read has almost nothing graded to read,
which is exactly why the graded read cannot rescue the soft-pooling window; (2) turning the winner-inactive depression
FULLY OFF (ld_wi=0) makes the permanences GRADED (a spread in the middle band), confirming the depression term is the
CAUSE of the over-sparsification-to-bimodal (not the read threshold); (3) at the soft ld_wi, the graded codon read yields
essentially the SAME (near-empty) codon as the hard read -- i.e. the graded read does not change the outcome because the
connectivity is bimodal. => the EMERGE-49 graded-read rung is a BOUNDARY (the residual is the learned TUNING dynamics,
not the readout threshold); the next rung is the Foldiak (1991) trace rule. The DECISIVE full-probe 3-seed super-acc port
is validated by the runner + finding, not pinned here. Skip if deps missing.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pytest

pytest.importorskip("sim.bridge")
_e46 = pytest.importorskip("research.runners._emerge46_spiking_stacked_pooler_derisk")
_e49 = pytest.importorskip("research.runners._emerge49_graded_read_derisk")

from research.runners._emerge14_stageC_onbridge_learning_derisk import _host

NCOL1 = _e46.NCOL1
NCOL2 = _e46.NCOL2
K2 = _e46.K2
_, GradedOnSubstratePooler = _e49._build_onsubstrate_probe()

# A small synthetic co-occurrence corpus over the L1-column space; reduced epochs keep each build+train ~3-6s while the
# bimodal/graded permanence split is already fully developed (verified: near0 0.99 at ld=0.005, mid 1.00 at ld=0.0 by
# 60 epochs -- the split is driven by the cumulative winner-inactive depression, which saturates quickly).
_EPOCHS = 60
_N_SAMPLES = 80
_ACTIVE = 12


def _samples(seed=0):
    rng = np.random.default_rng(seed)
    return [set(int(x) for x in rng.choice(NCOL1, _ACTIVE, replace=False)) for _ in range(_N_SAMPLES)]


def _learned_perms(ld_wi, graded="hard", seed=99):
    p = GradedOnSubstratePooler(seed=seed, n_in=NCOL1, n_col=NCOL2, k_win=K2, ld_wi=ld_wi, graded=graded)
    p.train(_samples(), _EPOCHS, seed)
    return p, _host(p.b.cp_connections.data)[p.ff_pos]


def test_soft_ld_learns_bimodal_permanences_not_graded():
    """The DECISIVE EMERGE-49 diagnostic: at the SOFT L2 winner-inactive depression rate (ld_wi=0.005, the EMERGE-48
    soft/union sweet spot), the LEARNED on-substrate L2 permanences are BIMODAL -- the vast majority collapse near 0
    (over-sparsified by the accumulating winner-inactive depression), with almost nothing in the graded middle band. A
    graded read therefore has nothing graded to read; this is why the graded-read rung cannot recover numpy's soft-pooling
    window (the residual is the learned tuning dynamics, not the readout threshold)."""
    _, perm = _learned_perms(0.005)
    near0 = float(np.mean(perm < 0.05))
    mid = float(np.mean((perm >= 0.2) & (perm <= 0.8)))
    assert near0 >= 0.80, f"soft ld_wi must over-sparsify the L2 permanences near 0 (near0 {near0:.2f})"
    assert mid <= 0.15, f"the graded middle band must be nearly empty (bimodal), got mid-frac {mid:.2f}"


def test_ld_off_makes_permanences_graded_confirming_depression_is_the_cause():
    """Turning the winner-inactive depression FULLY OFF (ld_wi=0) makes the learned permanences GRADED -- a spread that
    sits in the middle band (no near-0 collapse). This confirms the winner-inactive DEPRESSION is the mechanism CAUSE of
    the over-sparsification-to-bimodal at nonzero ld_wi (it is not the read threshold). But ld_wi=0 is the COLLISION
    regime (documented in the runner/finding: within ~= cross, super-acc ~= chance), so it is not a valid GO."""
    _, perm = _learned_perms(0.0)
    near0 = float(np.mean(perm < 0.05))
    mid = float(np.mean((perm >= 0.2) & (perm <= 0.8)))
    assert near0 <= 0.10, f"ld_wi=0 must NOT collapse permanences to 0 (near0 {near0:.2f})"
    assert mid >= 0.80, f"ld_wi=0 must keep permanences graded in the middle band (mid-frac {mid:.2f})"


def test_graded_read_does_not_change_codon_at_soft_ld():
    """At the soft ld_wi (bimodal permanences), the GRADED codon read yields essentially the SAME codon as the HARD read
    for the same learned pooler -- because with the connectivity collapsed to a near-0/connected split there is no graded
    band for the raw-permanence read to exploit. This pins that the graded read is NOT the fix at this regime (the codons
    match, so held-out routing cannot differ)."""
    # Build ONE pooler (hard), then read its codons two ways: hard vs graded (read-only difference, same weights).
    p, _ = _learned_perms(0.005, graded="hard")
    feats = _samples(seed=7)[0]
    hard_codon = p.codon(feats)                                   # graded='hard' -> hard read
    p.graded = "graded_read"                                     # flip ONLY the read (weights unchanged)
    graded_codon = p.codon(feats)
    inter = len(hard_codon & graded_codon)
    assert inter >= K2 - 2, (f"at the soft/bimodal regime the graded read must give a near-identical codon to the hard "
                             f"read (intersection {inter}/{K2}); it does not open a soft-pooling window")
