"""CI guard for analogical transfer via the parallelogram on learned co-occurrence codes (open-world inference #3,
CYCLE 1055): A:B :: C:? recovers the correct target, BEATS a "just C's neighbour" baseline (genuine analogy, not
retrieval), and collapses under permuted codes. Locks in the mechanism + its over-claim controls. numpy/offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
from research.runners._realcorpus_analogical_transfer_derisk import run_seed


def test_analogy_beats_baseline_and_permuted():
    r = run_seed(42)
    assert r["para"] >= 0.70, r                      # parallelogram recovers the analogy target
    assert r["para"] - r["base"] > 0.30, r           # BEATS the C-neighbour baseline -> genuine analogy, not retrieval
    assert r["para"] - r["perm"] > 0.30, r           # collapses under permuted codes -> learned structure load-bearing
    assert r["base"] < 0.30, r                       # the baseline is genuinely weak (the key over-claim control)
