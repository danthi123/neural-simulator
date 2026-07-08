"""CI guard for the multi-referent disambiguation REFRAME (CYCLE 1057): the predicate-COMPATIBILITY signal +
TEMPORAL INTEGRATION (a linear integrator) defeats the recency/salience confound, and no-bias stays ~chance. Locks
in the robust reframe claim (the WTA competition is not the necessary ingredient). numpy/offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
from research.runners._realcorpus_multireferent_wta_derisk import run_seed


def test_compatibility_integration_beats_recency_confound():
    for K in (3, 6):
        r = run_seed(42, K=K)
        assert r["linear"] - r["recency"] > 0.30, (K, r)     # compatibility+integration >> recency/salience confound
        assert r["recency"] < 0.10, (K, r)                   # the confound genuinely fails (binds the recent distractor)
        assert abs(r["nobias"] - 1.0 / K) < 0.12, (K, r)     # no-bias -> ~chance (no spurious winner)
