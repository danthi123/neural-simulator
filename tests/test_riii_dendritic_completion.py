"""CI guard for the R-iii dendritic-completion surpass de-risk (CYCLE 1065): a supra-linear dendritic integration
(NMDA-plateau + synaptic clustering) completes a partial cue where the point-neuron linear read-out fails, at the
same connectivity, specifically, and collapses under shuffled weights. numpy/offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
from research.runners._riii_dendritic_completion_derisk import run_seed


def test_dendritic_beats_linear_completion_specifically():
    for s in (42, 43):
        r = run_seed(s)
        assert r["den_held"] > 0.7, r                    # dendritic completes the held-out neurons
        assert r["den_held"] - r["lin_held"] > 0.4, r    # far above the linear read-out (non-linearity load-bearing)
        assert r["den_non"] < 0.15, r                    # specific: non-stored neurons don't fire
        assert r["den_shuf_held"] < 0.3, r               # shuffled weights collapse completion (rides the attractor)
