"""CI guard for counterfactual recombination (open-world inference #5 R-ii, CYCLE 1059): 'what if role R were F'?'
via unbind/substitute/re-bind -- clean substitution-fidelity (no mush), a plausibility gate ranking coherent above
incoherent substitutions (collapses under a shuffled graph), moat 0-leak. numpy/offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
from research.runners._counterfactual_recombination_derisk import run_seed


def test_counterfactual_fidelity_gate_and_shuffle():
    r = run_seed(42)
    rs = run_seed(42, shuffled=True)
    assert r["fidelity"] > 0.95, r                 # unbind/substitute/re-bind produces the exact counterfactual scene
    assert r["gate"] > 0.85, r                     # the plausibility gate ranks coherent > incoherent substitution
    assert r["plaus_mean"] - r["implaus_mean"] > 0.20, r   # plausible substitution clearly more coherent
    assert r["gate"] - rs["gate"] > 0.30, (r, rs)  # the gate collapses under a shuffled graph (signal load-bearing)
    assert r["moat_leak"] < 0.01, r                # counterfactual flagged, never asserted as a stored fact
