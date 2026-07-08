"""CI guard for imaginative scenario recombination (open-world inference #5 R-i, CYCLE 1058): a novel coherent
multi-role scene is bundled into one composite + FACTORED back (factor-recovery=1.0, no mush), coherence beats the
shuffled-graph control, novelty high, moat 0-leak. Locks in the mechanism + its anti-mush/shuffle controls. numpy.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
from research.runners._imaginative_scenario_recombination_derisk import run_seed


def test_scenario_recombination_factor_recovery_and_controls():
    r = run_seed(42)
    rs = run_seed(42, shuffled=True)
    assert r["factor_recovery"] > 0.95, r          # the composite unbinds to the correct fillers (no bundle mush)
    assert r["coherence"] > 0.85, r                # novel scenes are thematically plausible
    assert r["coherence"] - rs["coherence"] > 0.30, (r, rs)   # shuffled-graph collapses coherence (plausibility load-bearing)
    assert r["novelty"] > 0.85, r                  # scenes are genuinely novel (not stored)
    assert r["moat_leak"] < 0.01, r                # imagined flagged, never asserted as a stored fact


def test_bundle_capacity_holds_to_several_roles():
    for nr in (2, 4, 6):
        assert run_seed(42, n_role=nr)["factor_recovery"] > 0.95, nr
