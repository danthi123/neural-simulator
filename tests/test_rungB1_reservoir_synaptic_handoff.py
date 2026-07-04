"""CI for RUNG B-1 -- the reservoir's learned role output drives the composer's bind SYNAPTICALLY (role_route
gates), replacing the host {role:word} dict. Fast structural tests + one slow 6-anti-cheat GO gate (seed 42)."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import pytest

import research.runners._rungB1_reservoir_synaptic_handoff_derisk as b1


def test_imports_and_constants():
    # reused I5a instruments + the reservoir comprehender are importable (no drift in the reused APIs)
    assert callable(b1._gate_open) and callable(b1.lesion_route) and callable(b1.provenance_role_bank_current)
    assert b1.SYNAPTIC_ROUTE_ROLES == ("agent", "action", "patient")
    assert b1.N_TEST >= 3 and b1.PROJ_DIM >= 128


def test_reservoir_roles_shape():
    """A transitive SVO must yield exactly the three composer roles, in order, from the reservoir read-out."""
    corpus = b1.setup_corpus(seed=42)
    comp = b1.ReservoirComprehender(42, corpus["discovered"])
    rng = np.random.default_rng(42 * 101 + 5)
    comp.fit(b1._gen(b1._TRAIN_KINDS, b1._N_TRAIN_PER_CONSTRUCTION, rng, corpus["subj"], corpus["verb"], corpus["obj"]))
    toks = corpus["test"][0][0]
    pairs = b1._reservoir_roles(comp, toks)
    roles = [r for _w, r in pairs]
    assert set(roles) <= {"agent", "action", "patient"}
    assert set(roles) == {"agent", "action", "patient"}, f"expected all 3 roles, got {roles}"


@pytest.mark.slow
def test_seed42_go():
    """The full 6-anti-cheat de-risk on seed 42 is a GO: the synaptic hand-off recovers the facts, is never worse
    than the host-dict path, gated-by-firing, provenance-clean, and BOTH the route-lesion and the reservoir-lesion
    collapse recall (the route AND the reservoir are load-bearing)."""
    corpus = b1.setup_corpus(seed=42)
    r = b1.run_seed(42, corpus)
    assert r["route_correct"] >= 0.75 * r["n_queries"], r
    assert r["route_not_worse_than_dict"], r
    assert r["moat_false_accept"] <= 0.05, r
    assert r["gated_by_firing"], r["gate_trace"]
    assert r["provenance"]["clean"], r["provenance"]
    assert r["route_lesion_collapses"], r
    assert r["res_lesion_collapses"], r
    assert r["seed_GO"], r
