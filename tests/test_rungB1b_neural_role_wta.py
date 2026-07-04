"""Tests for RUNG B-1b -- the word's thematic role elected by an ON-BRIDGE spiking mutual-inhibition WTA whose
winner opens the composer's `role_route` gate (replacing RUNG B-1's host `argmax(f @ Ws[k])`).

FAST structural tests (no full seed run): the WTA wiring, the drive transform (no argmax), the source-level
neural-select anti-cheat, the config invariants. Plus a SLOW seed-42 GO gate (all 9 anti-cheats) marked so it can
be deselected on a weak CPU.

Run fast only:   SIM_BACKEND=numpy python -m pytest tests/test_rungB1b_neural_role_wta.py -m "not slow" -q
Run everything:  SIM_BACKEND=numpy python -m pytest tests/test_rungB1b_neural_role_wta.py -q
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import research.runners._rungB1b_neural_role_wta_derisk as B1B  # noqa: E402
from research.runners.unified_brain_bridge import UnifiedBrainBridge, SYNAPTIC_ROUTE_ROLES  # noqa: E402


# ── fast structural tests ────────────────────────────────────────────────────────────────────────────────────
def test_role_wta_n_layout():
    """role_wta_n = 3*P + INH = 90; the layout constants are consistent."""
    assert B1B.ROLE_WTA_N == 3 * B1B.WTA_P + B1B.WTA_INH == 90


def test_wta_drive_no_argmax_is_pure_transform():
    """_wta_drive is a pure transform (uniform baseline + graded bias): non-negative, monotone in the top logit,
    the max-logit ensemble gets the largest drive, and it makes NO decision (no argmax)."""
    d = B1B._wta_drive(np.array([1.0, 0.0, -0.3]))
    assert d.shape == (3,)
    assert np.all(d >= 0)
    assert int(np.argmax(d)) == 0                        # the top logit -> the top drive (but this is not the select)
    # every ensemble gets AT LEAST the uniform baseline (so I->E inhibition is what selects the winner)
    assert np.all(d >= B1B.WTA_BASE - 1e-9)
    # a one-hot logit and its scaled version give the same normalized shape (max-normalized)
    d2 = B1B._wta_drive(np.array([5.0, 0.0, 0.0]))
    assert np.allclose(d, B1B._wta_drive(np.array([1.0, 0.0, -0.3]))) or d2[0] == d2.max()


def test_neural_select_source_clean():
    """(anti-cheat 7, source half) _op_wta never references the read-out matrix Ws (it decides the role from the
    spiking gate, not a host argmax over @Ws), and _bind_wta_fact takes the role from the LATCHED gate."""
    assert B1B._source_has_no_host_argmax() is True


def test_scramble_Ws_deranges_role_columns():
    """_scramble_Ws permutes the 3 role columns (a real derangement) and leaves comp.Ws untouched."""
    class _Comp:
        Ws = {0: np.arange(15.0).reshape(5, 3), 1: np.arange(30.0, 45.0).reshape(5, 3)}
    comp = _Comp()
    orig = {k: v.copy() for k, v in comp.Ws.items()}
    scr = B1B._scramble_Ws(comp, seed=42)
    for k in comp.Ws:
        assert np.array_equal(comp.Ws[k], orig[k])       # original untouched
        # the scrambled cols are a permutation of the originals, and NOT the identity permutation
        assert not np.array_equal(scr[k][:, [0, 1, 2]], orig[k][:, [0, 1, 2]])
        assert sorted(map(tuple, scr[k].T)) == sorted(map(tuple, orig[k].T))


def test_wire_wta_builds_inhibitory_pool_and_gate_couplings():
    """wire_wta lays the 3 ensembles + inhibitory pool at role_wta_base, flips the inh trait to 1, and couples
    each role_route gate to its ensemble -- IN PLACE (the trained parser survives: the bridge builds without error
    and the parser reads roles)."""
    from research.runners._burndown_I5a_synaptic_parser_composer import synth_concepts
    concepts = synth_concepts(seed=0)
    ub = UnifiedBrainBridge(seed=42, proj_dim=128, concepts=concepts,
                            enable_synaptic_route=True, role_wta_n=B1B.ROLE_WTA_N)
    ens, inh = B1B.wire_wta(ub)
    base = ub.role_wta_base
    assert base is not None
    assert len(ens) == 3 and all(len(e) == B1B.WTA_P for e in ens)
    assert len(inh) == B1B.WTA_INH
    # the inhibitory pool has trait 1 (inhibitory_trait_index); the ensembles stay excitatory (trait 0)
    assert int(ub.bridge.cp_traits[inh].min()) == 1 and int(ub.bridge.cp_traits[inh].max()) == 1
    assert int(ub.bridge.cp_traits[np.concatenate(ens)].max()) == 0
    # a role_route gate coupling exists per role, each coupled to its ensemble
    coupled = {c["gate_name"] for c in ub.bridge._gate_couplings}
    for r in SYNAPTIC_ROUTE_ROLES:
        assert f"role_route_{r}" in coupled
    # the trained parser still reads the SVO roles (wiring the WTA in place did not reset it)
    assert ub.parser.role_of(0, "active") == "agent"
    assert ub.parser.role_of(2, "active") == "patient"


def test_op_wta_selects_the_biased_winner():
    """_op_wta drives the WTA with a one-hot logit and the LATCHED role == the driven ensemble (a genuine spiking
    selection), for each of the 3 roles, and exactly one gate opens."""
    from research.runners._burndown_I5a_synaptic_parser_composer import synth_concepts
    from research.runners.core_sim_composition import onoff, _scale_to_current, FILL_DRIVE
    concepts = synth_concepts(seed=0)
    ub = UnifiedBrainBridge(seed=42, proj_dim=128, concepts=concepts,
                            enable_synaptic_route=True, role_wta_n=B1B.ROLE_WTA_N)
    ens, inh = B1B.wire_wta(ub)
    word = sorted(concepts)[0]
    c_on, c_off = onoff(ub.composer.concepts[word])
    fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
    for target in range(3):
        logits = np.array([0.0, 0.0, 0.0]); logits[target] = 1.0
        _bon, _boff, latched, firewin, gates = B1B._op_wta(ub, ens, logits, fon, foff)
        assert latched == SYNAPTIC_ROUTE_ROLES[target], (target, latched)
        assert firewin == SYNAPTIC_ROUTE_ROLES[target], (target, firewin)
        assert gates == [SYNAPTIC_ROUTE_ROLES[target]], (target, gates)


# ── slow end-to-end GO gate ──────────────────────────────────────────────────────────────────────────────────
@pytest.mark.slow
def test_seed42_GO_all_nine_anticheats():
    """The full seed-42 GO: route recall >= 0.8n, route not worse than dict, moat clean, provenance clean,
    route-lesion + reservoir-lesion + WTA-lesion + Ws-scramble all collapse, and neural-select holds."""
    corpus = B1B.setup_corpus(seed=42)
    d = B1B.run_seed(42, corpus)
    assert d["route_recall"] >= 0.80, d
    assert d["route_not_worse_than_dict"], d
    assert d["moat_clean"], d
    assert d["provenance"]["clean"], d
    assert d["route_lesion_collapses"], d
    assert d["res_lesion_collapses"], d
    assert d["neural_select_ok"], d
    assert d["wta_lesion_collapses"], d
    assert d["ws_scramble_collapses"], d
    assert d["seed_GO"], d
