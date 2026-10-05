"""STEP4B step 2 (docs/drafts/long-delay-credit/STEP4B.md): the omission build's route-off identity.

With `omission_route=0`, research/runners/_ldc_credit_omission.py must build byte-identically to the original
bg_corticostriatal_credit.py (same region list, same cp_connections) at seed 7 -- the LHb-RMTg route must be a pure
addition. With `omission_route=1` exactly `lhb` / `lhb_inh` / `rmtg` are added with the five STEP4B pathways, and
`cut_rus_lhbinh=1` drops only `reward_us -> lhb_inh`.
"""
import numpy as np
import pytest


def _regions(bridge):
    return [r.name for r in bridge.region_manager.regions()]


def _conn(bridge):
    from sim.backend import to_host
    cp = bridge.cp_connections
    return (np.asarray(to_host(cp.indptr)), np.asarray(to_host(cp.indices)),
            np.asarray(to_host(cp.data)), int(cp.nnz))


@pytest.fixture(scope="module")
def builds():
    from research.runners.bg_corticostriatal_credit import CreditParams as OrigP, build_credit_selector as orig_build
    from research.runners._ldc_credit_omission import CreditParams as OmP, build_credit_selector as om_build
    return (orig_build(7, OrigP())[0], om_build(7, OmP())[0], om_build(7, OmP(omission_route=1))[0],
            om_build(7, OmP(omission_route=1, cut_rus_lhbinh=1))[0])


def test_route0_build_byte_identical_at_seed7(builds):
    b_orig, b_route0 = builds[0], builds[1]
    assert _regions(b_route0) == _regions(b_orig)
    i0, x0, d0, n0 = _conn(b_orig)
    i1, x1, d1, n1 = _conn(b_route0)
    assert n1 == n0, "nnz changed with the route off"
    assert np.array_equal(i1, i0), "cp_connections.indptr changed with the route off"
    assert np.array_equal(x1, x0), "cp_connections.indices changed with the route off"
    assert np.array_equal(d1, d0), "cp_connections.data changed with the route off"


def test_route1_adds_exactly_three_regions_and_the_five_pathways(builds):
    b_orig, b_route1 = builds[0], builds[2]
    assert set(_regions(b_route1)) - set(_regions(b_orig)) == {"lhb", "lhb_inh", "rmtg"}
    from research.runners.bg_corticostriatal_credit import Synapses
    syn = Synapses(b_route1)
    for a, b in (("ctx_o", "lhb"), ("lhb_inh", "lhb"), ("lhb", "rmtg"), ("rmtg", "snc")):
        assert syn.mask(a, b).size > 0, "missing pathway %s -> %s" % (a, b)
    assert syn.mask("reward_us", "lhb_inh").size > 0, "missing pathway reward_us -> lhb_inh"


def test_cut_drops_only_reward_us_to_lhb_inh(builds):
    b_route1, b_cut = builds[2], builds[3]
    from research.runners.bg_corticostriatal_credit import Synapses
    s_on, s_cut = Synapses(b_route1), Synapses(b_cut)
    assert s_cut.mask("reward_us", "lhb_inh").size == 0, "cut_rus_lhbinh must drop reward_us -> lhb_inh"
    for a, b in (("ctx_o", "lhb"), ("lhb_inh", "lhb"), ("lhb", "rmtg"), ("rmtg", "snc")):
        assert s_cut.mask(a, b).size > 0, "cut must keep %s -> %s" % (a, b)
