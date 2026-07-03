"""CPU tests for EMERGE-95 -- RUNG A.3: all three spiking components on ONE bridge (the one-brain substrate for the turn).

A light structural test (the shared bridge has disjoint reservoir/rf/slots slices, no cross pathways) + a slow
single-seed gate (the whole turn on the 3-spiking-slice bridge). CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import research.runners._emerge95_three_spiking_onebridge_turn_derisk as m95  # noqa: E402


def test_shared_bridge_has_three_disjoint_slices_no_pathways():
    b, rf_base = m95._build_shared_bridge(42, rf_size=2 * 6 * m95._D)
    res = np.asarray(b.region_manager.indices("reservoir"))
    rf = np.asarray(b.region_manager.indices("rf"))
    slots = np.asarray(b.region_manager.indices("slots"))
    for a in (res, rf, slots):
        assert len(a) > 0
    assert set(res.tolist()).isdisjoint(set(rf.tolist()))
    assert set(res.tolist()).isdisjoint(set(slots.tolist()))
    assert set(rf.tolist()).isdisjoint(set(slots.tolist()))       # three disjoint slices
    assert b.core_config.region_pathways == []                    # no cross-region synapses
    assert float(b.core_config.dt) == 1.0


def test_shared_bridge_reservoir_lsm_binds_the_reservoir_slice():
    from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder
    import research.runners._emerge62_discover_function_words_derisk as m62
    discovered, *_ = m62.discover_closed_class(*m62.compute_stats(m62.build_stream(42, n_sentences=2000))[:3])
    enc = Encoder(discovered)
    b, _rf = m95._build_shared_bridge(42, rf_size=2 * 6 * m95._D)
    lsm = m95.SharedBridgeReservoirLSM(enc.dim, 42, b)
    assert lsm.bridge is b                                          # the reservoir uses the SHARED bridge
    assert np.array_equal(lsm.res_idx, np.asarray(b.region_manager.indices("reservoir")))


@pytest.mark.slow
def test_seed42_three_spiking_onebridge_turn():
    d = m95._derisk_one(42)
    assert d["parse_acc"] >= 0.90              # reservoir (slice) comprehends
    assert d["recall"] >= 0.90                 # composer (slice) recalls
    assert d["render_exact"] >= 0.90           # producer (slice) speaks -- all three on ONE bridge
    assert d["moat_false_accept"] <= 0.05
    assert d["moat_producer_invoked_on_abstain"] == 0
    assert d["lesion_render_exact"] <= 0.30
    assert d["nolearn_render_exact"] <= 0.60
