"""CPU tests for EMERGE-93 -- RUNG A.2: the two spiking components (composer + producer) fold onto ONE bridge.

A light structural test (the shared bridge has disjoint rf/slots slices, no cross pathways) + a slow single-seed gate
(the whole turn on the shared-bridge composer + producer: parse/recall/render, gate-first moat, both lesions collapse).
CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import research.runners._emerge93_two_spiking_onebridge_turn_derisk as m93  # noqa: E402


def test_shared_bridge_has_disjoint_rf_and_slots_no_pathways():
    b, rf_base = m93._build_shared_bridge(42, rf_size=2 * 6 * m93._D)
    rf = np.asarray(b.region_manager.indices("rf"))
    slots = np.asarray(b.region_manager.indices("slots"))
    assert len(rf) > 0 and len(slots) > 0
    assert set(rf.tolist()).isdisjoint(set(slots.tolist()))      # disjoint slices
    assert b.core_config.region_pathways == []                   # no cross-region synapses
    assert rf_base == int(rf[0])


@pytest.mark.slow
def test_seed42_two_spiking_onebridge_turn():
    d = m93._derisk_one(42)
    assert d["parse_acc"] >= 0.90              # reservoir comprehends
    assert d["recall"] >= 0.90                 # composer (on the rf slice) recalls
    assert d["render_exact"] >= 0.90           # producer (on the slots slice) speaks -- both on ONE bridge
    assert d["moat_false_accept"] <= 0.05      # no-confab moat holds on the shared bridge
    assert d["moat_producer_invoked_on_abstain"] == 0    # gate-first
    assert d["lesion_render_exact"] <= 0.30    # comprehension is load-bearing
    assert d["nolearn_render_exact"] <= 0.60   # the learned spiking order is load-bearing
