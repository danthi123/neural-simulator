"""CPU tests for EMERGE-92 -- RUNG A.1: the spiking producer runs as a disjoint SLICE on a shared bridge.

A light structural test (the shared bridge has disjoint slots/coresident slices, no cross pathways; a co-resident
producer builds) + a slow single-seed gate (co-resident render == private render, exact, co-resident region active).
CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import research.runners._emerge92_producer_coresident_derisk as m92  # noqa: E402


def test_shared_bridge_has_disjoint_slots_and_coresident_no_pathways():
    b, cidx = m92._build_shared_bridge(42)
    slots = np.asarray(b.region_manager.indices("slots"))
    assert len(slots) > 0 and len(cidx) > 0
    assert set(slots.tolist()).isdisjoint(set(cidx.tolist()))    # disjoint slices
    assert b.core_config.region_pathways == []                   # no cross-region synapses


def test_shared_bridge_hosts_a_coresident_producer():
    from research.runners._emerge72_construction_registry_derisk import RegistryProducer, RegistryBrocaProducer
    from research.runners._emerge74_transitive_ditransitive_derisk import build_stream_svo, SVOConstructionRegistry
    reg = SVOConstructionRegistry(42).build(build_stream_svo(42))
    shared, _c = m92._build_shared_bridge(42)
    cq = RegistryProducer(seed=42, registry_slots=reg.registered_fits(), shared_bridge=shared, slot_region="slots")
    cq.learn()
    assert cq.bridge is shared                                   # the producer uses the SHARED bridge, not a private one
    out = RegistryBrocaProducer(cq).speak(
        __import__("research.runners._emerge72_construction_registry_derisk", fromlist=["decision"]).decision(
            "ANSWER", construction="C_TRANS", subject="dog", verb="chase", obj="ball"))
    assert out["surface"] == "the dog chases the ball"           # renders on the shared-bridge slice


@pytest.mark.slow
def test_seed42_producer_coresident():
    d = m92._derisk_one(42)
    assert d["render_match_coresident_vs_private"] >= 0.999      # co-resident render == private render
    assert d["render_exact_coresident"] >= 0.999                 # renders the ground-truth transitive
    assert d["render_exact_private"] >= 0.999
    assert d["coresident_region_rate"] > 0.01                    # the co-resident region is genuinely active
