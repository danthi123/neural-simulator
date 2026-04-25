"""Unit tests for the brain-region framework (Session E.2).

See docs/plans/2026-04-24-brain-region-framework.md for the full plan.
The framework declares multiple brain regions (PFC, Motor, etc.) as
configured submodules on a common bridge substrate, with cross-region
pathways and neuromodulator-gated plasticity.

Default OFF: when CoreSimConfig.brain_regions is empty, the bridge runs
as a single population (today's behavior unchanged).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------- Task 1: dataclasses ----------

def test_brain_region_defaults():
    from sim.regions import BrainRegion

    r = BrainRegion(name="PFC", n_neurons=200)
    assert r.name == "PFC"
    assert r.n_neurons == 200
    assert r.exc_fraction == 0.8
    assert r.internal_density == 0.1
    assert r.exc_weight_mean == 0.3
    assert r.inh_weight_mean == 0.8
    assert r.weight_jitter == 0.2
    assert r.plastic_internal is False
    assert r.nm_outputs == []


def test_brain_region_custom():
    from sim.regions import BrainRegion

    r = BrainRegion(
        name="Hippocampus",
        n_neurons=500,
        exc_fraction=0.9,
        internal_density=0.2,
        plastic_internal=True,
        nm_outputs=["acetylcholine"],
    )
    assert r.exc_fraction == 0.9
    assert r.plastic_internal is True
    assert r.nm_outputs == ["acetylcholine"]


def test_region_pathway_defaults():
    from sim.regions import RegionPathway

    p = RegionPathway(from_region="PFC", to_region="Motor")
    assert p.from_region == "PFC"
    assert p.to_region == "Motor"
    assert p.density == 0.5
    assert p.weight_mean == 1.0
    assert p.weight_jitter == 0.2
    assert p.plastic is True
    assert p.neuromodulator_gates == []


def test_region_pathway_with_nm_gating():
    from sim.regions import RegionPathway

    p = RegionPathway(
        from_region="Cortex",
        to_region="Striatum",
        density=0.3,
        weight_mean=0.5,
        plastic=True,
        neuromodulator_gates=["dopamine"],
    )
    assert p.neuromodulator_gates == ["dopamine"]


# ---------- Task 2: RegionManager allocation ----------

def test_region_manager_allocates_contiguous_indices():
    from sim.regions import BrainRegion, RegionManager

    regions = [
        BrainRegion(name="PFC", n_neurons=100),
        BrainRegion(name="Motor", n_neurons=20),
    ]
    mgr = RegionManager(regions, [])
    mgr.initialize()
    assert mgr.total_neurons() == 120
    assert mgr.indices("PFC") == list(range(0, 100))
    assert mgr.indices("Motor") == list(range(100, 120))
    with pytest.raises(KeyError):
        mgr.indices("Hippocampus")


def test_region_manager_inhibitory_indices_match_exc_fraction():
    from sim.regions import BrainRegion, RegionManager

    regions = [BrainRegion(name="PFC", n_neurons=100, exc_fraction=0.8)]
    mgr = RegionManager(regions, [])
    mgr.initialize(seed=42)
    inh_indices = mgr.inhibitory_indices("PFC")
    # 20% inhibitory of 100 = 20
    assert len(inh_indices) == 20
    # All within PFC range
    for idx in inh_indices:
        assert 0 <= idx < 100


def test_region_manager_inhibitory_indices_seed_deterministic():
    from sim.regions import BrainRegion, RegionManager

    regions = [BrainRegion(name="PFC", n_neurons=100, exc_fraction=0.8)]

    mgr1 = RegionManager(regions, [])
    mgr1.initialize(seed=42)

    mgr2 = RegionManager(regions, [])
    mgr2.initialize(seed=42)

    assert mgr1.inhibitory_indices("PFC") == mgr2.inhibitory_indices("PFC")


def test_region_manager_indices_dict_for_neuromod_groups():
    """region_indices_dict() returns {name: [int]} for nm_mgr.set_group_indices."""
    from sim.regions import BrainRegion, RegionManager

    regions = [
        BrainRegion(name="PFC", n_neurons=10),
        BrainRegion(name="Motor", n_neurons=4),
    ]
    mgr = RegionManager(regions, [])
    mgr.initialize()
    d = mgr.region_indices_dict()
    assert d["PFC"] == list(range(0, 10))
    assert d["Motor"] == list(range(10, 14))


def test_region_manager_empty_lists_yield_zero_total():
    from sim.regions import RegionManager

    mgr = RegionManager([], [])
    mgr.initialize()
    assert mgr.total_neurons() == 0
    assert mgr.region_indices_dict() == {}


# ---------- Task 3: internal connectivity in wiring plan ----------

def test_internal_wiring_plan_for_single_region():
    from sim.regions import BrainRegion, RegionManager

    regions = [BrainRegion(name="PFC", n_neurons=50,
                            exc_fraction=0.8, internal_density=0.1)]
    mgr = RegionManager(regions, [])
    mgr.initialize(seed=42)
    plan = mgr.build_wiring_plan(seed=42)
    assert "PFC_internal" in plan
    g = plan["PFC_internal"]
    n_pairs = 50 * 49  # ordered, no self
    expected = int(n_pairs * 0.1)
    # Allow ±25% slack for stochasticity at small N
    assert int(0.7 * expected) < g["count"] < int(1.3 * expected)
    # All endpoints inside [0, 50)
    for pre, post in zip(g["pre_indices"], g["post_indices"]):
        assert 0 <= pre < 50
        assert 0 <= post < 50
        assert pre != post


def test_internal_wiring_plan_inhibitory_synapses_have_inh_weights():
    """Synapses originating from inhibitory neurons should use inh_weight_mean."""
    from sim.regions import BrainRegion, RegionManager

    # Make all weights deterministic (no jitter) so we can check exact values
    regions = [BrainRegion(
        name="PFC", n_neurons=20, exc_fraction=0.5,
        internal_density=1.0,  # fully connected for fast verification
        exc_weight_mean=0.3, inh_weight_mean=0.8, weight_jitter=0.0,
    )]
    mgr = RegionManager(regions, [])
    mgr.initialize(seed=42)
    plan = mgr.build_wiring_plan(seed=42)
    g = plan["PFC_internal"]
    inh = set(mgr.inhibitory_indices("PFC"))
    for pre, w in zip(g["pre_indices"], g["initial_weights"]):
        if int(pre) in inh:
            assert abs(float(w) - 0.8) < 1e-6, f"inh pre {pre} should have w=0.8, got {w}"
        else:
            assert abs(float(w) - 0.3) < 1e-6, f"exc pre {pre} should have w=0.3, got {w}"


def test_internal_wiring_plan_plastic_flag_passes_through():
    from sim.regions import BrainRegion, RegionManager

    regions = [BrainRegion(name="PFC", n_neurons=10,
                            internal_density=0.5, plastic_internal=True)]
    mgr = RegionManager(regions, [])
    mgr.initialize(seed=42)
    plan = mgr.build_wiring_plan(seed=42)
    assert plan["PFC_internal"]["plastic"] is True

    regions2 = [BrainRegion(name="Motor", n_neurons=10,
                             internal_density=0.5, plastic_internal=False)]
    mgr2 = RegionManager(regions2, [])
    mgr2.initialize(seed=42)
    plan2 = mgr2.build_wiring_plan(seed=42)
    assert plan2["Motor_internal"]["plastic"] is False


# ---------- Task 4: cross-region pathway in wiring plan ----------

def test_cross_region_pathway_in_wiring_plan():
    from sim.regions import BrainRegion, RegionPathway, RegionManager

    regions = [
        BrainRegion(name="PFC", n_neurons=100, internal_density=0.0),
        BrainRegion(name="Motor", n_neurons=20, internal_density=0.0),
    ]
    pathways = [RegionPathway(from_region="PFC", to_region="Motor",
                                density=0.5, weight_mean=1.0,
                                weight_jitter=0.0)]
    mgr = RegionManager(regions, pathways)
    mgr.initialize(seed=42)
    plan = mgr.build_wiring_plan(seed=42)
    assert "pathway_PFC_to_Motor" in plan
    g = plan["pathway_PFC_to_Motor"]
    n_pairs = 100 * 20  # 2000 ordered pairs (PFC -> Motor)
    expected = int(n_pairs * 0.5)
    assert int(0.85 * expected) < g["count"] < int(1.15 * expected)
    # Endpoints respect region boundaries
    for pre, post in zip(g["pre_indices"], g["post_indices"]):
        assert 0 <= pre < 100  # PFC range
        assert 100 <= post < 120  # Motor range
    # Plastic by default for cross-region
    assert g["plastic"] is True


def test_cross_region_pathway_unknown_region_raises():
    from sim.regions import BrainRegion, RegionPathway, RegionManager

    regions = [BrainRegion(name="PFC", n_neurons=10)]
    pathways = [RegionPathway(from_region="PFC", to_region="Hippocampus")]
    mgr = RegionManager(regions, pathways)
    mgr.initialize()
    with pytest.raises(KeyError):
        mgr.build_wiring_plan(seed=42)
