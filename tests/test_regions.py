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
