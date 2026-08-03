"""Structural safety pins for Gate B v5 learning."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    from sim.backend import _reset_cache_for_tests

    _reset_cache_for_tests()
    name = "research.runners._vocal_action_credit_gate_v5_learning"
    module = importlib.import_module(name)
    module = importlib.reload(module)
    yield module
    _reset_cache_for_tests()


def test_formal_seeds_and_phases_are_assigned_but_sealed(gate):
    assert gate.OPEN_PHASES == ()
    assert gate.validate_smoke_seed(0) == 0
    assert tuple(gate._SEED_PARTITIONS) == (
        "calibration", "development", "held_out"
    )
    assert sum(map(len, gate._SEED_PARTITIONS.values())) == 8
    with pytest.raises(ValueError):
        gate.validate_smoke_seed(gate._SEED_PARTITIONS["calibration"][0])
    with pytest.raises(ValueError):
        gate.validate_phase("calibration")
    with pytest.raises(ValueError):
        gate.validate_formal_seeds(gate._SEED_PARTITIONS["calibration"])
    with pytest.raises(ValueError):
        gate.run_formal_seed(gate._SEED_PARTITIONS["calibration"][0])


def test_trace_and_expectation_are_distinct_populations(gate):
    regions = gate._learning_regions(gate.learning_config())
    names = {region.name for region in regions}
    for channel in gate.CHANNELS:
        assert gate._trace(channel) in names
        assert gate._expectation(channel) in names
        assert gate._trace(channel) != gate._expectation(channel)


def test_pathway_contract_removes_trace_critic_shortcut(gate):
    pathways = gate._learning_pathways(gate.learning_config())
    for channel in gate.CHANNELS:
        trace = gate._trace(channel)
        expectation = gate._expectation(channel)

        trace_loaders = [
            p for p in pathways
            if p.to_region == trace
            and p.from_region in {f"commit_{channel}", "practice_arousal"}
        ]
        assert len(trace_loaders) == 2
        assert all(p.coincidence_detector and not p.plastic for p in trace_loaders)

        learned = [
            p for p in pathways
            if p.from_region == trace and p.to_region == expectation
        ]
        assert len(learned) == 1
        assert learned[0].plastic
        assert learned[0].plasticity_gate == gate.EXPECTATION_PLASTICITY_GATE

        assert not any(
            p.from_region == trace
            and p.to_region in {gate.SNC, gate.v3.OMISSION_GATE}
            for p in pathways
        )
        outputs = [
            p for p in pathways
            if p.from_region == expectation
            and p.to_region in {gate.SNC, gate.v3.OMISSION_GATE}
        ]
        assert len(outputs) == 2
        assert all(not p.plastic for p in outputs)
        assert all(p.transmission_gate == gate.EXPECTATION_OUTPUT_GATE for p in outputs)


def test_outcome_afferents_are_symmetric_and_host_boundary_is_fixed(gate):
    pathways = gate._learning_pathways(gate.learning_config())
    routes = [
        p for p in pathways
        if p.from_region == gate.OUTCOME_ONSET
        and p.to_region.startswith(gate.EXPECTATION_PREFIX)
    ]
    assert len(routes) == 2
    assert {p.to_region for p in routes} == {
        gate._expectation(channel) for channel in gate.CHANNELS
    }
    assert len({p.weight_mean for p in routes}) == 1
    assert all(not p.plastic for p in routes)
    assert not gate.HOST_BOUNDARY["host_action_winner_latch"]
    assert not gate.HOST_BOUNDARY["host_action_timed_transmission_window"]
    assert not gate.HOST_BOUNDARY["host_copies_action_to_outcome"]


def test_bridge_masks_confine_plasticity_and_coincidence(gate):
    bridge, handles = gate.build_learning_bridge()
    audit = gate._structural_audit(bridge, handles)
    assert audit["plastic_synapses"] == audit["declared_plastic_synapses"]
    assert audit["plastic_outside_declared_routes"] == 0
    assert audit["fixed_inside_declared_routes"] == 0
    assert audit["reward_eligibility_matches_declared_routes"]
    assert audit["actor_gate_matches_actor_routes"]
    assert audit["expectation_gate_matches_expectation_routes"]
    assert audit["actor_gate_gain"] == gate.learning_config().actor_plasticity_gain
    assert audit["expectation_gate_gain"] == 1.0
    coincidence = audit["coincidence"]
    assert coincidence["enabled_synapses"] == coincidence["intended_synapses"]
    assert coincidence["enabled_outside_intended_routes"] == 0
    assert coincidence["disabled_inside_intended_routes"] == 0

    mask = np.asarray(gate.to_host(bridge.cp_synapse_plastic_mask), dtype=bool)
    declared = np.zeros(mask.shape, dtype=bool)
    declared[handles["routes"].all_indices()] = True
    assert np.array_equal(mask, declared)


def test_channel_permutation_moves_fixed_trace_loader_only(gate):
    bridge, handles = gate.build_learning_bridge(action_permutation=(1, 0))
    assert handles["action_permutation"] == (1, 0)
    audit = gate._structural_audit(bridge, handles)
    assert audit["plastic_outside_declared_routes"] == 0
    assert audit["reward_eligibility_matches_declared_routes"]
    coincidence = audit["coincidence"]
    assert coincidence["enabled_outside_intended_routes"] == 0
    assert coincidence["disabled_inside_intended_routes"] == 0
    assert coincidence["route_counts"] == {
        "commit_0->vocal_credit_value_1": 720,
        "commit_1->vocal_credit_value_0": 720,
        "practice_arousal->vocal_credit_value_0": 576,
        "practice_arousal->vocal_credit_value_1": 576,
    }

    pathways = gate._learning_pathways(
        gate.learning_config(), action_permutation=(1, 0)
    )
    outcome_targets = {
        pathway.to_region
        for pathway in pathways
        if pathway.from_region == gate.OUTCOME_ONSET
        and pathway.to_region.startswith(gate.EXPECTATION_PREFIX)
    }
    assert outcome_targets == {
        gate._expectation(channel) for channel in gate.CHANNELS
    }


def test_numpy_construction_smoke(gate):
    result = gate.run_construction_smoke()
    assert result["science_seed_executed"] is False
    assert result["status"] == "CONSTRUCTION_PASS", result["checks"]
