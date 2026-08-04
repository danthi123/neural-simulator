"""Structural safety pins for Gate B v8 dual-afferent expectation."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    from sim.backend import _reset_cache_for_tests

    _reset_cache_for_tests()
    module = importlib.import_module(
        "research.runners._vocal_action_credit_gate_v8_dual_afferent"
    )
    module = importlib.reload(module)
    yield module
    _reset_cache_for_tests()


def test_only_reserved_seed_and_preregistered_weights_are_open(gate):
    assert gate.OPEN_PHASES == ()
    assert gate.TRACE_SIZE == 200
    assert gate.FIXED_WEIGHT_LADDER == (2.0, 4.0, 8.0, 12.0, 16.0)
    assert gate.validate_smoke_seed(0) == 0
    with pytest.raises(ValueError):
        gate.validate_smoke_seed(1)
    with pytest.raises(ValueError):
        gate.v8_config(6.0)
    with pytest.raises(ValueError):
        gate.validate_phase("calibration")
    with pytest.raises(ValueError):
        gate.run_formal_seed(1)


def test_fixed_and_plastic_afferents_are_distinct(gate):
    config = gate.v8_config(8.0)
    pathways = gate._v8_pathways(config)
    learned = [
        pathway for pathway in pathways
        if pathway.from_region in {
            gate.v5l._trace(channel) for channel in gate.CHANNELS
        }
        and pathway.to_region in {
            gate.v5l._expectation(channel) for channel in gate.CHANNELS
        }
    ]
    fixed = [
        pathway for pathway in pathways
        if pathway.from_region in {
            gate._fixed_trace(channel) for channel in gate.CHANNELS
        }
        and pathway.to_region in {
            gate.v5l._expectation(channel) for channel in gate.CHANNELS
        }
    ]
    assert len(learned) == len(fixed) == 2
    assert all(pathway.plastic and pathway.density == 0.5 for pathway in learned)
    assert all(not pathway.plastic for pathway in fixed)
    assert all(pathway.density == 0.8 for pathway in fixed)
    assert all(pathway.weight_mean == 8.0 for pathway in fixed)
    assert all(pathway.transmission_gate == gate.FIXED_OUTPUT_GATE for pathway in fixed)


def test_both_trace_populations_use_neural_coincidence_loading(gate):
    config = gate.v8_config()
    pathways = gate._v8_pathways(config)
    for channel in gate.CHANNELS:
        for target in (gate.v5l._trace(channel), gate._fixed_trace(channel)):
            commit = [
                pathway for pathway in pathways
                if pathway.from_region == f"commit_{channel}"
                and pathway.to_region == target
            ]
            arousal = [
                pathway for pathway in pathways
                if pathway.from_region == "practice_arousal"
                and pathway.to_region == target
            ]
            assert len(commit) == len(arousal) == 1
            assert commit[0].coincidence_detector
            assert arousal[0].coincidence_detector
            assert not commit[0].plastic
            assert not arousal[0].plastic


def test_structural_audit_confines_learning_and_gates(gate):
    bridge, handles = gate.build_v8_bridge(config=gate.v8_config(4.0))
    audit = gate.structural_audit(bridge, handles)
    assert audit["plastic_synapses"] == audit["declared_plastic_synapses"]
    assert audit["plastic_outside_declared_routes"] == 0
    assert audit["fixed_inside_declared_routes"] == 0
    assert audit["fixed_upstate_routes_are_nonplastic"]
    assert audit["reward_eligibility_matches_declared_routes"]
    assert audit["actor_gate_matches_actor_routes"]
    assert audit["expectation_gate_matches_expectation_routes"]
    assert audit["fixed_output_gate_matches_fixed_routes"]
    assert audit["coincidence"]["enabled_outside_intended_routes"] == 0
    assert audit["coincidence"]["disabled_inside_intended_routes"] == 0


def test_fixed_arm_lesion_preserves_topology_and_weights(gate):
    bridge, handles = gate.build_v8_bridge(config=gate.v8_config(12.0))
    before = np.asarray(gate.to_host(bridge.cp_connections.data)).copy()
    indices = np.sort(np.concatenate(list(handles["fixed_routes"].values())))
    gate_indices = np.sort(np.asarray(gate.to_host(
        bridge._transmission_gate_indices_gpu[gate.FIXED_OUTPUT_GATE]
    )))
    bridge.set_transmission_gate(gate.FIXED_OUTPUT_GATE, 0.0)
    after = np.asarray(gate.to_host(bridge.cp_connections.data))
    assert np.array_equal(indices, gate_indices)
    assert np.array_equal(before, after)
    assert bridge._transmission_gate_values[gate.FIXED_OUTPUT_GATE] == 0.0


def test_numpy_construction_smoke(gate):
    result = gate.run_construction_smoke(16.0)
    assert result["fixed_weight"] == 16.0
    assert result["preconditions"]
    assert all(item["ok"] for item in result["preconditions"])
    assert result["status"] == "CONSTRUCTION_PASS", result["checks"]


def test_engagement_result_requires_both_afferents(gate, monkeypatch):
    def condition(mode, config, *, seed):
        intact = mode == "intact"
        rows = [
            {
                "winner": 0,
                "reward_delivered": True,
                "delay": {
                    "trace": [1, 0],
                    "fixed_trace": [1, 0],
                    "expectation": [1 if intact else 0, 0],
                },
            }
            for _ in range(config.smoke_training_trials)
        ]
        learns = mode != "expectation_learning_lesion"
        return {
            "mode": mode,
            "fixed_weight": config.fixed_trace_to_expectation_weight,
            "baseline_probe": {"delay": {"expectation": [0, 0]}},
            "expectation_weight_before": [0.1, 0.1],
            "expectation_weight_after": [0.2, 0.1] if learns else [0.1, 0.1],
            "fixed_route_weight": [config.fixed_trace_to_expectation_weight] * 2,
            "fixed_output_gate": 0.0 if mode == "fixed_arm_lesion" else 1.0,
            "clean_trials": len(rows),
            "rewarded_trials": len(rows),
            "changed_synapses": 1 if learns else 0,
            "changed_outside_declared_routes": 0,
            "rows": rows,
        }

    monkeypatch.setattr(gate, "run_engagement_condition", condition)
    result = gate.run_engagement_smoke(4.0)
    assert result["fixed_weight"] == 4.0
    assert all(item["ok"] for item in result["preconditions"])
    assert result["status"] == "ENGAGEMENT_PASS", result["checks"]


def test_cli_rejects_nonreserved_seed(gate):
    with pytest.raises(ValueError):
        gate.main(["--engagement", "--seed", "1"])
