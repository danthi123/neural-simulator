"""Structural safety pins for Gate B v7 dense convergence."""

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
        "research.runners._vocal_action_credit_gate_v7_dense_convergence"
    )
    module = importlib.reload(module)
    yield module
    _reset_cache_for_tests()


def test_only_reserved_seed_and_preregistered_trace_sizes_are_open(gate):
    assert gate.OPEN_PHASES == ()
    assert gate.TRACE_SIZE_LADDER == (24, 64, 128, 200)
    assert gate.validate_smoke_seed(0) == 0
    with pytest.raises(ValueError):
        gate.validate_smoke_seed(1)
    with pytest.raises(ValueError):
        gate.v7_config(201)
    with pytest.raises(ValueError):
        gate.validate_phase("calibration")
    with pytest.raises(ValueError):
        gate.run_formal_seed(1)


def test_dense_route_is_only_new_learning_surface_and_output_is_gabab(gate):
    config = gate.v7_config(128)
    pathways = gate._v7_pathways(config)
    learned = [
        pathway for pathway in pathways
        if pathway.from_region in {
            gate.v5l._trace(channel) for channel in gate.CHANNELS
        }
        and pathway.to_region in {
            gate.v5l._expectation(channel) for channel in gate.CHANNELS
        }
    ]
    assert len(learned) == 2
    assert all(pathway.plastic for pathway in learned)
    assert all(pathway.density == 0.5 for pathway in learned)
    outputs = [
        pathway for pathway in pathways
        if pathway.from_region in {
            gate.v5l._expectation(channel) for channel in gate.CHANNELS
        }
        and pathway.to_region == gate.v5l.SNC
    ]
    assert len(outputs) == 2
    assert all(pathway.receptor == "gaba_b" for pathway in outputs)
    assert all(not pathway.plastic for pathway in outputs)


def test_bridge_audit_confines_plasticity_and_coincidence(gate):
    config = gate.v7_config(64)
    bridge, handles = gate.build_v7_bridge(config=config)
    audit = gate.v5l._structural_audit(bridge, handles)
    assert audit["plastic_synapses"] == audit["declared_plastic_synapses"]
    assert audit["plastic_outside_declared_routes"] == 0
    assert audit["fixed_inside_declared_routes"] == 0
    assert audit["reward_eligibility_matches_declared_routes"]
    assert audit["actor_gate_matches_actor_routes"]
    assert audit["expectation_gate_matches_expectation_routes"]
    assert audit["coincidence"]["enabled_outside_intended_routes"] == 0
    assert audit["coincidence"]["disabled_inside_intended_routes"] == 0


def test_measured_trial_reset_clears_girk_and_target_state(gate):
    config = gate.v7_config()
    bridge, _ = gate.build_v7_bridge(config=config)
    bridge.cp_conductance_g_gabab[:] = 1.0
    target = np.concatenate([
        np.asarray(gate._indices(bridge, name), dtype=np.int64)
        for name in (
            gate.v5l.SNC,
            gate.v3.OMISSION_GATE,
            gate.v5l._expectation(0),
            gate.v5l._expectation(1),
        )
    ])
    bridge.cp_membrane_potential_v[target] = 0.0
    bridge.cp_recovery_variable_u[target] = 3.0
    gate._reset_measured_trial(bridge, config)
    assert np.allclose(gate.to_host(bridge.cp_conductance_g_gabab), 0.0)
    assert np.allclose(
        gate.to_host(bridge.cp_membrane_potential_v[target]),
        gate.to_host(bridge.cp_izh_vr[target]),
    )
    assert np.allclose(
        gate.to_host(bridge.cp_recovery_variable_u[target]), 0.0
    )


def test_numpy_construction_smoke(gate):
    result = gate.run_construction_smoke(64)
    assert result["science_seed_executed"] is False
    assert result["status"] == "CONSTRUCTION_PASS", result["checks"]


def test_cli_rejects_nonreserved_seed(gate):
    with pytest.raises(ValueError):
        gate.main(["--engagement", "--seed", "1"])
