"""Structural safety pins for Gate B v9 graded dendritic expectation."""

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
        "research.runners._vocal_action_credit_gate_v9_graded_dendritic"
    )
    module = importlib.reload(module)
    yield module
    _reset_cache_for_tests()


def test_only_reserved_seed_and_preregistered_centers_are_open(gate):
    assert gate.OPEN_PHASES == ()
    assert gate.TRACE_SIZE == 200
    assert gate.PLATEAU_CENTER_LADDER == (16.0, 8.0, 4.0, 2.0)
    assert gate.validate_smoke_seed(0) == 0
    with pytest.raises(ValueError):
        gate.validate_smoke_seed(1)
    with pytest.raises(ValueError):
        gate.v9_config(6.0)
    with pytest.raises(ValueError):
        gate.validate_phase("calibration")
    with pytest.raises(ValueError):
        gate.run_formal_seed(1)


def test_only_learned_expectation_route_is_new_dendritic_surface(gate):
    config = gate.v9_config(8.0)
    pathways = gate._v9_pathways(config)
    learned = [pathway for pathway in pathways if gate._is_expectation_route(pathway)]
    assert len(learned) == 2
    assert all(pathway.plastic for pathway in learned)
    assert all(pathway.density == 0.5 for pathway in learned)
    assert all(pathway.coincidence_detector for pathway in learned)
    assert all(pathway.weight_mean == 0.1 for pathway in learned)
    assert not any(hasattr(config, name) for name in ("n_fixed_trace",))


def test_structural_audit_confines_learning_dendrites_and_output(gate):
    bridge, handles = gate.build_v9_bridge(config=gate.v9_config(4.0))
    audit = gate.structural_audit(bridge, handles)
    assert audit["plastic_synapses"] == audit["declared_plastic_synapses"]
    assert audit["plastic_outside_declared_routes"] == 0
    assert audit["fixed_inside_declared_routes"] == 0
    assert audit["reward_eligibility_matches_declared_routes"]
    assert audit["actor_gate_matches_actor_routes"]
    assert audit["expectation_gate_matches_expectation_routes"]
    assert audit["coincidence"]["enabled_outside_intended_routes"] == 0
    assert audit["coincidence"]["disabled_inside_intended_routes"] == 0
    assert audit["expectation_output_gate_matches_routes"]
    assert audit["expectation_output_gate_value"] == 0.0


def test_plateau_lesion_changes_only_expectation_dendritic_mask(gate):
    bridge, handles = gate.build_v9_bridge(config=gate.v9_config(2.0))
    before_weights = np.asarray(gate.to_host(bridge.cp_connections.data)).copy()
    audit = gate.lesion_expectation_plateau(bridge, handles)
    after_weights = np.asarray(gate.to_host(bridge.cp_connections.data))
    assert audit["changed_only_expected_routes"]
    assert audit["expected_routes_disabled"]
    assert audit["other_dendritic_routes_unchanged"]
    assert audit["weights_unchanged"]
    assert np.array_equal(before_weights, after_weights)


def test_numpy_construction_smoke(gate):
    result = gate.run_construction_smoke(16.0)
    assert result["plateau_center"] == 16.0
    assert result["preconditions"]
    assert all(item["ok"] for item in result["preconditions"])
    assert result["status"] == "CONSTRUCTION_PASS", result["checks"]


def test_engagement_result_requires_learning_and_plateau(gate, monkeypatch):
    def condition(mode, config, *, seed):
        intact = mode == "intact"
        learns = mode != "expectation_learning_lesion"
        plateau = mode != "expectation_plateau_lesion"
        rows = []
        for _ in range(config.smoke_training_trials):
            rows.append({
                "winner": 0,
                "reward_delivered": True,
                "action": {
                    "graded_plateau_trace_integral": [1.0, 0.0],
                    "graded_plateau_expectation_integral": [1.0 if plateau else 0.0, 0.0],
                },
                "delay": {
                    "trace": [1, 0],
                    "expectation": [1 if intact else 0, 0],
                    "graded_plateau_trace_integral": [1.0, 0.0],
                    "graded_plateau_expectation_integral": [1.0 if plateau else 0.0, 0.0],
                },
            })
        return {
            "mode": mode,
            "plateau_center": config.action_tag_center,
            "baseline_probe": {"delay": {"expectation": [0, 0]}},
            "expectation_weight_before": [0.1, 0.1],
            "expectation_weight_after": [0.2, 0.1] if learns else [0.1, 0.1],
            "expectation_output_gate": 0.0,
            "plateau_lesion_audit": ({
                "changed_only_expected_routes": True,
                "expected_routes_disabled": True,
                "other_dendritic_routes_unchanged": True,
                "weights_unchanged": True,
            } if not plateau else None),
            "clean_trials": len(rows),
            "rewarded_trials": len(rows),
            "changed_synapses": 1 if learns else 0,
            "changed_outside_declared_routes": 0,
            "rows": rows,
        }

    monkeypatch.setattr(gate, "run_engagement_condition", condition)
    result = gate.run_engagement_smoke(8.0)
    assert result["plateau_center"] == 8.0
    assert all(item["ok"] for item in result["preconditions"])
    assert result["status"] == "ENGAGEMENT_PASS", result["checks"]


def test_output_result_requires_reward_and_omission_causality(gate, monkeypatch):
    def condition(mode, probe, *, seed):
        intact = mode == "output_intact"
        learned = mode != "expectation_learning_lesion"
        output_open = mode != "expectation_output_lesion"
        rows = []
        for index, action in enumerate((1, 0, 0, 1)):
            expectation = (
                ([0.24, 0.04] if index == 1 else [0.36, 0.06])
                if learned and action == 0
                else [0, 0.03] if learned
                else [0, 0]
            )
            if mode == "expectation_learning_lesion" and action == 0:
                expectation = [0.04, 0] if index == 1 else [0.08, 0]
            rows.append({
                "winner": action,
                "reward_delivered": probe == "reward" and action == 0,
                "delay": {"expectation": expectation},
                "gabab_snc_before_outcome_mean": (
                    (0.5 if index == 1 else 1.5)
                    if intact and action == 0 else 0.0
                ),
                "dopamine_burst_depth": (
                    (0.4 if index == 1 else 0.6)
                    if intact and action == 0
                    else (0.8 if index == 1 else 1.2)
                    if action == 0
                    else 99.0
                ),
                "dopamine_dip_depth": (
                    (0.4 if index == 1 else 0.6)
                    if intact and probe == "omission" and action == 0
                    else 99.0 if action == 1 else 0.0
                ),
                "outcome": {
                    "snc": (
                        (4 if index == 1 else 6)
                        if intact and action == 0
                        else (8 if index == 1 else 12)
                        if action == 0
                        else 999
                    ),
                    "lhb": (
                        (8 if index == 1 else 12)
                        if intact and probe == "omission" and action == 0
                        else 999 if action == 1 else 0
                    ),
                    "rmtg": (
                        (8 if index == 1 else 12)
                        if intact and probe == "omission" and action == 0
                        else 999 if action == 1 else 0
                    ),
                },
            })
        return {
            "mode": mode,
            "probe": probe,
            "plateau_center": 2.0,
            "expectation_output_gate_during_training": 0.0,
            "expectation_output_gate_during_probe": 1.0 if output_open else 0.0,
            "baseline_probe": {"delay": {"expectation": [0, 0]}},
            "clean_training_trials": 12,
            "changed_training_outside_declared_routes": 0,
            "probe_weights_unchanged": True,
            "probe_rows": rows,
        }

    monkeypatch.setattr(gate, "run_output_condition", condition)
    result = gate.run_output_smoke()
    assert all(item["ok"] for item in result["preconditions"])
    assert result["status"] == "OUTPUT_PASS", result["checks"]
    means = result["action_conditioned_means"]["reward"]["output_intact"]
    assert means["0"]["delay_expectation"] == pytest.approx([0.3, 0.05])
    assert means["0"]["dopamine_burst_depth"] == pytest.approx(0.5)
    assert means["0"]["outcome"]["snc"] == pytest.approx(5.0)
    assert result["reward_suppression_fraction"] == {
        "expectation_output_lesion": pytest.approx(0.5),
        "expectation_learning_lesion": pytest.approx(0.5),
    }


def test_output_is_undefined_when_block_samples_only_other_action(gate, monkeypatch):
    def condition(mode, probe, *, seed):
        row = {
            "winner": 1,
            "reward_delivered": False,
            "delay": {"expectation": [0, 0]},
            "gabab_snc_before_outcome_mean": 0.0,
            "dopamine_burst_depth": 0.0,
            "dopamine_dip_depth": 0.0,
            "outcome": {"snc": 0, "lhb": 0, "rmtg": 0},
        }
        return {
            "mode": mode,
            "probe": probe,
            "plateau_center": 2.0,
            "expectation_output_gate_during_training": 0.0,
            "expectation_output_gate_during_probe": (
                0.0 if mode == "expectation_output_lesion" else 1.0
            ),
            "baseline_probe": {"delay": {"expectation": [0, 0]}},
            "clean_training_trials": 12,
            "changed_training_outside_declared_routes": 0,
            "probe_weights_unchanged": True,
            "probe_rows": [dict(row) for _ in range(4)],
        }

    monkeypatch.setattr(gate, "run_output_condition", condition)
    result = gate.run_output_smoke()
    assert not all(item["ok"] for item in result["preconditions"])
    assert result["status"] == "UNDEFINED"


def test_output_is_undefined_when_lesions_change_action_sequence(gate, monkeypatch):
    def condition(mode, probe, *, seed):
        sequence = (
            (1, 0, 0, 1)
            if mode != "expectation_output_lesion"
            else (0, 1, 0, 1)
        )
        rows = [{
            "winner": action,
            "reward_delivered": probe == "reward" and action == 0,
            "delay": {"expectation": [10, 0]},
            "gabab_snc_before_outcome_mean": 0.0,
            "dopamine_burst_depth": 0.0,
            "dopamine_dip_depth": 0.0,
            "outcome": {"snc": 0, "lhb": 0, "rmtg": 0},
        } for action in sequence]
        return {
            "mode": mode,
            "probe": probe,
            "plateau_center": 2.0,
            "expectation_output_gate_during_training": 0.0,
            "expectation_output_gate_during_probe": (
                0.0 if mode == "expectation_output_lesion" else 1.0
            ),
            "baseline_probe": {"delay": {"expectation": [0, 0]}},
            "clean_training_trials": 12,
            "changed_training_outside_declared_routes": 0,
            "probe_weights_unchanged": True,
            "probe_rows": rows,
        }

    monkeypatch.setattr(gate, "run_output_condition", condition)
    result = gate.run_output_smoke()
    assert not all(item["ok"] for item in result["preconditions"])
    assert result["status"] == "UNDEFINED"


def test_output_protocol_failure_precedes_undefined_sampling(gate, monkeypatch):
    def condition(mode, probe, *, seed):
        row = {
            "winner": 1,
            "reward_delivered": True,
            "delay": {"expectation": [0, 0]},
            "gabab_snc_before_outcome_mean": 0.0,
            "dopamine_burst_depth": 0.0,
            "dopamine_dip_depth": 0.0,
            "outcome": {"snc": 0, "lhb": 0, "rmtg": 0},
        }
        return {
            "mode": mode,
            "probe": probe,
            "plateau_center": 2.0,
            "expectation_output_gate_during_training": (
                1.0 if mode == "output_intact" else 0.0
            ),
            "expectation_output_gate_during_probe": (
                0.0 if mode == "expectation_output_lesion" else 1.0
            ),
            "baseline_probe": {"delay": {"expectation": [0, 0]}},
            "clean_training_trials": 12,
            "changed_training_outside_declared_routes": 0,
            "probe_weights_unchanged": True,
            "probe_rows": [dict(row) for _ in range(3)],
        }

    monkeypatch.setattr(gate, "run_output_condition", condition)
    result = gate.run_output_smoke()
    assert not result["checks"]["all_conditions_train_with_output_closed"]
    assert not result["checks"]["fixed_four_probe_blocks"]
    assert not result["checks"][
        "reward_probe_is_delivered_and_omission_is_withheld"
    ]
    assert not all(item["ok"] for item in result["preconditions"])
    assert result["status"] == "OUTPUT_FAIL"


def test_fractional_learning_lesion_threshold_has_no_one_spike_floor(
    gate, monkeypatch
):
    def condition(mode, probe, *, seed):
        rows = []
        for action in (1, 0, 0, 1):
            expectation = (
                [0.1, 0.0]
                if mode == "expectation_learning_lesion" and action == 0
                else [0.3, 0.05]
                if action == 0
                else [0.0, 0.03]
            )
            intact_action = mode == "output_intact" and action == 0
            rows.append({
                "winner": action,
                "reward_delivered": probe == "reward" and action == 0,
                "delay": {"expectation": expectation},
                "gabab_snc_before_outcome_mean": (
                    1.0 if intact_action else 0.0
                ),
                "dopamine_burst_depth": 0.5 if intact_action else 1.0,
                "dopamine_dip_depth": (
                    0.5 if intact_action and probe == "omission" else 0.0
                ),
                "outcome": {
                    "snc": 5 if intact_action else 10,
                    "lhb": 10 if intact_action and probe == "omission" else 0,
                    "rmtg": 10 if intact_action and probe == "omission" else 0,
                },
            })
        return {
            "mode": mode,
            "probe": probe,
            "plateau_center": 2.0,
            "expectation_output_gate_during_training": 0.0,
            "expectation_output_gate_during_probe": (
                0.0 if mode == "expectation_output_lesion" else 1.0
            ),
            "baseline_probe": {"delay": {"expectation": [0, 0]}},
            "clean_training_trials": 12,
            "changed_training_outside_declared_routes": 0,
            "probe_weights_unchanged": True,
            "probe_rows": rows,
        }

    monkeypatch.setattr(gate, "run_output_condition", condition)
    result = gate.run_output_smoke()
    assert all(item["ok"] for item in result["preconditions"])
    assert not result["checks"][
        "learning_lesion_removes_80pct_of_probe_expectation"
    ]
    assert result["status"] == "OUTPUT_FAIL"


def test_cli_rejects_nonreserved_seed(gate):
    with pytest.raises(ValueError):
        gate.main(["--engagement", "--seed", "1"])
