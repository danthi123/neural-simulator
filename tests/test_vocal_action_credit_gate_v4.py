"""Focused guards for the sealed Gate B v4 seed-zero successor."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import _vocal_action_credit_gate_v4 as gate  # noqa: E402


@pytest.fixture(scope="module")
def smoke_result():
    return gate.run_smoke()


def test_seed_zero_is_the_only_open_execution_partition():
    formal = (
        set(gate.CALIBRATION_SEEDS)
        | set(gate.DEVELOPMENT_SEEDS)
        | set(gate.HELD_OUT_SEEDS)
    )
    inherited = (
        set(gate.v3.CALIBRATION_SEEDS)
        | set(gate.v3.DEVELOPMENT_SEEDS)
        | set(gate.v3.HELD_OUT_SEEDS)
    )
    assert gate.OPEN_PHASES == ()
    assert gate.SMOKE_SEED == 0
    assert gate.SMOKE_SEED not in formal
    assert not (formal & inherited)
    assert len(formal) == 8
    assert gate.validate_smoke_seed(0) == 0
    for seed in formal:
        with pytest.raises(ValueError, match="non-scientific seed"):
            gate.validate_smoke_seed(seed)


def test_formal_entry_points_fail_before_build(monkeypatch):
    monkeypatch.setattr(
        gate,
        "build_v4_bridge",
        lambda *_args, **_kwargs: pytest.fail("formal seed reached brain build"),
    )
    for phase in ("calibration", "development", "held_out"):
        with pytest.raises(ValueError, match="formal phases are sealed"):
            gate.validate_phase(phase)
    with pytest.raises(ValueError, match="formal seeds are sealed"):
        gate.run_formal_seed(gate.CALIBRATION_SEEDS[0])
    with pytest.raises(ValueError, match="formal seeds are sealed"):
        gate.validate_formal_seeds(gate.CALIBRATION_SEEDS)


def test_cli_requires_explicit_smoke_and_rejects_seed_arguments():
    assert gate.resolve_cli_request(smoke=True, phase=None, seeds=None) == "smoke"
    with pytest.raises(ValueError, match="does not accept"):
        gate.resolve_cli_request(
            smoke=True, phase=None, seeds=gate.CALIBRATION_SEEDS
        )
    with pytest.raises(ValueError, match="formal phases are sealed"):
        gate.resolve_cli_request(smoke=False, phase=None, seeds=None)


def test_configuration_is_exactly_locked_before_bridge_construction(monkeypatch):
    changed = replace(gate.v4_config(), outcome_to_value_weight=13.1)
    monkeypatch.setattr(
        gate,
        "build_selector_bridge",
        lambda *_args, **_kwargs: pytest.fail("bad config reached bridge build"),
    )
    with pytest.raises(ValueError, match="exact preregistered configuration"):
        gate.build_v4_bridge(gate.SMOKE_SEED, changed)


def test_v4_preserves_every_inherited_v3_setting_and_preregisters_controls():
    v3_config = asdict(gate.v3.v3_config())
    v4_config = asdict(gate.v4_config())
    assert all(v4_config[name] == value for name, value in v3_config.items())
    assert set(gate.v3.CONTROL_MODES) <= {
        "yoked" if name == "reward_count_matched_shifted_yoked" else name
        for name in gate.FORMAL_CONTROLS
    } | set(gate.v3.CONTROL_MODES)
    assert {
        "frozen_expectation_route",
        "plateau_lesion",
        "shared_outcome_read_lesion",
        "action_channel_permutation",
    } <= set(gate.FORMAL_CONTROLS)
    assert gate.HOST_BOUNDARY["host_expected_value_state"] is False
    assert gate.HOST_BOUNDARY["host_action_winner_latch"] is True
    assert gate.HOST_BOUNDARY["host_action_timed_transmission_window"] is True
    assert gate.RETIRED is True


def test_intact_build_preserves_v3_circuit_and_scopes_plateau_to_value_routes():
    config = gate.v4_config()
    bridge, handles = gate.build_v4_bridge(gate.SMOKE_SEED, config)
    structural = gate.structural_preconditions(bridge, handles, config)
    route_state = gate._route_telemetry(bridge, handles)

    assert all(structural.values()), structural
    assert bridge.core_config.enable_gabab is True
    assert bridge.core_config.gabab_propagation_strength == (
        gate.v3.v3_config().gabab_propagation_strength
    )
    assert bridge.core_config.enable_graded_dendritic_plateau is True
    assert bridge.core_config.coincidence_plateau_strength == 0.0
    assert route_state["enabled"] == route_state["total"] == 1440
    assert route_state[
        "all_enabled_coincidence_synapses_belong_to_expectation_route"
    ]
    assert bridge._transmission_gate_values[gate.v3.VALUE_TO_SNC_GATE] == 1.0
    assert bridge._transmission_gate_values[gate.v3.REWARD_VETO_GATE] == 1.0
    assert bridge._transmission_gate_values[gate.v3.OMISSION_PATH_GATE] == 1.0


def test_plateau_lesion_changes_only_the_dendritic_routing_mask():
    bridge, handles = gate.build_v4_bridge(gate.SMOKE_SEED, gate.v4_config())
    route_indices = np.concatenate(list(handles["expectation_routes"].values()))
    weights_before = np.asarray(
        gate.to_host(bridge.cp_connections.data[route_indices])
    ).copy()
    transmission_before = np.asarray(
        gate.to_host(bridge.cp_transmission_gain[route_indices])
    ).copy()

    gate._set_expectation_plateau_route(bridge, handles, False)
    route_state = gate._route_telemetry(bridge, handles)

    assert route_state["enabled"] == 0
    assert route_state[
        "all_enabled_coincidence_synapses_belong_to_expectation_route"
    ]
    np.testing.assert_array_equal(
        np.asarray(gate.to_host(bridge.cp_connections.data[route_indices])),
        weights_before,
    )
    np.testing.assert_array_equal(
        np.asarray(gate.to_host(bridge.cp_transmission_gain[route_indices])),
        transmission_before,
    )


def test_smoke_engages_bounded_outcome_read_and_both_decisive_lesions(smoke_result):
    assert smoke_result["status"] == "SMOKE_PASS"
    assert smoke_result["science_seed_executed"] is False
    assert all(smoke_result["checks"].values()), smoke_result["checks"]
    intact = smoke_result["conditions"]["intact"]
    expected = intact["expected_value_channel"]
    plateau_lesion = smoke_result["conditions"]["plateau_lesion"]
    outcome_lesion = smoke_result["conditions"][
        "shared_outcome_read_lesion"
    ]

    assert expected == intact["action"]["winner"]
    assert 1.0 <= intact["outcome_value_rate_hz_per_cell"][expected] <= 20.0
    assert max(intact["delay_value_rate_hz_per_cell"]) <= 0.5
    assert intact["action"]["load_value_fs_spikes"][expected] > 0
    assert plateau_lesion["plateau_before_outcome"] == [0.0, 0.0]
    assert (
        plateau_lesion["outcome_value_rate_hz_per_cell"][expected]
        < intact["outcome_value_rate_hz_per_cell"][expected]
    )
    assert outcome_lesion["outcome_value_rate_hz_per_cell"][expected] == 0.0


def test_smoke_frozen_route_permutation_and_gap_decay_are_causal(smoke_result):
    intact = smoke_result["conditions"]["intact"]
    frozen = smoke_result["conditions"]["frozen_expectation_route"]
    permuted = smoke_result["conditions"]["action_channel_permutation"]
    winner = intact["action"]["winner"]

    assert frozen["changed_expectation_synapses"] == 0
    assert frozen["changed_actor_synapses"] > 0
    assert frozen["changed_outside_declared_routes"] == 0
    assert permuted["action"]["winner"] == winner
    assert permuted["expected_value_channel"] == 1 - winner
    assert (
        permuted["plateau_before_outcome"][1 - winner]
        > permuted["plateau_before_outcome"][winner]
    )
    assert permuted["action"]["load_value_fs_spikes"][1 - winner] > 0

    plateaus = smoke_result["gap_decay"]["plateau_expected_channel"]
    assert plateaus[0] > plateaus[1] > plateaus[2] >= 0.0
    assert smoke_result["gap_decay"]["gaps_steps"] == [60, 100, 160]


def test_channel_selectivity_checks_reject_bilateral_neural_state(smoke_result):
    rows = copy.deepcopy(smoke_result["conditions"])
    for name in ("intact", "action_channel_permutation"):
        rows[name]["plateau_before_outcome"] = [83.329, 83.329]
        rows[name]["outcome_value_rate_hz_per_cell"] = [9.375, 9.375]

    checks = gate._smoke_checks(rows)

    assert checks["intact_action_tag_maps_to_executed_channel"]
    assert checks["permutation_moves_tag_to_opposite_value_channel"]
    assert not checks["intact_action_tag_is_neurally_channel_selective"]
    assert not checks["permuted_action_tag_is_neurally_channel_selective"]
