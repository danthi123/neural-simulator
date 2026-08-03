"""Focused tests for the bounded replay/cortical-consolidation calibration gate."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._replay_cortical_consolidation_gate import (  # noqa: E402
    CA3_GATE,
    CALIBRATION_SEEDS,
    CONDITIONS,
    CORTICAL_GATE,
    DEVELOPMENT_SEEDS,
    HELD_OUT_SEEDS,
    INDEX_CUE_GATE,
    INDEX_TARGET_GATE,
    SCHAFFER_GATE,
    _path_weights,
    _calibration_verdict,
    _shuffle_target_index,
    build_bridge,
    run_condition,
    smoke_config,
    validate_calibration_seeds,
)


def test_seed_policy_keeps_development_and_held_out_untouched():
    assert validate_calibration_seeds(CALIBRATION_SEEDS) == CALIBRATION_SEEDS
    for seed in DEVELOPMENT_SEEDS + HELD_OUT_SEEDS:
        with pytest.raises(ValueError, match="calibration seeds"):
            validate_calibration_seeds([seed])


def test_bridge_declares_required_regions_and_causal_gates():
    bridge, handles = build_bridge(212, smoke_config())
    assert set(handles["regions"]) == {"ca3", "ca1", "cortical_cue", "cortical_target"}
    plasticity_gates = set(bridge.list_plasticity_gates())
    assert {CA3_GATE, INDEX_CUE_GATE, INDEX_TARGET_GATE, CORTICAL_GATE, SCHAFFER_GATE} <= plasticity_gates
    assert SCHAFFER_GATE in bridge._transmission_gate_to_synapses
    assert handles["wiring_counts"]["cortical_association"] > 0
    assert handles["wiring_counts"]["ca3_to_ca1"] > 0


def test_shuffle_preserves_target_index_weight_multiset():
    bridge, _ = build_bridge(212, smoke_config())
    before = _path_weights(bridge, INDEX_TARGET_GATE)
    changed = _shuffle_target_index(bridge, 212)
    after = _path_weights(bridge, INDEX_TARGET_GATE)
    assert changed == before.size
    np.testing.assert_array_equal(np.sort(before), np.sort(after))


def test_smoke_runs_all_phases_on_one_persistent_bridge():
    row = run_condition(212, "intact", smoke_config())
    assert row["phase_trace"] == ["encode_A", "encode_B", "sleep", "retest"]
    assert row["single_bridge_persisted"] is True
    assert set(row["recall"]) == {"A", "B"}
    for memory in ("A", "B"):
        assert 0.0 <= row["recall"][memory]["false_recall_fraction"] <= 1.0
        assert row["recall"][memory]["partial_cue_cells"] > 0


def test_no_sleep_and_plasticity_off_are_mechanically_distinct():
    cfg = smoke_config()
    no_sleep = run_condition(212, "no_sleep", cfg)
    plasticity_off = run_condition(212, "cortical_plasticity_off", cfg)
    assert no_sleep["sleep"]["spikes"]["ca3"] == 0
    assert plasticity_off["sleep"]["spikes"]["ca3"] > 0
    assert abs(no_sleep["weight_deltas"]["cortical_during_sleep"]) < 1e-7
    assert abs(plasticity_off["weight_deltas"]["cortical_during_sleep"]) < 1e-7


def test_condition_names_are_closed():
    assert CONDITIONS == (
        "intact",
        "no_sleep",
        "shuffled_target_index",
        "ca3_ca1_lesion",
        "cortical_plasticity_off",
    )
    with pytest.raises(ValueError, match="Unknown condition"):
        run_condition(212, "not_a_condition", smoke_config())


def test_smoke_verdict_carries_measured_preconditions():
    cfg = smoke_config()
    conditions = {condition: run_condition(212, condition, cfg) for condition in CONDITIONS}
    verdict = _calibration_verdict(conditions)

    assert verdict["preconditions"]
    assert all(check["ok"] is not None for check in verdict["preconditions"])
