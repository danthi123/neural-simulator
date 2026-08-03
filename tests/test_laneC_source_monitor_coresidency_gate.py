"""Focused guards for the bounded Lane C source-monitor co-residency gate."""
from __future__ import annotations

import inspect
import os

import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import _laneC_source_monitor_coresidency_gate as gate_module
from research.runners._laneC_source_monitor_coresidency_gate import (
    ACC_GATE,
    APFC_GATE,
    CALIBRATION_SEEDS,
    DEVELOPMENT_SEEDS,
    HELD_OUT_SEEDS,
    SOURCE_AFFERENT_GATE,
    SOURCE_LEARNING_GATE,
    SOURCE_RECALL_GATE,
    SOURCES,
    SourceMonitorCoresidencyGate,
    evaluate_calibration_seed,
    make_episode_patterns,
)


@pytest.fixture(scope="module")
def calibration_result():
    return evaluate_calibration_seed(CALIBRATION_SEEDS[0])


def test_one_bridge_contains_episode_source_apfc_and_acc_populations():
    gate = SourceMonitorCoresidencyGate(seed=CALIBRATION_SEEDS[0])
    region_names = {region.name for region in gate.bridge.region_manager.regions()}

    assert gate.bridge.cp_firing_states.shape[0] == 304
    assert "episode_activity" in region_names
    assert "acc_source_monitor" in region_names
    for source in SOURCES:
        assert f"source_memory_{source}" in region_names
        assert f"apfc_source_{source}" in region_names

    assert set(gate.bridge.list_plasticity_gates()) == {SOURCE_LEARNING_GATE}
    assert set(gate.bridge._transmission_gate_to_synapses) == {
        SOURCE_RECALL_GATE,
        SOURCE_AFFERENT_GATE,
        APFC_GATE,
        ACC_GATE,
    }
    weights = gate.weight_summary()
    assert weights["n_synapses"] == 3 * 192 * 12
    assert weights["l1"] == 0.0


def test_inference_accepts_episode_activity_without_proposition_or_source_metadata():
    signature = inspect.signature(SourceMonitorCoresidencyGate.recall)
    source = inspect.getsource(gate_module)

    assert list(signature.parameters) == [
        "self",
        "episode_pattern",
        "source_path_lesion",
        "acc_lesion",
    ]
    assert "candidate" not in signature.parameters
    assert "confidence" not in signature.parameters
    assert "source" not in signature.parameters
    assert "hashlib" not in source
    assert "host_response_decision\": False" in source


def test_episode_patterns_are_explicit_disjoint_activity_not_complete_propositions():
    patterns = make_episode_patterns(CALIBRATION_SEEDS[0], 4)

    assert all(len(pattern) == 12 for pattern in patterns)
    assert len(set().union(*(set(pattern.tolist()) for pattern in patterns))) == 48


def test_calibration_learns_all_sources_and_follows_source_swap(calibration_result):
    components = calibration_result["components"]

    assert components["learned_routes_start_zero"]
    assert components["experience_changes_synaptic_weights"]
    assert components["seen_source_recalled"]
    assert components["heard_source_recalled"]
    assert components["self_source_recalled"]
    assert components["source_swap_follows_afferent_activity"]


def test_mixed_source_and_neural_monitor_controls(calibration_result):
    components = calibration_result["components"]
    records = calibration_result["records"]

    assert components["mixed_source_reinstates_both"]
    assert records["mixed"]["source_spikes"]["seen"] > 0.0
    assert records["mixed"]["source_spikes"]["heard"] > 0.0
    assert components["source_spikes_reach_apfc_and_acc"]
    assert components["source_path_lesion_collapses_recall"]
    assert components["acc_lesion_preserves_source_and_silences_acc"]


def test_learning_off_control_changes_neither_weights_nor_recall(calibration_result):
    components = calibration_result["components"]

    assert components["learning_off_keeps_weights_zero"]
    assert components["learning_off_has_no_source_recall"]


@pytest.mark.parametrize("seed", DEVELOPMENT_SEEDS + HELD_OUT_SEEDS)
def test_reserved_seeds_are_rejected_by_calibration_evaluator(seed):
    with pytest.raises(ValueError, match="is not a calibration seed"):
        evaluate_calibration_seed(seed)
