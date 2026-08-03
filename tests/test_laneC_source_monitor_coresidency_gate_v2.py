"""Focused guards for source-monitor co-residency v2 calibration."""
from __future__ import annotations

import inspect
import os

import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import _laneC_source_monitor_coresidency_gate_v2 as gate_module
from research.runners._laneC_source_monitor_coresidency_gate import SourceMonitorConfig
from research.runners._laneC_source_monitor_coresidency_gate_v2 import (
    CALIBRATION_SEEDS,
    DEVELOPMENT_SEEDS,
    HELD_OUT_SEEDS,
    MIN_ATTRIBUTION_FRACTION,
    MIN_SOURCE_MARGIN,
    SOURCE_COMPETITION_GATE,
    SOURCE_INTERNEURON,
    SOURCES,
    SourceMonitorConfigV2,
    SourceMonitorCoresidencyGateV2,
    evaluate_calibration_seed,
    validate_calibration_seed,
)


@pytest.fixture(scope="module")
def calibration_result():
    return evaluate_calibration_seed(CALIBRATION_SEEDS[0])


def test_v2_uses_one_bridge_with_local_source_competition():
    gate = SourceMonitorCoresidencyGateV2(seed=CALIBRATION_SEEDS[0])
    region_names = {region.name for region in gate.bridge.region_manager.regions()}

    assert gate.bridge.cp_firing_states.shape[0] == 322
    assert set(SOURCE_INTERNEURON.values()).issubset(region_names)
    assert SOURCE_COMPETITION_GATE in gate.bridge._transmission_gate_to_synapses
    regions = {region.name: region for region in gate.bridge.region_manager.regions()}
    for source in SOURCES:
        region = regions[SOURCE_INTERNEURON[source]]
        assert region.exc_fraction == 0.0
        assert region.izh_neuron_type == "IZH2007_FS_CORTICAL_INTERNEURON"
    assert gate.weight_summary()["l1"] == 0.0


def test_v2_preserves_input_strength_and_uses_no_host_gain_normalization():
    v1 = SourceMonitorConfig()
    v2 = SourceMonitorConfigV2()
    source = inspect.getsource(gate_module)

    assert v2.drive_pA == v1.drive_pA
    assert v2.source_afferent_weight == v1.source_afferent_weight
    assert "np.argsort" not in source
    assert "np.argpartition" not in source
    assert "host_gain_normalization\": False" in source


def test_inference_interface_exposes_no_source_metadata():
    signature = inspect.signature(SourceMonitorCoresidencyGateV2.recall)

    assert list(signature.parameters) == [
        "self",
        "episode_pattern",
        "source_path_lesion",
        "acc_lesion",
    ]
    assert "source" not in signature.parameters
    assert "confidence" not in signature.parameters
    assert "candidate" not in signature.parameters


def test_calibration_preserves_all_required_controls(calibration_result):
    components = calibration_result["components"]
    required = {
        "learned_routes_start_zero",
        "experience_changes_synaptic_weights",
        "seen_source_recalled",
        "heard_source_recalled",
        "self_source_recalled",
        "all_source_margins_meet_fixed_floor",
        "source_swap_follows_afferent_activity",
        "mixed_source_reinstates_both",
        "source_path_lesion_collapses_recall",
        "source_path_attribution_meets_fixed_floor",
        "acc_lesion_preserves_source_and_silences_acc",
        "acc_path_attribution_meets_fixed_floor",
        "learning_off_keeps_weights_zero",
        "learning_off_has_no_source_recall",
        "unseen_episode_has_no_source_recall",
        "source_spikes_reach_apfc_and_acc",
        "competition_circuit_is_active_and_lesionable",
        "competition_stabilizes_without_harming_source_margins",
    }

    assert required == set(components)
    assert all(components.values())
    assert calibration_result["status"] == "CALIBRATION_PASS"
    assert all(check["ok"] for check in calibration_result["preconditions"])


def test_fixed_floors_and_causal_attribution_are_not_weakened(calibration_result):
    metrics = calibration_result["metrics"]
    attribution = calibration_result["attribution"]

    assert MIN_SOURCE_MARGIN == 0.15
    assert MIN_ATTRIBUTION_FRACTION == 0.90
    assert metrics["minimum_source_margin"] >= MIN_SOURCE_MARGIN
    assert attribution["source_recall_path"]["attributable_fraction"] >= 0.90
    assert attribution["acc_output_path"]["attributable_fraction"] >= 0.90


def test_mixed_unseen_learning_off_and_competition_lesion_records(calibration_result):
    records = calibration_result["records"]

    assert records["mixed"]["source_spikes"]["seen"] > 0.0
    assert records["mixed"]["source_spikes"]["heard"] > 0.0
    assert sum(records["unseen"]["source_spikes"].values()) == 0.0
    assert sum(records["learning_off"]["source_spikes"].values()) == 0.0
    assert all(
        sum(record["competition_spikes"].values()) == 0.0
        for record in records["competition_lesions"].values()
    )
    gains = calibration_result["metrics"]["competition_margin_gains"]
    assert min(gains.values()) >= 0.0
    assert max(gains.values()) > 0.0


@pytest.mark.parametrize("seed", DEVELOPMENT_SEEDS + HELD_OUT_SEEDS)
def test_reserved_seeds_are_mechanically_rejected(seed):
    with pytest.raises(ValueError, match="is not a v2 calibration seed"):
        evaluate_calibration_seed(seed)


@pytest.mark.parametrize("seed", CALIBRATION_SEEDS)
def test_only_fresh_v2_calibration_seeds_are_open(seed):
    assert validate_calibration_seed(seed) == seed


def test_seed_sets_are_fresh_and_disjoint_from_v1():
    assert CALIBRATION_SEEDS == (216, 217)
    assert DEVELOPMENT_SEEDS == (218, 219, 314)
    assert HELD_OUT_SEEDS == (315, 316, 317)
    assert not set(CALIBRATION_SEEDS) & {212, 213, 214, 215, 310, 311, 312, 313}
