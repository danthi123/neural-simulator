"""Focused guards for source-monitor co-residency v3 calibration."""
from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import _laneC_source_monitor_coresidency_gate_v3 as gate_module
from research.runners._laneC_source_monitor_coresidency_gate import SOURCE_MEMORY, SOURCES
from research.runners._laneC_source_monitor_coresidency_gate_v2 import (
    SourceMonitorConfigV2,
)
from research.runners._laneC_source_monitor_coresidency_gate_v3 import (
    CALIBRATION_SEEDS,
    DEVELOPMENT_SEEDS,
    HELD_OUT_SEEDS,
    HOMEOSTASIS_ADAPT_RATE,
    HOMEOSTASIS_EMA_ALPHA,
    HOMEOSTASIS_SETTLING_STEPS,
    HOMEOSTASIS_TARGET_RATE,
    HOMEOSTASIS_THRESHOLD_MAX,
    HOMEOSTASIS_THRESHOLD_MIN,
    MIN_SOURCE_MARGIN,
    OPEN_PHASES,
    SMOKE_SEED,
    SourceMonitorConfigV3,
    SourceMonitorCoresidencyGateV3,
    bounded_tradeoff_assessment,
    evaluate_calibration_seed,
    validate_phase_seed,
)
from sim.backend import to_host


@pytest.fixture(scope="module")
def gate():
    return SourceMonitorCoresidencyGateV3(seed=SMOKE_SEED)


def test_v3_seed_sets_are_fresh_and_only_calibration_is_open():
    prior_source_seeds = {
        212,
        213,
        214,
        215,
        216,
        217,
        218,
        219,
        220,
        221,
        222,
        223,
        310,
        311,
        312,
        313,
        314,
        315,
        316,
        317,
        318,
        319,
        320,
        321,
    }

    assert SMOKE_SEED == 220
    assert CALIBRATION_SEEDS == (232, 233)
    assert DEVELOPMENT_SEEDS == (234, 235, 330)
    assert HELD_OUT_SEEDS == (331, 332, 333)
    assert OPEN_PHASES == ("calibration",)
    assert not prior_source_seeds.intersection(
        CALIBRATION_SEEDS + DEVELOPMENT_SEEDS + HELD_OUT_SEEDS
    )


@pytest.mark.parametrize("seed", CALIBRATION_SEEDS)
def test_calibration_seed_validation_accepts_only_open_seeds(seed):
    assert validate_phase_seed(seed, "calibration") == seed


def test_smoke_seed_is_rejected_by_formal_evaluator():
    with pytest.raises(ValueError, match="not a v3 calibration seed"):
        evaluate_calibration_seed(SMOKE_SEED)


@pytest.mark.parametrize("seed", DEVELOPMENT_SEEDS + HELD_OUT_SEEDS)
def test_reserved_seeds_are_rejected_before_evaluation(seed):
    with pytest.raises(ValueError, match="not a v3 calibration seed"):
        evaluate_calibration_seed(seed)


@pytest.mark.parametrize("phase", ("development", "held-out"))
def test_reserved_phases_are_mechanically_locked(phase):
    with pytest.raises(ValueError, match="is not open"):
        validate_phase_seed(CALIBRATION_SEEDS[0], phase)


def test_v3_freezes_v2_operating_point_and_canonical_homeostasis():
    v2 = SourceMonitorConfigV2()
    v3 = SourceMonitorConfigV3()

    for field_name in SourceMonitorConfigV2.__dataclass_fields__:
        assert getattr(v3, field_name) == getattr(v2, field_name)
    assert v3.source_homeostasis_target_rate == HOMEOSTASIS_TARGET_RATE == 0.02
    assert v3.source_homeostasis_ema_alpha == HOMEOSTASIS_EMA_ALPHA == 0.0002
    assert v3.source_homeostasis_adapt_rate == HOMEOSTASIS_ADAPT_RATE == 0.0005
    assert v3.source_homeostasis_threshold_min == HOMEOSTASIS_THRESHOLD_MIN == -55.0
    assert v3.source_homeostasis_threshold_max == HOMEOSTASIS_THRESHOLD_MAX == -30.0
    assert v3.source_homeostasis_settling_steps == HOMEOSTASIS_SETTLING_STEPS == 5000

    with pytest.raises(ValueError, match="freezes inherited v2 field 'drive_pA'"):
        SourceMonitorCoresidencyGateV3(
            seed=SMOKE_SEED,
            config=SourceMonitorConfigV3(drive_pA=v2.drive_pA + 1.0),
        )


def test_one_bridge_enables_homeostasis_only_on_source_memory(gate):
    regions = {
        region.name: region for region in gate.bridge.region_manager.regions()
    }
    enabled_regions = {
        name for name, region in regions.items() if region.enable_homeostasis
    }

    assert gate.bridge.cp_firing_states.shape[0] == 322
    assert enabled_regions == set(SOURCE_MEMORY.values())
    assert np.array_equal(gate.homeostasis_mask(), gate.expected_homeostasis_mask())
    update_mask = np.asarray(
        to_host(gate.bridge.cp_homeostasis_update_neuron_mask), dtype=bool
    )
    assert not update_mask.any()


def test_local_threshold_update_changes_only_source_neurons():
    intact = SourceMonitorCoresidencyGateV3(seed=SMOKE_SEED)
    lesion = SourceMonitorCoresidencyGateV3(seed=SMOKE_SEED)
    intact_before = np.asarray(
        to_host(intact.bridge.cp_neuron_firing_thresholds), dtype=np.float64
    ).copy()
    lesion_before = np.asarray(
        to_host(lesion.bridge.cp_neuron_firing_thresholds), dtype=np.float64
    ).copy()
    assert np.array_equal(intact_before, lesion_before)

    intact.set_local_homeostasis_updates(True)
    for _ in range(4):
        intact.bridge._run_one_simulation_step()
        lesion.bridge._run_one_simulation_step()
    intact.set_local_homeostasis_updates(False)

    intact_after = np.asarray(
        to_host(intact.bridge.cp_neuron_firing_thresholds), dtype=np.float64
    )
    lesion_after = np.asarray(
        to_host(lesion.bridge.cp_neuron_firing_thresholds), dtype=np.float64
    )
    changed = np.abs(intact_after - intact_before) > 0.0

    assert changed.any()
    assert not changed[~intact.expected_homeostasis_mask()].any()
    assert np.array_equal(lesion_before, lesion_after)


def test_inference_interface_exposes_no_source_metadata():
    signature = inspect.signature(SourceMonitorCoresidencyGateV3.recall)
    source = inspect.getsource(gate_module)

    assert list(signature.parameters) == [
        "self",
        "episode_pattern",
        "source_path_lesion",
        "acc_lesion",
    ]
    assert "source" not in signature.parameters
    assert "confidence" not in signature.parameters
    assert "candidate" not in signature.parameters
    assert "np.argsort" not in source
    assert "np.argpartition" not in source
    assert '"host_gain_normalization": False' in source
    assert '"host_response_decision": False' in source


def test_bounded_tradeoff_accepts_floor_protected_redistribution():
    assessment = bounded_tradeoff_assessment(
        {"seen": 0.18, "heard": 0.16, "self_generated": 0.21},
        {"seen": 0.20, "heard": 0.14, "self_generated": 0.24},
    )

    assert assessment["components"] == {
        "all_source_margins_meet_fixed_floor": True,
        "bounded_homeostasis_tradeoffs_protect_floor": True,
        "homeostasis_strictly_improves_weakest_source_margin": True,
    }
    assert assessment["losses"] == pytest.approx(
        {"seen": 0.02, "heard": 0.0, "self_generated": 0.03}
    )
    assert assessment["spendable_surplus"] == pytest.approx(
        {"seen": 0.05, "heard": 0.0, "self_generated": 0.09}
    )


def test_tradeoff_rejects_below_floor_or_no_weakest_source_improvement():
    below_floor = bounded_tradeoff_assessment(
        {"seen": 0.20, "heard": 0.13, "self_generated": 0.22},
        {"seen": 0.19, "heard": 0.14, "self_generated": 0.21},
    )
    no_improvement = bounded_tradeoff_assessment(
        {"seen": 0.20, "heard": 0.16, "self_generated": 0.22},
        {"seen": 0.19, "heard": 0.16, "self_generated": 0.21},
    )

    assert MIN_SOURCE_MARGIN == 0.15
    assert not below_floor["components"]["all_source_margins_meet_fixed_floor"]
    assert not below_floor["components"][
        "bounded_homeostasis_tradeoffs_protect_floor"
    ]
    assert not no_improvement["components"][
        "homeostasis_strictly_improves_weakest_source_margin"
    ]


def test_all_inherited_causal_controls_remain_in_the_v3_evaluator():
    source = inspect.getsource(gate_module.evaluate_calibration_seed)
    inherited = {
        "learned_routes_start_zero",
        "experience_changes_synaptic_weights",
        "seen_source_recalled",
        "heard_source_recalled",
        "self_source_recalled",
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
    }

    assert all(f'"{name}"' in source for name in inherited)
    assert '"homeostasis_mask_is_source_local"' in source
    assert '"homeostasis_thresholds_change_and_lesion_stays_fixed"' in source
    assert '"matched_arms_keep_identical_learned_weights"' in source
