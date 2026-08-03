"""Focused guards for adaptive-inhibition source monitor v4."""
from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import _laneC_source_monitor_coresidency_gate_v4 as gate_module
from research.runners._laneC_source_monitor_coresidency_gate import SOURCES
from research.runners._laneC_source_monitor_coresidency_gate_v2 import (
    SourceMonitorConfigV2,
    SourceMonitorCoresidencyGateV2,
)
from research.runners._laneC_source_monitor_coresidency_gate_v4 import (
    CALIBRATION_SEEDS,
    DEVELOPMENT_SEEDS,
    HELD_OUT_SEEDS,
    INHIBITORY_LEARNING_GATE,
    INHIBITORY_REHEARSAL_STEPS,
    ISTDP_ETA,
    ISTDP_INITIAL_WEIGHT,
    ISTDP_TARGET_RATE_PER_STEP,
    ISTDP_TAU_MS,
    ISTDP_WEIGHT_MAX,
    ISTDP_WEIGHT_MIN,
    OPEN_PHASES,
    SMOKE_SEED,
    SourceMonitorConfigV4,
    SourceMonitorCoresidencyGateV4,
    adaptive_inhibition_assessment,
    run_smoke,
    validate_individual_seed,
    validate_phase_seeds,
)


@pytest.fixture(scope="module")
def gate():
    return SourceMonitorCoresidencyGateV4(seed=SMOKE_SEED)


@pytest.fixture(scope="module")
def smoke():
    return run_smoke()


def test_seed_partition_is_fresh_exact_and_later_phases_are_locked():
    assert SMOKE_SEED == 600
    assert CALIBRATION_SEEDS == (601, 607)
    assert DEVELOPMENT_SEEDS == (613, 617, 619)
    assert HELD_OUT_SEEDS == (631, 641, 643)
    assert OPEN_PHASES == ("calibration",)
    assert validate_phase_seeds("calibration", CALIBRATION_SEEDS) == CALIBRATION_SEEDS
    for seed in CALIBRATION_SEEDS:
        assert validate_individual_seed(seed, "calibration") == seed
    for invalid in ((601,), (607,), (607, 601), (601, 601)):
        with pytest.raises(ValueError):
            validate_phase_seeds("calibration", invalid)
    for phase in ("development", "held-out"):
        with pytest.raises(ValueError, match="not open"):
            validate_individual_seed(CALIBRATION_SEEDS[0], phase)
    with pytest.raises(ValueError, match="not a v4 calibration seed"):
        validate_individual_seed(SMOKE_SEED, "calibration")


def test_v4_inherits_v2_and_freezes_operating_point_and_rule():
    assert issubclass(SourceMonitorCoresidencyGateV4, SourceMonitorCoresidencyGateV2)
    v2 = SourceMonitorConfigV2()
    v4 = SourceMonitorConfigV4()
    for field_name in SourceMonitorConfigV2.__dataclass_fields__:
        assert getattr(v4, field_name) == getattr(v2, field_name)
    assert v4.interneuron_to_rival_weight == ISTDP_INITIAL_WEIGHT == 3.0
    assert v4.inhibitory_stdp_tau_ms == ISTDP_TAU_MS == 20.0
    assert v4.inhibitory_stdp_target_rate_per_step == ISTDP_TARGET_RATE_PER_STEP == 0.02
    assert v4.inhibitory_stdp_eta == ISTDP_ETA == 0.001
    assert (v4.inhibitory_stdp_w_min, v4.inhibitory_stdp_w_max) == (
        ISTDP_WEIGHT_MIN,
        ISTDP_WEIGHT_MAX,
    ) == (0.0, 6.0)
    assert v4.inhibitory_rehearsal_steps == INHIBITORY_REHEARSAL_STEPS == 5000
    with pytest.raises(ValueError, match="freezes 'inhibitory_stdp_eta'"):
        SourceMonitorCoresidencyGateV4(
            seed=SMOKE_SEED,
            config=SourceMonitorConfigV4(inhibitory_stdp_eta=0.002),
        )


def test_one_bridge_has_only_expected_inhibitory_routes_plastic(gate):
    bridge = gate.bridge
    cfg = bridge.core_config
    indices = gate.inhibitory_synapse_indices()
    plastic = np.asarray(bridge.cp_synapse_plastic_mask, dtype=bool)

    assert cfg.enable_inhibitory_stdp is True
    assert cfg.enable_stdp is False
    assert cfg.enable_homeostasis is False
    assert cfg.inhibitory_stdp_tau_ms == ISTDP_TAU_MS
    assert cfg.inhibitory_stdp_target_rate_per_step == ISTDP_TARGET_RATE_PER_STEP
    assert cfg.inhibitory_stdp_eta == ISTDP_ETA
    assert bridge.cp_inhibitory_stdp_trace.shape == bridge.cp_firing_states.shape
    assert indices.size == len(SOURCES) * (len(SOURCES) - 1) * 6 * 12
    assert plastic[indices].all()
    assert bridge._plasticity_gate_values[INHIBITORY_LEARNING_GATE] == 0.0
    assert np.all(gate.inhibitory_weight_vector() == ISTDP_INITIAL_WEIGHT)


def test_rehearsal_separates_hebbian_and_inhibitory_learning_in_source():
    source = inspect.getsource(SourceMonitorCoresidencyGateV4.rehearse_inhibitory_competition)
    assert "enable_hebbian_learning = False" in source
    assert "SOURCE_LEARNING_GATE, 0.0" in source
    assert "INHIBITORY_LEARNING_GATE" in source
    assert "self._rest()" in source


def test_formal_evaluator_contains_all_preregistered_arms_and_controls():
    source = inspect.getsource(gate_module.evaluate_calibration_seed)
    required = {
        "learning_lesion",
        "expression_lesion",
        "source_lesion",
        "acc_lesion",
        "swapped",
        "mixed",
        "unseen",
        "learning_off",
        "inhibitory_weights_change_only_in_intact",
        "rehearsal_preserves_excitatory_weights_and_thresholds",
        "source_path_lesion_collapses_recall",
        "acc_lesion_preserves_source_and_silences_acc",
        "source_spikes_reach_apfc_and_acc",
        "competition_circuit_is_active_and_lesionable",
    }
    assert all(name in source for name in required)
    assert list(inspect.signature(SourceMonitorCoresidencyGateV4.recall).parameters) == [
        "self",
        "episode_pattern",
        "source_path_lesion",
        "acc_lesion",
    ]


def test_adaptive_assessment_enforces_floor_tradeoff_and_rival_burden():
    def record(target: str, target_spikes: float, rival_spikes: float) -> dict:
        spikes = {source: rival_spikes for source in SOURCES}
        spikes[target] = target_spikes
        return {
            "source_spikes": spikes,
            "source_rates": {source: count / 1200.0 for source, count in spikes.items()},
        }

    intact = {source: record(source, 400.0, 50.0) for source in SOURCES}
    learning_lesion = {source: record(source, 350.0, 100.0) for source in SOURCES}
    expression_lesion = {source: record(source, 350.0, 120.0) for source in SOURCES}
    assessment = adaptive_inhibition_assessment(
        intact, learning_lesion, expression_lesion
    )
    assert all(assessment["components"].values())


def test_smoke_is_non_scientific_and_activity_selective(smoke):
    assert smoke["seed"] == SMOKE_SEED
    assert smoke["status"] == "SMOKE_PASS"
    assert smoke["scientific_verdict"] is None
    assert all(smoke["checks"].values())
    heard = smoke["records"]["heard_coactive"]
    swapped = smoke["records"]["self_generated_coactive"]
    lesion = smoke["records"]["learning_lesion"]
    assert heard["coactive_delta"] > heard["silent_delta"]
    assert swapped["coactive_delta"] > swapped["silent_delta"]
    assert lesion["inhibitory_weights_before"] == lesion["inhibitory_weights_after"]
