"""Focused tests for Stage B GPU engineering triage."""

from __future__ import annotations

import numpy as np
import pytest

from tools.v14_stageB_gpu_triage import (
    StageBGPUTriageError,
    _classify,
    _event_metrics,
    _nap_metrics,
    _validate_phased_nap_intervention,
)


def test_event_metrics_follow_filed_101_spike_convention() -> None:
    times = np.arange(0.05, 6.0, 0.05, dtype=np.float64)
    spikes = np.zeros(times.size, dtype=np.bool_)
    spikes[:101] = True
    metrics = _event_metrics(spikes, times)
    assert metrics["spike_count"] == 101
    assert metrics["firing_rate_hz"] == 20.0
    assert metrics["isi_cv"] < 1e-12


def test_incomplete_event_count_keeps_rate_and_cv_unavailable() -> None:
    times = np.arange(0.05, 1.05, 0.05, dtype=np.float64)
    spikes = np.ones(times.size, dtype=np.bool_)
    metrics = _event_metrics(spikes, times)
    assert metrics == {"spike_count": 20, "firing_rate_hz": None, "isi_cv": None}


def test_classification_is_strict_and_noncompensating() -> None:
    metrics = {
        "intact_autonomous": {"spike_count": 101, "firing_rate_hz": 10.0, "isi_cv": 0.1},
        "nap_lesion": {"spike_count": 0, "firing_rate_hz": None, "isi_cv": None},
        "cav2_2_lesion": {"spike_count": 101, "firing_rate_hz": 11.0, "isi_cv": 0.2},
        "sk_lesion": {"spike_count": 101, "firing_rate_hz": 10.0, "isi_cv": 0.3},
        "hcn_baseline_lesion": {"spike_count": 101, "firing_rate_hz": 9.0, "isi_cv": 0.1},
    }
    classification, checks = _classify(metrics)
    assert classification == "engineering_pass"
    assert all(item["passed"] is True for item in checks)
    metrics["sk_lesion"]["isi_cv"] = 0.05
    assert _classify(metrics)[0] == "engineering_fail"
    metrics["sk_lesion"]["isi_cv"] = None
    assert _classify(metrics)[0] == "engineering_inconclusive"


def test_phased_nap_metrics_require_stable_baseline_and_hyperpolarization() -> None:
    dt = 0.00005
    times = np.arange(1, 60_001, dtype=np.float64) * dt
    spikes = np.zeros(times.size, dtype=np.bool_)
    voltage = np.full(times.size, -60.0, dtype=np.float64)
    voltage[times >= 2.0] = -65.0

    metrics = _nap_metrics(spikes, times, voltage)

    assert metrics["same_cell_phased"] is True
    assert metrics["baseline_stable"] is True
    assert metrics["post_lesion_delta_mV"] == -5.0
    assert metrics["spike_count"] == 0


def test_phased_nap_wrong_voltage_direction_is_an_engineering_failure() -> None:
    metrics = {
        "intact_autonomous": {"spike_count": 101, "firing_rate_hz": 10.0, "isi_cv": 0.1},
        "nap_lesion": {
            "spike_count": 0,
            "firing_rate_hz": None,
            "isi_cv": None,
            "same_cell_phased": True,
            "baseline_stable": True,
            "stability_delta_mV": 0.1,
            "post_lesion_delta_mV": 1.0,
        },
        "cav2_2_lesion": {"spike_count": 101, "firing_rate_hz": 11.0, "isi_cv": 0.2},
        "sk_lesion": {"spike_count": 101, "firing_rate_hz": 10.0, "isi_cv": 0.3},
        "hcn_baseline_lesion": {"spike_count": 101, "firing_rate_hz": 9.0, "isi_cv": 0.1},
    }

    classification, checks = _classify(metrics)

    assert classification == "engineering_fail"
    voltage_check = next(
        item for item in checks if item["metric"] == "median_membrane_voltage_change_mV"
    )
    assert voltage_check["passed"] is False


def test_phased_nap_intervention_is_fail_closed() -> None:
    intervention = {
        "kind": "complete_intrinsic_current_lesion",
        "operation": "set_conductance_density_to_zero_between_post_update_samples",
        "target": "nap",
        "runtime_conductance_field": "cp_snr_g_nap_max",
        "conductance_density_unit": "mS/cm^2",
        "before": 0.2,
        "after": 0.0,
        "timestamp_s": 2.0,
        "lesion_onset_sample_index": 39_999,
        "lesion_onset_sample_number": 40_000,
        "last_intact_sample_s": 1.99995,
        "first_lesion_sample_s": 2.0,
    }
    _validate_phased_nap_intervention(intervention)

    intervention["lesion_onset_sample_index"] = 40_000
    with pytest.raises(StageBGPUTriageError, match="exact lesion"):
        _validate_phased_nap_intervention(intervention)
