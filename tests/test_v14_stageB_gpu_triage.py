"""Focused tests for Stage B GPU engineering triage."""

from __future__ import annotations

import numpy as np

from tools.v14_stageB_gpu_triage import _classify, _event_metrics


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
