from __future__ import annotations

import copy

import pytest

from tools import v13_stage0_performance_continuation as continuation


def _performance(go: bool = True) -> dict:
    checks = {
        "old_baseline_supplied": True,
        "default_off_ratio": True,
        "normal_active_ratio": True,
        "v1_active_ratio": True,
        "v2_active_ratio": True,
        "feature_storage": True,
        "default_does_not_allocate": True,
        "v1_dispatches": True,
        "v2_dispatches": True,
    }
    if not go:
        checks["default_off_ratio"] = False
    cells = {}
    for mode in ("normal", "v1", "v2"):
        cells[f"{mode}_default"] = {
            "median_seconds": 1.0,
            "feature_bytes": [0, 0, 0],
            "megakernel_dispatch": [False, False, False],
        }
        cells[f"{mode}_active"] = {
            "median_seconds": 1.0,
            "feature_bytes": [2400, 2400, 2400],
            "megakernel_dispatch": [True, True, True],
        }
    if not go:
        cells["normal_default"]["median_seconds"] = 1.03
    return {
        "stage": "performance",
        "source_sha": continuation.CANDIDATE_REVISION,
        "backend": "cupy",
        "device": "NVIDIA GeForce RTX 3090",
        "checks": checks,
        "cells": cells,
        "old_baseline": {
            "source_sha": continuation.legacy.BASE_REVISION,
            "outcome": "BASELINE_RECORDED",
            "median_seconds": 1.0,
        },
        "ratios": {
            "default_vs_old": 1.0 if go else 1.03,
            "normal_active": 1.0 if go else 1.0 / 1.03,
            "v1_active": 1.0,
            "v2_active": 1.0,
        },
        "go": go,
        "outcome": "PERFORMANCE_GO" if go else "PERFORMANCE_NO_GO",
    }


def test_readiness_accepts_only_the_sealed_v6_inputs() -> None:
    result = continuation.readiness()
    assert result["status"] == "READY"
    assert set(result["v6_inputs"]) == set(continuation.MANIFEST_SHA256)
    assert result["stage1_seed_1031"] == "sealed-not-read-or-executed"


@pytest.mark.parametrize("go", [True, False])
def test_performance_accepts_positive_and_negative_earned_verdicts(go: bool) -> None:
    assert continuation._validate_performance(_performance(go)) is go


def test_performance_rejects_verdict_measurement_disagreement() -> None:
    artifact = copy.deepcopy(_performance(True))
    artifact["checks"]["v2_dispatches"] = False
    with pytest.raises(continuation.ContinuationError, match="checks differ"):
        continuation._validate_performance(artifact)


def test_performance_rejects_unexpected_check_surface() -> None:
    artifact = copy.deepcopy(_performance(True))
    artifact["checks"]["new_unregistered_check"] = True
    with pytest.raises(continuation.ContinuationError, match="invalid structure"):
        continuation._validate_performance(artifact)
