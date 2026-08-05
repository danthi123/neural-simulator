"""Tests for the consumed-candidate V3 GPU batch-width benchmark."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.v14_stageB_batch_width_benchmark import (
    StageBBatchWidthBenchmarkError,
    _select_width,
    load_benchmark_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research/specs/v14_snr_stageB_v3_batch_width_benchmark.json"


def test_filed_benchmark_uses_consumed_candidates_and_fixed_paired_order() -> None:
    digest = hashlib.sha256(SPEC.read_bytes()).hexdigest()

    spec, root = load_benchmark_spec(SPEC, digest, repository_root=ROOT)

    assert root == ROOT
    assert spec["run_order"] == [64, 128, 256, 512, 512, 256, 128, 64]
    assert "successor" not in spec["consumed_campaign"]["path"]
    assert spec["selection"]["near_tie_fraction"] == 0.05


def test_selection_chooses_smallest_width_within_five_percent() -> None:
    rows = []
    rates = {64: (90.0, 92.0), 128: (97.0, 99.0), 256: (100.0, 102.0), 512: (101.0, 103.0)}
    for width, values in rates.items():
        for value in values:
            rows.append({"width": width, "candidate_steps_per_second": value})

    selected, summary = _select_width(rows)

    assert selected == 128
    assert summary["512"]["replicates"] == 2


def test_spec_digest_mismatch_fails_before_execution() -> None:
    with pytest.raises(StageBBatchWidthBenchmarkError, match="digest does not match"):
        load_benchmark_spec(SPEC, "0" * 64, repository_root=ROOT)
