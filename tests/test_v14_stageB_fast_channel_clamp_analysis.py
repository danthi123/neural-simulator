import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import v14_stageB_fast_channel_clamp_analysis as analysis


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research/specs/v14_snr_stageB_fast_channel_clamp_execution_v1.json"
SPEC_SHA256 = "d99ed6a8fd3f1c6e871ded2358c69bf3807549ba62f4e178689ff5a29c1e330e"


def test_exact_sealed_evidence_produces_structural_no_go():
    assert hashlib.sha256(SPEC.read_bytes()).hexdigest() == SPEC_SHA256

    result = analysis.analyze(SPEC, SPEC_SHA256, repository_root=ROOT)

    assert result["schema"] == analysis.OUTPUT_SCHEMA
    assert result["scientific_verdict"] == "STAGE1_STRUCTURAL_NO_GO"
    assert result["source_transfer_status"] == "STRUCTURAL_NO_GO"
    assert result["failed_metric_count"] > 0
    assert result["backend_parity"]["status"] == "NOT_ESTABLISHED_NO_PREREGISTERED_TOLERANCE"
    assert result["backend_parity"]["promoting"] is False
    assert result["sha256"] == analysis._digest({key: value for key, value in result.items() if key != "sha256"})


def test_source_gate_does_not_allow_aggregate_compensation():
    result = analysis.analyze(SPEC, SPEC_SHA256, repository_root=ROOT)
    gates = {row["metric"]: row for row in result["source_transfer_gates"]}

    assert gates["fast_na.activation_vhalf_mV"]["passed"] is False
    assert gates["fast_na.activation_10_90_at_0_mV_ms"]["passed"] is False
    assert gates["fast_na.recovery_slow_tau_ms"]["passed"] is True
    assert gates["kv3_like.rise_20_80_at_plus_40_mV_ms"]["passed"] is True


def test_rise_crossings_are_linearly_interpolated():
    assay = {
        "elapsed_ms": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "trace_normalized_absolute_current": [[0.0, 0.2, 1.0, 0.9, 0.1, 0.0]],
    }

    rise, decay = analysis._rise_and_decay(assay, 0.1, 0.9)

    assert rise == pytest.approx(0.1375)
    assert decay == pytest.approx(0.1)


def test_recovery_fit_preserves_separate_components():
    duration = np.array([0.0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200])
    expected = (0.53, 0.6, 35.0)
    values = 1 - expected[0] * np.exp(-duration / expected[1]) - (1 - expected[0]) * np.exp(-duration / expected[2])

    observed = analysis._fit_recovery(duration, values)

    assert observed == pytest.approx(expected, rel=1e-6)


def test_create_only_output_refuses_overwrite(tmp_path):
    path = tmp_path / "analysis.json"
    path.write_text("occupied")

    with pytest.raises(analysis.FastChannelClampAnalysisError, match="overwrite"):
        analysis._write_new(path, {"value": 1})
