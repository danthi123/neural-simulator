import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.v14_stageB_scorer import (
    StageBScorerError,
    _main,
    score_raw_observation_file,
    score_raw_observations,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RELATIVE = Path("research/fixtures/v14_snr_stageB_scorer_fixtures.json")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _conductance(fixture_id: str, peak: float) -> dict:
    return {
        "fixture_id": fixture_id,
        "raw": {
            "kind": "conductance_trace",
            "time_s": [0.0, 0.001, 0.002, 0.003, 0.004, 0.005],
            "conductance_nS": [0.0, 0.0, peak / 2.0, peak, peak / 4.0, 0.0],
            "time_unit": "s",
            "conductance_unit": "nS",
            "sample_interval_s": 0.001,
            "recording_start_s": 0.0,
            "burn_in_start_s": 0.0,
            "burn_in_end_s": 0.002,
            "window_start_s": 0.002,
            "window_end_s": 0.005,
        },
    }


def _document() -> dict:
    fixture_path = ROOT / FIXTURE_RELATIVE
    return {
        "schema": "v14-snr-stageB-raw-observations-v1",
        "adaptive_candidate": {
            "candidate_id": "candidate-a",
            "candidate_sha256": "a" * 64,
            "effective_parameters": {"snr_g_nalcn_max": 0.01},
        },
        "fixture_packet": {"path": FIXTURE_RELATIVE.as_posix(), "sha256": _digest(fixture_path)},
        "observations": [
            {
                "fixture_id": "adult-autonomous-rate-observed-range",
                "raw": {
                    "kind": "spike_train",
                    "spike_times_s": [0.11, 0.16, 0.21, 0.26, 0.31, 0.36, 0.41, 0.46],
                    "time_unit": "s",
                    "sample_interval_s": 0.001,
                    "recording_start_s": 0.0,
                    "recording_end_s": 0.6,
                    "burn_in_start_s": 0.0,
                    "burn_in_end_s": 0.1,
                    "window_start_s": 0.1,
                    "window_end_s": 0.5,
                },
            },
            {
                "fixture_id": "nalcn-lesion-ratio-4mM-model-derived",
                "raw": {
                    "kind": "paired_spike_rate_ratio",
                    "intact": {
                        "spike_times_s": [0.11 + 0.025 * index for index in range(12)],
                        "time_unit": "s", "sample_interval_s": 0.001,
                        "recording_start_s": 0.0, "recording_end_s": 0.6,
                        "burn_in_start_s": 0.0, "burn_in_end_s": 0.1,
                        "window_start_s": 0.1, "window_end_s": 0.5,
                    },
                    "lesion": {
                        "spike_times_s": [0.11 + 0.05 * index for index in range(7)],
                        "time_unit": "s", "sample_interval_s": 0.001,
                        "recording_start_s": 0.0, "recording_end_s": 0.6,
                        "burn_in_start_s": 0.0, "burn_in_end_s": 0.1,
                        "window_start_s": 0.1, "window_end_s": 0.5,
                    },
                },
            },
            _conductance("direct-pathway-unitary-peak-observed-range", 1.0),
            _conductance("pallidonigral-unitary-peak-observed-range", 5.0),
            _conductance("pallidonigral-barrage-peak-selected-range", 2.0),
        ],
    }


def test_raw_scorer_recomputes_all_bounded_fixtures_and_preserves_boundary():
    result = score_raw_observations(_document(), root=ROOT)
    assert result["process_status"] == "completed"
    assert result["scientific_verdict"] == "GO"
    assert result["all_bounded_fixtures_passed"] is True
    assert result["adaptive_candidate"] == _document()["adaptive_candidate"]
    assert result["unscored_boundaries"] == ["hcn-baseline-non-significance-boundary"]
    by_id = {item["fixture_id"]: item for item in result["results"]}
    rate = by_id["adult-autonomous-rate-observed-range"]
    assert rate["value"] == 20.0
    assert rate["raw_metrics"]["spike_count"] == 8
    ratio = by_id["nalcn-lesion-ratio-4mM-model-derived"]
    assert ratio["value"] == pytest.approx(7 / 12)
    assert ratio["interval_provenance"] == "model-derived"
    assert ratio["raw_metrics"]["persistent_lesion_firing"] is True


def test_out_of_band_trace_is_valid_scientific_failure_not_scorer_error():
    document = _document()
    document["observations"][0]["raw"]["spike_times_s"] = [0.2]
    result = score_raw_observations(document, root=ROOT)
    assert result["process_status"] == "completed"
    assert result["scientific_verdict"] == "NO_GO"
    assert result["all_bounded_fixtures_passed"] is False
    failed = [item for item in result["results"] if item["passed"] is False]
    assert [item["fixture_id"] for item in failed] == ["adult-autonomous-rate-observed-range"]


def test_scorer_rejects_hidden_burn_in_instead_of_trusting_claimed_metrics():
    document = _document()
    raw = document["observations"][0]["raw"]
    raw["recording_start_s"] = -0.1
    raw["claimed_firing_rate_hz"] = 20.0
    with pytest.raises(StageBScorerError, match="invalid:.*burn_in_start_s"):
        score_raw_observations(document, root=ROOT)


def test_scorer_rejects_missing_duplicate_and_unknown_fixtures():
    missing = _document()
    missing["observations"].pop()
    with pytest.raises(StageBScorerError, match="do not cover every bounded fixture"):
        score_raw_observations(missing, root=ROOT)

    duplicate = _document()
    duplicate["observations"].append(copy.deepcopy(duplicate["observations"][0]))
    with pytest.raises(StageBScorerError, match="unknown or duplicate"):
        score_raw_observations(duplicate, root=ROOT)


def test_ratio_requires_matched_protocols_and_zero_firing_is_scientific_no_go():
    mismatch = _document()
    ratio_raw = mismatch["observations"][1]["raw"]
    ratio_raw["lesion"]["window_end_s"] = 0.4
    with pytest.raises(StageBScorerError, match="protocols must match exactly"):
        score_raw_observations(mismatch, root=ROOT)

    silent = _document()
    silent["observations"][1]["raw"]["lesion"]["spike_times_s"] = []
    result = score_raw_observations(silent, root=ROOT)
    assert result["scientific_verdict"] == "NO_GO"
    ratio = next(item for item in result["results"] if item["fixture_id"].startswith("nalcn"))
    assert ratio["raw_metrics"]["persistent_lesion_firing"] is False


def test_scorer_rejects_fixture_digest_tampering(tmp_path: Path):
    document = _document()
    document["fixture_packet"]["sha256"] = "0" * 64
    with pytest.raises(StageBScorerError, match="fixture packet digest does not match"):
        score_raw_observations(document, root=ROOT)


def test_scorer_rejects_malformed_candidate_echo():
    document = _document()
    document["adaptive_candidate"]["candidate_sha256"] = "not-a-digest"
    with pytest.raises(StageBScorerError, match="adaptive_candidate is malformed"):
        score_raw_observations(document, root=ROOT)


def test_file_boundary_writes_completed_no_go_but_no_infrastructure_result(tmp_path: Path):
    valid_input = tmp_path / "valid.json"
    valid_output = tmp_path / "valid-score.json"
    document = _document()
    document["observations"][0]["raw"]["spike_times_s"] = [0.2]
    valid_input.write_text(json.dumps(document), encoding="utf-8")
    result = score_raw_observation_file(valid_input, valid_output, root=ROOT)
    assert result["scientific_verdict"] == "NO_GO"
    assert json.loads(valid_output.read_text())["process_status"] == "completed"

    invalid_input = tmp_path / "invalid.json"
    invalid_output = tmp_path / "invalid-score.json"
    document["fixture_packet"]["sha256"] = "0" * 64
    invalid_input.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(StageBScorerError, match="digest does not match"):
        score_raw_observation_file(invalid_input, invalid_output, root=ROOT)
    assert not invalid_output.exists()


def test_cli_exits_nonzero_without_result_on_infrastructure_failure(tmp_path: Path):
    source = tmp_path / "bad.json"
    output = tmp_path / "score.json"
    source.write_text("not JSON", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        _main(["--input", str(source), "--output", str(output), "--root", str(ROOT)])
    assert raised.value.code == 2
    assert not output.exists()
