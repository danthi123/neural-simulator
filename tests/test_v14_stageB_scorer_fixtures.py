import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from tools.v14_stageB_scorer_fixtures import (
    StageBFixtureError,
    score_observation,
    validate_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "research/fixtures/v14_snr_stageB_scorer_fixtures.json"
TARGET_PATH = ROOT / "research/specs/v14_snr_stageB_target_packet.json"


def _packet():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _fixtures():
    return {fixture["id"]: fixture for fixture in _packet()["fixtures"]}


def _observation(fixture, value=None):
    observation = {
        "cohort": fixture["cohort"],
        "pathway": fixture["pathway"],
        "metric": fixture["metric"],
        "units": fixture["units"],
    }
    if value is not None:
        observation["value"] = value
    return observation


def test_fixture_packet_is_bound_to_the_source_target_packet():
    packet = _packet()
    source = packet["source_target_packet"]
    assert source["path"] == "research/specs/v14_snr_stageB_target_packet.json"
    assert hashlib.sha256(TARGET_PATH.read_bytes()).hexdigest() == source["sha256"]


def test_source_derived_fixtures_match_target_packet_and_score_boundaries():
    targets = {
        target["id"]: target
        for target in json.loads(TARGET_PATH.read_text(encoding="utf-8"))["accepted_targets"]
    }
    fixtures = _fixtures()
    expected = {
        "adult-autonomous-rate-observed-range": ("adult-autonomous-rate-support", 9.6, 27.8),
        "direct-pathway-unitary-peak-observed-range": ("adult-inhibitory-conductance-support", 0.35, 6.4),
        "pallidonigral-unitary-peak-observed-range": ("adult-pallidonigral-unitary-support", 2.4, 25.1),
        "pallidonigral-barrage-peak-selected-range": ("adult-pallidonigral-depressed-barrage", 1.6, 2.4),
    }
    for fixture_id, (target_id, low, high) in expected.items():
        fixture = fixtures[fixture_id]
        assert fixture["evidence"]["interval_provenance"] == "source-derived"
        assert validate_fixture(fixture)["interval"] == {"low": low, "high": high}
        assert targets[target_id]["source"] == fixture["source_id"]
        assert score_observation(fixture, _observation(fixture, low))["passed"] is True
        assert score_observation(fixture, _observation(fixture, high))["passed"] is True
        assert score_observation(fixture, _observation(fixture, high + 100))["passed"] is False


def test_k_resolved_nalcn_fixture_recomputes_its_model_derived_interval():
    packet = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    targets = {target["id"]: target for target in packet["accepted_targets"]}
    assert "nalcn-lesion-ratio" not in targets

    target = targets["nalcn-lesion-ratio-4mM"]
    fixture = _fixtures()["nalcn-lesion-ratio-4mM-model-derived"]
    measurements = fixture["evidence"]["source_measurements"]
    assert measurements == {
        "bath_KCl_mM": 4.0,
        "intact_mean_hz": 30.2,
        "intact_sem_hz": 3.9,
        "intact_n": 18,
        "lesion_mean_hz": 16.8,
        "lesion_sem_hz": 2.0,
        "lesion_n": 13,
        "confidence_multiplier": 1.96,
    }

    ratio = measurements["lesion_mean_hz"] / measurements["intact_mean_hz"]
    ratio_se = ratio * math.sqrt(
        (measurements["lesion_sem_hz"] / measurements["lesion_mean_hz"]) ** 2
        + (measurements["intact_sem_hz"] / measurements["intact_mean_hz"]) ** 2
    )
    expected_interval = {
        "low": ratio - measurements["confidence_multiplier"] * ratio_se,
        "high": ratio + measurements["confidence_multiplier"] * ratio_se,
    }

    assert target["value"]["lesion_over_intact"] == pytest.approx(ratio)
    assert target["model_derived_interval"]["low"] == pytest.approx(expected_interval["low"])
    assert target["model_derived_interval"]["high"] == pytest.approx(expected_interval["high"])
    assert fixture["interval"]["low"] == pytest.approx(expected_interval["low"])
    assert fixture["interval"]["high"] == pytest.approx(expected_interval["high"])
    assert fixture["evidence"]["interval_provenance"] == "model-derived"
    assert "first-order delta method" in fixture["evidence"]["derivation_method"]
    assert "Figure 6-figure supplement 1A" in fixture["evidence"]["source_locator"]
    assert validate_fixture(fixture)["interval"] == pytest.approx(expected_interval)
    assert score_observation(fixture, _observation(fixture, ratio))["passed"] is True
    assert score_observation(fixture, _observation(fixture, expected_interval["high"] + 0.01))["passed"] is False


def test_direct_and_pallidal_pathways_cannot_be_cross_scored():
    fixtures = _fixtures()
    direct = fixtures["direct-pathway-unitary-peak-observed-range"]
    pallidal = fixtures["pallidonigral-unitary-peak-observed-range"]
    observation = _observation(direct, 3.0)
    observation["pathway"] = pallidal["pathway"]
    with pytest.raises(StageBFixtureError, match="pathway does not match"):
        score_observation(direct, observation)


def test_cohorts_cannot_be_cross_scored():
    fixture = _fixtures()["adult-autonomous-rate-observed-range"]
    observation = _observation(fixture, 13.6)
    observation["cohort"] = "juvenile mouse transferred causal constraint"
    with pytest.raises(StageBFixtureError, match="cohort does not match"):
        score_observation(fixture, observation)


@pytest.mark.parametrize("missing", ["temperature", "solution", "blockers"])
def test_missing_preparation_metadata_is_rejected(missing):
    fixture = copy.deepcopy(_fixtures()["adult-autonomous-rate-observed-range"])
    del fixture["evidence"]["preparation"][missing]
    with pytest.raises(StageBFixtureError, match=rf"preparation\.{missing} is required"):
        validate_fixture(fixture)


@pytest.mark.parametrize("field", ["uncertainty", "source_locator"])
def test_missing_uncertainty_or_locator_is_rejected(field):
    fixture = copy.deepcopy(_fixtures()["adult-autonomous-rate-observed-range"])
    fixture["evidence"][field] = ""
    with pytest.raises(StageBFixtureError, match=field):
        validate_fixture(fixture)


def test_model_derived_interval_requires_derivation_and_source_measurements():
    fixture = copy.deepcopy(_fixtures()["adult-autonomous-rate-observed-range"])
    fixture["evidence"]["interval_provenance"] = "model-derived"
    with pytest.raises(StageBFixtureError, match="derivation_method"):
        validate_fixture(fixture)
    fixture["evidence"]["derivation_method"] = "first-order independent-group propagation"
    with pytest.raises(StageBFixtureError, match="source_measurements"):
        validate_fixture(fixture)
    fixture["evidence"]["source_measurements"] = {"mean": 13.6, "sd": 5.7, "n": 11}
    result = score_observation(fixture, _observation(fixture, 13.6))
    assert result["interval_provenance"] == "model-derived"


def test_source_derived_interval_cannot_hide_model_derivation():
    fixture = copy.deepcopy(_fixtures()["adult-autonomous-rate-observed-range"])
    fixture["evidence"]["derivation_method"] = "unfiled calculation"
    with pytest.raises(StageBFixtureError, match="cannot carry model-derivation"):
        validate_fixture(fixture)


def test_lutas_figure5_remains_blocked_and_has_no_executable_target_or_fixture():
    packet = _packet()
    blocked = packet["blocked_unscorable_source_observation"]
    assert blocked["record_id"] == "juvenile-mouse-lutas-figure5-k-unresolved"
    assert blocked["bath_KCl_mM"] == "unresolved_2.5_or_4"
    assert blocked["evidence_label"] == "blocked-unscorable-transfer-evidence"
    assert blocked["scorable"] is False
    assert "no target_id, interval, or scorer fixture" in blocked["reason"]
    assert "target_id" not in blocked
    assert "interval" not in blocked
    assert all(fixture["target_id"] != "nalcn-lesion-ratio" for fixture in _fixtures().values())


def test_mcelvain_cv_point_target_has_no_acceptance_fixture():
    targets = {
        target["id"]: target
        for target in json.loads(TARGET_PATH.read_text(encoding="utf-8"))["accepted_targets"]
    }
    target = targets["young-adult-intrinsic-isi-cv-point"]
    assert target["acceptance_bound"] is None
    assert target["evidence_label"] == "separate-cohort-source-measured-point-target"
    assert all(fixture["target_id"] != target["id"] for fixture in _fixtures().values())


def test_non_significance_is_never_scored_as_equivalence():
    fixture = _fixtures()["hcn-baseline-non-significance-boundary"]
    result = score_observation(fixture, _observation(fixture))
    assert result == {
        "fixture_id": fixture["id"],
        "status": "not-scorable-as-equivalence",
        "passed": None,
        "interval_provenance": "not-an-interval",
    }

    completed = copy.deepcopy(fixture)
    completed["interval"] = {"low": -1.0, "high": 1.0}
    with pytest.raises(StageBFixtureError, match="cannot carry equivalence bounds"):
        validate_fixture(completed)
