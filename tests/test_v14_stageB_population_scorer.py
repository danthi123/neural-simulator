from __future__ import annotations

import copy

import pytest

from tools import v14_stageB_population_targets as targets
from tools.v14_stageB_population_scorer import (
    OBSERVATION_SCHEMA,
    PopulationScorerError,
    digest,
    score_population_calibration,
)


def _measurement(median: float, uncertainty: float) -> dict:
    return {
        "median": median,
        "standard_uncertainty": uncertainty,
        "q025": median - uncertainty,
        "q975": median + uncertainty,
    }


def _target(
    identifier: str,
    family: str,
    x: float,
    y: float,
    *,
    biological: object = None,
    published_command: bool = False,
) -> dict:
    x_measurement = _measurement(x, 0.0)
    if published_command:
        x_measurement["q025"] = x
        x_measurement["q975"] = x
        x_measurement["authority"] = "published_command"
    return {
        "target_id": identifier,
        "target_family": family,
        "asset_id": "source",
        "panel": "P",
        "series_identity": "filled",
        "command_id": identifier.rsplit(":", 1)[1],
        "partition": "calibration",
        "x_quantity": "command_voltage",
        "x_unit": "mV",
        "y_quantity": "normalized_conductance",
        "y_unit": "G/Gmax",
        "sample_size": 5,
        "measurement_limitation": None,
        "status": "available",
        "unavailable_reason": None,
        "x": x_measurement,
        "y": _measurement(y, 0.1),
        "digitization_uncertainty": {"between_extractor_component": None},
        "biological_error": biological if biological is not None else {"status": "unavailable"},
    }


def _unavailable(identifier: str, family: str) -> dict:
    row = _target(identifier, family, 0.0, 0.0)
    row.update({
        "status": "unavailable", "unavailable_reason": "marker_not_bounded",
        "x": None, "y": None, "digitization_uncertainty": None, "biological_error": None,
    })
    return row


def _packet() -> dict:
    body = {
        "schema": targets.PACKET_SCHEMA,
        "scientific_verdict": None,
        "optimization_command": None,
        "optimization_allowed": False,
        "status": "sealed_source_measurements",
        "partition": "calibration",
        "proposal_visible": True,
        "measurement_protocol": {"path": "specs/protocol.json", "sha256": "a" * 64},
        "partition_protocol": {"path": "specs/partition.json", "sha256": "b" * 64},
        "evidence": [],
        "targets": [
            _target(
                "kv3:command_001", "kv3", -40.0, 0.6,
                biological={
                    "status": "available", "kind": "standard_error",
                    "lower_endpoint_digitization": _measurement(0.45, 0.01),
                    "upper_endpoint_digitization": _measurement(0.75, 0.01),
                },
            ),
            _unavailable("kv3:command_002", "kv3"),
            _target("na:command_001", "na", -60.0, 0.2),
            _target("na:command_002", "na", -50.0, 0.4),
        ],
    }
    return {**body, "sha256": digest(body)}


def _observation(packet: dict, predictions: list[dict] | None = None) -> dict:
    body = {
        "schema": OBSERVATION_SCHEMA,
        "status": "completed",
        "target_packet": {"sha256": packet["sha256"]},
        "candidate": {"id": "candidate-1", "sha256": "c" * 64},
        "predictions": predictions if predictions is not None else [
            {"target_id": "kv3:command_001", "x": -40.0, "y": 0.9},
            {"target_id": "na:command_001", "x": -60.0, "y": 0.1},
            {"target_id": "na:command_002", "x": -50.0, "y": 0.5},
        ],
        "scientific_verdict": None,
        "optimization_allowed": False,
    }
    return {**body, "sha256": digest(body)}


def _redigest(document: dict) -> dict:
    document["sha256"] = digest({key: value for key, value in document.items() if key != "sha256"})
    return document


def test_scores_each_family_separately_and_is_order_independent() -> None:
    packet = _packet()
    observation = _observation(packet)
    forward = score_population_calibration(packet, observation)
    reverse = score_population_calibration(packet, _observation(packet, list(reversed(observation["predictions"]))))

    # The sealed source receipt differs when its input array order differs, but
    # the deterministic analysis rows and family summaries do not.
    assert forward["per_target"] == reverse["per_target"]
    assert forward["target_families"] == reverse["target_families"]
    assert forward["sha256"] == digest({key: value for key, value in forward.items() if key != "sha256"})
    assert forward["scientific_verdict"] is None
    assert forward["optimization_allowed"] is False
    assert [row["target_family"] for row in forward["target_families"]] == ["kv3", "na"]
    assert [row["count"] for row in forward["target_families"]] == [1, 2]
    assert "objective" not in forward
    kv3 = forward["per_target"][0]
    assert kv3["biological_standard_error"] == pytest.approx(0.15)
    assert kv3["combined_standard_uncertainty"] == pytest.approx((0.1 ** 2 + 0.15 ** 2) ** 0.5)
    na = next(row for row in forward["per_target"] if row["target_id"] == "na:command_001")
    assert na["combined_standard_uncertainty"] is None


def test_rejects_tampered_packet_and_withheld_packet_leakage() -> None:
    packet = _packet()
    tampered = copy.deepcopy(packet)
    tampered["targets"][0]["y"]["median"] = 0.3
    with pytest.raises(PopulationScorerError, match="self digest"):
        score_population_calibration(tampered, _observation(packet))

    for partition in ("validation", "held_out"):
        withheld = copy.deepcopy(packet)
        withheld["partition"] = partition
        withheld["proposal_visible"] = False
        _redigest(withheld)
        with pytest.raises(PopulationScorerError, match="proposal-visible calibration"):
            score_population_calibration(withheld, _observation(withheld))


@pytest.mark.parametrize("change, message", [
    (lambda rows: rows[:-1], "exactly cover"),
    (lambda rows: [*rows, {"target_id": "kv3:command_002", "x": 0.0, "y": 0.0}], "unavailable or extra"),
    (lambda rows: [*rows, dict(rows[0])], "duplicate"),
])
def test_rejects_missing_extra_and_duplicate_predictions(change, message: str) -> None:
    packet = _packet()
    observation = _observation(packet, change(_observation(packet)["predictions"]))
    with pytest.raises(PopulationScorerError, match=message):
        score_population_calibration(packet, observation)


def test_rejects_x_mismatch_and_nonfinite_prediction_values() -> None:
    packet = _packet()
    wrong_x = _observation(packet)
    wrong_x["predictions"][0]["x"] = -39.0
    _redigest(wrong_x)
    with pytest.raises(PopulationScorerError, match="exactly match"):
        score_population_calibration(packet, wrong_x)

    nonfinite = _observation(packet)
    nonfinite["predictions"][0]["y"] = float("inf")
    with pytest.raises(PopulationScorerError, match="canonical JSON"):
        _redigest(nonfinite)


def test_rejects_nonpositive_digitization_uncertainty_and_observation_packet_mismatch() -> None:
    packet = _packet()
    bad_uncertainty = copy.deepcopy(packet)
    bad_uncertainty["targets"][0]["y"]["standard_uncertainty"] = 0.0
    _redigest(bad_uncertainty)
    with pytest.raises(PopulationScorerError, match="must be positive"):
        score_population_calibration(bad_uncertainty, _observation(bad_uncertainty))

    observation = _observation(packet)
    observation["target_packet"]["sha256"] = "d" * 64
    _redigest(observation)
    with pytest.raises(PopulationScorerError, match="not bound"):
        score_population_calibration(packet, observation)


def test_accepts_exact_published_command_x_and_rejects_inexact_authority() -> None:
    packet = _packet()
    packet["targets"][0] = _target(
        "kv3:command_001",
        "kv3",
        -40.0,
        0.6,
        published_command=True,
    )
    _redigest(packet)
    observation = _observation(packet)
    result = score_population_calibration(packet, observation)
    assert result["per_target"][0]["x"] == -40.0

    inexact = copy.deepcopy(packet)
    inexact["targets"][0]["x"]["q975"] = -39.9
    _redigest(inexact)
    with pytest.raises(PopulationScorerError, match="must be exact"):
        score_population_calibration(inexact, _observation(inexact))
