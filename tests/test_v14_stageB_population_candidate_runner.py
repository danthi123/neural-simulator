from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from sim import kv3_source_models as kv3
from sim import sodium_source_models as sodium
from tools import v14_stageB_population_candidate_runner as runner
from tools import v14_stageB_population_scorer as scorer
from tools import v14_stageB_population_targets as targets


ROOT = Path(__file__).resolve().parents[1]
COMMAND_PATH = ROOT / "research/specs/v14_snr_stageB_fast_channel_clamp_execution_v1.json"


FAMILY_X = {
    "fast_na_activation": -30.0,
    "fast_na_deactivation": -40.0,
    "fast_na_recovery": 2.0,
    "fast_na_steady_state_inactivation": -60.0,
    "kv3_activation": 0.0,
    "kv3_deactivation": -50.0,
    "kv3_steady_state_inactivation": -50.0,
}


def _target(family: str, command_id: str) -> dict:
    x = FAMILY_X[family]
    return {
        "target_id": f"{family}:{command_id}",
        "target_family": family,
        "asset_id": "source-figure",
        "panel": "P",
        "series_identity": "SNr_GABA",
        "command_id": command_id,
        "partition": "calibration",
        "x_quantity": "recovery_interval" if family == "fast_na_recovery" else "command_voltage",
        "x_unit": "ms" if family == "fast_na_recovery" else "mV",
        "y_quantity": "deactivation_time_constant" if "deactivation" in family else "normalized_current",
        "y_unit": "ms" if "deactivation" in family else "normalized",
        "sample_size": 5,
        "measurement_limitation": None,
        "status": "available",
        "unavailable_reason": None,
        "x": {
            "median": x,
            "standard_uncertainty": 0.0,
            "q025": x,
            "q975": x,
            "authority": "published_command",
        },
        "y": {
            "median": 0.5,
            "standard_uncertainty": 0.05,
            "q025": 0.4,
            "q975": 0.6,
        },
        "digitization_uncertainty": {"between_extractor_component": 0.01},
        "biological_error": {"status": "unavailable"},
    }


def _packet() -> dict:
    rows = [_target(family, "command_001") for family in sorted(FAMILY_X)]
    core = {
        "schema": targets.PACKET_SCHEMA,
        "scientific_verdict": None,
        "optimization_command": None,
        "optimization_allowed": False,
        "status": "sealed_source_measurements",
        "partition": "calibration",
        "proposal_visible": True,
        "measurement_protocol": {"path": "spec.json", "sha256": "1" * 64},
        "partition_protocol": {"path": "partition.json", "sha256": "2" * 64},
        "evidence": [],
        "unavailable_panels": [],
        "targets": rows,
    }
    return {**core, "sha256": targets._digest(core)}


def _commands():
    return runner.load_commands(COMMAND_PATH, hashlib.sha256(COMMAND_PATH.read_bytes()).hexdigest())


def _parameters():
    return (
        [dict(sodium.source_parameters(sodium.KHALIQ_RAMAN_13_STATE))],
        [dict(kv3.source_parameters(kv3.DESAI_2008_CONTROL))],
    )


def test_one_shared_sodium_and_kv3_vector_predict_every_population_family():
    sodium_parameters, kv3_parameters = _parameters()
    packet = _packet()
    target_ids, matrix = runner.predict_population_targets(
        packet,
        sodium_model_id=sodium.KHALIQ_RAMAN_13_STATE,
        sodium_parameters=sodium_parameters,
        sodium_temperature_c=None,
        kv3_model_id=kv3.DESAI_2008_CONTROL,
        kv3_parameters=kv3_parameters,
        kv3_temperature_c=None,
        command_authority=_commands(),
        xp=np,
    )
    assert target_ids == [row["target_id"] for row in packet["targets"]]
    assert matrix.shape == (1, 7)
    assert np.all(np.isfinite(matrix))
    for family in FAMILY_X:
        value = matrix[0, target_ids.index(f"{family}:command_001")]
        if "deactivation" in family:
            assert value > 0.0
        else:
            assert 0.0 <= value <= 1.0 + 1e-12

    observations = runner.build_candidate_observations(
        packet, [{"id": "candidate-001", "sha256": "a" * 64}], target_ids, matrix
    )
    scored = scorer.score_population_calibration(packet, observations[0])
    assert scored["candidate"]["id"] == "candidate-001"
    assert {row["target_family"] for row in scored["target_families"]} == set(FAMILY_X)


def test_labro_cannot_gain_an_inactivation_state_through_the_runner():
    sodium_parameters, _ = _parameters()
    with pytest.raises(runner.PopulationCandidateRunnerError, match="cannot predict"):
        runner.predict_population_targets(
            _packet(),
            sodium_model_id=sodium.KHALIQ_RAMAN_13_STATE,
            sodium_parameters=sodium_parameters,
            sodium_temperature_c=None,
            kv3_model_id=kv3.LABRO_2015,
            kv3_parameters=[dict(kv3.source_parameters(kv3.LABRO_2015))],
            kv3_temperature_c=np.array([22.5]),
            command_authority=_commands(),
            xp=np,
        )


def test_runner_rejects_withheld_packets_and_unpaired_candidate_counts():
    sodium_parameters, kv3_parameters = _parameters()
    packet = _packet()
    withheld = dict(packet)
    withheld["partition"] = "held_out"
    withheld["proposal_visible"] = False
    withheld["sha256"] = targets._digest(
        {key: value for key, value in withheld.items() if key != "sha256"}
    )
    with pytest.raises(runner.PopulationCandidateRunnerError, match="calibration only"):
        runner.predict_population_targets(
            withheld,
            sodium_model_id=sodium.KHALIQ_RAMAN_13_STATE,
            sodium_parameters=sodium_parameters,
            sodium_temperature_c=None,
            kv3_model_id=kv3.DESAI_2008_CONTROL,
            kv3_parameters=kv3_parameters,
            kv3_temperature_c=None,
            command_authority=_commands(),
            xp=np,
        )

    with pytest.raises(runner.PopulationCandidateRunnerError, match="counts differ"):
        runner.predict_population_targets(
            packet,
            sodium_model_id=sodium.KHALIQ_RAMAN_13_STATE,
            sodium_parameters=sodium_parameters * 2,
            sodium_temperature_c=None,
            kv3_model_id=kv3.DESAI_2008_CONTROL,
            kv3_parameters=kv3_parameters,
            kv3_temperature_c=None,
            command_authority=_commands(),
            xp=np,
        )


def test_optional_cupy_predictions_match_numpy():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    sodium_parameters, kv3_parameters = _parameters()
    kwargs = dict(
        target_packet=_packet(),
        sodium_model_id=sodium.KHALIQ_RAMAN_13_STATE,
        sodium_parameters=sodium_parameters,
        sodium_temperature_c=None,
        kv3_model_id=kv3.DESAI_2008_CONTROL,
        kv3_parameters=kv3_parameters,
        kv3_temperature_c=None,
        command_authority=_commands(),
    )
    expected_ids, expected = runner.predict_population_targets(**kwargs, xp=np)
    actual_ids, actual = runner.predict_population_targets(**kwargs, xp=cupy)
    assert actual_ids == expected_ids
    np.testing.assert_allclose(cupy.asnumpy(actual), expected, rtol=5e-8, atol=5e-10)

