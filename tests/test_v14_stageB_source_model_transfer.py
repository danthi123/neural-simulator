from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.runners import v14_stageB_source_model_transfer as runner
from sim import kv3_source_models as kv3
from sim import sodium_source_models as sodium


def _bound_file(root: Path, relative: str, content: bytes) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": relative, "sha256": hashlib.sha256(content).hexdigest()}


def _commands() -> dict:
    return {
        "schema": "v14-snr-stageB-fast-channel-clamp-execution-v1",
        "numeric_contract": {"clamp_sample_interval_ms": 0.005},
        "commands": {
            "fast_na_activation": {
                "hold_mV": -100.0,
                "test_mV": [-60.0, 0.0, 30.0],
                "test_duration_ms": 0.01,
            },
            "fast_na_inactivation": {
                "hold_mV": -100.0,
                "prepulse_mV": [-100.0, -40.0],
                "prepulse_duration_ms": 0.01,
                "test_mV": 0.0,
                "test_duration_ms": 0.01,
            },
            "fast_na_recovery": {
                "hold_mV": -90.0,
                "recovery_prepulse_mV": -120.0,
                "recovery_prepulse_duration_ms": 0.01,
                "inactivation_mV": 0.0,
                "inactivation_duration_ms": 0.01,
                "recovery_mV": -120.0,
                "recovery_duration_ms": [0.0, 0.01, 0.02],
                "test_mV": 0.0,
                "test_duration_ms": 0.01,
            },
            "fast_na_deactivation": {
                "hold_mV": -90.0,
                "prepulse_mV": -120.0,
                "prepulse_duration_ms": 0.01,
                "activation_mV": 0.0,
                "activation_duration_ms": 0.01,
                "test_mV": [-90.0, -40.0],
                "test_duration_ms": 0.01,
            },
            "kv3_activation": {
                "hold_mV": -100.0,
                "test_mV": [-40.0, 20.0, 40.0],
                "test_duration_ms": 0.01,
            },
            "kv3_inactivation": {
                "hold_mV": -90.0,
                "prepulse_mV": [-90.0, -20.0],
                "prepulse_duration_ms": 0.01,
                "test_mV": 50.0,
                "test_duration_ms": 0.01,
            },
            "kv3_deactivation": {
                "hold_mV": -90.0,
                "activation_mV": 20.0,
                "activation_duration_ms": 0.01,
                "test_mV": [-70.0, -30.0],
                "test_duration_ms": 0.01,
            },
        },
    }


def _model(
    model_id: str,
    conditions: list[dict],
    *,
    unavailable_assays: list[str] | None = None,
) -> dict:
    return {
        "model_id": model_id,
        "model_fingerprint": hashlib.sha256(model_id.encode("ascii")).hexdigest(),
        "calibration_allowed": False,
        "conductance_fitting_allowed": False,
        "conditions": conditions,
        "unavailable_assays": unavailable_assays or [],
        "fixed_voltage_solver": "source-faithful-test-solver",
    }


def _spec(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    original_runner = Path(runner.__file__).read_bytes()
    original_sodium = Path(sodium.__file__).read_bytes()
    original_kv3 = Path(kv3.__file__).read_bytes()
    implementation = {
        "sodium_models": _bound_file(root, "impl/sodium_source_models.py", original_sodium),
        "kv3_models": _bound_file(root, "impl/kv3_source_models.py", original_kv3),
        "runner": _bound_file(root, "impl/source_model_transfer.py", original_runner),
    }
    monkeypatch.setattr(sodium, "__file__", str(root / implementation["sodium_models"]["path"]))
    monkeypatch.setattr(kv3, "__file__", str(root / implementation["kv3_models"]["path"]))
    monkeypatch.setattr(runner, "__file__", str(root / implementation["runner"]["path"]))

    command_bytes = json.dumps(_commands(), sort_keys=True).encode("ascii")
    command_authority = _bound_file(root, "authority/commands.json", command_bytes)
    target_authority = _bound_file(root, "authority/target.json", b"target")
    research_gate = _bound_file(root, "authority/research.md", b"research")
    correction = _bound_file(root, "authority/correction.md", b"correction")
    estimator = _bound_file(root, "impl/estimator.py", b"estimator")
    analyzer = _bound_file(root, "impl/analyzer.py", b"analyzer")
    models = [
        _model(
            sodium.KHALIQ_RAMAN_13_STATE,
            [{"condition_id": "graph_stationary", "temperature_c": None}],
        ),
        _model(
            sodium.BALBI_NAV16_SIX_STATE,
            [{"condition_id": "source_22c", "temperature_c": 22.0}],
        ),
        _model(
            kv3.LABRO_2015,
            [
                {"condition_id": "room_20c", "temperature_c": 20.0},
                {"condition_id": "room_22p5c", "temperature_c": 22.5},
                {"condition_id": "room_25c", "temperature_c": 25.0},
            ],
            unavailable_assays=["kv3_inactivation"],
        ),
        _model(
            kv3.DESAI_2008_CONTROL,
            [{"condition_id": "no_temperature", "temperature_c": None}],
        ),
    ]
    matrix = []
    for model in models:
        for condition in model["conditions"]:
            stem = f'{model["model_id"]}-{condition["condition_id"]}'
            matrix.extend(
                [
                    {
                        "model_id": model["model_id"],
                        "condition_id": condition["condition_id"],
                        "backend": "numpy",
                        "device": "cpu",
                        "authority": "cpu_reference",
                        "output": f"raw/{stem}-numpy.json",
                    },
                    {
                        "model_id": model["model_id"],
                        "condition_id": condition["condition_id"],
                        "backend": "cupy",
                        "device": "cuda:0",
                        "authority": "gpu_parity_only",
                        "output": f"raw/{stem}-cupy.json",
                    },
                ]
            )
    return {
        "schema": runner.SCHEMA,
        "status": "preregistered_not_executed",
        "scientific_verdict": None,
        "candidate_calibration_allowed": False,
        "hybridization_allowed": False,
        "stage2_integration_allowed": False,
        "command_authority": command_authority,
        "target_authority": target_authority,
        "research_gate": research_gate,
        "khaliq_initialization_correction": correction,
        "analysis_contract": {
            "analyzer_authority": analyzer,
            "estimator_authority": estimator,
            "cpu_gpu_parity": {"rtol": 5e-8, "atol": 5e-10, "pointwise": True},
            "compensation_allowed": False,
            "stage2_verdict_allowed": False,
        },
        "implementation": implementation,
        "models": models,
        "execution_matrix": matrix,
    }


def _write_spec(root: Path, document: dict) -> tuple[Path, str]:
    path = root / "execution.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="ascii")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _numpy_job(document: dict, model_id: str, condition_id: str) -> dict:
    return next(
        row
        for row in document["execution_matrix"]
        if (row["model_id"], row["condition_id"], row["backend"])
        == (model_id, condition_id, "numpy")
    )


def test_temporary_contract_authenticates_exact_six_condition_matrix(tmp_path, monkeypatch):
    document = _spec(tmp_path, monkeypatch)
    path, digest = _write_spec(tmp_path, document)

    loaded, binding = runner.load_spec(path, digest, repository_root=tmp_path)

    assert binding == {"path": "execution.json", "sha256": digest}
    assert len(loaded["execution_matrix"]) == 12
    assert {
        (row["model_id"], row["condition_id"], row["backend"])
        for row in loaded["execution_matrix"]
    } == {
        (model["model_id"], condition["condition_id"], backend)
        for model in loaded["models"]
        for condition in model["conditions"]
        for backend in ("numpy", "cupy")
    }


@pytest.mark.parametrize(
    ("model_id", "condition_id", "expected_assays", "expected_shapes"),
    [
        (
            sodium.KHALIQ_RAMAN_13_STATE,
            "graph_stationary",
            {
                "fast_na_activation",
                "fast_na_inactivation",
                "fast_na_composite_zero",
                "fast_na_recovery",
                "fast_na_deactivation",
            },
            {"fast_na_activation": (3, 2), "fast_na_recovery": (3, 2)},
        ),
        (
            sodium.BALBI_NAV16_SIX_STATE,
            "source_22c",
            {
                "fast_na_activation",
                "fast_na_inactivation",
                "fast_na_composite_zero",
                "fast_na_recovery",
                "fast_na_deactivation",
            },
            {"fast_na_activation": (3, 2), "fast_na_recovery": (3, 2)},
        ),
        (
            kv3.LABRO_2015,
            "room_22p5c",
            {"kv3_activation", "kv3_rise", "kv3_deactivation"},
            {"kv3_activation": (3, 2), "kv3_deactivation": (2, 2)},
        ),
        (
            kv3.DESAI_2008_CONTROL,
            "no_temperature",
            {"kv3_activation", "kv3_rise", "kv3_inactivation", "kv3_deactivation"},
            {"kv3_activation": (3, 2), "kv3_inactivation": (2, 2)},
        ),
    ],
)
def test_cpu_run_emits_analyzer_ready_source_assays(
    tmp_path,
    monkeypatch,
    model_id,
    condition_id,
    expected_assays,
    expected_shapes,
):
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    document = _spec(tmp_path, monkeypatch)
    path, digest = _write_spec(tmp_path, document)
    job = _numpy_job(document, model_id, condition_id)
    backend_calls = []
    monkeypatch.setattr(
        runner.lab,
        "assert_backend",
        lambda expected, note="": backend_calls.append((expected, note)) or expected,
    )

    result = runner.run(
        spec_path=path,
        spec_sha256=digest,
        model_id=model_id,
        condition_id=condition_id,
        backend="numpy",
        output_path=job["output"],
        repository_root=tmp_path,
    )

    assert backend_calls == [("numpy", f"Stage B source-model transfer: {model_id}/{condition_id}")]
    assert result["schema"] == runner.OUTPUT_SCHEMA
    assert result["backend"] == "numpy"
    assert result["device"] == "cpu"
    assert result["authority"] == "cpu_reference"
    assert result["dtype"] == "float64"
    assert result["sample_interval_ms"] == 0.005
    assert set(result["assays"]) == expected_assays
    assert result["scientific_verdict"] is None
    assert result["analysis_status"] == "raw_unanalyzed"
    assert result["candidate_calibration_allowed"] is False
    assert result["conductance_fitting_allowed"] is False
    assert result["execution"]["per_time_step_host_loop"] is False
    assert result["execution"]["host_transfer_boundary"] == "final_serialization_only"
    assert result["execution"]["segment_operation_count"] > 0
    assert result["sha256"] == runner._semantic_digest(result)
    assert json.loads((tmp_path / job["output"]).read_text(encoding="ascii")) == result

    current_key = "normalized_na_current" if model_id in sodium.MODEL_METADATA else "normalized_kv3_current"
    for assay_name, shape in expected_shapes.items():
        values = np.asarray(result["assays"][assay_name][current_key])
        assert values.shape == shape
        assert np.isfinite(values).all()
    for assay in result["assays"].values():
        assert assay["elapsed_ms"] == [0.005, 0.01]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("digest", "digest mismatch"),
        ("calibration", "calibration is not forbidden"),
        ("matrix_count", "12 jobs"),
        ("duplicate_identity", "duplicates a job"),
        ("duplicate_output", "duplicates an output"),
        ("backend", "backend is invalid"),
        ("authority", "authority is invalid"),
    ],
)
def test_contract_and_matrix_mutations_fail_closed(tmp_path, monkeypatch, mutation, message):
    document = _spec(tmp_path, monkeypatch)
    path, digest = _write_spec(tmp_path, document)
    if mutation == "digest":
        digest = "f" * 64
    elif mutation == "calibration":
        document["candidate_calibration_allowed"] = True
        path, digest = _write_spec(tmp_path, document)
    elif mutation == "matrix_count":
        document["execution_matrix"].pop()
        path, digest = _write_spec(tmp_path, document)
    elif mutation == "duplicate_identity":
        document["execution_matrix"][1].update(document["execution_matrix"][0])
        document["execution_matrix"][1]["output"] = "raw/still-unique.json"
        path, digest = _write_spec(tmp_path, document)
    elif mutation == "duplicate_output":
        document["execution_matrix"][1]["output"] = document["execution_matrix"][0]["output"]
        path, digest = _write_spec(tmp_path, document)
    elif mutation == "backend":
        document["execution_matrix"][0]["backend"] = "jax"
        path, digest = _write_spec(tmp_path, document)
    elif mutation == "authority":
        document["execution_matrix"][0]["authority"] = "gpu_parity_only"
        path, digest = _write_spec(tmp_path, document)

    with pytest.raises(runner.SourceModelTransferError, match=message):
        runner.load_spec(path, digest, repository_root=tmp_path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("unknown_model", "model is absent"),
        ("unknown_condition", "condition is absent"),
        ("wrong_output", "output differs"),
        ("existing_output", "overwrite"),
    ],
)
def test_run_rejects_unsealed_or_existing_targets_before_backend_work(
    tmp_path, monkeypatch, change, message
):
    document = _spec(tmp_path, monkeypatch)
    path, digest = _write_spec(tmp_path, document)
    model_id = sodium.BALBI_NAV16_SIX_STATE
    condition_id = "source_22c"
    output = tmp_path / _numpy_job(document, model_id, condition_id)["output"]
    if change == "unknown_model":
        model_id = "not_sealed"
    elif change == "unknown_condition":
        condition_id = "not_sealed"
    elif change == "wrong_output":
        output = tmp_path / "raw/wrong.json"
    else:
        output.parent.mkdir(parents=True)
        output.write_text("banked", encoding="ascii")
    monkeypatch.setattr(
        runner.lab,
        "assert_backend",
        lambda *args, **kwargs: pytest.fail("backend touched before rejection"),
    )

    with pytest.raises(runner.SourceModelTransferError, match=message):
        runner.run(
            spec_path=path,
            spec_sha256=digest,
            model_id=model_id,
            condition_id=condition_id,
            backend="numpy",
            output_path=output,
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("duration", "interval", "message"),
    [(0.0, 0.005, "integral sample count"), (0.011, 0.005, "integral sample count")],
)
def test_filed_times_rejects_empty_or_fractional_commands(duration, interval, message):
    with pytest.raises(runner.SourceModelTransferError, match=message):
        runner._filed_times(duration, interval)
