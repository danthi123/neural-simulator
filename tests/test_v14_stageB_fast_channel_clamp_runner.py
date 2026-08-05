from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.runners import v14_stageB_fast_channel_clamp as runner
from sim.snr_structural_successor import EVIDENCE_CLASSES
from sim.snr_structural_successor import PROTOCOL_PATH as REAL_PROTOCOL_PATH
from sim.snr_structural_successor import PROTOCOL_SHA256 as REAL_PROTOCOL_SHA256


@pytest.fixture(autouse=True)
def _restore_production_protocol_binding():
    yield
    runner.PROTOCOL_PATH = REAL_PROTOCOL_PATH
    runner.PROTOCOL_SHA256 = REAL_PROTOCOL_SHA256


def _bound_file(root: Path, relative: str, content: bytes) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": relative, "sha256": hashlib.sha256(content).hexdigest()}


def _spec(root: Path, output_path: str = "raw/clamp.json") -> dict:
    protocol = _bound_file(root, "protocol.json", b"bound protocol\n")
    runner.PROTOCOL_PATH = protocol["path"]
    runner.PROTOCOL_SHA256 = protocol["sha256"]
    document = {
        "schema": runner.EXECUTION_SCHEMA,
        "execution_id": "fixture",
        "status": "preregistered_not_executed",
        "scientific_verdict": None,
        "candidate_calibration_allowed": False,
        "conductance_scale_fitting_allowed": False,
        "whole_cell_promotion_allowed": False,
        "protocol": protocol,
        "implementation": {
            "parameter_module": _bound_file(root, "impl/parameters.py", b"parameters\n"),
            "production_kernel_module": _bound_file(root, "impl/kernels.py", b"kernel\n"),
            "runner": _bound_file(root, "impl/runner.py", b"runner\n"),
        },
        "numeric_contract": {
            "state_dtype": "float32",
            "clamp_sample_interval_ms": 5.0,
            "conductance_scale": 1.0,
            "sodium_reversal_mV": 50.0,
            "potassium_reversal_mV": -90.0,
        },
        "commands": {
            "fast_na_activation": {
                "hold_mV": -100, "test_duration_ms": 20,
                "test_mV": list(range(-80, 31, 5)),
            },
            "fast_na_inactivation": {
                "hold_mV": -100, "prepulse_duration_ms": 50,
                "prepulse_mV": list(range(-120, -19, 10)),
                "test_mV": 0, "test_duration_ms": 20,
            },
            "fast_na_recovery": {
                "hold_mV": -90, "recovery_prepulse_mV": -120,
                "recovery_prepulse_duration_ms": 50,
                "inactivation_mV": 0, "inactivation_duration_ms": 300,
                "recovery_mV": -120,
                "recovery_duration_ms": list(runner.RECOVERY_LADDER_V2_MS),
                "test_mV": 0, "test_duration_ms": 20,
                "duration_evidence": "project_operational_sampling_not_source_reported",
            },
            "fast_na_deactivation": {
                "hold_mV": -90, "prepulse_mV": -120, "prepulse_duration_ms": 50,
                "activation_mV": 0, "activation_duration_ms": 0.2,
                "test_mV": list(range(-100, -19, 10)), "test_duration_ms": 50,
            },
            "kv3_activation": {
                "hold_mV": -100, "test_duration_ms": 100,
                "test_mV": list(range(-80, 51, 10)),
            },
            "kv3_inactivation": {
                "hold_mV": -90, "prepulse_duration_ms": 10_000,
                "prepulse_mV": list(range(-110, 1, 10)),
                "test_mV": 50, "test_duration_ms": 100,
            },
            "kv3_deactivation": {
                "hold_mV": -90, "activation_mV": 20, "activation_duration_ms": 100,
                "test_mV": [-30, -40, -50, -60, -70], "test_duration_ms": 50,
            },
        },
        "execution_matrix": [
            {"backend": "numpy", "device": "cpu", "output": output_path},
            {"backend": "cupy", "device": "cuda:0", "output": "raw/clamp-cupy.json"},
        ],
    }
    return document


def _write_spec(root: Path, document: dict) -> tuple[Path, str]:
    path = root / "execution.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_numpy_runner_emits_all_raw_curves_without_a_verdict(tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    spec, digest = _write_spec(tmp_path, _spec(tmp_path))
    calls = []
    monkeypatch.setattr(runner.lab, "assert_backend", lambda expected, note="": calls.append(expected) or expected)

    result = runner.run(
        spec_path=spec, spec_sha256=digest, output_path=tmp_path / "raw/clamp.json",
        backend="numpy", repository_root=tmp_path,
    )

    assert calls == ["numpy"]
    assert result["schema"] == runner.OUTPUT_SCHEMA
    assert result["execution_spec"] == {"path": "execution.json", "sha256": digest}
    assert result["protocol"]["sha256"] == runner.PROTOCOL_SHA256
    assert result["backend"] == "numpy"
    assert result["device"] == "cpu"
    assert result["dtype"] == "float32"
    assert result["sample_interval_ms"] == 5.0
    assert set(result["implementation"]) == {"parameter_module", "production_kernel_module", "runner"}
    assert result["scientific_verdict"] is None
    assert result["candidate_calibration_allowed"] is False
    assert result["analysis_status"] == "raw_unanalyzed"
    assert set(result["assays"]) == set(runner.ASSAY_NAMES)
    assert result["execution"]["per_time_step_host_loop"] is False
    assert result["execution"]["host_transfer_boundary"] == "final_serialization_only"
    assert result["execution"]["segment_launch_count"] == 26
    assert result["model_prior_boundaries"]["tau_between_voltage_endpoints"].endswith("model_prior")
    assert result["sha256"] == runner._semantic_digest(result)
    assert json.loads((tmp_path / "raw/clamp.json").read_text()) == result


def test_raw_shapes_and_normalized_ladders_are_analyzer_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    monkeypatch.setattr(runner.lab, "assert_backend", lambda expected, note="": expected)
    spec, digest = _write_spec(tmp_path, _spec(tmp_path))
    result = runner.run(
        spec_path=spec, spec_sha256=digest, output_path=tmp_path / "raw/clamp.json",
        backend="numpy", repository_root=tmp_path,
    )
    assays = result["assays"]
    assert np.asarray(assays["fast_na_activation"]["normalized_na_current"]).shape == (23, 4)
    assert np.asarray(assays["fast_na_inactivation"]["normalized_na_current"]).shape == (11, 4)
    assert np.asarray(assays["fast_na_recovery"]["normalized_na_current"]).shape == (12, 4)
    assert np.asarray(assays["fast_na_deactivation"]["normalized_na_current"]).shape == (9, 10)
    assert np.asarray(assays["kv3_activation"]["normalized_kv3_current"]).shape == (14, 20)
    assert np.asarray(assays["kv3_inactivation"]["normalized_kv3_current"]).shape == (12, 20)
    assert np.asarray(assays["kv3_deactivation"]["normalized_kv3_current"]).shape == (5, 10)
    assert assays["fast_na_activation"]["elapsed_ms"] == [5.0, 10.0, 15.0, 20.0]
    assert assays["fast_na_recovery"]["command"]["recovery_durations_ms"] == list(
        runner.RECOVERY_LADDER_V2_MS
    )
    for name, field in (
        ("fast_na_activation", "ladder_normalized_peak_conductance"),
        ("fast_na_inactivation", "ladder_normalized_peak_test_current"),
        ("fast_na_recovery", "ladder_normalized_peak_test_current"),
        ("kv3_activation", "ladder_normalized_peak_conductance"),
        ("kv3_inactivation", "ladder_normalized_peak_test_current"),
    ):
        values = np.asarray(assays[name][field])
        assert np.isfinite(values).all()
        assert np.isclose(values.max(), 1.0)


@pytest.mark.parametrize("mutation", ["file", "status", "protocol", "implementation", "command"])
def test_execution_spec_authentication_fails_closed(tmp_path, mutation):
    document = _spec(tmp_path)
    if mutation == "status":
        document["status"] = "draft"
    elif mutation == "protocol":
        document["protocol"]["sha256"] = "0" * 64
    elif mutation == "implementation":
        document["implementation"]["runner"]["sha256"] = "0" * 64
    elif mutation == "command":
        document["commands"]["fast_na_recovery"]["recovery_duration_ms"][-1] = 201
    path, digest = _write_spec(tmp_path, document)
    if mutation == "file":
        digest = "f" * 64
    with pytest.raises(runner.FastChannelClampError):
        runner.load_execution_spec(path, expected_file_sha256=digest, repository_root=tmp_path)


def test_existing_output_is_refused_before_backend_or_kernel_work(tmp_path, monkeypatch):
    spec, digest = _write_spec(tmp_path, _spec(tmp_path))
    output = tmp_path / "raw/clamp.json"
    output.parent.mkdir()
    output.write_text("banked evidence", encoding="ascii")
    monkeypatch.setattr(runner.lab, "assert_backend", lambda *args, **kwargs: pytest.fail("backend touched"))
    with pytest.raises(runner.FastChannelClampError, match="overwrite"):
        runner.run(
            spec_path=spec, spec_sha256=digest, output_path=output,
            backend="numpy", repository_root=tmp_path,
        )
    assert output.read_text() == "banked evidence"


def test_undeclared_output_is_refused_before_backend_assertion(tmp_path, monkeypatch):
    spec, digest = _write_spec(tmp_path, _spec(tmp_path))
    monkeypatch.setattr(runner.lab, "assert_backend", lambda *args, **kwargs: pytest.fail("backend touched"))
    with pytest.raises(runner.FastChannelClampError, match="output path"):
        runner.run(
            spec_path=spec, spec_sha256=digest, output_path=tmp_path / "raw/wrong.json",
            backend="numpy", repository_root=tmp_path,
        )


def test_assert_backend_failure_propagates_without_creating_output(tmp_path, monkeypatch):
    spec, digest = _write_spec(tmp_path, _spec(tmp_path))
    output = tmp_path / "raw/clamp.json"
    monkeypatch.setattr(
        runner.lab, "assert_backend",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("wrong device")),
    )
    with pytest.raises(AssertionError, match="wrong device"):
        runner.run(
            spec_path=spec, spec_sha256=digest, output_path=output,
            backend="numpy", repository_root=tmp_path,
        )
    assert not output.exists()


def test_repository_execution_spec_authenticates_exact_implementation():
    path = runner.EXECUTION_SPEC_PATH
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    document, binding = runner.load_execution_spec(
        path, expected_file_sha256=file_sha, repository_root=runner.ROOT,
    )
    assert binding == {
        "path": "research/specs/v14_snr_stageB_fast_channel_clamp_execution_v1.json",
        "sha256": file_sha,
    }
    assert document["production_kernel_base_revision"]["revision"] == (
        "ba769f3967d9a50fc859ec7cf20587b81280b243"
    )
    assert document["numeric_contract"]["clamp_sample_interval_ms"] == 0.005
