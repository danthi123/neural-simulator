"""Synthetic authenticated tests for the independent source-transfer analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import v14_stageB_source_model_transfer_analysis as analysis


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii") + b"\n"
    path.write_bytes(payload)
    return _sha(payload)


def _binding(root: Path, relative: str, value: object) -> dict[str, str]:
    return {"path": relative, "sha256": _write_json(root / relative, value)}


def _commands() -> dict:
    return {
        "schema": "v14-snr-stageB-fast-channel-clamp-execution-v1",
        "numeric_contract": {"clamp_sample_interval_ms": 0.005},
        "commands": {
            "fast_na_activation": {"test_mV": [-40.0, 0.0]},
            "fast_na_inactivation": {"prepulse_mV": [-100.0, -40.0]},
            "fast_na_recovery": {"recovery_duration_ms": [0.1, 1.0]},
            "fast_na_deactivation": {"test_mV": [-60.0, -50.0, -40.0]},
            "kv3_activation": {"test_mV": [-20.0, 40.0]},
            "kv3_inactivation": {"prepulse_mV": [-80.0, 0.0]},
            "kv3_deactivation": {"test_mV": [-60.0, -50.0, -40.0]},
        },
    }


def _assay(ladder_field: str | None = None, ladder: list[float] | None = None) -> dict:
    value = {"elapsed_ms": [0.005, 0.01], "signal": [[0.1, 0.2]]}
    if ladder_field is not None:
        value[ladder_field] = ladder
    return value


def _assays(model_id: str) -> tuple[dict, list[str]]:
    commands = _commands()["commands"]
    if model_id in {analysis.KHALIQ, analysis.BALBI}:
        return {
            "fast_na_activation": _assay("command_voltage_mv", commands["fast_na_activation"]["test_mV"]),
            "fast_na_inactivation": _assay("prepulse_voltage_mv", commands["fast_na_inactivation"]["prepulse_mV"]),
            "fast_na_composite_zero": _assay(),
            "fast_na_recovery": _assay("recovery_duration_ms", commands["fast_na_recovery"]["recovery_duration_ms"]),
            "fast_na_deactivation": _assay("command_voltage_mv", commands["fast_na_deactivation"]["test_mV"]),
        }, []
    result = {
        "kv3_activation": _assay("command_voltage_mv", commands["kv3_activation"]["test_mV"]),
        "kv3_rise": _assay(),
        "kv3_deactivation": _assay("command_voltage_mv", commands["kv3_deactivation"]["test_mV"]),
    }
    if model_id == analysis.DESAI:
        result["kv3_inactivation"] = _assay("prepulse_voltage_mv", commands["kv3_inactivation"]["prepulse_mV"])
        return result, []
    return result, ["kv3_inactivation"]


def _fake_metrics(document: dict) -> dict[str, float]:
    model_id = document["model_id"]
    names = (
        analysis.SODIUM_METRICS
        if model_id in {analysis.KHALIQ, analysis.BALBI}
        else analysis.LABRO_METRICS
        if model_id == analysis.LABRO
        else analysis.DESAI_METRICS
    )
    return {name: 1.0 for name in names}


def _fixture(tmp_path: Path, *, parity_failure: bool = False) -> dict:
    root = tmp_path / "repo"
    root.mkdir()
    command_binding = _binding(root, "authority/commands.json", _commands())
    targets = {
        name: {"mean": 1.0, "sem": 0.1}
        for name in set(analysis.SODIUM_METRICS) | set(analysis.DESAI_METRICS)
    }
    target_binding = _binding(
        root,
        "authority/targets.json",
        {
            "schema": "v14-snr-stageB-structural-successor-protocol-v2",
            "source_transfers": {
                "fast_sodium": {
                    "constraints": {
                        name.removeprefix("fast_na."): target
                        for name, target in targets.items()
                        if name.startswith("fast_na.")
                    }
                },
                "kv3_like": {
                    "constraints": {
                        **{
                            name.removeprefix("kv3_like."): target
                            for name, target in targets.items()
                            if name.startswith("kv3_like.") and "deactivation_at_" not in name
                        },
                        "deactivation_tau_ms": [
                            {"command_mV": -60.0, **targets["kv3_like.deactivation_at_minus_60_mV_ms"]},
                            {"command_mV": -50.0, **targets["kv3_like.deactivation_at_minus_50_mV_ms"]},
                            {"command_mV": -40.0, **targets["kv3_like.deactivation_at_minus_40_mV_ms"]},
                        ],
                    }
                },
            },
        },
    )
    implementation = {}
    for name in ("sodium_models", "kv3_models", "runner"):
        relative = f"implementation/{name}.py"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="ascii")
        implementation[name] = {"path": relative, "sha256": _sha(path.read_bytes())}

    models = [
        {
            "model_id": analysis.KHALIQ,
            "model_fingerprint": "1" * 64,
            "calibration_allowed": False,
            "conductance_fitting_allowed": False,
            "conditions": [{"condition_id": "native", "temperature_c": None}],
        },
        {
            "model_id": analysis.BALBI,
            "model_fingerprint": "2" * 64,
            "calibration_allowed": False,
            "conductance_fitting_allowed": False,
            "conditions": [{"condition_id": "22c", "temperature_c": 22.0}],
        },
        {
            "model_id": analysis.LABRO,
            "model_fingerprint": "3" * 64,
            "calibration_allowed": False,
            "conductance_fitting_allowed": False,
            "conditions": [
                {"condition_id": "20c", "temperature_c": 20.0},
                {"condition_id": "22_5c", "temperature_c": 22.5},
                {"condition_id": "25c", "temperature_c": 25.0},
            ],
        },
        {
            "model_id": analysis.DESAI,
            "model_fingerprint": "4" * 64,
            "calibration_allowed": False,
            "conductance_fitting_allowed": False,
            "conditions": [{"condition_id": "published", "temperature_c": None}],
        },
    ]
    matrix = []
    for model in models:
        for condition in model["conditions"]:
            for backend in ("numpy", "cupy"):
                stem = f"evidence/{model['model_id']}-{condition['condition_id']}-{backend}"
                matrix.append({
                    "model_id": model["model_id"],
                    "condition_id": condition["condition_id"],
                    "backend": backend,
                    "device": "cpu" if backend == "numpy" else "gpu",
                    "authority": "cpu_reference" if backend == "numpy" else "gpu_parity_only",
                    "output": f"{stem}.json",
                    "receipt": f"{stem}.receipt.json",
                })
    spec = {
        "schema": analysis.SPEC_SCHEMA,
        "status": "preregistered_not_executed",
        "scientific_verdict": None,
        "candidate_calibration_allowed": False,
        "hybridization_allowed": False,
        "stage2_integration_allowed": False,
        "analysis_contract": {
            "compensation_allowed": False,
            "stage2_verdict_allowed": False,
            "cpu_gpu_parity": {"rtol": analysis.PARITY_RTOL, "atol": analysis.PARITY_ATOL, "pointwise": True},
            "analyzer_authority": {
                "path": "tools/v14_stageB_source_model_transfer_analysis.py",
                "sha256": "placeholder",
            },
            "estimator_authority": {
                "path": "tools/v14_stageB_fast_channel_clamp_analysis.py",
                "sha256": "placeholder",
            },
        },
        "command_authority": command_binding,
        "target_authority": target_binding,
        "implementation": implementation,
        "models": models,
        "execution_matrix": matrix,
        "analysis_output": "results/analysis.json",
        "consumption_ledger": "results/analysis.consumed.json",
    }
    estimator = root / "tools/v14_stageB_fast_channel_clamp_analysis.py"
    estimator.parent.mkdir(parents=True, exist_ok=True)
    estimator.write_bytes((Path(analysis.__file__).resolve().parents[0] / "v14_stageB_fast_channel_clamp_analysis.py").read_bytes())
    spec["analysis_contract"]["estimator_authority"]["sha256"] = _sha(estimator.read_bytes())
    analyzer = root / "tools/v14_stageB_source_model_transfer_analysis.py"
    analyzer.write_bytes(Path(analysis.__file__).resolve().read_bytes())
    spec["analysis_contract"]["analyzer_authority"]["sha256"] = _sha(analyzer.read_bytes())
    spec_relative = "specs/transfer.json"
    spec_sha = _write_json(root / spec_relative, spec)
    spec_binding = {"path": spec_relative, "sha256": spec_sha}
    model_map = {model["model_id"]: model for model in models}
    for index, row in enumerate(matrix):
        model = model_map[row["model_id"]]
        condition = next(item for item in model["conditions"] if item["condition_id"] == row["condition_id"])
        assays, unavailable = _assays(row["model_id"])
        if parity_failure and row["model_id"] == analysis.DESAI and row["backend"] == "cupy":
            assays["kv3_rise"]["signal"][0][1] += 1e-3
        document = {
            "schema": analysis.OBSERVATION_SCHEMA,
            "execution_spec": spec_binding,
            "command_authority": command_binding,
            "implementation": implementation,
            "model_id": row["model_id"],
            "condition_id": row["condition_id"],
            "condition": condition,
            "model_fingerprint": model["model_fingerprint"],
            "backend": row["backend"],
            "device": row["device"],
            "authority": row["authority"],
            "dtype": "float64",
            "sample_interval_ms": 0.005,
            "assays": assays,
            "unavailable_assays": unavailable,
            "candidate_calibration_allowed": False,
            "conductance_fitting_allowed": False,
            "scientific_verdict": None,
            "analysis_status": "raw_unanalyzed",
        }
        document["sha256"] = analysis._semantic_digest(document)
        artifact_sha = _write_json(root / row["output"], document)
        run_id = f"run-{index}"
        argv = analysis._expected_argv(root, spec_binding, row)
        provenance_relative = f"{row['output']}.prov.json"
        provenance = {
            "schema": analysis.PROVENANCE_SCHEMA,
            "run_id": run_id,
            "runner": analysis.RUNNER,
            "argv": argv,
            "artifact": row["output"],
            "git_sha": "a" * 40,
            "source_kind": "git",
            "source_manifest_sha256": "b" * 64,
            "sim_backend_requested": row["backend"],
            "sim_backend": row["backend"],
            "started_utc_ns": 100 + index,
            "ended_utc_ns": 200 + index,
        }
        provenance_sha = _write_json(root / provenance_relative, provenance)
        receipt = {
            "schema": analysis.RECEIPT_SCHEMA,
            "status": "success",
            "exit_code": 0,
            "device": row["device"],
            "argv": ["python", "runner.py", *argv[1:]],
            "artifact": {
                "path": row["output"],
                "sha256": artifact_sha,
                "size_bytes": (root / row["output"]).stat().st_size,
            },
            "provenance": {
                "path": provenance_relative,
                "sha256": provenance_sha,
                "run_id": run_id,
                "started_utc_ns": 100 + index,
                "ended_utc_ns": 200 + index,
            },
            "source": {
                "git_sha": "a" * 40,
                "kind": "git",
                "manifest_sha256": "b" * 64,
            },
        }
        _write_json(root / row["receipt"], receipt)
    return {"root": root, "spec": spec_relative, "spec_sha": spec_sha, "matrix": matrix}


def test_authenticated_analysis_marks_labro_inactivation_unavailable(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(analysis, "_condition_metrics", _fake_metrics)

    result = analysis.analyze(fixture["spec"], fixture["spec_sha"], repository_root=fixture["root"])

    labro = [row for row in result["conditions"] if row["model_id"] == analysis.LABRO]
    assert len(labro) == 3
    assert all(row["inactivation_assay"] == "STRUCTURALLY_UNAVAILABLE" for row in labro)
    assert all(len(row["gates"]) == 6 for row in labro)
    assert result["labro_temperature_envelope"]["temperatures_c"] == [20.0, 22.5, 25.0]
    assert result["labro_temperature_envelope"]["post_hoc_temperature_selection_allowed"] is False
    assert result["stage2_go_issued"] is False


def test_tampered_observation_is_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(analysis, "_condition_metrics", _fake_metrics)
    path = fixture["root"] / fixture["matrix"][0]["output"]
    document = json.loads(path.read_text())
    document["condition_id"] = "tampered"
    _write_json(path, document)

    with pytest.raises(analysis.SourceModelTransferAnalysisError, match="self digest"):
        analysis.analyze(fixture["spec"], fixture["spec_sha"], repository_root=fixture["root"])


def test_tampered_provenance_sidecar_is_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(analysis, "_condition_metrics", _fake_metrics)
    path = fixture["root"] / f"{fixture['matrix'][0]['output']}.prov.json"
    provenance = json.loads(path.read_text())
    provenance["sim_backend"] = "cupy"
    _write_json(path, provenance)

    with pytest.raises(analysis.SourceModelTransferAnalysisError, match="actual backend"):
        analysis.analyze(fixture["spec"], fixture["spec_sha"], repository_root=fixture["root"])


def test_full_assay_pointwise_parity_failure_is_condition_local(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, parity_failure=True)
    monkeypatch.setattr(analysis, "_condition_metrics", _fake_metrics)

    result = analysis.analyze(fixture["spec"], fixture["spec_sha"], repository_root=fixture["root"])

    desai = next(row for row in result["conditions"] if row["model_id"] == analysis.DESAI)
    assert desai["backend_parity"]["pointwise_passed"] is False
    assert desai["verdict"] == "INCONCLUSIVE_INVALID_EVIDENCE"
    assert all(
        row["backend_parity"]["passed"]
        for row in result["conditions"]
        if row["model_id"] != analysis.DESAI
    )


def test_one_failed_metric_cannot_be_compensated(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)

    def metrics(document):
        values = _fake_metrics(document)
        if document["model_id"] == analysis.KHALIQ:
            values[analysis.SODIUM_METRICS[0]] = 1.21
        return values

    monkeypatch.setattr(analysis, "_condition_metrics", metrics)
    result = analysis.analyze(fixture["spec"], fixture["spec_sha"], repository_root=fixture["root"])
    khaliq = next(row for row in result["conditions"] if row["model_id"] == analysis.KHALIQ)

    assert khaliq["failed_gate_count"] == 1
    assert khaliq["source_transfer_candidate"] is False
    assert khaliq["aggregate_compensation_allowed"] is False


def test_analysis_and_ledger_are_create_once(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(analysis, "_condition_metrics", _fake_metrics)
    result = analysis.analyze(fixture["spec"], fixture["spec_sha"], repository_root=fixture["root"])

    ledger = analysis.write_analysis_bundle(
        fixture["root"], "results/analysis.json", "results/analysis.consumed.json", result
    )
    assert ledger["execution_spec"] == result["execution_spec"]
    assert ledger["model_fingerprints"] == result["model_fingerprints"]
    assert ledger["analysis_output"]["sha256"] == result["sha256"]
    assert ledger["analyzer"] == result["analyzer"]

    with pytest.raises(analysis.SourceModelTransferAnalysisError, match="refusing existing"):
        analysis.write_analysis_bundle(
            fixture["root"], "results/analysis.json", "results/analysis.consumed.json", result
        )
