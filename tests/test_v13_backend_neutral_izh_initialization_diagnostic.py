"""No-seed tests for the V13 backend-neutral initialization evidence flow."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from research.runners import (
    _v13_backend_neutral_izh_initialization_diagnostic as diagnostic,
)
from tools import v13_backend_neutral_izh_initialization_evidence as evidence


TEST_REVISION = "1" * 40


def _array_record(value: np.ndarray) -> dict:
    array = np.ascontiguousarray(value)
    payload = array.tobytes(order="C")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes_base64": base64.b64encode(payload).decode("ascii"),
        "bytes_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _cell(backend: str, *, delta: int = 0) -> dict:
    source = {
        "file_count": len(diagnostic.source_paths()),
        "git_sha": TEST_REVISION,
        "kind": "git",
        "manifest": "source.sha256",
        "manifest_sha256": "2" * 64,
        "tree_sha256": "3" * 64,
    }
    complete = {
        "backend_neutral_izh_initialization": True,
        "brain_regions": diagnostic._expected_config_summary()["brain_regions"],
        "heterogeneity_seed": diagnostic.LOCKED_SEED,
        "neuron_model_type": diagnostic._expected_config_summary()["neuron_model_type"],
        "num_neurons": 60,
        "seed": diagnostic.LOCKED_SEED,
    }
    arrays = {
        name: _array_record(np.array([index + delta], dtype=np.float32))
        for index, name in enumerate(diagnostic.ARRAYS)
    }
    return diagnostic._seal({
        "schema": diagnostic.SCHEMA_CELL,
        "promotion_value": "none",
        "diagnostic_only": True,
        "backend": backend,
        "seed": diagnostic.LOCKED_SEED,
        "spec_sha256": diagnostic.SPEC_SHA256,
        "source": source,
        "config": {
            "sha256": hashlib.sha256(diagnostic._canonical(complete)).hexdigest(),
            "complete": complete,
            "locked_summary": diagnostic._expected_config_summary(),
        },
        "runtime_state_before_capture": {"current_time_ms": 0.0, "current_time_step": 0},
        "runtime_state_after_capture": {"current_time_ms": 0.0, "current_time_step": 0},
        "simulation_steps_executed": 0,
        "arrays": arrays,
    })


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def test_authoritative_spec_is_exact_and_no_test_invokes_capture(monkeypatch):
    spec = diagnostic.load_locked_spec()
    assert spec["diagnostic_partition"]["seed"] == diagnostic.LOCKED_SEED
    assert tuple(spec["required_exact_arrays"]) == diagnostic.ARRAYS
    assert spec["promotion_value"] == "none"
    assert "sim/kernels.py" in diagnostic.source_paths()
    assert (
        "research/findings/2026-08-04-neural-vocal-credit-gateB-v13-"
        "backend-neutral-izh-initialization-correction-DIAGNOSTIC-PREREGISTRATION.md"
        in diagnostic.source_paths()
    )
    tracked_sim = set(_git(diagnostic.ROOT, "ls-files", "sim/**/*.py", "sim/*.py").splitlines())
    assert tracked_sim
    assert tracked_sim.issubset(diagnostic.source_paths())

    monkeypatch.setattr(
        diagnostic, "_build_bridge",
        lambda: pytest.fail("no-seed tests must not initialize the locked diagnostic"),
    )
    with pytest.raises(ValueError, match="spec digest mismatch"):
        diagnostic.load_locked_spec(expected_sha256="0" * 64)


def test_comparison_requires_complete_receipted_pair_and_exact_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    numpy_artifact = tmp_path / "numpy.json"
    cupy_artifact = tmp_path / "cupy.json"
    numpy_receipt = tmp_path / "numpy.receipt.json"
    cupy_receipt = tmp_path / "cupy.receipt.json"
    for path in (numpy_receipt, cupy_receipt):
        _write_json(path, {"receipt": path.name})
    _write_json(numpy_artifact, _cell("numpy"))
    _write_json(cupy_artifact, _cell("cupy"))

    monkeypatch.setattr(
        diagnostic, "_receipt_for_cell",
        lambda path, artifact_path, cell, backend: {
            "host": f"{backend}-host", "device": f"{backend}-device",
        },
    )
    out = tmp_path / "comparison.json"
    result = diagnostic.compare(
        numpy_artifact=numpy_artifact, numpy_receipt=numpy_receipt,
        cupy_artifact=cupy_artifact, cupy_receipt=cupy_receipt, out=out,
    )
    assert result["outcome"] == "DIAGNOSTIC_PASS"
    assert result["all_required_arrays_exact"] is True
    assert result["promotion_value"] == "none"
    assert result["scientific_verdict"] is None
    assert result["simulation_steps_executed"] == {"numpy": 0, "cupy": 0}
    assert all(row == {
        "dtype_equal": True, "shape_equal": True,
        "bytes_equal": True, "exact": True,
    } for row in result["array_comparisons"].values())

    with pytest.raises(FileExistsError, match="already exists"):
        diagnostic.compare(
            numpy_artifact=numpy_artifact, numpy_receipt=numpy_receipt,
            cupy_artifact=cupy_artifact, cupy_receipt=cupy_receipt, out=out,
        )


def test_comparison_reports_diagnostic_failure_without_promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    paths = {name: tmp_path / f"{name}.json" for name in (
        "numpy_artifact", "numpy_receipt", "cupy_artifact", "cupy_receipt",
    )}
    _write_json(paths["numpy_artifact"], _cell("numpy"))
    _write_json(paths["cupy_artifact"], _cell("cupy", delta=1))
    _write_json(paths["numpy_receipt"], {})
    _write_json(paths["cupy_receipt"], {})
    monkeypatch.setattr(
        diagnostic, "_receipt_for_cell",
        lambda path, artifact_path, cell, backend: {"host": "h", "device": "d"},
    )
    result = diagnostic.compare(out=tmp_path / "out.json", **paths)
    assert result["outcome"] == "DIAGNOSTIC_FAIL"
    assert result["all_required_arrays_exact"] is False
    assert result["promotion_value"] == "none"
    assert result["scientific_verdict"] is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value["arrays"].pop(diagnostic.ARRAYS[-1]), "cell contract mismatch"),
        (lambda value: value["config"].update(sha256="0" * 64), "config identity"),
        (lambda value: value.update(simulation_steps_executed=1), "cell contract mismatch"),
        (lambda value: value.update(promotion_value="eligible"), "cell contract mismatch"),
    ),
)
def test_cell_validation_rejects_missing_drifted_or_promotable_evidence(
    tmp_path, mutation, message,
):
    value = _cell("numpy")
    mutation(value)
    value["sha256"] = diagnostic._artifact_digest(value)
    path = tmp_path / "cell.json"
    _write_json(path, value)
    with pytest.raises(ValueError, match=message):
        diagnostic._load_cell(path, "numpy")


def test_receipt_must_bind_exact_capture_command(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "SPEC_PATH", tmp_path / "spec.json")
    artifact = tmp_path / "cell.json"
    receipt_path = tmp_path / "cell.receipt.json"
    manifest = tmp_path / "source.sha256"
    for path in (artifact, receipt_path, manifest, tmp_path / "spec.json"):
        path.write_text("{}", encoding="ascii")
    cell = _cell("numpy")
    source = cell["source"]
    receipt = {
        "artifact": {
            "path": "cell.json", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
        "env_allowlist": {"SIM_BACKEND": "numpy"},
        "source": source,
        "argv": [
            sys.executable, "-m",
            "research.runners._v13_backend_neutral_izh_initialization_diagnostic",
            "--spec", str((tmp_path / "spec.json").resolve()),
            "--spec-sha256", diagnostic.SPEC_SHA256,
            "--capture", "--backend", "numpy",
            "--source-manifest", str(manifest.resolve()),
            "--source-revision", TEST_REVISION,
            "--out", str(artifact.resolve()),
        ],
    }
    monkeypatch.setattr(diagnostic.execution_receipt, "verify_receipt", lambda root, path: receipt)
    assert diagnostic._receipt_for_cell(
        receipt_path, artifact, cell, "numpy"
    ) is receipt
    receipt["argv"][7] = "cupy"
    with pytest.raises(ValueError, match="frozen capture command"):
        diagnostic._receipt_for_cell(receipt_path, artifact, cell, "numpy")


def test_final_manifest_strictly_binds_comparison_command_and_receipt(tmp_path, monkeypatch):
    root = tmp_path
    base = root / "results"
    base.mkdir()
    for name in ("cell-numpy.json", "cell-numpy.receipt.json", "cell-cupy.json", "cell-cupy.receipt.json"):
        (base / name).write_text("{}\n", encoding="ascii")
    source = {
        "file_count": 2, "git_sha": TEST_REVISION, "kind": "git",
        "manifest": "results/source.sha256", "manifest_sha256": "2" * 64,
        "tree_sha256": "3" * 64,
    }
    artifact = diagnostic._seal({
        "schema": diagnostic.SCHEMA_COMPARISON,
        "promotion_value": "none", "diagnostic_only": True,
        "scientific_verdict": None, "seed": diagnostic.LOCKED_SEED,
        "spec_sha256": diagnostic.SPEC_SHA256, "source": source,
        "config_sha256": "4" * 64,
        "simulation_steps_executed": {"numpy": 0, "cupy": 0},
        "cell_artifacts": {
            "numpy": {
                "path": "results/cell-numpy.json",
                "receipt_path": "results/cell-numpy.receipt.json",
            },
            "cupy": {
                "path": "results/cell-cupy.json",
                "receipt_path": "results/cell-cupy.receipt.json",
            },
        },
        "array_comparisons": {
            name: {
                "dtype_equal": True, "shape_equal": True,
                "bytes_equal": True, "exact": True,
            }
            for name in diagnostic.ARRAYS
        },
        "all_required_arrays_exact": True,
        "outcome": "DIAGNOSTIC_PASS",
    })
    artifact_path = base / "comparison.json"
    receipt_path = base / "comparison.receipt.json"
    _write_json(artifact_path, artifact)
    _write_json(receipt_path, {"placeholder": True})
    runner_spec = str((root / diagnostic.SPEC_PATH.relative_to(diagnostic.ROOT)).resolve())
    receipt = {
        "artifact": {
            "path": "results/comparison.json",
            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        },
        "env_allowlist": {"SIM_BACKEND": "numpy"},
        "source": source,
        "host": "compare-host", "device": "CPU",
        "started_utc_ns": 10, "ended_utc_ns": 20,
        "argv": [
            sys.executable, "-m",
            "research.runners._v13_backend_neutral_izh_initialization_diagnostic",
            "--spec", runner_spec, "--spec-sha256", diagnostic.SPEC_SHA256,
            "--compare",
            "--numpy-artifact", str((base / "cell-numpy.json").resolve()),
            "--numpy-receipt", str((base / "cell-numpy.receipt.json").resolve()),
            "--cupy-artifact", str((base / "cell-cupy.json").resolve()),
            "--cupy-receipt", str((base / "cell-cupy.receipt.json").resolve()),
            "--out", str(artifact_path.resolve()),
        ],
    }
    monkeypatch.setattr(evidence.execution_receipt, "verify_receipt", lambda root, path: receipt)
    manifest = evidence.finalize_evidence(
        root=root, artifact_path="results/comparison.json",
        receipt_path="results/comparison.receipt.json",
        out="results/evidence-manifest.json",
    )
    assert manifest["promotion_value"] == "none"
    assert manifest["scientific_verdict"] is None
    assert manifest["comparison_receipt"]["host"] == "compare-host"
    with pytest.raises(evidence.EvidenceError, match="overwrite"):
        evidence.finalize_evidence(
            root=root, artifact_path="results/comparison.json",
            receipt_path="results/comparison.receipt.json",
            out="results/evidence-manifest.json",
        )

    receipt["argv"][-2] = str(root / "different.json")
    with pytest.raises(evidence.EvidenceError, match="frozen compare command"):
        evidence._comparison_receipt(
            root=root, artifact_path="results/comparison.json",
            receipt_path="results/comparison.receipt.json",
        )


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def test_evidence_adapter_freezes_committed_sources_and_emits_only_commands(
    tmp_path, monkeypatch,
):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    source = root / "source.py"
    source.write_text("VALUE = 1\n", encoding="ascii")
    _git(root, "add", "source.py")
    _git(root, "commit", "-m", "source")
    revision = _git(root, "rev-parse", "HEAD")
    output_dir = root / evidence.OUTPUT_DIR
    output_dir.mkdir(parents=True)
    monkeypatch.setattr(
        diagnostic, "source_paths", lambda root=None: ("source.py",),
    )

    frozen = evidence.freeze_source_manifest(
        root=root, revision=revision, out=f"{evidence.OUTPUT_DIR}/source.sha256",
    )
    assert frozen["revision"] == revision
    assert frozen["file_count"] == 1
    envelope = evidence.emit_command(
        root=root, action="capture_numpy", revision=revision,
        host="cpu-host", device="CPU", out=f"{evidence.OUTPUT_DIR}/capture-numpy.command.json",
        python=sys.executable,
    )
    assert envelope["execution"] == "not_executed"
    assert envelope["promotion_value"] == "none"
    assert envelope["env"] == {"SIM_BACKEND": "numpy"}
    assert not (root / envelope["artifact"]).exists()
    assert not (root / envelope["receipt"]).exists()
    assert diagnostic.LOCKED_SEED == 6_556_023

    with pytest.raises(evidence.EvidenceError, match="overwrite"):
        evidence.freeze_source_manifest(
            root=root, revision=revision, out=f"{evidence.OUTPUT_DIR}/source.sha256",
        )


def test_evidence_adapter_rejects_source_drift_and_partial_compare(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    sim = root / "sim"
    sim.mkdir()
    source = sim / "bridge.py"
    transitive = sim / "kernels.py"
    source.write_text("VALUE = 1\n", encoding="ascii")
    transitive.write_text("KERNEL = 1\n", encoding="ascii")
    _git(root, "add", "sim/bridge.py", "sim/kernels.py")
    _git(root, "commit", "-m", "source")
    revision = _git(root, "rev-parse", "HEAD")
    (root / evidence.OUTPUT_DIR).mkdir(parents=True)
    monkeypatch.setattr(diagnostic, "AUTHORITY_SOURCE_PATHS", ())
    assert diagnostic.source_paths(root) == ("sim/bridge.py", "sim/kernels.py")
    evidence.freeze_source_manifest(
        root=root, revision=revision, out=f"{evidence.OUTPUT_DIR}/source.sha256",
    )
    with pytest.raises(evidence.EvidenceError, match="partial evidence"):
        evidence.emit_command(
            root=root, action="compare", revision=revision, host="host", device="CPU",
            out=f"{evidence.OUTPUT_DIR}/compare.command.json", python=sys.executable,
        )

    transitive.write_text("KERNEL = 2\n", encoding="ascii")
    with pytest.raises(evidence.EvidenceError, match="source digest mismatch"):
        evidence.emit_command(
            root=root, action="capture_cupy", revision=revision, host="host", device="GPU",
            out=f"{evidence.OUTPUT_DIR}/cupy.command.json", python=sys.executable,
        )
