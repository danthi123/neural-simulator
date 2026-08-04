from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from research.runners import _v13_backend_neutral_izh_arithmetic_replay as common
from research.runners import _v13_backend_neutral_izh_arithmetic_replay_v2 as replay
from tools import v13_backend_neutral_izh_arithmetic_replay_evidence_v2 as evidence


def test_v2_locked_spec_is_seedless_and_uses_separate_output():
    spec = replay.load_locked_spec()
    raw = replay.SPEC_PATH.read_text(encoding="ascii")

    assert spec["id"].endswith("-v2")
    assert spec["status"] == "preregistered_not_executed"
    assert spec["rng"]["measured_replay_allowed"] is False
    assert spec["rng"]["measured_replay_seed"] is None
    assert spec["replay"]["total_steps"] == 1_200
    assert spec["execution"]["trajectory_arrays"] == ["v", "u", "spikes"]
    assert spec["output_directory"] == replay.OUTPUT_DIRECTORY
    assert spec["output_directory"].endswith("_v2")
    assert "4c685269c" not in raw
    assert spec["correction_under_test"][
        "source_revision_is_bound_by_execution_manifest"
    ] is True


def test_v2_locked_spec_digest_rejects_drift(tmp_path):
    path = tmp_path / "spec.json"
    path.write_bytes(replay.SPEC_PATH.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="locked spec digest mismatch"):
        replay.load_locked_spec(path, replay.SPEC_SHA256)


def test_v2_source_set_binds_simulator_shared_code_and_v2_authorities():
    paths = set(replay.source_paths())
    simulator = {
        path.relative_to(replay.ROOT).as_posix()
        for path in (replay.ROOT / "sim").rglob("*.py")
        if path.is_file()
    }

    assert simulator.issubset(paths)
    assert "sim/kernels.py" in paths
    assert common.V1_PROTOCOL.runner_module.replace(".", "/") + ".py" in paths
    assert replay.PROTOCOL.runner_module.replace(".", "/") + ".py" in paths
    assert "tools/v13_backend_neutral_izh_arithmetic_replay_evidence.py" in paths
    assert "tools/v13_backend_neutral_izh_arithmetic_replay_evidence_v2.py" in paths
    assert replay.SPEC_RELATIVE_PATH.as_posix() in paths
    assert any(path.endswith("replay-v1-DIAGNOSTIC-RESULT.md") for path in paths)
    assert (
        f"{common.V1_PROTOCOL.output_directory}/comparison.json" in paths
    )
    assert (
        f"{common.V1_PROTOCOL.output_directory}/evidence-manifest.json" in paths
    )
    assert {
        path for path in paths
        if path.startswith(common.V1_PROTOCOL.output_directory + "/")
    } == {
        f"{common.V1_PROTOCOL.output_directory}/comparison.json",
        f"{common.V1_PROTOCOL.output_directory}/evidence-manifest.json",
    }


def test_v2_consumes_same_completed_numpy_transplant_as_v1():
    v1 = common.load_locked_spec()
    v2 = replay.load_locked_spec()

    assert v2["completed_input"] == v1["completed_input"]
    bundle, transplant_spec = replay.load_completed_input(v2)
    assert bundle["origin"] == "numpy"
    assert transplant_spec["steps"] == {
        "baseline": 500,
        "inhibition": 200,
        "release": 500,
        "dt_ms": 1.0,
    }


def test_v2_paths_cannot_alias_v1_or_escape_v2_output_directory():
    v1_paths = set(
        __import__(
            "tools.v13_backend_neutral_izh_arithmetic_replay_evidence",
            fromlist=["_paths"],
        )._paths().values()
    )
    v2_paths = set(evidence._paths().values())

    assert v1_paths.isdisjoint(v2_paths)
    assert all(path.startswith(evidence.OUTPUT_DIR + "/") for path in v2_paths)


def test_v2_inner_commands_are_seedless_and_use_exact_v2_modules(tmp_path):
    paths = evidence._paths()
    revision = "a" * 40
    run = evidence._inner_command(
        root=tmp_path, action="run_cupy", revision=revision,
        paths=paths, python=sys.executable,
    )
    compare = evidence._inner_command(
        root=tmp_path, action="compare", revision=revision,
        paths=paths, python=sys.executable,
    )

    assert run[:3] == [sys.executable, "-m", replay.PROTOCOL.runner_module]
    assert compare[:3] == [sys.executable, "-m", replay.PROTOCOL.runner_module]
    assert str((tmp_path / replay.SPEC_RELATIVE_PATH).resolve()) in run
    assert replay.SPEC_SHA256 in run
    assert not any("seed" in argument.lower() for argument in run + compare)
    assert paths["numpy_receipt"] in " ".join(compare)
    assert paths["cupy_receipt"] in " ".join(compare)


def _cell_value(backend: str, source: dict, changed: bool = False) -> dict:
    arrays = {
        "v": np.zeros((1_200, 60), dtype=np.float32),
        "u": np.zeros((1_200, 60), dtype=np.float32),
        "spikes": np.zeros((1_200, 60), dtype=np.bool_),
    }
    if changed:
        arrays["u"][1_113, 0] = np.float32(1.0)
    encoded = {
        name: replay.transplant._encode_array(value) for name, value in arrays.items()
    }
    return replay._seal({
        "schema": replay.SCHEMA_CELL,
        "promotion_value": "none",
        "diagnostic_only": True,
        "scientific_verdict": None,
        "backend": backend,
        "spec_sha256": replay.SPEC_SHA256,
        "source": source,
        "completed_input": {"bundle_file_sha256": "1" * 64},
        "measured_replay_rng": {
            "allowed": False, "seed": None, "calls_observed": 0,
        },
        "simulation_steps_executed": 1_200,
        "trajectories": encoded,
        "trajectory_sha256": {
            name: value["sha256"] for name, value in encoded.items()
        },
        "trajectory_step_sha256": {
            name: replay.transplant._trajectory_step_hashes(value)
            for name, value in arrays.items()
        },
        "instrument_valid": True,
    })


def _write_cell(root: Path, backend: str, source: dict, changed: bool = False):
    artifact = root / f"cell-{backend}.json"
    receipt = root / f"cell-{backend}.receipt.json"
    artifact.write_text(
        json.dumps(_cell_value(backend, source, changed), sort_keys=True) + "\n",
        encoding="ascii",
    )
    receipt.write_text("{}\n", encoding="ascii")
    return artifact, receipt


def _receipt(root: Path, source: dict, artifact: Path, backend: str) -> dict:
    return {
        "argv": replay._expected_cell_argv(
            root=root, artifact=artifact, backend=backend, source=source,
            python=sys.executable,
        ),
        "artifact": {
            "path": artifact.relative_to(root).as_posix(),
            "sha256": replay._digest_file(artifact),
        },
        "env_allowlist": {"SIM_BACKEND": backend},
        "source": source,
    }


def test_v2_compare_is_exact_and_reports_step_1114_without_simulation(
    tmp_path, monkeypatch,
):
    output = tmp_path / replay.OUTPUT_DIRECTORY
    output.mkdir(parents=True)
    source = {
        "file_count": 1, "git_sha": "a" * 40, "kind": "git",
        "manifest": "source.sha256", "manifest_sha256": "b" * 64,
        "tree_sha256": "c" * 64,
    }
    numpy_artifact, numpy_receipt = _write_cell(output, "numpy", source)
    cupy_artifact, cupy_receipt = _write_cell(output, "cupy", source, True)
    receipts = {
        numpy_receipt: _receipt(tmp_path, source, numpy_artifact, "numpy"),
        cupy_receipt: _receipt(tmp_path, source, cupy_artifact, "cupy"),
    }
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "load_locked_spec", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        common.execution_receipt, "verify_receipt",
        lambda _root, relative: receipts[tmp_path / relative],
    )

    result = replay.compare_cells(
        numpy_artifact=numpy_artifact, numpy_receipt=numpy_receipt,
        cupy_artifact=cupy_artifact, cupy_receipt=cupy_receipt,
        out=output / "comparison.json",
    )

    assert result["schema"] == replay.SCHEMA_COMPARISON
    assert result["outcome"] == "DIAGNOSTIC_FAIL"
    difference = result["trajectory_comparisons"]["u"]["first_difference"]
    assert difference["step_number"] == 1_114
    assert result["trajectory_comparisons"]["v"]["all_1200_rows_exact"] is True


def test_v2_cell_receipt_rejects_v1_runner_argv(tmp_path, monkeypatch):
    output = tmp_path / replay.OUTPUT_DIRECTORY
    output.mkdir(parents=True)
    source = {"git_sha": "a" * 40, "manifest": "source.sha256"}
    artifact, receipt_path = _write_cell(output, "numpy", source)
    receipt = _receipt(tmp_path, source, artifact, "numpy")
    receipt["argv"][2] = common.V1_PROTOCOL.runner_module
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(
        common.execution_receipt, "verify_receipt", lambda *_args: receipt,
    )

    with pytest.raises(ValueError, match="differs from the frozen run command"):
        replay._load_cell(artifact, receipt_path, "numpy")


def test_v2_emit_is_create_only_and_manifest_bound(tmp_path, monkeypatch):
    paths = evidence._paths()
    output = tmp_path / evidence.OUTPUT_DIR
    output.mkdir(parents=True)
    (tmp_path / paths["source_manifest"]).write_text("source\n", encoding="ascii")
    revision = "a" * 40
    snapshot = {
        "manifest_sha256": "b" * 64,
        "tree_sha256": "c" * 64,
        "files": {"authority": {}},
    }
    monkeypatch.setattr(evidence.shared, "_git_head", lambda _root: revision)
    monkeypatch.setattr(
        evidence.execution_receipt, "verify_source_manifest",
        lambda *_args: snapshot,
    )
    monkeypatch.setattr(
        evidence.execution_receipt, "_source_revision", lambda *_args: "git",
    )
    monkeypatch.setattr(
        evidence.common, "_protocol_source_paths",
        lambda _root, _protocol: ("authority",),
    )
    envelope_path = f"{evidence.OUTPUT_DIR}/run-numpy.command.json"

    envelope = evidence.emit_command(
        root=tmp_path, action="run_numpy", revision=revision,
        host="local3090", device="CPU-NumPy", out=envelope_path,
        python=sys.executable,
    )

    assert envelope["schema"] == evidence.SCHEMA
    assert envelope["execution"] == "not_executed"
    assert envelope["source_revision"] == revision
    assert envelope["artifact"].startswith(evidence.OUTPUT_DIR + "/")
    with pytest.raises(evidence.EvidenceError, match="refusing stale output path"):
        evidence.emit_command(
            root=tmp_path, action="run_numpy", revision=revision,
            host="local3090", device="CPU-NumPy", out=envelope_path,
            python=sys.executable,
        )


def test_v2_evidence_rejects_output_outside_namespace(tmp_path, monkeypatch):
    revision = "a" * 40
    monkeypatch.setattr(evidence.shared, "_git_head", lambda _root: revision)

    with pytest.raises(evidence.EvidenceError, match="must be inside"):
        evidence.freeze_source_manifest(
            root=tmp_path, revision=revision, out="outside/source.sha256",
        )


def test_v2_freeze_binds_the_requested_current_head(tmp_path, monkeypatch):
    revision = "d" * 40
    authority = tmp_path / "authority"
    authority.write_bytes(b"eventual protocol commit bytes\n")
    output = tmp_path / evidence.OUTPUT_DIR
    output.mkdir(parents=True)
    monkeypatch.setattr(evidence.shared, "_git_head", lambda _root: revision)
    monkeypatch.setattr(
        evidence.shared, "_tracked_sim_python", lambda *_args: set(),
    )
    monkeypatch.setattr(
        evidence.shared, "_git_file",
        lambda _root, requested, relative: (
            authority.read_bytes()
            if requested == revision and relative == "authority"
            else b"wrong"
        ),
    )
    monkeypatch.setattr(
        evidence.common, "_protocol_source_paths",
        lambda _root, _protocol: ("authority",),
    )

    frozen = evidence.freeze_source_manifest(
        root=tmp_path, revision=revision,
        out=f"{evidence.OUTPUT_DIR}/source.sha256",
    )

    assert frozen["revision"] == revision
    assert frozen["file_count"] == 1
    assert (output / "source.sha256").is_file()


def test_v2_expected_comparison_argv_is_exact():
    root = Path("/tmp/v13-v2-root")
    artifact_file = root / evidence.OUTPUT_DIR / "comparison.json"
    cells = {
        backend: {
            "path": f"{evidence.OUTPUT_DIR}/cell-{backend}.json",
            "receipt_path": f"{evidence.OUTPUT_DIR}/cell-{backend}.receipt.json",
        }
        for backend in replay.BACKENDS
    }

    argv = evidence._expected_comparison_argv(
        root=root, artifact_file=artifact_file,
        artifact={"cell_artifacts": cells}, python=sys.executable,
    )

    assert argv[:3] == [sys.executable, "-m", replay.PROTOCOL.runner_module]
    assert argv[3:7] == [
        "--spec", str((root / replay.SPEC_RELATIVE_PATH).resolve()),
        "--spec-sha256", replay.SPEC_SHA256,
    ]
    assert argv[-2:] == ["--out", str(artifact_file.resolve())]
    assert "--numpy-receipt" in argv and "--cupy-receipt" in argv


def test_v2_runner_rejects_outputs_outside_its_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="must be inside"):
        replay.compare_cells(
            numpy_artifact=tmp_path / "cell-numpy.json",
            numpy_receipt=tmp_path / "cell-numpy.receipt.json",
            cupy_artifact=tmp_path / "cell-cupy.json",
            cupy_receipt=tmp_path / "cell-cupy.receipt.json",
            out=tmp_path / "comparison.json",
        )


def test_v2_finalize_rejects_tampered_comparison_argv(tmp_path, monkeypatch):
    output = tmp_path / replay.OUTPUT_DIRECTORY
    output.mkdir(parents=True)
    source = {
        "file_count": 1, "git_sha": "a" * 40, "kind": "git",
        "manifest": "source.sha256", "manifest_sha256": "b" * 64,
        "tree_sha256": "c" * 64,
    }
    numpy_artifact, numpy_receipt = _write_cell(output, "numpy", source)
    cupy_artifact, cupy_receipt = _write_cell(output, "cupy", source)
    cells = {
        backend: {
            "path": artifact.relative_to(tmp_path).as_posix(),
            "file_sha256": replay._digest_file(artifact),
            "receipt_path": receipt.relative_to(tmp_path).as_posix(),
            "artifact_sha256": json.loads(artifact.read_text())["sha256"],
        }
        for backend, artifact, receipt in (
            ("numpy", numpy_artifact, numpy_receipt),
            ("cupy", cupy_artifact, cupy_receipt),
        )
    }
    comparison = replay._seal({
        "schema": replay.SCHEMA_COMPARISON,
        "promotion_value": "none",
        "diagnostic_only": True,
        "scientific_verdict": None,
        "spec_sha256": replay.SPEC_SHA256,
        "source": source,
        "completed_input": {"bundle_file_sha256": "1" * 64},
        "simulation_steps_compared": {"numpy": 1_200, "cupy": 1_200},
        "cell_artifacts": cells,
        "trajectory_comparisons": {
            name: {"exact": True} for name in replay.TRAJECTORIES
        },
        "all_required_trajectories_exact": True,
        "outcome": "DIAGNOSTIC_PASS",
    })
    comparison_path = output / "comparison.json"
    comparison_path.write_text(json.dumps(comparison) + "\n", encoding="ascii")
    comparison_receipt_path = output / "comparison.receipt.json"
    comparison_receipt_path.write_text("{}\n", encoding="ascii")
    receipt = {
        "argv": evidence._expected_comparison_argv(
            root=tmp_path, artifact_file=comparison_path,
            artifact=comparison, python=sys.executable,
        ),
        "artifact": {
            "path": comparison_path.relative_to(tmp_path).as_posix(),
            "sha256": replay._digest_file(comparison_path),
        },
        "env_allowlist": {"SIM_BACKEND": "numpy"},
        "source": source,
    }
    receipt["argv"][-1] = str((tmp_path / "wrong-comparison.json").resolve())
    monkeypatch.setattr(
        evidence.execution_receipt, "verify_receipt", lambda *_args: receipt,
    )

    with pytest.raises(evidence.EvidenceError, match="frozen compare command"):
        evidence.finalize_evidence(
            root=tmp_path,
            artifact_path=comparison_path.relative_to(tmp_path).as_posix(),
            receipt_path=comparison_receipt_path.relative_to(tmp_path).as_posix(),
            out=f"{evidence.OUTPUT_DIR}/evidence-manifest.json",
        )
