from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from research.runners import _v13_backend_neutral_izh_arithmetic_replay as replay
from tools import v13_backend_neutral_izh_arithmetic_replay_evidence as evidence


def test_locked_spec_declares_no_seed_non_executed_protocol():
    spec = replay.load_locked_spec()

    assert spec["device"] == "not_applicable_non_executed_protocol"
    assert spec["promotion_value"] == "none"
    assert spec["scientific_verdict"] is None
    assert spec["rng"]["measured_replay_allowed"] is False
    assert spec["rng"]["measured_replay_seed"] is None
    assert spec["replay"]["total_steps"] == 1_200
    assert spec["execution"]["trajectory_arrays"] == ["v", "u", "spikes"]


def test_locked_spec_digest_rejects_drift(tmp_path):
    path = tmp_path / "spec.json"
    path.write_bytes(replay.SPEC_PATH.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="locked spec digest mismatch"):
        replay.load_locked_spec(path, replay.SPEC_SHA256)


def test_source_set_binds_every_simulator_python_and_replay_authorities():
    paths = set(replay.source_paths())
    simulator = {
        path.relative_to(replay.ROOT).as_posix()
        for path in (replay.ROOT / "sim").rglob("*.py")
        if path.is_file()
    }

    assert simulator.issubset(paths)
    assert "research/runners/_v13_backend_state_transplant.py" in paths
    assert "research/findings/raw/v13_backend_state_transplant/bundle-numpy.json" in paths
    assert "research/findings/raw/v13_backend_state_transplant/receipt-bundle-numpy.json" in paths
    assert "tools/v13_backend_neutral_izh_arithmetic_replay_evidence.py" in paths


def test_completed_input_is_exactly_the_sealed_numpy_origin_bundle():
    spec = replay.load_locked_spec()
    bundle, transplant_spec = replay.load_completed_input(spec)

    assert bundle["origin"] == "numpy"
    assert bundle["artifact_sha256"] == spec["completed_input"]["bundle"][
        "artifact_sha256"
    ]
    assert transplant_spec["steps"] == {
        "baseline": 500, "inhibition": 200, "release": 500, "dt_ms": 1.0,
    }


def test_completed_input_rejects_wrong_frozen_digest():
    spec = replay.load_locked_spec()
    changed = json.loads(json.dumps(spec))
    changed["completed_input"]["bundle"]["file_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="completed bundle file digest mismatch"):
        replay.load_completed_input(changed)


def test_rng_guard_fails_on_call_and_restores_api():
    original = np.random.random

    with replay._forbid_rng_calls() as audit:
        with pytest.raises(RuntimeError, match="RNG call forbidden"):
            np.random.random()
        assert audit["calls_observed"] == 1
        assert audit["active"] is True

    assert audit["active"] is False
    assert np.random.random is original


def test_first_difference_reports_late_exact_cell():
    left = np.zeros((1_200, 60), dtype=np.float32)
    right = left.copy()
    right[1_199, 59] = np.float32(1.0)

    assert replay._first_difference(left, right) == {
        "kind": "value",
        "step_index": 1_199,
        "step_number": 1_200,
        "cell": 59,
        "numpy": 0.0,
        "cupy": 1.0,
    }


def _cell_value(backend: str, source: dict, changed: bool = False) -> dict:
    arrays = {
        "v": np.zeros((1_200, 60), dtype=np.float32),
        "u": np.zeros((1_200, 60), dtype=np.float32),
        "spikes": np.zeros((1_200, 60), dtype=np.bool_),
    }
    if changed:
        arrays["u"][1_199, 59] = np.float32(1.0)
    encoded = {name: replay.transplant._encode_array(value) for name, value in arrays.items()}
    return replay._seal({
        "schema": replay.SCHEMA_CELL,
        "promotion_value": "none",
        "diagnostic_only": True,
        "scientific_verdict": None,
        "backend": backend,
        "spec_sha256": replay.SPEC_SHA256,
        "source": source,
        "completed_input": {
            "bundle_path": "bundle.json",
            "bundle_file_sha256": "1" * 64,
            "bundle_artifact_sha256": "2" * 64,
            "aggregate_file_sha256": "3" * 64,
            "historical_source_revision": "4" * 40,
        },
        "measured_replay_rng": {
            "allowed": False, "seed": None, "calls_observed": 0,
        },
        "simulation_steps_executed": 1_200,
        "trajectories": encoded,
        "trajectory_sha256": {name: value["sha256"] for name, value in encoded.items()},
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


def _receipt_for(root: Path, source: dict, path: Path, backend: str) -> dict:
    return {
        "argv": replay._expected_cell_argv(
            root=root, artifact=path, backend=backend, source=source,
            python=sys.executable,
        ),
        "artifact": {
            "path": path.relative_to(root).as_posix(),
            "sha256": replay._digest_file(path),
        },
        "env_allowlist": {"SIM_BACKEND": backend},
        "source": source,
    }


@pytest.mark.parametrize(
    ("changed", "expected_outcome"),
    [(False, "DIAGNOSTIC_PASS"), (True, "DIAGNOSTIC_FAIL")],
)
def test_compare_cells_checks_all_rows_without_running_simulation(
    tmp_path, monkeypatch, changed, expected_outcome,
):
    source = {
        "file_count": 1, "git_sha": "a" * 40, "kind": "git",
        "manifest": "source.sha256", "manifest_sha256": "b" * 64,
        "tree_sha256": "c" * 64,
    }
    numpy_artifact, numpy_receipt = _write_cell(tmp_path, "numpy", source)
    cupy_artifact, cupy_receipt = _write_cell(tmp_path, "cupy", source, changed)
    receipts = {
        numpy_receipt: _receipt_for(tmp_path, source, numpy_artifact, "numpy"),
        cupy_receipt: _receipt_for(tmp_path, source, cupy_artifact, "cupy"),
    }
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    monkeypatch.setattr(replay, "load_locked_spec", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        replay.execution_receipt, "verify_receipt",
        lambda _root, relative: receipts[tmp_path / relative],
    )
    out = tmp_path / "comparison.json"

    result = replay.compare_cells(
        numpy_artifact=numpy_artifact, numpy_receipt=numpy_receipt,
        cupy_artifact=cupy_artifact, cupy_receipt=cupy_receipt, out=out,
    )

    assert result["outcome"] == expected_outcome
    assert result["scientific_verdict"] is None
    assert result["promotion_value"] == "none"
    assert result["trajectory_comparisons"]["v"]["all_1200_rows_exact"] is True
    if changed:
        assert result["trajectory_comparisons"]["u"]["first_difference"][
            "step_number"
        ] == 1_200
    with pytest.raises(FileExistsError):
        replay.compare_cells(
            numpy_artifact=numpy_artifact, numpy_receipt=numpy_receipt,
            cupy_artifact=cupy_artifact, cupy_receipt=cupy_receipt, out=out,
        )


def test_load_cell_rejects_receipt_environment_mismatch(tmp_path, monkeypatch):
    source = {"git_sha": "a" * 40, "manifest": "source.sha256"}
    artifact, receipt_path = _write_cell(tmp_path, "numpy", source)
    receipt = _receipt_for(tmp_path, source, artifact, "numpy")
    receipt["env_allowlist"] = {"SIM_BACKEND": "cupy"}
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    monkeypatch.setattr(
        replay.execution_receipt, "verify_receipt", lambda *_args: receipt,
    )

    with pytest.raises(ValueError, match="does not bind its cell"):
        replay._load_cell(artifact, receipt_path, "numpy")


def test_load_cell_rejects_tampered_frozen_argv(tmp_path, monkeypatch):
    source = {"git_sha": "a" * 40, "manifest": "source.sha256"}
    artifact, receipt_path = _write_cell(tmp_path, "numpy", source)
    receipt = _receipt_for(tmp_path, source, artifact, "numpy")
    receipt["argv"][-1] = str((tmp_path / "other.json").resolve())
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    monkeypatch.setattr(
        replay.execution_receipt, "verify_receipt", lambda *_args: receipt,
    )

    with pytest.raises(ValueError, match="differs from the frozen run command"):
        replay._load_cell(artifact, receipt_path, "numpy")


def test_evidence_inner_commands_have_no_seed_and_bind_both_receipts(tmp_path):
    paths = evidence._paths()
    run = evidence._inner_command(
        root=tmp_path, action="run_cupy", revision="a" * 40,
        paths=paths, python=sys.executable,
    )
    compare = evidence._inner_command(
        root=tmp_path, action="compare", revision="a" * 40,
        paths=paths, python=sys.executable,
    )

    assert "--run" in run and "--backend" in run and "cupy" in run
    assert "--source-manifest" in run and "--source-revision" in run
    assert not any("seed" in argument.lower() for argument in run)
    assert "--numpy-receipt" in compare
    assert "--cupy-receipt" in compare


def test_emit_command_is_create_only_and_compare_requires_complete_cells(
    tmp_path, monkeypatch,
):
    paths = evidence._paths()
    output_dir = tmp_path / evidence.OUTPUT_DIR
    output_dir.mkdir(parents=True)
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
    monkeypatch.setattr(evidence.replay, "source_paths", lambda _root: ("authority",))
    envelope_path = f"{evidence.OUTPUT_DIR}/run-numpy.command.json"

    envelope = evidence.emit_command(
        root=tmp_path, action="run_numpy", revision=revision,
        host="local3090", device="CPU-NumPy", out=envelope_path,
        python=sys.executable,
    )

    assert envelope["execution"] == "not_executed"
    assert envelope["promotion_value"] == "none"
    assert envelope["env"] == {"SIM_BACKEND": "numpy"}
    with pytest.raises(evidence.EvidenceError, match="refusing stale output path"):
        evidence.emit_command(
            root=tmp_path, action="run_numpy", revision=revision,
            host="local3090", device="CPU-NumPy", out=envelope_path,
            python=sys.executable,
        )
    with pytest.raises(evidence.EvidenceError, match="missing prerequisite"):
        evidence.emit_command(
            root=tmp_path, action="compare", revision=revision,
            host="local3090", device="comparison-only", out="compare.command.json",
            python=sys.executable,
        )


def test_freeze_source_manifest_requires_committed_bytes(tmp_path, monkeypatch):
    (tmp_path / "sim").mkdir()
    source = tmp_path / "sim/input.py"
    source.write_text("VALUE = 1\n", encoding="ascii")
    output = tmp_path / "evidence"
    output.mkdir()
    revision = "a" * 40
    monkeypatch.setattr(evidence.shared, "_git_head", lambda _root: revision)
    monkeypatch.setattr(
        evidence.shared, "_tracked_sim_python", lambda *_args: {"sim/input.py"},
    )
    monkeypatch.setattr(
        evidence.shared, "_git_file", lambda *_args: b"VALUE = 2\n",
    )
    monkeypatch.setattr(
        evidence.replay, "source_paths", lambda _root: ("sim/input.py",),
    )

    with pytest.raises(evidence.EvidenceError, match="differs from committed revision"):
        evidence.freeze_source_manifest(
            root=tmp_path, revision=revision, out="evidence/source.sha256",
        )


def test_finalize_rejects_tampered_comparison_argv(tmp_path, monkeypatch):
    source = {
        "file_count": 1, "git_sha": "a" * 40, "kind": "git",
        "manifest": "source.sha256", "manifest_sha256": "b" * 64,
        "tree_sha256": "c" * 64,
    }
    numpy_artifact, numpy_receipt = _write_cell(tmp_path, "numpy", source)
    cupy_artifact, cupy_receipt = _write_cell(tmp_path, "cupy", source)
    cells = {
        "numpy": {
            "path": numpy_artifact.relative_to(tmp_path).as_posix(),
            "file_sha256": replay._digest_file(numpy_artifact),
            "receipt_path": numpy_receipt.relative_to(tmp_path).as_posix(),
            "artifact_sha256": json.loads(numpy_artifact.read_text())["sha256"],
        },
        "cupy": {
            "path": cupy_artifact.relative_to(tmp_path).as_posix(),
            "file_sha256": replay._digest_file(cupy_artifact),
            "receipt_path": cupy_receipt.relative_to(tmp_path).as_posix(),
            "artifact_sha256": json.loads(cupy_artifact.read_text())["sha256"],
        },
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
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison) + "\n", encoding="ascii")
    comparison_receipt_path = tmp_path / "comparison.receipt.json"
    comparison_receipt_path.write_text("{}\n", encoding="ascii")
    receipt = {
        "argv": evidence._expected_comparison_argv(
            root=tmp_path, artifact_file=comparison_path,
            artifact=comparison, python=sys.executable,
        ),
        "artifact": {
            "path": "comparison.json",
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
            root=tmp_path, artifact_path="comparison.json",
            receipt_path="comparison.receipt.json", out="evidence-manifest.json",
        )
