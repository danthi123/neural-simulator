from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from tools import v13_legacy_source_package as package


ROOT = Path(__file__).resolve().parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob(revision: str, relative: str) -> bytes:
    return bytes(package._git(ROOT, "show", f"{revision}:{relative}", binary=True))


@pytest.fixture
def built(tmp_path: Path) -> Path:
    output = tmp_path / "legacy-source"
    result = package.build_package(root=ROOT, output=output)
    assert result["status"] == "verified"
    return output


def test_builds_exact_audited_git_object_package(built: Path) -> None:
    lock = json.loads((built / package.LOCK_NAME).read_text(encoding="ascii"))
    manifest = (built / package.MANIFEST_NAME).read_bytes()

    assert lock["sha256"] == package._canonical_digest(lock)
    assert lock["base"]["revision"] == package.BASE_REVISION
    assert lock["candidate"]["revision"] == package.CANDIDATE_REVISION
    assert lock["overlay"] == {
        "path": package.OVERLAY_PATH,
        "source_revision": package.CANDIDATE_REVISION,
        "sha256": _sha256(_git_blob(package.CANDIDATE_REVISION, package.OVERLAY_PATH)),
    }
    assert lock["execution"] == {
        "runner": package.OVERLAY_PATH,
        "runner_module": package.RUNNER_MODULE,
        "python_flags": ["-B"],
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "SIM_BACKEND": "cupy",
            "SIM_NO_PROVENANCE": "1",
        },
        "working_directory": ".",
        "output_directory": package.RUN_DIRECTORY,
        "source_identity": lock["execution"]["source_identity"],
    }
    assert lock["execution"]["source_identity"]["reported_revision"] == (
        package.BASE_REVISION
    )
    assert [item["path"] for item in lock["files"]] == list(package.SOURCE_PATHS)
    assert len(lock["files"]) == len(package.BASE_PATHS) + 1

    for record in lock["files"]:
        revision = (
            package.CANDIDATE_REVISION
            if record["path"] == package.OVERLAY_PATH
            else package.BASE_REVISION
        )
        expected = _git_blob(revision, record["path"])
        assert (built / record["path"]).read_bytes() == expected
        assert record["sha256"] == _sha256(expected)
        assert record["source_revision"] == revision

    assert manifest.decode("utf-8").splitlines() == [
        f"{item['sha256']}  {item['path']}" for item in lock["files"]
    ]
    actual = {
        path.relative_to(built).as_posix()
        for path in built.rglob("*")
        if path.is_file()
    }
    assert actual == {
        *package.SOURCE_PATHS,
        package.IDENTITY_HEAD,
        package.IDENTITY_CONFIG,
        package.IDENTITY_OBJECT,
        package.MANIFEST_NAME,
        package.LOCK_NAME,
    }
    assert (built / package.RUN_DIRECTORY).is_dir()
    assert not any((built / package.RUN_DIRECTORY).iterdir())
    assert package.verify_package(built)["source_file_count"] == len(package.SOURCE_PATHS)


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_refuses_every_existing_output(tmp_path: Path, kind: str) -> None:
    output = tmp_path / "existing"
    if kind == "file":
        output.write_text("occupied\n")
    elif kind == "directory":
        output.mkdir()
    else:
        output.symlink_to(tmp_path / "missing-target")

    with pytest.raises(package.PackageError, match="refusing existing output"):
        package.build_package(root=ROOT, output=output)


def test_refuses_symlinked_output_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(package.PackageError, match="symlink ancestor"):
        package.build_package(root=ROOT, output=linked / "package")


def test_rejects_any_unexpected_overlay_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "legacy-source"
    overlays = dict(package.EXPECTED_OVERLAYS)
    overlays["sim/bridge.py"] = package.CANDIDATE_REVISION

    with pytest.raises(package.PackageError, match="only the audited"):
        package.build_package(root=ROOT, output=output, overlays=overlays)
    assert not os.path.lexists(output)


def test_rejects_symlink_git_entry_and_removes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "legacy-source"
    original = package._git_entry

    def fake_entry(root: Path, revision: str, relative: str) -> dict[str, str]:
        if relative == package.BASE_PATHS[0]:
            raise package.PackageError(
                f"source must be a regular Git blob: {relative} (120000 blob)"
            )
        return original(root, revision, relative)

    monkeypatch.setattr(package, "_git_entry", fake_entry)
    with pytest.raises(package.PackageError, match="regular Git blob"):
        package.build_package(root=ROOT, output=output)
    assert not output.exists()


@pytest.mark.parametrize("tamper", ["extra", "source", "symlink", "lock", "mode"])
def test_verifier_fails_closed_on_package_tampering(
    built: Path, tmp_path: Path, tamper: str,
) -> None:
    if tamper == "extra":
        (built / "unexpected.py").write_text("pass\n")
        match = "file set differs"
    elif tamper == "source":
        (built / package.BASE_PATHS[0]).write_text("changed\n")
        match = "source digest mismatch"
    elif tamper == "symlink":
        target = built / package.BASE_PATHS[0]
        target.unlink()
        target.symlink_to(tmp_path / "outside.py")
        match = "symlink or non-regular"
    elif tamper == "lock":
        lock_path = built / package.LOCK_NAME
        lock = json.loads(lock_path.read_text())
        lock["status"] = "changed"
        lock_path.write_text(json.dumps(lock))
        match = "schema or status|self-digest"
    else:
        target = built / package.BASE_PATHS[0]
        target.chmod(0o755)
        match = "source mode mismatch"

    with pytest.raises(package.PackageError, match=match):
        package.verify_package(built)


def test_summary_contains_no_seed_material(built: Path) -> None:
    summary = package.verify_package(built)
    rendered = json.dumps(summary, sort_keys=True).lower()
    assert "seed" not in rendered


def test_seed_free_execution_probe_uses_package_identity_without_bytecode(
    built: Path,
) -> None:
    result = package.probe_execution_package(built)

    assert result == {
        "schema": package.LOCK_SCHEMA,
        "status": "execution_controls_verified",
        "lock_sha256": package.verify_package(built)["lock_sha256"],
        "manifest_sha256": package.verify_package(built)["manifest_sha256"],
        "reported_source_revision": package.BASE_REVISION,
        "python_no_bytecode": True,
        "runtime_file_count": 0,
    }
    assert not list(built.rglob("__pycache__"))
    assert not list(built.rglob("*.pyc"))


@pytest.mark.parametrize("location", ["source", "runtime"])
def test_verifier_rejects_python_bytecode_anywhere(
    built: Path, location: str,
) -> None:
    if location == "source":
        cache = built / "sim" / "__pycache__"
    else:
        cache = built / package.RUN_DIRECTORY / "__pycache__"
    cache.mkdir()
    (cache / "unsafe.pyc").write_bytes(b"bytecode")

    with pytest.raises(package.PackageError, match="bytecode cache|runtime directory"):
        package.verify_package(built)


def test_executor_controls_identity_flags_and_isolated_output(
    built: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run = package.subprocess.run
    observed: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        if package.RUNNER_MODULE not in argv:
            return original_run(argv, **kwargs)
        observed.update(argv=list(argv), kwargs=kwargs)
        output = Path(argv[argv.index("--out") + 1])
        output.write_text(json.dumps({
            "stage": "legacy_performance_baseline",
            "source_sha": package.BASE_REVISION,
            "outcome": "BASELINE_RECORDED",
            "backend": "cupy",
            "device": "NVIDIA GeForce RTX 3090",
            "median_seconds": 1.0,
        }))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(package.subprocess, "run", fake_run)
    result = package.execute_legacy_baseline(built, python="/exact/python")

    assert observed["argv"] == [
        "/exact/python", "-B", "-m", package.RUNNER_MODULE,
        "--legacy-performance-baseline", "--out",
        str(built / package.RUN_DIRECTORY / package.LEGACY_OUTPUT_NAME),
    ]
    kwargs = observed["kwargs"]
    assert kwargs["cwd"] == built
    assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert kwargs["env"]["SIM_BACKEND"] == "cupy"
    assert kwargs["env"]["SIM_NO_PROVENANCE"] == "1"
    assert kwargs["env"]["GIT_DIR"] == str(built / package.IDENTITY_DIRECTORY)
    assert "PYTHONPATH" not in kwargs["env"]
    assert result["artifact"]["path"] == (
        f"{package.RUN_DIRECTORY}/{package.LEGACY_OUTPUT_NAME}"
    )
    assert set(package._runtime_files(built)) == {package.LEGACY_OUTPUT_NAME}


def test_executor_requires_an_empty_isolated_runtime(built: Path) -> None:
    (built / package.RUN_DIRECTORY / "occupied.json").write_text("{}\n")

    with pytest.raises(package.PackageError, match="empty runtime"):
        package.execute_legacy_baseline(built)


def _write_legacy_artifact(built: Path) -> bytes:
    value = {
        "stage": "legacy_performance_baseline",
        "source_sha": package.BASE_REVISION,
        "outcome": "BASELINE_RECORDED",
        "backend": "cupy",
        "device": "NVIDIA GeForce RTX 3090",
        "median_seconds": 1.0,
    }
    data = (json.dumps(value, sort_keys=True) + "\n").encode("ascii")
    (built / package.RUN_DIRECTORY / package.LEGACY_OUTPUT_NAME).write_bytes(data)
    return data


def test_transfer_is_create_only_and_binds_package_manifest_and_artifact(
    built: Path, tmp_path: Path,
) -> None:
    artifact_bytes = _write_legacy_artifact(built)
    evidence = tmp_path / "candidate-evidence"
    evidence.mkdir()

    result = package.transfer_legacy_artifact(
        built, candidate_evidence=evidence, transfer_name="legacy-baseline-v2",
    )
    transfer = evidence / "legacy-baseline-v2"
    manifest = json.loads(
        (transfer / package.TRANSFER_MANIFEST_NAME).read_text(encoding="ascii")
    )

    assert manifest == result
    assert manifest["sha256"] == package._canonical_digest(manifest)
    assert manifest["package"]["lock_file_sha256"] == _sha256(
        (built / package.LOCK_NAME).read_bytes()
    )
    assert manifest["package"]["source_manifest_sha256"] == _sha256(
        (built / package.MANIFEST_NAME).read_bytes()
    )
    assert manifest["artifact"]["sha256"] == _sha256(artifact_bytes)
    assert (transfer / package.LEGACY_OUTPUT_NAME).read_bytes() == artifact_bytes

    with pytest.raises(package.PackageError, match="existing transfer destination"):
        package.transfer_legacy_artifact(
            built, candidate_evidence=evidence, transfer_name="legacy-baseline-v2",
        )


def test_transfer_rejects_symlinked_candidate_evidence(
    built: Path, tmp_path: Path,
) -> None:
    _write_legacy_artifact(built)
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(package.PackageError, match="symlink ancestor"):
        package.transfer_legacy_artifact(
            built, candidate_evidence=linked, transfer_name="legacy-baseline-v2",
        )


def test_transfer_rejects_wrong_artifact_identity(built: Path, tmp_path: Path) -> None:
    artifact = built / package.RUN_DIRECTORY / package.LEGACY_OUTPUT_NAME
    artifact.write_text(json.dumps({
        "stage": "legacy_performance_baseline", "source_sha": "0" * 40,
    }))
    evidence = tmp_path / "candidate-evidence"
    evidence.mkdir()

    with pytest.raises(package.PackageError, match="wrong source"):
        package.transfer_legacy_artifact(
            built, candidate_evidence=evidence, transfer_name="legacy-baseline-v2",
        )
