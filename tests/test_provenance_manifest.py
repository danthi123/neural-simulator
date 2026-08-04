"""Focused tests for the non-ledger provenance manifest CLI."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import provenance_manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(
    root: Path,
    name: str,
    *,
    run_id: str,
    declared: str | None = None,
    backend: str = "cupy",
    sidecar_device: str | None = None,
) -> tuple[Path, Path]:
    artifact = root / name
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps({"runtime": {"backend": backend, "device": "RTX 3090"}, "value": run_id}),
        encoding="utf-8",
    )
    sidecar = Path(str(artifact) + ".prov.json")
    record = {
        "run_id": run_id,
        "artifact": declared if declared is not None else name,
        "git_sha": "0123456789abcdef",
        "git_dirty": False,
        "source_kind": "git_archive",
        "source_manifest_sha256": "a" * 64,
        "sim_backend": backend,
        "argv": ["python", "runner.py", "--out", name],
        "started": "2026-08-04T06:23:05+00:00",
    }
    if sidecar_device is not None:
        record["device"] = sidecar_device
    sidecar.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return artifact, sidecar


def test_cli_writes_sorted_deterministic_manifest_with_byte_hashes(tmp_path, capsys):
    root = tmp_path / "experiment"
    root.mkdir()
    artifact_b, sidecar_b = _artifact(root, "nested/b.json", run_id="run-b")
    artifact_a, sidecar_a = _artifact(
        root, "a.json", run_id="run-a", sidecar_device="cuda:0 / NVIDIA GeForce RTX 3090"
    )
    first = tmp_path / "manifest-one.json"
    second = tmp_path / "manifest-two.json"

    args = [
        str(root), "--artifact", "nested/b.json", "--artifact", "a.json", "--output", str(first)
    ]
    assert provenance_manifest.main(args) == 0
    assert provenance_manifest.main([*args[:-1], str(second)]) == 0

    assert first.read_bytes() == second.read_bytes()
    manifest = json.loads(first.read_text(encoding="utf-8"))
    assert manifest["schema"] == "sim-provenance-manifest-v1"
    assert manifest["artifact_root"] == "."
    assert manifest["record_count"] == 2
    assert [record["artifact"] for record in manifest["records"]] == ["a.json", "nested/b.json"]
    a_record, b_record = manifest["records"]
    assert a_record == {
        "artifact": "a.json",
        "artifact_sha256": _sha(artifact_a),
        "backend": "cupy",
        "command": ["python", "runner.py", "--out", "a.json"],
        "device": "cuda:0 / NVIDIA GeForce RTX 3090",
        "run_id": "run-a",
        "sidecar": "a.json.prov.json",
        "sidecar_sha256": _sha(sidecar_a),
        "source": {
            "dirty": False,
            "git_sha": "0123456789abcdef",
            "kind": "git_archive",
            "manifest_sha256": "a" * 64,
        },
        "timestamp": "2026-08-04T06:23:05+00:00",
    }
    assert b_record["artifact_sha256"] == _sha(artifact_b)
    assert b_record["sidecar_sha256"] == _sha(sidecar_b)
    assert b_record["device"] == "RTX 3090"
    assert "WROTE" in capsys.readouterr().out


def test_glob_selection_is_sorted_and_requires_every_sidecar(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    _artifact(root, "cells/z.json", run_id="z")
    _artifact(root, "cells/a.json", run_id="a")
    missing = root / "cells/m.json"
    missing.write_text("not required to be JSON", encoding="utf-8")

    selected = provenance_manifest.select_artifacts(root, patterns=["cells/*.json"])
    with pytest.raises(provenance_manifest.ManifestError, match="missing provenance sidecar"):
        provenance_manifest.build_manifest(root, selected)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("{not-json", "malformed sidecar"),
        (json.dumps(["not", "an", "object"]), "expected a JSON object"),
        (json.dumps({"artifact": "result.json"}), "missing backend"),
    ],
)
def test_malformed_sidecars_are_rejected(tmp_path, replacement, message):
    root = tmp_path / "experiment"
    root.mkdir()
    artifact, sidecar = _artifact(root, "result.json", run_id="run-1")
    sidecar.write_text(replacement, encoding="utf-8")

    with pytest.raises(provenance_manifest.ManifestError, match=message):
        provenance_manifest.build_manifest(root, [artifact])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_id", "", "run_id"),
        ("git_sha", None, "source identity"),
        ("sim_backend", "", "backend"),
        ("argv", [], "command/argv"),
        ("started", "not-a-timestamp", "invalid timestamp"),
    ],
)
def test_required_provenance_fields_are_validated(tmp_path, field, value, message):
    root = tmp_path / field
    root.mkdir()
    artifact, sidecar = _artifact(root, "result.json", run_id="run-1")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    record[field] = value
    sidecar.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(provenance_manifest.ManifestError, match=message):
        provenance_manifest.build_manifest(root, [artifact])


def test_missing_device_is_rejected_instead_of_inferred(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    artifact, _ = _artifact(root, "result.json", run_id="run-1")
    artifact.write_text(json.dumps({"value": 1}), encoding="utf-8")

    with pytest.raises(provenance_manifest.ManifestError, match="missing device identity"):
        provenance_manifest.build_manifest(root, [artifact])


def test_repository_relative_declared_artifact_path_is_accepted(tmp_path):
    repository = tmp_path / "repo"
    (repository / ".git").mkdir(parents=True)
    root = repository / "results"
    root.mkdir()
    artifact, _ = _artifact(
        root, "result.json", run_id="run-1", declared="results/result.json"
    )

    manifest = provenance_manifest.build_manifest(root, [artifact])

    assert manifest["records"][0]["artifact"] == "result.json"


@pytest.mark.parametrize("declared", ["other.json", "../result.json", "/tmp/result.json"])
def test_mismatched_sidecar_artifact_paths_are_rejected(tmp_path, declared):
    root = tmp_path / "experiment"
    root.mkdir()
    artifact, _ = _artifact(root, "result.json", run_id="run-1", declared=declared)

    with pytest.raises(provenance_manifest.ManifestError, match="artifact path mismatch"):
        provenance_manifest.build_manifest(root, [artifact])


def test_duplicate_and_escaping_selections_are_rejected(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    _artifact(root, "result.json", run_id="run-1")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(provenance_manifest.ManifestError, match="duplicate artifact record"):
        provenance_manifest.select_artifacts(
            root, artifacts=["result.json", "result.json"]
        )
    with pytest.raises(provenance_manifest.ManifestError, match="escapes artifact directory"):
        provenance_manifest.select_artifacts(root, artifacts=["../outside.json"])


def test_manifest_is_create_only_and_preserves_existing_bytes(tmp_path):
    output = tmp_path / "manifest.json"
    output.write_bytes(b"original\n")

    with pytest.raises(provenance_manifest.ManifestError, match="refusing to overwrite"):
        provenance_manifest.write_manifest(output, {"schema": "replacement"})
    assert output.read_bytes() == b"original\n"
