"""Pool snapshots retain and verify source identity without a Git checkout."""

import hashlib
import json

import pytest

import research.runners as provenance


def _write_archive_snapshot(root, files):
    lines = []
    for relative_path, content in sorted(files.items()):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        lines.append(f"{hashlib.sha256(content.encode()).hexdigest()}  {relative_path}\n")
    manifest = "".join(lines)
    (root / ".source_manifest.sha256").write_text(manifest, encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest.encode()).hexdigest()
    (root / ".source_revision").write_text(
        "git_sha=0123456789abcdef\n"
        "source_kind=git_archive\n"
        f"source_manifest_sha256={manifest_hash}\n",
        encoding="utf-8",
    )
    return manifest_hash


def test_git_head_falls_back_to_clean_source_snapshot(tmp_path, monkeypatch):
    (tmp_path / ".source_revision").write_text(
        "git_sha=0123456789abcdef\n"
        "source_kind=git_archive\n"
        "source_manifest_sha256=abc123\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(provenance, "_ROOT", str(tmp_path))

    sha, dirty = provenance._git_head()

    assert sha == "0123456789abcdef"
    assert dirty is False
    assert provenance._source_snapshot() == {
        "git_sha": "0123456789abcdef",
        "source_kind": "git_archive",
        "source_manifest_sha256": "abc123",
    }


def test_output_sidecar_carries_snapshot_identity(tmp_path, monkeypatch):
    raw = tmp_path / "research" / "findings" / "raw"
    raw.mkdir(parents=True)
    artifact = raw / "result.json"
    artifact.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(provenance, "_ROOT", str(tmp_path))
    monkeypatch.setattr(provenance, "_RAW_DIR", str(raw))
    monkeypatch.setattr(provenance, "_START", 0.0)
    rec = {
        "run_id": "pool-test",
        "runner": "research/runners/example.py",
        "argv": ["example.py", "--out", str(artifact)],
        "git_sha": "0123456789abcdef",
        "git_dirty": False,
        "source_kind": "git_archive",
        "source_manifest_sha256": "abc123",
        "source_manifest_verified": True,
        "source_manifest_verification_error": None,
        "started": "2026-08-03T13:00:00",
        "cwd": str(tmp_path),
        "env": {"POOL_CHECKED_REASON": "prior record read"},
    }

    assert provenance._stamp_outputs(rec) == [str(artifact)]
    sidecar = json.loads((raw / "result.json.prov.json").read_text())
    assert sidecar["source_kind"] == "git_archive"
    assert sidecar["source_manifest_sha256"] == "abc123"
    assert sidecar["source_manifest_verified_at_start"] is True
    assert sidecar["source_manifest_verified_at_exit"] is False
    assert sidecar["source_manifest_exit_error"]
    assert sidecar["env"]["POOL_CHECKED_REASON"] == "prior record read"
    assert "schema" not in sidecar
    assert "started_utc_ns" not in sidecar
    assert "ended_utc_ns" not in sidecar


def _set_v2_identity(monkeypatch, *, source_kind="git"):
    monkeypatch.setenv("SIM_PROVENANCE_V2", "1")
    monkeypatch.setenv("SIM_PROVENANCE_RUN_ID", "receipt-owned-run")
    monkeypatch.setenv("SIM_PROVENANCE_SOURCE_KIND", source_kind)
    monkeypatch.setenv("SIM_PROVENANCE_SOURCE_MANIFEST_SHA256", "b" * 64)


@pytest.mark.parametrize(
    "missing",
    (
        "SIM_PROVENANCE_RUN_ID",
        "SIM_PROVENANCE_SOURCE_KIND",
        "SIM_PROVENANCE_SOURCE_MANIFEST_SHA256",
    ),
)
def test_v2_record_requires_private_identity(tmp_path, monkeypatch, missing):
    _set_v2_identity(monkeypatch)
    monkeypatch.delenv(missing)
    monkeypatch.setattr(provenance, "_PROV_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="requires private provenance identity"):
        provenance._record_start()


def test_v2_record_uses_full_identity_and_redacts_private_environment(tmp_path, monkeypatch):
    _set_v2_identity(monkeypatch)
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    monkeypatch.delenv("SIM_RUN_ID", raising=False)
    monkeypatch.setattr(provenance, "_PROV_DIR", str(tmp_path))
    monkeypatch.setattr(provenance, "_git_head", lambda full=False: ("a" * 40, False))

    rec = provenance._record_start()

    assert rec["run_id"] == "receipt-owned-run"
    assert rec["git_sha"] == "a" * 40
    assert rec["source_kind"] == "git"
    assert rec["source_manifest_sha256"] == "b" * 64
    assert rec["provenance_schema"] == "sim-run-provenance-v2"
    assert isinstance(rec["started_utc_ns"], int)
    assert rec["started_utc_ns"] > 0
    assert rec["env"]["SIM_BACKEND"] == "numpy"
    assert not any(key.startswith("SIM_PROVENANCE_") for key in rec["env"])
    assert provenance.os.environ["SIM_RUN_ID"] == "receipt-owned-run"


def test_v2_git_sidecar_has_timing_identity_and_null_archive_verification(tmp_path, monkeypatch):
    raw = tmp_path / "research" / "findings" / "raw"
    raw.mkdir(parents=True)
    artifact = raw / "result.json"
    artifact.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(provenance, "_ROOT", str(tmp_path))
    monkeypatch.setattr(provenance, "_RAW_DIR", str(raw))
    monkeypatch.setattr(provenance, "_START", 0.0)
    rec = {
        "run_id": "receipt-owned-run",
        "runner": "research/runners/example.py",
        "argv": ["example.py", "--out", str(artifact)],
        "git_sha": "a" * 40,
        "git_dirty": True,
        "source_kind": "git",
        "source_manifest_sha256": "b" * 64,
        "started": "2026-08-04T13:00:00",
        "started_utc_ns": 123456789,
        "provenance_schema": "sim-run-provenance-v2",
        "cwd": str(tmp_path),
        "env": {"SIM_BACKEND": "numpy"},
    }

    assert provenance._stamp_outputs(rec) == [str(artifact)]

    sidecar = json.loads((raw / "result.json.prov.json").read_text())
    assert sidecar["schema"] == "sim-run-provenance-v2"
    assert sidecar["run_id"] == "receipt-owned-run"
    assert sidecar["git_sha"] == "a" * 40
    assert sidecar["source_kind"] == "git"
    assert sidecar["source_manifest_sha256"] == "b" * 64
    assert sidecar["started_utc_ns"] == 123456789
    assert sidecar["ended_utc_ns"] >= sidecar["started_utc_ns"]
    assert sidecar["source_manifest_verified_at_start"] is None
    assert sidecar["source_manifest_start_error"] is None
    assert sidecar["source_manifest_verified_at_exit"] is None
    assert sidecar["source_manifest_exit_error"] is None
    assert not any(key.startswith("SIM_PROVENANCE_") for key in sidecar["env"])
    assert "exit_code" not in sidecar


def test_v2_archive_sidecar_records_start_and_exit_verification(tmp_path, monkeypatch):
    files = {
        "research/__init__.py": "\n",
        "research/runners/example.py": "VALUE = 1\n",
    }
    manifest_hash = _write_archive_snapshot(tmp_path, files)
    revision = "a" * 40
    (tmp_path / ".source_revision").write_text(
        f"git_sha={revision}\n"
        "source_kind=git_archive\n"
        f"source_manifest_sha256={manifest_hash}\n",
        encoding="utf-8",
    )
    raw = tmp_path / "research" / "findings" / "raw"
    raw.mkdir(parents=True)
    artifact = raw / "result.json"
    _set_v2_identity(monkeypatch, source_kind="git_archive")
    monkeypatch.setenv("SIM_PROVENANCE_SOURCE_MANIFEST_SHA256", manifest_hash)
    monkeypatch.setattr(provenance, "_ROOT", str(tmp_path))
    monkeypatch.setattr(provenance, "_RAW_DIR", str(raw))
    monkeypatch.setattr(provenance, "_PROV_DIR", str(raw / "_provenance"))
    monkeypatch.setattr(provenance, "_git_head", lambda full=False: (revision, False))

    rec = provenance._record_start()
    artifact.write_text("{}", encoding="utf-8")
    rec["argv"] = ["example.py", "--out", str(artifact)]

    assert provenance._stamp_outputs(rec) == [str(artifact)]

    sidecar = json.loads((raw / "result.json.prov.json").read_text())
    assert sidecar["source_manifest_verified_at_start"] is True
    assert sidecar["source_manifest_start_error"] is None
    assert sidecar["source_manifest_verified_at_exit"] is True
    assert sidecar["source_manifest_exit_error"] is None


def test_archive_manifest_verification_rejects_tampering_and_extra_source(tmp_path, monkeypatch):
    snapshot_hash = _write_archive_snapshot(
        tmp_path,
        {
            "research/__init__.py": "\n",
            "research/runners/example.py": "VALUE = 1\n",
        },
    )
    monkeypatch.setattr(provenance, "_ROOT", str(tmp_path))
    assert provenance.verify_immutable_source_manifest()["source_manifest_verified"] is True

    (tmp_path / "research/runners/example.py").write_text("VALUE = 2\n", encoding="utf-8")
    result = provenance.verify_immutable_source_manifest()
    assert result["source_manifest_verified"] is False
    assert "digest mismatch" in result["source_manifest_verification_error"]

    (tmp_path / "research/runners/example.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "tools/extra.py").parent.mkdir(parents=True)
    (tmp_path / "tools/extra.py").write_text("pass\n", encoding="utf-8")
    result = provenance.verify_immutable_source_manifest()
    assert result["source_manifest_verified"] is False
    assert "file set differs" in result["source_manifest_verification_error"]

    (tmp_path / "tools/extra.py").unlink()
    (tmp_path / "research/__init__.py").write_text("MUTATED = True\n", encoding="utf-8")
    result = provenance.verify_immutable_source_manifest()
    assert result["source_manifest_verified"] is False
    assert "research/__init__.py" in result["source_manifest_verification_error"]

    captured = {
        "source_kind": "git_archive",
        "source_manifest_sha256": snapshot_hash,
    }
    _write_archive_snapshot(
        tmp_path,
        {
            "research/__init__.py": "REPLACED = True\n",
            "research/runners/example.py": "VALUE = 3\n",
        },
    )
    result = provenance.verify_immutable_source_manifest(captured)
    assert result["source_manifest_verified"] is False
    assert "manifest file digest" in result["source_manifest_verification_error"]


def test_archive_manifest_accepts_bound_specs_and_ancestry_attestation(tmp_path, monkeypatch):
    _write_archive_snapshot(
        tmp_path,
        {
            ".source_ancestry.json": '{"schema":"sim-source-ancestry-v1"}\n',
            "research/__init__.py": "\n",
            "research/runners/example.py": "VALUE = 1\n",
            "research/specs/locked.json": '{}\n',
            "tools/pool/provisioning/ancestry_attestation.py": "SCHEMA = 'test'\n",
        },
    )
    monkeypatch.setattr(provenance, "_ROOT", str(tmp_path))

    result = provenance.verify_immutable_source_manifest()

    assert result == {
        "source_manifest_verified": True,
        "source_manifest_verification_error": None,
    }


def test_archive_manifest_accepts_deployed_test_helpers(tmp_path, monkeypatch):
    _write_archive_snapshot(
        tmp_path,
        {
            "research/__init__.py": "\n",
            "research/runners/example.py": "VALUE = 1\n",
            "tests/_capture_helper.py": "VALUE = 2\n",
        },
    )
    monkeypatch.setattr(provenance, "_ROOT", str(tmp_path))

    assert provenance.verify_immutable_source_manifest() == {
        "source_manifest_verified": True,
        "source_manifest_verification_error": None,
    }
