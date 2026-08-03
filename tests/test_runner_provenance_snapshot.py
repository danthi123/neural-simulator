"""Pool snapshots retain and verify source identity without a Git checkout."""

import hashlib
import json

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
