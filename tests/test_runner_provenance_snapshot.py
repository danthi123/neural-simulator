"""Pool snapshots retain source identity without carrying a Git checkout."""

import research.runners as provenance


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
        "started": "2026-08-03T13:00:00",
        "cwd": str(tmp_path),
        "env": {"POOL_CHECKED_REASON": "prior record read"},
    }

    assert provenance._stamp_outputs(rec) == [str(artifact)]
    import json
    sidecar = json.loads((raw / "result.json.prov.json").read_text())
    assert sidecar["source_kind"] == "git_archive"
    assert sidecar["source_manifest_sha256"] == "abc123"
    assert sidecar["env"]["POOL_CHECKED_REASON"] == "prior record read"
