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
