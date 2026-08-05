"""Tests for complete immutable pool source manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.pool.provisioning.source_manifest import (
    SourceManifestError,
    verify_manifest,
    write_manifest,
)


def _tree(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "source"
    (root / "tools").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "tools/run.py").write_text("VALUE = 1\n", encoding="ascii")
    (root / "docs/readme.md").write_text("source\n", encoding="ascii")
    manifest = tmp_path / "manifest.sha256"
    digest = write_manifest(root, manifest)
    return root, manifest, digest


def test_complete_manifest_verifies_exact_tree(tmp_path: Path) -> None:
    root, manifest, digest = _tree(tmp_path)
    result = verify_manifest(root, manifest, digest)
    assert result["source_manifest_verified"] is True
    assert result["source_file_count"] == 2


@pytest.mark.parametrize("mutation", ["add", "remove", "modify"])
def test_added_missing_and_modified_source_fail_closed(tmp_path: Path, mutation: str) -> None:
    root, manifest, digest = _tree(tmp_path)
    if mutation == "add":
        (root / "tools/injected.py").write_text("ATTACK = 1\n", encoding="ascii")
        message = "file set differs"
    elif mutation == "remove":
        (root / "tools/run.py").unlink()
        message = "file set differs"
    else:
        (root / "tools/run.py").write_text("VALUE = 2\n", encoding="ascii")
        message = "digest mismatch"
    with pytest.raises(SourceManifestError, match=message):
        verify_manifest(root, manifest, digest)


def test_manifest_file_digest_and_symlink_fail_closed(tmp_path: Path) -> None:
    root, manifest, digest = _tree(tmp_path)
    with pytest.raises(SourceManifestError, match="file digest"):
        verify_manifest(root, manifest, "0" * 64)
    (root / "tools/link.py").symlink_to(root / "tools/run.py")
    with pytest.raises(SourceManifestError, match="non-regular"):
        verify_manifest(root, manifest, digest)


def test_nested_venv_named_directory_is_authoritative_source(tmp_path: Path) -> None:
    root, manifest, digest = _tree(tmp_path)
    nested = root / "tools/.venv/injected.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("ATTACK = 1\n", encoding="ascii")
    with pytest.raises(SourceManifestError, match="file set differs"):
        verify_manifest(root, manifest, digest)
