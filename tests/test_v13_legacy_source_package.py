from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

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
    assert actual == {*package.SOURCE_PATHS, package.MANIFEST_NAME, package.LOCK_NAME}
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
