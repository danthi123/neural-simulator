"""Archive-safe source ancestry verification for scientific pool runners."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools.pool.provisioning import ancestry_attestation as ancestry


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _repository(root: Path) -> tuple[Path, str, str]:
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    (root / "source.py").write_text(f"VALUE = {root.name!r}\n", encoding="ascii")
    subprocess.run(["git", "add", "source.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "anchor"], cwd=root, check=True)
    anchor = _git(root, "rev-parse", "HEAD")
    (root / "source.py").write_text(f"VALUE = {root.name + '-head'!r}\n", encoding="ascii")
    subprocess.run(["git", "commit", "-qam", "head"], cwd=root, check=True)
    return root, anchor, _git(root, "rev-parse", "HEAD")


def _archive(repo: Path, destination: Path, head: str) -> Path:
    destination.mkdir()
    attestation_path = destination / ancestry.ATTESTATION_PATH
    ancestry.create_attestation(repo, attestation_path, head)
    (destination / "source.py").write_bytes(_git(repo, "show", f"{head}:source.py").encode())
    entries = []
    for path in (destination / ancestry.ATTESTATION_PATH, destination / "source.py"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.name}\n")
    manifest = "".join(sorted(entries)).encode("ascii")
    (destination / ancestry.MANIFEST_PATH).write_bytes(manifest)
    attestation_digest = hashlib.sha256(attestation_path.read_bytes()).hexdigest()
    (destination / ancestry.REVISION_PATH).write_text(
        "source_kind=git_archive\n"
        f"git_sha={head}\n"
        f"source_manifest_sha256={hashlib.sha256(manifest).hexdigest()}\n"
        f"source_ancestry_sha256={attestation_digest}\n",
        encoding="ascii",
    )
    return destination


def test_git_checkout_and_bound_archive_accept_same_ancestor(tmp_path):
    repo, anchor, head = _repository(tmp_path / "repo")
    archive = _archive(repo, tmp_path / "archive", head)

    checkout_result = ancestry.require_source_ancestor(repo, anchor, expected_head=head)
    archive_result = ancestry.require_source_ancestor(archive, anchor, expected_head=head)

    assert checkout_result == {
        "anchor": anchor, "git_sha": head, "is_ancestor": True, "kind": "git",
    }
    assert archive_result["anchor"] == anchor
    assert archive_result["git_sha"] == head
    assert archive_result["is_ancestor"] is True
    assert archive_result["kind"] == "git_archive"
    assert len(archive_result["source_ancestry_sha256"]) == 64


def test_nonancestor_and_wrong_expected_head_fail_closed(tmp_path):
    repo, anchor, head = _repository(tmp_path / "repo")
    archive = _archive(repo, tmp_path / "archive", head)
    unrelated_repo, _, unrelated = _repository(tmp_path / "unrelated")
    assert unrelated_repo != repo

    with pytest.raises(ancestry.AncestryError, match="not an ancestor"):
        ancestry.require_source_ancestor(repo, unrelated)
    with pytest.raises(ancestry.AncestryError, match="not an ancestor"):
        ancestry.require_source_ancestor(archive, unrelated)
    with pytest.raises(ancestry.AncestryError, match="revision mismatch"):
        ancestry.require_source_ancestor(archive, anchor, expected_head="f" * 40)


def test_archive_rejects_attestation_or_binding_tampering(tmp_path):
    repo, anchor, head = _repository(tmp_path / "repo")

    direct = _archive(repo, tmp_path / "direct", head)
    value = json.loads((direct / ancestry.ATTESTATION_PATH).read_text())
    value["ancestors"].remove(anchor)
    (direct / ancestry.ATTESTATION_PATH).write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(ancestry.AncestryError, match="does not match source manifest"):
        ancestry.require_source_ancestor(direct, anchor)

    rebound = _archive(repo, tmp_path / "rebound", head)
    value = json.loads((rebound / ancestry.ATTESTATION_PATH).read_text())
    value["ancestors"].remove(anchor)
    value["ancestor_count"] -= 1
    value["ancestors_sha256"] = ancestry._ancestry_digest(value["ancestors"])
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    (rebound / ancestry.ATTESTATION_PATH).write_bytes(payload)
    source_digest = hashlib.sha256((rebound / "source.py").read_bytes()).hexdigest()
    attestation_digest = hashlib.sha256(payload).hexdigest()
    manifest = (
        f"{attestation_digest}  {ancestry.ATTESTATION_PATH}\n"
        f"{source_digest}  source.py\n"
    ).encode("ascii")
    (rebound / ancestry.MANIFEST_PATH).write_bytes(manifest)
    with pytest.raises(ancestry.AncestryError, match="manifest digest does not match source revision"):
        ancestry.require_source_ancestor(rebound, anchor)


def test_archive_rejects_unbound_or_malformed_attestation(tmp_path):
    repo, anchor, head = _repository(tmp_path / "repo")
    archive = _archive(repo, tmp_path / "archive", head)
    manifest_path = archive / ancestry.MANIFEST_PATH
    manifest = "".join(
        line for line in manifest_path.read_text().splitlines(keepends=True)
        if ancestry.ATTESTATION_PATH not in line
    ).encode("ascii")
    manifest_path.write_bytes(manifest)
    revision_path = archive / ancestry.REVISION_PATH
    revision = revision_path.read_text().replace(
        next(line for line in revision_path.read_text().splitlines(keepends=True)
             if line.startswith("source_manifest_sha256=")),
        f"source_manifest_sha256={hashlib.sha256(manifest).hexdigest()}\n",
    )
    revision_path.write_text(revision, encoding="ascii")

    with pytest.raises(ancestry.AncestryError, match="does not bind"):
        ancestry.require_source_ancestor(archive, anchor)


def test_create_is_deterministic_and_refuses_overwrite(tmp_path):
    repo, _, head = _repository(tmp_path / "repo")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    value = ancestry.create_attestation(repo, first, head)
    ancestry.create_attestation(repo, second, head)

    assert first.read_bytes() == second.read_bytes()
    assert value["git_sha"] == head
    assert value["ancestors"] == sorted(value["ancestors"])
    with pytest.raises(ancestry.AncestryError, match="refusing existing"):
        ancestry.create_attestation(repo, first, head)

    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing-target")
    with pytest.raises(ancestry.AncestryError, match="refusing existing"):
        ancestry.create_attestation(repo, dangling, head)


def test_git_lookup_does_not_escape_archive_root(tmp_path):
    outer, anchor, head = _repository(tmp_path / "outer")
    archive = _archive(outer, outer / "nested-archive", head)

    result = ancestry.require_source_ancestor(archive, anchor)

    assert result["kind"] == "git_archive"


def test_provisioning_executes_the_generator_extracted_from_head():
    root = Path(__file__).resolve().parents[1]
    script = (root / "tools/pool_provision.sh").read_text(encoding="utf-8")
    archive = script.index('git archive "$SOURCE_SHA"')
    staged_generator = script.index(
        'python3 "$STAGE/tools/pool/provisioning/ancestry_attestation.py" create'
    )

    assert archive < staged_generator
    assert "python3 tools/pool/provisioning/ancestry_attestation.py create" not in script
    assert '--repo . --revision "$SOURCE_SHA"' in script
    assert (
        "docs CLAUDE.md GAP_CLOSURE_MISSION.md README.md ROADMAP.md "
        "requirements.txt requirements-dev.txt"
    ) in script
    assert "find docs -type f -name '*.md' -print0" in script
    assert (
        "CLAUDE.md\\0GAP_CLOSURE_MISSION.md\\0README.md\\0ROADMAP.md\\0"
        "requirements-dev.txt\\0.source_ancestry.json\\0"
    ) in script
    assert '"$STAGE/docs/" "$h:~/$REMOTE_ROOT/docs/"' in script
    assert '"$STAGE/CLAUDE.md" "$STAGE/GAP_CLOSURE_MISSION.md" "$STAGE/README.md"' in script
    assert "numpy scipy h5py pyyaml pytest" in script
    assert 'source_ancestry_sha256=%s' in script


def test_provisioning_can_archive_an_explicit_revision():
    root = Path(__file__).resolve().parents[1]
    script = (root / "tools/pool_provision.sh").read_text(encoding="utf-8")

    assert 'REVISION_REF=HEAD' in script
    assert 'REVISION_REF=$2' in script
    assert 'git rev-parse --verify "${REVISION_REF}^{commit}"' in script
    assert 'git archive "$SOURCE_SHA"' in script
