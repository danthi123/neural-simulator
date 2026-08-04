"""Tests for isolated, seedless pool regression receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import pool_regression_receipt as receipt
from tools.pool.provisioning import ancestry_attestation as ancestry


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _deployed_root(tmp_path: Path, script: str) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
    (repository / "tracked.txt").write_text("immutable\n", encoding="ascii")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=repository, check=True)
    revision = _git(repository, "rev-parse", "HEAD")

    root = tmp_path / "deployed" / revision
    (root / "tools").mkdir(parents=True)
    (root / ".venv/bin").mkdir(parents=True)
    (root / "tracked.txt").write_text("immutable\n", encoding="ascii")
    bundle = root / receipt.BUNDLE_PATH
    bundle.write_text(script, encoding="ascii")
    bundle.chmod(0o755)
    (root / ".venv/bin/python").symlink_to(Path(sys.executable).resolve())
    ancestry.create_attestation(repository, root / ancestry.ATTESTATION_PATH, revision)

    entries = []
    for relative in (ancestry.ATTESTATION_PATH, "tracked.txt", receipt.BUNDLE_PATH):
        payload = (root / relative).read_bytes()
        entries.append(f"{hashlib.sha256(payload).hexdigest()}  {relative}\n")
    manifest = "".join(sorted(entries)).encode("ascii")
    (root / ancestry.MANIFEST_PATH).write_bytes(manifest)
    ancestry_digest = hashlib.sha256((root / ancestry.ATTESTATION_PATH).read_bytes()).hexdigest()
    (root / ancestry.REVISION_PATH).write_text(
        "source_kind=git_archive\n"
        f"git_sha={revision}\n"
        f"source_manifest_sha256={hashlib.sha256(manifest).hexdigest()}\n"
        f"source_ancestry_sha256={ancestry_digest}\n",
        encoding="ascii",
    )
    return root, revision


def _collect(tmp_path: Path, script: str) -> tuple[Path, str, Path, dict]:
    root, revision = _deployed_root(tmp_path, script)
    output = tmp_path / "receipts" / "pool.json"
    output.parent.mkdir()
    value = receipt.collect_receipt(
        root=root, expected_revision=revision, receipt_path=output,
    )
    return root, revision, output, value


def test_success_receipt_binds_source_command_output_and_environment(tmp_path: Path) -> None:
    script = "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'bundle ok\\n'\nprintf 'warning-free\\n' >&2\n"
    root, revision, output, value = _collect(tmp_path, script)

    assert value["schema"] == receipt.SCHEMA
    assert value["status"] == "passed"
    assert value["exit_code"] == 0
    assert value["expected_revision"] == revision
    assert value["command"] == {
        "argv": ["bash", receipt.BUNDLE_PATH],
        "cwd": ".",
        "scientific_arguments": [],
    }
    assert value["stdout"]["text"] == "bundle ok\n"
    assert value["stderr"]["text"] == "warning-free\n"
    assert value["source"]["git_sha"] == revision
    assert value["source"]["kind"] == "git_archive"
    assert value["source"]["manifest_sha256"] == hashlib.sha256(
        (root / ancestry.MANIFEST_PATH).read_bytes()
    ).hexdigest()
    assert value["environment"]["hostname"]
    assert value["environment"]["python"]["version"]
    assert value["sha256"] == receipt._self_digest(value)
    assert receipt.verify_receipt(output) == value
    assert not any(path.suffix == ".pyc" for path in root.rglob("*.pyc"))


def test_nonzero_regression_is_captured_in_create_only_failure_receipt(tmp_path: Path) -> None:
    script = "#!/usr/bin/env bash\nprintf 'failed safely\\n' >&2\nexit 17\n"
    root, _, output, value = _collect(tmp_path, script)

    assert value["status"] == "failed"
    assert value["exit_code"] == 17
    assert value["stderr"]["text"] == "failed safely\n"
    original = output.read_bytes()
    with pytest.raises(receipt.PoolReceiptError, match="refusing existing receipt"):
        receipt.collect_receipt(
            root=root,
            expected_revision=value["expected_revision"],
            receipt_path=output,
        )
    assert output.read_bytes() == original


def test_rejects_root_level_python_shadow_before_execution(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    root, revision = _deployed_root(
        tmp_path,
        f"#!/usr/bin/env bash\ntouch {marker}\n",
    )
    (root / "sitecustomize.py").write_text("raise RuntimeError('shadowed')\n", encoding="ascii")

    with pytest.raises(receipt.PoolReceiptError, match="import-shadowing"):
        receipt.collect_receipt(
            root=root, expected_revision=revision, receipt_path=tmp_path / "receipt.json",
        )
    assert not marker.exists()
    assert not (tmp_path / "receipt.json").exists()


def test_rejects_revision_or_ancestry_binding_mismatch(tmp_path: Path) -> None:
    root, revision = _deployed_root(tmp_path, "#!/usr/bin/env bash\nexit 0\n")

    with pytest.raises(receipt.PoolReceiptError, match="revision mismatch"):
        receipt.collect_receipt(
            root=root, expected_revision="f" * 40, receipt_path=tmp_path / "wrong.json",
        )

    attestation_path = root / ancestry.ATTESTATION_PATH
    value = json.loads(attestation_path.read_text(encoding="ascii"))
    value["ancestor_count"] = 0
    attestation_path.write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(receipt.PoolReceiptError, match="deployed source verification failed"):
        receipt.collect_receipt(
            root=root, expected_revision=revision, receipt_path=tmp_path / "tampered.json",
        )


def test_manifest_drift_during_bundle_refuses_receipt(tmp_path: Path) -> None:
    script = "#!/usr/bin/env bash\nprintf 'changed\\n' > tracked.txt\n"
    root, revision = _deployed_root(tmp_path, script)
    output = tmp_path / "receipt.json"

    with pytest.raises(receipt.PoolReceiptError, match="source verification failed"):
        receipt.collect_receipt(root=root, expected_revision=revision, receipt_path=output)
    assert not output.exists()


def test_scientific_seed_output_is_never_written_or_repeated(tmp_path: Path) -> None:
    root, revision = _deployed_root(
        tmp_path, "#!/usr/bin/env bash\nprintf 'scientific_seed=123456\\n'\n",
    )
    output = tmp_path / "receipt.json"

    with pytest.raises(receipt.PoolReceiptError, match="seedless receipt contract") as raised:
        receipt.collect_receipt(root=root, expected_revision=revision, receipt_path=output)
    assert "123456" not in str(raised.value)
    assert not output.exists()


def test_receipt_must_be_external_and_self_digest_detects_tampering(tmp_path: Path) -> None:
    script = "#!/usr/bin/env bash\nprintf 'ok\\n'\n"
    root, revision = _deployed_root(tmp_path / "external", script)
    with pytest.raises(receipt.PoolReceiptError, match="outside the deployed root"):
        receipt.collect_receipt(
            root=root, expected_revision=revision, receipt_path=root / "receipt.json",
        )

    output = tmp_path / "receipt.json"
    value = receipt.collect_receipt(root=root, expected_revision=revision, receipt_path=output)
    value["status"] = "failed"
    output.write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(receipt.PoolReceiptError, match="self-digest"):
        receipt.verify_receipt(output)

    value = receipt.collect_receipt(
        root=root,
        expected_revision=revision,
        receipt_path=tmp_path / "second-receipt.json",
    )
    value["unexpected"] = True
    value["sha256"] = receipt._self_digest(value)
    (tmp_path / "second-receipt.json").write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(receipt.PoolReceiptError, match="fields"):
        receipt.verify_receipt(tmp_path / "second-receipt.json")


def test_cli_surface_has_no_command_or_scientific_seed_options() -> None:
    option_strings = {
        option
        for action in receipt._parser()._actions
        for option in action.option_strings
    }
    assert option_strings == {"-h", "--help", "--root", "--expected-revision", "--receipt"}
