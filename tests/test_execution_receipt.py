"""Tests for fail-closed command execution receipts."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from tools import execution_receipt


GIT_SHA = "0123456789abcdef0123456789abcdef01234567"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source(root: Path) -> None:
    source = root / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    manifest = f"{_sha(source.read_bytes())}  source.py\n"
    (root / ".source_manifest.sha256").write_text(manifest, encoding="utf-8")
    (root / ".source_revision").write_text(
        "source_kind=git_archive\n"
        f"git_sha={GIT_SHA}\n"
        f"source_manifest_sha256={_sha(manifest.encode())}\n",
        encoding="utf-8",
    )


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True)
    _source(root)
    return root


def _run(root: Path, code: str, *, env=(), environ=None):
    return execution_receipt.run_and_receipt(
        root=root,
        artifact_path="result.json",
        receipt_path="result.receipt.json",
        source_manifest=".source_manifest.sha256",
        git_sha=GIT_SHA,
        host="pool40",
        device="cpu",
        argv=[sys.executable, "-c", code],
        env_allowlist=env,
        environ={} if environ is None else environ,
    )


def test_success_receipt_binds_execution_source_and_artifact(tmp_path):
    root = _root(tmp_path)
    receipt = _run(root, "from pathlib import Path; Path('result.json').write_text('ok')")

    stored = json.loads((root / "result.receipt.json").read_text(encoding="ascii"))
    assert stored == receipt
    assert stored["schema"] == "sim-execution-receipt-v1"
    assert stored["status"] == "success"
    assert stored["exit_code"] == 0
    assert stored["host"] == "pool40"
    assert stored["device"] == "cpu"
    assert stored["started_utc_ns"] <= stored["ended_utc_ns"]
    assert stored["duration_monotonic_ns"] >= 0
    assert stored["artifact"] == {
        "path": "result.json",
        "sha256": _sha(b"ok"),
        "size_bytes": 2,
    }
    assert stored["source"]["git_sha"] == GIT_SHA
    assert stored["source"]["kind"] == "git_archive"
    assert stored["source"]["file_count"] == 1
    assert execution_receipt.verify_receipt(root, "result.receipt.json") == stored


def test_environment_is_explicitly_allowlisted_and_command_is_not_a_shell(tmp_path):
    root = _root(tmp_path)
    code = (
        "import json,os,sys; from pathlib import Path; "
        "Path('result.json').write_text(json.dumps({'env':dict(os.environ),'arg':sys.argv[1]}))"
    )
    marker = root / "must-not-exist"
    receipt = execution_receipt.run_and_receipt(
        root=root,
        artifact_path="result.json",
        receipt_path="result.receipt.json",
        source_manifest=".source_manifest.sha256",
        git_sha=GIT_SHA,
        host="local",
        device="test",
        argv=[sys.executable, "-c", code, f"; touch {marker}"],
        env_allowlist=["VISIBLE"],
        environ={"VISIBLE": "yes", "SECRET": "no"},
    )

    output = json.loads((root / "result.json").read_text())
    assert output["arg"] == f"; touch {marker}"
    assert output["env"] == {"LC_CTYPE": "C.UTF-8", "VISIBLE": "yes"}
    assert receipt["env_allowlist"] == {"VISIBLE": "yes"}
    assert not marker.exists()


@pytest.mark.parametrize("existing", ["result.json", "result.receipt.json"])
def test_existing_outputs_are_refused_without_modification(tmp_path, existing):
    root = _root(tmp_path)
    target = root / existing
    target.write_bytes(b"preserve")

    with pytest.raises(execution_receipt.ReceiptError, match="refusing existing"):
        _run(root, "raise SystemExit('must not launch')")
    assert target.read_bytes() == b"preserve"


def test_missing_artifact_and_nonzero_exit_never_write_receipt(tmp_path):
    missing_root = _root(tmp_path / "missing")
    with pytest.raises(execution_receipt.ReceiptError, match="cannot open artifact"):
        _run(missing_root, "pass")
    assert not (missing_root / "result.receipt.json").exists()

    nonzero_root = _root(tmp_path / "nonzero")
    with pytest.raises(execution_receipt.ReceiptError, match="exited nonzero: 7"):
        _run(
            nonzero_root,
            "from pathlib import Path; Path('result.json').write_text('partial'); raise SystemExit(7)",
        )
    assert (nonzero_root / "result.json").read_text() == "partial"
    assert not (nonzero_root / "result.receipt.json").exists()


@pytest.mark.parametrize(
    "code",
    [
        "from pathlib import Path; Path('source.py').write_text('changed'); Path('result.json').write_text('ok')",
        "from pathlib import Path; Path('.source_manifest.sha256').write_text('changed'); Path('result.json').write_text('ok')",
    ],
)
def test_source_drift_never_writes_receipt(tmp_path, code):
    root = _root(tmp_path)
    with pytest.raises(execution_receipt.ReceiptError, match="source"):
        _run(root, code)
    assert not (root / "result.receipt.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_path", "../escape.json"),
        ("artifact_path", "/tmp/escape.json"),
        ("receipt_path", "../escape.json"),
        ("source_manifest", "../manifest"),
    ],
)
def test_unsafe_paths_are_rejected_before_launch(tmp_path, field, value):
    root = _root(tmp_path)
    arguments = {
        "root": root,
        "artifact_path": "result.json",
        "receipt_path": "result.receipt.json",
        "source_manifest": ".source_manifest.sha256",
        "git_sha": GIT_SHA,
        "host": "host",
        "device": "device",
        "argv": [sys.executable, "-c", "raise SystemExit('must not launch')"],
    }
    arguments[field] = value
    with pytest.raises(execution_receipt.ReceiptError, match="safe repository-relative path"):
        execution_receipt.run_and_receipt(**arguments)
    assert not (root / "result.receipt.json").exists()


def test_symlink_artifact_and_git_identity_mismatch_are_refused(tmp_path):
    root = _root(tmp_path / "symlink")
    outside = tmp_path / "outside"
    outside.write_text("outside")
    (root / "result.json").symlink_to(outside)
    with pytest.raises(execution_receipt.ReceiptError, match="artifact path escapes|existing artifact"):
        _run(root, "pass")

    mismatch_root = _root(tmp_path / "mismatch")
    with pytest.raises(execution_receipt.ReceiptError, match="Git identity mismatch"):
        execution_receipt.run_and_receipt(
            root=mismatch_root,
            artifact_path="result.json",
            receipt_path="result.receipt.json",
            source_manifest=".source_manifest.sha256",
            git_sha="f" * 40,
            host="host",
            device="device",
            argv=[sys.executable, "-c", "raise SystemExit('must not launch')"],
            environ={},
        )


def test_verify_rejects_artifact_tampering_and_invalid_schema(tmp_path):
    root = _root(tmp_path / "artifact")
    _run(root, "from pathlib import Path; Path('result.json').write_text('ok')")
    (root / "result.json").write_text("tampered")
    with pytest.raises(execution_receipt.ReceiptError, match="artifact does not match"):
        execution_receipt.verify_receipt(root, "result.receipt.json")

    schema_root = _root(tmp_path / "schema")
    _run(schema_root, "from pathlib import Path; Path('result.json').write_text('ok')")
    receipt_path = schema_root / "result.receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["unexpected"] = True
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(execution_receipt.ReceiptError, match="invalid receipt fields"):
        execution_receipt.verify_receipt(schema_root, "result.receipt.json")


def test_verify_rejects_later_source_drift(tmp_path):
    root = _root(tmp_path)
    _run(root, "from pathlib import Path; Path('result.json').write_text('ok')")
    (root / "source.py").write_text("VALUE = 2\n")

    with pytest.raises(execution_receipt.ReceiptError, match="source digest mismatch"):
        execution_receipt.verify_receipt(root, "result.receipt.json")


def test_cli_run_and_verify(tmp_path, capsys):
    root = _root(tmp_path)
    code = "from pathlib import Path; Path('result.json').write_text('cli')"
    assert execution_receipt.main(
        [
            "run", "--root", str(root), "--artifact", "result.json", "--receipt",
            "result.receipt.json", "--source-manifest", ".source_manifest.sha256",
            "--git-sha", GIT_SHA, "--host", "host", "--device", "cpu", "--",
            sys.executable, "-c", code,
        ]
    ) == 0
    assert "WROTE" in capsys.readouterr().out
    assert execution_receipt.main(
        ["verify", "--root", str(root), "--receipt", "result.receipt.json"]
    ) == 0
    assert "VERIFIED" in capsys.readouterr().out
