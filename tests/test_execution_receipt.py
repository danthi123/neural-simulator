"""Tests for fail-closed command execution receipts."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
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


_V2_CHILD = """
import json
import os
import sys
import time
from pathlib import Path

artifact = Path("result.json")
started = time.time_ns()
artifact.write_text(json.dumps({
    "run_id": os.environ["SIM_PROVENANCE_RUN_ID"],
    "source_kind": os.environ["SIM_PROVENANCE_SOURCE_KIND"],
    "source_manifest_sha256": os.environ["SIM_PROVENANCE_SOURCE_MANIFEST_SHA256"],
    "v2": os.environ["SIM_PROVENANCE_V2"],
}))
ended = time.time_ns()
source_kind = os.environ["SIM_PROVENANCE_SOURCE_KIND"]
sidecar = {
    "schema": "sim-run-provenance-v2",
    "run_id": os.environ["SIM_PROVENANCE_RUN_ID"],
    "artifact": "result.json",
    "git_sha": sys.argv[1],
    "git_dirty": False,
    "source_kind": source_kind,
    "source_manifest_sha256": os.environ["SIM_PROVENANCE_SOURCE_MANIFEST_SHA256"],
    "started_utc_ns": started,
    "ended_utc_ns": ended,
    "env": {"SIM_BACKEND": os.environ["SIM_BACKEND"]},
    "sim_backend_requested": os.environ["SIM_BACKEND"],
    "sim_backend": os.environ["SIM_BACKEND"],
    "sim_backend_cupy_importable": os.environ["SIM_BACKEND"] == "cupy",
    "source_manifest_verified_at_start": True if source_kind == "git_archive" else None,
    "source_manifest_start_error": None,
    "source_manifest_verified_at_exit": True if source_kind == "git_archive" else None,
    "source_manifest_exit_error": None,
}
overrides = json.loads(sys.argv[2])
for field in overrides.pop("__drop__", []):
    sidecar.pop(field, None)
sidecar.update(overrides)
Path("result.json.prov.json").write_text(json.dumps(sidecar))
"""


def _run_v2(root: Path, *, git_sha=GIT_SHA, overrides=None):
    return execution_receipt.run_and_receipt(
        root=root,
        artifact_path="result.json",
        receipt_path="result.receipt.json",
        source_manifest=".source_manifest.sha256",
        git_sha=git_sha,
        host="pool40",
        device="cpu",
        argv=[sys.executable, "-c", _V2_CHILD, git_sha, json.dumps(overrides or {})],
        env_allowlist=["SIM_BACKEND"],
        environ={"SIM_BACKEND": "numpy", "SECRET": "not-forwarded"},
        provenance_v2=True,
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
    with pytest.raises(
        execution_receipt.ReceiptError,
        match="artifact path escapes|existing artifact|cannot contain a symlink",
    ):
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


def test_symlinked_parent_directory_is_refused_before_launch(tmp_path):
    root = _root(tmp_path)
    real = root / "real"
    real.mkdir()
    (root / "linked").symlink_to(real, target_is_directory=True)

    with pytest.raises(execution_receipt.ReceiptError, match="cannot contain a symlink"):
        execution_receipt.run_and_receipt(
            root=root,
            artifact_path="linked/result.json",
            receipt_path="result.receipt.json",
            source_manifest=".source_manifest.sha256",
            git_sha=GIT_SHA,
            host="host",
            device="device",
            argv=[sys.executable, "-c", "raise SystemExit('must not launch')"],
            environ={},
        )
    assert not (real / "result.json").exists()


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


def test_v2_success_binds_private_run_identity_and_sidecar(tmp_path):
    root = _root(tmp_path)
    receipt = _run_v2(root)

    assert receipt["schema"] == execution_receipt.SCHEMA_V2
    assert receipt["env_allowlist"] == {"SIM_BACKEND": "numpy"}
    assert not execution_receipt._PRIVATE_PROVENANCE_ENV.intersection(
        receipt["env_allowlist"]
    )
    provenance = receipt["provenance"]
    assert provenance["path"] == "result.json.prov.json"
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["run_id"])
    artifact = json.loads((root / "result.json").read_text())
    assert artifact == {
        "run_id": provenance["run_id"],
        "source_kind": "git_archive",
        "source_manifest_sha256": receipt["source"]["manifest_sha256"],
        "v2": "1",
    }
    sidecar_path = root / provenance["path"]
    assert provenance["sha256"] == _sha(sidecar_path.read_bytes())
    assert execution_receipt.verify_receipt(root, "result.receipt.json") == receipt


def test_v2_refuses_existing_sidecar_before_launch(tmp_path):
    root = _root(tmp_path)
    sidecar = root / "result.json.prov.json"
    sidecar.write_text("preserve")

    with pytest.raises(execution_receipt.ReceiptError, match="existing provenance sidecar"):
        _run_v2(root)
    assert sidecar.read_text() == "preserve"
    assert not (root / "result.json").exists()
    assert not (root / "result.receipt.json").exists()


def test_v2_requires_allowlisted_backend_and_full_git_sha(tmp_path):
    root = _root(tmp_path / "backend")
    with pytest.raises(execution_receipt.ReceiptError, match="requires SIM_BACKEND"):
        execution_receipt.run_and_receipt(
            root=root,
            artifact_path="result.json",
            receipt_path="result.receipt.json",
            source_manifest=".source_manifest.sha256",
            git_sha=GIT_SHA,
            host="host",
            device="cpu",
            argv=[sys.executable, "-c", "raise SystemExit('must not launch')"],
            environ={},
            provenance_v2=True,
        )

    private_root = _root(tmp_path / "private")
    with pytest.raises(execution_receipt.ReceiptError, match="cannot be allowlisted"):
        execution_receipt.run_and_receipt(
            root=private_root,
            artifact_path="result.json",
            receipt_path="result.receipt.json",
            source_manifest=".source_manifest.sha256",
            git_sha=GIT_SHA,
            host="host",
            device="cpu",
            argv=[sys.executable, "-c", "raise SystemExit('must not launch')"],
            env_allowlist=["SIM_BACKEND", "SIM_PROVENANCE_RUN_ID"],
            environ={"SIM_BACKEND": "numpy", "SIM_PROVENANCE_RUN_ID": "caller-value"},
            provenance_v2=True,
        )

    short_root = _root(tmp_path / "short")
    with pytest.raises(execution_receipt.ReceiptError, match="full Git SHA"):
        execution_receipt.run_and_receipt(
            root=short_root,
            artifact_path="result.json",
            receipt_path="result.receipt.json",
            source_manifest=".source_manifest.sha256",
            git_sha=GIT_SHA[:12],
            host="host",
            device="cpu",
            argv=[sys.executable, "-c", "raise SystemExit('must not launch')"],
            env_allowlist=["SIM_BACKEND"],
            environ={"SIM_BACKEND": "numpy"},
            provenance_v2=True,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"run_id": "wrong"}, "run ID"),
        ({"artifact": "other.json"}, "artifact path"),
        ({"git_sha": "f" * 40}, "Git SHA"),
        ({"source_kind": "git"}, "source kind"),
        ({"source_manifest_sha256": "f" * 64}, "source manifest"),
        ({"sim_backend": "cupy"}, "backend"),
        ({"started_utc_ns": 0}, "timestamps"),
        ({"source_manifest_verified_at_exit": False}, "archive provenance"),
        ({"__drop__": ["source_manifest_start_error"]}, "missing fields"),
    ],
)
def test_v2_rejects_invalid_child_sidecar(tmp_path, overrides, message):
    root = _root(tmp_path)
    with pytest.raises(execution_receipt.ReceiptError, match=message):
        _run_v2(root, overrides=overrides)
    assert not (root / "result.receipt.json").exists()


def test_v2_verify_rejects_sidecar_tampering_and_symlink(tmp_path):
    tamper_root = _root(tmp_path / "tamper")
    _run_v2(tamper_root)
    sidecar = tamper_root / "result.json.prov.json"
    value = json.loads(sidecar.read_text())
    value["run_id"] = "f" * 64
    sidecar.write_text(json.dumps(value))
    with pytest.raises(execution_receipt.ReceiptError, match="does not match receipt"):
        execution_receipt.verify_receipt(tamper_root, "result.receipt.json")

    symlink_root = _root(tmp_path / "symlink")
    _run_v2(symlink_root)
    sidecar = symlink_root / "result.json.prov.json"
    outside = tmp_path / "outside-sidecar.json"
    outside.write_bytes(sidecar.read_bytes())
    sidecar.unlink()
    sidecar.symlink_to(outside)
    with pytest.raises(
        execution_receipt.ReceiptError,
        match=(
            "provenance sidecar path escapes|invalid provenance sidecar|"
            "cannot contain a symlink"
        ),
    ):
        execution_receipt.verify_receipt(symlink_root, "result.receipt.json")


def test_v2_accepts_git_source_with_null_archive_verification(tmp_path):
    root = tmp_path / "git-run"
    root.mkdir()
    (root / "source.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Receipt Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "source.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=root, check=True)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = f"{_sha((root / 'source.py').read_bytes())}  source.py\n"
    (root / ".source_manifest.sha256").write_text(manifest)

    receipt = _run_v2(root, git_sha=git_sha)
    assert receipt["source"]["kind"] == "git"
    assert execution_receipt.verify_receipt(root, "result.receipt.json") == receipt


def test_cli_v2_run_and_verify(tmp_path, capsys, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    assert execution_receipt.main(
        [
            "run", "--root", str(root), "--artifact", "result.json", "--receipt",
            "result.receipt.json", "--source-manifest", ".source_manifest.sha256",
            "--git-sha", GIT_SHA, "--host", "host", "--device", "cpu",
            "--env", "SIM_BACKEND", "--provenance-v2", "--", sys.executable,
            "-c", _V2_CHILD, GIT_SHA, "{}",
        ]
    ) == 0
    assert "WROTE" in capsys.readouterr().out
    stored = json.loads((root / "result.receipt.json").read_text())
    assert stored["schema"] == execution_receipt.SCHEMA_V2
    assert execution_receipt.main(
        ["verify", "--root", str(root), "--receipt", "result.receipt.json"]
    ) == 0
    assert "VERIFIED" in capsys.readouterr().out
