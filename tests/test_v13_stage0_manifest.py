"""Synthetic evidence tests for the V13 Stage-0 manifest adapter.

The tests create only inert JSON artifacts and hand-built success receipts. They do
not import or execute the scientific runner.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import execution_receipt
from tools import v13_stage0_controller as controller
from tools import v13_stage0_manifest as manifest_tool


CANDIDATE = "a" * 40
LEGACY = "b" * 40


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(value: dict) -> dict:
    result = dict(value)
    result["sha256"] = controller._canonical_digest(result)
    return result


def _write_replay_evidence(root: Path, revision: str) -> Path:
    evidence_root = root / Path(controller.STRICT_REPLAY_PATH).parent
    evidence_root.mkdir(parents=True, exist_ok=True)
    replay_sources = sorted(
        path for path in controller._required_candidate_source_paths(root)
        if path.startswith(controller.REPLAY_SENSITIVE_PREFIX) and path.endswith(".py")
    )
    source_manifest = evidence_root / "source.sha256"
    source_manifest.write_text("".join(
        f"{_digest(root / relative)}  {relative}\n" for relative in replay_sources
    ))
    source = {
        "file_count": len(replay_sources),
        "git_sha": revision,
        "kind": "git",
        "manifest": source_manifest.relative_to(root).as_posix(),
        "manifest_sha256": _digest(source_manifest),
        "tree_sha256": _digest(source_manifest),
    }
    cells = {}
    for backend in ("numpy", "cupy"):
        cell_path = _write_json(evidence_root / f"cell-{backend}.json", _seal({
            "schema": "v13-backend-neutral-izh-arithmetic-replay-cell-v2",
            "backend": backend,
            "source": source,
        }))
        receipt_path = evidence_root / f"cell-{backend}.receipt.json"
        _write_json(receipt_path, {
            "schema": execution_receipt.SCHEMA,
            "status": "success",
            "exit_code": 0,
            "source": source,
            "artifact": {
                "path": cell_path.relative_to(root).as_posix(),
                "sha256": _digest(cell_path),
                "size_bytes": cell_path.stat().st_size,
            },
            "argv": [
                "/usr/bin/python3", "-m", controller.REPLAY_RUNNER_MODULE,
                "--run", "--backend", backend, "--out", str(cell_path.resolve()),
            ],
            "env_allowlist": {"SIM_BACKEND": backend},
        })
        cell = json.loads(cell_path.read_text())
        cells[backend] = {
            "artifact_sha256": cell["sha256"],
            "file_sha256": _digest(cell_path),
            "path": cell_path.relative_to(root).as_posix(),
            "receipt_path": receipt_path.relative_to(root).as_posix(),
        }
    comparison_path = _write_json(evidence_root / "comparison.json", _seal({
        "outcome": "DIAGNOSTIC_PASS",
        "all_required_trajectories_exact": True,
        "trajectory_comparisons": {
            name: {"all_1200_rows_exact": True}
            for name in ("v", "u", "spikes")
        },
        "cell_artifacts": cells,
        "source": source,
    }))
    comparison_receipt = evidence_root / "comparison.receipt.json"
    _write_json(comparison_receipt, {
        "schema": execution_receipt.SCHEMA,
        "status": "success",
        "exit_code": 0,
        "source": source,
        "artifact": {
            "path": comparison_path.relative_to(root).as_posix(),
            "sha256": _digest(comparison_path),
            "size_bytes": comparison_path.stat().st_size,
        },
        "argv": [
            "/usr/bin/python3", "-m", controller.REPLAY_RUNNER_MODULE,
            "--compare", "--out", str(comparison_path.resolve()),
        ],
        "env_allowlist": {"SIM_BACKEND": "numpy"},
    })
    return _write_json(root / controller.STRICT_REPLAY_PATH, _seal({
        "schema": "v13-backend-neutral-izh-arithmetic-replay-evidence-manifest-v2",
        "outcome": "DIAGNOSTIC_PASS",
        "diagnostic_only": True,
        "scientific_verdict": None,
        "source": source,
        "cells": cells,
        "comparison": {
            "path": comparison_path.relative_to(root).as_posix(),
            "sha256": _digest(comparison_path),
            "artifact_sha256": json.loads(comparison_path.read_text())["sha256"],
            "receipt_path": comparison_receipt.relative_to(root).as_posix(),
            "receipt_sha256": _digest(comparison_receipt),
        },
    }))


class Fixture:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(controller, "SEED_DERIVATION_NAMESPACE", "TEST_STAGE0")
        monkeypatch.setattr(controller, "SEED_DERIVATION_SOURCE_REVISION", CANDIDATE)
        self.root = (tmp_path / "candidate").resolve()
        self.root.mkdir()
        compatibility = _write_json(
            self.root / controller.COMPATIBILITY_PATH,
            {"outcome": "DETERMINISTIC_COMPATIBILITY_GO", "go": True},
        )
        self.seeds = {
            "calibration": controller._derive_replacement_seed(
                role="calibration", original_seed=controller.OLD_CALIBRATION_SEED
            ),
            "replication": controller._derive_replacement_seed(
                role="replication", original_seed=controller.OLD_REPLICATION_SEED
            ),
            "held_out": controller.LOCKED_HELD_OUT_SEED,
        }
        self.seed_derivation = {
            "algorithm": controller.SEED_DERIVATION_ALGORITHM,
            "namespace": controller.SEED_DERIVATION_NAMESPACE,
            "source_revision": controller.SEED_DERIVATION_SOURCE_REVISION,
            "original_seeds": {
                "calibration": controller.OLD_CALIBRATION_SEED,
                "replication": controller.OLD_REPLICATION_SEED,
            },
        }
        spec = _write_json(
            self.root / controller.SEED_SPEC_PATH,
            {
                "partitions": {name: [seed] for name, seed in self.seeds.items()},
                "seed_derivation": self.seed_derivation,
            },
        )
        for relative in controller.REQUIRED_SOURCE_MANIFEST_PATHS:
            path = self.root / relative
            if path == spec:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# frozen source: {relative}\n")
        source_identity = {
            relative: _digest(self.root / relative)
            for relative in controller.CRITICAL_SOURCE_PATHS
        }
        legacy_runner_relative = (
            "research/runners/_vocal_action_credit_gate_v13_tonic_output.py"
        )
        legacy_runner = self.root / legacy_runner_relative
        source_manifest = self.root / "source.sha256"
        source_manifest.write_text("".join(
            f"{_digest(self.root / relative)}  {relative}\n"
            for relative in sorted(controller.REQUIRED_SOURCE_MANIFEST_PATHS)
        ))
        source_snapshot = execution_receipt.verify_source_manifest(
            self.root, "source.sha256"
        )

        replay_path = _write_replay_evidence(self.root, CANDIDATE)
        self.artifacts = {
            "calibration_numpy": "evidence/calibration-numpy.json",
            "calibration_cupy": "evidence/calibration-cupy.json",
            "calibration_selection": "evidence/calibration-selection.json",
            "replication_numpy": "evidence/replication-numpy.json",
            "replication_cupy": "evidence/replication-cupy.json",
            "held_out_cupy": "evidence/held-out-cupy.json",
            "held_out_numpy": "evidence/held-out-numpy.json",
            "performance_baseline": "evidence/performance-baseline.json",
            "performance_candidate": "evidence/performance-candidate.json",
            "final_stage0": "evidence/final-stage0.json",
        }
        body = {
            "schema": controller.CONFIG_SCHEMA,
            "status": "frozen",
            "correction_id": "stage0-process-correction-test",
            "candidate_source_revision": CANDIDATE,
            "candidate_source_identity": source_identity,
            "candidate_source_manifest": {
                "path": "source.sha256",
                "sha256": source_snapshot["manifest_sha256"],
                "tree_sha256": source_snapshot["tree_sha256"],
                "file_count": source_snapshot["file_count"],
            },
            "python": "/usr/bin/python3",
            "runner_module": controller.RUNNER_MODULE,
            "seeds": self.seeds,
            "seed_derivation": self.seed_derivation,
            "seed_binding": {"path": controller.SEED_SPEC_PATH, "sha256": _digest(spec)},
            "strict_arithmetic_replay": {
                "path": controller.STRICT_REPLAY_PATH,
                "sha256": _digest(replay_path),
                "source_revision": CANDIDATE,
            },
            "compatibility": {
                "path": controller.COMPATIBILITY_PATH,
                "sha256": _digest(compatibility),
            },
            "legacy_performance": {
                "source_revision": LEGACY,
                "runner_path": legacy_runner_relative,
                "runner_sha256": _digest(legacy_runner),
            },
            "artifacts": self.artifacts,
        }
        self.config = _seal(body)
        self.config_path = _write_json(self.root / "correction.json", self.config)
        monkeypatch.setattr(controller, "_git_head", lambda root: CANDIDATE)
        monkeypatch.setattr(
            controller,
            "_revision_file_digest",
            lambda root, revision, relative: _digest(root / relative),
        )
        monkeypatch.setattr(
            execution_receipt,
            "_source_revision",
            lambda root, expected_git_sha, manifest_sha256: "git",
        )
        self.envelope_path = self.root / "commands/calibration-numpy.json"
        controller.emit_calibration(
            config_path=self.config_path,
            backend="numpy",
            emit=self.envelope_path,
            root=self.root,
        )
        self.artifact_path = _write_json(
            self.root / self.artifacts["calibration_numpy"], {"dummy": "result"}
        )
        self.source_manifest = source_manifest
        self.receipt_path = self.root / "receipts/calibration-numpy.json"
        self.receipt = self._receipt()
        _write_json(self.receipt_path, self.receipt)

    def _receipt(self) -> dict:
        envelope = json.loads(self.envelope_path.read_text())
        source = execution_receipt.verify_source_manifest(self.root, "source.sha256")
        artifact_relative = self.artifact_path.relative_to(self.root).as_posix()
        return {
            "argv": envelope["argv"],
            "artifact": {
                "path": artifact_relative,
                "sha256": _digest(self.artifact_path),
                "size_bytes": self.artifact_path.stat().st_size,
            },
            "device": "AMD EPYC fixture CPU",
            "duration_monotonic_ns": 10,
            "ended_utc_ns": 110,
            "env_allowlist": envelope["env"],
            "execution_root": ".",
            "exit_code": 0,
            "host": "fixture-host",
            "schema": execution_receipt.SCHEMA,
            "source": {
                "file_count": source["file_count"],
                "git_sha": CANDIDATE,
                "kind": "git",
                "manifest": source["manifest"],
                "manifest_sha256": source["manifest_sha256"],
                "tree_sha256": source["tree_sha256"],
            },
            "started_utc_ns": 100,
            "status": "success",
        }

    def write_receipt(self, mutate) -> None:
        value = json.loads(json.dumps(self.receipt))
        mutate(value)
        _write_json(self.receipt_path, value)

    def write_envelope(self, mutate) -> None:
        value = json.loads(self.envelope_path.read_text())
        mutate(value)
        _write_json(self.envelope_path, value)

    def create(self, emit: str = "manifests/calibration-numpy.json") -> dict:
        return manifest_tool.create_manifest(
            root=self.root,
            config_path="correction.json",
            envelope_path="commands/calibration-numpy.json",
            receipt_path="receipts/calibration-numpy.json",
            kind="calibration_numpy",
            emit=emit,
        )


@pytest.fixture
def fx(tmp_path, monkeypatch):
    return Fixture(tmp_path, monkeypatch)


def test_emits_controller_consumable_self_digested_manifest(fx: Fixture):
    manifest = fx.create()
    stored = json.loads((fx.root / "manifests/calibration-numpy.json").read_text())

    assert stored == manifest
    assert set(manifest) == controller.MANIFEST_FIELDS
    assert manifest["schema"] == controller.MANIFEST_SCHEMA
    assert manifest["kind"] == "calibration_numpy"
    assert manifest["config_sha256"] == fx.config["sha256"]
    assert manifest["source_revision"] == CANDIDATE
    assert manifest["artifact"] == {
        "path": fx.artifacts["calibration_numpy"],
        "sha256": _digest(fx.artifact_path),
    }
    assert manifest["command_envelope"] == {
        "path": "commands/calibration-numpy.json",
        "sha256": _digest(fx.envelope_path),
    }
    assert manifest["execution_receipt"]["host"] == "fixture-host"
    assert manifest["sha256"] == controller._canonical_digest(manifest)

    artifact, loaded, _ = controller.load_manifest(
        fx.root / "manifests/calibration-numpy.json",
        config=fx.config,
        kind="calibration_numpy",
        root=fx.root,
    )
    assert artifact == {"dummy": "result"}
    assert loaded == manifest


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "wrong", "schema"),
        ("action", "calibration_cupy", "action"),
        ("source_revision", LEGACY, "source revision"),
        ("cwd", "/tmp", "cwd"),
        ("output", "/tmp/result.json", "canonical artifact"),
        ("argv", ["false"], "argv"),
        ("env", {"SIM_BACKEND": "cupy"}, "environment"),
        ("execution", "executed", "execution marker"),
    ],
)
def test_rejects_envelope_mismatches(fx: Fixture, field: str, value, message: str):
    fx.write_envelope(lambda envelope: envelope.__setitem__(field, value))
    with pytest.raises(manifest_tool.ManifestError, match=message):
        fx.create()


def test_rejects_envelope_config_digest_and_extra_fields(fx: Fixture):
    fx.write_envelope(lambda envelope: envelope["config"].__setitem__("sha256", "f" * 64))
    with pytest.raises(manifest_tool.ManifestError, match="config digest"):
        fx.create()

    fx.write_envelope(lambda envelope: envelope.__setitem__("unexpected", True))
    with pytest.raises(manifest_tool.ManifestError, match="missing or extra"):
        fx.create()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt.__setitem__("argv", ["false"]), "receipt argv"),
        (
            lambda receipt: receipt.__setitem__("env_allowlist", {"SIM_BACKEND": "cupy"}),
            "receipt environment",
        ),
        (
            lambda receipt: receipt["source"].__setitem__("git_sha", LEGACY),
            "receipt source revision",
        ),
    ],
)
def test_rejects_receipt_mismatches(fx: Fixture, mutation, message: str):
    fx.write_receipt(mutation)
    with pytest.raises(manifest_tool.ManifestError, match=message):
        fx.create()


def test_rejects_receipt_from_different_valid_source_manifest(fx: Fixture):
    extra = fx.root / "extra-source.txt"
    extra.write_text("extra source\n")
    alternate = fx.root / "alternate-source.sha256"
    alternate.write_text(
        fx.source_manifest.read_text()
        + f"{_digest(extra)}  extra-source.txt\n"
    )
    source = execution_receipt.verify_source_manifest(
        fx.root, "alternate-source.sha256"
    )

    def mutate(receipt):
        receipt["source"].update({
            "file_count": source["file_count"],
            "manifest": source["manifest"],
            "manifest_sha256": source["manifest_sha256"],
            "tree_sha256": source["tree_sha256"],
        })

    fx.write_receipt(mutate)
    with pytest.raises(manifest_tool.ManifestError, match="frozen candidate source"):
        fx.create()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt: receipt.__setitem__("status", "failed"),
        lambda receipt: receipt.__setitem__("host", ""),
        lambda receipt: receipt.__setitem__("device", ""),
        lambda receipt: receipt.__setitem__("started_utc_ns", 111),
        lambda receipt: receipt.__setitem__("extra", True),
    ],
)
def test_rejects_invalid_or_non_success_receipt(fx: Fixture, mutation):
    fx.write_receipt(mutation)
    with pytest.raises(manifest_tool.ManifestError, match="receipt is invalid"):
        fx.create()


def test_rejects_receipt_for_noncanonical_artifact(fx: Fixture):
    other = _write_json(fx.root / "evidence/other.json", {"dummy": "other"})

    def mutate(receipt):
        receipt["artifact"] = {
            "path": "evidence/other.json",
            "sha256": _digest(other),
            "size_bytes": other.stat().st_size,
        }

    fx.write_receipt(mutate)
    with pytest.raises(manifest_tool.ManifestError, match="canonical destination"):
        fx.create()


def test_rejects_artifact_tamper_after_receipt(fx: Fixture):
    fx.artifact_path.write_text('{"tampered":true}\n')
    with pytest.raises(manifest_tool.ManifestError, match="receipt is invalid"):
        fx.create()


def test_manifest_is_exclusive_create_only(fx: Fixture):
    fx.create()
    with pytest.raises(manifest_tool.ManifestError, match="refusing to overwrite"):
        fx.create()


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("config_path", "/tmp/config.json"),
        ("envelope_path", "../command.json"),
        ("receipt_path", "../receipt.json"),
        ("emit", "../manifest.json"),
    ],
)
def test_rejects_unsafe_paths(fx: Fixture, argument: str, value: str):
    arguments = {
        "root": fx.root,
        "config_path": "correction.json",
        "envelope_path": "commands/calibration-numpy.json",
        "receipt_path": "receipts/calibration-numpy.json",
        "kind": "calibration_numpy",
        "emit": "manifests/calibration-numpy.json",
    }
    arguments[argument] = value
    with pytest.raises(manifest_tool.ManifestError, match="safe repository-relative"):
        manifest_tool.create_manifest(**arguments)


def test_baseline_uses_legacy_revision_while_other_kinds_use_candidate(fx: Fixture):
    assert manifest_tool._expected_source(fx.config, "performance_baseline") == LEGACY
    for kind in set(manifest_tool.KINDS) - {"performance_baseline"}:
        assert manifest_tool._expected_source(fx.config, kind) == CANDIDATE


@pytest.mark.parametrize(
    ("kind", "action", "backend", "mode"),
    [
        ("calibration_numpy", "calibration_numpy", "numpy", "--calibration"),
        ("calibration_cupy", "calibration_cupy", "cupy", "--calibration"),
        ("calibration_selection", "merge_calibration", None, "--merge-calibration"),
        ("replication_numpy", "replication_numpy", "numpy", "--replication"),
        ("replication_cupy", "replication_cupy", "cupy", "--replication"),
        ("held_out_cupy", "held_out_cupy", "cupy", "--held-out"),
        ("held_out_numpy", "held_out_numpy", "numpy", "--held-out"),
        (
            "performance_baseline",
            "performance_baseline",
            "cupy",
            "--legacy-performance-baseline",
        ),
        (
            "performance_candidate",
            "performance_candidate",
            "cupy",
            "--performance",
        ),
        ("final_stage0", "final_stage0_merge", None, "--merge-final"),
    ],
)
def test_all_kind_command_contracts_are_explicit(
    fx: Fixture, kind: str, action: str, backend: str | None, mode: str,
):
    output = (fx.root / fx.artifacts[kind]).resolve()
    argv = manifest_tool._expected_argv(
        config=fx.config, kind=kind, root=fx.root, output=output
    )

    assert manifest_tool._ACTION_BY_KIND[kind] == action
    assert manifest_tool._expected_env(kind) == (
        {} if backend is None else {"SIM_BACKEND": backend}
    )
    assert argv[:3] == ["/usr/bin/python3", "-m", controller.RUNNER_MODULE]
    assert mode in argv
    assert argv[-2:] == ["--out", str(output)]


def test_main_reports_success_without_executing_a_command(fx: Fixture, capsys):
    result = manifest_tool.main([
        "--root", str(fx.root),
        "--config", "correction.json",
        "--envelope", "commands/calibration-numpy.json",
        "--receipt", "receipts/calibration-numpy.json",
        "--kind", "calibration_numpy",
        "--emit", "manifests/calibration-numpy.json",
    ])
    assert result == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "calibration_numpy"
