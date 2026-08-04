"""Synthetic state-machine tests for the external V13 Stage-0 controller.

No scientific runner is imported or executed by this test module.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import execution_receipt
from tools import v13_stage0_controller as controller


CANDIDATE = "a" * 40
LEGACY = "b" * 40


@pytest.mark.parametrize(
    "script",
    [
        "tools/execution_receipt.py",
        "tools/v13_stage0_controller.py",
        "tools/v13_stage0_freeze.py",
        "tools/v13_stage0_manifest.py",
    ],
)
def test_control_tools_support_direct_cli_invocation(script: str):
    root = Path(controller.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(value: dict) -> dict:
    result = dict(value)
    result["sha256"] = hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return result


def _rewrite_sealed(path: Path, mutate) -> dict:
    value = json.loads(path.read_text())
    value.pop("sha256", None)
    mutate(value)
    sealed = _seal(value)
    _write_json(path, sealed)
    return sealed


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

    cells: dict[str, dict] = {}
    for backend in ("numpy", "cupy"):
        cell_path = _write_json(evidence_root / f"cell-{backend}.json", _seal({
            "schema": "v13-backend-neutral-izh-arithmetic-replay-cell-v2",
            "backend": backend,
            "source": source,
        }))
        receipt_path = evidence_root / f"cell-{backend}.receipt.json"
        receipt = {
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
        }
        _write_json(receipt_path, receipt)
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
    comparison_receipt_path = evidence_root / "comparison.receipt.json"
    _write_json(comparison_receipt_path, {
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
            "receipt_path": comparison_receipt_path.relative_to(root).as_posix(),
            "receipt_sha256": _digest(comparison_receipt_path),
        },
    }))


def _load_fixture_manifest(fx: "Fixture", path: Path, kind: str) -> dict:
    config = controller.load_config(fx.config_path, root=fx.root)
    _, manifest, _ = controller.load_manifest(
        path, config=config, kind=kind, root=fx.root
    )
    return manifest


class Fixture:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(controller, "SEED_DERIVATION_NAMESPACE", "TEST_STAGE0")
        monkeypatch.setattr(controller, "SEED_DERIVATION_SOURCE_REVISION", CANDIDATE)
        self.root = tmp_path / "candidate"
        self.root.mkdir()
        self.legacy_root = tmp_path / "legacy"
        self.legacy_root.mkdir()
        self.manifest_dir = self.root / "manifests"
        self.manifest_dir.mkdir()
        process_spec = json.loads(
            (controller.ROOT / controller.SEED_SPEC_PATH).read_text()
        )

        compatibility = _write_json(
            self.root / controller.COMPATIBILITY_PATH,
            {
                "stage": "cross_twin_compare",
                "outcome": "DETERMINISTIC_COMPATIBILITY_GO",
                "verdict_status": "GO",
                "preconditions": [{"name": "sealed twins", "ok": True}],
                "undefined_reasons": [],
                "go": True,
            },
        )
        self.seeds = {
            "calibration": controller._derive_replacement_seed(
                role="calibration", prior_seed=controller.PRIOR_PARTITION_SEEDS["calibration"]
            ),
            "replication": controller._derive_replacement_seed(
                role="replication", prior_seed=controller.PRIOR_PARTITION_SEEDS["replication"]
            ),
            "held_out": controller.LOCKED_HELD_OUT_SEED,
        }
        self.seed_derivation = controller._expected_seed_derivation()
        spec_path = self.root / controller.SEED_SPEC_PATH
        for relative in controller.REQUIRED_SOURCE_MANIFEST_PATHS:
            path = self.root / relative
            if path == spec_path:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# frozen candidate source: {relative}\n")
        _write_json(spec_path, {})
        replay_path = _write_replay_evidence(self.root, CANDIDATE)
        compatibility_binding = {
            "path": controller.COMPATIBILITY_PATH,
            "file_sha256": _digest(compatibility),
            "canonical_json_sha256": hashlib.sha256(
                controller._canonical_bytes(json.loads(compatibility.read_text()))
            ).hexdigest(),
            "canonicalization": controller.COMPATIBILITY_CANONICALIZATION,
        }
        process_spec.update({
            "base_spec": {
                "path": controller.BASE_SPEC_PATH,
                "sha256": _digest(self.root / controller.BASE_SPEC_PATH),
            },
            "strict_arithmetic_replay": {
                "path": controller.STRICT_REPLAY_PATH,
                "sha256": _digest(replay_path),
                "outcome": "DIAGNOSTIC_PASS",
            },
            "seed_derivation": self.seed_derivation,
            "partitions": {name: [seed] for name, seed in self.seeds.items()},
            "compatibility": {
                **compatibility_binding,
                "verification": process_spec["compatibility"]["verification"],
            },
        })
        spec = _write_json(spec_path, process_spec)
        self.source_identity = {
            relative: _digest(self.root / relative)
            for relative in controller.CRITICAL_SOURCE_PATHS
        }
        legacy_runner = self.legacy_root / "research/runners/_vocal_action_credit_gate_v13_tonic_output.py"
        legacy_runner.parent.mkdir(parents=True)
        candidate_runner = self.root / controller.CRITICAL_SOURCE_PATHS[0]
        legacy_runner.write_bytes(candidate_runner.read_bytes())

        source_manifest = self.root / "source.sha256"
        source_manifest.write_text("".join(
            f"{_digest(self.root / relative)}  {relative}\n"
            for relative in sorted(controller.REQUIRED_SOURCE_MANIFEST_PATHS)
        ))
        source_snapshot = execution_receipt.verify_source_manifest(
            self.root, "source.sha256"
        )

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
        config_body = {
            "schema": controller.CONFIG_SCHEMA,
            "status": "frozen",
            "correction_id": "stage0-process-correction-1",
            "candidate_source_revision": CANDIDATE,
            "candidate_source_identity": self.source_identity,
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
            "seed_binding": {
                "path": controller.SEED_SPEC_PATH,
                "sha256": _digest(spec),
            },
            "strict_arithmetic_replay": {
                "path": controller.STRICT_REPLAY_PATH,
                "sha256": _digest(replay_path),
                "source_revision": CANDIDATE,
            },
            "compatibility": compatibility_binding,
            "legacy_performance": {
                "source_revision": LEGACY,
                "runner_path": "research/runners/_vocal_action_credit_gate_v13_tonic_output.py",
                "runner_sha256": _digest(legacy_runner),
            },
            "artifacts": self.artifacts,
        }
        self.config = _seal(config_body)
        self.config_path = _write_json(self.root / "correction.json", self.config)
        monkeypatch.setattr(
            controller, "_git_head",
            lambda path: LEGACY if path.resolve() == self.legacy_root.resolve() else CANDIDATE,
        )
        monkeypatch.setattr(
            controller, "_revision_file_digest",
            lambda root, revision, relative: _digest(root / relative),
        )
        monkeypatch.setattr(
            execution_receipt,
            "_source_revision",
            lambda root, expected_git_sha, manifest_sha256: "git",
        )
        self.source_manifest = source_manifest

    def artifact_path(self, kind: str) -> Path:
        return self.root / self.artifacts[kind]

    def manifest(
        self, kind: str, artifact: dict, *,
        prerequisites: list[dict] | None = None,
    ) -> Path:
        artifact_path = _write_json(self.artifact_path(kind), artifact)
        source_revision = LEGACY if kind == "performance_baseline" else CANDIDATE
        command = {
            "schema": controller.COMMAND_SCHEMA,
            "action": controller.MANIFEST_ACTIONS[kind],
            "correction_id": self.config["correction_id"],
            "config": {
                "path": str(self.config_path.resolve()),
                "sha256": self.config["sha256"],
            },
            "source_revision": source_revision,
            "cwd": str(self.root.resolve()),
            "env": controller._expected_manifest_env(kind),
            "argv": controller._expected_manifest_argv(
                config=self.config, kind=kind, root=self.root,
                output=artifact_path.resolve(),
            ),
            "output": str(artifact_path.resolve()),
            "prerequisites": prerequisites or [],
            "execution": "not_executed",
        }
        if kind == "final_stage0":
            command["expected_result"] = {
                "stage": "final_cross_backend",
                "outcome": "TONIC_OUTPUT_GO",
                "go": True,
            }
        command_path = _write_json(
            self.root / f"evidence-commands/{kind}.json", command
        )
        source = execution_receipt.verify_source_manifest(self.root, "source.sha256")
        run_id = "a" * 64
        sidecar_path = _write_json(
            Path(f"{artifact_path}.prov.json"),
            {
                "schema": execution_receipt.PROVENANCE_SCHEMA_V2,
                "run_id": run_id,
                "runner": controller.RUNNER_MODULE.replace(".", "/") + ".py",
                "argv": [
                    str((self.root / controller.RUNNER_MODULE.replace(".", "/")).with_suffix(".py")),
                    *command["argv"][3:],
                ],
                "git_sha": source_revision,
                "git_dirty": False,
                "source_kind": "git",
                "source_manifest_sha256": source["manifest_sha256"],
                "source_manifest_verified_at_start": None,
                "source_manifest_start_error": None,
                "source_manifest_verified_at_exit": None,
                "source_manifest_exit_error": None,
                "started": "fixture",
                "started_utc_ns": 101,
                "ended_utc_ns": 109,
                "env": command["env"],
                "sim_backend": command["env"]["SIM_BACKEND"],
                "sim_backend_requested": command["env"]["SIM_BACKEND"],
                "sim_backend_cupy_importable": True,
                "artifact": self.artifacts[kind],
            },
        )
        receipt = {
            "argv": command["argv"],
            "artifact": {
                "path": self.artifacts[kind],
                "sha256": _digest(artifact_path),
                "size_bytes": artifact_path.stat().st_size,
            },
            "device": "synthetic fixture device",
            "duration_monotonic_ns": 10,
            "ended_utc_ns": 110,
            "env_allowlist": command["env"],
            "execution_root": ".",
            "exit_code": 0,
            "host": "fixture-host",
            "schema": execution_receipt.SCHEMA_V2,
            "source": {
                "file_count": source["file_count"],
                "git_sha": source_revision,
                "kind": "git",
                "manifest": source["manifest"],
                "manifest_sha256": source["manifest_sha256"],
                "tree_sha256": source["tree_sha256"],
            },
            "started_utc_ns": 100,
            "status": "success",
            "provenance": {
                "path": sidecar_path.relative_to(self.root).as_posix(),
                "sha256": _digest(sidecar_path),
                "run_id": run_id,
                "started_utc_ns": 101,
                "ended_utc_ns": 109,
            },
        }
        receipt_path = _write_json(
            self.root / f"evidence-receipts/{kind}.json", receipt
        )
        body = {
            "schema": controller.MANIFEST_SCHEMA,
            "kind": kind,
            "config_sha256": self.config["sha256"],
            "source_revision": source_revision,
            "controller_config": {
                "path": self.config_path.relative_to(self.root).as_posix(),
                "file_sha256": _digest(self.config_path),
                "canonical_sha256": self.config["sha256"],
            },
            "process_correction_spec": dict(self.config["seed_binding"]),
            "candidate_source_manifest": dict(
                self.config["candidate_source_manifest"]
            ),
            "compatibility": dict(self.config["compatibility"]),
            "artifact": {"path": self.artifacts[kind], "sha256": _digest(artifact_path)},
            "provenance_sidecar": {
                "path": sidecar_path.relative_to(self.root).as_posix(),
                "sha256": _digest(sidecar_path),
            },
            "command_envelope": {
                "path": command_path.relative_to(self.root).as_posix(),
                "sha256": _digest(command_path),
            },
            "execution_receipt": {
                "path": receipt_path.relative_to(self.root).as_posix(),
                "sha256": _digest(receipt_path),
                "host": receipt["host"],
                "device": receipt["device"],
                "started_utc_ns": receipt["started_utc_ns"],
                "ended_utc_ns": receipt["ended_utc_ns"],
            },
        }
        return _write_json(self.manifest_dir / f"{kind}.manifest.json", _seal(body))

    def manifest_reference(self, kind: str, path: Path) -> dict:
        manifest = json.loads(path.read_text())
        return {
            "kind": kind,
            "manifest_path": str(path.resolve()),
            "manifest_sha256": manifest["sha256"],
            "artifact_path": str(self.artifact_path(kind).resolve()),
            "artifact_sha256": manifest["artifact"]["sha256"],
            "command_envelope_path": manifest["command_envelope"]["path"],
            "command_envelope_sha256": manifest["command_envelope"]["sha256"],
            "execution_receipt_path": manifest["execution_receipt"]["path"],
            "execution_receipt_sha256": manifest["execution_receipt"]["sha256"],
            "provenance_sidecar_path": manifest["provenance_sidecar"]["path"],
            "provenance_sidecar_sha256": manifest["provenance_sidecar"]["sha256"],
        }

    def calibration(self, backend: str) -> dict:
        rows = []
        for value in (75, 100, 125, 150, 175):
            passed = value == 100
            rows.append({
                "current_pA": value,
                "audit": {"pass": True},
                "physiology": {"pass": passed},
                "pass": passed,
            })
        return {
            "stage": "calibration_backend",
            "backend": backend,
            "seed": self.seeds["calibration"],
            "source_sha": CANDIDATE,
            "source_identity": self.source_identity,
            "spec_sha256": "6" * 64,
            "compatibility_correction": {
                **self.config["compatibility"],
            },
            "rows": rows,
            "passing_currents_pA": [100],
        }

    def selection(self, *, go: bool = True) -> dict:
        return {
            "stage": "calibration_cross_backend",
            "backend": "cross_backend",
            "seed": self.seeds["calibration"],
            "source_shas": {"numpy": CANDIDATE, "cupy": CANDIDATE},
            "source_identity": self.source_identity,
            "compatibility_correction": {
                **self.config["compatibility"],
            },
            "selected_current_pA": 100,
            "calibration_go": go,
            "go": go,
            "verdict_status": "GO" if go else "NO-GO",
            "preconditions": [{"name": "matched calibration", "ok": True}],
            "undefined_reasons": [],
            "outcome": "CALIBRATION_GO" if go else "CALIBRATION_NO_GO",
        }

    def stage(self, stage: str, backend: str, selection: dict) -> dict:
        seed = self.seeds[stage if stage == "replication" else "held_out"]
        return {
            "stage": stage,
            "backend": backend,
            "seed": seed,
            "source_sha": CANDIDATE,
            "source_identity": self.source_identity,
            "selected_current_pA": 100,
            "selection": selection,
            "go": True,
            "verdict_status": "GO",
            "preconditions": [{"name": f"complete {stage}", "ok": True}],
            "undefined_reasons": [],
            "outcome": f"{stage.upper()}_GO",
        }

    def performance(self) -> dict:
        return {
            "stage": "performance",
            "seed": 314159,
            "backend": "cupy",
            "device": "NVIDIA GeForce RTX 3090",
            "source_sha": CANDIDATE,
            "source_identity": self.source_identity,
            "old_baseline_artifact": self.artifacts["performance_baseline"],
            "old_baseline": {
                "stage": "legacy_performance_baseline",
                "outcome": "BASELINE_RECORDED",
                "source_sha": LEGACY,
                "backend": "cupy",
                "device": "NVIDIA GeForce RTX 3090",
                "median_seconds": 2.5,
            },
            "go": True,
            "verdict_status": "GO",
            "preconditions": [{"name": "complete benchmark matrix", "ok": True}],
            "undefined_reasons": [],
            "outcome": "PERFORMANCE_GO",
        }

    def final_manifests(
        self, *, performance: dict | None = None,
        performance_bindings: dict | None = None,
    ) -> dict[str, Path]:
        selection = self.selection()
        manifests = {
            "selection": self.manifest("calibration_selection", selection),
            "replication_numpy": self.manifest(
                "replication_numpy", self.stage("replication", "numpy", selection)
            ),
            "replication_cupy": self.manifest(
                "replication_cupy", self.stage("replication", "cupy", selection)
            ),
            "held_out_cupy": self.manifest(
                "held_out_cupy", self.stage("held_out", "cupy", selection)
            ),
            "held_out_numpy": self.manifest(
                "held_out_numpy", self.stage("held_out", "numpy", selection)
            ),
        }
        selection_reference = self.manifest_reference(
            "calibration_selection", manifests["selection"]
        )
        if performance_bindings is not None:
            expected_bindings = {
                "selection_manifest_sha256": selection_reference["manifest_sha256"],
                "selected_current_pA": selection["selected_current_pA"],
                "compatibility_path": controller.COMPATIBILITY_PATH,
                "compatibility_sha256": self.config["compatibility"]["canonical_json_sha256"],
            }
            if performance_bindings != expected_bindings:
                selection_reference = dict(selection_reference)
                selection_reference["manifest_sha256"] = "f" * 64
        manifests["performance"] = self.manifest(
            "performance_candidate", performance or self.performance(),
            prerequisites=[selection_reference],
        )
        return manifests


@pytest.fixture
def fx(tmp_path, monkeypatch):
    return Fixture(tmp_path, monkeypatch)


def _emit_final(fx: Fixture, manifests: dict[str, Path], command: str = "final.json"):
    return controller.emit_final_merge(
        config_path=fx.config_path,
        selection_manifest=manifests["selection"],
        replication_numpy_manifest=manifests["replication_numpy"],
        replication_cupy_manifest=manifests["replication_cupy"],
        held_out_cupy_manifest=manifests["held_out_cupy"],
        held_out_numpy_manifest=manifests["held_out_numpy"],
        performance_manifest=manifests["performance"],
        emit=fx.root / f"commands/{command}", root=fx.root,
    )


def test_numpy_calibration_is_first_and_only_emits_command(fx: Fixture):
    emitted = fx.root / "commands/calibration-numpy.json"
    envelope = controller.emit_calibration(
        config_path=fx.config_path, backend="numpy", emit=emitted, root=fx.root,
    )

    assert envelope["execution"] == "not_executed"
    assert envelope["env"] == {"SIM_BACKEND": "numpy"}
    assert envelope["argv"][1:3] == ["-m", controller.RUNNER_MODULE]
    assert envelope["argv"][3:5] == [
        "--process-correction-spec", str((fx.root / controller.SEED_SPEC_PATH).resolve())
    ]
    assert "--calibration" in envelope["argv"]
    assert emitted.is_file()
    assert not fx.artifact_path("calibration_numpy").exists()


def test_cupy_calibration_requires_digested_completed_numpy(fx: Fixture):
    with pytest.raises(controller.ControllerError, match="requires a digested NumPy"):
        controller.emit_calibration(
            config_path=fx.config_path, backend="cupy",
            emit=fx.root / "commands/rejected.json", root=fx.root,
        )

    numpy_manifest = fx.manifest("calibration_numpy", fx.calibration("numpy"))
    envelope = controller.emit_calibration(
        config_path=fx.config_path, backend="cupy", numpy_manifest=numpy_manifest,
        emit=fx.root / "commands/calibration-cupy.json", root=fx.root,
    )
    assert envelope["action"] == "calibration_cupy"
    assert envelope["prerequisites"][2]["artifact_sha256"] == _digest(
        fx.artifact_path("calibration_numpy")
    )


def test_config_requires_replacement_seeds_and_matching_locked_spec(fx: Fixture):
    config = dict(fx.config)
    config["seeds"] = dict(config["seeds"], calibration=1013)
    config.pop("sha256")
    bad_config = _write_json(fx.root / "bad-correction.json", _seal(config))
    with pytest.raises(
        controller.ControllerError,
        match="exclude consumed and retired partitions",
    ):
        controller.load_config(bad_config, root=fx.root)

    spec = fx.root / controller.SEED_SPEC_PATH
    spec.write_text('{"partitions":{"calibration":[999]}}\n')
    with pytest.raises(controller.ControllerError, match="source digest mismatch"):
        controller.load_config(fx.config_path, root=fx.root)


def test_config_rejects_arbitrary_replacement_seed(fx: Fixture):
    config = dict(fx.config)
    config["seeds"] = dict(
        config["seeds"], calibration=config["seeds"]["calibration"] + 1
    )
    config.pop("sha256")
    bad_config = _write_json(fx.root / "arbitrary-seed.json", _seal(config))

    with pytest.raises(controller.ControllerError, match="mechanically derived replacement"):
        controller.load_config(bad_config, root=fx.root)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(schema="wrong"), "schema"),
        (lambda value: value.update(status="executed"), "status"),
        (
            lambda value: value.update(forbidden_consumed_seeds=[1013, 1019]),
            "consumed-seed",
        ),
        (lambda value: value.update(retired_unexecuted_seeds=[]), "retired-seed"),
        (
            lambda value: value["sealed_future_seeds"].update(stage_1=999),
            "future seed",
        ),
        (
            lambda value: value["seed_derivation"]["source_anchor"].update(
                committed_at="after-observation"
            ),
            "seed derivation",
        ),
        (
            lambda value: value["calibration"].update(ladder_pA=[100]),
            "calibration contract",
        ),
        (
            lambda value: value["merge_environment"]["calibration_selection"].clear(),
            "merge environment contract",
        ),
        (
            lambda value: value["artifact_manifest"].update(
                required_sealed_entries=[]
            ),
            "manifest seals",
        ),
        (lambda value: value.update(execution_order=[]), "execution order"),
        (lambda value: value.update(stop_rules=[]), "stop rules"),
    ],
)
def test_process_spec_execution_contract_is_interpreted_before_emission(
    fx: Fixture, mutate, message: str,
):
    config = controller.load_config(fx.config_path, root=fx.root)
    spec = copy.deepcopy(
        json.loads((fx.root / controller.SEED_SPEC_PATH).read_text())
    )
    mutate(spec)

    with pytest.raises(controller.ControllerError, match=message):
        controller._validate_process_correction_spec(spec, config=config)


def test_numpy_calibration_readiness_is_non_emitting_and_seed_free(fx: Fixture):
    command_dir = fx.root / "commands"
    readiness = controller.check_calibration_readiness(
        config_path=fx.config_path, backend="numpy", root=fx.root
    )

    assert readiness["ready"] is True
    assert readiness["backend"] == "numpy"
    assert readiness["prerequisite_kinds"] == [
        "strict_arithmetic_replay_v2", "compatibility",
    ]
    assert readiness["command_emitted"] is False
    assert readiness["execution"] == "not_executed"
    assert "seed" not in json.dumps(readiness).lower()
    assert not command_dir.exists()


def test_config_requires_passing_strict_arithmetic_replay(fx: Fixture):
    replay_path = fx.root / controller.STRICT_REPLAY_PATH
    replay = json.loads(replay_path.read_text())
    replay.pop("sha256")
    replay["outcome"] = "DIAGNOSTIC_FAIL"
    replay = _seal(replay)
    _write_json(replay_path, replay)
    config = dict(fx.config)
    config["strict_arithmetic_replay"] = dict(
        config["strict_arithmetic_replay"], sha256=_digest(replay_path)
    )
    config.pop("sha256")
    bad_config = _write_json(fx.root / "failed-replay.json", _seal(config))

    with pytest.raises(controller.ControllerError, match="has not earned"):
        controller.load_config(bad_config, root=fx.root)


def test_strict_replay_receipt_is_portable_across_worktree_roots(fx: Fixture):
    replay = json.loads((fx.root / controller.STRICT_REPLAY_PATH).read_text())
    receipt_path = fx.root / replay["cells"]["numpy"]["receipt_path"]
    receipt = json.loads(receipt_path.read_text())
    artifact = Path(receipt["artifact"]["path"])
    receipt["argv"][-1] = str(Path("/different/checkout") / artifact)
    _write_json(receipt_path, receipt)

    assert controller.load_config(fx.config_path, root=fx.root)["status"] == "frozen"


def test_source_manifest_must_be_bound_to_candidate_revision(
    fx: Fixture, monkeypatch: pytest.MonkeyPatch,
):
    original = controller._revision_file_digest

    def revision_digest(root: Path, revision: str, relative: str) -> str:
        if relative == fx.config["candidate_source_manifest"]["path"]:
            return "f" * 64
        return original(root, revision, relative)

    monkeypatch.setattr(controller, "_revision_file_digest", revision_digest)
    with pytest.raises(controller.ControllerError, match="manifest is not bound"):
        controller.load_config(fx.config_path, root=fx.root)


def test_candidate_source_closure_includes_every_sim_python_file(fx: Fixture):
    unimported = fx.root / "sim/dynamic_only.py"
    unimported.write_text("# dynamically selected simulator module\n")

    assert "sim/dynamic_only.py" in controller._required_candidate_source_paths(fx.root)


def test_merge_requires_matching_seed_source_compatibility_and_digests(fx: Fixture):
    numpy_manifest = fx.manifest("calibration_numpy", fx.calibration("numpy"))
    bad_cupy = fx.calibration("cupy")
    bad_cupy["source_identity"] = {**fx.source_identity, "sim/bridge.py": "f" * 64}
    cupy_manifest = fx.manifest("calibration_cupy", bad_cupy)
    with pytest.raises(controller.ControllerError, match="source identities differ"):
        controller.emit_merge_calibration(
            config_path=fx.config_path, numpy_manifest=numpy_manifest,
            cupy_manifest=cupy_manifest, emit=fx.root / "commands/rejected.json", root=fx.root,
        )


def test_merge_emits_only_after_both_calibration_artifacts_match(fx: Fixture):
    numpy_manifest = fx.manifest("calibration_numpy", fx.calibration("numpy"))
    cupy_manifest = fx.manifest("calibration_cupy", fx.calibration("cupy"))
    envelope = controller.emit_merge_calibration(
        config_path=fx.config_path, numpy_manifest=numpy_manifest,
        cupy_manifest=cupy_manifest, emit=fx.root / "commands/merge.json", root=fx.root,
    )
    assert envelope["action"] == "merge_calibration"
    assert envelope["env"] == controller._expected_manifest_env("calibration_selection")
    assert len(envelope["prerequisites"]) == 2
    assert not fx.artifact_path("calibration_selection").exists()


def test_replication_backends_schedule_independently_after_selection_go(fx: Fixture):
    selection_manifest = fx.manifest("calibration_selection", fx.selection())
    numpy = controller.emit_replication(
        config_path=fx.config_path, backend="numpy", selection_manifest=selection_manifest,
        emit=fx.root / "commands/repl-numpy.json", root=fx.root,
    )
    cupy = controller.emit_replication(
        config_path=fx.config_path, backend="cupy", selection_manifest=selection_manifest,
        emit=fx.root / "commands/repl-cupy.json", root=fx.root,
    )
    assert numpy["env"] == {"SIM_BACKEND": "numpy"}
    assert cupy["env"] == {"SIM_BACKEND": "cupy"}


def test_replication_refuses_non_go_selection(fx: Fixture):
    selection_manifest = fx.manifest("calibration_selection", fx.selection(go=False))
    with pytest.raises(controller.ControllerError, match="has not earned GO"):
        controller.emit_replication(
            config_path=fx.config_path, backend="numpy", selection_manifest=selection_manifest,
            emit=fx.root / "commands/rejected.json", root=fx.root,
        )


def test_held_out_requires_both_replications_and_enforces_cupy_first(fx: Fixture):
    selection = fx.selection()
    selection_manifest = fx.manifest("calibration_selection", selection)
    repl_numpy_manifest = fx.manifest(
        "replication_numpy", fx.stage("replication", "numpy", selection)
    )
    repl_cupy_manifest = fx.manifest(
        "replication_cupy", fx.stage("replication", "cupy", selection)
    )

    with pytest.raises(controller.ControllerError, match="completed digested CuPy"):
        controller.emit_held_out(
            config_path=fx.config_path, backend="numpy",
            selection_manifest=selection_manifest,
            replication_numpy_manifest=repl_numpy_manifest,
            replication_cupy_manifest=repl_cupy_manifest,
            emit=fx.root / "commands/rejected.json", root=fx.root,
        )

    cupy_envelope = controller.emit_held_out(
        config_path=fx.config_path, backend="cupy",
        selection_manifest=selection_manifest,
        replication_numpy_manifest=repl_numpy_manifest,
        replication_cupy_manifest=repl_cupy_manifest,
        emit=fx.root / "commands/held-cupy.json", root=fx.root,
    )
    assert cupy_envelope["action"] == "held_out_cupy"

    held_cupy_manifest = fx.manifest(
        "held_out_cupy", fx.stage("held_out", "cupy", selection)
    )
    numpy_envelope = controller.emit_held_out(
        config_path=fx.config_path, backend="numpy",
        selection_manifest=selection_manifest,
        replication_numpy_manifest=repl_numpy_manifest,
        replication_cupy_manifest=repl_cupy_manifest,
        cupy_held_out_manifest=held_cupy_manifest,
        emit=fx.root / "commands/held-numpy.json", root=fx.root,
    )
    assert numpy_envelope["prerequisites"][-1]["kind"] == "held_out_cupy"


def test_held_out_refuses_any_replication_no_go(fx: Fixture):
    selection = fx.selection()
    selection_manifest = fx.manifest("calibration_selection", selection)
    repl_numpy = fx.stage("replication", "numpy", selection)
    repl_numpy["go"] = False
    repl_numpy["outcome"] = "REPLICATION_NO_GO"
    repl_numpy_manifest = fx.manifest("replication_numpy", repl_numpy)
    repl_cupy_manifest = fx.manifest(
        "replication_cupy", fx.stage("replication", "cupy", selection)
    )
    with pytest.raises(controller.ControllerError, match="has not earned GO"):
        controller.emit_held_out(
            config_path=fx.config_path, backend="cupy",
            selection_manifest=selection_manifest,
            replication_numpy_manifest=repl_numpy_manifest,
            replication_cupy_manifest=repl_cupy_manifest,
            emit=fx.root / "commands/rejected.json", root=fx.root,
        )


def test_performance_baseline_requires_exact_old_source_and_runner(fx: Fixture, monkeypatch):
    monkeypatch.setattr(
        controller, "_git_head",
        lambda path: "c" * 40 if path.resolve() == fx.legacy_root.resolve() else CANDIDATE,
    )
    with pytest.raises(controller.ControllerError, match="exact required old revision"):
        controller.emit_performance_baseline(
            config_path=fx.config_path, source_root=fx.legacy_root,
            emit=fx.root / "commands/rejected.json", root=fx.root,
        )

    monkeypatch.setattr(
        controller, "_git_head",
        lambda path: LEGACY if path.resolve() == fx.legacy_root.resolve() else CANDIDATE,
    )
    envelope = controller.emit_performance_baseline(
        config_path=fx.config_path, source_root=fx.legacy_root,
        emit=fx.root / "commands/performance-baseline.json", root=fx.root,
    )
    assert envelope["source_revision"] == LEGACY
    assert envelope["cwd"] == str(fx.legacy_root.resolve())
    assert "--legacy-performance-baseline" in envelope["argv"]


def test_candidate_performance_requires_digested_baseline_and_held_out_go(fx: Fixture):
    selection = fx.selection()
    selection_manifest = fx.manifest("calibration_selection", selection)
    held_cupy_manifest = fx.manifest(
        "held_out_cupy", fx.stage("held_out", "cupy", selection)
    )
    held_numpy_manifest = fx.manifest(
        "held_out_numpy", fx.stage("held_out", "numpy", selection)
    )
    baseline_manifest = fx.manifest("performance_baseline", {
        "stage": "legacy_performance_baseline",
        "outcome": "BASELINE_RECORDED",
        "source_sha": LEGACY,
        "backend": "cupy",
        "device": "NVIDIA GeForce RTX 3090",
        "median_seconds": 2.5,
    })
    envelope = controller.emit_performance_candidate(
        config_path=fx.config_path, baseline_manifest=baseline_manifest,
        selection_manifest=selection_manifest,
        held_out_cupy_manifest=held_cupy_manifest,
        held_out_numpy_manifest=held_numpy_manifest,
        emit=fx.root / "commands/performance.json", root=fx.root,
    )
    assert envelope["action"] == "performance_candidate"
    assert envelope["prerequisites"][0]["kind"] == "performance_baseline"
    assert "--old-baseline" in envelope["argv"]


def test_manifest_byte_tamper_and_create_only_outputs_fail_closed(fx: Fixture):
    selection_manifest = fx.manifest("calibration_selection", fx.selection())
    fx.artifact_path("calibration_selection").write_text("{}\n")
    with pytest.raises(controller.ControllerError, match="digest changed"):
        controller.emit_replication(
            config_path=fx.config_path, backend="numpy", selection_manifest=selection_manifest,
            emit=fx.root / "commands/rejected.json", root=fx.root,
        )

    fx.artifact_path("calibration_selection").unlink()
    emitted = fx.root / "commands/existing.json"
    emitted.parent.mkdir(parents=True, exist_ok=True)
    emitted.write_text("keep\n")
    with pytest.raises(controller.ControllerError, match="refusing to overwrite command envelope"):
        controller.emit_calibration(
            config_path=fx.config_path, backend="numpy", emit=emitted, root=fx.root,
        )
    assert emitted.read_text() == "keep\n"


def test_manifest_requires_exact_adapter_field_set(fx: Fixture):
    path = fx.manifest("calibration_selection", fx.selection())
    _rewrite_sealed(path, lambda manifest: manifest.__setitem__("unexpected", True))

    with pytest.raises(controller.ControllerError, match="missing or extra fields"):
        _load_fixture_manifest(fx, path, "calibration_selection")


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("artifact", "artifact reference"),
        ("command_envelope", "command envelope reference"),
        ("execution_receipt", "execution receipt reference"),
    ],
)
def test_manifest_requires_exact_nested_reference_fields(
    fx: Fixture, field: str, message: str,
):
    path = fx.manifest("calibration_selection", fx.selection())
    _rewrite_sealed(
        path,
        lambda manifest: manifest[field].__setitem__("unexpected", True),
    )

    with pytest.raises(controller.ControllerError, match=message):
        _load_fixture_manifest(fx, path, "calibration_selection")


def test_manifest_rejects_unsafe_envelope_and_receipt_paths(fx: Fixture):
    path = fx.manifest("calibration_selection", fx.selection())
    _rewrite_sealed(
        path,
        lambda manifest: manifest["command_envelope"].__setitem__(
            "path", "../outside-command.json"
        ),
    )
    with pytest.raises(controller.ControllerError, match="safe repository-relative"):
        _load_fixture_manifest(fx, path, "calibration_selection")

    path = fx.manifest("calibration_selection", fx.selection())
    _rewrite_sealed(
        path,
        lambda manifest: manifest["execution_receipt"].__setitem__(
            "path", "../outside-receipt.json"
        ),
    )
    with pytest.raises(controller.ControllerError, match="safe repository-relative"):
        _load_fixture_manifest(fx, path, "calibration_selection")


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("command_envelope", "command envelope is missing or its digest changed"),
        ("execution_receipt", "execution receipt is missing or its digest changed"),
    ],
)
def test_manifest_rehashes_referenced_evidence(
    fx: Fixture, reference: str, message: str,
):
    path = fx.manifest("calibration_selection", fx.selection())
    manifest = json.loads(path.read_text())
    referenced = fx.root / manifest[reference]["path"]
    referenced.write_text(referenced.read_text() + " ")

    with pytest.raises(controller.ControllerError, match=message):
        _load_fixture_manifest(fx, path, "calibration_selection")


def test_manifest_revalidates_frozen_command_fields(fx: Fixture):
    path = fx.manifest("calibration_selection", fx.selection())
    manifest = json.loads(path.read_text())
    command_path = fx.root / manifest["command_envelope"]["path"]
    command = json.loads(command_path.read_text())
    command["env"] = {"SIM_BACKEND": "cupy"}
    _write_json(command_path, command)
    _rewrite_sealed(
        path,
        lambda value: value["command_envelope"].__setitem__(
            "sha256", _digest(command_path)
        ),
    )

    with pytest.raises(controller.ControllerError, match="environment differs"):
        _load_fixture_manifest(fx, path, "calibration_selection")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("host", "other-host", "host or device differs"),
        ("device", "other-device", "host or device differs"),
        ("started_utc_ns", 111, "timestamps are invalid"),
    ],
)
def test_manifest_revalidates_receipt_identity_and_ordering(
    fx: Fixture, field: str, value, message: str,
):
    path = fx.manifest("calibration_selection", fx.selection())
    _rewrite_sealed(
        path,
        lambda manifest: manifest["execution_receipt"].__setitem__(field, value),
    )

    with pytest.raises(controller.ControllerError, match=message):
        _load_fixture_manifest(fx, path, "calibration_selection")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt.__setitem__("argv", ["false"]), "receipt argv differs"),
        (
            lambda receipt: receipt.__setitem__(
                "env_allowlist", {"SIM_BACKEND": "cupy"}
            ),
            "receipt environment differs|environment backend does not match receipt",
        ),
        (
            lambda receipt: receipt["source"].__setitem__("git_sha", LEGACY),
            "receipt source revision is invalid|Git SHA does not match receipt",
        ),
        (lambda receipt: receipt.__setitem__("status", "failed"), "receipt is invalid"),
    ],
)
def test_manifest_reverifies_receipt_binding_and_success(
    fx: Fixture, mutation, message: str,
):
    path = fx.manifest("calibration_selection", fx.selection())
    manifest = json.loads(path.read_text())
    receipt_path = fx.root / manifest["execution_receipt"]["path"]
    receipt = json.loads(receipt_path.read_text())
    mutation(receipt)
    _write_json(receipt_path, receipt)
    _rewrite_sealed(
        path,
        lambda value: value["execution_receipt"].__setitem__(
            "sha256", _digest(receipt_path)
        ),
    )

    with pytest.raises(controller.ControllerError, match=message):
        _load_fixture_manifest(fx, path, "calibration_selection")


def test_manifest_rejects_resealed_semantically_false_provenance(fx: Fixture):
    path = fx.manifest("calibration_selection", fx.selection())
    manifest = json.loads(path.read_text())
    sidecar_path = fx.root / manifest["provenance_sidecar"]["path"]
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["sim_backend"] = "cupy"
    _write_json(sidecar_path, sidecar)

    receipt_path = fx.root / manifest["execution_receipt"]["path"]
    receipt = json.loads(receipt_path.read_text())
    receipt["provenance"]["sha256"] = _digest(sidecar_path)
    _write_json(receipt_path, receipt)

    def reseal(value: dict) -> None:
        value["provenance_sidecar"]["sha256"] = _digest(sidecar_path)
        value["execution_receipt"]["sha256"] = _digest(receipt_path)

    _rewrite_sealed(path, reseal)

    with pytest.raises(controller.ControllerError, match="backend"):
        _load_fixture_manifest(fx, path, "calibration_selection")


def test_manifest_rejects_symlinked_artifact(fx: Fixture):
    path = fx.manifest("calibration_selection", fx.selection())
    artifact_path = fx.artifact_path("calibration_selection")
    target = artifact_path.with_name("selection-target.json")
    artifact_path.rename(target)
    artifact_path.symlink_to(target)

    with pytest.raises(controller.ControllerError, match="symlink"):
        _load_fixture_manifest(fx, path, "calibration_selection")


def test_manifest_rejects_receipt_for_noncanonical_artifact(fx: Fixture):
    path = fx.manifest("calibration_selection", fx.selection())
    manifest = json.loads(path.read_text())
    other = _write_json(fx.root / "evidence/other.json", {"dummy": "other"})
    receipt_path = fx.root / manifest["execution_receipt"]["path"]
    receipt = json.loads(receipt_path.read_text())
    receipt["artifact"] = {
        "path": other.relative_to(fx.root).as_posix(),
        "sha256": _digest(other),
        "size_bytes": other.stat().st_size,
    }
    _write_json(receipt_path, receipt)
    _rewrite_sealed(
        path,
        lambda value: value["execution_receipt"].__setitem__(
            "sha256", _digest(receipt_path)
        ),
    )

    with pytest.raises(
        controller.ControllerError,
        match="non-canonical artifact|provenance path is not adjacent",
    ):
        _load_fixture_manifest(fx, path, "calibration_selection")


def test_existing_scientific_artifact_target_is_never_overwritten(fx: Fixture):
    target = fx.artifact_path("calibration_numpy")
    target.parent.mkdir(parents=True)
    target.write_text("existing evidence\n")
    with pytest.raises(controller.ControllerError, match="existing artifact"):
        controller.emit_calibration(
            config_path=fx.config_path, backend="numpy",
            emit=fx.root / "commands/rejected.json", root=fx.root,
        )
    assert target.read_text() == "existing evidence\n"


def test_manifest_rejects_formatting_only_controller_config_rewrite(fx: Fixture):
    manifest = fx.manifest("calibration_numpy", fx.calibration("numpy"))
    fx.config_path.write_text(
        json.dumps(fx.config, separators=(",", ":")), encoding="utf-8"
    )
    config = controller.load_config(fx.config_path, root=fx.root)

    with pytest.raises(controller.ControllerError, match="exact-byte digest differs"):
        controller.load_manifest(
            manifest,
            config=config,
            kind="calibration_numpy",
            root=fx.root,
        )


def test_final_merge_emits_complete_digested_runner_command(fx: Fixture):
    manifests = fx.final_manifests()
    envelope = _emit_final(fx, manifests)

    paths = {kind: str(fx.artifact_path(kind)) for kind in fx.artifacts}
    merge_index = envelope["argv"].index("--merge-final")
    assert envelope["action"] == "final_stage0_merge"
    assert envelope["env"] == controller._expected_manifest_env("final_stage0")
    assert envelope["execution"] == "not_executed"
    assert envelope["expected_result"] == {
        "stage": "final_cross_backend", "outcome": "TONIC_OUTPUT_GO", "go": True,
    }
    assert envelope["argv"][merge_index + 1:merge_index + 7] == [
        str((fx.root / controller.COMPATIBILITY_PATH).resolve()),
        paths["replication_numpy"], paths["replication_cupy"],
        paths["held_out_cupy"], paths["held_out_numpy"],
        paths["performance_candidate"],
    ]
    assert [item["kind"] for item in envelope["prerequisites"]] == [
        "compatibility", "calibration_selection", "replication_numpy",
        "replication_cupy", "held_out_cupy", "held_out_numpy",
        "performance_candidate",
    ]
    assert "artifact_sha256" not in envelope["prerequisites"][0]
    assert all(
        "artifact_sha256" in item for item in envelope["prerequisites"][1:]
    )
    assert not fx.artifact_path("final_stage0").exists()


def test_final_merge_requires_every_manifest(fx: Fixture):
    manifests = fx.final_manifests()
    manifests["held_out_numpy"] = fx.manifest_dir / "missing-held-out-numpy.json"
    with pytest.raises(controller.ControllerError, match="does not exist"):
        _emit_final(fx, manifests)


def test_final_merge_requires_exact_performance_go_outcome(fx: Fixture):
    performance = fx.performance()
    performance.update({
        "go": False,
        "verdict_status": "NO-GO",
        "outcome": "PERFORMANCE_NO_GO",
    })
    manifests = fx.final_manifests(performance=performance)
    with pytest.raises(controller.ControllerError, match="earned PERFORMANCE_GO"):
        _emit_final(fx, manifests)


def test_final_merge_refuses_undefined_input_even_if_go_fields_claim_success(fx: Fixture):
    manifests = fx.final_manifests()
    replication = json.loads(fx.artifact_path("replication_numpy").read_text())
    replication["verdict_status"] = "UNDEFINED"
    replication["undefined_reasons"] = ["execution receipt missing"]
    manifests["replication_numpy"] = fx.manifest("replication_numpy", replication)
    with pytest.raises(controller.ControllerError, match="earned REPLICATION_GO"):
        _emit_final(fx, manifests)


def test_final_merge_refuses_performance_selection_or_compatibility_mismatch(fx: Fixture):
    selection = fx.selection()
    selection_path = fx.manifest("calibration_selection", selection)
    selection_seal = json.loads(selection_path.read_text())
    manifests = fx.final_manifests(performance_bindings={
        "selection_manifest_sha256": selection_seal["sha256"],
        "selected_current_pA": 125,
        "compatibility_path": controller.COMPATIBILITY_PATH,
        "compatibility_sha256": fx.config["compatibility"]["canonical_json_sha256"],
    })
    with pytest.raises(controller.ControllerError, match="not bound to the selected"):
        _emit_final(fx, manifests)


def test_final_merge_refuses_existing_final_artifact(fx: Fixture):
    manifests = fx.final_manifests()
    target = fx.artifact_path("final_stage0")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("existing final evidence\n")
    with pytest.raises(controller.ControllerError, match="existing artifact"):
        _emit_final(fx, manifests)
    assert target.read_text() == "existing final evidence\n"


def test_final_merge_cli_dispatches_without_executing_runner(fx: Fixture):
    manifests = fx.final_manifests()
    command = fx.root / "commands/final-cli.json"
    result = controller.main([
        "--root", str(fx.root),
        "--config", str(fx.config_path),
        "--emit", str(command),
        "final-merge",
        "--selection-manifest", str(manifests["selection"]),
        "--replication-numpy-manifest", str(manifests["replication_numpy"]),
        "--replication-cupy-manifest", str(manifests["replication_cupy"]),
        "--held-out-cupy-manifest", str(manifests["held_out_cupy"]),
        "--held-out-numpy-manifest", str(manifests["held_out_numpy"]),
        "--performance-manifest", str(manifests["performance"]),
    ])
    assert result == 0
    assert json.loads(command.read_text())["execution"] == "not_executed"
    assert not fx.artifact_path("final_stage0").exists()
