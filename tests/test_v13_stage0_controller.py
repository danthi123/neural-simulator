"""Synthetic state-machine tests for the external V13 Stage-0 controller.

No scientific runner is imported or executed by this test module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import v13_stage0_controller as controller


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
    result["sha256"] = hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return result


class Fixture:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.root = tmp_path / "candidate"
        self.root.mkdir()
        self.legacy_root = tmp_path / "legacy"
        self.legacy_root.mkdir()
        self.manifest_dir = tmp_path / "manifests"
        self.manifest_dir.mkdir()

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
        spec = _write_json(
            self.root / controller.SEED_SPEC_PATH,
            {"partitions": {"calibration": [2003], "replication": [2009], "held_out": [2011]}},
        )
        for relative in controller.CRITICAL_SOURCE_PATHS:
            path = self.root / relative
            if path == spec:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# frozen candidate source: {relative}\n")
        self.source_identity = {
            relative: _digest(self.root / relative)
            for relative in controller.CRITICAL_SOURCE_PATHS
        }
        legacy_runner = self.legacy_root / "research/runners/_vocal_action_credit_gate_v13_tonic_output.py"
        legacy_runner.parent.mkdir(parents=True)
        legacy_runner.write_text("# sealed old runner\n")

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
            "python": "/usr/bin/python3",
            "runner_module": controller.RUNNER_MODULE,
            "seeds": {"calibration": 2003, "replication": 2009, "held_out": 2011},
            "seed_binding": {
                "path": controller.SEED_SPEC_PATH,
                "sha256": _digest(spec),
            },
            "compatibility": {
                "path": controller.COMPATIBILITY_PATH,
                "sha256": _digest(compatibility),
            },
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

    def artifact_path(self, kind: str) -> Path:
        return self.root / self.artifacts[kind]

    def manifest(self, kind: str, artifact: dict, *, bindings: dict | None = None) -> Path:
        artifact_path = _write_json(self.artifact_path(kind), artifact)
        source_revision = LEGACY if kind == "performance_baseline" else CANDIDATE
        body = {
            "schema": controller.MANIFEST_SCHEMA,
            "kind": kind,
            "config_sha256": self.config["sha256"],
            "source_revision": source_revision,
            "artifact": {"path": self.artifacts[kind], "sha256": _digest(artifact_path)},
        }
        if bindings is not None:
            body["bindings"] = bindings
        return _write_json(self.manifest_dir / f"{kind}.manifest.json", _seal(body))

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
            "seed": 2003,
            "source_sha": CANDIDATE,
            "source_identity": self.source_identity,
            "spec_sha256": "6" * 64,
            "compatibility_correction": {
                "path": controller.COMPATIBILITY_PATH,
                "sha256": self.config["compatibility"]["sha256"],
            },
            "rows": rows,
            "passing_currents_pA": [100],
        }

    def selection(self, *, go: bool = True) -> dict:
        return {
            "stage": "calibration_cross_backend",
            "backend": "cross_backend",
            "seed": 2003,
            "source_identity": self.source_identity,
            "compatibility_correction": {
                "path": controller.COMPATIBILITY_PATH,
                "sha256": self.config["compatibility"]["sha256"],
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
        seed = 2009 if stage == "replication" else 2011
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
        selection_seal = json.loads(manifests["selection"].read_text())
        bindings = performance_bindings or {
            "selection_manifest_sha256": selection_seal["sha256"],
            "selected_current_pA": selection["selected_current_pA"],
            "compatibility_path": controller.COMPATIBILITY_PATH,
            "compatibility_sha256": self.config["compatibility"]["sha256"],
        }
        manifests["performance"] = self.manifest(
            "performance_candidate", performance or self.performance(), bindings=bindings
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
    assert envelope["argv"][1:4] == ["-m", controller.RUNNER_MODULE, "--calibration"]
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
    assert envelope["prerequisites"][1]["artifact_sha256"] == _digest(
        fx.artifact_path("calibration_numpy")
    )


def test_config_requires_replacement_seeds_and_matching_locked_spec(fx: Fixture):
    config = dict(fx.config)
    config["seeds"] = dict(config["seeds"], calibration=1013)
    config.pop("sha256")
    bad_config = _write_json(fx.root / "bad-correction.json", _seal(config))
    with pytest.raises(controller.ControllerError, match="replace consumed seed 1013"):
        controller.load_config(bad_config, root=fx.root)

    spec = fx.root / controller.SEED_SPEC_PATH
    spec.write_text('{"partitions":{"calibration":[999]}}\n')
    with pytest.raises(controller.ControllerError, match="working source has changed"):
        controller.load_config(fx.config_path, root=fx.root)


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


def test_final_merge_emits_complete_digested_runner_command(fx: Fixture):
    manifests = fx.final_manifests()
    envelope = _emit_final(fx, manifests)

    paths = {kind: str(fx.artifact_path(kind)) for kind in fx.artifacts}
    merge_index = envelope["argv"].index("--merge-final")
    assert envelope["action"] == "final_stage0_merge"
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
    assert all("artifact_sha256" in item for item in envelope["prerequisites"])
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
        "compatibility_sha256": fx.config["compatibility"]["sha256"],
    })
    with pytest.raises(controller.ControllerError, match="bindings do not match"):
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
