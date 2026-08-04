from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import experiment_automation_lifecycle as lifecycle


def _write_evidence(root: Path, stage: str) -> dict[str, str]:
    path = root / f"evidence/{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stage": stage}) + "\n", encoding="utf-8")
    return {
        "path": path.relative_to(root).as_posix(),
        "role": lifecycle.REQUIRED_ROLES[stage],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _manifest(root: Path, frontier: str | None = "execute", status: str = "ready") -> dict:
    stages = {}
    frontier_index = len(lifecycle.STAGES) if frontier is None else lifecycle.STAGES.index(frontier)
    for index, stage in enumerate(lifecycle.STAGES):
        if index < frontier_index or frontier is None:
            stage_status = "complete"
            evidence = [_write_evidence(root, stage)]
        elif index == frontier_index:
            stage_status = status
            evidence = []
        else:
            stage_status = "pending"
            evidence = []
        stages[stage] = {
            "status": stage_status,
            "evidence": evidence,
            "blockers": ["waiting for a verified source archive"] if stage_status == "blocked" else [],
        }
    return {
        "schema": lifecycle.SCHEMA,
        "experiment_id": "bounded-test-v1",
        "stages": stages,
        "compute_targets": {
            "local_cpu": {"disposition": "planned", "reason": "validation and comparison"},
            "local_gpu": {"disposition": "planned", "reason": "candidate execution"},
            "mini_pc_cluster": {"disposition": "not_applicable", "reason": "single-cell diagnostic"},
        },
    }


def test_reports_one_ready_frontier_and_all_compute_lanes(tmp_path: Path) -> None:
    result = lifecycle.validate_manifest(_manifest(tmp_path), root=tmp_path)

    assert result["valid"] is True
    assert result["state"] == "ready"
    assert result["frontier"] == "execute"
    assert tuple(result["compute_targets"]) == lifecycle.COMPUTE_TARGETS


def test_accepts_a_complete_lifecycle(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, frontier=None)
    manifest["compute_targets"]["local_cpu"]["disposition"] = "complete"
    manifest["compute_targets"]["local_gpu"]["disposition"] = "complete"

    result = lifecycle.validate_manifest(manifest, root=tmp_path)

    assert result["state"] == "complete"
    assert result["frontier"] is None


def test_blocked_frontier_requires_a_specific_blocker(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, frontier="research", status="blocked")
    manifest["stages"]["research"]["blockers"] = []

    with pytest.raises(lifecycle.LifecycleError, match="must name at least one blocker"):
        lifecycle.validate_manifest(manifest, root=tmp_path)


def test_rejects_completed_stage_after_frontier(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, frontier="research")
    manifest["stages"]["preregister"] = {
        "status": "complete",
        "evidence": [_write_evidence(tmp_path, "preregister")],
        "blockers": [],
    }

    with pytest.raises(lifecycle.LifecycleError, match="complete after the lifecycle frontier"):
        lifecycle.validate_manifest(manifest, root=tmp_path)


def test_rejects_pending_stage_with_claimed_evidence(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, frontier="execute")
    manifest["stages"]["validate"]["evidence"] = [_write_evidence(tmp_path, "validate")]

    with pytest.raises(lifecycle.LifecycleError, match="pending stage validate cannot claim evidence"):
        lifecycle.validate_manifest(manifest, root=tmp_path)


def test_rejects_missing_required_evidence_role(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, frontier="research")
    manifest["stages"]["scope"]["evidence"][0]["role"] = "unrelated_note"

    with pytest.raises(lifecycle.LifecycleError, match="lacks required 'scope_definition'"):
        lifecycle.validate_manifest(manifest, root=tmp_path)


def test_rejects_changed_evidence(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, frontier="research")
    evidence = tmp_path / manifest["stages"]["scope"]["evidence"][0]["path"]
    evidence.write_text("changed\n", encoding="utf-8")

    with pytest.raises(lifecycle.LifecycleError, match="evidence digest mismatch"):
        lifecycle.validate_manifest(manifest, root=tmp_path)


def test_rejects_evidence_reused_across_stages(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, frontier="preregister")
    duplicate = dict(manifest["stages"]["scope"]["evidence"][0])
    duplicate["role"] = lifecycle.REQUIRED_ROLES["research"]
    manifest["stages"]["research"]["evidence"] = [duplicate]

    with pytest.raises(lifecycle.LifecycleError, match="assigned to more than one stage"):
        lifecycle.validate_manifest(manifest, root=tmp_path)


def test_rejects_symlink_evidence(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, frontier="research")
    evidence = manifest["stages"]["scope"]["evidence"][0]
    target = tmp_path / evidence["path"]
    link = tmp_path / "evidence/scope-link.json"
    link.symlink_to(target)
    evidence["path"] = link.relative_to(tmp_path).as_posix()

    with pytest.raises(lifecycle.LifecycleError, match="cannot be a symlink"):
        lifecycle.validate_manifest(manifest, root=tmp_path)


def test_requires_explicit_disposition_for_all_compute_targets(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    del manifest["compute_targets"]["mini_pc_cluster"]

    with pytest.raises(lifecycle.LifecycleError, match="must name exactly"):
        lifecycle.validate_manifest(manifest, root=tmp_path)


def test_not_applicable_compute_target_requires_a_reason(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["compute_targets"]["mini_pc_cluster"]["reason"] = ""

    with pytest.raises(lifecycle.LifecycleError, match="reason must be a non-empty string"):
        lifecycle.validate_manifest(manifest, root=tmp_path)


def test_execute_complete_rejects_still_planned_compute(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, frontier="validate")

    with pytest.raises(lifecycle.LifecycleError, match="still planned"):
        lifecycle.validate_manifest(manifest, root=tmp_path)


def test_execute_complete_requires_a_used_compute_target(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, frontier="validate")
    for record in manifest["compute_targets"].values():
        record["disposition"] = "not_applicable"

    with pytest.raises(lifecycle.LifecycleError, match="planned or complete"):
        lifecycle.validate_manifest(manifest, root=tmp_path)


def test_cli_is_read_only_and_returns_nonzero_on_invalid_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _manifest(tmp_path)
    manifest["schema"] = "wrong"
    manifest_path = tmp_path / "lifecycle.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    assert lifecycle.main([str(manifest_path), "--root", str(tmp_path)]) == 1

    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before
    assert "NOT READY" in capsys.readouterr().err
