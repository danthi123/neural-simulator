"""Focused tests for the bounded V14 Stage B campaign supervisor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sim.snr_executable_packet import canonical_bytes
from tools.compact_trace import save_compact_trace
import tools.v14_stageB_campaign_supervisor as supervisor


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _write(path: Path, value: object) -> str:
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _candidate_sha(arm: str, batch_index: int) -> str:
    return hashlib.sha256(f"{arm}:{batch_index}".encode()).hexdigest()


def _campaign_fixture(
    tmp_path: Path,
    *,
    arms: tuple[str, ...] = supervisor.ARM_ORDER,
    batches_per_arm: int = 1,
) -> tuple[Path, str, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    declarations = []
    for arm in arms:
        for batch_index in range(batches_per_arm):
            declaration_body = {
                "schema": "v14-snr-stageB-batch-declaration-test",
                "arm": arm,
                "analysis_protocol": {"path": "protocol.json", "sha256": "0" * 64},
                "candidates": [{
                    "candidate_id": f"candidate-{batch_index:04d}",
                    "candidate_sha256": _candidate_sha(arm, batch_index),
                }],
            }
            declaration = {**declaration_body, "sha256": _digest(declaration_body)}
            declaration_path = root / "campaign" / "declarations" / arm / f"batch-{batch_index:03d}.json"
            declaration_file_sha = _write(declaration_path, declaration)
            declarations.append({
                "arm": arm,
                "batch_index": batch_index,
                "candidate_count": 1,
                "path": declaration_path.relative_to(root).as_posix(),
                "sha256": declaration_file_sha,
                "declaration_sha256": declaration["sha256"],
            })
    body = {
        "schema": supervisor.PHASED_CAMPAIGN_SCHEMA,
        "status": "materialized-not-executed",
        "engineering_screening_only": True,
        "scientific_verdict": None,
        "numpy_confirmation_required": True,
        "candidate_manifest": {"path": "manifest.json", "sha256": "2" * 64},
        "analysis_protocol": {"path": "protocol.json", "sha256": "3" * 64},
        "packet_template": {"path": "template.json", "sha256": "4" * 64},
        "candidate_count": 512,
        "arm_count": 5,
        "batch_size": 512,
        "batch_count": len(arms) * batches_per_arm,
        "declarations": declarations,
    }
    campaign = {**body, "sha256": _digest(body)}
    campaign_path = root / "campaign" / "campaign.json"
    campaign_file_sha = _write(campaign_path, campaign)
    return root, campaign_file_sha, campaign_path


def _fake_receipt(
    root: Path,
    output: Path,
    campaign_sha: str,
    campaign_path: Path,
    arm: str,
    batch_index: int = 0,
) -> None:
    batch = output / arm / f"batch-{batch_index:03d}"
    batch.mkdir(parents=True, exist_ok=True)
    trace = batch / "candidate-0000.trace.zip"
    times = np.arange(4, dtype=np.float64) * 0.001 + 0.001
    voltage = np.zeros(4, dtype=np.float64)
    spikes = np.zeros(4, dtype=np.bool_)
    trace_sha = save_compact_trace(trace, times, voltage, spikes)
    declaration_path = root / "campaign" / "declarations" / arm / f"batch-{batch_index:03d}.json"
    declaration = json.loads(declaration_path.read_text())
    declaration_ref = {
        "path": declaration_path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(canonical_bytes(declaration)).hexdigest(),
        "declaration_sha256": declaration["sha256"],
    }
    campaign_ref = {"path": campaign_path.relative_to(root).as_posix(), "sha256": campaign_sha}
    body = {
        "schema": supervisor.PHASED_GPU_BATCH_RECEIPT_SCHEMA,
        "process_status": "completed",
        "engineering_screening_only": True,
        "scientific_verdict": None,
        "numpy_confirmation_required": True,
        "campaign": campaign_ref,
        "declaration": declaration_ref,
        "arm": arm,
        "batch_index": 0,
        "execution": {},
        "provenance": {},
        "traces": [{
            "candidate_id": f"candidate-{batch_index:04d}",
            "candidate_sha256": _candidate_sha(arm, batch_index),
            "termination": "test",
            "compact_trace": {
                "path": trace.relative_to(root).as_posix(),
                "sha256": trace_sha,
                "sample_count": 4,
            },
            "runtime_intervention": {"test": True},
        }],
    }
    _write(batch / "receipt.json", {**body, "sha256": _digest(body)})


def test_status_is_read_only_and_selects_cheapest_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, campaign_sha, campaign_path = _campaign_fixture(tmp_path)
    output = root / "results"
    _fake_receipt(root, output, campaign_sha, campaign_path, "nap_lesion")
    before = sorted(path.relative_to(root).as_posix() for path in output.rglob("*"))

    def forbidden(*args, **kwargs):
        raise AssertionError("status must not preflight or execute")

    monkeypatch.setattr(supervisor, "preflight_gpu", forbidden)
    result = supervisor.supervise_campaign(campaign_path, campaign_sha, output, repository_root=root, status=True)
    assert result["next"] == {"arm": "intact_autonomous", "batch_index": 0}
    assert result["completed_count"] == 1
    assert not (output / supervisor.STATE_FILENAME).exists()
    assert before == sorted(path.relative_to(root).as_posix() for path in output.rglob("*"))


def test_executes_one_missing_batch_and_persists_digest_bound_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, campaign_sha, campaign_path = _campaign_fixture(tmp_path)
    output = root / "results"
    _fake_receipt(root, output, campaign_sha, campaign_path, "nap_lesion")
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(supervisor, "preflight_gpu", lambda **kwargs: {"interpreter": "test", "cupy_version": "test"})

    def fake_run(campaign, digest, arm, index, output_dir, **kwargs):
        calls.append((arm, index))
        _fake_receipt(root, output, digest, campaign_path, arm)
        return {"ok": True}

    monkeypatch.setattr(supervisor, "run_gpu_batch", fake_run)
    result = supervisor.supervise_campaign(campaign_path, campaign_sha, output, repository_root=root)
    assert calls == [("intact_autonomous", 0)]
    assert result["executed"] is True
    state = json.loads((output / supervisor.STATE_FILENAME).read_text())
    assert state["status"] == "ready"
    assert state["campaign"] == {"path": campaign_path.relative_to(root).as_posix(), "sha256": campaign_sha}
    assert state["sha256"] == _digest({key: value for key, value in state.items() if key != "sha256"})


def test_rejects_wrong_interpreter_and_output_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, campaign_sha, campaign_path = _campaign_fixture(tmp_path)
    monkeypatch.setattr(supervisor, "DOCUMENTED_INTERPRETER", tmp_path / "missing-python")
    with pytest.raises(supervisor.StageBCampaignSupervisorError, match="wrong interpreter"):
        supervisor.preflight_gpu(repository_root=root)
    with pytest.raises(supervisor.StageBCampaignSupervisorError, match="inside repository_root"):
        supervisor.supervise_campaign(campaign_path, campaign_sha, tmp_path / "outside", repository_root=root, status=True)


def test_preflight_binds_virtual_environment_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    environment = root / ".venv"
    interpreter = environment / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("")
    monkeypatch.setattr(supervisor, "DOCUMENTED_INTERPRETER", tmp_path / "missing-python")
    monkeypatch.setattr(supervisor.sys, "prefix", str(environment))
    assert supervisor._sanctioned_environments(root) == (environment.resolve(),)


def test_rejects_partial_or_corrupt_existing_batch_without_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, campaign_sha, campaign_path = _campaign_fixture(tmp_path)
    output = root / "results"
    partial = output / "nap_lesion" / "batch-000"
    partial.mkdir(parents=True)
    (partial / "candidate-0000.trace.zip").write_bytes(b"partial")
    monkeypatch.setattr(supervisor, "run_gpu_batch", lambda *args, **kwargs: pytest.fail("must not rerun partial output"))
    with pytest.raises(supervisor.StageBCampaignSupervisorError, match="partial batch output"):
        supervisor.supervise_campaign(campaign_path, campaign_sha, output, repository_root=root, status=True)


def test_rejects_state_campaign_identity_change(tmp_path: Path) -> None:
    root, campaign_sha, campaign_path = _campaign_fixture(tmp_path)
    output = root / "results"
    output.mkdir()
    body = {
        "schema": supervisor.SUPERVISOR_SCHEMA,
        "campaign": {"path": "other.json", "sha256": "0" * 64},
        "output_root": "results",
        "environment": None,
        "status": "ready",
        "completed": [],
        "in_flight": None,
        "triage": None,
    }
    _write(output / supervisor.STATE_FILENAME, {**body, "sha256": _digest(body)})
    with pytest.raises(supervisor.StageBCampaignSupervisorError, match="different campaign"):
        supervisor.supervise_campaign(campaign_path, campaign_sha, output, repository_root=root, status=True)


def test_complete_campaign_invokes_strict_triage_and_records_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, campaign_sha, campaign_path = _campaign_fixture(tmp_path)
    output = root / "results"
    for arm in supervisor.ARM_ORDER:
        _fake_receipt(root, output, campaign_sha, campaign_path, arm)
    monkeypatch.setattr(
        supervisor,
        "preflight_gpu",
        lambda **kwargs: {"interpreter": "test", "cupy_version": "test"},
    )
    triage_body = {
        "schema": "test-triage",
        "process_status": "completed",
        "engineering_screening_only": True,
        "scientific_verdict": None,
    }
    triage = {**triage_body, "sha256": _digest(triage_body)}
    calls: list[tuple[Path, Path]] = []

    def fake_triage(campaign, digest, results, *, repository_root):
        calls.append((Path(campaign), Path(results)))
        return triage

    monkeypatch.setattr(supervisor, "triage_gpu_campaign", fake_triage)
    result = supervisor.supervise_campaign(campaign_path, campaign_sha, output, repository_root=root)
    assert result["executed"] is False
    assert calls == [(campaign_path, output)]
    state = json.loads((output / supervisor.STATE_FILENAME).read_text())
    assert state["status"] == "complete"
    assert state["triage"]["path"] == "results/triage.json"
    assert json.loads((output / "triage.json").read_text()) == triage


def test_allows_authenticated_partial_progress_in_a_multi_batch_arm(tmp_path: Path) -> None:
    root, campaign_sha, campaign_path = _campaign_fixture(tmp_path, batches_per_arm=2)
    output = root / "results"
    _fake_receipt(root, output, campaign_sha, campaign_path, "nap_lesion", batch_index=0)
    result = supervisor.supervise_campaign(
        campaign_path, campaign_sha, output, repository_root=root, status=True
    )
    assert result["completed_count"] == 1
    assert result["next"] == {"arm": "nap_lesion", "batch_index": 1}


def test_accepts_authenticated_receipt_superset_from_another_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, campaign_sha, campaign_path = _campaign_fixture(tmp_path)
    output = root / "results"
    _fake_receipt(root, output, campaign_sha, campaign_path, "nap_lesion")
    monkeypatch.setattr(
        supervisor,
        "preflight_gpu",
        lambda **kwargs: {"interpreter": "test", "cupy_version": "test"},
    )

    def fake_run(campaign, digest, arm, index, output_dir, **kwargs):
        _fake_receipt(root, output, digest, campaign_path, arm, index)
        return {"ok": True}

    monkeypatch.setattr(supervisor, "run_gpu_batch", fake_run)
    supervisor.supervise_campaign(campaign_path, campaign_sha, output, repository_root=root)
    _fake_receipt(root, output, campaign_sha, campaign_path, "cav2_2_lesion")
    result = supervisor.supervise_campaign(
        campaign_path, campaign_sha, output, repository_root=root, status=True
    )
    assert result["completed_count"] == 3
    assert result["next"] == {"arm": "sk_lesion", "batch_index": 0}
