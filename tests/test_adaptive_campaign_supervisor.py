from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import adaptive_campaign_supervisor as supervisor
from tools.experiment_observation import digest


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="ascii")


def _load_receipt(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def _plan() -> dict:
    body = {
        "schema": "sim-experiment-controller-dry-run-v1",
        "decision": "propose",
        "candidate_materialization": {
            "count": 1,
            "candidates": [{"candidate_id": "c1", "order": 0, "partition": "calibration"}],
        },
    }
    return {**body, "sha256": digest(body)}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    root = tmp_path / "repo"
    root.mkdir()
    design = root / "design.json"
    _write(design, {"schema": "fixture", "id": "d1"})
    monkeypatch.setattr(supervisor, "load_adaptive_design", lambda *args, **kwargs: {"id": "d1"})
    monkeypatch.setattr(supervisor, "build_dry_run_plan", lambda *args, **kwargs: _plan())

    def write_plan(plan, destination, **kwargs):
        _write(Path(destination), plan)
        return Path(destination)

    monkeypatch.setattr(supervisor, "write_dry_run_plan", write_plan)
    return {"root": root, "design": design, "campaign": root / "campaign"}


def _advance(fixture: dict, **kwargs):
    return supervisor.advance_campaign(
        fixture["design"], fixture["campaign"], repository_root=fixture["root"], **kwargs,
    )


def test_creates_one_self_digested_transition_per_invocation(repo: dict) -> None:
    first = _advance(repo)

    assert first["transition"] == "controller_plan_created"
    assert first["scientific_verdict"] is None
    assert first["held_out_partitions_accessed"] == []
    assert first["sha256"] == digest({key: value for key, value in first.items() if key != "sha256"})
    assert len(list((repo["campaign"] / "state").glob("*.json"))) == 1


def test_authorization_is_idempotent_until_seal_appears(
    repo: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _advance(repo)

    def materialize(plan, destination, **kwargs):
        _write(Path(destination), {"schema": "candidate-spec"})
        return Path(destination)

    monkeypatch.setattr(supervisor, "materialize_candidate_spec", materialize)
    second = _advance(repo)
    third = _advance(repo)
    repeated = _advance(repo)

    assert second["transition"] == "candidate_spec_created"
    assert third["transition"] == "seal_authorized"
    assert "tools/experiment.py seal" in third["authorized_command"]
    assert repeated == third
    assert len(list((repo["campaign"] / "state").glob("*.json"))) == 3


def test_rejects_held_out_candidate_partition(repo: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = _plan()
    bad["candidate_materialization"]["candidates"][0]["partition"] = "held_out"
    bad["sha256"] = digest({key: value for key, value in bad.items() if key != "sha256"})
    monkeypatch.setattr(supervisor, "build_dry_run_plan", lambda *args, **kwargs: bad)
    _advance(repo)

    def materialize(plan, destination, **kwargs):
        _write(Path(destination), {"schema": "candidate-spec"})
        return Path(destination)

    monkeypatch.setattr(supervisor, "materialize_candidate_spec", materialize)
    _advance(repo)
    _write(repo["campaign"] / "experiment-seal.json", {"seal": True})
    handoff_body = {
        "schema": "sim-experiment-controller-handoff-v1", "sealed": True,
        "held_out_partitions_accessed": [],
    }
    handoff = {**handoff_body, "sha256": digest(handoff_body)}
    monkeypatch.setattr(supervisor, "validate_experiment_handoff", lambda *args, **kwargs: handoff)
    _advance(repo)
    materialization_body = {"schema": "materialization", "sha256": "x"}
    monkeypatch.setattr(supervisor, "materialize_execution_manifest",
                        lambda *args, **kwargs: materialization_body)
    _advance(repo)

    with pytest.raises(supervisor.CampaignSupervisorError, match="non-held-out"):
        _advance(repo)


def test_tampered_state_chain_fails_closed(repo: dict) -> None:
    _advance(repo)
    state_path = repo["campaign"] / "state/000001.json"
    state = json.loads(state_path.read_text(encoding="ascii"))
    state["requirements"].append("tampered")
    state_path.chmod(0o644)
    _write(state_path, state)

    with pytest.raises(supervisor.CampaignSupervisorError, match="state chain is invalid"):
        _advance(repo)


def test_running_job_emits_recovery_not_execution(
    repo: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = repo["campaign"]
    campaign.mkdir()
    for name in ("controller-plan.json", "candidate-spec.json", "experiment-seal.json",
                 "sealed-handoff.json", "materialization.json"):
        _write(campaign / name, _plan() if name == "controller-plan.json" else {"name": name})
    (campaign / "sealed-jobs").mkdir()
    _write(campaign / "sealed-jobs/plan.json", {"job_ids": ["job-a"]})
    state = campaign / "executor-state"
    _write(state / "manifest.json", {
        "jobs": [{"job_id": "job-a"}], "sha256": "manifest",
    })
    _write(campaign / "executor-manifest.json", {
        "jobs": [{"job_id": "job-a"}], "sha256": "manifest",
    })
    receipt_body = {"job_id": "job-a", "status": "running"}
    receipt = {**receipt_body, "sha256": digest(receipt_body)}
    _write(state / "receipts/job-a.json", receipt)
    monkeypatch.setattr(
        supervisor, "validate_experiment_handoff",
        lambda *args, **kwargs: {"name": "sealed-handoff.json"},
    )
    monkeypatch.setattr(
        supervisor, "materialize_execution_manifest",
        lambda *args, **kwargs: {"name": "materialization.json"},
    )
    monkeypatch.setattr(supervisor, "_validate_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        supervisor, "_read_receipt",
        lambda state, job_id, plan: _load_receipt(state / "receipts" / f"{job_id}.json"),
    )

    result = _advance(repo)

    assert result["transition"] == "recovery_check_authorized"
    assert "experiment_executor.py recover" in result["authorized_command"]
    assert result["execution_performed"] is False


def test_missing_observation_contract_stops_after_successful_jobs(
    repo: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = repo["campaign"]
    campaign.mkdir()
    for name in ("controller-plan.json", "candidate-spec.json", "experiment-seal.json",
                 "sealed-handoff.json", "materialization.json"):
        _write(campaign / name, _plan() if name == "controller-plan.json" else {"name": name})
    (campaign / "sealed-jobs").mkdir()
    _write(campaign / "sealed-jobs/plan.json", {"job_ids": ["job-a"]})
    state = campaign / "executor-state"
    _write(state / "manifest.json", {"jobs": [{"job_id": "job-a"}], "sha256": "manifest"})
    _write(campaign / "executor-manifest.json", {
        "jobs": [{"job_id": "job-a"}], "sha256": "manifest",
    })
    receipt_body = {"job_id": "job-a", "status": "succeeded"}
    _write(state / "receipts/job-a.json", {**receipt_body, "sha256": digest(receipt_body)})
    monkeypatch.setattr(
        supervisor, "validate_experiment_handoff",
        lambda *args, **kwargs: {"name": "sealed-handoff.json"},
    )
    monkeypatch.setattr(
        supervisor, "materialize_execution_manifest",
        lambda *args, **kwargs: {"name": "materialization.json"},
    )
    monkeypatch.setattr(supervisor, "_validate_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        supervisor, "_read_receipt",
        lambda state, job_id, plan: _load_receipt(state / "receipts" / f"{job_id}.json"),
    )

    result = _advance(repo)

    assert result["transition"] == "observation_contract_required"
    assert result["authorized_command"] is None
    assert result["scientific_verdict"] is None


def test_changed_controller_plan_fails_before_next_transition(repo: dict) -> None:
    _advance(repo)
    path = repo["campaign"] / "controller-plan.json"
    plan = json.loads(path.read_text(encoding="ascii"))
    plan["candidate_materialization"]["candidates"][0]["partition"] = "other"
    plan["sha256"] = digest({key: value for key, value in plan.items() if key != "sha256"})
    path.chmod(0o644)
    _write(path, plan)

    with pytest.raises(supervisor.CampaignSupervisorError, match="deterministic design proposal"):
        _advance(repo)


def test_invalid_executor_receipt_cannot_advance(
    repo: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = repo["campaign"]
    campaign.mkdir()
    for name in ("controller-plan.json", "candidate-spec.json", "experiment-seal.json",
                 "sealed-handoff.json", "materialization.json"):
        _write(campaign / name, _plan() if name == "controller-plan.json" else {"name": name})
    (campaign / "sealed-jobs").mkdir()
    _write(campaign / "sealed-jobs/plan.json", {"job_ids": ["job-a"]})
    state = campaign / "executor-state"
    executor = {"jobs": [{"job_id": "job-a"}], "sha256": "manifest"}
    _write(state / "manifest.json", executor)
    _write(campaign / "executor-manifest.json", executor)
    _write(state / "receipts/job-a.json", {"job_id": "job-a", "status": "succeeded",
                                            "sha256": "tampered"})
    monkeypatch.setattr(supervisor, "validate_experiment_handoff", lambda *args, **kwargs: {"name": "sealed-handoff.json"})
    monkeypatch.setattr(supervisor, "materialize_execution_manifest", lambda *args, **kwargs: {"name": "materialization.json"})
    monkeypatch.setattr(supervisor, "_validate_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        supervisor, "_read_receipt",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("receipt digest mismatch")),
    )

    with pytest.raises(supervisor.CampaignSupervisorError, match="receipt is invalid"):
        _advance(repo)
