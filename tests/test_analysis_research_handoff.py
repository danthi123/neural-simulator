import hashlib
import json
from pathlib import Path

import pytest

from tools import analysis_research_handoff as handoff


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_digest(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": handoff._semantic_digest(body)}


def _fixture(tmp_path: Path) -> dict:
    root = tmp_path / "repo"
    root.mkdir()
    execution_spec_path = root / "spec.json"
    execution_spec_sha = _write_json(execution_spec_path, {"schema": "fixture-execution-v1"})
    analysis = _self_digest({
        "schema": "fixture-analysis-v1",
        "scientific_verdict": "STRUCTURAL_NO_GO",
        "execution_spec": {"path": "spec.json", "sha256": execution_spec_sha},
        "failed_metric_count": 2,
        "gates": [
            {"metric": "na.activation", "passed": False},
            {"metric": "na.recovery", "passed": True},
            {"metric": "kv3.deactivation", "passed": False},
        ],
    })
    analysis_path = root / "evidence/analysis.json"
    analysis_sha = _write_json(analysis_path, analysis)
    provenance = {
        "run_id": "fixture-analysis-run",
        "runner": "research/runners/fixture_analysis.py",
        "argv": ["fixture_analysis.py", "--out", "evidence/analysis.json"],
        "git_sha": "a" * 9,
        "git_dirty": False,
        "artifact": "evidence/analysis.json",
        "sim_backend_requested": "numpy",
        "sim_backend": "numpy",
    }
    provenance_path = root / "evidence/analysis.json.prov.json"
    provenance_sha = _write_json(provenance_path, provenance)
    implementation = {}
    for name in ("compiler", "research_escalation"):
        path = root / f"tools/{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="ascii")
        implementation[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    prior = []
    for index in range(2):
        path = root / f"findings/prior-{index}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"prior attempt {index}\n", encoding="ascii")
        prior.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "summary": f"Attempt {index} failed its declared current gate.",
        })
    contract = _self_digest({
        "schema": handoff.CONTRACT_SCHEMA,
        "status": "preregistered",
        "scientific_verdict": None,
        "source_claim_acceptance_allowed": False,
        "successor_dispatch_allowed": False,
        "implementation": implementation,
        "analysis": {
            "path": analysis_path.relative_to(root).as_posix(),
            "sha256": analysis_sha,
            "schema": "fixture-analysis-v1",
            "governing_binding": {
                "path": ["execution_spec"],
                "equals": {"path": "spec.json", "sha256": execution_spec_sha},
            },
            "provenance": {
                "path": provenance_path.relative_to(root).as_posix(),
                "sha256": provenance_sha,
                "runner": "research/runners/fixture_analysis.py",
                "backend": "numpy",
            },
        },
        "trigger": {
            "verdict_path": ["scientific_verdict"],
            "allowed_verdicts": ["STRUCTURAL_NO_GO"],
            "items_path": ["gates"],
            "item_id_field": "metric",
            "failed_field": "passed",
            "failed_value": False,
            "expected_failed_ids": ["na.activation", "kv3.deactivation"],
            "reject_unmapped": True,
        },
        "research_gate": {
            "slug": "fixture-current-wall",
            "title": "Fixture current wall",
            "blocked_experiment": "Two current families fail complete voltage-clamp transfer.",
            "wall_reason": "Gate-level constants do not reproduce composite current kinetics.",
            "query": "transient sodium Kv3 current state model",
            "output": "research/findings/fixture-research-gate.md",
        },
        "prior_attempts": prior,
        "questions": [
            {
                "id": "P1", "kind": "parameter",
                "text": "Which measured transition-rate equations reproduce the failed sodium activation current?",
                "trigger_ids": ["na.activation"],
            },
            {
                "id": "P2", "kind": "parameter",
                "text": "Which measured transition-rate equations reproduce the failed Kv3 deactivation current?",
                "trigger_ids": ["kv3.deactivation"],
            },
            {
                "id": "W1", "kind": "wiring",
                "text": "Which channel states and transitions are required to couple activation and deactivation?",
                "trigger_ids": ["na.activation", "kv3.deactivation"],
            },
        ],
        "receipt_output": "research/findings/raw/fixture-handoff.json",
    })
    contract_path = root / "research/specs/handoff.json"
    contract_sha = _write_json(contract_path, contract)
    return {
        "root": root,
        "analysis": analysis_path,
        "contract": contract_path,
        "contract_sha": contract_sha,
        "gate": root / contract["research_gate"]["output"],
        "receipt": root / contract["receipt_output"],
        "execution_spec": execution_spec_path,
    }


def _fake_start(captured: list, args, root: Path) -> Path:
    captured.append(args)
    path = root / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture research gate\n", encoding="ascii")
    return path


def _bind_authenticated_escalation_module(monkeypatch, fixture: dict) -> None:
    monkeypatch.setattr(
        handoff.research_escalation,
        "__file__",
        str(fixture["root"] / "tools/research_escalation.py"),
    )


def test_authenticated_failure_creates_fixed_questions_and_receipt(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    _bind_authenticated_escalation_module(monkeypatch, fixture)
    captured = []
    monkeypatch.setattr(
        handoff.research_escalation, "start",
        lambda args, root: _fake_start(captured, args, root),
    )

    receipt = handoff.compile_handoff(
        fixture["contract"], fixture["contract_sha"], fixture["receipt"],
        repository_root=fixture["root"],
    )

    assert receipt["failed_ids"] == ["kv3.deactivation", "na.activation"]
    assert receipt["selected_question_ids"] == ["P1", "P2", "W1"]
    assert receipt["source_claims_accepted"] is False
    assert receipt["successor_dispatched"] is False
    assert receipt["scientific_verdict"] is None
    assert receipt["sha256"] == handoff._semantic_digest(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    assert len(captured[0].parameter_question) == 2
    assert len(captured[0].wiring_question) == 1
    assert fixture["gate"].is_file() and fixture["receipt"].is_file()


def test_analysis_tamper_is_rejected_before_gate_creation(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    _bind_authenticated_escalation_module(monkeypatch, fixture)
    analysis = json.loads(fixture["analysis"].read_text())
    analysis["gates"][0]["passed"] = True
    _write_json(fixture["analysis"], analysis)
    monkeypatch.setattr(handoff.research_escalation, "start", lambda *_: pytest.fail("must not create gate"))

    with pytest.raises(handoff.AnalysisResearchHandoffError, match="digest"):
        handoff.compile_handoff(
            fixture["contract"], fixture["contract_sha"], fixture["receipt"],
            repository_root=fixture["root"],
        )


def test_unexpected_failure_set_is_not_mapped_post_hoc(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    _bind_authenticated_escalation_module(monkeypatch, fixture)
    analysis = json.loads(fixture["analysis"].read_text())
    analysis["gates"][1]["passed"] = False
    analysis["failed_metric_count"] = 3
    analysis = _self_digest(analysis)
    analysis_sha = _write_json(fixture["analysis"], analysis)
    contract = json.loads(fixture["contract"].read_text())
    contract["analysis"]["sha256"] = analysis_sha
    contract = _self_digest(contract)
    contract_sha = _write_json(fixture["contract"], contract)
    monkeypatch.setattr(handoff.research_escalation, "start", lambda *_: pytest.fail("must not create gate"))

    with pytest.raises(handoff.AnalysisResearchHandoffError, match="failed IDs differ"):
        handoff.compile_handoff(
            fixture["contract"], contract_sha, fixture["receipt"],
            repository_root=fixture["root"],
        )


def test_existing_gate_or_receipt_is_never_overwritten(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    _bind_authenticated_escalation_module(monkeypatch, fixture)
    fixture["gate"].parent.mkdir(parents=True, exist_ok=True)
    fixture["gate"].write_text("occupied\n")
    monkeypatch.setattr(handoff.research_escalation, "start", lambda *_: pytest.fail("must not overwrite"))

    with pytest.raises(handoff.AnalysisResearchHandoffError, match="existing partial research gate is invalid"):
        handoff.compile_handoff(
            fixture["contract"], fixture["contract_sha"], fixture["receipt"],
            repository_root=fixture["root"],
        )


def test_missing_governing_execution_spec_is_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    _bind_authenticated_escalation_module(monkeypatch, fixture)
    fixture["execution_spec"].unlink()
    monkeypatch.setattr(handoff.research_escalation, "start", lambda *_: pytest.fail("must not create gate"))

    with pytest.raises(handoff.AnalysisResearchHandoffError, match="governing execution spec"):
        handoff.compile_handoff(
            fixture["contract"], fixture["contract_sha"], fixture["receipt"],
            repository_root=fixture["root"],
        )


def test_receipt_failure_removes_gate_created_by_same_attempt(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    _bind_authenticated_escalation_module(monkeypatch, fixture)
    captured = []
    monkeypatch.setattr(
        handoff.research_escalation, "start",
        lambda args, root: _fake_start(captured, args, root),
    )
    monkeypatch.setattr(
        handoff,
        "_write_new",
        lambda *_: (_ for _ in ()).throw(OSError("injected receipt failure")),
    )

    with pytest.raises(OSError, match="injected receipt failure"):
        handoff.compile_handoff(
            fixture["contract"], fixture["contract_sha"], fixture["receipt"],
            repository_root=fixture["root"],
        )

    assert not fixture["gate"].exists()
    assert not fixture["receipt"].exists()


def test_nonsequential_question_identity_is_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    _bind_authenticated_escalation_module(monkeypatch, fixture)
    contract = json.loads(fixture["contract"].read_text())
    contract["questions"][0]["id"] = "sodium-question"
    contract = _self_digest(contract)
    contract_sha = _write_json(fixture["contract"], contract)
    monkeypatch.setattr(handoff.research_escalation, "start", lambda *_: pytest.fail("must not create gate"))

    with pytest.raises(handoff.AnalysisResearchHandoffError, match="sequential P/W identity"):
        handoff.compile_handoff(
            fixture["contract"], contract_sha, fixture["receipt"],
            repository_root=fixture["root"],
        )


def test_forged_failed_metric_count_is_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    _bind_authenticated_escalation_module(monkeypatch, fixture)
    analysis = json.loads(fixture["analysis"].read_text())
    analysis["failed_metric_count"] = 1
    analysis = _self_digest(analysis)
    analysis_sha = _write_json(fixture["analysis"], analysis)
    contract = json.loads(fixture["contract"].read_text())
    contract["analysis"]["sha256"] = analysis_sha
    contract = _self_digest(contract)
    contract_sha = _write_json(fixture["contract"], contract)
    monkeypatch.setattr(handoff.research_escalation, "start", lambda *_: pytest.fail("must not create gate"))

    with pytest.raises(handoff.AnalysisResearchHandoffError, match="failed metric count"):
        handoff.compile_handoff(
            fixture["contract"], contract_sha, fixture["receipt"],
            repository_root=fixture["root"],
        )


def test_tampered_analysis_provenance_is_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    _bind_authenticated_escalation_module(monkeypatch, fixture)
    provenance = fixture["root"] / "evidence/analysis.json.prov.json"
    value = json.loads(provenance.read_text())
    value["artifact"] = "evidence/substituted.json"
    _write_json(provenance, value)
    monkeypatch.setattr(handoff.research_escalation, "start", lambda *_: pytest.fail("must not create gate"))

    with pytest.raises(handoff.AnalysisResearchHandoffError, match="provenance digest"):
        handoff.compile_handoff(
            fixture["contract"], fixture["contract_sha"], fixture["receipt"],
            repository_root=fixture["root"],
        )
