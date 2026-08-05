"""Tests for exact GPU-survivor to NumPy confirmation handoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tools.v14_stageB_confirmation as confirmation
from sim.snr_executable_packet import canonical_bytes


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "research/specs/v14_snr_stageB_sobol_candidates.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_versioned_contract_fixture(root: Path) -> tuple[Path, Path]:
    causal_gate = root / "research/specs/custom-causal-gate.json"
    analysis_protocol = root / "research/specs/custom-analysis-protocol.json"
    analysis_protocol.write_bytes(canonical_bytes({"schema": "test-analysis-protocol"}))
    causal_gate.write_bytes(canonical_bytes({
        "schema": "test-causal-gate",
        "authorized_analysis_protocol": {
            "path": "research/specs/custom-analysis-protocol.json",
            "sha256": _sha(analysis_protocol),
        },
    }))
    return causal_gate, analysis_protocol


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repository"
    specs = root / "research/specs"
    runtime = root / "research/experiment-runtime"
    specs.mkdir(parents=True)
    runtime.mkdir(parents=True)
    candidates_path = specs / CANDIDATES.name
    candidates_path.write_bytes(CANDIDATES.read_bytes())
    candidates = json.loads(candidates_path.read_bytes())
    rows = []
    counts = {"engineering_fail": 510, "engineering_pass": 2}
    for row in candidates["candidates"]:
        candidate = row["candidate"]
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "candidate_sha256": row["candidate_sha256"],
            "classification": (
                "engineering_pass" if row["point_index"] in {284, 404}
                else "engineering_fail"
            ),
        })
    body = {
        "schema": "v14-snr-stageB-gpu-triage-v1",
        "process_status": "completed",
        "engineering_screening_only": True,
        "scientific_verdict": None,
        "source_equivalence_claimed": False,
        "numpy_confirmation_required": True,
        "campaign": {"path": "runtime/campaign.json", "sha256": "0" * 64},
        "candidate_count": 512,
        "classification_counts": counts,
        "candidates": rows,
    }
    triage = {**body, "sha256": confirmation._digest(body)}
    triage_path = runtime / "triage.json"
    triage_path.write_bytes(canonical_bytes(triage))
    return root, candidates_path, triage_path


def test_screen_freezes_exact_two_engineering_passes(tmp_path: Path) -> None:
    root, candidates, triage = _fixture(tmp_path)
    manifest = confirmation.build_confirmation_manifest(
        candidates, _sha(candidates), triage, _sha(triage), repository_root=root,
    )
    assert manifest["selected_count"] == 2
    assert manifest["selection_rule"] == "all_and_only_engineering_pass_candidates"
    assert [item["point_index"] for item in manifest["selected_candidates"]] == [284, 404]
    assert manifest["backend"] == "numpy"
    assert manifest["authoritative_source_required"] is True
    assert manifest["sha256"] == confirmation._digest({
        key: value for key, value in manifest.items() if key != "sha256"
    })


def test_tampered_triage_selection_is_rejected(tmp_path: Path) -> None:
    root, candidates, triage_path = _fixture(tmp_path)
    triage = json.loads(triage_path.read_bytes())
    selected = next(
        item for item in triage["candidates"]
        if item["classification"] == "engineering_fail"
    )
    selected["classification"] = "engineering_pass"
    body = {key: value for key, value in triage.items() if key != "sha256"}
    triage["sha256"] = confirmation._digest(body)
    triage["classification_counts"] = {"engineering_fail": 509, "engineering_pass": 3}
    triage_path.write_bytes(canonical_bytes(triage))
    with pytest.raises(confirmation.StageBConfirmationError, match="self digest"):
        confirmation.build_confirmation_manifest(
            candidates, _sha(candidates), triage_path, _sha(triage_path), repository_root=root,
        )


def test_manifest_writer_is_write_once_and_repository_scoped(tmp_path: Path) -> None:
    root, candidates, triage = _fixture(tmp_path)
    manifest = confirmation.build_confirmation_manifest(
        candidates, _sha(candidates), triage, _sha(triage), repository_root=root,
    )
    output = root / "research/experiment-runtime/confirmation.json"
    try:
        assert confirmation.write_manifest(manifest, output, repository_root=root) == output
        with pytest.raises(confirmation.StageBConfirmationError, match="refusing to replace"):
            confirmation.write_manifest(manifest, output, repository_root=root)
        with pytest.raises(confirmation.StageBConfirmationError, match="inside repository_root"):
            confirmation.write_manifest(manifest, tmp_path / "outside.json", repository_root=root)
    finally:
        output.unlink(missing_ok=True)


def test_run_binds_selected_candidate_and_requires_authoritative_source(
    tmp_path: Path, monkeypatch,
) -> None:
    root, candidates, triage = _fixture(tmp_path)
    for name in (
        "v14_snr_stageB_packet_template.json",
        "v14_snr_stageB_causal_gates.json",
        "v14_snr_stageB_intrinsic_protocol.json",
    ):
        target = root / "research/specs" / name
        target.write_bytes((ROOT / "research/specs" / name).read_bytes())
    manifest = confirmation.build_confirmation_manifest(
        candidates, _sha(candidates), triage, _sha(triage), repository_root=root,
    )
    manifest_path = root / "research/specs/confirmation.json"
    confirmation.write_manifest(manifest, manifest_path, repository_root=root)
    selected = manifest["selected_candidates"][0]
    assignments = [
        {
            "candidate_id": item["candidate_id"],
            "primary_host": "primary-host",
            "recovery_host": "recovery-host",
        }
        for item in manifest["selected_candidates"]
    ]
    plan = confirmation.build_job_plan(
        manifest_path, _sha(manifest_path), "a" * 40, "b" * 64,
        assignments, repository_root=root,
    )
    assert plan["schema"] == "v14-snr-stageB-numpy-confirmation-job-plan-v1"
    assert "causal_gate" not in plan
    assert "analysis_protocol" not in plan
    plan_path = root / "runtime/job-plan.json"
    confirmation.write_job_plan(plan, plan_path, repository_root=root)
    observed = {}

    def fake_run(*args, **kwargs):
        observed["causal_gate_path"] = args[4]
        observed["causal_gate_sha256"] = args[5]
        observed.update(kwargs)
        output = Path(args[6])
        output.mkdir(parents=True)
        scorer_input = {"schema": "synthetic-scorer-input"}
        stored_score = {"schema": "synthetic-score", "readiness_contract_result": "UNAVAILABLE"}
        scorer_input_path = output / "scorer-input.json"
        score_path = output / "score.json"
        scorer_input_path.write_bytes(canonical_bytes(scorer_input))
        score_path.write_bytes(canonical_bytes(stored_score))
        inner = {
            "backend": "numpy",
            "device": "cpu",
            "scientific_verdict": None,
            "provenance": {
                "source_identity": {
                    "kind": "git_archive",
                    "revision": "a" * 40,
                    "source_manifest_sha256": "b" * 64,
                    "source_ancestry_sha256": "c" * 64,
                    "authoritative": True,
                },
                "source_verified_at_start": True,
                "source_verified_at_exit": True,
            },
            "candidate": {
                "candidate_id": selected["candidate_id"],
                "sha256": selected["candidate_sha256"],
            },
            "scorer_input": {
                "path": scorer_input_path.relative_to(root).as_posix(),
                "sha256": _sha(scorer_input_path),
            },
            "score": {
                "path": score_path.relative_to(root).as_posix(),
                "sha256": _sha(score_path),
                "readiness_contract_result": "UNAVAILABLE",
            },
        }
        (output / "readiness-receipt.json").write_bytes(canonical_bytes(inner))
        return inner

    monkeypatch.setattr(confirmation, "run_intrinsic_readiness", fake_run)
    monkeypatch.setattr(
        confirmation, "_runtime_environment", lambda: dict(confirmation.EXPECTED_ENVIRONMENT)
    )
    monkeypatch.setattr(confirmation.socket, "gethostname", lambda: "primary-host")
    receipt = confirmation.run_confirmation_candidate(
        manifest_path, _sha(manifest_path), selected["candidate_id"],
        root / "runtime/result", repository_root=root,
        execution_argv=["test-confirmation"],
        expected_source_revision="a" * 40,
        expected_source_manifest_sha256="b" * 64,
        job_plan_path=plan_path,
        job_plan_sha256=_sha(plan_path),
        job_id=f"confirm-{selected['point_index']}",
    )
    assert observed["require_authoritative_source"] is True
    assert Path(observed["causal_gate_path"]) == (
        root / "research/specs/v14_snr_stageB_causal_gates.json"
    )
    assert observed["causal_gate_sha256"] == _sha(
        root / "research/specs/v14_snr_stageB_causal_gates.json"
    )
    assert Path(observed["analysis_protocol_path"]) == (
        root / "research/specs/v14_snr_stageB_intrinsic_protocol.json"
    )
    assert observed["analysis_protocol_sha256"] == _sha(
        root / "research/specs/v14_snr_stageB_intrinsic_protocol.json"
    )
    assert receipt["candidate"]["candidate_sha256"] == selected["candidate_sha256"]
    assert receipt["source_identity"]["authoritative"] is True
    assert receipt["environment"] == confirmation.EXPECTED_ENVIRONMENT
    assert receipt["job_plan"]["execution_host"] == "primary-host"
    receipt_path = root / "runtime/result/confirmation-receipt.json"
    monkeypatch.setattr(
        confirmation, "score_intrinsic_lesion_observations",
        lambda document, root: {"schema": "synthetic-score", "readiness_contract_result": "UNAVAILABLE"},
    )
    verified = confirmation.verify_collected_confirmation(
        receipt_path, _sha(receipt_path), repository_root=root,
    )
    assert verified["verified"] is True
    (root / "runtime/result/score.json").write_text("corrupt", encoding="ascii")
    with pytest.raises(confirmation.StageBConfirmationError, match="corrupt"):
        confirmation.verify_collected_confirmation(
            receipt_path, _sha(receipt_path), repository_root=root,
        )


def test_recomputed_score_comparison_allows_only_machine_scale_float_drift() -> None:
    stored = {
        "passed": True,
        "values": [0.0032847258422412456, -78.3174934387207],
        "status": "scored",
    }
    replayed = {
        "passed": True,
        "values": [0.0032847258422412443, -78.31749343872069],
        "status": "scored",
    }
    assert confirmation._recomputed_score_matches(stored, replayed)
    assert not confirmation._recomputed_score_matches(
        stored, {**replayed, "values": [0.0033, -78.31749343872069]}
    )
    assert not confirmation._recomputed_score_matches(
        stored, {**replayed, "passed": 1}
    )


def test_job_plan_v2_binds_exact_causal_gate_and_analysis_protocol(
    tmp_path: Path,
) -> None:
    root, candidates, triage = _fixture(tmp_path)
    manifest = confirmation.build_confirmation_manifest(
        candidates, _sha(candidates), triage, _sha(triage), repository_root=root,
    )
    manifest_path = root / "research/specs/confirmation.json"
    confirmation.write_manifest(manifest, manifest_path, repository_root=root)
    causal_gate, analysis_protocol = _write_versioned_contract_fixture(root)
    assignments = [
        {
            "candidate_id": item["candidate_id"],
            "primary_host": "primary-host",
            "recovery_host": "recovery-host",
        }
        for item in manifest["selected_candidates"]
    ]

    plan = confirmation.build_job_plan(
        manifest_path,
        _sha(manifest_path),
        "a" * 40,
        "b" * 64,
        assignments,
        repository_root=root,
        causal_gate_path=causal_gate,
        causal_gate_sha256=_sha(causal_gate),
        analysis_protocol_path=analysis_protocol,
        analysis_protocol_sha256=_sha(analysis_protocol),
    )

    assert plan["schema"] == "v14-snr-stageB-numpy-confirmation-job-plan-v2"
    assert plan["contract"]["causal_gate"] == {
        "path": "research/specs/custom-causal-gate.json",
        "sha256": _sha(causal_gate),
    }
    assert plan["contract"]["analysis_protocol"] == {
        "path": "research/specs/custom-analysis-protocol.json",
        "sha256": _sha(analysis_protocol),
    }
    assert plan["sha256"] == confirmation._digest({
        key: value for key, value in plan.items() if key != "sha256"
    })


def test_v2_run_uses_bound_contract_files_and_rejects_digest_tampering(
    tmp_path: Path, monkeypatch,
) -> None:
    root, candidates, triage = _fixture(tmp_path)
    template = root / "research/specs/v14_snr_stageB_packet_template.json"
    template.write_bytes(
        (ROOT / "research/specs/v14_snr_stageB_packet_template.json").read_bytes()
    )
    causal_gate, analysis_protocol = _write_versioned_contract_fixture(root)
    manifest = confirmation.build_confirmation_manifest(
        candidates, _sha(candidates), triage, _sha(triage), repository_root=root,
    )
    manifest_path = root / "research/specs/confirmation.json"
    confirmation.write_manifest(manifest, manifest_path, repository_root=root)
    selected = manifest["selected_candidates"][0]
    assignments = [
        {
            "candidate_id": item["candidate_id"],
            "primary_host": "primary-host",
            "recovery_host": "recovery-host",
        }
        for item in manifest["selected_candidates"]
    ]
    plan = confirmation.build_job_plan(
        manifest_path,
        _sha(manifest_path),
        "a" * 40,
        "b" * 64,
        assignments,
        repository_root=root,
        causal_gate_path=causal_gate,
        causal_gate_sha256=_sha(causal_gate),
        analysis_protocol_path=analysis_protocol,
        analysis_protocol_sha256=_sha(analysis_protocol),
    )
    plan_path = root / "runtime/job-plan.json"
    confirmation.write_job_plan(plan, plan_path, repository_root=root)
    observed: dict[str, object] = {}

    def stop_after_binding(*args, **kwargs):
        observed["causal_gate_path"] = args[4]
        observed["causal_gate_sha256"] = args[5]
        observed.update(kwargs)
        raise RuntimeError("binding observed")

    monkeypatch.setattr(confirmation, "run_intrinsic_readiness", stop_after_binding)
    monkeypatch.setattr(
        confirmation, "_runtime_environment", lambda: dict(confirmation.EXPECTED_ENVIRONMENT)
    )
    monkeypatch.setattr(confirmation.socket, "gethostname", lambda: "primary-host")
    run_kwargs = {
        "repository_root": root,
        "execution_argv": ["test-confirmation"],
        "expected_source_revision": "a" * 40,
        "expected_source_manifest_sha256": "b" * 64,
        "job_plan_path": plan_path,
        "job_plan_sha256": _sha(plan_path),
        "job_id": f"confirm-{selected['point_index']}",
    }

    with pytest.raises(RuntimeError, match="binding observed"):
        confirmation.run_confirmation_candidate(
            manifest_path,
            _sha(manifest_path),
            selected["candidate_id"],
            root / "runtime/result",
            **run_kwargs,
        )
    assert Path(observed["causal_gate_path"]) == causal_gate
    assert observed["causal_gate_sha256"] == _sha(causal_gate)
    assert Path(observed["analysis_protocol_path"]) == analysis_protocol
    assert observed["analysis_protocol_sha256"] == _sha(analysis_protocol)

    causal_gate.write_bytes(canonical_bytes({"schema": "tampered-causal-gate"}))
    with pytest.raises(confirmation.StageBConfirmationError, match="digest"):
        confirmation.run_confirmation_candidate(
            manifest_path,
            _sha(manifest_path),
            selected["candidate_id"],
            root / "runtime/tampered-result",
            **run_kwargs,
        )
