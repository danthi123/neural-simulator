"""Focused tests for the resolved V14 Stage B screen aggregator."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import tools.v14_stageB_screen_aggregator as aggregator
from sim.snr_executable_packet import canonical_bytes
from tools.v14_stageB_screen_aggregator import (
    REQUIRED_METRICS,
    RESULT_SCHEMA,
    StageBScreenAggregatorError,
    aggregate_stageB_screen,
    main,
)


ROOT = Path(__file__).resolve().parents[1]
CAUSAL_SOURCE = ROOT / "research/specs/v14_snr_stageB_causal_gates.json"
REAL_REGENERATE = aggregator._regenerate_candidate_manifest


@pytest.fixture(autouse=True)
def _isolate_structural_aggregator_tests(monkeypatch):
    # Candidate regeneration and raw-trace rescoring have their own focused
    # suites. These fixtures exercise the aggregator's join/classification.
    monkeypatch.setattr(aggregator, "_regenerate_candidate_manifest", lambda _root, value: value)
    monkeypatch.setattr(aggregator, "_recompute_scorer", lambda _root, value: value)


def _write_json(path: Path, value: dict) -> str:
    path.write_bytes(canonical_bytes(value) + b"\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(candidate_id: str, value: float) -> dict:
    return {
        "schema": "sim-adaptive-candidate-v1",
        "candidate_id": candidate_id,
        "parameters": {"g_nap": value, "g_sk": value + 1.0},
    }


def _fixture(tmp_path: Path, count: int = 1):
    causal_path = tmp_path / "causal-gates.json"
    causal_path.write_bytes(CAUSAL_SOURCE.read_bytes())
    causal_sha = hashlib.sha256(causal_path.read_bytes()).hexdigest()
    causal = json.loads(causal_path.read_text(encoding="utf-8"))

    rows = []
    candidates = []
    for index in range(count):
        candidate = _candidate(f"candidate-{index}", float(index + 1))
        candidate_sha = hashlib.sha256(canonical_bytes(candidate)).hexdigest()
        rows.append({"point_index": index, "candidate_sha256": candidate_sha, "candidate": candidate})
        candidates.append((candidate, candidate_sha))
    manifest_body = {
        "schema": "v14-snr-stageB-sobol-candidate-manifest-v1",
        "status": "preregistered-seed-free-candidate-generation",
        "device": "not_applicable_non_executed_candidate_design",
        "provenance_exempt": "deterministic non-executed Sobol candidate design; contains no measured result",
        "template": {},
        "design": {"scientific_seed": None},
        "search_space": {},
        "candidates": rows,
    }
    manifest = {**manifest_body, "sha256": hashlib.sha256(canonical_bytes(manifest_body)).hexdigest()}
    manifest_path = tmp_path / "candidate-manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)

    def score(candidate: dict, candidate_sha: str, outcomes: dict[tuple[str, str], bool | None] | None = None) -> dict:
        outcomes = outcomes or {}
        results = []
        for gate in causal["causal_gates"]:
            if gate["id"] not in {
                "nap-complete-lesion", "cav2.2-complete-lesion", "sk-complete-lesion", "hcn-complete-lesion"
            }:
                continue
            hard_results = []
            for contract in gate["hard_gates"]:
                key = (gate["id"], contract["metric"])
                passed = outcomes.get(key)
                hard = {
                    **{field: contract[field] for field in ("metric", "operator", "evidence_class", "value", "window_s", "cohort_n") if field in contract},
                    "source_equivalence_claimed": False,
                    "status": "scored" if passed is not None else "unavailable",
                    "passed": passed,
                }
                if passed is None:
                    hard["reason"] = "operational timeout or missing contract"
                else:
                    hard["observed"] = 1.0
                hard_results.append(hard)
            gate_passed = (
                False if any(item["passed"] is False for item in hard_results)
                else None if any(item["passed"] is None for item in hard_results)
                else True
            )
            results.append({
                "gate_id": gate["id"],
                "source": gate["source"],
                "preparation": gate["preparation"],
                "passed": gate_passed,
                "hard_gates": hard_results,
            })
        all_passed = False if any(item["passed"] is False for item in results) else None if any(item["passed"] is None for item in results) else True
        return {
            "schema": "v14-snr-stageB-intrinsic-lesion-score-v1",
            "process_status": "completed",
            "scientific_verdict": None,
            "readiness_only": {"enabled": True, "reserved_seed_count": 0, "scientific_seed": None},
            "adaptive_candidate": {
                "candidate_id": candidate["candidate_id"],
                "candidate_sha256": candidate_sha,
                "effective_parameters": candidate["parameters"],
            },
            "causal_gate_packet": {"path": "causal-gates.json", "sha256": causal_sha},
            "runner_observations": {},
            "all_intrinsic_lesion_gates_passed": all_passed,
            "readiness_contract_result": "FAIL" if all_passed is False else "UNAVAILABLE" if all_passed is None else "PASS",
            "source_equivalence_claimed": False,
            "results": results,
        }

    return manifest_path, manifest_sha, candidates, score


def _run(tmp_path: Path, outcomes=None):
    manifest_path, manifest_sha, candidates, score = _fixture(tmp_path)
    score_path = tmp_path / "score-0.json"
    score_sha = _write_json(score_path, score(*candidates[0], outcomes=outcomes))
    return aggregate_stageB_screen(
        {"path": manifest_path.name, "sha256": manifest_sha},
        [{"path": score_path.name, "sha256": score_sha}],
        root=tmp_path,
    )


def test_all_five_resolved_subgates_true_is_screen_pass_and_digest_is_deterministic(tmp_path: Path):
    outcomes = {key: True for key in REQUIRED_METRICS}
    # Build two independent identical roots so file paths and bytes are equal.
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    first = _run(left, outcomes)
    second = _run(right, outcomes)
    assert first["schema"] == RESULT_SCHEMA
    assert first == second
    assert first["candidates"][0]["classification"] == "screen_pass"
    assert first["sha256"] == hashlib.sha256(canonical_bytes({k: v for k, v in first.items() if k != "sha256"})).hexdigest()


def test_any_false_wins_over_unavailable_and_does_not_create_go(tmp_path: Path):
    outcomes = {key: True for key in REQUIRED_METRICS}
    outcomes[("cav2.2-complete-lesion", "isi_cv")] = False
    outcomes[("sk-complete-lesion", "medium_ahp_depth_mV")] = None
    result = _run(tmp_path, outcomes)
    candidate = result["candidates"][0]
    assert candidate["classification"] == "screen_fail"
    assert all("GO" not in str(value) for value in result.values())
    assert "scientific_verdict" in result and result["scientific_verdict"] is None


def test_required_unavailable_is_inconclusive_and_missing_contract_stays_listed(tmp_path: Path):
    outcomes = {key: True for key in REQUIRED_METRICS}
    outcomes[("hcn-complete-lesion", "lesion_spike_count")] = None
    result = _run(tmp_path, outcomes)
    candidate = result["candidates"][0]
    assert candidate["classification"] == "screen_inconclusive"
    assert any(item["metric"] == "lesion_spike_count" and item["passed"] is None for item in candidate["resolved_metrics"])
    assert any(item["metric"] == "medium_ahp_depth_mV" and item["status"] == "unavailable" for item in candidate["missing_contract_metrics"])


def test_duplicate_manifest_candidates_are_rejected(tmp_path: Path):
    manifest_path, _, candidates, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidates"].append(copy.deepcopy(manifest["candidates"][0]))
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    manifest["sha256"] = hashlib.sha256(canonical_bytes(body)).hexdigest()
    manifest_sha = _write_json(manifest_path, manifest)
    with pytest.raises(StageBScreenAggregatorError, match="duplicate"):
        aggregate_stageB_screen({"path": manifest_path.name, "sha256": manifest_sha}, [], root=tmp_path)


def test_unknown_extra_gate_is_rejected(tmp_path: Path):
    manifest_path, manifest_sha, candidates, score = _fixture(tmp_path)
    value = score(*candidates[0])
    value["results"].append(copy.deepcopy(value["results"][0]))
    path = tmp_path / "score.json"
    digest = _write_json(path, value)
    with pytest.raises(StageBScreenAggregatorError, match="extra gates"):
        aggregate_stageB_screen({"path": manifest_path.name, "sha256": manifest_sha}, [{"path": path.name, "sha256": digest}], root=tmp_path)


def test_candidate_parameter_mismatch_and_tampering_are_rejected(tmp_path: Path):
    manifest_path, manifest_sha, candidates, score = _fixture(tmp_path)
    value = score(*candidates[0])
    value["adaptive_candidate"]["effective_parameters"]["g_nap"] = 99.0
    path = tmp_path / "score.json"
    digest = _write_json(path, value)
    with pytest.raises(StageBScreenAggregatorError, match="parameter identity"):
        aggregate_stageB_screen({"path": manifest_path.name, "sha256": manifest_sha}, [{"path": path.name, "sha256": digest}], root=tmp_path)
    with pytest.raises(StageBScreenAggregatorError, match="digest does not match"):
        aggregate_stageB_screen({"path": manifest_path.name, "sha256": manifest_sha}, [{"path": path.name, "sha256": "0" * 64}], root=tmp_path)


def test_recomputed_score_mismatch_is_rejected(tmp_path: Path, monkeypatch):
    manifest_path, manifest_sha, candidates, score = _fixture(tmp_path)
    document = score(*candidates[0], outcomes={key: True for key in REQUIRED_METRICS})
    path = tmp_path / "score.json"
    digest = _write_json(path, document)
    recomputed = copy.deepcopy(document)
    recomputed["results"][0]["hard_gates"][0]["passed"] = False
    monkeypatch.setattr(aggregator, "_recompute_scorer", lambda _root, _value: recomputed)
    with pytest.raises(StageBScreenAggregatorError, match="does not equal the score recomputed"):
        aggregate_stageB_screen(
            {"path": manifest_path.name, "sha256": manifest_sha},
            [{"path": path.name, "sha256": digest}],
            root=tmp_path,
        )


def test_manifest_must_equal_exact_regenerated_sobol_design(tmp_path: Path, monkeypatch):
    specs = tmp_path / "research/specs"
    specs.mkdir(parents=True)
    for name in (
        "v14_snr_stageB_sobol_candidates.json",
        "v14_snr_stageB_packet_template.json",
    ):
        (specs / name).write_bytes((ROOT / "research/specs" / name).read_bytes())
    manifest_path = specs / "v14_snr_stageB_sobol_candidates.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["candidates"] = manifest["candidates"][:-1]
    manifest["design"]["exact_count"] -= 1
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    manifest["sha256"] = hashlib.sha256(canonical_bytes(body)).hexdigest()
    manifest_path.write_bytes(canonical_bytes(manifest))
    monkeypatch.setattr(aggregator, "_regenerate_candidate_manifest", REAL_REGENERATE)
    with pytest.raises(StageBScreenAggregatorError, match="exact regenerated Sobol design"):
        aggregator._validate_candidate_manifest(
            tmp_path,
            {
                "path": manifest_path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            },
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scientific_verdict", "NO_GO", "scientific verdict"),
        ("source_equivalence_claimed", True, "source-equivalence"),
    ],
)
def test_forbidden_scorer_claims_are_rejected(tmp_path: Path, field, value, message):
    manifest_path, manifest_sha, candidates, score = _fixture(tmp_path)
    document = score(*candidates[0])
    document[field] = value
    path = tmp_path / "score.json"
    digest = _write_json(path, document)
    with pytest.raises(StageBScreenAggregatorError, match=message):
        aggregate_stageB_screen({"path": manifest_path.name, "sha256": manifest_sha}, [{"path": path.name, "sha256": digest}], root=tmp_path)


def test_seed_and_held_out_data_are_rejected(tmp_path: Path):
    manifest_path, manifest_sha, candidates, score = _fixture(tmp_path)
    document = score(*candidates[0])
    document["runner_observations"] = {"held_out": [42]}
    path = tmp_path / "score.json"
    digest = _write_json(path, document)
    with pytest.raises(StageBScreenAggregatorError, match="held-out"):
        aggregate_stageB_screen({"path": manifest_path.name, "sha256": manifest_sha}, [{"path": path.name, "sha256": digest}], root=tmp_path)


def test_mixed_causal_gate_bindings_are_rejected(tmp_path: Path):
    manifest_path, manifest_sha, candidates, score = _fixture(tmp_path, count=2)
    second_causal = tmp_path / "causal-gates-copy.json"
    second_causal.write_bytes(CAUSAL_SOURCE.read_bytes())
    second_sha = hashlib.sha256(second_causal.read_bytes()).hexdigest()
    first_score = score(*candidates[0])
    second_score = score(*candidates[1])
    second_score["causal_gate_packet"] = {"path": second_causal.name, "sha256": second_sha}
    first_path = tmp_path / "score-0.json"
    second_path = tmp_path / "score-1.json"
    first_sha = _write_json(first_path, first_score)
    second_digest = _write_json(second_path, second_score)
    with pytest.raises(StageBScreenAggregatorError, match="causal-gate bindings"):
        aggregate_stageB_screen(
            {"path": manifest_path.name, "sha256": manifest_sha},
            [{"path": first_path.name, "sha256": first_sha}, {"path": second_path.name, "sha256": second_digest}],
            root=tmp_path,
        )


def test_cli_writes_once_to_new_repository_relative_output(tmp_path: Path):
    manifest_path, manifest_sha, candidates, score = _fixture(tmp_path)
    outcomes = {key: True for key in REQUIRED_METRICS}
    score_path = tmp_path / "score.json"
    score_sha = _write_json(score_path, score(*candidates[0], outcomes=outcomes))
    argv = [
        "--root", str(tmp_path),
        "--candidate-manifest", manifest_path.name,
        "--candidate-manifest-sha256", manifest_sha,
        "--scorer", score_path.name, score_sha,
        "--output", "aggregate.json",
    ]

    assert main(argv) == 0
    output = json.loads((tmp_path / "aggregate.json").read_text(encoding="ascii"))
    assert output["candidates"][0]["classification"] == "screen_pass"
    with pytest.raises(SystemExit):
        main(argv)
