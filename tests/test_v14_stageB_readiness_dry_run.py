import hashlib
import json
from pathlib import Path

from tools import v14_stageB_readiness_dry_run as readiness


ROOT = Path(__file__).resolve().parents[1]


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_two_candidate_readiness_dry_run_is_seed_free_isolated_and_exact(tmp_path: Path):
    report = readiness.run_readiness_dry_run(tmp_path / "artifacts", root=ROOT)

    assert report["process_status"] == "completed"
    assert report["exit_code"] == 0
    assert report["readiness_only"] == {
        "synthetic": True, "non_scientific": True, "reserved_seed_count": 0,
    }
    assert report["candidate_count"] == 2
    assert report["backend"] == "numpy"
    assert report["device"] == "cpu"
    assert report["provenance"]["seed"] is None
    assert report["backend_partition_pairs"] == [{"backend": "numpy", "partition": "readiness"}]
    assert report["seeds_selected"] is False
    assert report["held_out_partitions_accessed"] == []
    assert len(set(report["candidate_digests"])) == 2
    assert all(item["readiness_only"]["reserved_seed_count"] == 0 for item in report["results"])
    assert all(item["backend"] == "numpy" and item["device"] == "cpu"
               and item["partition"] == "readiness" and item["seed"] is None
               for item in report["results"])
    assert "590297" not in json.dumps(report, sort_keys=True)

    results = {item["adaptive_candidate"]["candidate_id"]: item for item in report["results"]}
    assert set(results) == {"readiness-synthetic-in-band", "readiness-synthetic-no-go"}
    assert results["readiness-synthetic-in-band"]["scientific_verdict"] == "GO"
    no_go = results["readiness-synthetic-no-go"]
    assert no_go["process_status"] == "completed"
    assert no_go["exit_code"] == 0
    assert no_go["scientific_verdict"] == "NO_GO"
    assert no_go["verdict_semantics"] == "synthetic readiness transport only; not a physiology claim"

    for result in report["results"]:
        candidate_path = tmp_path / "artifacts" / result["candidate_artifact"]["path"]
        raw_path = tmp_path / "artifacts" / result["raw_observation_artifact"]["path"]
        assert candidate_path.parent == raw_path.parent
        candidate = _load(candidate_path)
        raw = _load(raw_path)
        command = _load(candidate_path.with_name("candidate.json.prov.json"))
        assert _digest(candidate) == result["candidate_artifact"]["sha256"]
        assert candidate["candidate_id"] == result["adaptive_candidate"]["candidate_id"]
        assert _digest(candidate) == result["adaptive_candidate"]["candidate_sha256"]
        assert raw["adaptive_candidate"] == result["adaptive_candidate"]
        assert raw["readiness_only"]["synthetic"] is True
        assert raw["readiness_only"]["non_scientific"] is True
        assert raw["readiness_only"]["reserved_seed_count"] == 0
        assert raw["backend"] == "numpy" and raw["device"] == "cpu"
        assert raw["provenance"]["seed"] is None
        assert command["runner"] == "tools/v14_stageB_readiness_dry_run.py"
        assert command["seed"] is None
        assert _digest(raw) == result["raw_observation_artifact"]["sha256"]


def test_candidate_mixing_is_detected_as_an_infrastructure_failure(tmp_path: Path, monkeypatch):
    original = readiness.score_raw_observations
    calls = 0

    def mixed_score(document, *, root):
        nonlocal calls
        score = original(document, root=root)
        calls += 1
        if calls == 2:
            score["adaptive_candidate"] = report_first_echo
        return score

    report_first_echo = readiness._candidate_echo(
        readiness._candidate_document("other", {"readiness_trace_profile": "other"})
    )
    monkeypatch.setattr(readiness, "score_raw_observations", mixed_score)
    report = readiness.run_readiness_dry_run(tmp_path / "artifacts", root=ROOT)

    assert report["process_status"] == "failed"
    assert report["exit_code"] == 1
    assert "scientific_verdict" not in report
    assert "candidate echo does not match" in report["infrastructure_error"]


def test_scorer_failure_is_infrastructure_failure_without_a_verdict(tmp_path: Path, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise readiness.StageBScorerError("fixture transport is unavailable")

    monkeypatch.setattr(readiness, "score_raw_observations", unavailable)
    report = readiness.run_readiness_dry_run(tmp_path / "artifacts", root=ROOT)

    assert report["process_status"] == "failed"
    assert report["exit_code"] == 1
    assert "scientific_verdict" not in report
    assert report["completed_candidates"] == []
    assert "fixture transport is unavailable" in report["infrastructure_error"]


def test_cli_uses_process_semantic_exit_codes(tmp_path: Path, monkeypatch):
    assert readiness.main(["--output-dir", str(tmp_path / "success"), "--root", str(ROOT)]) == 0

    monkeypatch.setattr(
        readiness,
        "score_raw_observations",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(readiness.StageBScorerError("broken")),
    )
    assert readiness.main(["--output-dir", str(tmp_path / "failure"), "--root", str(ROOT)]) == 1
