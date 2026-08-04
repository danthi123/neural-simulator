"""Focused guards for deterministic, sealed experiment job planning."""

import hashlib
import json
import subprocess

import pytest

from tools.experiment import (
    HarnessError,
    create_experiment_seal,
    expand_experiment_jobs,
    load_experiment_spec,
    write_experiment_plan,
)


def _canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _base_spec():
    return {
        "schema": "sim-experiment-spec-v0",
        "id": "planner-test",
        "status": "preregistered",
        "partitions": {"held_out": [21], "calibration": [12, 11]},
        "backends": ["numpy", "cupy"],
        "execution": {
            "command": [
                ".venv/bin/python", "-u", "-m", "research.runners.example",
                "--seed", "{seed}", "--phase", "{partition}", "--arm", "{arm}",
                "--out", "{output}",
            ],
            "output": "research/findings/raw/planner/{partition}/{backend}/{arm}-{seed}.json",
            "arms": ["treatment", "control"],
            "targets": {
                "numpy": {"device": "cpu", "lane": "pool"},
                "cupy": {"device": "cuda:0", "lane": "gpu", "env": {"CUDA_VISIBLE_DEVICES": "0"}},
            },
            "corpus_reason": "new-config",
        },
    }


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def experiment_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "planner@example.invalid")
    _git(root, "config", "user.name", "Experiment Planner Test")
    (root / "research/specs").mkdir(parents=True)
    (root / "research/runners").mkdir(parents=True)
    (root / "research/runners/example.py").write_text("pass\n", encoding="utf-8")
    spec_path = root / "research/specs/test.json"
    spec_path.write_text(json.dumps(_base_spec(), indent=2), encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "test fixture")
    return root, spec_path, tmp_path


def test_repository_preregistration_loads_without_execution_contract():
    spec = load_experiment_spec("research/specs/v13_tonic_output_substrate.json")
    assert spec["partitions"]["calibration"] == [1013]
    assert spec["partitions"]["held_out"] == [1021]


def test_loader_refuses_seed_partition_overlap(experiment_repo):
    _, spec_path, _ = experiment_repo
    spec = json.loads(spec_path.read_text())
    spec["partitions"]["held_out"] = [12]
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(HarnessError, match="overlaps partitions"):
        load_experiment_spec(spec_path)


def test_matrix_expansion_is_deterministic_and_queue_compatible(experiment_repo):
    root, spec_path, _ = experiment_repo
    first = expand_experiment_jobs(spec_path, ["calibration"], root=root)
    second = expand_experiment_jobs(spec_path, ["calibration"], root=root)

    assert first == second
    assert len(first) == 8
    assert [(job["backend"], job["arm"], job["seed"]) for job in first[:4]] == [
        ("cupy", "control", 11),
        ("cupy", "control", 12),
        ("cupy", "treatment", 11),
        ("cupy", "treatment", 12),
    ]
    assert first[0]["sealed"] is False
    assert first[0]["lane"] == "gpu"
    assert "tools/queue_add.sh gpu" in first[0]["enqueue_command"]
    assert "SIM_BACKEND=cupy" in first[0]["command"]
    assert "CUDA_VISIBLE_DEVICES=0" in first[0]["command"]
    assert "mkdir research/findings/raw/planner/calibration/cupy/control-11.json.claim" in first[0]["command"]

    spec = json.loads(spec_path.read_text())
    spec["execution"]["targets"]["numpy"]["lane"] = "local"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    local = next(job for job in expand_experiment_jobs(spec_path, ["calibration"], root=root)
                 if job["backend"] == "numpy")
    assert local["lane"] == "local"
    assert local["enqueue_command"] is None


def test_missing_backend_device_declaration_is_refused(experiment_repo):
    root, spec_path, _ = experiment_repo
    spec = json.loads(spec_path.read_text())
    del spec["execution"]["targets"]["cupy"]
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(HarnessError, match="no execution target/device declaration"):
        expand_experiment_jobs(spec_path, ["calibration"], root=root)


def test_heldout_requires_matching_clean_seal(experiment_repo):
    root, spec_path, outside = experiment_repo
    with pytest.raises(HarnessError, match="held-out jobs are locked"):
        expand_experiment_jobs(spec_path, ["held_out"], root=root)

    seal_path = outside / "test.seal.json"
    seal = create_experiment_seal(spec_path, seal_path, root=root)
    jobs = expand_experiment_jobs(spec_path, ["held_out"], seal_path=seal_path, root=root)
    assert len(jobs) == 4
    assert all(job["sealed"] and job["source"] == seal["source"] for job in jobs)

    (root / "research/runners/example.py").write_text("CHANGED = True\n", encoding="utf-8")
    with pytest.raises(HarnessError, match="source changed or became dirty"):
        expand_experiment_jobs(spec_path, ["held_out"], seal_path=seal_path, root=root)


def test_prerequisites_and_stop_decisions_gate_downstream_jobs(experiment_repo):
    root, spec_path, _ = experiment_repo
    prerequisite = root / "evidence/calibration.json"
    prerequisite.parent.mkdir()
    prerequisite.write_text('{"status":"measured"}\n', encoding="utf-8")
    spec = json.loads(spec_path.read_text())
    spec["partitions"]["replication"] = [15]
    spec["prerequisites"] = [{
        "id": "calibration-artifact",
        "path": "evidence/calibration.json",
        "sha256": hashlib.sha256(prerequisite.read_bytes()).hexdigest(),
        "partitions": ["replication"],
    }]
    spec["stop_rules"] = [{
        "id": "calibration-decision",
        "blocks": ["replication"],
        "decision_file": "evidence/decision.json",
    }]
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(HarnessError, match="no valid decision record"):
        expand_experiment_jobs(spec_path, ["replication"], root=root)

    decision_path = root / "evidence/decision.json"
    decision = {
        "rule_id": "calibration-decision",
        "spec_sha256": _canonical_hash(spec),
        "decision": "stop",
    }
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    with pytest.raises(HarnessError, match="recorded STOP"):
        expand_experiment_jobs(spec_path, ["replication"], root=root)

    decision["decision"] = "continue"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    assert len(expand_experiment_jobs(spec_path, ["replication"], root=root)) == 4

    prerequisite.write_text("changed\n", encoding="utf-8")
    with pytest.raises(HarnessError, match="wrong sha256"):
        expand_experiment_jobs(spec_path, ["replication"], root=root)


def test_output_collisions_are_refused_before_dispatch(experiment_repo):
    root, spec_path, _ = experiment_repo
    spec = json.loads(spec_path.read_text())
    spec["execution"]["output"] = "research/findings/raw/planner/shared.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(HarnessError, match="more than one job"):
        expand_experiment_jobs(spec_path, ["calibration"], root=root)

    spec = _base_spec()
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = root / "research/findings/raw/planner/calibration/cupy/control-11.json"
    output.parent.mkdir(parents=True)
    output.write_text("old result\n", encoding="utf-8")
    with pytest.raises(HarnessError, match="mutable output collision"):
        expand_experiment_jobs(spec_path, ["calibration"], root=root)


def test_plan_files_are_fresh_and_read_only(experiment_repo):
    root, spec_path, outside = experiment_repo
    jobs = expand_experiment_jobs(spec_path, ["calibration"], root=root)
    plan_dir = outside / "plan"

    index = write_experiment_plan(jobs, plan_dir)

    assert index["count"] == 8
    assert len(list(plan_dir.glob("*.json"))) == 9
    assert len(list(plan_dir.glob("*.command"))) == 8
    assert all(path.stat().st_mode & 0o222 == 0 for path in plan_dir.iterdir())
    assert plan_dir.stat().st_mode & 0o222 == 0
    with pytest.raises(HarnessError, match="refusing to mutate existing plan directory"):
        write_experiment_plan(jobs, plan_dir)


def test_seal_file_itself_is_create_only(experiment_repo):
    root, spec_path, outside = experiment_repo
    seal_path = outside / "immutable.seal.json"
    create_experiment_seal(spec_path, seal_path, root=root)
    assert seal_path.stat().st_mode & 0o222 == 0
    with pytest.raises(HarnessError, match="refusing to replace immutable file"):
        create_experiment_seal(spec_path, seal_path, root=root)


def test_archive_seal_verifies_complete_source_manifest(tmp_path):
    root = tmp_path / "archive"
    spec_path = root / "research/specs/test.json"
    runner = root / "research/runners/example.py"
    runner.parent.mkdir(parents=True)
    spec_path.parent.mkdir(parents=True)
    runner.write_text("pass\n", encoding="utf-8")
    spec_path.write_text(json.dumps(_base_spec()), encoding="utf-8")
    runner_hash = hashlib.sha256(runner.read_bytes()).hexdigest()
    manifest = f"{runner_hash}  research/runners/example.py\n"
    (root / ".source_manifest.sha256").write_text(manifest, encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest.encode()).hexdigest()
    (root / ".source_revision").write_text(
        "git_sha=abcdef0123456789\n"
        "source_kind=git_archive\n"
        f"source_manifest_sha256={manifest_hash}\n",
        encoding="utf-8",
    )

    seal = create_experiment_seal(spec_path, tmp_path / "archive.seal.json", root=root)
    assert seal["source"]["kind"] == "git_archive"
    runner.write_text("TAMPERED = True\n", encoding="utf-8")
    with pytest.raises(HarnessError, match="source digest mismatch"):
        create_experiment_seal(spec_path, tmp_path / "tampered.seal.json", root=root)
