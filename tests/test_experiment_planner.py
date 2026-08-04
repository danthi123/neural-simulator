"""Integration guards for digest-bound experiment planning and execution."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from tools.experiment import (
    HarnessError,
    _verify_runtime_snapshot,
    create_experiment_seal,
    execute_job_contract,
    expand_experiment_jobs,
    load_experiment_spec,
    write_experiment_plan,
)


QUERY = "Has this exact planner experiment already been run?"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
            "corpus_check": {
                "path": "research/queue/planner-corpus-check.json",
                "sha256": "pending",
                "query": QUERY,
                "max_age_seconds": 3600,
            },
            "claim_stale_seconds": 60,
        },
    }


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def _commit(root, message="fixture update"):
    _git(root, "add", ".")
    _git(root, "commit", "-qm", message)


def _write_corpus(root, spec_path, *, status="success", rag_status="success", age=0):
    spec = json.loads(spec_path.read_text())
    path = root / spec["execution"]["corpus_check"]["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "sim-corpus-check-v1",
        "experiment_id": spec["id"],
        "query": spec["execution"]["corpus_check"]["query"],
        "status": status,
        "completed_at": time.time() - age,
        "rag": {"status": rag_status, "index_digest": "catalog-index-abc123"},
    }
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    spec["execution"]["corpus_check"]["sha256"] = _sha(path)
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def experiment_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "planner@example.invalid")
    _git(root, "config", "user.name", "Experiment Planner Test")
    (root / "research/specs").mkdir(parents=True)
    (root / "research/runners").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "tools/helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    runner = root / "research/runners/example.py"
    runner.write_text(
        "import argparse, json, pathlib\n"
        "p=argparse.ArgumentParser(); p.add_argument('--seed'); p.add_argument('--phase'); "
        "p.add_argument('--arm'); p.add_argument('--out'); p.add_argument('--fail', action='store_true')\n"
        "a=p.parse_args();\n"
        "if a.fail: raise SystemExit(7)\n"
        "pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); "
        "pathlib.Path(a.out).write_text(json.dumps({'seed': a.seed}))\n",
        encoding="utf-8",
    )
    spec_path = root / "research/specs/test.json"
    spec_path.write_text(json.dumps(_base_spec(), indent=2), encoding="utf-8")
    _write_corpus(root, spec_path)
    _commit(root, "test fixture")
    venv_bin = root / ".venv/bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)
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


def test_matrix_is_deterministic_and_binds_manifest_and_rag(experiment_repo):
    root, spec_path, _ = experiment_repo
    first = expand_experiment_jobs(spec_path, ["calibration"], root=root)
    second = expand_experiment_jobs(spec_path, ["calibration"], root=root)
    assert first == second
    assert len(first) == 8
    assert [(job["backend"], job["arm"], job["seed"]) for job in first[:4]] == [
        ("cupy", "control", 11), ("cupy", "control", 12),
        ("cupy", "treatment", 11), ("cupy", "treatment", 12),
    ]
    job = first[0]
    assert job["sealed"] is False
    assert job["corpus_check_sha256"] == _sha(root / "research/queue/planner-corpus-check.json")
    assert job["execution_manifest_sha256"] == job["execution_contract"]["execution_snapshot"]["manifest_sha256"]
    assert "tools/experiment.py execute-job" in job["command"]
    assert "-m research.runners.example" in job["command"]
    assert "tools/queue_add.sh gpu" in job["enqueue_command"]


def test_missing_backend_device_declaration_is_refused(experiment_repo):
    root, spec_path, _ = experiment_repo
    spec = json.loads(spec_path.read_text())
    del spec["execution"]["targets"]["cupy"]
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(HarnessError, match="no execution target/device declaration"):
        expand_experiment_jobs(spec_path, ["calibration"], root=root)


def test_authoritative_runtime_dirt_and_post_calibration_outputs_are_allowed(experiment_repo):
    root, spec_path, outside = experiment_repo
    provenance = root / "research/findings/raw/_provenance/runs.jsonl"
    provenance.parent.mkdir(parents=True)
    provenance.write_text('{"run":"existing"}\n', encoding="utf-8")
    log = root / "research/queue/.corpus_checks.jsonl"
    log.write_text('{"query":"existing"}\n', encoding="utf-8")
    seal_path = outside / "seal.json"
    create_experiment_seal(spec_path, seal_path, root=root)

    calibration = root / "research/findings/raw/planner/calibration/numpy/control-11.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text("{}\n", encoding="utf-8")
    Path(str(calibration) + ".prov.json").write_text("{}\n", encoding="utf-8")
    jobs = expand_experiment_jobs(spec_path, ["held_out"], seal_path=seal_path, root=root)
    assert len(jobs) == 4 and all(job["sealed"] for job in jobs)


@pytest.mark.parametrize("tamper", ["code", "config"])
def test_code_and_config_tampering_invalidate_seal(experiment_repo, tamper):
    root, spec_path, outside = experiment_repo
    seal_path = outside / "seal.json"
    create_experiment_seal(spec_path, seal_path, root=root)
    if tamper == "code":
        (root / "research/runners/example.py").write_text("CHANGED = True\n", encoding="utf-8")
    else:
        spec = json.loads(spec_path.read_text())
        spec["execution"]["claim_stale_seconds"] = 120
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(HarnessError, match="wrong sha256|changed|unsealed"):
        expand_experiment_jobs(spec_path, ["held_out"], seal_path=seal_path, root=root)


def test_noncanonical_checkout_revision_mismatch_fails_before_runner(experiment_repo):
    root, spec_path, outside = experiment_repo
    job = expand_experiment_jobs(spec_path, ["calibration"], root=root)[0]
    canonical = outside / "canonical"
    subprocess.run(["git", "clone", "-q", str(root), str(canonical)], check=True)
    _git(canonical, "config", "user.email", "planner@example.invalid")
    _git(canonical, "config", "user.name", "Experiment Planner Test")
    (canonical / "tools/helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    _commit(canonical, "canonical advanced")
    with pytest.raises(HarnessError, match="source revision mismatch"):
        _verify_runtime_snapshot(job["execution_contract"], canonical)


def _make_archive(source, destination, revision):
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__"))
    paths = []
    for relative_root in ("sim", "research/runners", "experiment", "tools"):
        base = destination / relative_root
        if base.is_dir():
            paths.extend(path for path in base.rglob("*") if path.is_file() and path.suffix in (".py", ".sh"))
    init = destination / "research/__init__.py"
    if init.is_file():
        paths.append(init)
    lines = [f"{_sha(path)}  {path.relative_to(destination).as_posix()}\n" for path in sorted(paths)]
    manifest = destination / ".source_manifest.sha256"
    manifest.write_text("".join(lines), encoding="utf-8")
    (destination / ".source_revision").write_text(
        f"git_sha={revision}\nsource_kind=git_archive\nsource_manifest_sha256={_sha(manifest)}\n",
        encoding="utf-8",
    )


def test_stale_cluster_archive_revision_is_rejected(experiment_repo):
    root, spec_path, outside = experiment_repo
    job = expand_experiment_jobs(spec_path, ["calibration"], root=root)[0]
    archive = outside / "archive"
    _make_archive(root, archive, "stale-cluster-revision")
    with pytest.raises(HarnessError, match="source revision mismatch"):
        _verify_runtime_snapshot(job["execution_contract"], archive)


@pytest.mark.parametrize(
    ("status", "rag_status", "age", "message"),
    [("failed", "success", 0, "successful matching retrieval"),
     ("success", "failed", 0, "successful matching retrieval"),
     ("success", "success", 7200, "stale")],
)
def test_failed_or_stale_rag_checks_block_planning(experiment_repo, status, rag_status, age, message):
    root, spec_path, _ = experiment_repo
    _write_corpus(root, spec_path, status=status, rag_status=rag_status, age=age)
    _commit(root, "corpus state")
    with pytest.raises(HarnessError, match=message):
        expand_experiment_jobs(spec_path, ["calibration"], root=root)


def test_declared_extra_input_tampering_invalidates_seal(experiment_repo):
    root, spec_path, outside = experiment_repo
    data = root / "data/operating-point.json"
    data.parent.mkdir()
    data.write_text('{"current": 100}\n', encoding="utf-8")
    spec = json.loads(spec_path.read_text())
    spec["execution"]["inputs"] = [{"path": "data/operating-point.json", "sha256": _sha(data)}]
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    _commit(root, "declare operating point")
    seal_path = outside / "seal.json"
    create_experiment_seal(spec_path, seal_path, root=root)
    data.write_text('{"current": 101}\n', encoding="utf-8")
    with pytest.raises(HarnessError, match="wrong sha256|changed|unsealed"):
        expand_experiment_jobs(spec_path, ["held_out"], seal_path=seal_path, root=root)


def test_failed_job_releases_claim_and_stale_claim_is_recoverable(experiment_repo):
    root, spec_path, _ = experiment_repo
    spec = json.loads(spec_path.read_text())
    spec["execution"]["targets"]["numpy"]["lane"] = "local"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    _commit(root, "local execution")
    job = next(item for item in expand_experiment_jobs(spec_path, ["calibration"], root=root)
               if item["backend"] == "numpy")
    contract = json.loads(json.dumps(job["execution_contract"]))
    failed_command = [*contract["runner_command"], "--fail"]
    contract["runner_command"] = failed_command
    with pytest.raises(HarnessError, match="exit code 7"):
        execute_job_contract(contract, failed_command, root=root)
    claim = root / job["output_claim"]
    assert not claim.exists()

    claim.parent.mkdir(parents=True, exist_ok=True)
    claim.write_text(json.dumps({"started_at": time.time() - 120, "hostname": "dead-worker", "pid": 1}))
    result = execute_job_contract(job["execution_contract"], job["execution_contract"]["runner_command"], root=root)
    assert result["status"] == "complete"
    assert not claim.exists()
    assert (root / job["output"]).is_file()


def test_active_claim_and_successful_output_block_duplicates(experiment_repo):
    root, spec_path, _ = experiment_repo
    job = expand_experiment_jobs(spec_path, ["calibration"], root=root)[0]
    claim = root / job["output_claim"]
    claim.parent.mkdir(parents=True)
    claim.write_text(json.dumps({"started_at": time.time(), "hostname": os.uname().nodename, "pid": os.getpid()}))
    with pytest.raises(HarnessError, match="active"):
        execute_job_contract(job["execution_contract"], job["execution_contract"]["runner_command"], root=root)


def test_prerequisites_and_stop_decisions_gate_downstream_jobs(experiment_repo):
    root, spec_path, _ = experiment_repo
    prerequisite = root / "evidence/calibration.json"
    prerequisite.parent.mkdir()
    prerequisite.write_text('{"status":"measured"}\n', encoding="utf-8")
    spec = json.loads(spec_path.read_text())
    spec["partitions"]["replication"] = [15]
    spec["prerequisites"] = [{"id": "calibration", "path": "evidence/calibration.json",
                              "sha256": _sha(prerequisite), "partitions": ["replication"]}]
    spec["stop_rules"] = [{"id": "decision", "blocks": ["replication"],
                           "decision_file": "evidence/decision.json"}]
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    _commit(root, "add downstream gate")
    with pytest.raises(HarnessError, match="no valid decision"):
        expand_experiment_jobs(spec_path, ["replication"], root=root)
    decision = {"rule_id": "decision", "spec_sha256": _canonical_hash(spec), "decision": "continue"}
    (root / "evidence/decision.json").write_text(json.dumps(decision), encoding="utf-8")
    assert len(expand_experiment_jobs(spec_path, ["replication"], root=root)) == 4


def test_output_collisions_and_plan_mutation_are_refused(experiment_repo):
    root, spec_path, outside = experiment_repo
    jobs = expand_experiment_jobs(spec_path, ["calibration"], root=root)
    output = root / jobs[0]["output"]
    output.parent.mkdir(parents=True)
    output.write_text("old result\n", encoding="utf-8")
    with pytest.raises(HarnessError, match="output collision"):
        expand_experiment_jobs(spec_path, ["calibration"], root=root)
    output.unlink()
    plan = outside / "plan"
    index = write_experiment_plan(jobs, plan)
    assert index["count"] == 8
    assert all(path.stat().st_mode & 0o222 == 0 for path in plan.iterdir())
    with pytest.raises(HarnessError, match="refusing to mutate"):
        write_experiment_plan(jobs, plan)
