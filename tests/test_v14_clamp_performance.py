from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/v14_clamp_performance.py"
SPEC = importlib.util.spec_from_file_location("v14_clamp_performance", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def test_full_run_requires_explicit_controller_approval():
    with pytest.raises(benchmark.BenchmarkRefused, match="approve-full"):
        benchmark.validate_request(
            steps=20_000, warmup_steps=100, workers=1,
            declared_worker_vram_mib=2048, vram_cap_mib=20_000,
            approve_full=False,
        )


def test_smoke_run_does_not_require_full_approval():
    benchmark.validate_request(
        steps=200, warmup_steps=10, workers=1,
        declared_worker_vram_mib=2048, vram_cap_mib=20_000,
        approve_full=False,
    )


def test_concurrency_is_bounded_to_two_workers():
    with pytest.raises(benchmark.BenchmarkRefused, match="workers"):
        benchmark.validate_request(
            steps=200, warmup_steps=10, workers=3,
            declared_worker_vram_mib=1024, vram_cap_mib=20_000,
            approve_full=False,
        )


def test_two_workers_require_aggregate_declared_vram_headroom():
    with pytest.raises(benchmark.BenchmarkRefused, match="aggregate"):
        benchmark.validate_request(
            steps=200, warmup_steps=10, workers=2,
            declared_worker_vram_mib=6000, vram_cap_mib=10_000,
            approve_full=False,
        )


def test_queue_snapshot_fails_closed_for_queued_or_running_work(tmp_path):
    queue = tmp_path / "gpu.queue"
    running = tmp_path / "gpu.queue.running"
    proc = tmp_path / "proc"
    proc.mkdir()
    queue.write_text("scientific command\n", encoding="utf-8")
    running.write_text("", encoding="utf-8")
    snapshot = benchmark.safety_snapshot(
        queue_path=queue, running_path=running, proc_root=proc
    )
    assert snapshot["scientific_gpu_idle"] is False
    assert snapshot["queued_scientific_jobs"] == ["scientific command"]


def test_process_scan_detects_scientific_runner(tmp_path):
    process = tmp_path / "123"
    process.mkdir()
    (process / "cmdline").write_bytes(
        b"python\0-m\0research.runners.stage_b\0"
    )
    assert benchmark._active_scientific_processes(tmp_path)


def test_worker_command_is_exact_and_contains_no_scientific_seed(tmp_path):
    args = SimpleNamespace(steps=200, warmup_steps=100, neurons=600)
    command = benchmark._worker_command(args, tmp_path / "receipt.json", 1)
    assert command[2] == "worker"
    assert "--worker-index" in command
    assert not any("scientific" in token or token == "--seed" for token in command)


def test_worker_bootstrap_adds_repository_root_before_sim_import(monkeypatch):
    source = MODULE_PATH.read_text(encoding="utf-8")
    worker_start = source.index("def run_worker")
    cupy_import = source.index("    import cupy as cp", worker_start)
    path_insert = source.index("sys.path.insert(0, str(ROOT))", worker_start)
    assert path_insert < cupy_import


def test_plan_emits_reviewable_command_without_executing(capsys):
    assert benchmark.main(["plan", "--workers", "2"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["executed"] is False
    assert plan["scientific_seeds"] == []
    assert plan["scientific_verdict"] is None
    assert "--approve-full" in plan["command"]
    assert "20000" in plan["command"]


def test_controller_rechecks_scientific_state_after_lease(monkeypatch, tmp_path):
    snapshots = iter([
        {"scientific_gpu_idle": True},
        {"scientific_gpu_idle": False},
    ])
    monkeypatch.setattr(benchmark, "safety_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(benchmark, "GPU_LEASE", tmp_path / "lease")
    args = SimpleNamespace(
        steps=10, warmup_steps=1, workers=1, neurons=5,
        declared_worker_vram_mib=1, vram_cap_mib=2, approve_full=False,
    )
    with pytest.raises(benchmark.BenchmarkRefused, match="appeared"):
        benchmark.run_controller(args)


def test_source_scope_excludes_uncommitted_research_files():
    source = benchmark._git("status", "--porcelain", "--", "sim", "experiment")
    assert source == ""


def test_benchmark_declares_no_biology_authority():
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert '"scientific_seeds": []' in text
    assert '"scientific_verdict": None' in text
    assert '"biology_claim": None' in text
