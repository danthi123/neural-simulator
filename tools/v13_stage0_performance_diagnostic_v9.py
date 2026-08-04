#!/usr/bin/env python3
"""Run the preregistered V13 V8 normal-path performance diagnostic.

This is process-only evidence. It cannot promote V8 or open Stage 1. The GPU
worker is intentionally run in a fresh subprocess for every cell repetition so
source and cache conditions are explicit and the host process cannot retain a
previous simulation module state.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "research/specs/v13_stage0_performance_diagnostic_v9.json"
RUNNER_REL = Path("research/runners/_vocal_action_credit_gate_v13_tonic_output.py")
SOURCE_RELS = (
    RUNNER_REL,
    Path("sim/bridge.py"),
    Path("sim/regions.py"),
    Path("sim/kernels.py"),
    Path("research/specs/v13_tonic_output_substrate.json"),
    Path("research/specs/v13_tonic_output_stage0_process_correction_v6.json"),
)
CANDIDATE_LINEAGE = "1ecc85cd698539a6ef92e112d2c49092cfa21f1e"
CONTROL_REVISION = "1bec3c22ad7c535a2cbb27860e5bf4cfd51d6d6f"
DEFAULT_CONTROL_ROOT = Path("/tmp/sim-v13-v6-candidate-1bec")
DEFAULT_QUEUE_RUNNING = Path("/home/dant123/Projects/sim/research/queue/gpu.queue.running")
DEFAULT_LEASE = Path("/tmp/sim-local-model-gpu0.lock")
HISTORICAL_BASELINE = 5.78883048100397
CELL_DEFINITIONS = {
    "A": {"source": "candidate", "cache_state": "cold-process", "warm_v2": False},
    "B": {"source": "control", "cache_state": "cold-process", "warm_v2": False},
    "C": {"source": "candidate", "cache_state": "after-v2-warmup", "warm_v2": True},
    "D": {"source": "control", "cache_state": "after-v2-warmup", "warm_v2": True},
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _source_snapshot(root: Path) -> dict:
    files = {}
    missing = []
    for relative in SOURCE_RELS:
        path = root / relative
        if not path.is_file():
            missing.append(str(relative))
        else:
            files[str(relative)] = _sha256(path)
    try:
        revision = _git(root, "rev-parse", "HEAD")
        status = _git(root, "status", "--porcelain", "--", *map(str, SOURCE_RELS))
    except (OSError, subprocess.CalledProcessError) as exc:
        revision = None
        status = f"git-error:{type(exc).__name__}"
    return {
        "root": str(root),
        "revision": revision,
        "files": files,
        "missing": missing,
        "source_files_clean": status == "",
        "source_status": status,
    }


def _lease_available(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    return True


def _queue_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def readiness(
    *,
    candidate_root: Path = ROOT,
    control_root: Path = DEFAULT_CONTROL_ROOT,
    queue_running: Path = DEFAULT_QUEUE_RUNNING,
    lease_path: Path = DEFAULT_LEASE,
) -> dict:
    candidate_root = candidate_root.resolve()
    control_root = control_root.resolve()
    candidate = _source_snapshot(candidate_root)
    control = _source_snapshot(control_root)
    candidate_lineage_match = False
    if candidate["revision"]:
        try:
            candidate_lineage_match = (
                _git(candidate_root, "diff", "--quiet", CANDIDATE_LINEAGE, "--", *map(str, SOURCE_RELS))
                == ""
            )
        except subprocess.CalledProcessError as exc:
            candidate_lineage_match = exc.returncode == 0
        except (OSError, subprocess.CalledProcessError):
            candidate_lineage_match = False
    queue = _queue_lines(queue_running)
    checks = {
        "candidate_sources_present": not candidate["missing"],
        "control_sources_present": not control["missing"],
        "candidate_matches_v8_source_files": candidate_lineage_match,
        "control_revision_locked": control["revision"] == CONTROL_REVISION,
        "candidate_sources_clean": candidate["source_files_clean"],
        "control_sources_clean": control["source_files_clean"],
        "runner_identical_between_sources": (
            candidate["files"].get(str(RUNNER_REL))
            == control["files"].get(str(RUNNER_REL))
        ),
        "gpu_queue_running_empty": not queue,
        "shared_gpu_lease_available": _lease_available(lease_path),
    }
    return {
        "schema": "v13-stage0-performance-diagnostic-v9-readiness-v1",
        "status": "ready" if all(checks.values()) else "refused",
        "checks": checks,
        "candidate": candidate,
        "control": control,
        "queue_running_path": str(queue_running),
        "queue_running": queue,
        "lease_path": str(lease_path),
        "spec_sha256": _sha256(SPEC_PATH),
    }


def build_run_plan(*, repetitions: int = 3, order_seed: int = 20260804) -> list[dict]:
    jobs = [
        {"cell": cell, "rep": rep, **CELL_DEFINITIONS[cell]}
        for rep in range(1, int(repetitions) + 1)
        for cell in sorted(CELL_DEFINITIONS)
    ]
    random.Random(int(order_seed)).shuffle(jobs)
    for index, job in enumerate(jobs, 1):
        job["sequence"] = index
    return jobs


def _nvidia_telemetry() -> dict:
    query = "temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used,memory.total,pstate"
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "query": query,
        "returncode": completed.returncode,
        "raw": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "captured_at": time.time(),
    }


def _runtime_metadata(cp) -> dict:
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cupy": getattr(cp, "__version__", "unknown"),
        "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
        "gpu": str(name),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _pool_bytes(cp) -> int:
    return int(cp.get_default_memory_pool().used_bytes())


def _exact_zero(cp, array) -> bool:
    return bool(cp.all(array == 0).item())


def _worker(*, source_root: Path, cell: str, warm_v2: bool, warmup_steps: int, timed_steps: int, chunk_steps: int) -> dict:
    source_root = source_root.resolve()
    os.environ["SIM_BACKEND"] = "cupy"
    sys.path.insert(0, str(source_root))
    from research.runners import _vocal_action_credit_gate_v13_tonic_output as runner
    import cupy as cp

    started = time.time()
    source = _source_snapshot(source_root)
    metadata = _runtime_metadata(cp)
    structural = {
        "cell": cell,
        "source": source,
        "metadata": metadata,
        "warm_v2_requested": bool(warm_v2),
    }
    warm_bridge = None
    bridge = None
    try:
        if warm_v2:
            warm_bridge = runner._performance_bridge(active=False, step_mode="v2")
            for _ in range(int(warmup_steps)):
                warm_bridge._run_one_simulation_step()
            runner.synchronize()
            warm_bridge.clear_simulation_state_and_gpu_memory()
            warm_bridge = None

        bridge = runner._performance_bridge(active=False, step_mode="normal")
        pool_after_build = _pool_bytes(cp)
        cfg = bridge.core_config
        structural.update({
            "normal_step_flags": {
                "enable_step_megakernel": bool(getattr(cfg, "enable_step_megakernel", False)),
                "enable_step_megakernel_v2": bool(getattr(cfg, "enable_step_megakernel_v2", False)),
                "enable_step_cudagraph": bool(getattr(cfg, "enable_step_cudagraph", False)),
            },
            "intrinsic_is_none": bridge.cp_intrinsic_current_pA is None,
            "external_current_exact_zero": _exact_zero(cp, bridge.cp_external_input_current),
            "dispatch_preflight": bool(bridge._step_megakernel_can_dispatch()),
            "learning_disabled": not any(
                bool(getattr(cfg, name, False)) for name in (
                    "enable_hebbian_learning", "enable_short_term_plasticity",
                    "enable_homeostasis", "enable_stdp", "enable_structural_plasticity",
                    "enable_reward_modulation", "enable_inhibitory_stdp",
                )
            ),
            "external_input_shape": list(bridge.cp_external_input_current.shape),
        })
        for _ in range(int(warmup_steps)):
            bridge._run_one_simulation_step()
        runner.synchronize()
        pool_before_timing = _pool_bytes(cp)
        telemetry_before = _nvidia_telemetry()

        event_pairs = []
        host_chunks = []
        timed_started = time.perf_counter()
        remaining = int(timed_steps)
        while remaining:
            n_steps = min(int(chunk_steps), remaining)
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            start_event.record()
            chunk_started = time.perf_counter()
            for _ in range(n_steps):
                bridge._run_one_simulation_step()
            host_chunks.append(time.perf_counter() - chunk_started)
            end_event.record()
            event_pairs.append((start_event, end_event))
            remaining -= n_steps
        runner.synchronize()
        timed_wall = time.perf_counter() - timed_started
        cuda_chunks = [float(cp.cuda.get_elapsed_time(start, end) / 1000.0) for start, end in event_pairs]
        telemetry_after = _nvidia_telemetry()
        pool_after_timing = _pool_bytes(cp)

        trace_calls = 0
        trace_dispatch_true = 0
        original_dispatch = bridge._step_megakernel_can_dispatch

        def counted_dispatch():
            nonlocal trace_calls, trace_dispatch_true
            trace_calls += 1
            result = bool(original_dispatch())
            trace_dispatch_true += int(result)
            return result

        bridge._step_megakernel_can_dispatch = counted_dispatch
        for _ in range(64):
            bridge._run_one_simulation_step()
        runner.synchronize()
        structural["trace"] = {
            "steps": 64,
            "normal_step_calls": trace_calls,
            "megakernel_dispatch_true": trace_dispatch_true,
        }
        structural["timed_state_reset"] = False
        return {
            "schema": "v13-stage0-performance-diagnostic-v9-worker-v1",
            "status": "completed",
            "cell": cell,
            "source_root": str(source_root),
            "source_revision": source["revision"],
            "warm_v2": bool(warm_v2),
            "warmup_steps": int(warmup_steps),
            "timed_steps": int(timed_steps),
            "chunk_steps": int(chunk_steps),
            "structural": structural,
            "timing": {
                "wall_seconds": float(timed_wall),
                "host_chunk_seconds": host_chunks,
                "cuda_chunk_seconds": cuda_chunks,
                "cuda_seconds": float(sum(cuda_chunks)),
                "host_minus_device_seconds": float(timed_wall - sum(cuda_chunks)),
            },
            "memory_pool_used_bytes": {
                "after_build_and_before_warmup": int(pool_after_build),
                "after_warmup": int(pool_before_timing),
                "after_timing": int(pool_after_timing),
            },
            "gpu_telemetry": {"before": telemetry_before, "after": telemetry_after},
            "elapsed_seconds": float(time.time() - started),
        }
    finally:
        if bridge is not None:
            bridge.clear_simulation_state_and_gpu_memory()
        if warm_bridge is not None:
            warm_bridge.clear_simulation_state_and_gpu_memory()


def _run_worker(job: dict, *, candidate_root: Path, control_root: Path, warmup_steps: int, timed_steps: int, chunk_steps: int) -> dict:
    source_root = candidate_root if job["source"] == "candidate" else control_root
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--source-root",
        str(source_root),
        "--cell",
        str(job["cell"]),
        "--warm-v2",
        "1" if job["warm_v2"] else "0",
        "--warmup-steps",
        str(warmup_steps),
        "--timed-steps",
        str(timed_steps),
        "--chunk-steps",
        str(chunk_steps),
    ]
    completed = subprocess.run(
        command,
        cwd=source_root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "SIM_BACKEND": "cupy"},
        timeout=max(300, int(timed_steps / 20)),
    )
    stdout = completed.stdout.strip()
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        payload = {
            "schema": "v13-stage0-performance-diagnostic-v9-worker-v1",
            "status": "failed",
            "cell": job["cell"],
            "returncode": completed.returncode,
            "stdout": stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    payload["sequence"] = job["sequence"]
    payload["rep"] = job["rep"]
    payload["cache_state"] = job["cache_state"]
    payload["source_label"] = job["source"]
    if completed.returncode != 0 and payload.get("status") == "completed":
        payload["status"] = "failed"
        payload["returncode"] = completed.returncode
    return payload


def _median(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def summarize(results: list[dict]) -> dict:
    cells = {}
    for cell in sorted(CELL_DEFINITIONS):
        rows = [row for row in results if row.get("cell") == cell and row.get("status") == "completed"]
        walls = [float(row["timing"]["wall_seconds"]) for row in rows]
        devices = [float(row["timing"]["cuda_seconds"]) for row in rows]
        cells[cell] = {
            "source": CELL_DEFINITIONS[cell]["source"],
            "cache_state": CELL_DEFINITIONS[cell]["cache_state"],
            "completed_repetitions": len(rows),
            "wall_seconds": walls,
            "cuda_seconds": devices,
            "median_wall_seconds": _median(walls),
            "median_cuda_seconds": _median(devices),
            "median_host_minus_device_seconds": _median([
                float(row["timing"]["host_minus_device_seconds"]) for row in rows
            ]),
            "all_structural_checks_pass": all(
                bool(row.get("structural", {}).get("intrinsic_is_none"))
                and bool(row.get("structural", {}).get("external_current_exact_zero"))
                and not bool(row.get("structural", {}).get("dispatch_preflight"))
                and row.get("structural", {}).get("trace", {}).get("megakernel_dispatch_true") == 0
                for row in rows
            ) and len(rows) == 3,
        }

    def ratio(numerator: str, denominator: str) -> float | None:
        a = cells[numerator]["median_wall_seconds"]
        b = cells[denominator]["median_wall_seconds"]
        return None if a is None or b in (None, 0) else float(a / b)

    return {
        "historical_baseline_seconds": HISTORICAL_BASELINE,
        "cells": cells,
        "comparisons": {
            "candidate_cold_vs_control_cold": ratio("A", "B"),
            "candidate_after_v2_vs_control_after_v2": ratio("C", "D"),
            "candidate_cold_vs_historical": (
                None if cells["A"]["median_wall_seconds"] is None
                else cells["A"]["median_wall_seconds"] / HISTORICAL_BASELINE
            ),
            "candidate_after_v2_vs_historical": (
                None if cells["C"]["median_wall_seconds"] is None
                else cells["C"]["median_wall_seconds"] / HISTORICAL_BASELINE
            ),
        },
        "interpretation_status": "requires_review; no automatic gate verdict",
        "sealed_v8_boundary": "unchanged PERFORMANCE_NO_GO",
        "stage1_seed_1031": "sealed-not-read-or-executed",
    }


def run_diagnostic(
    *,
    candidate_root: Path,
    control_root: Path,
    output: Path,
    queue_running: Path,
    lease_path: Path,
    repetitions: int,
    order_seed: int,
    warmup_steps: int,
    timed_steps: int,
    chunk_steps: int,
) -> dict:
    ready = readiness(
        candidate_root=candidate_root,
        control_root=control_root,
        queue_running=queue_running,
        lease_path=lease_path,
    )
    if ready["status"] != "ready":
        result = {
            "schema": "v13-stage0-performance-diagnostic-v9-result-v1",
            "status": "refused",
            "reason": "readiness gate failed",
            "readiness": ready,
            "promotion_effect": "none; sealed V8 NO_GO remains authoritative",
        }
        _write_exclusive(output, result)
        return result

    lease_path.parent.mkdir(parents=True, exist_ok=True)
    with lease_path.open("a+") as lease:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result = {
                "schema": "v13-stage0-performance-diagnostic-v9-result-v1",
                "status": "refused",
                "reason": "shared GPU lease became busy after readiness",
                "readiness": ready,
            }
            _write_exclusive(output, result)
            return result
        queue_after_lease = _queue_lines(queue_running)
        if queue_after_lease:
            result = {
                "schema": "v13-stage0-performance-diagnostic-v9-result-v1",
                "status": "refused",
                "reason": "GPU queue became nonempty after readiness",
                "readiness": {**ready, "queue_running": queue_after_lease},
            }
            _write_exclusive(output, result)
            return result
        jobs = build_run_plan(repetitions=repetitions, order_seed=order_seed)
        results = []
        for job in jobs:
            results.append(_run_worker(
                job,
                candidate_root=candidate_root,
                control_root=control_root,
                warmup_steps=warmup_steps,
                timed_steps=timed_steps,
                chunk_steps=chunk_steps,
            ))
        result = {
            "schema": "v13-stage0-performance-diagnostic-v9-result-v1",
            "status": "complete" if all(row.get("status") == "completed" for row in results) else "incomplete",
            "preregistration": str(SPEC_PATH.relative_to(ROOT)),
            "preregistration_sha256": _sha256(SPEC_PATH),
            "candidate_root": str(candidate_root.resolve()),
            "control_root": str(control_root.resolve()),
            "backend": "cupy",
            "device": "NVIDIA GeForce RTX 3090",
            "repetitions": int(repetitions),
            "order_seed": int(order_seed),
            "run_plan": jobs,
            "readiness": ready,
            "results": results,
            "summary": summarize(results),
            "promotion_effect": "none; sealed V8 NO_GO remains authoritative",
        }
        _write_exclusive(output, result)
        return result


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--queue-running", type=Path, default=DEFAULT_QUEUE_RUNNING)
    parser.add_argument("--lease-path", type=Path, default=DEFAULT_LEASE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--order-seed", type=int, default=20260804)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--timed-steps", type=int, default=20000)
    parser.add_argument("--chunk-steps", type=int, default=5000)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--cell")
    parser.add_argument("--warm-v2", choices=("0", "1"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker:
        if not args.source_root or args.cell not in CELL_DEFINITIONS or args.warm_v2 is None:
            raise SystemExit("worker arguments are incomplete")
        result = _worker(
            source_root=args.source_root,
            cell=args.cell,
            warm_v2=args.warm_v2 == "1",
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
            chunk_steps=args.chunk_steps,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.out is None:
        raise SystemExit("--out is required for the controller")
    result = run_diagnostic(
        candidate_root=args.candidate_root,
        control_root=args.control_root,
        output=args.out,
        queue_running=args.queue_running,
        lease_path=args.lease_path,
        repetitions=args.repetitions,
        order_seed=args.order_seed,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
        chunk_steps=args.chunk_steps,
    )
    print(json.dumps({
        "status": result["status"],
        "out": str(args.out),
        "summary": result.get("summary", {}).get("interpretation_status"),
    }, sort_keys=True))
    return 0 if result["status"] == "complete" else 75


if __name__ == "__main__":
    raise SystemExit(main())
