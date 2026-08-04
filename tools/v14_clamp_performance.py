#!/usr/bin/env python3
"""Bounded engineering benchmark for the V14 inhibitory clamp.

This tool produces performance evidence only.  It never supplies scientific
seeds or a biological verdict.  A 20,000-step run requires an explicit
``--approve-full`` acknowledgement; shorter smoke runs remain available for
validating the command and the real CUDA path.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = Path(os.environ.get("SIM_SHARED_ROOT", "/home/dant123/Projects/sim"))
GPU_LEASE = Path(os.environ.get("SIM_GPU_LEASE_PATH", "/tmp/sim-local-model-gpu0.lock"))
GPU_QUEUE = SHARED_ROOT / "research/queue/gpu.queue"
GPU_RUNNING = SHARED_ROOT / "research/queue/gpu.queue.running"
FULL_STEPS = 20_000
DEFAULT_SMOKE_STEPS = 200
DEFAULT_WARMUP_STEPS = 100
DEFAULT_NEURONS = 600
DEFAULT_WORKER_VRAM_MIB = 2048
DEFAULT_VRAM_CAP_MIB = 20_000
SCIENTIFIC_PROCESS_MARKERS = ("-m research.runners", "/research/runners/")


class BenchmarkRefused(RuntimeError):
    """The requested benchmark would violate a mechanical safety gate."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _queue_lines(path: Path) -> list[str]:
    try:
        return [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except FileNotFoundError:
        return []


def _active_scientific_processes(proc_root: Path = Path("/proc")) -> list[str]:
    active: list[str] = []
    for cmdline in proc_root.glob("[0-9]*/cmdline"):
        try:
            command = cmdline.read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            ).strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(marker in command for marker in SCIENTIFIC_PROCESS_MARKERS):
            active.append(f"{cmdline.parent.name} {command}")
    return active


def safety_snapshot(
    *, queue_path: Path = GPU_QUEUE, running_path: Path = GPU_RUNNING,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    queued = _queue_lines(queue_path)
    running = _queue_lines(running_path)
    processes = _active_scientific_processes(proc_root)
    return {
        "gpu_queue_path": str(queue_path),
        "gpu_running_path": str(running_path),
        "queued_scientific_jobs": queued,
        "running_scientific_jobs": running,
        "active_scientific_processes": processes,
        "scientific_gpu_idle": not (queued or running or processes),
    }


def validate_request(
    *, steps: int, warmup_steps: int, workers: int,
    declared_worker_vram_mib: int, vram_cap_mib: int, approve_full: bool,
) -> None:
    if steps <= 0 or warmup_steps < 0:
        raise BenchmarkRefused("steps must be positive and warmup steps nonnegative")
    if workers not in {1, 2}:
        raise BenchmarkRefused("workers must be 1 or 2")
    if declared_worker_vram_mib <= 0 or vram_cap_mib <= 0:
        raise BenchmarkRefused("VRAM declarations and cap must be positive")
    if workers * declared_worker_vram_mib > vram_cap_mib:
        raise BenchmarkRefused(
            "aggregate declared worker VRAM exceeds the configured cap"
        )
    if steps >= FULL_STEPS and not approve_full:
        raise BenchmarkRefused(
            "the full 20,000-step benchmark requires controller --approve-full"
        )


class SharedGpuLease:
    def __init__(self, path: Path = GPU_LEASE):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise BenchmarkRefused("shared GPU lease is already held") from exc
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def _device_snapshot(cp) -> dict[str, Any]:
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    return {
        "backend": "cupy",
        "device_index": 0,
        "device_name": str(name),
        "compute_capability": [int(props["major"]), int(props["minor"])],
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "vram_total_bytes": int(total_bytes),
        "vram_free_bytes": int(free_bytes),
        "cupy_version": cp.__version__,
    }


def _clamp_channels(duration_ms: float):
    from sim.config import InhibitoryConductanceClampConfig

    direct_events = [float(step) for step in range(0, max(1, int(duration_ms)), 20)]
    pallidal_events = [
        float(step) for step in range(0, max(1, int(duration_ms)), 11)
    ]
    return [
        InhibitoryConductanceClampConfig(
            pathway="engineering_direct", target_region="snr",
            tau_rise_ms=0.9, tau_decay_ms=6.2, reversal_mV=-70.0,
            event_peak_nS=0.9, membrane_area_um2=2000.0,
            event_times_ms=direct_events,
        ),
        InhibitoryConductanceClampConfig(
            pathway="engineering_pallidal", target_region="snr",
            tau_rise_ms=0.4, tau_decay_ms=2.1, reversal_mV=-70.0,
            event_peak_nS=2.0, membrane_area_um2=2000.0,
            event_times_ms=pallidal_events,
        ),
    ]


def _build_bridge(*, enabled: bool, neurons: int, total_steps: int):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
    from sim.enums import NeuronModel
    from sim.regions import BrainRegion

    dt_ms = 0.05
    duration_ms = max(1.0, total_steps * dt_ms)
    region = BrainRegion(
        name="snr", n_neurons=neurons, internal_density=0.0,
        snr_g_nalcn_max=0.01, snr_g_nap_max=0.02,
        snr_g_ca_max=0.03, snr_g_sk_max=0.04, snr_g_h_max=0.005,
    )
    config = CoreSimConfig(
        num_neurons=neurons, connections_per_neuron=0, seed=0,
        neuron_model_type=NeuronModel.HODGKIN_HUXLEY.name,
        default_neuron_type_hh="HH_EXCITATORY_DEFAULT_LEGACY",
        dt_ms=dt_ms, total_simulation_time_ms=duration_ms,
        enable_brain_region_framework=True, brain_regions=[region],
        region_pathways=[], enable_parameter_heterogeneity=False,
        enable_ou_process=False, enable_conductance_noise=False,
        enable_hebbian_learning=False, enable_short_term_plasticity=False,
        enable_homeostasis=False, enable_stdp=False,
        enable_structural_plasticity=False, enable_reward_modulation=False,
        enable_inhibitory_stdp=False, enable_nmda=False, enable_gabab=False,
        hh_external_drive_scale=0.0,
        enable_inhibitory_conductance_clamp=enabled,
        inhibitory_conductance_clamps=(
            _clamp_channels(duration_ms) if enabled else []
        ),
    )
    bridge = SimulationBridge(
        core_config=config, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(enable_profiling=False),
    )
    bridge.strict_step_errors = True
    bridge._initialize_simulation_data()
    if not bridge.is_initialized:
        raise RuntimeError("benchmark bridge failed to initialize")
    return bridge


def _advance(bridge, steps: int) -> None:
    for _ in range(steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge.runtime_state.current_time_step += 1


def _measure_condition(cp, *, enabled: bool, neurons: int, steps: int,
                       warmup_steps: int) -> dict[str, Any]:
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    before_used = int(pool.used_bytes())
    bridge = _build_bridge(
        enabled=enabled, neurons=neurons, total_steps=warmup_steps + steps
    )
    try:
        initialized_used = int(pool.used_bytes())
        _advance(bridge, warmup_steps)
        cp.cuda.runtime.deviceSynchronize()
        warmed_used = int(pool.used_bytes())
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        host_start = time.perf_counter()
        _advance(bridge, steps)
        stop.record()
        stop.synchronize()
        host_seconds = float(time.perf_counter() - host_start)
        cuda_seconds = float(cp.cuda.get_elapsed_time(start, stop) / 1000.0)
        final_used = int(pool.used_bytes())
        return {
            "condition": "clamp_enabled" if enabled else "clamp_disabled",
            "steps": steps,
            "warmup_steps": warmup_steps,
            "host_seconds": host_seconds,
            "cuda_event_seconds": cuda_seconds,
            "host_microseconds_per_step": host_seconds * 1e6 / steps,
            "cuda_microseconds_per_step": cuda_seconds * 1e6 / steps,
            "launch_observation": {
                "launch_count": None,
                "reason": "not observed without an external profiler",
                "synchronization_inside_measured_loop": False,
            },
            "memory_pool": {
                "before_used_bytes": before_used,
                "initialized_used_bytes": initialized_used,
                "warmed_used_bytes": warmed_used,
                "final_used_bytes": final_used,
                "footprint_bytes": max(initialized_used, warmed_used, final_used)
                - before_used,
            },
        }
    finally:
        bridge.clear_simulation_state_and_gpu_memory()
        pool.free_all_blocks()


def run_worker(*, worker_index: int, steps: int, warmup_steps: int,
               neurons: int) -> dict[str, Any]:
    os.environ["SIM_BACKEND"] = "cupy"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import cupy as cp

    device_before = _device_snapshot(cp)
    disabled = _measure_condition(
        cp, enabled=False, neurons=neurons, steps=steps,
        warmup_steps=warmup_steps,
    )
    enabled = _measure_condition(
        cp, enabled=True, neurons=neurons, steps=steps,
        warmup_steps=warmup_steps,
    )
    ratio = enabled["cuda_event_seconds"] / disabled["cuda_event_seconds"]
    full_pair_seconds = FULL_STEPS * (
        enabled["cuda_microseconds_per_step"]
        + disabled["cuda_microseconds_per_step"]
    ) / 1e6
    return {
        "schema": "v14-clamp-performance-worker-v1",
        "status": "completed",
        "worker_index": worker_index,
        "device": device_before,
        "configuration": {
            "neurons": neurons, "dt_ms": 0.05,
            "engineering_construction_seed": 0, "scientific_seeds": [],
        },
        "conditions": [disabled, enabled],
        "comparison": {
            "clamp_enabled_to_disabled_cuda_ratio": ratio,
            "incremental_cuda_microseconds_per_step": (
                enabled["cuda_microseconds_per_step"]
                - disabled["cuda_microseconds_per_step"]
            ),
        },
        "projected_cost": {
            "basis": "linear projection from this engineering observation",
            "steps_per_condition": FULL_STEPS,
            "projected_pair_gpu_seconds": full_pair_seconds,
            "projected_pair_gpu_hours": full_pair_seconds / 3600.0,
        },
        "scientific_verdict": None,
    }


def _worker_command(args, output: Path, worker_index: int) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "worker",
        "--steps", str(args.steps), "--warmup-steps", str(args.warmup_steps),
        "--neurons", str(args.neurons), "--worker-index", str(worker_index),
        "--output", str(output),
    ]


def run_controller(args) -> dict[str, Any]:
    validate_request(
        steps=args.steps, warmup_steps=args.warmup_steps, workers=args.workers,
        declared_worker_vram_mib=args.declared_worker_vram_mib,
        vram_cap_mib=args.vram_cap_mib, approve_full=args.approve_full,
    )
    before = safety_snapshot()
    if not before["scientific_gpu_idle"]:
        raise BenchmarkRefused("scientific GPU queue or workload is active")
    with SharedGpuLease():
        after_lease = safety_snapshot()
        if not after_lease["scientific_gpu_idle"]:
            raise BenchmarkRefused("scientific GPU work appeared after lease acquisition")
        with tempfile.TemporaryDirectory(prefix="v14-clamp-benchmark-") as tmp:
            paths = [Path(tmp) / f"worker-{index}.json" for index in range(args.workers)]
            processes = [
                subprocess.Popen(_worker_command(args, path, index), cwd=ROOT)
                for index, path in enumerate(paths)
            ]
            returncodes = [process.wait() for process in processes]
            if any(returncodes):
                raise RuntimeError(f"benchmark worker failure: {returncodes}")
            workers = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    projected_gpu_seconds = sum(
        worker["projected_cost"]["projected_pair_gpu_seconds"]
        for worker in workers
    )
    receipt = {
        "schema": "v14-clamp-engineering-performance-receipt-v1",
        "status": "completed",
        "classification": "non-scientific-engineering-benchmark",
        "created_unix_seconds": time.time(),
        "source": {
            "root": str(ROOT), "revision": _git("rev-parse", "HEAD"),
            "status": _git("status", "--porcelain", "--", "sim", "experiment"),
            "benchmark_sha256": _sha256(Path(__file__)),
        },
        "request": {
            "steps": args.steps, "warmup_steps": args.warmup_steps,
            "neurons": args.neurons, "workers": args.workers,
            "declared_worker_vram_mib": args.declared_worker_vram_mib,
            "aggregate_declared_vram_mib": (
                args.workers * args.declared_worker_vram_mib
            ),
            "vram_cap_mib": args.vram_cap_mib,
            "full_run_approved": bool(args.approve_full),
        },
        "safety": {
            "lease_path": str(GPU_LEASE), "lease_held_for_workers": True,
            "prelease": before, "postlease": after_lease,
            "never_overlap_active_scientific_queue": True,
        },
        "workers": workers,
        "projected_cost": {
            "basis": "sum of worker linear projections",
            "full_steps_per_condition": FULL_STEPS,
            "aggregate_gpu_seconds": projected_gpu_seconds,
            "aggregate_gpu_hours": projected_gpu_seconds / 3600.0,
            "concurrent_wall_seconds_upper_estimate": max(
                worker["projected_cost"]["projected_pair_gpu_seconds"]
                for worker in workers
            ),
        },
        "scientific_seeds": [], "scientific_verdict": None,
        "biology_claim": None,
    }
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="emit the reviewed full-run command")
    plan.add_argument("--workers", type=int, choices=(1, 2), default=1)
    plan.add_argument("--declared-worker-vram-mib", type=int, default=DEFAULT_WORKER_VRAM_MIB)
    plan.add_argument("--vram-cap-mib", type=int, default=DEFAULT_VRAM_CAP_MIB)
    run = subparsers.add_parser("run", help="run a bounded smoke or approved benchmark")
    run.add_argument("--steps", type=int, default=DEFAULT_SMOKE_STEPS)
    run.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    run.add_argument("--neurons", type=int, default=DEFAULT_NEURONS)
    run.add_argument("--workers", type=int, choices=(1, 2), default=1)
    run.add_argument("--declared-worker-vram-mib", type=int, default=DEFAULT_WORKER_VRAM_MIB)
    run.add_argument("--vram-cap-mib", type=int, default=DEFAULT_VRAM_CAP_MIB)
    run.add_argument("--approve-full", action="store_true")
    run.add_argument("--output", type=Path, required=True)
    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--steps", type=int, required=True)
    worker.add_argument("--warmup-steps", type=int, required=True)
    worker.add_argument("--neurons", type=int, required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            validate_request(
                steps=FULL_STEPS, warmup_steps=DEFAULT_WARMUP_STEPS,
                workers=args.workers,
                declared_worker_vram_mib=args.declared_worker_vram_mib,
                vram_cap_mib=args.vram_cap_mib, approve_full=True,
            )
            command = [
                sys.executable, str(Path(__file__).resolve()), "run",
                "--steps", str(FULL_STEPS), "--warmup-steps", str(DEFAULT_WARMUP_STEPS),
                "--workers", str(args.workers),
                "--declared-worker-vram-mib", str(args.declared_worker_vram_mib),
                "--vram-cap-mib", str(args.vram_cap_mib), "--approve-full",
                "--output", str(ROOT / "research/receipts/v14-clamp-performance.json"),
            ]
            print(json.dumps({
                "classification": "non-scientific-engineering-plan",
                "scientific_seeds": [], "scientific_verdict": None,
                "command": command, "executed": False,
            }, indent=2))
            return 0
        if args.command == "worker":
            receipt = run_worker(
                worker_index=args.worker_index, steps=args.steps,
                warmup_steps=args.warmup_steps, neurons=args.neurons,
            )
        else:
            receipt = run_controller(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return 0
    except BenchmarkRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
