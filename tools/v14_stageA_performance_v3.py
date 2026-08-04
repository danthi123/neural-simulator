#!/usr/bin/env python3
"""Run the prospective V14 Stage-A CuPy performance protocol v3.

The caller owns the GPU lease and externally stabilizes the GPU. This harness
validates that state but never acquires a lease, uses sudo, or changes GPU or
host power controls. Candidate and historical-control observations run in two
persistent, source-isolated workers with retained CUDA contexts.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import selectors
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "research/specs/v14_snr_conductance_stageA_implementation.json"
PARENT_PROTOCOL_SPEC_PATH = ROOT / "research/specs/v14_stageA_performance_v2.json"
PROTOCOL_SPEC_PATH = ROOT / "research/specs/v14_stageA_performance_v3.json"
OUTER_PAIRS = 3
WARMUP_STEPS = 5000
REWARM_STEPS = 500
TIMING_BLOCKS = 5
STEPS_PER_BLOCK = 3000
NUM_NEURONS = 600
ACTIVE_BYTES_PER_NEURON = 48
DEFAULT_RATIO_MAX = 1.02
ACTIVE_RATIO_MAX = 1.25
DIRECT_OUTPUT_RATIO_MAX = 0.85
CONSTRUCTION_RNG_SEED = 0
WORKER_TIMEOUT_SECONDS = 1800
WORKER_SHUTDOWN_SECONDS = 10

MAX_INNER_BLOCK_RELATIVE_RANGE = 0.10
MAX_PAIRED_RATIO_RELATIVE_RANGE = 0.05

WORKER_CPU = 10
CONTROLLER_CPU = 16
REQUIRED_NICE = -4
REQUIRED_SCHEDULER = getattr(os, "SCHED_OTHER", 0)
GPU_CLOCK_ENV = "SIM_BENCHMARK_GPU_CLOCK_MHZ"
REQUIRED_GPU_CLOCK_MHZ = 1800.0
GPU_CLOCK_TOLERANCE_MHZ = 15.0
REQUIRED_MEMORY_CLOCK_MHZ = 9751.0
MEMORY_CLOCK_TOLERANCE_MHZ = 0.0
REQUIRED_POWER_LIMIT_W = 300.0
POWER_LIMIT_TOLERANCE_W = 0.0

BLOCKING_PROCESS_MARKERS = (
    "tools/rag/update_indexes.py",
    "-m research.runners",
    "/research/runners/",
)

BUNDLE_ARRAYS = (
    "cp_snr_g_nalcn_max",
    "cp_snr_g_nap_max",
    "cp_snr_g_ca_max",
    "cp_snr_g_sk_max",
    "cp_snr_g_h_max",
    "cp_snr_nap_activation",
    "cp_snr_nap_inactivation",
    "cp_snr_ca_activation",
    "cp_snr_ca_inactivation",
    "cp_snr_calcium",
    "cp_snr_sk_activation",
    "cp_snr_h_activation",
)

CELL_DEFINITIONS = {
    "candidate-default": {
        "source": "candidate", "active": False, "direct_outputs": False,
    },
    "prechange-control-default": {
        "source": "prechange-control", "active": False, "direct_outputs": False,
    },
    "candidate-active-unfused": {
        "source": "candidate", "active": True, "direct_outputs": False,
    },
    "candidate-active": {
        "source": "candidate", "active": True, "direct_outputs": True,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def source_snapshot(root: Path) -> dict[str, Any]:
    root = root.resolve()
    rels = (
        "sim/backend.py",
        "sim/bridge.py",
        "sim/config.py",
        "sim/kernels.py",
        "sim/regions.py",
    )
    return {
        "root": str(root),
        "revision": _git(root, "rev-parse", "HEAD"),
        "status_porcelain": _git(root, "status", "--porcelain", "--", *rels),
        "files": {
            rel: _sha256(root / rel) if (root / rel).is_file() else None
            for rel in rels
        },
    }


def build_run_plan() -> list[dict[str, Any]]:
    """Build the sealed reversible four-cell chains."""
    forward = (
        "candidate-active-unfused",
        "candidate-active",
        "candidate-default",
        "prechange-control-default",
    )
    jobs: list[dict[str, Any]] = []
    sequence = 1
    for outer_pair in range(1, OUTER_PAIRS + 1):
        pair_order = "AB" if outer_pair % 2 else "BA"
        cells = forward if pair_order == "AB" else tuple(reversed(forward))
        for chain_position, cell in enumerate(cells, 1):
            jobs.append(
                {
                    "sequence": sequence,
                    "outer_pair": outer_pair,
                    "pair_order": pair_order,
                    "chain_position": chain_position,
                    "cell": cell,
                    **CELL_DEFINITIONS[cell],
                }
            )
            sequence += 1
    return jobs


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _command_capture(command: list[str], *, timeout: float = 15.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _scheduler_name(policy: int) -> str:
    for name in ("SCHED_OTHER", "SCHED_BATCH", "SCHED_IDLE", "SCHED_FIFO", "SCHED_RR"):
        if getattr(os, name, object()) == policy:
            return name
    return str(policy)


def process_scheduling(pid: int = 0) -> dict[str, Any]:
    try:
        affinity = sorted(os.sched_getaffinity(pid))
        nice = os.getpriority(os.PRIO_PROCESS, pid)
        scheduler = os.sched_getscheduler(pid)
        return {
            "available": True,
            "affinity": affinity,
            "nice": nice,
            "scheduler": scheduler,
            "scheduler_name": _scheduler_name(scheduler),
        }
    except (AttributeError, OSError) as exc:
        return {"available": False, "error": str(exc)}


def pin_current_process(cpu: int) -> dict[str, Any]:
    try:
        os.sched_setaffinity(0, {cpu})
    except (AttributeError, OSError) as exc:
        return {"requested_cpu": cpu, "set_error": str(exc), **process_scheduling()}
    return {"requested_cpu": cpu, **process_scheduling()}


def _power_profile() -> dict[str, Any]:
    sysfs_path = Path("/sys/firmware/acpi/platform_profile")
    sysfs_value = _read_text(sysfs_path)
    if sysfs_value is not None:
        return {"source": str(sysfs_path), "value": sysfs_value}
    command = _command_capture(["powerprofilesctl", "get"])
    return {
        "source": "powerprofilesctl get",
        "value": command["stdout"] if command["returncode"] == 0 else None,
        "command": command,
    }


def _turbo_state() -> dict[str, Any]:
    candidates = (
        (Path("/sys/devices/system/cpu/intel_pstate/no_turbo"), "zero_is_enabled"),
        (Path("/sys/devices/system/cpu/cpufreq/boost"), "one_is_enabled"),
        (Path("/sys/devices/system/cpu/amd_pstate/cpufreq_boost"), "one_is_enabled"),
    )
    records = []
    enabled_values = []
    for path, convention in candidates:
        value = _read_text(path)
        if value is None:
            continue
        enabled = value == ("0" if convention == "zero_is_enabled" else "1")
        records.append({"path": str(path), "value": value, "enabled": enabled})
        enabled_values.append(enabled)
    return {
        "available": bool(records),
        "enabled": bool(records) and all(enabled_values),
        "records": records,
    }


def cpu_controls(cpu: int) -> dict[str, Any]:
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    cpufreq = base / "cpufreq"
    topology = base / "topology"
    governor = _read_text(cpufreq / "scaling_governor")
    epp = _read_text(cpufreq / "energy_performance_preference")
    profile = _power_profile()
    turbo = _turbo_state()
    return {
        "cpu": cpu,
        "online": _read_text(base / "online") or "1",
        "governor": governor,
        "energy_performance_preference": epp,
        "power_profile": profile,
        "turbo": turbo,
        "topology": {
            "core_id": _read_text(topology / "core_id"),
            "physical_package_id": _read_text(topology / "physical_package_id"),
            "thread_siblings_list": _read_text(topology / "thread_siblings_list"),
        },
        "passed": (
            governor == "performance"
            and epp == "performance"
            and profile.get("value") == "performance"
            and turbo.get("enabled") is True
        ),
    }


def active_blocking_workloads() -> list[dict[str, Any]]:
    rows = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if command and any(marker in command for marker in BLOCKING_PROCESS_MARKERS):
            rows.append({"pid": int(entry.name), "command": command[:1000]})
    return sorted(rows, key=lambda row: row["pid"])


GPU_QUERY_FIELDS = (
    "index", "uuid", "pstate", "clocks.sm", "clocks.mem", "power.limit",
    "power.draw", "temperature.gpu", "utilization.gpu", "memory.used", "memory.total",
)


def _parse_csv_rows(text: str, fields: tuple[str, ...]) -> list[dict[str, str]]:
    rows = []
    for line in text.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values)))
    return rows


def gpu_telemetry() -> dict[str, Any]:
    gpu = _command_capture([
        "nvidia-smi", f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ])
    compute_fields = ("pid", "process_name", "used_gpu_memory")
    compute = _command_capture([
        "nvidia-smi", f"--query-compute-apps={','.join(compute_fields)}",
        "--format=csv,noheader,nounits",
    ])
    process_monitor = _command_capture(["nvidia-smi", "pmon", "-c", "1", "-s", "um"])
    gpu_rows = _parse_csv_rows(gpu["stdout"], GPU_QUERY_FIELDS)
    compute_rows = _parse_csv_rows(compute["stdout"], compute_fields)
    return {
        "captured_at_unix": time.time(),
        "gpu": {**gpu, "rows": gpu_rows},
        "compute_processes": {**compute, "rows": compute_rows},
        "process_monitor": process_monitor,
    }


def validate_gpu_telemetry(
    record: dict[str, Any], *, require_active_clocks: bool = True
) -> dict[str, Any]:
    failures = []
    gpu_capture = record.get("gpu", {})
    rows = gpu_capture.get("rows", [])
    if gpu_capture.get("returncode") != 0 or len(rows) != 1:
        failures.append("exactly one queryable NVIDIA GPU is required")
    if record.get("compute_processes", {}).get("returncode") != 0:
        failures.append("compute process inventory is unavailable")
    if record.get("process_monitor", {}).get("returncode") != 0:
        failures.append("GPU process monitor inventory is unavailable")
    if len(rows) == 1:
        row = rows[0]
        checks = [("power.limit", REQUIRED_POWER_LIMIT_W, POWER_LIMIT_TOLERANCE_W)]
        if require_active_clocks:
            checks.extend((
                ("clocks.sm", REQUIRED_GPU_CLOCK_MHZ, GPU_CLOCK_TOLERANCE_MHZ),
                ("clocks.mem", REQUIRED_MEMORY_CLOCK_MHZ, MEMORY_CLOCK_TOLERANCE_MHZ),
            ))
        for field, expected, tolerance in checks:
            try:
                observed = float(row[field])
            except (KeyError, TypeError, ValueError):
                failures.append(f"{field} is unavailable")
                continue
            if abs(observed - expected) > tolerance:
                failures.append(
                    f"{field}={observed} outside {expected} +/- {tolerance}"
                )
    return {"passed": not failures, "failures": failures}


def _direct_output_config_kwargs(config_type, *, direct_outputs: bool) -> dict[str, bool]:
    fields = getattr(config_type, "__dataclass_fields__", {})
    if "enable_snr_direct_outputs" in fields:
        return {"enable_snr_direct_outputs": direct_outputs}
    if direct_outputs:
        raise RuntimeError("source does not support SNr direct outputs")
    return {}


def _build_bridge(*, active: bool, direct_outputs: bool):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
    from sim.enums import NeuronModel
    from sim.regions import BrainRegion

    region_args = {}
    if active:
        region_args = {
            "snr_g_nalcn_max": 0.01,
            "snr_g_nap_max": 0.02,
            "snr_g_ca_max": 0.03,
            "snr_g_sk_max": 0.04,
            "snr_g_h_max": 0.005,
        }
    region = BrainRegion(
        name="performance_population", n_neurons=NUM_NEURONS,
        internal_density=0.0, **region_args,
    )
    config_type = CoreSimConfig
    config = config_type(
        num_neurons=NUM_NEURONS,
        connections_per_neuron=0,
        seed=CONSTRUCTION_RNG_SEED,
        neuron_model_type=NeuronModel.HODGKIN_HUXLEY.name,
        default_neuron_type_hh="HH_EXCITATORY_DEFAULT_LEGACY",
        dt_ms=0.05,
        enable_brain_region_framework=True,
        brain_regions=[region],
        enable_parameter_heterogeneity=False,
        enable_conductance_noise=False,
        enable_hebbian_learning=False,
        enable_short_term_plasticity=False,
        enable_structural_plasticity=False,
        enable_ou_process=False,
        hh_external_drive_scale=0.0,
        **_direct_output_config_kwargs(config_type, direct_outputs=direct_outputs),
    )
    bridge = SimulationBridge(
        core_config=config,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(enable_profiling=False),
    )
    bridge._initialize_simulation_data()
    if not bridge.is_initialized:
        raise RuntimeError("bridge initialization failed")
    bridge.strict_step_errors = True
    return bridge


def _relative_range(values: list[float]) -> float | None:
    if not values:
        return None
    median = float(statistics.median(values))
    if median <= 0.0:
        return None
    return float((max(values) - min(values)) / median)


def run_observation(*, source_root: Path, source_kind: str, cell: str, cp) -> dict[str, Any]:
    definition = CELL_DEFINITIONS[cell]
    if definition["source"] != source_kind:
        raise RuntimeError(f"{cell} cannot run in {source_kind} worker")
    bridge = _build_bridge(
        active=definition["active"], direct_outputs=definition["direct_outputs"]
    )
    try:
        arrays = [getattr(bridge, name, None) for name in BUNDLE_ARRAYS]
        attributes_present = [hasattr(bridge, name) for name in BUNDLE_ARRAYS]
        bundle_bytes = sum(int(array.nbytes) for array in arrays if array is not None)
        default_none = all(array is None for array in arrays)
        active_bytes_exact = bundle_bytes == ACTIVE_BYTES_PER_NEURON * NUM_NEURONS
        structural_ok = (
            active_bytes_exact and all(array is not None for array in arrays)
            if definition["active"] else default_none
        )

        for _ in range(WARMUP_STEPS):
            bridge._run_one_simulation_step()
        cp.cuda.runtime.deviceSynchronize()
        warmup_telemetry = gpu_telemetry()
        warmup_telemetry_validation = validate_gpu_telemetry(warmup_telemetry)
        if not warmup_telemetry_validation["passed"]:
            return {
                "schema": "v14-stageA-performance-worker-response-v3",
                "status": "infrastructure_invalid",
                "cell": cell,
                "warmup_telemetry": warmup_telemetry,
                "warmup_telemetry_validation": warmup_telemetry_validation,
            }
        for _ in range(REWARM_STEPS):
            bridge._run_one_simulation_step()
        cp.cuda.runtime.deviceSynchronize()
        blocks = []
        for block_index in range(1, TIMING_BLOCKS + 1):
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            start_event.record()
            host_started = time.perf_counter()
            for _ in range(STEPS_PER_BLOCK):
                bridge._run_one_simulation_step()
            end_event.record()
            end_event.synchronize()
            blocks.append({
                "block": block_index,
                "steps": STEPS_PER_BLOCK,
                "host_seconds": float(time.perf_counter() - host_started),
                "cuda_event_seconds": float(
                    cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
                ),
            })
        host_values = [block["host_seconds"] for block in blocks]
        cuda_values = [block["cuda_event_seconds"] for block in blocks]
        post_timing_telemetry = gpu_telemetry()
        post_timing_telemetry_validation = validate_gpu_telemetry(
            post_timing_telemetry
        )
        return {
            "schema": "v14-stageA-performance-worker-response-v3",
            "status": "completed" if structural_ok else "infrastructure_invalid",
            "cell": cell,
            "configuration": {
                "num_neurons": NUM_NEURONS,
                "dt_ms": 0.05,
                "warmup_steps": WARMUP_STEPS,
                "rewarm_steps": REWARM_STEPS,
                "timing_blocks": TIMING_BLOCKS,
                "steps_per_block": STEPS_PER_BLOCK,
                "construction_rng_seed": CONSTRUCTION_RNG_SEED,
                "scientific_seeds": [],
                "active": definition["active"],
                "direct_outputs": definition["direct_outputs"],
            },
            "structural": {
                "bundle_attributes_present": attributes_present,
                "default_bundle_arrays_none": default_none,
                "active_bundle_bytes": bundle_bytes,
                "expected_active_bundle_bytes": ACTIVE_BYTES_PER_NEURON * NUM_NEURONS,
                "active_bundle_bytes_exact": active_bytes_exact,
                "passed": structural_ok,
            },
            "timing": {
                "blocks": blocks,
                "total_host_seconds": float(sum(host_values)),
                "total_cuda_event_seconds": float(sum(cuda_values)),
                "median_block_host_seconds": float(statistics.median(host_values)),
                "median_block_cuda_event_seconds": float(statistics.median(cuda_values)),
                "host_block_relative_range": _relative_range(host_values),
                "cuda_block_relative_range": _relative_range(cuda_values),
            },
            "warmup_telemetry": warmup_telemetry,
            "warmup_telemetry_validation": warmup_telemetry_validation,
            "post_timing_telemetry": post_timing_telemetry,
            "post_timing_telemetry_validation": post_timing_telemetry_validation,
        }
    finally:
        bridge.clear_simulation_state_and_gpu_memory()


def persistent_worker(*, source_root: Path, source_kind: str) -> int:
    source_root = source_root.resolve()
    explicit_root = os.environ.get("SIM_SOURCE_ROOT")
    if not explicit_root or Path(explicit_root).resolve() != source_root:
        raise RuntimeError("SIM_SOURCE_ROOT must explicitly match --source-root")
    if source_kind not in {"candidate", "prechange-control"}:
        raise RuntimeError("invalid source kind")
    os.environ["SIM_BACKEND"] = "cupy"
    sys.path.insert(0, str(source_root))
    with contextlib.redirect_stdout(sys.stderr):
        import cupy as cp
        properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = properties.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    scheduling = process_scheduling()
    controls = cpu_controls(WORKER_CPU)
    startup_ok = (
        scheduling.get("affinity") == [WORKER_CPU]
        and scheduling.get("nice") == REQUIRED_NICE
        and scheduling.get("scheduler") == REQUIRED_SCHEDULER
        and controls["passed"]
    )
    startup = {
        "schema": "v14-stageA-performance-worker-startup-v3",
        "status": "ready" if startup_ok else "infrastructure_invalid",
        "source_kind": source_kind,
        "pid": os.getpid(),
        "source": source_snapshot(source_root),
        "scheduling": scheduling,
        "cpu_controls": controls,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "cupy": cp.__version__,
            "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
            "device": str(device_name),
        },
    }
    print(json.dumps(startup, sort_keys=True), flush=True)
    if not startup_ok:
        return 75

    for line in sys.stdin:
        try:
            command = json.loads(line)
            request_id = command.get("request_id")
            if command.get("command") == "shutdown":
                print(json.dumps({"request_id": request_id, "status": "shutdown"}), flush=True)
                return 0
            if command.get("command") != "observe":
                raise ValueError("unknown worker command")
            with contextlib.redirect_stdout(sys.stderr):
                response = run_observation(
                    source_root=source_root,
                    source_kind=source_kind,
                    cell=command["cell"],
                    cp=cp,
                )
            response["request_id"] = request_id
        except BaseException as exc:
            traceback.print_exc(file=sys.stderr)
            response = {
                "schema": "v14-stageA-performance-worker-response-v3",
                "request_id": locals().get("request_id"),
                "status": "infrastructure_failure",
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(response, sort_keys=True), flush=True)
    return 0


class PersistentWorker:
    def __init__(self, *, source_root: Path, source_kind: str):
        self.source_root = source_root.resolve()
        self.source_kind = source_kind
        self._stderr = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--persistent-worker",
            "--source-root", str(self.source_root),
            "--source-kind", source_kind,
        ]
        env = {
            **os.environ,
            "SIM_BACKEND": "cupy",
            "SIM_SOURCE_ROOT": str(self.source_root),
        }

        def pin_worker() -> None:
            os.sched_setaffinity(0, {WORKER_CPU})

        self.process = subprocess.Popen(
            command,
            cwd=self.source_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
            preexec_fn=pin_worker,
        )
        try:
            self.startup = self._read_response(
                request_id=None, timeout=WORKER_TIMEOUT_SECONDS
            )
        except BaseException:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=WORKER_SHUTDOWN_SECONDS)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=WORKER_SHUTDOWN_SECONDS)
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
            self._stderr.close()
            raise

    @property
    def pid(self) -> int:
        return self.process.pid

    def _read_response(self, *, request_id: str | None, timeout: float) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("worker stdout is unavailable")
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout
        ignored = []
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"{self.source_kind} worker response timeout")
                if not selector.select(remaining):
                    raise TimeoutError(f"{self.source_kind} worker response timeout")
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError(
                        f"{self.source_kind} worker exited with {self.process.poll()}: "
                        f"{self.stderr_tail()}"
                    )
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    ignored.append(line.rstrip())
                    continue
                if request_id is None or payload.get("request_id") == request_id:
                    if ignored:
                        payload["ignored_stdout"] = ignored[-20:]
                    return payload
                ignored.append(line.rstrip())
        finally:
            selector.close()

    def observe(self, job: dict[str, Any]) -> dict[str, Any]:
        if self.process.stdin is None:
            raise RuntimeError("worker stdin is unavailable")
        request_id = f"observation-{job['sequence']}"
        self.process.stdin.write(json.dumps({
            "command": "observe", "request_id": request_id, "cell": job["cell"],
        }) + "\n")
        self.process.stdin.flush()
        return self._read_response(request_id=request_id, timeout=WORKER_TIMEOUT_SECONDS)

    def stderr_tail(self) -> str:
        try:
            self._stderr.flush()
            self._stderr.seek(0)
            return self._stderr.read()[-8000:]
        except OSError:
            return ""

    def close(self) -> dict[str, Any]:
        outcome: dict[str, Any] = {"pid": self.pid, "source_kind": self.source_kind}
        try:
            if self.process.poll() is None and self.process.stdin is not None:
                request_id = f"shutdown-{self.pid}"
                self.process.stdin.write(json.dumps({
                    "command": "shutdown", "request_id": request_id,
                }) + "\n")
                self.process.stdin.flush()
                try:
                    outcome["response"] = self._read_response(
                        request_id=request_id, timeout=WORKER_SHUTDOWN_SECONDS
                    )
                except (RuntimeError, TimeoutError) as exc:
                    outcome["shutdown_error"] = str(exc)
            try:
                self.process.wait(timeout=WORKER_SHUTDOWN_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=WORKER_SHUTDOWN_SECONDS)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=WORKER_SHUTDOWN_SECONDS)
            outcome["returncode"] = self.process.returncode
            outcome["stderr_tail"] = self.stderr_tail()
            return outcome
        finally:
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
            self._stderr.close()


def _row_by_pair(rows: list[dict[str, Any]], outer_pair: int, cell: str) -> dict[str, Any] | None:
    return next((
        row for row in rows
        if row.get("outer_pair") == outer_pair
        and row.get("cell") == cell
        and row.get("status") == "completed"
    ), None)


def _timing_value(row: dict[str, Any], channel: str) -> float:
    return float(row["timing"][f"total_{channel}_seconds"])


def _pair_ratios(
    rows: list[dict[str, Any]], numerator: str, denominator: str, channel: str
) -> list[float]:
    ratios = []
    for outer_pair in range(1, OUTER_PAIRS + 1):
        top = _row_by_pair(rows, outer_pair, numerator)
        bottom = _row_by_pair(rows, outer_pair, denominator)
        if top is not None and bottom is not None:
            denominator_value = _timing_value(bottom, channel)
            if denominator_value > 0.0:
                ratios.append(_timing_value(top, channel) / denominator_value)
    return ratios


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    complete = len(completed) == OUTER_PAIRS * len(CELL_DEFINITIONS)
    structural = complete and all(
        row.get("structural", {}).get("passed") is True for row in completed
    )
    definitions = {
        "default_host": ("candidate-default", "prechange-control-default", "host"),
        "default_cuda_event": (
            "candidate-default", "prechange-control-default", "cuda_event",
        ),
        "active_host": ("candidate-active", "candidate-default", "host"),
        "active_cuda_event": ("candidate-active", "candidate-default", "cuda_event"),
        "direct_output_host": (
            "candidate-active", "candidate-active-unfused", "host",
        ),
        "direct_output_cuda_event": (
            "candidate-active", "candidate-active-unfused", "cuda_event",
        ),
    }
    paired_ratios = {
        name: _pair_ratios(rows, numerator, denominator, channel)
        for name, (numerator, denominator, channel) in definitions.items()
    }
    medians = {
        name: float(statistics.median(values)) if values else None
        for name, values in paired_ratios.items()
    }
    paired_dispersion = {
        name: _relative_range(values) for name, values in paired_ratios.items()
    }
    observation_dispersion = []
    for row in completed:
        for channel, field in (
            ("host", "host_block_relative_range"),
            ("cuda_event", "cuda_block_relative_range"),
        ):
            value = row["timing"].get(field)
            observation_dispersion.append({
                "outer_pair": row["outer_pair"],
                "cell": row["cell"],
                "channel": channel,
                "relative_range": value,
                "limit": MAX_INNER_BLOCK_RELATIVE_RANGE,
                "passed": value is not None and value <= MAX_INNER_BLOCK_RELATIVE_RANGE,
            })
    dispersion_valid = complete and all(
        item["passed"] for item in observation_dispersion
    ) and all(
        value is not None and value <= MAX_PAIRED_RATIO_RELATIVE_RANGE
        for value in paired_dispersion.values()
    )
    infrastructure_valid = structural and dispersion_valid
    thresholds_complete = all(value is not None for value in medians.values())
    thresholds_pass = thresholds_complete and (
        medians["default_host"] <= DEFAULT_RATIO_MAX
        and medians["default_cuda_event"] <= DEFAULT_RATIO_MAX
        and medians["active_host"] <= ACTIVE_RATIO_MAX
        and medians["active_cuda_event"] <= ACTIVE_RATIO_MAX
        and medians["direct_output_host"] <= DIRECT_OUTPUT_RATIO_MAX
        and medians["direct_output_cuda_event"] <= DIRECT_OUTPUT_RATIO_MAX
    )
    if not complete:
        status = "pending"
    elif not infrastructure_valid:
        status = "infrastructure_invalid"
    else:
        status = "GO" if thresholds_pass else "NO_GO"
    return {
        "completed_observations": len(completed),
        "expected_observations": OUTER_PAIRS * len(CELL_DEFINITIONS),
        "paired_ratios": paired_ratios,
        "median_paired_ratios": medians,
        "dispersion": {
            "observation_blocks": observation_dispersion,
            "paired_ratio_relative_ranges": paired_dispersion,
            "limits": {
                "max_inner_block_relative_range": MAX_INNER_BLOCK_RELATIVE_RANGE,
                "max_paired_ratio_relative_range": MAX_PAIRED_RATIO_RELATIVE_RANGE,
            },
            "passed": dispersion_valid,
        },
        "fixed_thresholds": {
            "default_off_ratio_max": DEFAULT_RATIO_MAX,
            "active_ratio_max": ACTIVE_RATIO_MAX,
            "direct_output_ratio_max": DIRECT_OUTPUT_RATIO_MAX,
        },
        "infrastructure_valid": infrastructure_valid,
        "performance_status": status,
        "physiology_verdict": None,
        "promotion_effect": "none",
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _preconditions(
    *, source_roots: dict[str, dict[str, Any]], controller_state: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    controls = cpu_controls(CONTROLLER_CPU)
    blockers = active_blocking_workloads()
    env_clock = os.environ.get(GPU_CLOCK_ENV)
    initial_telemetry = gpu_telemetry()
    telemetry_check = validate_gpu_telemetry(
        initial_telemetry, require_active_clocks=False
    )
    checks = [
        {
            "name": "clean source boundaries",
            "ok": all(
                source.get("revision") and not source.get("status_porcelain")
                for source in source_roots.values()
            ),
            "observed": source_roots,
        },
        {
            "name": "engineering-only seed boundary",
            "ok": True,
            "observed": {"scientific_seeds": []},
        },
        {
            "name": "controller scheduling",
            "ok": (
                controller_state.get("affinity") == [CONTROLLER_CPU]
                and controller_state.get("nice") == REQUIRED_NICE
                and controller_state.get("scheduler") == REQUIRED_SCHEDULER
            ),
            "observed": controller_state,
        },
        {"name": "controller CPU and power controls", "ok": controls["passed"], "observed": controls},
        {
            "name": "no RAG indexer or research runner workload",
            "ok": not blockers,
            "observed": blockers,
        },
        {
            "name": "external GPU clock declaration",
            "ok": env_clock == str(int(REQUIRED_GPU_CLOCK_MHZ)),
            "observed": {GPU_CLOCK_ENV: env_clock},
        },
        {
            "name": "initial GPU controls and telemetry",
            "ok": telemetry_check["passed"],
            "observed": {"validation": telemetry_check, "telemetry": initial_telemetry},
        },
    ]
    return checks, initial_telemetry


def run_matrix(*, candidate_root: Path, control_root: Path, output: Path) -> dict[str, Any]:
    controller_state = pin_current_process(CONTROLLER_CPU)
    plan = build_run_plan()
    source_roots = {
        "candidate": source_snapshot(candidate_root),
        "prechange_control": source_snapshot(control_root),
    }
    result: dict[str, Any] = {
        "schema": "v14-stageA-performance-result-v3",
        "status": "running",
        "created_at_unix": time.time(),
        "provenance": {"argv": list(sys.argv), "cwd": str(Path.cwd().resolve())},
        "specification": str(SPEC_PATH.relative_to(ROOT)),
        "specification_sha256": _sha256(SPEC_PATH),
        "parent_protocol_specification": str(PARENT_PROTOCOL_SPEC_PATH.relative_to(ROOT)),
        "parent_protocol_specification_sha256": _sha256(PARENT_PROTOCOL_SPEC_PATH),
        "protocol_specification": str(PROTOCOL_SPEC_PATH.relative_to(ROOT)),
        "protocol_specification_sha256": _sha256(PROTOCOL_SPEC_PATH),
        "harness_sha256": _sha256(Path(__file__).resolve()),
        "lease_policy": "caller-owned; harness acquires no lease",
        "state_mutation_policy": "validation only; no sudo or host/GPU state changes",
        "backend": "cupy",
        "scientific_seeds": [],
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "protocol": {
            "outer_pairs": OUTER_PAIRS,
            "chain_order": ["AB", "BA", "AB"],
            "forward_chain": [job["cell"] for job in plan[:4]],
            "warmup_steps": WARMUP_STEPS,
            "rewarm_steps": REWARM_STEPS,
            "timing_blocks": TIMING_BLOCKS,
            "steps_per_block": STEPS_PER_BLOCK,
            "persistent_source_isolated_workers": True,
            "json_line_protocol": True,
            "worker_cpu": WORKER_CPU,
            "controller_cpu": CONTROLLER_CPU,
            "required_nice": REQUIRED_NICE,
            "required_scheduler": _scheduler_name(REQUIRED_SCHEDULER),
            "gpu_controls": {
                "clock_mhz": REQUIRED_GPU_CLOCK_MHZ,
                "clock_tolerance_mhz": GPU_CLOCK_TOLERANCE_MHZ,
                "memory_clock_mhz": REQUIRED_MEMORY_CLOCK_MHZ,
                "power_limit_w": REQUIRED_POWER_LIMIT_W,
            },
            "dispersion_limits": {
                "max_inner_block_relative_range": MAX_INNER_BLOCK_RELATIVE_RANGE,
                "max_paired_ratio_relative_range": MAX_PAIRED_RATIO_RELATIVE_RANGE,
            },
        },
        "run_plan": plan,
        "source_roots": source_roots,
        "results": [],
        "summary": summarize([]),
        "workers": {},
        "worker_shutdown": [],
    }
    preconditions, initial_telemetry = _preconditions(
        source_roots=source_roots, controller_state=controller_state
    )
    result["preconditions"] = preconditions
    result["initial_telemetry"] = initial_telemetry
    if not all(check["ok"] for check in preconditions):
        result["status"] = "infrastructure_invalid"
        result["summary"]["performance_status"] = "infrastructure_invalid"
        write_json_atomic(output, result)
        return result

    write_json_atomic(output, result)
    workers: dict[str, PersistentWorker] = {}
    try:
        workers["candidate"] = PersistentWorker(
            source_root=candidate_root, source_kind="candidate"
        )
        workers["prechange-control"] = PersistentWorker(
            source_root=control_root, source_kind="prechange-control"
        )
        result["workers"] = {
            kind: worker.startup for kind, worker in workers.items()
        }
        if not all(worker.startup.get("status") == "ready" for worker in workers.values()):
            result["status"] = "infrastructure_invalid"
            result["summary"]["performance_status"] = "infrastructure_invalid"
            write_json_atomic(output, result)
            return result
        write_json_atomic(output, result)

        for job in plan:
            before = gpu_telemetry()
            before_check = validate_gpu_telemetry(
                before, require_active_clocks=False
            )
            row: dict[str, Any] = {
                **job,
                "status": "dispatching",
                "telemetry_before": before,
                "telemetry_before_validation": before_check,
            }
            result["results"].append(row)
            write_json_atomic(output, result)
            if not before_check["passed"]:
                row["status"] = "infrastructure_invalid"
                result["status"] = "infrastructure_invalid"
                result["summary"]["performance_status"] = "infrastructure_invalid"
                write_json_atomic(output, result)
                return result

            worker_key = "candidate" if job["source"] == "candidate" else "prechange-control"
            response = workers[worker_key].observe(job)
            after = gpu_telemetry()
            after_check = validate_gpu_telemetry(
                after, require_active_clocks=False
            )
            row.update(response)
            row.update({key: job[key] for key in job})
            row["telemetry_after"] = after
            row["telemetry_after_validation"] = after_check
            worker_clock_valid = (
                response.get("warmup_telemetry_validation", {}).get("passed") is True
                and response.get("post_timing_telemetry_validation", {}).get("passed") is True
            )
            if (response.get("status") != "completed" or not after_check["passed"]
                    or not worker_clock_valid):
                row["status"] = "infrastructure_invalid"
                row["worker_status"] = response.get("status")
                result["status"] = "infrastructure_invalid"
                result["summary"] = summarize(result["results"])
                result["summary"]["performance_status"] = "infrastructure_invalid"
                write_json_atomic(output, result)
                return result
            result["summary"] = summarize(result["results"])
            write_json_atomic(output, result)

        result["status"] = (
            "complete"
            if result["summary"]["performance_status"] in {"GO", "NO_GO"}
            else "infrastructure_invalid"
        )
        write_json_atomic(output, result)
        return result
    except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
        result["status"] = "infrastructure_invalid"
        result["failure"] = f"{type(exc).__name__}: {exc}"
        result["summary"] = summarize(result["results"])
        result["summary"]["performance_status"] = "infrastructure_invalid"
        write_json_atomic(output, result)
        return result
    finally:
        for worker in reversed(list(workers.values())):
            try:
                result["worker_shutdown"].append(worker.close())
            except BaseException as exc:
                result["worker_shutdown"].append({
                    "source_kind": worker.source_kind,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        write_json_atomic(output, result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=ROOT)
    parser.add_argument("--prechange-control-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--persistent-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--source-kind", choices=("candidate", "prechange-control"), help=argparse.SUPPRESS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.persistent_worker:
        if args.source_root is None or args.source_kind is None:
            raise SystemExit("persistent worker requires --source-root and --source-kind")
        return persistent_worker(source_root=args.source_root, source_kind=args.source_kind)
    if args.prechange_control_root is None or args.out is None:
        raise SystemExit("controller requires --prechange-control-root and --out")
    result = run_matrix(
        candidate_root=args.candidate_root,
        control_root=args.prechange_control_root,
        output=args.out,
    )
    print(json.dumps({"status": result["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if result["status"] == "complete" else 75


if __name__ == "__main__":
    raise SystemExit(main())
