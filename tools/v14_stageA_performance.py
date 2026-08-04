#!/usr/bin/env python3
"""Run the seed-free V14 Stage-A CuPy performance matrix.

The caller owns the GPU lease. This process never acquires or releases one.
Each matrix cell runs in a fresh subprocess with an explicit SIM_SOURCE_ROOT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "research/specs/v14_snr_conductance_stageA_implementation.json"
PROCESS_ORDER_SEED = 20260804
WARMUP_STEPS = 500
TIMED_STEPS = 2000
REPETITIONS = 3
NUM_NEURONS = 600
ACTIVE_BYTES_PER_NEURON = 48
DEFAULT_RATIO_MAX = 1.02
ACTIVE_RATIO_MAX = 1.25
CONSTRUCTION_RNG_SEED = 0
WORKER_TIMEOUT_SECONDS = 1800
PROJECTED_TOTAL_SECONDS = 4800

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
    "candidate-default": {"source": "candidate", "active": False},
    "prechange-control-default": {"source": "prechange-control", "active": False},
    "candidate-active": {"source": "candidate", "active": True},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def source_snapshot(root: Path) -> dict:
    root = root.resolve()
    rels = ("sim/bridge.py", "sim/config.py", "sim/kernels.py", "sim/regions.py")
    return {
        "root": str(root),
        "revision": _git(root, "rev-parse", "HEAD"),
        "status_porcelain": _git(root, "status", "--porcelain", "--", *rels),
        "files": {
            rel: _sha256(root / rel) if (root / rel).is_file() else None
            for rel in rels
        },
    }


def build_run_plan() -> list[dict]:
    jobs = [
        {"cell": cell, "rep": rep, **definition}
        for rep in range(1, REPETITIONS + 1)
        for cell, definition in CELL_DEFINITIONS.items()
    ]
    random.Random(PROCESS_ORDER_SEED).shuffle(jobs)
    for sequence, job in enumerate(jobs, 1):
        job["sequence"] = sequence
    return jobs


def _nvidia_smi() -> dict:
    query = (
        "name,uuid,pci.bus_id,driver_version,pstate,temperature.gpu,clocks.sm,"
        "power.draw,utilization.gpu,memory.used,memory.total"
    )
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "query": query,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "captured_at_unix": time.time(),
    }


def _build_bridge(*, active: bool):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
    from sim.enums import NeuronModel
    from sim.regions import BrainRegion

    region_args = {}
    if active:
        # Numerical Stage-A values only; these are not adult SNr parameters.
        region_args = {
            "snr_g_nalcn_max": 0.01,
            "snr_g_nap_max": 0.02,
            "snr_g_ca_max": 0.03,
            "snr_g_sk_max": 0.04,
            "snr_g_h_max": 0.005,
        }
    region = BrainRegion(
        name="performance_population",
        n_neurons=NUM_NEURONS,
        internal_density=0.0,
        **region_args,
    )
    config = CoreSimConfig(
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
    return bridge


def worker(*, source_root: Path, cell: str) -> dict:
    source_root = source_root.resolve()
    if Path(os.environ.get("SIM_SOURCE_ROOT", "")).resolve() != source_root:
        raise RuntimeError("SIM_SOURCE_ROOT must explicitly match --source-root")
    os.environ["SIM_BACKEND"] = "cupy"
    sys.path.insert(0, str(source_root))
    import cupy as cp

    definition = CELL_DEFINITIONS[cell]
    if definition["active"] and definition["source"] != "candidate":
        raise RuntimeError("active cell is candidate-only")
    bridge = _build_bridge(active=definition["active"])
    try:
        arrays = [getattr(bridge, name, None) for name in BUNDLE_ARRAYS]
        attributes_present = [hasattr(bridge, name) for name in BUNDLE_ARRAYS]
        bundle_bytes = sum(int(array.nbytes) for array in arrays if array is not None)
        default_none = all(array is None for array in arrays)
        active_bytes_exact = bundle_bytes == ACTIVE_BYTES_PER_NEURON * NUM_NEURONS
        structural_ok = (
            active_bytes_exact and all(array is not None for array in arrays)
            if definition["active"]
            else default_none
        )
        for _ in range(WARMUP_STEPS):
            bridge._run_one_simulation_step()
        cp.cuda.runtime.deviceSynchronize()
        telemetry_before = _nvidia_smi()
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record()
        host_started = time.perf_counter()
        for _ in range(TIMED_STEPS):
            bridge._run_one_simulation_step()
        end_event.record()
        end_event.synchronize()
        host_seconds = time.perf_counter() - host_started
        cuda_seconds = float(cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0)
        telemetry_after = _nvidia_smi()
        properties = cp.cuda.runtime.getDeviceProperties(0)
        device_name = properties.get("name", "unknown")
        if isinstance(device_name, bytes):
            device_name = device_name.decode("utf-8", errors="replace")
        return {
            "schema": "v14-stageA-performance-worker-v1",
            "status": "completed" if structural_ok else "failed",
            "cell": cell,
            "source": source_snapshot(source_root),
            "backend": "cupy",
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "hostname": platform.node(),
                "cupy": cp.__version__,
                "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
                "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
                "device": str(device_name),
            },
            "configuration": {
                "num_neurons": NUM_NEURONS,
                "dt_ms": 0.05,
                "warmup_steps": WARMUP_STEPS,
                "timed_steps": TIMED_STEPS,
                "construction_rng_seed": CONSTRUCTION_RNG_SEED,
                "scientific_seeds": [],
                "active": definition["active"],
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
                "host_seconds": float(host_seconds),
                "cuda_event_seconds": cuda_seconds,
            },
            "nvidia_smi": {"before": telemetry_before, "after": telemetry_after},
        }
    finally:
        bridge.clear_simulation_state_and_gpu_memory()


def run_worker(job: dict, *, candidate_root: Path, control_root: Path) -> dict:
    source_root = candidate_root if job["source"] == "candidate" else control_root
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--source-root",
        str(source_root.resolve()),
        "--cell",
        job["cell"],
    ]
    env = {**os.environ, "SIM_BACKEND": "cupy", "SIM_SOURCE_ROOT": str(source_root.resolve())}
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=source_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=WORKER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "schema": "v14-stageA-performance-worker-v1",
            "status": "infrastructure_failure",
            "failure": "worker_timeout",
            "timeout_seconds": WORKER_TIMEOUT_SECONDS,
            "elapsed_seconds": time.time() - started,
            "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "sequence": job["sequence"],
            "rep": job["rep"],
            "cell": job["cell"],
            "source": source_snapshot(source_root),
        }
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        payload = {
            "schema": "v14-stageA-performance-worker-v1",
            "status": "failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    payload.update(sequence=job["sequence"], rep=job["rep"])
    if completed.returncode != 0:
        payload["status"] = "failed"
        payload["returncode"] = completed.returncode
    return payload


def _median(rows: list[dict], cell: str, field: str) -> float | None:
    values = [
        float(row["timing"][field])
        for row in rows
        if row.get("cell") == cell and row.get("status") == "completed"
    ]
    return float(statistics.median(values)) if values else None


def summarize(rows: list[dict]) -> dict:
    cells = {}
    for cell in CELL_DEFINITIONS:
        selected = [row for row in rows if row.get("cell") == cell and row.get("status") == "completed"]
        cells[cell] = {
            "completed_repetitions": len(selected),
            "median_host_seconds": _median(rows, cell, "host_seconds"),
            "median_cuda_event_seconds": _median(rows, cell, "cuda_event_seconds"),
            "all_structural_checks_pass": len(selected) == REPETITIONS and all(
                row.get("structural", {}).get("passed") is True for row in selected
            ),
        }

    def ratio(numerator: str, denominator: str, field: str) -> float | None:
        a, b = cells[numerator][field], cells[denominator][field]
        return None if a is None or b in (None, 0.0) else float(a / b)

    ratios = {
        "default_host": ratio("candidate-default", "prechange-control-default", "median_host_seconds"),
        "default_cuda_event": ratio("candidate-default", "prechange-control-default", "median_cuda_event_seconds"),
        "active_host": ratio("candidate-active", "candidate-default", "median_host_seconds"),
        "active_cuda_event": ratio("candidate-active", "candidate-default", "median_cuda_event_seconds"),
    }
    complete = all(cell["completed_repetitions"] == REPETITIONS for cell in cells.values())
    structural = all(cell["all_structural_checks_pass"] for cell in cells.values())
    thresholds_pass = complete and all(value is not None for value in ratios.values()) and (
        ratios["default_host"] <= DEFAULT_RATIO_MAX
        and ratios["default_cuda_event"] <= DEFAULT_RATIO_MAX
        and ratios["active_host"] <= ACTIVE_RATIO_MAX
        and ratios["active_cuda_event"] <= ACTIVE_RATIO_MAX
    )
    return {
        "cells": cells,
        "ratios_against_matching_medians": ratios,
        "fixed_thresholds": {
            "default_off_ratio_max": DEFAULT_RATIO_MAX,
            "active_ratio_max": ACTIVE_RATIO_MAX,
        },
        "performance_status": "GO" if structural and thresholds_pass else "NO_GO",
        "physiology_verdict": None,
        "promotion_effect": "none",
    }


def write_json_atomic(path: Path, value: dict) -> None:
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


def run_matrix(*, candidate_root: Path, control_root: Path, output: Path) -> dict:
    plan = build_run_plan()
    result = {
        "schema": "v14-stageA-performance-result-v1",
        "status": "running",
        "created_at_unix": time.time(),
        "specification": str(SPEC_PATH.relative_to(ROOT)),
        "specification_sha256": _sha256(SPEC_PATH),
        "harness_sha256": _sha256(Path(__file__).resolve()),
        "lease_policy": "caller-owned; harness acquires no lease",
        "backend": "cupy",
        "scientific_seeds": [],
        "process_order_seed": PROCESS_ORDER_SEED,
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "projected_total_seconds": PROJECTED_TOTAL_SECONDS,
        "run_plan": plan,
        "source_roots": {
            "candidate": source_snapshot(candidate_root),
            "prechange_control": source_snapshot(control_root),
        },
        "results": [],
        "summary": summarize([]),
    }
    write_json_atomic(output, result)
    for job in plan:
        row = run_worker(job, candidate_root=candidate_root, control_root=control_root)
        result["results"].append(row)
        result["summary"] = summarize(result["results"])
        if row.get("status") != "completed":
            result["status"] = "infrastructure_failure"
            result["failed_sequence"] = job["sequence"]
            write_json_atomic(output, result)
            return result
        write_json_atomic(output, result)
    result["status"] = "complete"
    write_json_atomic(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=ROOT)
    parser.add_argument("--prechange-control-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cell", choices=tuple(CELL_DEFINITIONS), help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker:
        if args.source_root is None or args.cell is None:
            raise SystemExit("worker requires --source-root and --cell")
        print(json.dumps(worker(source_root=args.source_root, cell=args.cell), sort_keys=True))
        return 0
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
