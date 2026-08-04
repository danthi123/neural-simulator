#!/usr/bin/env python3
"""Run V14 performance v3 with reversible host clock controls."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tools/v14_stageA_performance_v3.py"
CONTROLLER_CPU = 16
GPU_CLOCK_MHZ = 1800
MEMORY_CLOCK_MHZ = 9751
BENCHMARK_CPUS = (10, CONTROLLER_CPU)
HOST_HEAVY_LEASE = Path(os.environ.get("SIM_HOST_HEAVY_LEASE", "/tmp/sim-host-heavy.lock"))


def _run(command: list[str], *, check: bool = True, env: dict | None = None):
    return subprocess.run(command, check=check, text=True, env=env)


def _output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _project_workloads() -> list[str]:
    workloads = []
    for cmdline_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            values = [
                value.decode(errors="replace")
                for value in cmdline_path.read_bytes().split(b"\0")
                if value
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        is_rag = any(value.endswith("tools/rag/update_indexes.py") for value in values)
        is_pending_rag = "rag-refresh" in values
        is_runner = any("research/runners/" in value for value in values)
        is_module_runner = any(
            values[index] == "-m" and values[index + 1].startswith("research.runners")
            for index in range(len(values) - 1)
        )
        if is_rag or is_pending_rag or is_runner or is_module_runner:
            workloads.append(f"{cmdline_path.parent.name} {' '.join(values)}")
    return workloads


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=ROOT)
    parser.add_argument("--prechange-control-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not HARNESS.is_file():
        raise SystemExit(f"missing sealed harness: {HARNESS}")
    host_lease = HOST_HEAVY_LEASE.open("a+", encoding="utf-8")
    try:
        fcntl.flock(host_lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        host_lease.close()
        raise SystemExit("another host-heavy project workload owns the scheduling lease")
    workloads = _project_workloads()
    if workloads:
        host_lease.close()
        raise SystemExit("project workload already active:\n" + "\n".join(workloads))
    if _run(["sudo", "-n", "true"], check=False).returncode != 0:
        host_lease.close()
        raise SystemExit("passwordless sudo is required for reversible GPU clock locks")

    previous_profile = _output(["powerprofilesctl", "get"])
    governor_paths = [
        Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
        for cpu in BENCHMARK_CPUS
    ]
    previous_governors = [path.read_text(encoding="utf-8").strip() for path in governor_paths]
    prior_affinity = os.sched_getaffinity(0)
    interrupted = False

    def interrupt(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        raise InterruptedError("benchmark interrupted")

    old_handlers = {
        signum: signal.signal(signum, interrupt)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    gpu_clock_locked = False
    memory_clock_locked = False
    try:
        if previous_profile != "performance":
            _run(["powerprofilesctl", "set", "performance"])
        for path in governor_paths:
            _run([
                "sudo", "-n", "sh", "-c",
                f"printf %s performance > {path}",
            ])
        os.sched_setaffinity(0, {CONTROLLER_CPU})
        if os.sched_getaffinity(0) != {CONTROLLER_CPU}:
            raise RuntimeError("controller CPU affinity did not apply")
        _run([
            "sudo", "-n", "nvidia-smi", "-i", "0", "-lgc",
            f"{GPU_CLOCK_MHZ},{GPU_CLOCK_MHZ}",
        ])
        gpu_clock_locked = True
        _run([
            "sudo", "-n", "nvidia-smi", "-i", "0", "-lmc",
            f"{MEMORY_CLOCK_MHZ},{MEMORY_CLOCK_MHZ}",
        ])
        memory_clock_locked = True
        env = {
            **os.environ,
            "SIM_BENCHMARK_GPU_CLOCK_MHZ": str(GPU_CLOCK_MHZ),
            "SIM_BENCHMARK_MEMORY_CLOCK_MHZ": str(MEMORY_CLOCK_MHZ),
            "SIM_BENCHMARK_POWER_LIMIT_W": "300",
            "SIM_BENCHMARK_WORKER_CPU": "10",
            "SIM_BENCHMARK_CONTROLLER_CPU": str(CONTROLLER_CPU),
        }
        completed = _run([
            sys.executable,
            str(HARNESS),
            "--candidate-root", str(args.candidate_root.resolve()),
            "--prechange-control-root", str(args.prechange_control_root.resolve()),
            "--out", str(args.out.resolve()),
        ], check=False, env=env)
        return completed.returncode
    except InterruptedError:
        return 130 if interrupted else 75
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if gpu_clock_locked:
            _run(["sudo", "-n", "nvidia-smi", "-i", "0", "-rgc"], check=False)
        if memory_clock_locked:
            _run(["sudo", "-n", "nvidia-smi", "-i", "0", "-rmc"], check=False)
        os.sched_setaffinity(0, prior_affinity)
        if previous_profile != "performance":
            _run(["powerprofilesctl", "set", previous_profile], check=False)
        for path, governor in zip(governor_paths, previous_governors):
            _run([
                "sudo", "-n", "sh", "-c",
                f"printf %s {governor} > {path}",
            ], check=False)
        host_lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
