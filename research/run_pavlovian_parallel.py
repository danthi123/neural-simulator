"""Session D Part C — parallelized Pavlovian/RL probe driver.

Route A from the GPU-utilization conversation: launches multiple
experiment runs concurrently instead of serially. The GPU time-slices
between them, and since each individual process is CPU/sync-bound (not
GPU-compute-bound), total wall time drops roughly 1/N-ish on an idle
GPU with N concurrent processes.

Memory safety: each process is passed CUPY_MEMORY_POOL_LIMIT so the
6 concurrent 10k-neuron processes can fit in a 24GB 3090 without
each one grabbing the default 80% and OOMing everyone else.

Usage:
    # Re-run everything in parallel from scratch:
    python research/run_pavlovian_parallel.py --experiments associative,reinforcement \
        --num-trials 300 --seeds 42,43,44 --concurrency 6

    # Run only missing outputs (what I used after the serial probe was killed):
    python research/run_pavlovian_parallel.py --experiments reinforcement \
        --num-trials 300 --seeds 43,44 --concurrency 2
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_one(experiment: str, seed: int, num_trials: int, out_dir: Path,
            mem_pool_fraction: float) -> subprocess.Popen:
    """Launch a single experiment run as a non-blocking subprocess."""
    out_path = out_dir / f"pavlovian_{experiment}_seed{seed}_n{num_trials}.json"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # CuPy honors CUPY_GPU_MEMORY_LIMIT (in bytes) to cap the memory pool.
    # Calculate from fraction of reported 24GB 3090.
    env["CUPY_GPU_MEMORY_LIMIT"] = str(int(24 * 1024**3 * mem_pool_fraction))
    cmd = [
        sys.executable, "run_experiment_headless.py",
        "--experiment", experiment,
        "--num-trials", str(num_trials),
        "--num-neurons", "10000",
        "--output", str(out_path),
        "--seed", str(seed),
    ]
    log_path = out_dir / f"pavlovian_{experiment}_seed{seed}_n{num_trials}.stdout.txt"
    log_f = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    p._log_handle = log_f  # keep alive so buffer doesn't get GC'd
    p._out_path = str(out_path)
    p._log_path = str(log_path)
    p._tag = f"{experiment}_seed{seed}"
    return p


def wait_all(procs: list[subprocess.Popen], poll_every: float = 5.0):
    """Poll until all subprocesses finish, printing status updates."""
    remaining = list(procs)
    start = time.time()
    while remaining:
        time.sleep(poll_every)
        still_running = []
        for p in remaining:
            rc = p.poll()
            if rc is None:
                still_running.append(p)
            else:
                p._log_handle.close()
                elapsed = time.time() - start
                status = "OK" if rc == 0 else f"FAIL rc={rc}"
                print(f"  [{elapsed:5.0f}s] {p._tag} finished ({status})  "
                      f"output={p._out_path}", flush=True)
        remaining = still_running
        if remaining:
            elapsed = time.time() - start
            print(f"  [{elapsed:5.0f}s] {len(remaining)} still running: "
                  f"{[p._tag for p in remaining]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default="associative,reinforcement")
    ap.add_argument("--num-trials", type=int, default=300)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--concurrency", type=int, default=6,
                    help="Max concurrent runs (default 6; reduce if OOM).")
    args = ap.parse_args()

    out_dir = Path("research/findings/raw/pavlovian")
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]

    # Build the full (experiment, seed) job list, skip jobs whose output already exists.
    jobs = []
    for e in experiments:
        for s in seeds:
            out_path = out_dir / f"pavlovian_{e}_seed{s}_n{args.num_trials}.json"
            if out_path.exists():
                print(f"  SKIP (already done): {out_path.name}")
            else:
                jobs.append((e, s))

    if not jobs:
        print("Nothing to do; all outputs exist.")
        return

    print(f"\n  Launching {len(jobs)} runs with concurrency={args.concurrency}")
    # Memory fraction per process — concurrent processes should share VRAM safely
    mem_pool_fraction = max(0.08, 0.8 / max(args.concurrency, 1))
    print(f"  Each process capped at {mem_pool_fraction*24:.1f} GB ({mem_pool_fraction*100:.0f}% of 24GB)")
    print(f"  Target: {args.num_trials} trials per run, 10000 neurons\n")

    t0 = time.time()
    procs: list[subprocess.Popen] = []
    next_job = 0

    while next_job < len(jobs) or procs:
        # Reap any finished
        still = []
        for p in procs:
            if p.poll() is None:
                still.append(p)
            else:
                p._log_handle.close()
                elapsed = time.time() - t0
                rc = p.returncode
                status = "OK" if rc == 0 else f"FAIL rc={rc}"
                print(f"  [{elapsed:5.0f}s] {p._tag} done ({status}) "
                      f"-> {Path(p._out_path).name}", flush=True)
        procs = still
        # Launch more if we have room
        while len(procs) < args.concurrency and next_job < len(jobs):
            e, s = jobs[next_job]
            p = run_one(e, s, args.num_trials, out_dir, mem_pool_fraction)
            procs.append(p)
            next_job += 1
            elapsed = time.time() - t0
            print(f"  [{elapsed:5.0f}s] launched {p._tag} (pid={p.pid}, "
                  f"{len(procs)}/{args.concurrency} slots used)", flush=True)
        if procs:
            time.sleep(5.0)

    elapsed = time.time() - t0
    print(f"\n  ALL DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
