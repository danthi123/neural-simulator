#!/usr/bin/env python
"""Parallel multi-seed nav batch launcher — perf win #1 (2026-06-08).

EXACT speedup (separate OS processes -> separate CUDA contexts -> byte-identical math
with --deterministic; only WHEN a run executes changes, never WHAT it computes). Measured
~1.7x effective at 3 concurrent procs on this ~20K-neuron grid-8 net
(2026-05-04-perf-speedup-stack.md). Replaces the strict sequential `foreach` in the
de-risk launchers. See research/findings/2026-06-08-performance-audit.md #1.

SAFETY GUARDS baked in (from the audit + MEMORY's over-subscription trap):
  * parallelism HARD-CAPPED at 3 (the measured ceiling; parallelism=6 HALVED throughput).
  * each child inherits CUPY_GPU_MEMORY_LIMIT (default "30%") so N pools share VRAM
    cleanly instead of each grabbing the default 0.8 and colliding.
  * a child that fails does NOT abort the batch (its failure is reported; others finish).
  * NO sim/ edit, NO runner edit — launcher only.

VALIDATION before trusting a verdict from this (the audit's falsifiable check): run ONE
seed solo and the same seed in a pool, diff the output JSON's per-phase
final_quarter_mean_distance — must be byte-identical under --deterministic. If it differs,
something shares state across processes (it shouldn't — separate processes) -> STOP.

Spec JSON (pass via --spec):
{
  "common": ["--moving-goal", "--goal-schedule", "multi", "--deterministic", ...],
  "parallelism": 2,
  "cwd": "E:\\\\Documents\\\\Projects\\\\sim",        # tree whose runner code to import
  "cupy_mem_limit": "30%",                              # per-child VRAM cap
  "runs": [
     {"label": "neural_s42", "args": ["--seed","42","--perceived-approach-reward","--spiking-snc"],
      "out": "E:\\\\...\\\\_biorda_neural_s42.json"},
     ...
  ]
}
"""
import argparse
import json
import os
import subprocess
import sys
import time

HARD_CAP = 3  # never exceed — over-subscription is a net SLOWDOWN on this net.


def run_batch(spec):
    common = list(spec.get("common", []))
    runs = spec["runs"]
    par = max(1, min(HARD_CAP, int(spec.get("parallelism", 2))))
    cwd = spec.get("cwd", os.getcwd())
    mem_limit = str(spec.get("cupy_mem_limit", "30%"))

    base_env = dict(os.environ)
    base_env["CUPY_GPU_MEMORY_LIMIT"] = mem_limit

    print(f"[parallel-batch] {len(runs)} runs, parallelism={par} (cap {HARD_CAP}), "
          f"cwd={cwd}, CUPY_GPU_MEMORY_LIMIT={mem_limit}\n", flush=True)

    pending = list(enumerate(runs))
    active = {}          # proc -> (label, start_time, idx)
    results = {}         # idx -> (label, returncode, elapsed)
    t0 = time.perf_counter()

    def launch(idx, run):
        label = run.get("label", f"run{idx}")
        cmd = [sys.executable, "-m", "research.runners.g11_bg_runner",
               *common, *run["args"], "--out", run["out"]]
        log_path = run.get("log")
        log_fh = open(log_path, "w", encoding="utf-8") if log_path else subprocess.DEVNULL
        p = subprocess.Popen(cmd, cwd=cwd, env=base_env,
                             stdout=log_fh, stderr=subprocess.STDOUT)
        active[p] = (label, time.perf_counter(), idx, log_fh)
        print(f"[parallel-batch] START  {label}  (pid {p.pid})", flush=True)

    while pending or active:
        while pending and len(active) < par:
            idx, run = pending.pop(0)
            launch(idx, run)
        # poll
        done = [p for p in active if p.poll() is not None]
        if not done:
            time.sleep(2.0)
            continue
        for p in done:
            label, st, idx, log_fh = active.pop(p)
            if log_fh not in (subprocess.DEVNULL, None):
                try:
                    log_fh.close()
                except OSError:
                    pass
            elapsed = time.perf_counter() - st
            results[idx] = (label, p.returncode, elapsed)
            status = "OK" if p.returncode == 0 else f"FAIL(rc={p.returncode})"
            print(f"[parallel-batch] DONE   {label}  {status}  {elapsed/60:.1f} min", flush=True)

    wall = time.perf_counter() - t0
    n_ok = sum(1 for (_, rc, _) in results.values() if rc == 0)
    print(f"\n[parallel-batch] BATCH COMPLETE — {n_ok}/{len(runs)} OK, "
          f"wall {wall/60:.1f} min (vs ~{sum(e for _,_,e in results.values())/60:.1f} min sequential)",
          flush=True)
    return all(rc == 0 for (_, rc, _) in results.values())


def main():
    ap = argparse.ArgumentParser(description="Parallel multi-seed nav batch (perf win #1).")
    ap.add_argument("--spec", required=True, help="path to the batch spec JSON")
    args = ap.parse_args()
    with open(args.spec, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    ok = run_batch(spec)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
