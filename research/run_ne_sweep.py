"""Session E.1.5 — NE parameter sweep on the silent-motor-trap probe.

E.1's NE-with-(sens=120, threshold=0.4, window=2000ms) was net-negative
because NE fired during phase-1 reward variability, interfering with
argmax consolidation BEFORE the goal change had even happened.

This sweep explores tighter parameter regimes where NE fires only on
sustained error (i.e., truly during phase 2). 4 promising configs based
on the failure analysis in 2026-04-24-session-e1-neuromodulator-subsystem.md.

Each config x 3 seeds = 12 runs. Run 3 in parallel at a time on the 3090
(each capped at 8 GB VRAM). Total wall time ~30-40 min.

Each run is a separate subprocess so we get true parallelism with no
shared cupy state.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# (sensitivity, threshold, window_ms, label)
CONFIGS = [
    # E.1 baseline (for control / sanity-check)
    (120.0, 0.4, 2000.0, "baseline"),
    # Tighter threshold + longer window: NE fires only on sustained error
    (60.0, 0.6, 5000.0, "tight"),
    # Even tighter: should rarely fire at all
    (30.0, 0.7, 8000.0, "very_tight"),
    # Same threshold but stronger boost when it does fire (test if the issue
    # is threshold or magnitude)
    (90.0, 0.6, 5000.0, "tight_strong"),
]
SEEDS = [42, 43, 44]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=3,
                    help="Max concurrent subprocesses (default 3 on 24GB GPU)")
    args = ap.parse_args()

    out_dir = Path("research/findings/raw/g9_ne_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the full job list, skipping any that already exist
    jobs = []  # (sens, thr, win, label, seed, out_path)
    for (sens, thr, win, label) in CONFIGS:
        for seed in SEEDS:
            out_path = out_dir / f"ne_{label}_seed{seed}.json"
            if out_path.exists():
                print(f"  SKIP (already done): {out_path.name}")
                continue
            jobs.append((sens, thr, win, label, seed, out_path))

    if not jobs:
        print("Nothing to do; all outputs exist.")
        return

    # Memory budget per process (8 GB cap × 3 = 24 GB headroom)
    mem_pool_bytes = int(8 * 1024**3)

    print(f"\n  Launching {len(jobs)} runs with concurrency={args.concurrency}")
    print(f"  Each capped at {mem_pool_bytes/1024**3:.0f} GB VRAM")

    procs: list[subprocess.Popen] = []
    next_job = 0
    t0 = time.time()

    while next_job < len(jobs) or procs:
        # Reap finished
        still = []
        for p in procs:
            if p.poll() is None:
                still.append(p)
            else:
                p._log_handle.close()
                rc = p.returncode
                elapsed = time.time() - t0
                status = "OK" if rc == 0 else f"FAIL rc={rc}"
                print(f"  [{elapsed:5.0f}s] {p._tag} done ({status}) -> {p._out_path.name}")
        procs = still

        # Launch more
        while len(procs) < args.concurrency and next_job < len(jobs):
            sens, thr, win, label, seed, out_path = jobs[next_job]
            next_job += 1
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["CUPY_GPU_MEMORY_LIMIT"] = str(mem_pool_bytes)
            cmd = [
                sys.executable, "-c",
                f"""
import sys
sys.path.insert(0, '.')
from research.runners.g9_runner import run_g9_episode
from sim.neuromodulators import NeuromodulatorConfig, ProductionRule, ModulatorTarget

nm_configs = [
    NeuromodulatorConfig(
        name='dopamine', baseline=0.0, decay_tau_ms=500.0,
        production_rules=[ProductionRule(rule_type='from_reward', sensitivity=1.0)],
        targets=[],
    ),
    NeuromodulatorConfig(
        name='noradrenaline',
        baseline=0.05,
        decay_tau_ms=3000.0,
        concentration_min=0.0, concentration_max=2.0,
        production_rules=[ProductionRule(
            rule_type='from_error_persistence',
            sensitivity=1.0, threshold={thr}, window_ms={win},
        )],
        targets=[ModulatorTarget(
            target_type='excitability_drive',
            scope='group:motor', sensitivity={sens},
        )],
    ),
]
run_g9_episode(
    out_path={str(out_path)!r},
    seed={seed}, n_steps=1800,
    start_pos=(1, 1), goal_pos=(6, 6),
    goal_schedule=[(0, (6, 6)), (300, (1, 6))],
    learning_rate=0.01,
    reward_eligibility_tau_ms=500.0, reward_hold_steps=10,
    action_selection='argmax',
    nm_configs=nm_configs,
    verbose=True,
)
"""
            ]
            log_path = out_path.with_suffix(".stdout.txt")
            log_f = open(log_path, "w")
            p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            p._log_handle = log_f
            p._out_path = out_path
            p._tag = f"{label}_seed{seed}"
            procs.append(p)
            elapsed = time.time() - t0
            print(f"  [{elapsed:5.0f}s] launched {p._tag} (pid={p.pid})  "
                  f"sens={sens} thr={thr} win={win}")

        if procs:
            time.sleep(10.0)

    elapsed = time.time() - t0
    print(f"\n  ALL DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
