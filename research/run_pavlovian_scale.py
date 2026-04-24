"""Session D Part C: Pavlovian and R-STDP demonstrations at scale.

Exercises the existing ExperimentPresets.associative_conditioning and
ExperimentPresets.reinforcement_learning built-in presets across multiple
seeds and analyses against Rescorla-Wagner analytical predictions.

The goal is to produce a biology-canonical learning demonstration that
the sim handles cleanly, as a complement to the moving-goal saga. These
are the tasks the sim was designed for (per CLAUDE.md §Experiment &
Stimulus System) and have been probed only at small scale via smoke
tests.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_one(experiment: str, seed: int, num_trials: int, out_dir: Path):
    """Invoke run_experiment_headless.py for one config and return the log path."""
    out_path = out_dir / f"pavlovian_{experiment}_seed{seed}_n{num_trials}.json"
    print(f"\n{'='*70}")
    print(f"PAVLOVIAN SCALE: experiment={experiment} seed={seed} num_trials={num_trials}")
    print(f"{'='*70}")
    env = {"PYTHONUNBUFFERED": "1"}
    cmd = [
        sys.executable, "run_experiment_headless.py",
        "--experiment", experiment,
        "--num-trials", str(num_trials),
        "--num-neurons", "10000",
        "--output", str(out_path),
        "--seed", str(seed),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-2000:]}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default="associative,reinforcement",
                    help="comma-separated: associative, reinforcement, or both")
    ap.add_argument("--num-trials", type=int, default=300,
                    help="trials per experiment (default 300, up from preset default 100-200)")
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()

    out_dir = Path("research/findings/raw/pavlovian")
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]

    for exp in experiments:
        for seed in seeds:
            run_one(exp, seed, args.num_trials, out_dir)

    print(f"\n{'='*70}")
    print(f"DONE. Outputs in {out_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
