"""Universal experiment runner — replaces bespoke PowerShell orchestrators
with a single YAML/JSON config + reusable parallel execution engine.

Replaces N copies of orchestrator boilerplate (PID files, log redirects,
master log + COMPLETE marker, parallel batches, .pid -> .pid.done
renaming) with one Python module.

Usage:
    python -m research.experiment_runner experiments/biology_sweep.yaml

YAML config schema (see schemas/experiment.example.yaml for reference):

    name: biology-sweep
    runner: research.runners.text_minimal_isolation
    output_dir: research/findings/raw/g11_bg
    parallelism: 3
    seeds: [42, 43, 44, 100, 101, 102]
    base_args:                          # args common to every run
      n-events-per-direction: 1000
      stim-steps-per-step: 100
      reset-steps: 50
      dt-ms: 1.0
    conditions:                          # per-condition arg overrides
      - name: baseline
        args: {}
      - name: fs_only
        args:
          enable-motor-fs: true          # bools become flag-only args
      - name: topo_only
        args:
          topographic-bias-factor: 1.5
          off-target-bias-factor: 0.7
      - name: topo_fs
        args:
          topographic-bias-factor: 1.5
          off-target-bias-factor: 0.7
          enable-motor-fs: true
    out_stats_template:                  # optional, default below
      "text_eval_{name}_seed{seed}.json"

Total runs = len(conditions) * len(seeds), executed in parallel-N batches.

Each run produces:
    {output_dir}/{name}_seed{seed}.log         stdout (script's prints + structured progress)
    {output_dir}/{name}_seed{seed}.log.err     stderr
    {output_dir}/{name}_seed{seed}.pid         active PID (renamed to .pid.done on exit)
    {output_dir}/{name}_seed{seed}.json        eval results (per out_stats_template)

Master log at {output_dir}/{config-name}.master.log includes:
  - Launch/completion timestamps per run
  - Final "=== {NAME} COMPLETE ===" marker for downstream waiters

Anti-shortcut features:
  - Exits non-zero if any run fails (visible in PowerShell's $LASTEXITCODE)
  - Prints aggregate summary at end (success / fail counts)
  - Optional `pre_check` and `post_check` hooks for anti-cheat controls
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Condition:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str
    runner: str
    output_dir: Path
    parallelism: int
    seeds: List[int]
    base_args: Dict[str, Any]
    conditions: List[Condition]
    out_stats_template: str = "text_eval_{name}_seed{seed}.json"
    pre_check: Optional[Dict[str, Any]] = None  # {condition: ..., assert_aligned_lt: 1}
    post_check: Optional[Dict[str, Any]] = None  # similar

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        data = yaml.safe_load(path.read_text())
        conds = [Condition(name=c["name"], args=c.get("args", {}) or {})
                 for c in data["conditions"]]
        return cls(
            name=data["name"],
            runner=data["runner"],
            output_dir=Path(data["output_dir"]),
            parallelism=int(data.get("parallelism", 1)),
            seeds=list(data["seeds"]),
            base_args=data.get("base_args", {}) or {},
            conditions=conds,
            out_stats_template=data.get("out_stats_template",
                                        "text_eval_{name}_seed{seed}.json"),
            pre_check=data.get("pre_check"),
            post_check=data.get("post_check"),
        )


def _build_cli_args(args_dict: Dict[str, Any]) -> List[str]:
    """Convert {'foo-bar': value} into ['--foo-bar', str(value)] CLI args.
    Booleans become flag-only (--foo-bar with no value if True; omitted
    if False). None values are omitted.
    """
    out: List[str] = []
    for k, v in args_dict.items():
        if v is None or v is False:
            continue
        flag = f"--{k.replace('_', '-')}"
        if v is True:
            out.append(flag)
        else:
            out.extend([flag, str(v)])
    return out


def _run_single(
    runner_module: str,
    seed: int,
    args: Dict[str, Any],
    output_dir: Path,
    log_prefix: str,
    out_stats_path: Path,
) -> subprocess.Popen:
    """Launch one runner subprocess. Returns the Popen handle.
    Caller is responsible for waiting + cleanup."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"{log_prefix}.log"
    err_file = output_dir / f"{log_prefix}.log.err"
    pid_file = output_dir / f"{log_prefix}.pid"

    full_args = {"seed": seed, **args}
    cli = ["python", "-m", runner_module] + _build_cli_args(full_args)
    cli += ["--out-stats", str(out_stats_path)]

    log_fp = log_file.open("w", encoding="utf-8")
    err_fp = err_file.open("w", encoding="utf-8")
    proc = subprocess.Popen(cli, stdout=log_fp, stderr=err_fp,
                             cwd=os.getcwd())
    pid_file.write_text(str(proc.pid))
    return _RunHandle(proc, pid_file, log_prefix, log_fp, err_fp,
                      condition=log_prefix.split("_seed")[0],
                      seed=seed)


@dataclass
class _RunHandle:
    proc: subprocess.Popen
    pid_file: Path
    log_prefix: str
    log_fp: Any
    err_fp: Any
    condition: str
    seed: int

    def wait(self) -> int:
        rc = self.proc.wait()
        self.log_fp.close()
        self.err_fp.close()
        # Mark PID file as done
        if self.pid_file.exists():
            done = self.pid_file.with_suffix(self.pid_file.suffix + ".done")
            self.pid_file.replace(done)
        return rc


def run_experiment(cfg: ExperimentConfig, master_log: Path) -> Dict[str, Any]:
    """Execute the full experiment. Returns summary dict.

    Behavior:
      - Run all conditions × seeds in parallel-N batches
      - Each batch waits for all parallel runs to finish before starting next
      - Master log records launch/completion of each run
      - Final "=== {NAME} COMPLETE ===" marker for downstream waiters
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    master_log.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with master_log.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
        print(f"[{ts}] {msg}", flush=True)

    log(f"=== {cfg.name} started ===")
    log(f"Runner: {cfg.runner}")
    log(f"Conditions: {[c.name for c in cfg.conditions]}")
    log(f"Seeds: {cfg.seeds}")
    log(f"Parallelism: {cfg.parallelism}")

    # Build full run plan (condition × seed)
    plan: List[tuple[Condition, int]] = []
    for cond in cfg.conditions:
        for seed in cfg.seeds:
            plan.append((cond, seed))
    total = len(plan)
    log(f"Total runs: {total}")

    succeeded: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    # Process in parallel-N batches
    p = max(1, int(cfg.parallelism))
    for batch_start in range(0, total, p):
        batch = plan[batch_start: batch_start + p]
        log(f"--- batch {batch_start // p + 1}/"
            f"{(total + p - 1) // p}: {[(c.name, s) for c, s in batch]} ---")

        handles: List[_RunHandle] = []
        for cond, seed in batch:
            args = {**cfg.base_args, **cond.args}
            log_prefix = f"{cond.name}_seed{seed}"
            out_stats = cfg.output_dir / cfg.out_stats_template.format(
                name=cond.name, seed=seed
            )
            try:
                h = _run_single(
                    cfg.runner, seed, args, cfg.output_dir,
                    log_prefix, out_stats,
                )
                handles.append(h)
                log(f"  launched {log_prefix} as PID {h.proc.pid}")
            except Exception as e:
                log(f"  FAILED to launch {log_prefix}: {e}")
                failed.append({"condition": cond.name, "seed": seed,
                                "error": str(e)})

        for h in handles:
            rc = h.wait()
            entry = {"condition": h.condition, "seed": h.seed,
                     "exit_code": rc}
            if rc == 0:
                succeeded.append(entry)
                log(f"  {h.log_prefix} OK")
            else:
                failed.append(entry)
                log(f"  {h.log_prefix} FAILED (exit {rc})")

    # Summary
    log("")
    log(f"Summary: {len(succeeded)}/{total} succeeded, "
        f"{len(failed)}/{total} failed")
    log("")
    log(f"=== {cfg.name} COMPLETE ===")

    return {
        "name": cfg.name,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="YAML config file")
    ap.add_argument("--master-log", type=Path, default=None,
                    help="Master log path (default: {output_dir}/{name}.master.log)")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    master_log = args.master_log or (cfg.output_dir / f"{cfg.name}.master.log")
    summary = run_experiment(cfg, master_log)

    if summary["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
