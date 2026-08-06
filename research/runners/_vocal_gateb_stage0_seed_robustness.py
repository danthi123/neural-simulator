"""Gate B Stage-0 tonic-output substrate: multi-seed robustness smoke.

Reuses the VALIDATED physiology primitives from the locked v13 tonic-output
runner (build_tonic_bridge / _run_steps / _physiology_metrics) and sweeps fresh
seeds at the selected 100 pA intrinsic drive. This is an ENGINEERING robustness
confirmation of the load-bearing Stage-0 phenotype (GPi/SNr fires 40-80 Hz
tonically at zero external current, all cells fire, intrinsic + weights
immutable) that the sealed harness only exercised at ~1 held-out seed. It is
NOT a formal capability-seed partition and consumes no sealed seed.

Run:
  SIM_BACKEND=numpy python -m research.runners._vocal_gateb_stage0_seed_robustness \
      --seeds 810001-810012 --out research/findings/raw/gateb_stage0_seed_robustness/numpy.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from research.runners._vocal_action_credit_gate_v13_tonic_output import (
    CRITERIA,
    build_tonic_bridge,
    _backend_info,
    _git_sha,
    _physiology_metrics,
    _population_audit,
    _run_steps,
)

SELECTED_CURRENT_PA = 100.0
STEPS = 1000
BIN_STEPS = 100
N = 40


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="810001-810012")
    parser.add_argument("--current-pA", type=float, default=SELECTED_CURRENT_PA)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    backend = _backend_info()
    started = time.perf_counter()
    rows = []
    for seed in seeds:
        bridge = build_tonic_bridge(seed, args.current_pA, n=N)
        audit = _population_audit(bridge)
        run = _run_steps(bridge, STEPS)
        phys = _physiology_metrics(run, n=N, bin_steps=BIN_STEPS)
        rows.append({
            "seed": seed,
            "population_rate_hz": phys["metrics"]["population_rate_hz"],
            "cells_firing": phys["metrics"]["cells_firing"],
            "min_bin_hz": min(phys["metrics"]["bin_rates_hz"]) if phys["metrics"]["bin_rates_hz"] else None,
            "max_bin_hz": max(phys["metrics"]["bin_rates_hz"]) if phys["metrics"]["bin_rates_hz"] else None,
            "audit_pass": bool(audit["pass"]),
            "physiology_pass": bool(phys["pass"]),
            "failed_checks": [k for k, v in phys["checks"].items() if not v],
            "pass": bool(audit["pass"] and phys["pass"]),
        })
        bridge.clear_simulation_state_and_gpu_memory()

    n_pass = sum(1 for r in rows if r["pass"])
    result = {
        "probe": "gateB_stage0_seed_robustness",
        "stage": "engineering_robustness_smoke",
        "note": "not a formal capability partition; confirms Stage-0 tonic-output phenotype across fresh seeds",
        "backend": backend["backend"],
        "device": backend["device"],
        "source_sha": _git_sha(),
        "current_pA": args.current_pA,
        "steps": STEPS,
        "criteria": CRITERIA,
        "n_seeds": len(seeds),
        "n_pass": n_pass,
        "all_pass": n_pass == len(seeds),
        "rows": rows,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "backend": result["backend"], "n_pass": n_pass, "n_seeds": len(seeds),
        "all_pass": result["all_pass"], "out": str(out),
    }, indent=2))
    return 0 if result["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
