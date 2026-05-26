# research/findings/raw/direction_Q_tertiary_nmda_ratio_runner.py
"""Direction Q-tertiary NMDA-AMPA conductance-ratio sweep runner.

Tests whether the NMDA:AMPA conductance ratio (cfg.nmda_ratio) is the
constraint that prevents the Wang 2002 NMDA recurrent attractor from
self-sustaining at the best-signal cell from prior Direction Q sweeps:
n_dlpfc=1000, dlpfc_density=0.20, inh_weight_mean=2.0.

Pre-registered hypothesis (per the Direction Q-secondary findings doc):
the prior Q / Q-prime / Q-secondary bridges all used the CoreSimConfig
default nmda_ratio=0.4 (AMPA carries 60% of recurrent conductance).
Wang 2002 used NMDA-dominated AMPA-light ratios (~0.05–0.10 in some
figures) so that the slow NMDA tail (tau~100 ms) outweighs the fast
AMPA decay (~2–5 ms) and sustains the bistable attractor across the
delay period. Our sweep tests {0.4, 0.6, 0.8} — the opposite direction
(MORE NMDA, LESS AMPA leakage) — because at our current substrate the
AMPA decay rate appears to be the leak channel that drains the
recurrent attractor before NMDA can build it up. If any nmda_ratio
produces sustained_sec >= 3.0 multi-seed, that's a Q PASS candidate.
Otherwise the diagnosis is sharpened (NMDA-AMPA ratio is NOT the
constraint and the bottleneck is in neuron model kinetics).

Reuses validated infrastructure byte-unchanged where possible:
- direction_Q_bridge_builder.build_q_test_bridge (parameterized with
  nmda_ratio as of 2026-05-26; default 0.4 preserves prior behavior
  so all Q-prime / Q-secondary byte-identical reads remain valid)
- direction_Q_protocol.run_baseline_period / apply_cue_stimulus /
  measure_delay_period (frozen; unmodified)
- direction_Q_verdict.compute_verdict (frozen thresholds at
  ratio>=2.0, sustained>=3.0s, min_seeds_pass=3; module not imported
  reflectively, not re-wrapped, called as a black-box scorer)

Discipline (binding):
- Verdict thresholds frozen in direction_Q_verdict.py; NOT overridable
- Mandatory NMDA-off control runs per nmda_ratio value (no
  test-vs-control collapse; each ratio has its own control sweep)
- Standalone bridges built fresh per (seed, ratio, condition) combo
  (full reset; no state leakage across nmda_ratio values or conditions)
- No protected module modified; the bridge_builder modification adds
  ONE new keyword parameter with default 0.4 (default-preserving)
- Pre-registered next-action per nmda_ratio value documented in the
  findings doc rather than auto-applied
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.findings.raw.direction_Q_bridge_builder import build_q_test_bridge
from research.findings.raw.direction_Q_protocol import (
    run_baseline_period, apply_cue_stimulus, measure_delay_period,
)
from research.findings.raw.direction_Q_verdict import (
    compute_verdict,
    _Q_RATE_RATIO_MIN, _Q_DELAY_MIN_SEC, _Q_MIN_SEEDS_PASS,
)
from sim.backend import get_backend, is_gpu_backend


DEFAULT_OUT = os.path.join(
    _HERE, "direction_Q_tertiary_nmda_ratio_sweep.json"
)


def _parse_seeds(seeds_str: str) -> List[int]:
    """Parse "42,43,44" -> [42, 43, 44]."""
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def _parse_ratio_values(ratio_str: str) -> List[float]:
    """Parse "0.4,0.6,0.8" -> [0.4, 0.6, 0.8]."""
    return [float(s.strip()) for s in ratio_str.split(",") if s.strip()]


def _compute_sustained_sec(delay_rates: List[float],
                                baseline_rate: float,
                                ratio_min: float,
                                bin_ms: float) -> float:
    """Compute the total duration (sec) of elevated firing in the
    delay period. Same convention as the prior Q-prime / Q-secondary
    runners: count bins with rate >= ratio_min * baseline_rate (silent
    baseline uses 1.0 Hz absolute floor).
    """
    if not delay_rates:
        return 0.0
    if baseline_rate <= 0.0:
        threshold = 1.0
    else:
        threshold = ratio_min * baseline_rate
    n_elevated = sum(1 for r in delay_rates if r >= threshold)
    return float(n_elevated) * float(bin_ms) / 1000.0


def _run_one_condition_ratio(seed: int, n_dlpfc: int, dlpfc_density: float,
                                  inh_weight_mean: float, nmda_ratio: float,
                                  enable_nmda: bool,
                                  baseline_ms: float, cue_ms: float,
                                  cue_amplitude_pA: float,
                                  cue_fraction: float,
                                  delay_ms: float, bin_ms: float,
                                  verbose: bool = True) -> dict:
    """Run a single (seed, nmda_ratio, condition) combo through the
    Wang 2002 protocol. Returns a dict with baseline_rate, cue_rate,
    delay_rates, mean_delay_rate, rate_ratio, sustained_sec,
    wall_seconds, and the nmda_ratio for traceability.

    Behavior identical to direction_Q_secondary_ei_balance_runner
    ._run_one_condition_ei except for the nmda_ratio pass-through.
    """
    label = "NMDA-ON" if enable_nmda else "NMDA-OFF (AMPA-only control)"
    if verbose:
        print(
            "[Q-RATIO] seed=" + str(seed) + " " + label
            + " nmda_ratio=" + str(nmda_ratio)
            + " inh_w=" + str(inh_weight_mean)
            + " n_dlpfc=" + str(n_dlpfc)
            + " density=" + str(dlpfc_density),
            flush=True,
        )
    t0 = time.time()
    bridge = build_q_test_bridge(
        seed=seed, n_dlpfc=n_dlpfc, dlpfc_density=dlpfc_density,
        enable_nmda=enable_nmda, inh_weight_mean=inh_weight_mean,
        nmda_ratio=nmda_ratio, verbose=verbose,
    )
    if verbose:
        print(
            "[Q-RATIO]   build " + str(round(time.time() - t0, 1)) + "s",
            flush=True,
        )

    t1 = time.time()
    baseline_rate = run_baseline_period(bridge, baseline_ms)
    if verbose:
        print(
            "[Q-RATIO]   baseline=" + str(round(baseline_rate, 3))
            + " Hz (" + str(round(time.time() - t1, 1)) + "s)",
            flush=True,
        )

    t2 = time.time()
    cue_rate = apply_cue_stimulus(
        bridge, cue_amplitude_pA=cue_amplitude_pA,
        duration_ms=cue_ms, cue_fraction=cue_fraction,
    )
    if verbose:
        print(
            "[Q-RATIO]   cue=" + str(round(cue_rate, 3))
            + " Hz (" + str(round(time.time() - t2, 1)) + "s)",
            flush=True,
        )

    t3 = time.time()
    delay_rates = measure_delay_period(
        bridge, duration_ms=delay_ms, bin_ms=bin_ms,
    )
    if verbose:
        n_bins = len(delay_rates)
        last3 = delay_rates[-3:] if n_bins >= 3 else delay_rates
        print(
            "[Q-RATIO]   delay=" + str(n_bins) + " bins ("
            + str(round(time.time() - t3, 1))
            + "s); final-3=" + str([round(r, 3) for r in last3]),
            flush=True,
        )

    if delay_rates:
        mean_delay_rate = float(sum(delay_rates)) / float(len(delay_rates))
    else:
        mean_delay_rate = 0.0

    if baseline_rate > 1e-9:
        rate_ratio = mean_delay_rate / baseline_rate
    elif mean_delay_rate > 1e-9:
        rate_ratio = float(mean_delay_rate) * 1e9
    else:
        rate_ratio = 0.0

    sustained_sec = _compute_sustained_sec(
        delay_rates=delay_rates,
        baseline_rate=baseline_rate,
        ratio_min=_Q_RATE_RATIO_MIN,
        bin_ms=bin_ms,
    )

    wall_seconds = time.time() - t0
    if verbose:
        print(
            "[Q-RATIO]   rate_ratio=" + str(round(rate_ratio, 2))
            + "  sustained_sec=" + str(round(sustained_sec, 3))
            + "  wall=" + str(round(wall_seconds, 1)) + "s",
            flush=True,
        )

    return {
        "seed": seed,
        "nmda_ratio": float(nmda_ratio),
        "inh_weight_mean": float(inh_weight_mean),
        "enable_nmda": enable_nmda,
        "baseline_rate": baseline_rate,
        "cue_rate": cue_rate,
        "delay_rates": delay_rates,
        "mean_delay_rate": mean_delay_rate,
        "rate_ratio": float(rate_ratio),
        "sustained_sec": float(sustained_sec),
        "wall_seconds": float(wall_seconds),
    }


def _sweep_one_ratio(nmda_ratio: float, seeds: List[int],
                          n_dlpfc: int, dlpfc_density: float,
                          inh_weight_mean: float,
                          baseline_ms: float, cue_ms: float,
                          cue_amplitude_pA: float, cue_fraction: float,
                          delay_ms: float, bin_ms: float) -> dict:
    """Run a full per-cell sweep at the given nmda_ratio:
    3 test seeds (NMDA-on) + 3 control seeds (NMDA-off) + verdict.
    Returns a dict with per_seed_data, control_per_seed_data, verdict.
    """
    print("\n" + "=" * 70, flush=True)
    print("=== NMDA-RATIO CELL nmda_ratio=" + str(nmda_ratio)
          + " (n=" + str(n_dlpfc)
          + " d=" + str(dlpfc_density)
          + " inh=" + str(inh_weight_mean) + ") ===", flush=True)
    print("=" * 70, flush=True)

    # ---- TEST condition: NMDA-on ----
    print("\n--- TEST CONDITION (NMDA-on) ---", flush=True)
    test_runs: List[dict] = []
    per_seed: List[Tuple[float, float]] = []
    for seed in seeds:
        run = _run_one_condition_ratio(
            seed=seed, n_dlpfc=n_dlpfc, dlpfc_density=dlpfc_density,
            inh_weight_mean=inh_weight_mean, nmda_ratio=nmda_ratio,
            enable_nmda=True,
            baseline_ms=baseline_ms, cue_ms=cue_ms,
            cue_amplitude_pA=cue_amplitude_pA, cue_fraction=cue_fraction,
            delay_ms=delay_ms, bin_ms=bin_ms,
        )
        test_runs.append(run)
        per_seed.append((run["rate_ratio"], run["sustained_sec"]))

    # ---- MANDATORY CONTROL: NMDA-off ----
    print("\n--- CONTROL CONDITION (NMDA-off / AMPA-only) ---", flush=True)
    control_runs: List[dict] = []
    control_per_seed: List[Tuple[float, float]] = []
    for seed in seeds:
        run = _run_one_condition_ratio(
            seed=seed, n_dlpfc=n_dlpfc, dlpfc_density=dlpfc_density,
            inh_weight_mean=inh_weight_mean, nmda_ratio=nmda_ratio,
            enable_nmda=False,
            baseline_ms=baseline_ms, cue_ms=cue_ms,
            cue_amplitude_pA=cue_amplitude_pA, cue_fraction=cue_fraction,
            delay_ms=delay_ms, bin_ms=bin_ms,
        )
        control_runs.append(run)
        control_per_seed.append((run["rate_ratio"], run["sustained_sec"]))

    # ---- Verdict (frozen thresholds; no tuning) ----
    verdict = compute_verdict(per_seed, control_per_seed)

    # Compute multi-seed aggregates for inline reporting
    test_ratios = [r["rate_ratio"] for r in test_runs]
    test_sustained = [r["sustained_sec"] for r in test_runs]
    ctrl_ratios = [r["rate_ratio"] for r in control_runs]
    ctrl_sustained = [r["sustained_sec"] for r in control_runs]

    print("\n--- CELL RESULT nmda_ratio=" + str(nmda_ratio)
          + " ---", flush=True)
    for r in test_runs:
        print(
            "  TEST    seed=" + str(r["seed"])
            + " baseline=" + str(round(r["baseline_rate"], 3))
            + "Hz  delay=" + str(round(r["mean_delay_rate"], 3))
            + "Hz  ratio=" + str(round(r["rate_ratio"], 2))
            + "  sustained=" + str(round(r["sustained_sec"], 2)) + "s",
            flush=True,
        )
    for r in control_runs:
        print(
            "  CONTROL seed=" + str(r["seed"])
            + " baseline=" + str(round(r["baseline_rate"], 3))
            + "Hz  delay=" + str(round(r["mean_delay_rate"], 3))
            + "Hz  ratio=" + str(round(r["rate_ratio"], 2))
            + "  sustained=" + str(round(r["sustained_sec"], 2)) + "s",
            flush=True,
        )
    print("  verdict: " + str(verdict), flush=True)

    return {
        "nmda_ratio": float(nmda_ratio),
        "inh_weight_mean": float(inh_weight_mean),
        "n_dlpfc": n_dlpfc,
        "dlpfc_density": dlpfc_density,
        "seeds": list(seeds),
        "per_seed_data": test_runs,
        "control_per_seed_data": control_runs,
        "per_seed_tuples": per_seed,
        "control_per_seed_tuples": control_per_seed,
        "verdict": verdict,
        "test_ratio_mean": (sum(test_ratios) / len(test_ratios)
                             if test_ratios else 0.0),
        "test_ratio_min": min(test_ratios) if test_ratios else 0.0,
        "test_ratio_max": max(test_ratios) if test_ratios else 0.0,
        "test_sustained_max": (max(test_sustained)
                                if test_sustained else 0.0),
        "test_sustained_mean": (sum(test_sustained) / len(test_sustained)
                                  if test_sustained else 0.0),
        "control_ratio_min": min(ctrl_ratios) if ctrl_ratios else 0.0,
        "control_ratio_max": max(ctrl_ratios) if ctrl_ratios else 0.0,
        "control_sustained_max": (max(ctrl_sustained)
                                    if ctrl_sustained else 0.0),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Direction Q-tertiary NMDA-AMPA conductance-ratio sweep "
            "(varies cfg.nmda_ratio across {0.4, 0.6, 0.8} by default "
            "at n=1000 d=0.20 inh=2.0; tests whether NMDA-AMPA ratio "
            "is the constraint preventing the Wang 2002 attractor "
            "from self-sustaining in the prior Q-secondary E/I "
            "envelope)."
        )
    )
    parser.add_argument("--nmda-ratios", type=str, default="0.4,0.6,0.8",
                        help="comma-separated nmda_ratio values")
    parser.add_argument("--n-dlpfc", type=int, default=1000,
                        help="dlpfc_wm region size (default 1000)")
    parser.add_argument("--dlpfc-density", type=float, default=0.20,
                        help="recurrent density (default 0.20, Wang)")
    parser.add_argument("--inh-weight-mean", type=float, default=2.0,
                        help="dlpfc_wm internal inhibitory weight "
                             "(default 2.0, lowest-inhibition cell "
                             "from Q-secondary)")
    parser.add_argument("--seeds", type=str, default="42,43,44",
                        help="comma-separated seed list")
    parser.add_argument("--baseline-ms", type=float, default=500.0,
                        help="baseline (pre-cue) duration in ms")
    parser.add_argument("--cue-ms", type=float, default=500.0,
                        help="cue stimulus duration in ms")
    parser.add_argument("--cue-amplitude-pA", type=float, default=1500.0,
                        help="cue current amplitude (pA)")
    parser.add_argument("--cue-fraction", type=float, default=0.5,
                        help="fraction of dlpfc_wm exc neurons cued")
    parser.add_argument("--delay-ms", type=float, default=3000.0,
                        help="post-cue delay duration in ms")
    parser.add_argument("--bin-ms", type=float, default=50.0,
                        help="delay-period bin width in ms")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT,
                        help="output JSON path")
    args = parser.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    ratio_values = _parse_ratio_values(args.nmda_ratios)
    if not seeds:
        print("[Q-RATIO] FATAL: empty seed list", flush=True)
        return 2
    if not ratio_values:
        print("[Q-RATIO] FATAL: empty nmda_ratio list", flush=True)
        return 2

    _, backend_name = get_backend()
    gpu = is_gpu_backend()

    print("=" * 70, flush=True)
    print("=== Direction Q-tertiary: NMDA-AMPA conductance-ratio sweep ===",
          flush=True)
    print("=" * 70, flush=True)
    print("  backend=" + backend_name + " (GPU=" + str(gpu) + ")",
          flush=True)
    print("  nmda_ratios=" + str(ratio_values), flush=True)
    print("  n_dlpfc=" + str(args.n_dlpfc)
          + "  density=" + str(args.dlpfc_density)
          + "  inh=" + str(args.inh_weight_mean)
          + "  seeds=" + str(seeds), flush=True)
    print("  baseline_ms=" + str(args.baseline_ms)
          + "  cue_ms=" + str(args.cue_ms)
          + "  cue_amp_pA=" + str(args.cue_amplitude_pA)
          + "  cue_fraction=" + str(args.cue_fraction), flush=True)
    print("  delay_ms=" + str(args.delay_ms)
          + "  bin_ms=" + str(args.bin_ms), flush=True)
    print("  pre-registered thresholds:"
          + " ratio>=" + str(_Q_RATE_RATIO_MIN)
          + ", sustained>=" + str(_Q_DELAY_MIN_SEC) + "s,"
          + " min_seeds_pass=" + str(_Q_MIN_SEEDS_PASS),
          flush=True)
    print("  out=" + args.out, flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()

    cells: List[dict] = []
    for nmda_ratio in ratio_values:
        cell = _sweep_one_ratio(
            nmda_ratio=nmda_ratio, seeds=seeds,
            n_dlpfc=args.n_dlpfc, dlpfc_density=args.dlpfc_density,
            inh_weight_mean=args.inh_weight_mean,
            baseline_ms=args.baseline_ms, cue_ms=args.cue_ms,
            cue_amplitude_pA=args.cue_amplitude_pA,
            cue_fraction=args.cue_fraction,
            delay_ms=args.delay_ms, bin_ms=args.bin_ms,
        )
        cells.append(cell)

    wall_minutes = (time.time() - t_start) / 60.0

    # ---- Cross-cell aggregate table ----
    print("\n" + "=" * 70, flush=True)
    print("=== NMDA-RATIO SWEEP TABLE "
          "(multi-seed mean ratio / max sustained) ===",
          flush=True)
    print("=" * 70, flush=True)
    print("  nmda_r | TEST ratio mean | TEST sustained max | "
          "CTRL ratio max | CTRL sustained max | verdict",
          flush=True)
    for cell in cells:
        print(
            "  " + str(round(cell["nmda_ratio"], 2)).ljust(6)
            + " | " + str(round(cell["test_ratio_mean"], 2)).ljust(15)
            + " | " + str(round(cell["test_sustained_max"], 2)).ljust(18)
            + " | " + str(round(cell["control_ratio_max"], 2)).ljust(14)
            + " | " + str(round(cell["control_sustained_max"], 2)).ljust(18)
            + " | " + str(cell["verdict"]),
            flush=True,
        )

    # Determine if any cell produced sustained_sec >= 3.0 (the bar)
    any_passed_bar = any(
        cell["test_sustained_max"] >= _Q_DELAY_MIN_SEC for cell in cells
    )
    any_full_verdict_pass = any(
        cell["verdict"] == "Q_BISTABILITY_PASS" for cell in cells
    )

    print("\n=== SWEEP-LEVEL DIAGNOSIS ===", flush=True)
    print("  any cell with sustained_sec >= 3.0s: "
          + str(any_passed_bar), flush=True)
    print("  any cell with full verdict PASS:     "
          + str(any_full_verdict_pass), flush=True)
    print("  wall: " + str(round(wall_minutes, 2)) + " min", flush=True)

    out = {
        "backend": backend_name,
        "gpu": gpu,
        "experiment": "direction_Q_tertiary_nmda_ratio_sweep",
        "nmda_ratios": ratio_values,
        "seeds": seeds,
        "n_dlpfc": args.n_dlpfc,
        "dlpfc_density": args.dlpfc_density,
        "inh_weight_mean": args.inh_weight_mean,
        "baseline_ms": args.baseline_ms,
        "cue_ms": args.cue_ms,
        "cue_amplitude_pA": args.cue_amplitude_pA,
        "cue_fraction": args.cue_fraction,
        "delay_ms": args.delay_ms,
        "bin_ms": args.bin_ms,
        "cells": cells,
        "any_cell_sustained_bar_met": bool(any_passed_bar),
        "any_cell_full_verdict_pass": bool(any_full_verdict_pass),
        "pre_registered_thresholds": {
            "_Q_RATE_RATIO_MIN": _Q_RATE_RATIO_MIN,
            "_Q_DELAY_MIN_SEC": _Q_DELAY_MIN_SEC,
            "_Q_MIN_SEEDS_PASS": _Q_MIN_SEEDS_PASS,
        },
        "wall_minutes": wall_minutes,
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote " + args.out, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
