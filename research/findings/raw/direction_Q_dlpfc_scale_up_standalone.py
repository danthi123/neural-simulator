# research/findings/raw/direction_Q_dlpfc_scale_up_standalone.py
"""Direction Q multi-seed standalone runner.

Orchestrates the full Wang 2002 delayed-response protocol across
3 seeds for BOTH the NMDA-on (test) condition AND the mandatory
NMDA-off (AMPA-only) control condition. Writes JSON with the
verdict + per-seed data.

Per docs/plans/2026-05-25-direction-Q-dlpfc-scale-up-implementation.md
Task 4. The decisive full-scale run (n_dlpfc=1000) is Task 6
(controller-only); this runner supports both smoke (n_dlpfc=200) and
decisive (n_dlpfc=1000) modes via CLI args.

Discipline (binding):
- Verdict thresholds frozen in direction_Q_verdict.py; NOT overridable
- Mandatory NMDA-off control is part of the runner (enable_nmda=False)
- No protected module modified; uses only Direction Q Task 1/2/3 + sim.backend
- Standalone bridges are built fresh per seed-condition combo (full
  reset; no state leakage)
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
    _HERE, "direction_Q_dlpfc_scale_up_standalone.json"
)


def _parse_seeds(seeds_str: str) -> List[int]:
    """Parse "42,43,44" -> [42, 43, 44]."""
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def _compute_sustained_sec(delay_rates: List[float],
                                baseline_rate: float,
                                ratio_min: float,
                                bin_ms: float) -> float:
    """Compute the total duration (sec) of elevated firing in the
    delay period.

    Per Task 4 spec: "total time with rate >= ratio_min * baseline as the
    sustained_sec value". Uses ratio_min (frozen threshold from verdict
    module) so the sustained-sec metric is consistent with the verdict's
    rate_ratio bar.

    A baseline of ~0 Hz (silent population) is handled by treating the
    elevation threshold as a tiny absolute rate (1e-9 Hz) to avoid
    div-by-zero or accepting any nonzero spike as "elevated". This is
    conservative: if baseline is truly silent, even tiny delay activity
    won't trigger the threshold unless it's well above noise.
    """
    if not delay_rates:
        return 0.0
    if baseline_rate <= 0.0:
        # Silent baseline: use absolute floor (1.0 Hz) for elevation.
        # Conservative; a real persistent attractor produces tens of Hz.
        threshold = 1.0
    else:
        threshold = ratio_min * baseline_rate
    n_elevated = sum(1 for r in delay_rates if r >= threshold)
    return float(n_elevated) * float(bin_ms) / 1000.0


def _run_one_condition(seed: int, n_dlpfc: int, dlpfc_density: float,
                            enable_nmda: bool,
                            baseline_ms: float, cue_ms: float,
                            cue_amplitude_pA: float, cue_fraction: float,
                            delay_ms: float, bin_ms: float,
                            verbose: bool = True) -> dict:
    """Run a single seed-condition combo through the full protocol.

    Returns a dict with baseline_rate, cue_rate, delay_rates,
    mean_delay_rate, rate_ratio, sustained_sec, wall_seconds.
    """
    label = "NMDA-ON" if enable_nmda else "NMDA-OFF (AMPA-only control)"
    if verbose:
        print(
            "[Q-RUN] seed=" + str(seed) + " " + label
            + " n_dlpfc=" + str(n_dlpfc)
            + " density=" + str(dlpfc_density),
            flush=True,
        )
    t0 = time.time()
    bridge = build_q_test_bridge(
        seed=seed, n_dlpfc=n_dlpfc, dlpfc_density=dlpfc_density,
        enable_nmda=enable_nmda, verbose=verbose,
    )
    if verbose:
        print(
            "[Q-RUN]   build " + str(round(time.time() - t0, 1)) + "s",
            flush=True,
        )

    t1 = time.time()
    baseline_rate = run_baseline_period(bridge, baseline_ms)
    if verbose:
        print(
            "[Q-RUN]   baseline=" + str(round(baseline_rate, 3))
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
            "[Q-RUN]   cue=" + str(round(cue_rate, 3))
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
            "[Q-RUN]   delay=" + str(n_bins) + " bins ("
            + str(round(time.time() - t3, 1))
            + "s); final-3=" + str([round(r, 3) for r in last3]),
            flush=True,
        )

    if delay_rates:
        mean_delay_rate = float(sum(delay_rates)) / float(len(delay_rates))
    else:
        mean_delay_rate = 0.0

    # rate_ratio = mean_delay_rate / baseline_rate
    # Use small epsilon for silent-baseline safety; the verdict
    # compares against fixed bar _Q_RATE_RATIO_MIN=2.0 either way.
    if baseline_rate > 1e-9:
        rate_ratio = mean_delay_rate / baseline_rate
    elif mean_delay_rate > 1e-9:
        # Silent baseline + nonzero delay: treat as large ratio so the
        # verdict can decide via the absolute sustained_sec criterion.
        # (Note: at baseline=0, this branch could fire on any noise; the
        # downstream verdict still requires sustained_sec >= 3.0s, so
        # this is safe as a metric-level convention.)
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
            "[Q-RUN]   rate_ratio=" + str(round(rate_ratio, 2))
            + "  sustained_sec=" + str(round(sustained_sec, 3))
            + "  wall=" + str(round(wall_seconds, 1)) + "s",
            flush=True,
        )

    return {
        "seed": seed,
        "enable_nmda": enable_nmda,
        "baseline_rate": baseline_rate,
        "cue_rate": cue_rate,
        "delay_rates": delay_rates,
        "mean_delay_rate": mean_delay_rate,
        "rate_ratio": float(rate_ratio),
        "sustained_sec": float(sustained_sec),
        "wall_seconds": float(wall_seconds),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Direction Q dlpfc_wm scale-up multi-seed runner "
            "(Wang 2002 delayed-response protocol with mandatory "
            "NMDA-off control)."
        )
    )
    parser.add_argument("--n-dlpfc", type=int, default=200,
                        help="dlpfc_wm region size (smoke=200, decisive=1000)")
    parser.add_argument("--dlpfc-density", type=float, default=0.10,
                        help="recurrent density within dlpfc_wm")
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
    if not seeds:
        print("[Q-RUN] FATAL: empty seed list", flush=True)
        return 2

    _, backend_name = get_backend()
    gpu = is_gpu_backend()

    print("=" * 70, flush=True)
    print("=== Direction Q: dlpfc_wm scale-up multi-seed runner ===",
          flush=True)
    print("=" * 70, flush=True)
    print("  backend=" + backend_name + " (GPU=" + str(gpu) + ")",
          flush=True)
    print("  n_dlpfc=" + str(args.n_dlpfc)
          + "  density=" + str(args.dlpfc_density)
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

    # ---- TEST condition: NMDA-on ----
    print("\n--- TEST CONDITION (NMDA-on) ---", flush=True)
    test_runs: List[dict] = []
    per_seed: List[Tuple[float, float]] = []
    for seed in seeds:
        run = _run_one_condition(
            seed=seed, n_dlpfc=args.n_dlpfc,
            dlpfc_density=args.dlpfc_density,
            enable_nmda=True,
            baseline_ms=args.baseline_ms, cue_ms=args.cue_ms,
            cue_amplitude_pA=args.cue_amplitude_pA,
            cue_fraction=args.cue_fraction,
            delay_ms=args.delay_ms, bin_ms=args.bin_ms,
        )
        test_runs.append(run)
        per_seed.append((run["rate_ratio"], run["sustained_sec"]))

    # ---- MANDATORY CONTROL: NMDA-off (AMPA-only) ----
    print("\n--- CONTROL CONDITION (NMDA-off / AMPA-only) ---",
          flush=True)
    control_runs: List[dict] = []
    control_per_seed: List[Tuple[float, float]] = []
    for seed in seeds:
        run = _run_one_condition(
            seed=seed, n_dlpfc=args.n_dlpfc,
            dlpfc_density=args.dlpfc_density,
            enable_nmda=False,  # AMPA-only control
            baseline_ms=args.baseline_ms, cue_ms=args.cue_ms,
            cue_amplitude_pA=args.cue_amplitude_pA,
            cue_fraction=args.cue_fraction,
            delay_ms=args.delay_ms, bin_ms=args.bin_ms,
        )
        control_runs.append(run)
        control_per_seed.append((run["rate_ratio"], run["sustained_sec"]))

    # ---- Verdict (frozen thresholds; no tuning) ----
    verdict = compute_verdict(per_seed, control_per_seed)

    wall_minutes = (time.time() - t_start) / 60.0

    print("\n" + "=" * 70, flush=True)
    print("=== AGGREGATE (3-seed NMDA-on vs NMDA-off) ===", flush=True)
    print("=" * 70, flush=True)
    for r in test_runs:
        print(
            "  TEST    seed=" + str(r["seed"])
            + " baseline=" + str(round(r["baseline_rate"], 3))
            + "Hz  cue=" + str(round(r["cue_rate"], 3))
            + "Hz  delay=" + str(round(r["mean_delay_rate"], 3))
            + "Hz  ratio=" + str(round(r["rate_ratio"], 2))
            + "  sustained=" + str(round(r["sustained_sec"], 2)) + "s",
            flush=True,
        )
    for r in control_runs:
        print(
            "  CONTROL seed=" + str(r["seed"])
            + " baseline=" + str(round(r["baseline_rate"], 3))
            + "Hz  cue=" + str(round(r["cue_rate"], 3))
            + "Hz  delay=" + str(round(r["mean_delay_rate"], 3))
            + "Hz  ratio=" + str(round(r["rate_ratio"], 2))
            + "  sustained=" + str(round(r["sustained_sec"], 2)) + "s",
            flush=True,
        )

    print("\n=== VERDICT (pre-registered frozen thresholds) ===",
          flush=True)
    print("  verdict: " + str(verdict), flush=True)
    print("  bar: ratio>=" + str(_Q_RATE_RATIO_MIN)
          + ", sustained>=" + str(_Q_DELAY_MIN_SEC) + "s,"
          + " seeds_needed=" + str(_Q_MIN_SEEDS_PASS), flush=True)
    print("  wall: " + str(round(wall_minutes, 2)) + " min",
          flush=True)

    out = {
        "backend": backend_name,
        "gpu": gpu,
        "seeds": seeds,
        "n_dlpfc": args.n_dlpfc,
        "dlpfc_density": args.dlpfc_density,
        "baseline_ms": args.baseline_ms,
        "cue_ms": args.cue_ms,
        "cue_amplitude_pA": args.cue_amplitude_pA,
        "cue_fraction": args.cue_fraction,
        "delay_ms": args.delay_ms,
        "bin_ms": args.bin_ms,
        "per_seed_data": test_runs,
        "control_per_seed_data": control_runs,
        "per_seed_tuples": per_seed,
        "control_per_seed_tuples": control_per_seed,
        "verdict": verdict,
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
