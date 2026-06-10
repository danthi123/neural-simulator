"""Aggregate the navigation-gate(a) six-seed campaign for roadmap step 2a.

Roadmap step 2 consolidates the navigation brain and the conversational
brain onto a single ``SimulationBridge``. Step 2a holds the conversational
half **frozen** (its synapses cannot change) and on its **own block of
neuron indices** (disjoint from the navigation neurons), and then asks one
question:

    Does carrying that frozen conversational half change how well the
    bridge navigates?

To answer it, each seed is run twice with identical settings: once as a
navigation-only bridge ("standalone") and once on the merged bridge that
also holds the conversational neurons ("merged"). The navigation score for
a run is the sum, over the four goal phases, of the final-quarter mean
distance to the goal. **Lower is better** (the agent is sitting closer to
the goal at the end of each phase).

This script reads the twelve run files::

    research/findings/raw/nav_gate_2a/gate6_{standalone,merged}_seed{42..47}.json

and reports, per seed, the standalone score, the merged score, and their
difference (merged minus standalone). It then renders a verdict:

  - GREEN_INERT        every seed's difference is essentially zero. The
                       conversational half is inert; merging is free. (This
                       is the intended result — disjoint plus frozen means
                       the navigation computation is deterministically
                       unchanged.)
  - GREEN_WITHIN_NOISE differences are non-zero but stay within the
                       deterministic run-to-run noise floor (about 0.7 on
                       this benchmark). Merging is within noise.
  - REGRESS            at least one seed's difference exceeds the noise
                       floor. Carrying the conversational half measurably
                       changed navigation. This is a real finding (the
                       measured cost of merging) to report honestly, not to
                       hide.
  - INCOMPLETE         not all twelve files are present yet.

Usage::

    python -m research.runners.nav_gate2a_aggregate \
        --raw-dir research/findings/raw/nav_gate_2a \
        --out research/findings/raw/nav_gate_2a/_gate2a_verdict.json
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional

# The deterministic run-to-run noise floor on this navigation benchmark.
# CLAUDE.md documents the ``--deterministic`` flag tightening seed-to-seed
# noise from +/-3-5 down to about +/-0.7. A matched-seed difference at or
# below this is "within noise"; above it is a real change.
NOISE_FLOOR = 0.7

# Below this, a matched-seed difference counts as effectively zero (the
# conversational half is inert). 0.05 is comfortably below the noise floor
# and well above floating-point summation error on four ~0.5 terms.
INERT_EPSILON = 0.05

DEFAULT_SEEDS = range(42, 48)
DEFAULT_RAW_DIR = os.path.join("research", "findings", "raw", "nav_gate_2a")


def score_from_data(data: dict) -> float:
    """Return a run's navigation score: sum of per-phase final-quarter means.

    Raises ``ValueError`` if the run file has no usable ``phase_stats`` — a
    missing score must never be silently treated as zero (which would look
    like a perfect navigation result).
    """
    phases = data.get("phase_stats")
    if not phases:
        raise ValueError("run data has no non-empty 'phase_stats'")
    total = 0.0
    for p in phases:
        val = p.get("final_quarter_mean_distance")
        if val is None:
            raise ValueError(
                f"phase {p.get('phase')} missing 'final_quarter_mean_distance'"
            )
        total += float(val)
    return total


def _run_path(raw_dir: str, seed: int, arm: str) -> str:
    return os.path.join(raw_dir, f"gate6_{arm}_seed{seed}.json")


def _load_score(raw_dir: str, seed: int, arm: str) -> Optional[float]:
    """Load one run's score, or ``None`` if the file is not present yet."""
    path = _run_path(raw_dir, seed, arm)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return score_from_data(data)


def aggregate_gate2a(
    raw_dir: str = DEFAULT_RAW_DIR,
    seeds: Iterable[int] = DEFAULT_SEEDS,
) -> Dict:
    """Aggregate per-seed standalone vs merged navigation scores.

    Returns a dict with per-seed rows, the per-arm mean/std over the seeds
    that have both arms present, the maximum absolute matched-seed delta,
    the mean delta (merged minus standalone), the count of complete seeds,
    and the list of missing (seed, arm) pairs.
    """
    seeds = list(seeds)
    rows: List[Dict] = []
    missing: List[Dict] = []
    standalone_scores: List[float] = []
    merged_scores: List[float] = []
    deltas: List[float] = []

    for seed in seeds:
        s = _load_score(raw_dir, seed, "standalone")
        m = _load_score(raw_dir, seed, "merged")
        if s is None:
            missing.append({"seed": seed, "arm": "standalone"})
        if m is None:
            missing.append({"seed": seed, "arm": "merged"})
        delta = (m - s) if (s is not None and m is not None) else None
        if delta is not None:
            standalone_scores.append(s)
            merged_scores.append(m)
            deltas.append(delta)
        rows.append(
            {
                "seed": seed,
                "standalone": s,
                "merged": m,
                "delta": delta,
                "complete": delta is not None,
            }
        )

    def _mean(xs: List[float]) -> Optional[float]:
        return mean(xs) if xs else None

    def _std(xs: List[float]) -> Optional[float]:
        # Population std: descriptive spread over the available seeds, not an
        # inferential estimate.
        return pstdev(xs) if len(xs) >= 2 else (0.0 if len(xs) == 1 else None)

    return {
        "raw_dir": raw_dir,
        "seeds": seeds,
        "rows": rows,
        "missing": missing,
        "n_complete": len(deltas),
        "n_expected": len(seeds),
        "standalone_mean": _mean(standalone_scores),
        "standalone_std": _std(standalone_scores),
        "merged_mean": _mean(merged_scores),
        "merged_std": _std(merged_scores),
        "max_abs_delta": max((abs(d) for d in deltas), default=None),
        "mean_delta": _mean(deltas),
    }


def verdict(
    agg: Dict,
    inert_epsilon: float = INERT_EPSILON,
    noise_floor: float = NOISE_FLOOR,
) -> Dict:
    """Map an aggregate into a pass/fail label with a one-line reason.

    The decision is driven by ``max_abs_delta`` (the largest matched-seed
    difference) because gate (a) is fundamentally a per-seed inertness
    question: at the same seed, with the same deterministic settings, does
    the merged bridge navigate the same as the standalone one?
    """
    if agg["n_complete"] < agg["n_expected"]:
        return {
            "label": "INCOMPLETE",
            "reason": (
                f"{agg['n_complete']}/{agg['n_expected']} seeds complete; "
                f"{len(agg['missing'])} run file(s) still missing"
            ),
        }

    max_abs = agg["max_abs_delta"]
    mean_d = agg["mean_delta"]

    if max_abs <= inert_epsilon:
        return {
            "label": "GREEN_INERT",
            "reason": (
                f"max |merged-standalone| = {max_abs:.4f} <= {inert_epsilon} "
                "across all seeds: the frozen conversational half is inert; "
                "merging does not change navigation"
            ),
        }
    if max_abs <= noise_floor:
        return {
            "label": "GREEN_WITHIN_NOISE",
            "reason": (
                f"max |merged-standalone| = {max_abs:.4f} <= {noise_floor} "
                f"(deterministic noise floor); mean delta {mean_d:+.4f}: "
                "merging is within run-to-run noise"
            ),
        }
    return {
        "label": "REGRESS",
        "reason": (
            f"max |merged-standalone| = {max_abs:.4f} > {noise_floor} "
            f"(noise floor); mean delta {mean_d:+.4f} (positive = merged "
            "navigates worse): carrying the conversational half measurably "
            "changed navigation -- report as a finding"
        ),
    }


def format_report(agg: Dict, v: Dict) -> str:
    lines = []
    lines.append("nav-gate(a) aggregate -- step 2a (merged vs standalone nav)")
    lines.append("=" * 64)
    lines.append(
        f"{'seed':>5} {'standalone':>12} {'merged':>12} {'delta(m-s)':>12}"
    )
    lines.append("-" * 64)
    for r in agg["rows"]:
        s = "   --" if r["standalone"] is None else f"{r['standalone']:>12.4f}"
        m = "   --" if r["merged"] is None else f"{r['merged']:>12.4f}"
        d = "   --" if r["delta"] is None else f"{r['delta']:>+12.4f}"
        lines.append(f"{r['seed']:>5} {s} {m} {d}")
    lines.append("-" * 64)
    sm = agg["standalone_mean"]
    mm = agg["merged_mean"]
    sm_s = "--" if sm is None else f"{sm:.4f}"
    mm_s = "--" if mm is None else f"{mm:.4f}"
    ss_s = "--" if agg["standalone_std"] is None else f"{agg['standalone_std']:.4f}"
    ms_s = "--" if agg["merged_std"] is None else f"{agg['merged_std']:.4f}"
    lines.append(f"standalone mean+/-std : {sm_s} +/- {ss_s}")
    lines.append(f"merged     mean+/-std : {mm_s} +/- {ms_s}")
    if agg["max_abs_delta"] is not None:
        lines.append(f"max |delta|           : {agg['max_abs_delta']:.4f}")
        lines.append(f"mean delta (m-s)      : {agg['mean_delta']:+.4f}")
    lines.append(f"complete seeds        : {agg['n_complete']}/{agg['n_expected']}")
    lines.append("=" * 64)
    lines.append(f"VERDICT: {v['label']}")
    lines.append(f"  {v['reason']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    ap.add_argument(
        "--seeds",
        default="42,43,44,45,46,47",
        help="comma-separated seed list",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="optional path to write the JSON aggregate + verdict",
    )
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    agg = aggregate_gate2a(args.raw_dir, seeds=seeds)
    v = verdict(agg)
    report = format_report(agg, v)
    print(report)

    if args.out:
        payload = dict(agg)
        payload["verdict"] = v
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[wrote {args.out}]")

    return 0 if v["label"].startswith("GREEN") else (2 if v["label"] == "REGRESS" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
