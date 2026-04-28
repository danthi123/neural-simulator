"""Aggregate cheat #5 v4 (developmental pretraining) results vs flagship baseline.

Decision matrix (per docs/plans/2026-04-28-cheat5-v4-design.md):

| Eval-phase mean sum (n=6) | P0   | P1   | Verdict        |
|---------------------------|------|------|----------------|
| <= 4.1                    | <=2.5| <=2.5| GO             |
| 4.1 - 4.5                 | OK   | OK   | GO MARGINAL    |
| 4.5 - 6.0                 | OK   | high | PARTIAL        |
| > 6.0 OR P0 high          |  -   |  -   | NO-GO v4       |

Tier 2 short-circuit (n=3): mean <= 4.5 -> proceed to Tier 3; > 6 -> NO-GO; otherwise review.

Identification: a v4 run is one whose cmd.json `extra_args` contains
`--developmental-pretraining`. The aggregator finds them, parses their
result JSONs, and applies the matrix.

Usage:
    python scripts/analyze_cheat5_v4.py
    python scripts/analyze_cheat5_v4.py --tier 1   # just tier 1
    python scripts/analyze_cheat5_v4.py --tier 2
    python scripts/analyze_cheat5_v4.py --tier 3
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

RAW = Path("research/findings/raw/g11_bg")
FLAGSHIP_BASELINE = 4.08  # 6-seed reference from research/findings/2026-04-27-perception-arc-COMPLETE


def load_v4_runs(tier_filter: str | None = None) -> list[dict]:
    """Return all v4 runs whose cmd.json extra_args contain --developmental-pretraining.

    A "tier" is implicit in the steps-per-goal value:
      tier 1: 1 goal x 1000 steps
      tier 2: 5 goals x 1000 steps
      tier 3: 10 goals x 3000 steps
    """
    out = []
    for cmd_path in sorted(RAW.glob("g11_seed*.cmd.json")):
        try:
            cmd = json.load(open(cmd_path))
        except Exception:
            continue
        extras = cmd.get("extra_args", [])
        cli = " ".join(cmd.get("cmd", []))
        if "--developmental-pretraining" not in extras and "--developmental-pretraining" not in cli:
            continue
        # parse pretraining params (n_goals, steps_per_goal) from the cli
        n_goals = _extract_int(cmd, "--pretraining-n-goals", default=10)
        steps_per_goal = _extract_int(cmd, "--pretraining-steps-per-goal", default=3000)
        total = n_goals * steps_per_goal
        tier = "1" if total < 2000 else ("2" if total < 10000 else "3")
        if tier_filter and tier != str(tier_filter):
            continue
        result_path = cmd.get("out_path")
        if not result_path or not Path(result_path).exists():
            continue
        try:
            data = json.load(open(result_path))
        except Exception:
            continue
        out.append({
            "seed": cmd.get("seed"),
            "tier": tier,
            "n_goals": n_goals,
            "steps_per_goal": steps_per_goal,
            "result": data,
            "cmd_path": str(cmd_path),
            "out_path": result_path,
        })
    return out


def _extract_int(cmd: dict, flag: str, default: int) -> int:
    """Find `flag` in cmd['cmd'] or cmd['extra_args'] and return the next token as int."""
    for src in (cmd.get("cmd") or [], cmd.get("extra_args") or []):
        for i, tok in enumerate(src):
            if tok == flag and i + 1 < len(src):
                try:
                    return int(src[i + 1])
                except ValueError:
                    pass
    return default


def summarize(runs: list[dict]) -> dict:
    if not runs:
        return {"n": 0}
    rows = []
    for r in runs:
        ps = r["result"].get("phase_stats", [])
        per_phase = [p.get("final_quarter_mean_distance", 0) for p in ps]
        rows.append({
            "seed": r["seed"],
            "tier": r["tier"],
            "n_phases": len(ps),
            "per_phase": per_phase,
            "sum": sum(per_phase),
            "pretraining_total": r["n_goals"] * r["steps_per_goal"],
        })
    sums = [r["sum"] for r in rows]
    p0s = [r["per_phase"][0] for r in rows if r["per_phase"]]
    p1s = [r["per_phase"][1] for r in rows if len(r["per_phase"]) > 1]
    return {
        "n": len(rows),
        "rows": rows,
        "mean_sum": statistics.mean(sums),
        "stdev_sum": statistics.stdev(sums) if len(sums) > 1 else 0.0,
        "mean_p0": statistics.mean(p0s) if p0s else None,
        "mean_p1": statistics.mean(p1s) if p1s else None,
    }


def verdict(summary: dict, n_required: int) -> str:
    if summary["n"] < n_required:
        return f"INSUFFICIENT DATA ({summary['n']}/{n_required} seeds)"
    s = summary["mean_sum"]
    p0 = summary["mean_p0"] or 0
    p1 = summary["mean_p1"] or 0
    if s <= 4.1 and p0 <= 2.5 and p1 <= 2.5:
        return f"GO - sum {s:.2f}, p0 {p0:.2f}, p1 {p1:.2f}; cheat #5 CLOSED via developmental pretraining"
    if s <= 4.5 and p0 < 4 and p1 < 4:
        return f"GO MARGINAL - sum {s:.2f}; closure-without-improvement"
    if s <= 6.0:
        return f"PARTIAL - sum {s:.2f} (p0 {p0:.2f}, p1 {p1:.2f}); try longer pretraining or more goals"
    return f"NO-GO v4 - sum {s:.2f}; cross-projections may be off-axis even developmentally"


def print_tier(tier: str, runs: list[dict], required_seeds: int) -> dict | None:
    print(f"\n--- Tier {tier} (developmental pretraining) ---")
    if not runs:
        print(f"  No tier-{tier} v4 runs found yet.")
        return None
    s = summarize(runs)
    for r in sorted(s["rows"], key=lambda x: x["seed"] or 0):
        phases = ", ".join(f"{p:.2f}" for p in r["per_phase"])
        print(f"  seed={r['seed']:>3} sum={r['sum']:.2f}  phases=[{phases}]  pretraining={r['pretraining_total']}")
    print(f"  mean sum={s['mean_sum']:.3f} +/- {s['stdev_sum']:.3f}  "
          f"p0={s['mean_p0']:.2f} p1={s['mean_p1']:.2f}  (n={s['n']})")
    print(f"  Verdict: {verdict(s, required_seeds)}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=str, default=None,
                    help="Restrict to one tier: 1, 2, or 3. Default: all tiers.")
    args = ap.parse_args()

    print("=" * 64)
    print(f"Cheat #5 v4 results vs flagship baseline ({FLAGSHIP_BASELINE})")
    print("=" * 64)

    all_runs = load_v4_runs(tier_filter=args.tier)
    if not all_runs:
        print("\nNo v4 runs found.")
        print("Looking for: research/findings/raw/g11_bg/g11_seed*.cmd.json with"
              " --developmental-pretraining in extra_args.")
        return

    if args.tier:
        # single tier
        n_required = 1 if args.tier == "1" else (3 if args.tier == "2" else 6)
        print_tier(args.tier, all_runs, n_required)
        return

    # all tiers
    tier1 = [r for r in all_runs if r["tier"] == "1"]
    tier2 = [r for r in all_runs if r["tier"] == "2"]
    tier3 = [r for r in all_runs if r["tier"] == "3"]
    print_tier("1", tier1, 1)
    print_tier("2", tier2, 3)
    print_tier("3", tier3, 6)


if __name__ == "__main__":
    main()
