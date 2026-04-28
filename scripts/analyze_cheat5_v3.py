"""Aggregate cheat #5 v3 + v3.1 results vs flagship baseline.

Decision matrix (per docs/plans/2026-04-28-cheat5-v3-lateral-inhibition.md):

v3.1cross (cheat #5 closure test):
| Mean sum | P0  | P1  | Verdict   | Next                                       |
|----------|-----|-----|-----------|--------------------------------------------|
| <= 4.1   | <=2.5 | <=2.5 | GO        | 6-seed validation, propagate, cheat #5 closed |
| 4.1-4.5  | OK  | OK  | GO MARGINAL | 6-seed; document closure-without-improvement |
| 4.5-6.0  | OK  | high | PARTIAL   | Try slower phase-3 gain (0.2), longer warmup |
| > 6.0 OR P0 high | - | - | NO-GO v3.1 | Move to v4 (developmental phase)         |

v3lateral (no-regression test):
- mean sum <= 4.5: GO (no regression vs flagship 4.08)
- mean sum > 4.5:  NO-GO (lateral inhibition itself regresses flagship)

Usage:
    python scripts/analyze_cheat5_v3.py
"""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

RAW = Path("research/findings/raw/g11_bg")


def load_runs(label_substring: str) -> list[dict]:
    """Find runs whose cmd.json extra_args match the v3 or v3.1 signature.

    The launcher saves with name patterns like g11_seed42_flagship_<id>.json.
    The cmd.json sidecar has the extra_args we used to identify which experiment.
    """
    out = []
    for cmd_path in RAW.glob("g11_seed*_flagship_*.cmd.json"):
        try:
            cmd = json.load(open(cmd_path))
        except Exception:
            continue
        extras = " ".join(cmd.get("extra_args", []))
        if label_substring not in extras:
            continue
        # v3 lateral has --bg-lateral-inhibition but NOT --bg-cross-projections.
        # v3.1 cross has both. Disambiguate.
        if label_substring == "--bg-lateral-inhibition" and "--bg-cross-projections" in extras:
            continue
        # The result file path is in out_path
        result_path = cmd.get("out_path")
        if not result_path or not Path(result_path).exists():
            continue
        try:
            data = json.load(open(result_path))
        except Exception:
            continue
        out.append({"seed": cmd.get("seed"), "result": data, "cmd_path": str(cmd_path)})
    # Also pick up legacy-named files (g11_seed{N}_v3lateral.json etc.)
    legacy_pattern = "g11_seed*_v3lateral.json" if "lateral" in label_substring else None
    if legacy_pattern:
        for path in RAW.glob(legacy_pattern):
            if "cmd" in path.name:
                continue
            try:
                data = json.load(open(path))
            except Exception:
                continue
            seed = data.get("seed")
            if not any(r["seed"] == seed for r in out):
                out.append({"seed": seed, "result": data, "cmd_path": str(path)})
    return out


def summary(runs: list[dict]) -> dict:
    """Per-seed sum of final_quarter_mean_distance + per-phase means."""
    if not runs:
        return {"n": 0}
    rows = []
    for r in runs:
        ps = r["result"].get("phase_stats", [])
        per_phase = [p.get("final_quarter_mean_distance", 0) for p in ps]
        rows.append({
            "seed": r["seed"],
            "n_phases": len(ps),
            "per_phase": per_phase,
            "sum": sum(per_phase),
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


def verdict_v3_lateral(s: dict) -> str:
    if s["n"] < 3:
        return f"INSUFFICIENT DATA ({s['n']} seeds)"
    if s["mean_sum"] <= 4.5:
        return f"GO — mean sum {s['mean_sum']:.2f} <= 4.5 (no regression vs flagship 4.08)"
    return f"NO-GO — mean sum {s['mean_sum']:.2f} > 4.5 (lateral inhibition regresses flagship)"


def verdict_v3_1_cross(s: dict) -> str:
    if s["n"] < 3:
        return f"INSUFFICIENT DATA ({s['n']} seeds)"
    sum_ = s["mean_sum"]
    p0 = s["mean_p0"] or 0
    p1 = s["mean_p1"] or 0
    if sum_ <= 4.1 and p0 <= 2.5 and p1 <= 2.5:
        return f"GO — sum {sum_:.2f}, p0 {p0:.2f}, p1 {p1:.2f}; cheat #5 CLOSED"
    if sum_ <= 4.5 and p0 < 4 and p1 < 4:
        return f"GO MARGINAL — sum {sum_:.2f}; closure-without-improvement"
    if sum_ <= 6.0:
        return f"PARTIAL — sum {sum_:.2f} (p0 {p0:.2f}, p1 {p1:.2f}); try v3.1.1"
    return f"NO-GO — sum {sum_:.2f}; pivot to v4 developmental phase"


def main():
    print("=" * 64)
    print("Cheat #5 v3 / v3.1 results vs flagship baseline (4.08)")
    print("=" * 64)

    print("\n--- v3lateral (--bg-lateral-inhibition, no cross) ---")
    v3 = load_runs("--bg-lateral-inhibition")
    s = summary(v3)
    if s["n"] == 0:
        print("  No completed runs yet.")
    else:
        for r in sorted(s["rows"], key=lambda x: x["seed"] or 0):
            phases = ", ".join(f"{p:.2f}" for p in r["per_phase"])
            print(f"  seed={r['seed']:3} sum={r['sum']:.2f}  phases=[{phases}]")
        print(f"  mean sum={s['mean_sum']:.3f} ± {s['stdev_sum']:.3f}  "
              f"p0={s['mean_p0']:.2f} p1={s['mean_p1']:.2f}  (n={s['n']})")
        print(f"  Verdict: {verdict_v3_lateral(s)}")

    print("\n--- v3.1cross (lateral + cross-projections, cheat #5 closure) ---")
    v31 = load_runs("--bg-cross-projections")
    s = summary(v31)
    if s["n"] == 0:
        print("  No completed runs yet.")
    else:
        for r in sorted(s["rows"], key=lambda x: x["seed"] or 0):
            phases = ", ".join(f"{p:.2f}" for p in r["per_phase"])
            print(f"  seed={r['seed']:3} sum={r['sum']:.2f}  phases=[{phases}]")
        print(f"  mean sum={s['mean_sum']:.3f} ± {s['stdev_sum']:.3f}  "
              f"p0={s['mean_p0']:.2f} p1={s['mean_p1']:.2f}  (n={s['n']})")
        print(f"  Verdict: {verdict_v3_1_cross(s)}")


if __name__ == "__main__":
    main()
