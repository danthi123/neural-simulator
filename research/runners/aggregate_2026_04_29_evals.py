"""Aggregate 2026-04-29 cheat-5 eval results across conditions and seeds.

Six conditions × three seeds = eighteen data points. Outputs a markdown table
suitable for the findings docs.

Conditions:
  baseline:  --bg-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-fsis
  +A:        + --enable-cluster-a-closed-loop
  +C:        + --enable-tonic-da
  A+C+B.3:   + --enable-cluster-a-closed-loop --enable-tonic-da --enable-tans
  D:         + --enable-cluster-d-hippocampus --hippocampus --landmarks
  A+D:       + --enable-cluster-a-closed-loop --enable-cluster-d-hippocampus --hippocampus --landmarks

Run after all evals complete:
    python -m research.runners.aggregate_2026_04_29_evals
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional


CONDITIONS = {
    "baseline (post-R)": "g11_seed{seed}_clusterB_postR.json",
    "+ Cluster A": "g11_seed{seed}_clusterA.json",
    "+ Cluster C v1 only": "g11_seed{seed}_clusterC.json",
    "A + C v1 + B.3": "g11_seed{seed}_combo_ACB3.json",
    "+ Cluster D only": "g11_seed{seed}_clusterD.json",
    "A + D": "g11_seed{seed}_clusterAD.json",
    "+ Cluster C v2 only": "g11_seed{seed}_clusterCv2.json",
    "A + C v2": "g11_seed{seed}_clusterACv2.json",
    "+ Cluster E only": "g11_seed{seed}_clusterE.json",
    "A + E": "g11_seed{seed}_clusterAE.json",
    "no-heuristic baseline": "g11_seed{seed}_noHeur_baseline.json",
    "no-heuristic A+C+E": "g11_seed{seed}_noHeur_ACE.json",
    "FIXED cascade baseline": "g11_seed{seed}_FIX_baseline.json",
    "FIXED cascade A+C+E": "g11_seed{seed}_FIX_ACE.json",
}
SEEDS = [42, 43, 44]
RAW_DIR = Path("research/findings/raw/g11_bg")


def parse(file: Path) -> Optional[Dict]:
    if not file.exists():
        return None
    d = json.load(open(file))
    ps = d.get("phase_stats", [])
    if len(ps) != 4:
        return {"missing_phases": True, "n_phases": len(ps)}
    fq = [p["final_quarter_mean_distance"] for p in ps]
    return {"finalQs": fq, "sum": sum(fq)}


def aggregate() -> None:
    print("# Cheat-5 multi-goal eval - 8 conditions x 3 seeds (n=3)\n")
    print("| Condition | Seed 42 | Seed 43 | Seed 44 | Mean +/- std |")
    print("|---|---|---|---|---|")
    summary: List[Dict] = []
    for label, tmpl in CONDITIONS.items():
        sums: List[Optional[float]] = []
        for seed in SEEDS:
            fp = RAW_DIR / tmpl.format(seed=seed)
            r = parse(fp)
            if r is None:
                sums.append(None)
            elif r.get("missing_phases"):
                sums.append(None)
            else:
                sums.append(r["sum"])
        complete = [s for s in sums if s is not None]
        if len(complete) >= 2:
            m = statistics.mean(complete)
            s_ = statistics.stdev(complete) if len(complete) > 1 else 0.0
            cell = f"{m:.2f} +/- {s_:.2f}"
            summary.append({"label": label, "mean": m, "std": s_})
        elif len(complete) == 1:
            cell = f"{complete[0]:.2f} (n=1)"
        else:
            cell = "PEND"
        seed_cells = [f"{s:.2f}" if s is not None else "PEND" for s in sums]
        print(f"| {label} | {seed_cells[0]} | {seed_cells[1]} | {seed_cells[2]} | **{cell}** |")

    print("\n## Per-phase mean (mean across seeds)\n")
    print("| Condition | P0 | P1 | P2 | P3 |")
    print("|---|---|---|---|---|")
    for label, tmpl in CONDITIONS.items():
        phase_data: Dict[int, List[float]] = {0: [], 1: [], 2: [], 3: []}
        for seed in SEEDS:
            fp = RAW_DIR / tmpl.format(seed=seed)
            r = parse(fp)
            if r and not r.get("missing_phases"):
                for i, v in enumerate(r["finalQs"]):
                    phase_data[i].append(v)
        cells = []
        for i in range(4):
            vals = phase_data[i]
            if len(vals) >= 1:
                cells.append(f"{statistics.mean(vals):.2f}")
            else:
                cells.append("PEND")
        print(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

    print("\n## Decision matrix (Cluster A primary test)\n")
    baseline_entry = next((s for s in summary if s["label"] == "baseline (post-R)"), None)
    a_entry = next((s for s in summary if s["label"] == "+ Cluster A"), None)
    if baseline_entry and a_entry:
        delta_mean = a_entry["mean"] - baseline_entry["mean"]
        delta_std = a_entry["std"] - baseline_entry["std"]
        print(f"baseline: {baseline_entry['mean']:.2f} +/- {baseline_entry['std']:.2f}")
        print(f"+ Cluster A: {a_entry['mean']:.2f} +/- {a_entry['std']:.2f}")
        print(f"delta mean: {delta_mean:+.2f}")
        print(f"delta std: {delta_std:+.2f}")
        if delta_mean <= -1.0 and delta_std <= 0:
            verdict = "**GO** — tier-3 (6-seed) validation next"
        elif delta_mean <= 0.0:
            verdict = "**PARTIAL** — composability with C/D may still help"
        else:
            verdict = "**NO-GO** — closed loop alone doesn't help; check combo conditions"
        print(f"Verdict: {verdict}")


if __name__ == "__main__":
    aggregate()
