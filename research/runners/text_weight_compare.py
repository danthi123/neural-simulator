"""Compare weight diagnostics across multiple bridge checkpoints.

Loads the JSON output of text_weight_diagnostic.py for each checkpoint
and prints a side-by-side comparison of pathway means and token-targeted
diff_target values. Useful for tracking learning progress across
training runs with different fixes applied.

Usage:
  python -m research.runners.text_weight_compare \\
      run_a:research/findings/raw/g11_bg/text_weight_diag_R3R6_NoT1_seed42.json \\
      run_b:research/findings/raw/g11_bg/text_weight_diag_R3R6_HebOff_seed42.json \\
      run_c:research/findings/raw/g11_bg/text_weight_diag_R3R6_HebOff_v2_seed42.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+",
                    help="LABEL:path pairs to weight-diag JSONs")
    args = ap.parse_args()

    runs = {}
    for spec in args.inputs:
        if ":" not in spec:
            ap.error(f"expected LABEL:path, got {spec}")
        label, path = spec.split(":", 1)
        p = Path(path)
        if not p.exists():
            print(f"  WARN: {p} not found, skipping {label}")
            continue
        runs[label] = json.loads(p.read_text())

    if not runs:
        ap.error("no valid input files")

    labels = list(runs.keys())

    # Pathway-mean comparison
    print("=" * 95)
    print(f"PATHWAY MEAN WEIGHTS  (>0.05 = above floor; near 2.0+ = real learning)")
    print("=" * 95)
    print(f"{'pathway':<28} | " + " | ".join(f"{l:>14}" for l in labels))
    print("-" * 95)

    # Get pathway names from first run (assume same set across runs)
    first_pathways = {p["name"]: i for i, p in enumerate(runs[labels[0]]["pathways"])}
    for pname, idx in first_pathways.items():
        row = []
        for lab in labels:
            paths = runs[lab].get("pathways", [])
            match = next((p for p in paths if p["name"] == pname), None)
            if match and match.get("mean") is not None:
                cell = f"{match['mean']:.3f} ({match['n']})"
            else:
                cell = "-"
            row.append(cell)
        print(f"{pname:<28} | " + " | ".join(f"{c:>14}" for c in row))

    # Token-targeted diff_target comparison
    print()
    print("=" * 95)
    print("TOKEN-TARGETED DIFF (target_motor mean - non-target_motor mean)")
    print(">0 = LEARNED target preference  |  <0 = REVERSED  |  ~0 = no signal")
    print("=" * 95)
    DIRECTIONS = ["north", "east", "south", "west"]
    print(f"{'token':<10} | " + " | ".join(f"{l:>14}" for l in labels))
    print("-" * 95)
    for tok in DIRECTIONS:
        row = []
        for lab in labels:
            tta = runs[lab].get("token_to_motor_analysis", {})
            entry = tta.get(tok)
            if entry:
                diff = entry.get("diff_target", 0)
                marker = "OK" if diff > 0.05 else "(weak)" if diff > 0 else "REV" if diff < -0.05 else "0"
                cell = f"{diff:+.4f} {marker}"
            else:
                cell = "-"
            row.append(cell)
        print(f"{tok:<10} | " + " | ".join(f"{c:>14}" for c in row))

    # Verdict comparison
    print()
    print("=" * 95)
    print("VERDICT (n_learned tokens out of 4)")
    print("=" * 95)
    for lab in labels:
        v = runs[lab].get("verdict", {})
        n = v.get("n_learned", "?")
        s = v.get("summary", "?")
        print(f"  {lab:<14}: {n}/4 tokens learned -- {s}")


if __name__ == "__main__":
    main()
