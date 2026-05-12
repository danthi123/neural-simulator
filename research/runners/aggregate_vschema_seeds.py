"""Aggregate V_SCHEMA-only multi-seed results.

Reads invivo_seed{N}_v_schema.json files and reports per-seed which
novel keys bound correctly. Each seed's "true bind" pattern shows
whether V_SCHEMA's mountain→south seed-42 success was lucky or
systematic.

Usage:
    python -m research.runners.aggregate_vschema_seeds \
        --raw-root research/findings/raw/g11_bg/invivo_binding \
        --seeds 42,43,44,100,101,102 \
        --out research/findings/2026-05-12-V_SCHEMA-multiseed.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_seed(seed, root):
    """Load V_SCHEMA result for a seed. Try multiple file patterns.

    Priority: explicit _v_schema.json > _v0_v_schema.json (arch-fixed
    combined run) > generic invivo_seed42.json (legacy, may be from
    broken-arch run for seed 42 only).
    """
    candidates = [
        root / f"invivo_seed{seed}_v_schema.json",
        root / f"invivo_seed{seed}_v0_v_schema.json",
        root / f"invivo_seed{seed}.json",
    ]
    for fp in candidates:
        if not fp.exists():
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        # Find v_schema result
        for r in d.get("results", []):
            if r["variant"] == "v_schema":
                return r, fp
    return None, candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=str,
                    default="research/findings/raw/g11_bg/invivo_binding")
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    root = Path(args.raw_root)

    rows = []
    for s in seeds:
        r, fp = load_seed(s, root)
        if r is None:
            print(f"  seed {s}: {fp} MISSING", file=sys.stderr)
            continue
        details = {d["key"]: d for d in r["details"]}
        rows.append({
            "seed": s,
            "n_correct": r["n_correct"],
            "n_total": r["n_total"],
            "apple": details.get("apple", {}),
            "river": details.get("river", {}),
            "mountain": details.get("mountain", {}),
            "forest": details.get("forest", {}),
        })

    if not rows:
        print("No seeds found.")
        return 1

    n = len(rows)
    n_pass_3of4 = sum(1 for r in rows if r["n_correct"] >= 3)
    n_pass_2of4 = sum(1 for r in rows if r["n_correct"] >= 2)
    n_any = sum(1 for r in rows if r["n_correct"] >= 1)

    # Per-key success rate
    key_pass = {k: 0 for k in ["apple", "river", "mountain", "forest"]}
    for r in rows:
        for k in key_pass:
            if r[k].get("correct"):
                key_pass[k] += 1

    print(f"=== V_SCHEMA multi-seed (n={n}) ===")
    print(f"  >=3/4 PASS: {n_pass_3of4}/{n}")
    print(f"  >=2/4 PASS: {n_pass_2of4}/{n}")
    print(f"  >=1/4 (any bind): {n_any}/{n}")
    print(f"Per-key pass rate across seeds:")
    for k, np in key_pass.items():
        print(f"  {k:>9}: {np}/{n}")
    print(f"\nPer-seed details:")
    for r in rows:
        marks = " ".join(
            ("OK" if r[k].get("correct") else "X") + ":" + r[k].get("got_value", "?")
            for k in ["apple", "river", "mountain", "forest"]
        )
        print(f"  seed {r['seed']:>3}: {r['n_correct']}/4 [{marks}]")

    if args.out:
        md = [
            f"# V_SCHEMA multi-seed result\n\n",
            f"**Date:** 2026-05-12\n",
            f"**Method:** Schema-supported novel-key binding (Tse 2007)\n",
            f"**Seeds:** {[r['seed'] for r in rows]}\n\n",
            f"## Headline\n\n",
            f"- >=3/4 bindings correct: **{n_pass_3of4}/{n}**\n",
            f"- >=2/4 bindings correct: {n_pass_2of4}/{n}\n",
            f"- >=1/4 (any bind): {n_any}/{n}\n\n",
            f"## Per-key pass rate\n\n",
            f"| Key (target) | Pass rate |\n",
            f"|---|---|\n",
        ]
        for k, np in key_pass.items():
            md.append(f"| {k} ({{'apple':'N','river':'E','mountain':'S','forest':'W'}}[k]) | {np}/{n} |\n".format(k=k))
        # fix the template above — use simple mapping
        md = md[:-len(key_pass)]
        target_map = {"apple": "N", "river": "E", "mountain": "S", "forest": "W"}
        for k in ["apple", "river", "mountain", "forest"]:
            md.append(f"| {k} → {target_map[k]} | {key_pass[k]}/{n} |\n")
        md.append("\n## Per-seed details\n\n")
        md.append("| Seed | Total | apple | river | mountain | forest |\n")
        md.append("|---|---|---|---|---|---|\n")
        for r in rows:
            row = [f"{r['seed']}", f"{r['n_correct']}/4"]
            for k in ["apple", "river", "mountain", "forest"]:
                d = r[k]
                ok = "✓" if d.get("correct") else "✗"
                row.append(f"{ok} {d.get('got_value', '?')}")
            md.append("| " + " | ".join(row) + " |\n")

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("".join(md), encoding="utf-8")
        print(f"\n[OUT] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
