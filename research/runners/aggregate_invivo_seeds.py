"""Aggregate in-vivo new-vocab binding multi-seed results.

Reads research/findings/raw/g11_bg/invivo_binding/invivo_seed{N}.json
and reports per-variant pass rate (≥3/4 bindings correct = seed PASS).

Usage:
    python -m research.runners.aggregate_invivo_seeds \
        --raw-root research/findings/raw/g11_bg/invivo_binding \
        --seeds 42,43,44,100,101,102 \
        --out research/findings/2026-05-12-invivo-binding-multiseed.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def aggregate(seeds, root):
    seed_data = {}
    for s in seeds:
        fp = root / f"invivo_seed{s}.json"
        if not fp.exists():
            print(f"  seed {s}: {fp} MISSING", file=sys.stderr)
            continue
        seed_data[s] = json.loads(fp.read_text(encoding="utf-8"))
    return seed_data


def render_markdown(seed_data, root):
    if not seed_data:
        return "# No results found\n"
    n = len(seed_data)

    # Collect per-variant per-seed accuracy
    variants = set()
    for s, d in seed_data.items():
        for r in d.get("results", []):
            variants.add(r["variant"])
    variants = sorted(variants)

    # Per-seed table
    md = []
    md.append("# In-vivo new-vocab binding multi-seed result\n\n")
    md.append("**Date:** 2026-05-12\n")
    md.append("**Phase:** Step 1 of realigned plan (Sim as standalone "
              "conversational agent)\n")
    md.append("**Catalog:** McClelland 1995 (CLS) + Buzsáki 2015 (SWR) + "
              "Tse 2007 (schema-supported)\n")
    md.append(f"**Seeds:** {list(seed_data.keys())}\n")
    md.append("**Test bindings:** apple→north, river→east, "
              "mountain→south, forest→west\n\n")

    md.append("## Per-variant pass rate (>=3/4 bindings correct)\n\n")
    md.append("| Variant | " + " | ".join(f"s{s}" for s in seed_data) +
              " | Pass rate |\n")
    md.append("|---|" + "|".join("---" for _ in seed_data) + "|---|\n")
    for v in variants:
        row = [f"| {v}"]
        n_pass = 0
        for s in seed_data:
            n_correct = next((r["n_correct"] for r in seed_data[s].get("results", [])
                               if r["variant"] == v), 0)
            n_total = next((r["n_total"] for r in seed_data[s].get("results", [])
                             if r["variant"] == v), 4)
            row.append(f"{n_correct}/{n_total}")
            if n_correct >= 3:  # ≥3/4 bindings correct
                n_pass += 1
        row.append(f"**{n_pass}/{n} ({100*n_pass/n:.0f}%)**")
        md.append(" | ".join(row) + " |\n")
    md.append("\n")

    md.append("## Per-seed detailed binding accuracy\n\n")
    for s, d in seed_data.items():
        md.append(f"### Seed {s}\n\n")
        md.append("| Variant | apple | river | mountain | forest | "
                  "Total |\n")
        md.append("|---|---|---|---|---|---|\n")
        for r in d.get("results", []):
            details = r.get("details", [])
            row = [r["variant"]]
            for key in ["apple", "river", "mountain", "forest"]:
                d_item = next((x for x in details if x["key"] == key), {})
                row.append("✓" if d_item.get("correct") else "✗")
            row.append(f"{r['n_correct']}/{r['n_total']}")
            md.append("| " + " | ".join(row) + " |\n")
        md.append("\n")

    return "".join(md)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=str,
                    default="research/findings/raw/g11_bg/invivo_binding")
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    root = Path(args.raw_root)
    seed_data = aggregate(seeds, root)

    print(f"Aggregated {len(seed_data)} seeds")
    # Per-variant stdout summary
    variants = set()
    for s, d in seed_data.items():
        for r in d.get("results", []):
            variants.add(r["variant"])
    for v in sorted(variants):
        n_pass = 0
        for s in seed_data:
            for r in seed_data[s].get("results", []):
                if r["variant"] == v and r["n_correct"] >= 3:
                    n_pass += 1
        print(f"  {v:>20}: {n_pass}/{len(seed_data)} seeds pass (>=3/4 bindings correct)")

    md = render_markdown(seed_data, root)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n[OUT] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
