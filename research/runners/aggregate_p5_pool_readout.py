"""Aggregate P5 iter AA / iter KK / future ventral semantic runs by
the BEHAVIORAL pool_readout metric (apple_stim -> pool_0 wins,
river_stim -> pool_1 wins). This is the iter AA 4/6 breakthrough
metric and the operationally meaningful one for conversational
demos (which pool fires when concept is stimulated).

Reads naming.pool_readout = {"apple": {"0": spikes, "1": spikes},
"river": {...}} from each seed's JSON output.

Usage:
    python -m research.runners.aggregate_p5_pool_readout \
        --raw-root research/findings/raw/g11_bg/iter_KK \
        --prefix iter_KK_seed --seeds 42,43,44,100,101,102 \
        --out research/findings/2026-05-12-P5-iterKK-multiseed.md \
        --label "iter KK (Tier 1 canon + biological scale)"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_seed(seed: int, prefix: str, root: Path):
    fp = root / f"{prefix}{seed}.json"
    if not fp.exists():
        return None, fp
    return json.loads(fp.read_text(encoding="utf-8")), fp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=str, required=True)
    ap.add_argument("--prefix", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--label", type=str, default="iter ??")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    root = Path(args.raw_root)

    rows = []
    for s in seeds:
        d, fp = load_seed(s, args.prefix, root)
        if d is None:
            print(f"  seed {s}: {fp} MISSING")
            continue
        try:
            pr = d["naming"]["pool_readout"]
            apple = pr["apple"]
            river = pr["river"]
            apple_p0 = float(apple.get("0", 0.0))
            apple_p1 = float(apple.get("1", 0.0))
            river_p0 = float(river.get("0", 0.0))
            river_p1 = float(river.get("1", 0.0))
        except KeyError as e:
            print(f"  seed {s}: missing key {e}")
            continue
        apple_ok = apple_p0 > apple_p1
        river_ok = river_p1 > river_p0
        bidir = apple_ok and river_ok
        row = {
            "seed": s,
            "apple_p0": apple_p0, "apple_p1": apple_p1,
            "river_p0": river_p0, "river_p1": river_p1,
            "apple_margin": apple_p0 - apple_p1,
            "river_margin": river_p1 - river_p0,
            "apple_ok": apple_ok,
            "river_ok": river_ok,
            "bidir": bidir,
            "n_neurons": d.get("n_neurons", 0),
            "n_synapses": d.get("n_synapses", 0),
            "total_seconds": d.get("total_seconds", 0.0),
        }
        # Iter OO_visual: optional visual_pool_readout (TEST 2c)
        vpr = d.get("naming", {}).get("visual_pool_readout")
        if vpr:
            v_apple = vpr.get("apple", {})
            v_river = vpr.get("river", {})
            row["v_apple_p0"] = float(v_apple.get("0", 0.0))
            row["v_apple_p1"] = float(v_apple.get("1", 0.0))
            row["v_river_p0"] = float(v_river.get("0", 0.0))
            row["v_river_p1"] = float(v_river.get("1", 0.0))
            row["v_apple_ok"] = row["v_apple_p0"] > row["v_apple_p1"]
            row["v_river_ok"] = row["v_river_p1"] > row["v_river_p0"]
            row["v_bidir"] = row["v_apple_ok"] and row["v_river_ok"]
        rows.append(row)

    if not rows:
        print("No seeds found.")
        return 1

    n = len(rows)
    n_apple = sum(int(r["apple_ok"]) for r in rows)
    n_river = sum(int(r["river_ok"]) for r in rows)
    n_bidir = sum(int(r["bidir"]) for r in rows)
    avg_total = sum(r["total_seconds"] for r in rows) / n
    has_visual = any("v_bidir" in r for r in rows)
    if has_visual:
        n_v_apple = sum(int(r.get("v_apple_ok", False)) for r in rows)
        n_v_river = sum(int(r.get("v_river_ok", False)) for r in rows)
        n_v_bidir = sum(int(r.get("v_bidir", False)) for r in rows)

    print(f"=== {args.label} ===")
    print(f"Seeds: {[r['seed'] for r in rows]}")
    print(f"AUDITORY (CA3-stim) recognition:")
    print(f"  Apple: {n_apple}/{n}")
    print(f"  River: {n_river}/{n}")
    print(f"  BIDIRECTIONAL: {n_bidir}/{n}")
    if has_visual:
        print(f"VISUAL-ONLY (retina-stim) recognition:")
        print(f"  Apple: {n_v_apple}/{n}")
        print(f"  River: {n_v_river}/{n}")
        print(f"  BIDIRECTIONAL: {n_v_bidir}/{n}")
    print(f"Mean wall clock: {avg_total/60:.1f} min/seed")
    print()
    for r in rows:
        line = (
            f"  seed {r['seed']:>3}: "
            f"audio apple {r['apple_margin']:+5.0f} "
            f"({'OK' if r['apple_ok'] else 'X '}) "
            f"river {r['river_margin']:+5.0f} "
            f"({'OK' if r['river_ok'] else 'X '}) "
            f"BIDIR={'YES' if r['bidir'] else 'no '}"
        )
        if "v_bidir" in r:
            line += (
                f" | visual apple {r['v_apple_p0']-r['v_apple_p1']:+5.0f} "
                f"({'OK' if r['v_apple_ok'] else 'X '}) "
                f"river {r['v_river_p1']-r['v_river_p0']:+5.0f} "
                f"({'OK' if r['v_river_ok'] else 'X '}) "
                f"BIDIR={'YES' if r['v_bidir'] else 'no '}"
            )
        print(line)

    if args.out:
        md = []
        md.append(f"# P5 multi-seed pool_readout: {args.label}\n\n")
        md.append("**Date:** 2026-05-12\n")
        md.append("**Catalog:** G.11 (Hickok & Poeppel ventral stream) + "
                  "G.13 (Wernicke's area)\n")
        md.append("**Metric:** behavioral pool spike count "
                  "(stim concept tag → which lang_output_pool wins)\n")
        md.append(f"**Seeds:** {[r['seed'] for r in rows]}\n")
        md.append(f"**Headline:** apple {n_apple}/{n}, river {n_river}/{n}, "
                  f"**BIDIRECTIONAL {n_bidir}/{n}**\n")
        md.append(f"**Mean wall clock:** {avg_total/60:.1f} min/seed\n\n")
        if rows:
            md.append(f"**Arch:** {rows[0]['n_neurons']:,} neurons, "
                      f"{rows[0]['n_synapses']:,} synapses\n\n")

        md.append("## Per-seed pool spike counts\n\n")
        md.append("| Seed | apple→p0 | apple→p1 | "
                  "Apple OK | river→p0 | river→p1 | "
                  "River OK | Bidir |\n")
        md.append("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            md.append(
                f"| {r['seed']} | "
                f"{r['apple_p0']:.0f} | {r['apple_p1']:.0f} | "
                f"{'OK' if r['apple_ok'] else 'X'} | "
                f"{r['river_p0']:.0f} | {r['river_p1']:.0f} | "
                f"{'OK' if r['river_ok'] else 'X'} | "
                f"{'YES' if r['bidir'] else 'no'} |\n"
            )

        md.append("\n## Comparison to iter AA (toy scale baseline)\n\n")
        md.append("| Iter | apple | river | BIDIR |\n")
        md.append("|---|---|---|---|\n")
        md.append("| AA (toy, 100/200) | 6/6 | 4/6 | 4/6 |\n")
        md.append(f"| **{args.label}** | "
                  f"**{n_apple}/{n}** | **{n_river}/{n}** | "
                  f"**{n_bidir}/{n}** |\n")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(md), encoding="utf-8")
        print(f"\n[OUT] {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
