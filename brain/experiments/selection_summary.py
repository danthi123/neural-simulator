# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-02-record-brief-consolidation-and-selection.md
"""Table from selection_probe JSONs: mean over seeds (min..max) per variant.

  python -m brain.experiments.selection_summary <json> [<json> ...]
"""
import json
import sys

COLS = ("sel_orig", "sel_para_a", "sel_para_b", "auroc", "rec_orig", "rec_para_a", "rec_para_b")
MAIN = ("S0", "S_mid", "S_all", "att_all", "ffn_all", "last_all", "pool_mid", "pool_all",
        "learned_selfq", "learned_shuf", "learned_teacher", "Ssk_all", "learnedS_selfq", "learnedS_shuf",
        "learnedS_teacher", "oracle", "none")


def fmt(xs):
    if not xs:
        return "-"
    return f"{sum(xs) / len(xs):.2f} ({min(xs):.2f}-{max(xs):.2f})"


def main():
    for path in sys.argv[1:]:
        r = json.load(open(path))
        seeds = r["seeds"]
        print(f"\n### {r['model'].split('/')[-1]}  seeds {list(seeds)}  stored {r['n_store']} (chance {1 / r['n_store']:.3f})")
        print("| variant | " + " | ".join(COLS) + " |")
        print("|---|" + "---|" * len(COLS))
        for v in MAIN:
            row = [fmt([s["variants"][v][c] for s in seeds.values() if c in s["variants"].get(v, {})]) for c in COLS]
            print(f"| {v} | " + " | ".join(row) + " |")
        # wins of the best fixed key over the default, per seed
        for v in ("S_all", "learned_selfq", "learned_teacher", "learnedS_selfq"):
            for c in ("sel_para_a", "sel_para_b", "rec_para_a", "rec_para_b"):
                d = [s["variants"][v][c] - s["variants"]["S0"][c] for s in seeds.values() if c in s["variants"].get(v, {})]
                if d:
                    print(f"{v} - S0 on {c}: per seed {[round(x, 3) for x in d]}  (better on {sum(x > 0 for x in d)}/{len(d)})")
        for p, base in (("learned", "pool_all"), ("learnedS", "Ssk_all")):
            if f"{p}_selfq" not in next(iter(seeds.values()))["variants"]:
                continue
            for ctl in (f"{p}_shuf", base):
                d = [s["variants"][f"{p}_selfq"][c] - s["variants"][ctl][c] for s in seeds.values()
                     for c in ("sel_para_a", "sel_para_b")]
                print(f"{p}_selfq - {ctl} on para sel (seeds x 2 cues): mean {sum(d) / len(d):+.3f}, "
                      f"better {sum(x > 0 for x in d)}/{len(d)}")
        print("per-layer selection on para_b (S / pool):")
        L = r["n_layers"]
        for kind in ("S", "pool"):
            print(" ", kind, [round(sum(s["variants"][f"{kind}{l}"]["sel_para_b"] for s in seeds.values()) / len(seeds), 2)
                              for l in range(L)])


if __name__ == "__main__":
    main()
