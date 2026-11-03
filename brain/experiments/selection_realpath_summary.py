# Capability: Memory selection (pattern completion and separation)
# Record: brain/experiments/results/2026-11-03-meaning-selection-realpath.md
"""Tables and the pre-registered criteria P1-P4 from selection_realpath JSONs.

  python -m brain.experiments.selection_realpath_summary <json> [<json> ...]
"""
import json
import sys

CUES = ("orig", "para_a", "para_b")
ARMS = ("S0_forced", "S_all_forced", "S0_default", "S_all_default", "S0_refit", "S_all_refit", "oracle", "none")
GRAM = {12: 0.69, 24: 0.76}  # 11-03 Gram-probe S_all para_b selection means (0.1B has 12 layers, 0.4B 24)


def fmt(xs):
    return f"{sum(xs) / len(xs):.2f} ({min(xs):.2f}-{max(xs):.2f})"


def jtest(arm):
    told = sum(arm[c]["sel"] for c in CUES) / 3
    untold = (arm["untold_fact"]["fire"] + arm["never_told"]["fire"]) / 2
    return told - untold, told, untold


def main():
    for path in sys.argv[1:]:
        r = json.load(open(path))
        S = r["seeds"]
        print(f"\n### {r['model'].split('/')[-1]}  seeds {list(S)}")
        print("dev fits:", json.dumps(r["fits"]))
        print("| arm | sel orig | sel para_a | sel para_b | rec orig | rec para_a | rec para_b | untold fire | J_test |")
        print("|---|---|---|---|---|---|---|---|---|")
        for a in ARMS:
            row = []
            for m in ("sel", "rec"):
                for c in CUES:
                    xs = [s["arms"][a][c].get(m) for s in S.values()]
                    row.append(fmt(xs) if None not in xs else "")
            if "untold_fact" in next(iter(S.values()))["arms"][a]:
                row.append(fmt([jtest(s["arms"][a])[2] for s in S.values()]))
                row.append(fmt([jtest(s["arms"][a])[0] for s in S.values()]))
            else:
                row += ["", ""]
            print(f"| {a} | " + " | ".join(row) + " |")
        for k in ("S0", "S_all"):
            print(f"AUROC told-vs-untold top score {k}: {fmt([s[f'auroc_{k}'] for s in S.values()])}")

        def wins(fa, fb):
            d = [fa(s) - fb(s) for s in S.values()]
            return sum(x > 0 for x in d), sum(d) / len(d), [round(x, 3) for x in d]

        print("P1 forced selection S_all > S0:")
        for c in ("para_a", "para_b"):
            print(f"  {c}: {wins(lambda s: s['arms']['S_all_forced'][c]['sel'], lambda s: s['arms']['S0_forced'][c]['sel'])}")
        w = wins(lambda s: s["arms"]["S_all_refit"]["para_b"]["rec"], lambda s: s["arms"]["S0_refit"]["para_b"]["rec"])
        print(f"P2 refit recall para_b S_all > S0: {w}  PASS={w[0] >= 5 and w[1] >= 0.10}")
        w = wins(lambda s: jtest(s["arms"]["S_all_refit"])[0], lambda s: jtest(s["arms"]["S0_refit"])[0])
        print(f"P3 J_test refit S_all > S0: {w}  PASS={w[0] >= 5}")
        m = sum(s["arms"]["S_all_forced"]["para_b"]["sel"] for s in S.values()) / len(S)
        g = GRAM.get(r["n_layers"])
        print(f"P4 forced S_all para_b sel {m:.3f} vs Gram probe {g}: PASS={g is not None and abs(m - g) <= 0.10}")
        print("bytes per episode (after):", r["bytes_per_episode_after"], "delta bytes:", r["delta_bytes"])


if __name__ == "__main__":
    main()
