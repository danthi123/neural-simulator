# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-03-high-use-ceiling/README.md
"""Print per-seed and mean tables from use_strength_probe jsons (--use-strength) or the reachable rescore (--rescore)."""
import glob
import json
import sys


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def rescore(path):
    d = json.load(open(path))
    print("reachable (orig):", d["reachable"], "\npara-reachable:", d["para_reachable"])
    for kind in ("recall", "paraphrase"):
        for arm in ("sleep_use", "sleep_equal"):
            print(f"\n{kind} {arm}  (hits/reachable facts in group)  k=1  k=3  k=10")
            tot = {"1": [0, 0], "3": [0, 0], "10": [0, 0]}
            for s, c in d["per_seed"].items():
                cells = [c[arm][f"{kind}_k{k}"] for k in ("1", "3", "10")]
                for k, v in zip(("1", "3", "10"), cells):
                    tot[k][0] += v[0]
                    tot[k][1] += v[1]
                print(f"  s{s}: " + "  ".join(f"{h}/{n}" for h, n in cells))
            print("  pooled: " + "  ".join(f"{tot[k][0]}/{tot[k][1]}" for k in ("1", "3", "10")))


def probe(folder):
    seeds = []
    for p in sorted(glob.glob(f"{folder}/s*.json")):
        seeds.append(json.load(open(p)))
    print("seed | arm | recall k1/k3/k10 (of 4) | total | para k1/k3/k10 | false/4 | KL")
    agg = {}
    for d in seeds:
        for arm, r in d["arms"].items():
            rb, pb = r["recall_by_k"], r["paraphrase_by_k"]
            print(f"{d['seed']} | {arm} | {rb['1']}/{rb['3']}/{rb['10']} | {r['recall']} | {pb['1']}/{pb['3']}/{pb['10']} | "
                  f"{r['false_recall']} | {r['generic_kl']:.3f}")
            a = agg.setdefault(arm, {"k1": [], "k3": [], "k10": [], "tot": [], "p1": [], "p3": [], "p10": [], "f": [], "kl": []})
            a["k1"].append(rb["1"]); a["k3"].append(rb["3"]); a["k10"].append(rb["10"]); a["tot"].append(r["recall"])
            a["p1"].append(pb["1"]); a["p3"].append(pb["3"]); a["p10"].append(pb["10"]); a["f"].append(r["false_recall"])
            a["kl"].append(r["generic_kl"])
    print("\nMEANS over", len(seeds), "seeds")
    for arm, a in agg.items():
        print(arm, {k: round(mean(v), 3) for k, v in a.items()})
    if "sleep_use" in agg:
        print("k=10 4/4 seeds (use):", sum(x == 4 for x in agg["sleep_use"]["k10"]), "/", len(seeds),
              "| (equal):", sum(x == 4 for x in agg["sleep_equal"]["k10"]), "/", len(seeds))
        print("k=10 use minus equal, per seed:", [u - e for u, e in zip(agg["sleep_use"]["k10"], agg["sleep_equal"]["k10"])])


if __name__ == "__main__":
    (rescore if sys.argv[1] == "--rescore" else probe)(sys.argv[2])
