# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-03-high-use-ceiling/README.md
"""Re-score the six-seed high-use results (hand-set strength) on facts reachable in context only.

  python -m brain.experiments.rescore_reachable --ceiling <ceiling json> --results <dir with s*.json> --para
Reachable = the original question is answered in context (ceiling json). Paraphrase is scored against facts whose reworded
question is also reachable in context. Prints per seed and mean, per arm and k group: hits/reachable facts in that group."""
import argparse
import glob
import json

from brain.experiments.facts import FACTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ceiling", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    rows = {r["key"]: r for r in json.load(open(a.ceiling))["rows"]}
    keys = [f[2] for f in FACTS]
    out = {"per_seed": {}, "reachable": [k for k in keys if rows[k]["original"]],
           "para_reachable": [k for k in keys if rows[k]["para"]]}
    for p in sorted(glob.glob(f"{a.results}/s*.json")):
        d = json.load(open(p))
        seed = str(d["seed"])
        cell = {}
        for arm, r in d["arms"].items():
            cell[arm] = {}
            for kind, items, ok in (("recall", r["recall_items"], "original"), ("paraphrase", r["paraphrase_items"], "para")):
                for k in ("1", "3", "10"):
                    sel = [i for i, key in enumerate(keys) if str(d["k_of_fact"][key]) == k and rows[key][ok]]
                    cell[arm][f"{kind}_k{k}"] = [sum(items[i] for i in sel), len(sel)]
        out["per_seed"][seed] = cell
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
