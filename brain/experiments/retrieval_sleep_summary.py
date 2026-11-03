# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-03-confab-m3-retrieval-sleep/README.md
"""Aggregate retrieval_sleep_probe jsons against the pre-registered M3 criteria.

  python -m brain.experiments.retrieval_sleep_summary <dir>
"""
import glob
import json
import os
import sys

ARMS = ("no_sleep", "sleep_true", "sleep_retrieval", "sleep_retrieval_shuffled")
COLS = ("recall", "recall_strict", "paraphrase", "false_recall", "false_recall_fresh24", "generic_kl")


def main(d):
    runs = {}
    for p in sorted(glob.glob(os.path.join(d, "s*.json"))):
        r = json.load(open(p))
        runs[r["seed"]] = r
    print(f"seeds: {sorted(runs)}")
    for arm in ARMS:
        rows = [runs[s]["arms"][arm] for s in sorted(runs) if arm in runs[s]["arms"]]
        if not rows:
            continue
        print(arm, " ".join(f"{c}={sum(x[c] for x in rows) / len(rows):.3g}" for c in COLS),
              "| recall per seed", [x["recall"] for x in rows])
    gains = [runs[s]["arms"]["sleep_retrieval"]["recall"] - runs[s]["arms"]["sleep_true"]["recall"] for s in sorted(runs)]
    fr_r = [runs[s]["arms"]["sleep_retrieval"]["false_recall"] for s in sorted(runs)]
    fr_t = [runs[s]["arms"]["sleep_true"]["false_recall"] for s in sorted(runs)]
    shuf = [runs[s]["arms"]["sleep_retrieval_shuffled"]["recall"] for s in sorted(runs)]
    n = len(gains)
    mg = sum(gains) / n
    print("gain retrieval - true per seed:", gains, f"mean {mg:+.2f}; seeds >= +2: {sum(g >= 2 for g in gains)}/{n}")
    print(f"false recall 4 never-told mean: retrieval {sum(fr_r) / n:.2f} vs true {sum(fr_t) / n:.2f}")
    print(f"shuffled-retrieval recall mean {sum(shuf) / n:.2f}")
    for s in sorted(runs):
        r = runs[s]
        a = r["arms"]["sleep_retrieval"]
        print(f"s{s}: cues {r['n_cues_total']} (= test question {r['cues_equal_to_test_question']}), facts with key in a "
              f"teacher answer {r['facts_with_key_in_some_teacher_answer']}/12, failed first40 {a.get('failed_first40')} "
              f"last40 {a.get('failed_last40')}")
    lead = (n == 6 and sum(g >= 2 for g in gains) >= 5 and sum(fr_r) <= sum(fr_t) and sum(shuf) / n <= 0.5)
    falsified = mg < 1 or sum(fr_r) > sum(fr_t)
    print("VERDICT:", "LEAD" if lead else ("FALSIFIED at this scale" if falsified else "INCONCLUSIVE"),
          "" if n == 6 else f"(PARTIAL: {n}/6 seeds)")


if __name__ == "__main__":
    main(sys.argv[1])
