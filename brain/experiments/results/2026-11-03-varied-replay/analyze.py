# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-02-sleep-0.1b-six-seeds.md
"""Aggregate the varied-replay six-seed JSONs (pre-registered in ../2026-11-03-varied-replay.md).

  python brain/experiments/results/2026-11-03-varied-replay/analyze.py
"""
import json
import pathlib
import sys

D = pathlib.Path(__file__).parent
SEEDS = [42, 43, 44, 100, 101, 102]
ARMS = ["no_sleep", "sleep_true", "sleep_varied", "sleep_shuffled"]
LEAK = {4, 11}  # 0-indexed items 5 and 12 (rewordings reuse their PARA wording)
CLEAN = [i for i in range(12) if i not in LEAK]


def main():
    R = {s: json.loads((D / f"s{s}_0.1b_varied.json").read_text())["arms"] for s in SEEDS}

    def row(a, s):
        r = R[s][a]
        return (r["recall"], r["paraphrase"], sum(r["paraphrase_items"][i] for i in CLEAN), r["false_recall"],
                r["generic_kl"])

    print("Cells: recall/12 / paraphrase/12 / clean paraphrase/10 / false recall/4 / KL")
    print("| seed | " + " | ".join(ARMS) + " |\n|---|" + "---|" * len(ARMS))
    for s in SEEDS:
        print(f"| {s} | " + " | ".join("%d / %d / %d / %d / %.3f" % row(a, s) for a in ARMS) + " |")
    m = {a: [sum(row(a, s)[k] for s in SEEDS) / 6 for k in range(5)] for a in ARMS}
    print("| mean | " + " | ".join("%.1f / %.1f / %.1f / %.1f / %.3f" % tuple(m[a]) for a in ARMS) + " |")

    cp = lambda a, s: row(a, s)[2]
    c1n = sum(cp("sleep_varied", s) > cp("sleep_true", s) for s in SEEDS)
    c1t = sum(cp("sleep_varied", s) == cp("sleep_true", s) for s in SEEDS)
    gain = m["sleep_varied"][2] - m["sleep_true"][2]
    c2n = sum(cp("sleep_varied", s) > cp("sleep_shuffled", s) for s in SEEDS)
    c = [c1n >= 5 and gain >= 1.0, c2n == 6, m["sleep_varied"][0] >= m["sleep_true"][0] - 0.5,
         m["sleep_varied"][3] <= m["sleep_true"][3] + 0.5]
    print(f"\nC1 clean para varied>true on {c1n}/6 seeds (ties {c1t}), mean gain {gain:+.2f} -> {c[0]}")
    print(f"C2 clean para varied>shuffled on {c2n}/6 seeds -> {c[1]}")
    print(f"C3 recall varied {m['sleep_varied'][0]:.2f} vs true {m['sleep_true'][0]:.2f} -> {c[2]}")
    print(f"C4 false recall varied {m['sleep_varied'][3]:.2f} vs true {m['sleep_true'][3]:.2f} -> {c[3]}")

    print("\nPer-item: seeds passing /6; * = leaky item (excluded from clean)")
    print("| item | true para | varied para | shuffled para | true recall | varied recall | shuffled recall |")
    print("|---|---|---|---|---|---|---|")
    for i in range(12):
        n = lambda a, k: sum(R[s][a][k][i] for s in SEEDS)
        print(f"| {i + 1}{'*' if i in LEAK else ''} | {n('sleep_true', 'paraphrase_items')} | "
              f"{n('sleep_varied', 'paraphrase_items')} | {n('sleep_shuffled', 'paraphrase_items')} | "
              f"{n('sleep_true', 'recall_items')} | {n('sleep_varied', 'recall_items')} | "
              f"{n('sleep_shuffled', 'recall_items')} |")

    print()
    for a in ["sleep_true", "sleep_varied", "sleep_shuffled"]:
        rec = sum(R[s][a]["recall_items"][i] for s in SEEDS for i in CLEAN)
        both = sum(R[s][a]["recall_items"][i] and R[s][a]["paraphrase_items"][i] for s in SEEDS for i in CLEAN)
        para_only = sum((not R[s][a]["recall_items"][i]) and R[s][a]["paraphrase_items"][i]
                        for s in SEEDS for i in CLEAN)
        print(f"{a}: clean item-seeds recalled {rec}/60, also paraphrased {both}; paraphrased without recall {para_only}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
