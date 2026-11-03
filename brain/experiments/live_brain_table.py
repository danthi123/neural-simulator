# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-03-live-brain-12facts.md
"""Read the per-seed live_brain_probe JSONs, print per-seed/per-arm tables and the pre-registered P1-P4 checks.

  python -m brain.experiments.live_brain_table brain/experiments/results/2026-11-03-live-brain-12facts
"""
import glob
import json
import statistics as st
import sys

ARMS = ["a_intact_sleep", "b_intact_nosleep", "c_hippo_off", "d_hippo_off_consol_off", "e_shuffled_hippo_off"]
M = ["recall", "paraphrase", "false_recall", "echo"]


def main(d):
    R = {json.load(open(f))["seed"]: json.load(open(f))["arms"] for f in sorted(glob.glob(d + "/seed*.json"))}
    seeds = sorted(R)
    print("seeds", seeds)
    for m in M + ["recall_lenient"]:
        print(f"\n{m} (per seed {seeds}, mean)")
        for a in ARMS:
            v = [R[s][a][m] for s in seeds]
            print(f"  {a:24s} {v} mean {st.mean(v):.2f}")
    mean = lambda a, m: st.mean(R[s][a][m] for s in seeds)
    p1 = all(R[s]["c_hippo_off"]["recall"] > max(R[s]["d_hippo_off_consol_off"]["recall"], R[s]["e_shuffled_hippo_off"]["recall"]) for s in seeds) \
        and mean("c_hippo_off", "recall") >= 2
    p2 = sum(R[s]["a_intact_sleep"]["recall"] >= R[s]["b_intact_nosleep"]["recall"] for s in seeds) >= 5 \
        and mean("a_intact_sleep", "recall") - mean("b_intact_nosleep", "recall") >= 1
    p3 = mean("a_intact_sleep", "false_recall") <= mean("b_intact_nosleep", "false_recall") + 0.5 \
        and mean("c_hippo_off", "false_recall") <= mean("e_shuffled_hippo_off", "false_recall")
    p4 = mean("c_hippo_off", "paraphrase") > mean("e_shuffled_hippo_off", "paraphrase") + 1
    print(f"\nP1 {p1}  P2 {p2}  P3 {p3}  P4 {p4}")
    print("sleep adds over no-sleep (a-b): recall", round(mean("a_intact_sleep", "recall") - mean("b_intact_nosleep", "recall"), 2),
          "paraphrase", round(mean("a_intact_sleep", "paraphrase") - mean("b_intact_nosleep", "paraphrase"), 2),
          "false", round(mean("a_intact_sleep", "false_recall") - mean("b_intact_nosleep", "false_recall"), 2))


if __name__ == "__main__":
    main(sys.argv[1])
