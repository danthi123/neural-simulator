# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-03-hippo-routing-1.5b.md (pre-registered criteria F1-F5 this prints)
"""Read live_brain_probe JSONs run with --fix-key (arms a-g) and print the pre-registered F1-F5 checks.

  python -m brain.experiments.hippo_route_live_table <dir> --glob 'hroute_s*.json'
Recall / paraphrase = recall_correct / paraphrase_correct (strict_scorer.correct_answer). F1 reads each told row's
`recalled` against the fact's day-1 sentence.
"""
import argparse
import glob
import json
import statistics as st

from brain.experiments.facts import FACTS

ARMS = ["a_intact_sleep", "b_intact_nosleep", "c_hippo_off", "d_hippo_off_consol_off", "e_shuffled_hippo_off",
        "f_fix_sleep", "g_fix_nosleep"]


def routing(rows):
    told = [r for r in rows if r["kind"] == "told"]
    right = sum(r["recalled"] == f"User: {s}" for r, (s, _, _) in zip(told, FACTS))
    wrong = sum(r["recalled"] is not None and r["recalled"] != f"User: {s}" for r, (s, _, _) in zip(told, FACTS))
    return right, wrong


def main(d, pattern):
    R = {}
    for f in sorted(glob.glob(f"{d}/{pattern}")):
        j = json.load(open(f))
        R[j["seed"]] = j["arms"]
    seeds = sorted(R)
    print("seeds", seeds)
    m = lambda a, k: st.mean(R[s][a][k] for s in seeds)
    for k in ("recall_correct", "paraphrase_correct", "false_recall", "echo"):
        print(f"\n{k} (per seed, mean)")
        for a in ARMS:
            print(f"  {a:24s} {[R[s][a][k] for s in seeds]} mean {m(a, k):.2f}")
    rt = {a: [routing(R[s][a]["rows"]) for s in seeds] for a in ("a_intact_sleep", "b_intact_nosleep", "f_fix_sleep", "g_fix_nosleep")}
    print("\nrouting of told questions (right, wrong) per seed:", rt)
    rc = "recall_correct"
    f1 = all(r >= 9 and w <= 1 for r, w in rt["f_fix_sleep"] + rt["g_fix_nosleep"])
    f2 = m("f_fix_sleep", rc) - m("a_intact_sleep", rc) >= 2.0 and \
        sum(R[s]["f_fix_sleep"][rc] > R[s]["a_intact_sleep"][rc] for s in seeds) >= 5
    f3 = m("f_fix_sleep", rc) >= m("c_hippo_off", rc) - 1.0
    f4 = m("f_fix_sleep", "false_recall") <= m("a_intact_sleep", "false_recall") + 0.5
    f5 = m("g_fix_nosleep", rc) - m("b_intact_nosleep", rc) >= 2.0 and \
        sum(R[s]["g_fix_nosleep"][rc] > R[s]["b_intact_nosleep"][rc] for s in seeds) >= 5
    print(f"\nF1 (instrument: routing) {f1}  F2 (f - a >= 2, 5/6) {f2}  F3 (f >= c - 1) {f3}  "
          f"F4 (false f <= a + 0.5) {f4}  F5 (g - b >= 2, 5/6) {f5}")
    print("verdict:", "INVALID (routing differs from the CPU measurement)" if not f1 else
          ("LEAD" if f2 and f3 and f4 and f5 and len(seeds) == 6 else "not a lead (see failing criteria)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--glob", default="hroute_s*.json")
    a = ap.parse_args()
    main(a.dir, a.glob)
