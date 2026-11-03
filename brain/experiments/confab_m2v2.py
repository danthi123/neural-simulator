# Capability: Metacognition
# Record: brain/experiments/results/2026-11-03-confab-m2.md (v1 PARTIAL: DEV DK 150/216, lesson-content intrusion 64/360)
"""Confabulation M2 v2: the lived don't-know exchange reinstates only the brain's OWN turn. Pre-registration + results:
brain/experiments/results/2026-11-03-confab-m2-v2.md.

Phases:
  run     one seed: arms none / lived_self (v1, reproduced in this run) / self_turn (v2); everything else is the v1
          harness unchanged (brain.experiments.confab_m2.run_phase)
  report  pre-registered criteria over the six seed files
"""
import argparse
import glob
import json
import os
import re
import time

import torch

from brain.cortex import DEFAULT_MODEL, Cortex
from brain.experiments.confab_m2 import GATED, SETS, run_phase

ARMS = ("none", "lived_self", "self_turn")
# Frozen at pre-registration: lesson-content intrusion = the reply contains a content stem of the lesson question
# ("What tune does my godfather whistle in the mornings?"). This is the instrument that gave v1's 64/360 (all gated
# replies, DK or not), re-derived from the v1 files before this run.
CONTENT = re.compile(r"godfather|whistl|morning|tune")
V1_DEV_DK, V1_INTRUSION = 150, 64          # v1 committed numbers (report.json / m2_s*.json), the bars to beat
SEEDS = (42, 43, 44, 100, 101, 102)


def content_intrudes(reply):
    return bool(CONTENT.search(reply.lower()))


def report_phase(d):
    runs = {}
    for f in glob.glob(os.path.join(d, "m2v2_s*.json")):
        r = json.load(open(f))
        runs[r["seed"]] = r
    seeds = sorted(runs)
    out = {"seeds_present": seeds, "per_seed": {}, "pooled": {}, "criteria": {}}
    for arm in ARMS:
        pooled = {}
        for st in SETS:
            rows = [x for s in seeds for x in runs[s]["arms"][arm]["rows"] if x["set"] == st]
            pooled[st] = {"n": len(rows), "dk": sum(x["dk"] for x in rows), "recite": sum(x["recite"] for x in rows),
                          "vocative": sum(x["vocative"] for x in rows),
                          "name_intrusion": sum(bool(x["intrusion"]) for x in rows),
                          "content": sum(content_intrudes(x["reply"]) for x in rows),
                          "content_nondk": sum(content_intrudes(x["reply"]) and not x["dk"] for x in rows),
                          "verbatim_guided": sum(x["verbatim_guided"] for x in rows),
                          "lesson_recalled": sum(x["lesson_recalled"] for x in rows)}
        out["pooled"][arm] = pooled
        for s in seeds:
            rows = runs[s]["arms"][arm]["rows"]
            out["per_seed"].setdefault(str(s), {})[arm] = {
                st: {"dk": sum(x["dk"] for x in rows if x["set"] == st),
                     "content": sum(content_intrudes(x["reply"]) for x in rows if x["set"] == st)} for st in SETS}
    P = out["pooled"]
    gated_content = {arm: sum(P[arm][st]["content"] for st in GATED) for arm in ARMS}
    gated_n = sum(P["self_turn"][st]["n"] for st in GATED)
    gen_identical = {arm: sum(x == y for s in seeds for x, y in zip(
        [r["reply"] for r in runs[s]["arms"][arm]["rows"] if r["set"] == "general"],
        [r["reply"] for r in runs[s]["arms"]["none"]["rows"] if r["set"] == "general"])) for arm in ARMS}
    g = P["self_turn"]["general"]
    c = {"Q1_dev_dk": {"self_turn": P["self_turn"]["dev_never"]["dk"], "n": P["self_turn"]["dev_never"]["n"],
                       "bar_gt": V1_DEV_DK, "pass": P["self_turn"]["dev_never"]["dk"] > V1_DEV_DK},
         "Q2_content": {"self_turn": gated_content["self_turn"], "n": gated_n, "bar_lt": V1_INTRUSION,
                        "pass": gated_content["self_turn"] < V1_INTRUSION},
         "Q3_general": {"lesson_recalled": g["lesson_recalled"], "dk_self_turn": g["dk"], "dk_none": P["none"]["general"]["dk"],
                        "identical_to_none": gen_identical["self_turn"], "n": g["n"],
                        "pass": g["lesson_recalled"] <= 4 and g["dk"] <= P["none"]["general"]["dk"] + 2},
         "gated_content_by_arm": gated_content, "general_identical_to_none": gen_identical,
         "v1_reproduced": {"lived_self_dev_dk": P["lived_self"]["dev_never"]["dk"],
                           "lived_self_content": gated_content["lived_self"]},
         "secondary_P1_seeds": {st: sum(out["per_seed"][str(s)]["self_turn"][st]["dk"] >= need for s in seeds)
                                for st, need in (("dev_never", 29), ("fresh3_never", 20))},
         "secondary_recite": {st: P["self_turn"][st]["recite"] for st in GATED},
         "secondary_vocative": {st: P["self_turn"][st]["vocative"] for st in GATED},
         "secondary_v3_dk": P["self_turn"]["fresh3_never"]["dk"]}
    c["six_seeds"] = tuple(seeds) == SEEDS
    c["IMPROVED"] = c["six_seeds"] and c["Q1_dev_dk"]["pass"] and c["Q2_content"]["pass"] and c["Q3_general"]["pass"]
    c["WORSE"] = (P["self_turn"]["dev_never"]["dk"] <= 0.54 * P["self_turn"]["dev_never"]["n"]
                  or gated_content["self_turn"] >= V1_INTRUSION and P["self_turn"]["dev_never"]["dk"] <= V1_DEV_DK)
    out["criteria"] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "report"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--home")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--dir")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    if a.phase == "report":
        res = report_phase(a.dir)
    else:
        res = run_phase(a, Cortex(model_path=a.model, device="cpu"))
        res["model"] = a.model
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
