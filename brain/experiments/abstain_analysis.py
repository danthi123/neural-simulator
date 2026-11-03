# Capability: Metacognition
# Record: research/findings/2026-05-22-attractor-cleanup-biologization-shortcut-3-RESOLVED-abstention-is-a-separate-familiarity-signal-not-a-basin-property.md
"""Read-out analysis for confidence_probe: does the cortex's own confidence separate true recall from confabulation,
and what does an abstention threshold keep and remove?

Separation: AUC (Mann-Whitney) of each confidence signal, correct answers vs confabulations (never-told questions
answered with a learned name), and correct vs every non-correct answer. Abstention: a threshold the brain applies to
its own answer ("answer only if my confidence >= t"), chosen leave-one-seed-out (Youden on the other five seeds:
kept-correct fraction minus kept-confab fraction) and scored on the held-out seed; optionally AND-ed with the
hippocampal familiarity check (score >= the hippocampus's own FAMILIARITY bar) for when the hippocampus is on.

  python -m brain.experiments.abstain_analysis <dir with s<seed>_st<steps>.json> [--out summary.json]
"""
import argparse
import glob
import json
import os

from brain.hippocampus import Hippocampus

SEEDS = [42, 43, 44, 100, 101, 102]
SIGNALS = ["mean_logp", "min_logp", "first_logp", "name_logp", "neg_ent", "shift", "cue_gain", "cue_gain_max", "cue_kl",
           "cue_kl_max", "hippo_score"]


def sig(it, s):
    if s == "neg_ent":
        return -it["mean_ent"]
    if s == "name_logp" and it[s] is None:  # no learned name in the answer: fall back to the least certain token
        return it["min_logp"]
    return it[s]


def auc(pos, neg):
    if not pos or not neg:
        return None
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def youden(items, s):
    cor = [sig(i, s) for i in items if i["cls"] == "correct"]
    con = [sig(i, s) for i in items if i["cls"] == "confab"]
    best, bt = -2.0, None
    for t in sorted(set(cor + con)):
        j = sum(v >= t for v in cor) / max(1, len(cor)) - sum(v >= t for v in con) / max(1, len(con))
        if j > best:
            best, bt = j, t
    return bt


def score(items, keep):
    k = [i for i in items if keep(i)]
    return {"correct": sum(i["cls"] == "correct" for i in k), "confab": sum(i["cls"] == "confab" for i in k),
            "confab_4": sum(i["cls"] == "confab" and i["kind"] == "never" for i in k),
            "wrong_named": sum(i["cls"] == "wrong" and bool(i["named"]) for i in k)}


def analyse(runs):
    pool = [i for r in runs.values() for i in r["items"]]
    out = {"n": {c: sum(i["cls"] == c for i in pool) for c in ("correct", "wrong", "confab", "nt_other")}, "auc": {},
           "auc_per_seed_mean": {}, "abstain": {}}
    sigs = [s for s in SIGNALS if s == "neg_ent" or s in pool[0]] if pool else []
    for s in sigs:
        cor = [sig(i, s) for i in pool if i["cls"] == "correct"]
        out["auc"][s] = {"correct_vs_confab": auc(cor, [sig(i, s) for i in pool if i["cls"] == "confab"]),
                         "correct_vs_all_noncorrect": auc(cor, [sig(i, s) for i in pool if i["cls"] != "correct"])}
        per = [auc([sig(i, s) for i in r["items"] if i["cls"] == "correct"],
                   [sig(i, s) for i in r["items"] if i["cls"] == "confab"]) for r in runs.values()]
        per = [p for p in per if p is not None]
        out["auc_per_seed_mean"][s] = (sum(per) / len(per), len(per)) if per else None
    out["abstain"]["no_readout"] = score(pool, lambda i: True)
    fam = Hippocampus.FAMILIARITY
    out["abstain"]["hippo_bar_only"] = score(pool, lambda i: i["hippo_score"] >= fam)
    for s in sigs:
        tot, tot_h, thr = {}, {}, {}
        for seed, r in runs.items():
            train = [i for sd, rr in runs.items() if sd != seed for i in rr["items"]]
            t = youden(train, s)
            thr[seed] = t
            for d, keep in ((tot, lambda i: sig(i, s) >= t),
                            (tot_h, lambda i: sig(i, s) >= t and i["hippo_score"] >= fam)):
                for k, v in score(r["items"], keep).items():
                    d[k] = d.get(k, 0) + v
        out["abstain"][f"loso_{s}"] = tot
        out["abstain"][f"loso_{s}+hippo"] = tot_h
        out["abstain"][f"loso_{s}_thresholds"] = thr
        # stricter: the held-out item's seed AND question are both unseen when the threshold is set (the hippocampal
        # score is a property of the question, identical on every seed, so leave-one-seed-out alone is in-sample)
        tot = {}
        for seed, r in runs.items():
            for q in {i["q"] for i in r["items"]}:
                train = [i for sd, rr in runs.items() if sd != seed for i in rr["items"] if i["q"] != q]
                t = youden(train, s)
                for k, v in score([i for i in r["items"] if i["q"] == q], lambda i: sig(i, s) >= t).items():
                    tot[k] = tot.get(k, 0) + v
        out["abstain"][f"lsqo_{s}"] = tot
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {}
    for st in sorted({int(os.path.basename(p).split("_st")[1][:-5]) for p in glob.glob(f"{a.dir}/s*_st*.json")}):
        runs = {}
        for seed in SEEDS:
            p = f"{a.dir}/s{seed}_st{st}.json"
            if os.path.exists(p):
                runs[seed] = json.load(open(p))
        res[f"steps_{st}"] = {"seeds": sorted(runs), "per_seed": {sd: r["summary"] for sd, r in runs.items()},
                              **analyse(runs)}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
