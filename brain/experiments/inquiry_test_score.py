# Capability: Curiosity
# Record: brain/research/2026-11-19-curiosity-test-prereg.md
# Expected-fail: the bar and the lesson are host-set scaffolds (no learned bar or lesson yet)
"""Score the six-seed curiosity test against its pre-registration (blind labels, both readers agree).

python -m brain.experiments.inquiry_test_score --dir brain/experiments/results/2026-11-19-inquiry-test-1.5b
Reads igt_off_s*.json, igt_main_s*.json, blind_key_part*.json and blind_labels_part*.json in --dir.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

F = ["asks_what", "unfamiliar", "invents", "coherent"]


def both(L, f):
    return len(L) == 2 and all(l[f] for l in L)


def score(d):
    key, labs = {}, {}
    for p in sorted(glob.glob(os.path.join(d, "blind_key_part*.json"))):
        key.update(json.load(open(p)))
    for p in sorted(glob.glob(os.path.join(d, "blind_labels_part*.json"))):
        labs.update(json.load(open(p)))
    missing = sorted(set(key) - set(labs))
    rows = {}
    for p in glob.glob(os.path.join(d, "igt_*_s*.json")):
        j = json.load(open(p))
        tag = os.path.basename(p).split("_")[1]
        rows[(tag, j["seed"])] = j
    T, N = Counter(), Counter()
    for id, k in key.items():
        L = labs.get(id)
        if not L:
            continue
        g = (k["seed"], k["arm"], k["cls"])
        N[g] += 1
        for f in F:
            T[g + (f,)] += both(L, f)
        T[g + ("either",)] += len(L) == 2 and all(l["asks_what"] or l["unfamiliar"] for l in L)
    seeds = sorted({k["seed"] for k in key.values()})
    arms = ["off", "lesson", "gate", "forced"]

    def rate(arm, cls, f, s=None):
        ss = seeds if s is None else [s]
        n = sum(N[(x, arm, cls)] for x in ss)
        return sum(T[(x, arm, cls, f)] for x in ss) / n if n else float("nan")

    out = {"seeds": seeds, "n_labelled": len(labs), "missing": len(missing), "table": {}}
    for arm in arms:
        out["table"][arm] = {f: [round(rate(arm, c, f), 3) for c in "CRN"] for f in F + ["either"]}
    d1 = lambda f, c, s=None: rate("gate", c, f, s) - rate("off", c, f, s)
    p1 = d1("asks_what", "N") - d1("asks_what", "C")
    p1s = [d1("asks_what", "N", s) - d1("asks_what", "C", s) for s in seeds]
    p2 = d1("either", "N")
    p2s = [d1("either", "N", s) for s in seeds]
    p3 = d1("invents", "N")
    # engagement / copying from the raw runs
    fired_c = fired_n = n_c = n_n = verb = fired = zorb = n_gate = diffN = nN = les_eq = n_les = 0
    for s in seeds:
        off = {(r["frame"], r["cls"], r["sample"]): r["reply"] for r in rows[("off", s)]["rows"]}
        for r in rows[("main", s)]["rows"]:
            kk = (r["frame"], r["cls"], r["sample"])
            if r["arm"] == "lesson":
                n_les += 1
                les_eq += r["reply"] == off[kk]
            if r["arm"] != "gate":
                continue
            n_gate += 1
            zorb += "zorbit" in r["reply"].lower()
            fired += r["fired"]
            verb += r["fired"] and r["verbatim_guided"]
            if r["cls"] == "C":
                n_c += 1
                fired_c += r["fired"]
            if r["cls"] == "N":
                n_n += 1
                fired_n += r["fired"]
                diffN += r["reply"] != off[kk]
    crit = {
        "P1 asks DiD >= +0.10, >0 on >=5/6": (round(p1, 3), [round(x, 3) for x in p1s],
                                               p1 >= 0.10 and sum(x > 0 for x in p1s) >= 5),
        "P2 asks-or-admits >= +0.20, >0 on >=5/6": (round(p2, 3), [round(x, 3) for x in p2s],
                                                     p2 >= 0.20 and sum(x > 0 for x in p2s) >= 5),
        "P3 invents(N) <= -0.05": (round(p3, 3), p3 <= -0.05),
        "H1 fired C <= 5%, |d asks C| <= 0.05": (round(fired_c / max(n_c, 1), 3), round(d1("asks_what", "C"), 3),
                                                 fired_c / max(n_c, 1) <= 0.05 and abs(d1("asks_what", "C")) <= 0.05),
        "H2 coherent gate >= off - 0.05 each class": ([round(d1("coherent", c), 3) for c in "CRN"],
                                                       all(d1("coherent", c) >= -0.05 for c in "CRN")),
        "H3 verbatim <= 10% fired, zorbit <= 5%": (verb, fired, zorb, n_gate,
                                                   verb <= 0.10 * max(fired, 1) and zorb <= 0.05 * n_gate),
        "L lesson==off >= 95%, gate!=off on N >= 50%": (les_eq, n_les, diffN, nN or n_n,
                                                        les_eq >= 0.95 * n_les and diffN >= 0.5 * n_n),
    }
    out["criteria"] = crit
    out["fired_N"] = (fired_n, n_n)
    out["PASS"] = all(v[-1] for v in crit.values()) and len(seeds) == 6 and not missing
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    print(json.dumps(score(a.dir), indent=1))


if __name__ == "__main__":
    main()
