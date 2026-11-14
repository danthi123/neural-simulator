# Capability: Episodic recall
# Record: brain/research/2026-11-13-live-gain-1.5b.md (pre-registration, L0-L4)
"""Table + registered criteria for mixed_day_probe --gain-arm JSONs (arms c, f gain 4, r gain 0.5).

  python -m brain.experiments.live_gain_table <dir> [--glob 'livegain_s*.json']
"""
import argparse
import glob
import json
import os
import re
from fractions import Fraction

from brain.experiments.strict_scorer import has_key

ABSTAIN = re.compile(r"\b(i don't know|i do not know|i'm sorry|i am sorry|i don't have|not mentioned|no information)\b",
                     re.I)


def cnt(rows, arm, kind, st=None):
    return sum(r[arm]["ok"] for r in rows if r["kind"] == kind and (st is None or r["set"] == st))


def same_route(r):
    a, b = r["f"]["recalled"], r["r"]["recalled"]
    return (a or None) == (b or None)


def verdict(R):
    seeds = sorted(R)
    S = len(seeds)
    mean = lambda arm, kind, st=None: Fraction(sum(cnt(R[s], arm, kind, st) for s in seeds), S)
    rows = [r for s in seeds for r in R[s]]
    l0 = sum(same_route(r) for r in rows) / max(1, len(rows)) >= 0.95
    per = {s: {a: cnt(R[s], a, "told", "new") for a in "cfr"} for s in seeds}
    l1 = mean("r", "told", "new") - mean("c", "told", "new") >= Fraction(3, 2) and \
        sum(per[s]["r"] > per[s]["c"] for s in seeds) >= 5
    l2 = mean("r", "told", "new") - mean("f", "told", "new") >= 1 and sum(per[s]["r"] >= per[s]["f"] for s in seeds) >= 5
    l3 = mean("r", "never") <= mean("f", "never") + Fraction(1, 2)
    l4 = mean("r", "told", "slept") >= mean("f", "told", "slept") - Fraction(1, 2)
    v = "VOID" if not l0 else ("GO" if (l1 and l2 and l3 and l4) else "NO-GO")
    return {"L0": l0, "L1": l1, "L2": l2, "L3": l3, "L4": l4, "verdict": v, "per": per, "mean": mean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--glob", default="livegain_s*.json")
    a = ap.parse_args()
    R = {}
    for p in sorted(glob.glob(os.path.join(a.dir, a.glob))):
        d = json.load(open(p))
        assert d.get("gain_arm"), p
        R[d["seed"]] = d["rows"]
    v = verdict(R)
    m = v["mean"]
    print(f"seeds {sorted(R)}")
    print("seed   new c/f/r   slept c/f/r   para c/f/r   false c/f/r   abstain c/f/r")
    for s in sorted(R):
        rs = R[s]
        f3 = lambda kind, st=None: "/".join(str(cnt(rs, x, kind, st)) for x in "cfr")
        ab = "/".join(str(sum(bool(ABSTAIN.search(r[x]["reply"] or "")) for r in rs if r["kind"] != "never")) for x in "cfr")
        print(f"{s:<6} {f3('told', 'new'):<11} {f3('told', 'slept'):<13} {f3('para'):<12} {f3('never'):<13} {ab}")
    for x in "cfr":
        print(f"mean {x}: new {float(m(x, 'told', 'new')):.2f}  slept {float(m(x, 'told', 'slept')):.2f}  "
              f"told {float(m(x, 'told')):.2f}  para {float(m(x, 'para')):.2f}  false {float(m(x, 'never')):.2f}")
    rows = [r for s in R for r in R[s]]
    kw = "/".join(str(sum(has_key(r[x]["reply"] or "", r["key"]) for r in rows if r["kind"] == "told" and r["set"] == "new"))
                  for x in "cfr")
    print(f"report: new told replies containing the fact word (incl. first-person echoes), all seeds c/f/r: {kw}")
    print(f"\nL0 r routes as f: {sum(same_route(r) for r in rows)}/{len(rows)}  {'holds' if v['L0'] else 'FAILS (VOID)'}")
    for k, t in (("L1", "new r - c >= 1.5, r > c on >= 5/6"), ("L2", "new r - f >= 1, r >= f on >= 5/6"),
                 ("L3", "false r <= f + 0.5"), ("L4", "slept r >= f - 0.5")):
        print(f"{k} ({t}): {'holds' if v[k] else 'FAILS'}")
    print(f"VERDICT: {v['verdict']}")


if __name__ == "__main__":
    main()
