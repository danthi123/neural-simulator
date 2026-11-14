# Capability: Episodic recall
# Record: brain/research/2026-11-14-speaker-contrast-2-1.5b.md (pre-registration, S0-S3)
"""Speaker-contrast follow-up: pick (format, gain) on dev only, judge on the fresh 36-fact bank.
Reads every dev_*.json and fresh_*.json under <dir> (rows de-duplicated by format/gain/q).

  python -m brain.experiments.speaker_contrast_table <dir>
"""
import glob
import json
import os
import sys
from collections import Counter

KINDS = ("orig", "para_a", "para_b")


def load(d, pre):
    seen, rows = set(), []
    for p in sorted(glob.glob(os.path.join(d, f"{pre}_*.json"))):
        for r in json.load(open(p))["rows"]:
            k = (r["format"], r["gain"], r["q"])
            if k not in seen:
                seen.add(k)
                rows.append(r)
    return rows


def right(rows, f, g):
    return sum(r["correct"] for r in rows if r["format"] == f and r["gain"] == g and r["kind"] in KINDS)


def echo(rows, f, g):
    return sum(r["class"] == "echo" for r in rows if r["format"] == f and r["gain"] == g)


def mu(f):
    return 0.0 if f == "tell" else int(f[len("contrast"):]) / 100


def verdict(dev, fresh):
    cells = sorted({(r["format"], r["gain"]) for r in dev})
    contrast = [c for c in cells if c[0] != "tell"]
    best = max(contrast, key=lambda c: (right(dev, *c), -mu(c[0]), -c[1]))
    gt = max([c for c in cells if c[0] == "tell"], key=lambda c: (right(dev, *c), -c[1]))
    s0 = right(fresh, *gt) > 0 and all((c in {(r["format"], r["gain"]) for r in fresh}) for c in (best, gt))
    s1 = right(fresh, *best) >= right(fresh, *gt) + 8
    s2 = echo(fresh, *best) * 2 <= echo(fresh, *gt)
    s3 = sum(r["class"] == "generic" for r in fresh if (r["format"], r["gain"]) == best) <= \
        sum(r["class"] == "generic" for r in fresh if (r["format"], r["gain"]) == gt) + 5
    v = "VOID" if not s0 else ("LEAD" if (s1 and s2 and s3) else "NOT A LEAD")
    return {"best": best, "tell": gt, "S0": s0, "S1": s1, "S2": s2, "S3": s3, "verdict": v}


def main():
    d = sys.argv[1]
    dev, fresh = load(d, "dev"), load(d, "fresh")
    for name, rows in (("DEV", dev), ("FRESH", fresh)):
        print(f"{name} right of 108 (echo / generic)")
        for f, g in sorted({(r["format"], r["gain"]) for r in rows}, key=lambda c: (mu(c[0]), c[1])):
            cl = Counter(r["class"] for r in rows if r["format"] == f and r["gain"] == g)
            print(f"  {f:<12} gain {g:<5} {right(rows, f, g):>3}  ({cl['echo']} / {cl['generic']})")
    v = verdict(dev, fresh)
    print(f"\ndev pick: {v['best']}  tell: {v['tell']}")
    for k, t in (("S0", "cells present"), ("S1", "fresh right pick >= tell + 8"), ("S2", "fresh echoes pick <= tell / 2"),
                 ("S3", "fresh generic pick <= tell + 5")):
        print(f"{k} ({t}): {'holds' if v[k] else 'FAILS'}")
    print(f"VERDICT: {v['verdict']}")


if __name__ == "__main__":
    main()
