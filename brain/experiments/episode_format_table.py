# Capability: Episodic recall
# Record: brain/research/2026-11-14-lived-episode-1.5b.md (pre-registration, E0-E3)
"""Table + registered criteria for episode_format_probe JSONs (dev_*.json, test_*.json in one dir).

  python -m brain.experiments.episode_format_table <dir>
"""
import glob
import json
import os
import sys
from collections import Counter

DEV_KINDS = ("orig", "para_a", "para_b")


def load(d, pre):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, f"{pre}_*.json"))):
        rows += json.load(open(p))["rows"]
    return rows


def right(rows, fmt, g, kinds):
    return sum(r["correct"] for r in rows if r["format"] == fmt and r["gain"] == g and r["kind"] in kinds)


def echoes(rows, fmt, g):
    return sum(r["class"] == "echo" for r in rows if r["format"] == fmt and r["gain"] == g)


def fit(dev, fmt):
    gs = sorted({r["gain"] for r in dev if r["format"] == fmt})
    return max(gs, key=lambda g: (right(dev, fmt, g, DEV_KINDS), -g))


def verdict(dev, test, other="lived"):
    gt, gl = fit(dev, "tell"), fit(dev, other)
    e0 = right(dev, "tell", 0.5, DEV_KINDS) == 58 and right(test, "tell", 0.5, ("orig",)) == 7 and \
        right(test, "tell", 0.5, ("para",)) == 7
    e1 = right(dev, other, gl, DEV_KINDS) >= right(dev, "tell", gt, DEV_KINDS) + 10
    e2 = right(test, other, gl, ("orig",)) >= right(test, "tell", gt, ("orig",)) + 2 and \
        right(test, other, gl, ("para",)) >= right(test, "tell", gt, ("para",))
    e3 = echoes(dev, other, gl) * 2 <= echoes(dev, "tell", gt)
    v = "VOID" if not e0 else ("LEAD" if (e1 and e2 and e3) else "NOT A LEAD")
    return {"G_tell": gt, "G_lived": gl, "E0": e0, "E1": e1, "E2": e2, "E3": e3, "verdict": v}


def main():
    d = sys.argv[1]
    other = sys.argv[2] if len(sys.argv) > 2 else "lived"
    dev, test = load(d, "dev"), load(d, "test")
    for name, rows, kinds in (("DEV (of 108)", dev, DEV_KINDS), ("TEST told / para (of 12)", test, None)):
        print(name)
        for fmt in ("tell", other):
            for g in sorted({r["gain"] for r in rows if r["format"] == fmt}):
                cl = Counter(r["class"] for r in rows if r["format"] == fmt and r["gain"] == g)
                n = (f"{right(rows, fmt, g, kinds):>3}" if kinds else
                     f"{right(rows, fmt, g, ('orig',)):>2} / {right(rows, fmt, g, ('para',)):>2}")
                print(f"  {fmt:<6} gain {g:<5} right {n}  {dict(cl)}")
    v = verdict(dev, test, other)
    print(f"\nG* tell {v['G_tell']}, {other} {v['G_lived']}")
    for k, t in (("E0", "tell 0.5 = 11-13 refit (58, 7, 7)"), ("E1", "dev right lived >= tell + 10"),
                 ("E2", "test told lived >= tell + 2, para lived >= tell"), ("E3", "dev echoes lived <= tell / 2")):
        print(f"{k} ({t}): {'holds' if v[k] else 'FAILS'}")
    print(f"VERDICT: {v['verdict']}")


if __name__ == "__main__":
    main()
