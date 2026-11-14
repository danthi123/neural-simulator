# Capability: Episodic recall
# Record: brain/research/2026-11-13-gain-refit-1.5b.md (pre-registration, R0-R3)
"""Table + registered criteria for hippo_gain_refit JSONs.

Fit (dev only): G* = the gain with the most right replies over all 108 dev cues (orig + para_a + para_b); ties go to the
smaller gain. Test (12 facts) is read at G* only for the verdict; the full test curve is report-only.

  python -m brain.experiments.hippo_gain_refit_table <dir>      (reads dev_*.json and test_*.json)
"""
import glob
import json
import os
import sys
from collections import Counter

BASELINE_LIVE = 0.5      # 11-05 arms b and g re-scored (told, of 12)
GAIN4_TEST = 1           # 11-03 oracle gain 4 re-scored (told, of 12)


def load(d, pre):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, f"{pre}_*.json"))):
        rows += json.load(open(p))["rows"]
    return rows


def curve(rows, kinds):
    out = {}
    for g in sorted({r["gain"] for r in rows}):
        rs = [r for r in rows if r["gain"] == g and r["kind"] in kinds]
        out[g] = (sum(r["correct"] for r in rs), len(rs), Counter(r["class"] for r in rs))
    return out


def fit(dev):
    cv = curve(dev, ("orig", "para_a", "para_b"))
    return max(cv, key=lambda g: (cv[g][0], -g))


def verdict(dev, test):
    g0 = [r for r in dev if r["gain"] == 0]
    r0 = sum(r["correct"] for r in g0) <= 3
    gs = fit(dev)
    dv = curve(dev, ("orig", "para_a", "para_b"))[gs][0]
    t_orig = sum(r["correct"] for r in test if r["gain"] == gs and r["kind"] == "orig")
    t_para = sum(r["correct"] for r in test if r["gain"] == gs and r["kind"] == "para")
    r1 = dv >= 27                    # 25% of 108 dev cues
    r2 = t_orig >= 6
    r3 = t_para >= 4
    if not r0:
        v = "VOID"
    elif r1 and r2 and r3:
        v = "LEAD"
    elif not r1:
        v = "NO GAIN WORKS"
    else:
        v = "NOT A LEAD"
    return {"G*": gs, "dev_right": dv, "test_orig": t_orig, "test_para": t_para,
            "R0": r0, "R1": r1, "R2": r2, "R3": r3, "verdict": v}


def main():
    d = sys.argv[1]
    dev, test = load(d, "dev"), load(d, "test")
    print("DEV (right of 108; classes)")
    for g, (n, m, cl) in curve(dev, ("orig", "para_a", "para_b")).items():
        print(f"  gain {g:<4} {n:>3}/{m}  {dict(cl)}")
    print("TEST orig (of 12) / para (of 12), report-only curve")
    co, cp = curve(test, ("orig",)), curve(test, ("para",))
    for g in co:
        print(f"  gain {g:<4} {co[g][0]:>2} / {cp[g][0]:>2}  {dict(co[g][2])}")
    v = verdict(dev, test)
    print(f"\nG* = {v['G*']} (dev right {v['dev_right']}/108)")
    print(f"R0 instrument (gain 0 dev right <= 3): {'holds' if v['R0'] else 'FAILS (VOID)'}")
    print(f"R1 some gain works on dev (>= 27/108): {'holds' if v['R1'] else 'FAILS'}")
    print(f"R2 test told at G* >= 6/12 (live b/g {BASELINE_LIVE}, oracle gain 4 {GAIN4_TEST}): {v['test_orig']} "
          f"{'holds' if v['R2'] else 'FAILS'}")
    print(f"R3 test para at G* >= 4/12: {v['test_para']} {'holds' if v['R3'] else 'FAILS'}")
    print(f"VERDICT: {v['verdict']}")


if __name__ == "__main__":
    main()
