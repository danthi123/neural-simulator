# Capability: Episodic recall
# Record: brain/research/2026-11-14-sall-contrast-1.5b.md (pre-registration, Q0-Q3)
"""Paired table: part A unslept runs (unslept_s*.json, no contrast, arms c/s/r) vs the same with --speaker-contrast 3
(contrast_s*.json). Primary arm r (S_all + size bar, gain 0.5); arm s reported.

  python -m brain.experiments.sall_contrast_table --tell <dir A> --contrast <dir>
"""
import argparse
import glob
import json
import os
from fractions import Fraction


def load(d, pre):
    R = {}
    for p in sorted(glob.glob(os.path.join(d, f"{pre}_s*.json"))):
        x = json.load(open(p))
        R[x["seed"]] = x
    return R


def cnt(x, arm, kind):
    return sum(r[arm]["ok"] for r in x["rows"] if r["kind"] == kind)


def verdict(T, C, arm="r"):
    seeds = sorted(set(T) & set(C))
    S = len(seeds)
    m = lambda R, kind: Fraction(sum(cnt(R[s], arm, kind) for s in seeds), S)
    rows = [(a, b) for s in seeds for a, b in zip(T[s]["rows"], C[s]["rows"])]
    same = sum(a[arm]["recalled"] == b[arm]["recalled"] for a, b in rows) / max(1, len(rows))
    q0 = S == 6 and all(not T[s].get("speaker_contrast") and C[s].get("speaker_contrast") == 3 for s in seeds) \
        and same >= 0.95
    q1 = m(C, "told") - m(T, "told") >= 1 and sum(cnt(C[s], arm, "told") >= cnt(T[s], arm, "told") for s in seeds) >= 5
    q2 = m(C, "never") <= m(T, "never") + Fraction(1, 2)
    q3 = m(C, "self") >= m(T, "self") - Fraction(1, 2)
    v = "VOID" if not q0 else ("GO" if (q1 and q2 and q3) else "NO-GO")
    return {"Q0": q0, "same_route": same, "Q1": q1, "Q2": q2, "Q3": q3, "verdict": v, "m": m, "seeds": seeds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tell", required=True)
    ap.add_argument("--contrast", required=True)
    a = ap.parse_args()
    T, C = load(a.tell, "unslept"), load(a.contrast, "contrast")
    for arm in ("s", "r"):
        v = verdict(T, C, arm)
        print(f"== arm {arm} ({'S0' if arm == 's' else 'S_all'})  same route {v['same_route']:.3f}")
        print("seed   told tell/contrast   para   false   self")
        for s in v["seeds"]:
            f = lambda kind: f"{cnt(T[s], arm, kind)}/{cnt(C[s], arm, kind)}"
            print(f"{s:<6} {f('told'):<20}{f('para'):<7}{f('never'):<8}{f('self')}")
        m = v["m"]
        for name, R in (("tell", T), ("contrast", C)):
            print(f"mean {name:<8}: told {float(m(R, 'told')):.2f}  para {float(m(R, 'para')):.2f}  "
                  f"false {float(m(R, 'never')):.2f}  self {float(m(R, 'self')):.2f}")
    v = verdict(T, C, "r")
    print()
    for k, t in (("Q0", "contrast recorded, arm r routes as before (>= 95%)"), ("Q1", "r told contrast - tell >= 1, >= 5/6"),
                 ("Q2", "r false contrast <= tell + 0.5"), ("Q3", "r self contrast >= tell - 0.5")):
        print(f"{k} ({t}): {'holds' if v[k] else 'FAILS'}")
    print(f"VERDICT (primary arm r): {v['verdict']}")


if __name__ == "__main__":
    main()
