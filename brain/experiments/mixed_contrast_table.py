# Capability: Episodic recall
# Record: brain/research/2026-11-14-sall-contrast-mixed-1.5b.md (pre-registration, X0-X4)
"""Paired mixed-day table: mixed_day_probe --arms c,s,r --self-q without (tellmix_s*.json) and with --speaker-contrast 3
(contmix_s*.json), same AWS batch. Primary arm r (S_all + size bar, gain 0.5); arm s reported.

  python -m brain.experiments.mixed_contrast_table <dir>
"""
import glob
import json
import os
import sys
from fractions import Fraction


def load(d, pre):
    R = {}
    for p in sorted(glob.glob(os.path.join(d, f"{pre}_s*.json"))):
        x = json.load(open(p))
        R[x["seed"]] = x
    return R


def cnt(x, arm, kind, st=None):
    return sum(r[arm]["ok"] for r in x["rows"] if r["kind"] == kind and (st is None or r.get("set") == st))


def verdict(T, C, arm="r"):
    seeds = sorted(set(T) & set(C))
    S = len(seeds)
    m = lambda R, kind, st=None: Fraction(sum(cnt(R[s], arm, kind, st) for s in seeds), S)
    rows = [(a, b) for s in seeds for a, b in zip(T[s]["rows"], C[s]["rows"])]
    same = sum(a[arm]["recalled"] == b[arm]["recalled"] for a, b in rows) / max(1, len(rows))
    x0 = S == 6 and all(not T[s].get("speaker_contrast") and C[s].get("speaker_contrast") == 3 for s in seeds) \
        and same >= 0.95
    x1 = m(C, "told") - m(T, "told") >= 1 and sum(cnt(C[s], arm, "told") >= cnt(T[s], arm, "told") for s in seeds) >= 5
    x2 = m(C, "told", "slept") >= m(T, "told", "slept") - Fraction(1, 2)
    x3 = m(C, "never") <= m(T, "never") + Fraction(1, 2)
    x4 = m(C, "self") >= m(T, "self") - Fraction(1, 2)
    v = "VOID" if not x0 else ("GO" if (x1 and x2 and x3 and x4) else "NO-GO")
    return {"X0": x0, "same": same, "X1": x1, "X2": x2, "X3": x3, "X4": x4, "verdict": v, "m": m, "seeds": seeds}


def main():
    d = sys.argv[1]
    T, C = load(d, "tellmix"), load(d, "contmix")
    for arm in ("s", "r"):
        v = verdict(T, C, arm)
        m = v["m"]
        print(f"== arm {arm} ({'S0' if arm == 's' else 'S_all'})  same route {v['same']:.3f}")
        for s in v["seeds"]:
            f = lambda kind, st=None: f"{cnt(T[s], arm, kind, st)}/{cnt(C[s], arm, kind, st)}"
            print(f"  {s:<5} told {f('told'):<6} slept {f('told', 'slept'):<5} new {f('told', 'new'):<5} para "
                  f"{f('para'):<6} false {f('never'):<6} self {f('self')}")
        for name, R in (("tell", T), ("contrast", C)):
            print(f"  mean {name:<8}: told {float(m(R, 'told')):.2f}  slept {float(m(R, 'told', 'slept')):.2f}  new "
                  f"{float(m(R, 'told', 'new')):.2f}  para {float(m(R, 'para')):.2f}  false {float(m(R, 'never')):.2f}  "
                  f"self {float(m(R, 'self')):.2f}")
    v = verdict(T, C, "r")
    for k, t in (("X0", "contrast recorded, r routes as before"), ("X1", "r told +1, >= 5/6"), ("X2", "r slept >= -0.5"),
                 ("X3", "r false <= +0.5"), ("X4", "r self >= -0.5")):
        print(f"{k} ({t}): {'holds' if v[k] else 'FAILS'}")
    print(f"VERDICT (primary arm r): {v['verdict']}")


if __name__ == "__main__":
    main()
