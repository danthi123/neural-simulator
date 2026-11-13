# Capability: Metacognition
# Record: brain/research/2026-11-13-confidence-gate-1.5b.md (pre-registration, G0-G4)
"""Table + registered criteria for confidence_gate_probe JSONs (one per seed).

Gate h: answer with c's reply (no reinstatement) when signal >= t, else with f's reply. t is fitted leave-one-seed-out:
on the other seeds' told + paraphrase rows, the t that maximizes correct answers (ties: the highest t, i.e. the most
reinstatement); applied to the held-out seed. Never-told rows are never used for fitting.

  python -m brain.experiments.confidence_gate_table <dir> [--glob 'cgate_s*.json'] [--signal shift]
"""
import argparse
import glob
import json
import os
from fractions import Fraction

PRIMARY = "mean_logp"
SECONDARY = ("shift", "cue", "first_logp", "min_logp")


def auc(pos, neg):
    if not pos or not neg:
        return None
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def gated(row, sig, t):
    return row["c"] if row[sig] >= t else row["f"]


def fit(rows, sig):
    tr = [r for r in rows if r["kind"] in ("told", "para")]
    cands = sorted({r[sig] for r in tr}) + [float("inf")]
    best = max(cands, key=lambda t: (sum(gated(r, sig, t)["ok"] for r in tr), t))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--glob", default="cgate_s*.json")
    ap.add_argument("--signal", default=PRIMARY)
    a = ap.parse_args()
    R = {}
    for p in sorted(glob.glob(os.path.join(a.dir, a.glob))):
        d = json.load(open(p))
        R[d["seed"]] = d["rows"]
    seeds = sorted(R)
    print(f"seeds {seeds}")
    # G0 instrument: f's reply when nothing fired equals c's reply
    nf = [(r["c"]["reply"] == r["f"]["reply"]) for s in seeds for r in R[s] if not r["f"]["recalled"]]
    g0 = sum(nf) / max(1, len(nf))
    print(f"G0 f==c when nothing fired: {sum(nf)}/{len(nf)} = {g0:.3f}  (bar >= 0.95)  {'holds' if g0 >= 0.95 else 'FAILS'}")
    # G1 signal: AUC c-correct vs c-wrong on told + para
    for sig in (a.signal,) + tuple(x for x in SECONDARY if x != a.signal):
        tp = lambda s: [r for r in R[s] if r["kind"] in ("told", "para")]
        pooled = auc([r[sig] for s in seeds for r in tp(s) if r["c"]["ok"]],
                     [r[sig] for s in seeds for r in tp(s) if not r["c"]["ok"]])
        per = [auc([r[sig] for r in tp(s) if r["c"]["ok"]], [r[sig] for r in tp(s) if not r["c"]["ok"]]) for s in seeds]
        per = [x for x in per if x is not None]
        pm = sum(per) / len(per) if per else None
        tag = "PRIMARY" if sig == a.signal else "secondary"
        line = f"AUC[{sig}] ({tag}) pooled {pooled:.3f}, per-seed mean {pm:.3f}" if pm is not None else f"AUC[{sig}] undefined"
        if sig == a.signal:
            g1 = pooled is not None and pooled >= 0.70 and pm is not None and pm >= 0.65
            line += f"  G1 (pooled >= 0.70 and per-seed mean >= 0.65): {'holds' if g1 else 'FAILS'}"
        print(line)
    # Gated recall, leave-one-seed-out
    sig = a.signal
    per = {}
    for s in seeds:
        t = fit([r for o in seeds if o != s for r in R[o]], sig)
        cnt = lambda arm, kind: sum((gated(r, sig, t) if arm == "h" else r[arm])["ok"] for r in R[s] if r["kind"] == kind)
        per[s] = {arm: {k: cnt(arm, k) for k in ("told", "para", "never")} for arm in ("c", "f", "h")}
        per[s]["t"] = t
        per[s]["reinstated"] = sum(1 for r in R[s] if r["kind"] in ("told", "para") and r[sig] < t and r["f"]["recalled"])
    q = lambda arm, k: Fraction(sum(per[s][arm][k] for s in seeds), len(seeds))
    print("\nseed  t        told c/f/h   para c/f/h   false(of 12) c/f/h   reinstated(told+para, h)")
    for s in seeds:
        p = per[s]
        print(f"{s:<5} {p['t']:<8.4f} {p['c']['told']:>2}/{p['f']['told']:>2}/{p['h']['told']:>2}     "
              f"{p['c']['para']:>2}/{p['f']['para']:>2}/{p['h']['para']:>2}     "
              f"{p['c']['never']:>2}/{p['f']['never']:>2}/{p['h']['never']:>2}            {p['reinstated']}")
    for k in ("told", "para", "never"):
        print(f"mean {k}: c {float(q('c', k)):.2f}  f {float(q('f', k)):.2f}  h {float(q('h', k)):.2f}")
    g2 = q("h", "told") >= q("c", "told") - Fraction(1, 2)
    g3 = q("h", "told") - q("f", "told") >= 1 and sum(per[s]["h"]["told"] >= per[s]["f"]["told"] for s in seeds) >= 5
    g4 = q("h", "never") <= q("f", "never") + Fraction(1, 2)
    print(f"G2 (h told >= c - 0.5): {'holds' if g2 else 'FAILS'}")
    print(f"G3 (h - f told >= 1, h >= f on >= 5/6 seeds): {'holds' if g3 else 'FAILS'}")
    print(f"G4 (false h <= f + 0.5): {'holds' if g4 else 'FAILS'}")


if __name__ == "__main__":
    main()
