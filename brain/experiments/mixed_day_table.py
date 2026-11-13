# Capability: Metacognition
# Record: brain/research/2026-11-13-mixed-day-gate-1.5b.md (pre-registration, M0-M5)
"""Table + registered criteria for mixed_day_probe JSONs (one per seed).

Primary gate h: c's reply when mean_logp >= T_FIXED, else f's reply. T_FIXED is the bar fitted on ALL six seeds of the
11-13 run (confidence_gate_table.fit over brain/experiments/results/2026-11-13-confidence-gate-1.5b/cgate_s*.json,
signal mean_logp), fixed before this run; nothing is fitted on this run's data for the verdict. Secondary (report only):
the leave-one-seed-out bar on this run, a gate on reply length (n_tok >= 6, fitted the same way on 11-13), the oracle.

  python -m brain.experiments.mixed_day_table <dir> [--glob 'mixday_s*.json']
"""
import argparse
import glob
import json
import os
import random
from fractions import Fraction

from brain.experiments.confidence_gate_table import auc, fit, gated

T_FIXED = -0.16947272575063316   # 11-13 pooled fit, mean_logp
T_NTOK = 6                       # 11-13 pooled fit, n_tok (control)
N_PERM = 2000


def told(rows, pick, st=None):
    return sum(pick(r)["ok"] for r in rows if r["kind"] == "told" and (st is None or r["set"] == st))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--glob", default="mixday_s*.json")
    a = ap.parse_args()
    R, SL = {}, {}
    for p in sorted(glob.glob(os.path.join(a.dir, a.glob))):
        d = json.load(open(p))
        R[d["seed"]] = d["rows"]
        SL[d["seed"]] = d.get("slept", "?")
    seeds = sorted(R)
    S = len(seeds)
    print(f"seeds {seeds}")
    nf = [r["c"]["reply"] == r["f"]["reply"] for s in seeds for r in R[s] if not r["f"]["recalled"]]
    m0 = sum(nf) / max(1, len(nf)) >= 0.95
    print(f"M0 f==c when nothing fired: {sum(nf)}/{len(nf)}  (bar >= 0.95)  {'holds' if m0 else 'FAILS'}")
    tp = [r for s in seeds for r in R[s] if r["kind"] in ("told", "para")]
    print(f"AUC[mean_logp] c-right vs c-wrong, told+para: {auc([r['mean_logp'] for r in tp if r['c']['ok']], [r['mean_logp'] for r in tp if not r['c']['ok']])}")
    H = lambda t, sig="mean_logp": (lambda r: gated(r, sig, t))
    arms = {"c": lambda r: r["c"], "f": lambda r: r["f"], "h": H(T_FIXED), "len": H(T_NTOK, "n_tok"),
            "orc": lambda r: r["c"] if r["c"]["ok"] else r["f"]}
    loso = {s: fit([r for o in seeds if o != s for r in R[o]], "mean_logp") for s in seeds}
    print("\nseed  slept  told-slept c/f/h   told-new c/f/h   told c/f/h/loso/len/oracle   para c/f/h   false c/f/h")
    per = {}
    for s in seeds:
        rs = R[s]
        per[s] = {k: told(rs, fn) for k, fn in arms.items()}
        per[s]["loso"] = told(rs, H(loso[s]))
        sp = lambda st: "/".join(f"{told(rs, arms[k], st):>1}" for k in "cfh")
        para = "/".join(str(sum(arms[k](r)["ok"] for r in rs if r["kind"] == "para")) for k in "cfh")
        fal = "/".join(str(sum(arms[k](r)["ok"] for r in rs if r["kind"] == "never")) for k in "cfh")
        p = per[s]
        print(f"{s:<5} {SL[s]:<6} "
              f"{sp('slept'):<18} {sp('new'):<16} {p['c']}/{p['f']}/{p['h']}/{p['loso']}/{p['len']}/{p['orc']:<16} "
              f"{para:<12} {fal}")
    q = lambda fn, kind="told", st=None: Fraction(sum(sum(fn(r)["ok"] for r in R[s] if r["kind"] == kind and (st is None or r["set"] == st)) for s in seeds), S)
    for k in ("c", "f", "h", "len", "orc"):
        print(f"mean {k:<4} told {float(q(arms[k])):.2f}  (slept {float(q(arms[k], st='slept')):.2f}, new "
              f"{float(q(arms[k], st='new')):.2f})  para {float(q(arms[k], 'para')):.2f}  false {float(q(arms[k], 'never')):.2f}")
    print(f"mean loso told {sum(per[s]['loso'] for s in seeds) / S:.2f}")
    m1 = q(arms["f"], st="new") - q(arms["c"], st="new") >= Fraction(3, 2)
    m2 = q(arms["h"]) - q(arms["c"]) >= 1 and sum(per[s]["h"] >= per[s]["c"] for s in seeds) >= 5
    m3 = q(arms["h"]) - q(arms["f"]) >= Fraction(1, 2) and sum(per[s]["h"] >= per[s]["f"] for s in seeds) >= 5
    m4 = q(arms["h"], "never") <= q(arms["f"], "never") + Fraction(1, 2)
    # M5: within-seed permutation of mean_logp over told+para rows, fixed bar, fixed RNG
    rng = random.Random(1113)
    obs = sum(per[s]["h"] for s in seeds)
    ge = 0
    for _ in range(N_PERM):
        tot = 0
        for s in seeds:
            rs = [r for r in R[s] if r["kind"] in ("told", "para")]
            v = [r["mean_logp"] for r in rs]
            rng.shuffle(v)
            tot += sum(gated({**r, "mean_logp": x}, "mean_logp", T_FIXED)["ok"] for r, x in zip(rs, v) if r["kind"] == "told")
        ge += tot >= obs
    p5 = (ge + 1) / (N_PERM + 1)
    m5 = p5 < 0.05
    print(f"\nM1 instrument (new facts: f - c >= 1.5): {'holds' if m1 else 'FAILS (VOID)'}")
    print(f"M2 (h - c told >= 1, h >= c on >= 5/6): {'holds' if m2 else 'FAILS'}")
    print(f"M3 (h - f told >= 0.5, h >= f on >= 5/6): {'holds' if m3 else 'FAILS'}")
    print(f"M4 (false h <= f + 0.5): {'holds' if m4 else 'FAILS'}")
    print(f"M5 (permutation p < 0.05): p = {p5:.4f}  {'holds' if m5 else 'FAILS'}")
    v = "VOID" if not (m0 and m1) else ("LEAD" if (m2 and m3 and m4 and m5) else "NOT A LEAD")
    print(f"VERDICT: {v}")


if __name__ == "__main__":
    main()
