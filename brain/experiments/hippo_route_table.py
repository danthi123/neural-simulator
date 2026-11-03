# Capability: Memory selection (pattern completion and separation)
# Record: brain/experiments/hippo_route_probe.py (the measurement this tabulates)
"""Tabulate hippo_route_probe JSON: per key, forced top-1 accuracy (right episode), margins, and dev-fit bars.

  python -m brain.experiments.hippo_route_table <json or json.gz> [--top 12] [--show S0,S_all] [--fit]

Per key: test orig / para forced top-1 accuracy (of 12), mean right-minus-best-wrong margin on told cues, how many told
cues fire and pick the right episode under (a) S0's class bar 0.12 / 0.02 and (b) the bar fit on the dev banks only
(grid and objective of sall_live_bar_fit: J = sel(orig + para_a) - fire(untold + never)), never-told fire under (b).
"""
import argparse
import gzip
import json

import numpy as np

from brain.experiments.sall_live_bar_fit import BARS, MARGINS


def decide(s, bar, margin):
    o = np.argsort(-np.asarray(s), kind="stable")
    top, run = s[o[0]], (s[o[1]] if len(o) > 1 else -1.0)
    return (top >= bar and top - run >= margin), int(o[0])


def right_idx(eps, r):
    return eps.index(r["right"]) if r["right"] else None


def sel_fire(bank, v, kinds, bar, margin):
    eps, n, sel, fire = bank["episodes"], 0, 0, 0
    for r in bank["rows"]:
        if r["kind"] not in kinds:
            continue
        f, j = decide(r["scores"][v], bar, margin)
        n += 1
        fire += f
        sel += f and j == right_idx(eps, r)
    return sel, fire, n


def fit(dev, v):
    best = None
    for bar in BARS:
        for m in MARGINS:
            s = [sel_fire(b, v, ("orig", "para_a"), bar, m) for b in dev]
            u = [sel_fire(b, v, ("untold", "never"), bar, m) for b in dev]
            J = sum(x[0] for x in s) / sum(x[2] for x in s) - sum(x[1] for x in u) / sum(x[2] for x in u)
            k = (round(J, 9), bar, m)
            if best is None or k > best:
                best = k
    return best


def margins(bank, v, kinds):
    eps, out = bank["episodes"], []
    for r in bank["rows"]:
        if r["kind"] in kinds:
            s = list(r["scores"][v])
            i = right_idx(eps, r)
            rs = s.pop(i)
            out.append(rs - max(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--show", default="S0,S_all,att_all,ffn_all,last_all,pool_all")
    ap.add_argument("--fit", action="store_true", help="dev-fit a bar per shown/top key (slow: grid)")
    a = ap.parse_args()
    d = json.load(gzip.open(a.json, "rt") if a.json.endswith(".gz") else open(a.json))
    test = d["banks"]["test"]
    dev = [b for k, b in d["banks"].items() if k.startswith("dev")]
    eps = test["episodes"]
    # fidelity: live S0 path vs read-out S0 (routing must agree)
    agree = sum(int(np.argmax(r["live_S0"])) == int(np.argmax(r["scores"]["S0"])) for r in test["rows"])
    maxdiff = max(abs(x - y) for r in test["rows"] for x, y in zip(r["live_S0"], r["scores"]["S0"]))
    print(f"model {d['model']}  fidelity: live S0 top-1 == read-out S0 top-1 on {agree}/{len(test['rows'])} cues, "
          f"max |score diff| {maxdiff:.4f}")
    acc = {}
    for v in d["variants"]:
        o = sel_fire(test, v, ("orig",), -9, 0)[0]
        p = sel_fire(test, v, ("para",), -9, 0)[0]
        do = [sel_fire(b, v, ("orig",), -9, 0)[0] for b in dev]
        dp = [sel_fire(b, v, ("para_a",), -9, 0)[0] for b in dev]
        db = [sel_fire(b, v, ("para_b",), -9, 0)[0] for b in dev]
        acc[v] = (o, p, sum(do) / max(len(dev), 1), sum(dp) / max(len(dev), 1), sum(db) / max(len(dev), 1))
    shown = [v for v in a.show.split(",") if v in acc]
    ranked = sorted(acc, key=lambda v: (acc[v][0] + acc[v][1] + acc[v][2] + acc[v][3], v), reverse=True)[:a.top]
    print("\nForced top-1 right episode (of 12). test orig / test PARA | dev mean orig / para_a / para_b | "
          "test told margin mean (min)")
    for v in shown + [x for x in ranked if x not in shown]:
        m = margins(test, v, ("orig",))
        print(f"{v:9s} {acc[v][0]:2d} / {acc[v][1]:2d} | {acc[v][2]:5.2f} / {acc[v][3]:5.2f} / {acc[v][4]:5.2f} | "
              f"{np.mean(m):+.3f} ({min(m):+.3f})")
    print("\nTest told cues: who wins under S0 (forced), with top score and margin over runner-up")
    for r in test["rows"]:
        if r["kind"] != "orig":
            continue
        s = r["scores"]["S0"]
        o = np.argsort(-np.asarray(s))
        print(f"  {r['q'][:40]:40s} -> {eps[o[0]][6:40]:34s} top {s[o[0]]:.3f} gap {s[o[0]] - s[o[1]]:.3f}"
              f"{'' if eps[o[0]] == r['right'] else '  WRONG'}")
    if a.fit and dev:
        print("\nDev-fit bar per key (dev banks only), then on TEST: told sel orig / PARA, never-told fire (/16)")
        for v in shown + [x for x in ranked[:4] if x not in shown]:
            J, bar, m = fit(dev, v)
            so = sel_fire(test, v, ("orig",), bar, m)[0]
            sp = sel_fire(test, v, ("para",), bar, m)[0]
            fn = sel_fire(test, v, ("never",), bar, m)[1]
            s0o = sel_fire(test, v, ("orig",), 0.12, 0.02)[0]
            s0p = sel_fire(test, v, ("para",), 0.12, 0.02)[0]
            s0n = sel_fire(test, v, ("never",), 0.12, 0.02)[1]
            print(f"{v:9s} fit bar {bar:.3f} margin {m:.3f} devJ {J:.2f} | test {so}/{sp} never {fn} "
                  f"| at 0.12/0.02: {s0o}/{s0p} never {s0n}")


if __name__ == "__main__":
    main()
