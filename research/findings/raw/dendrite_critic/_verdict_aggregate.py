#!/usr/bin/env python3
"""Aggregate the shortcut-#9 dendrite-critic deploy nav table (dendcritic vs baseline vs lesion + ctrl)."""
import json, os, statistics as st
D = os.path.dirname(os.path.abspath(__file__))

def load(arm, s):
    p = os.path.join(D, f"{arm}_seed{s}.json")
    if not (os.path.exists(p) and os.path.getsize(p) > 0):
        return None
    d = json.load(open(p))
    return {
        "sum": sum(d["mean_distance_quarters"]),
        "q": [round(x, 3) for x in d["mean_distance_quarters"]],
        "atgoal": d.get("n_steps_at_goal"),
        "striov": (lambda r: round(sum(r) / len(r), 1) if r else None)(d.get("striov_rate_log", [])),
        "snc": (lambda r: round(sum(r) / len(r), 1) if r else None)(d.get("snc_rate_log", [])),
        "cw_final": round(d.get("critic_weight_final", 0), 2),
    }

seeds = [42, 43, 44]
arms = ["dendcritic", "baseline", "lesion"]
print(f"{'arm':<11} {'seed':>4} {'sum':>7} {'atgoal':>6} {'striov':>7} {'snc':>5} {'cw_final':>9}  quarters")
agg = {a: [] for a in arms}
for a in arms:
    for s in seeds:
        r = load(a, s)
        if r is None:
            print(f"{a:<11} {s:>4}   ==INCOMPLETE==")
            continue
        agg[a].append(r["sum"])
        print(f"{a:<11} {s:>4} {r['sum']:>7.3f} {r['atgoal']:>6} {r['striov']:>7} {r['snc']:>5} {r['cw_final']:>9}  {r['q']}")
print()
print("MEANS (sum, lower=better):")
for a in arms:
    if agg[a]:
        m = st.mean(agg[a]); sd = st.pstdev(agg[a]) if len(agg[a]) > 1 else 0.0
        print(f"  {a:<11}: mean={m:.3f}  sd={sd:.3f}  n={len(agg[a])}")
# deploy bar check
if agg["dendcritic"] and agg["baseline"]:
    dm, bm = st.mean(agg["dendcritic"]), st.mean(agg["baseline"])
    print()
    print(f"  dendcritic vs baseline: {dm:.3f} vs {bm:.3f}  -> dendcritic is {(bm-dm)/bm*100:.1f}% BETTER than point-neuron baseline")
if agg["dendcritic"] and agg["lesion"]:
    dm, lm = st.mean(agg["dendcritic"]), st.mean(agg["lesion"])
    print(f"  dendcritic vs lesion(value silenced): {dm:.3f} vs {lm:.3f}  -> delta {abs(dm-lm)/dm*100:.1f}% (LESION ~= DENDCRITIC => value NOT load-bearing for nav)")
# attribution control
c = load("ctrl_nmda", 42)
if c:
    print()
    print(f"  ctrl_nmda (baseline + global NMDA, NO dendrite value) seed42: sum={c['sum']:.3f}  striov={c['striov']}  snc={c['snc']}  cw={c['cw_final']}")
