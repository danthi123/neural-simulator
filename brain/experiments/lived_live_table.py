# Capability: Episodic recall
# Record: brain/research/2026-11-14-lived-episode-live-1.5b.md (pre-registration, P0-P3)
"""Paired table: unslept_day_probe runs with --episode tell (tell_s*.json) and --episode lived (lived_s*.json), arm s
(S0, gain 0.5 = the 1.5B default). python -m brain.experiments.lived_live_table <dir> [lived|lived_key]"""
import glob
import json
import os
import sys
from fractions import Fraction

from brain.experiments.strict_scorer import has_key, is_echo
from brain.experiments.facts import FACTS

TOLD = {k: s for s, _, k in FACTS}


def load(d, pre):
    R = {}
    for p in sorted(glob.glob(os.path.join(d, f"{pre}_s*.json"))):
        x = json.load(open(p))
        R[x["seed"]] = x
    return R


def cnt(x, kind, arm="s"):
    return sum(r[arm]["ok"] for r in x["rows"] if r["kind"] == kind)


def echoes(x, arm="s"):
    return sum(is_echo(r[arm]["reply"] or "", r["key"], TOLD[r["key"]]) and has_key(r[arm]["reply"] or "", r["key"])
               for r in x["rows"] if r["kind"] in ("told", "para"))


def verdict(T, L):
    seeds = sorted(set(T) & set(L))
    S = len(seeds)
    m = lambda R, kind: Fraction(sum(cnt(R[s], kind) for s in seeds), S)
    p0 = S == 6 and all(T[s]["episode"] == "tell" and L[s]["episode"] in ("lived", "lived_key") for s in seeds) and \
        all("\n\nAssistant: " in t for s in seeds for t in L[s]["stored"]) and \
        all("Assistant:" not in t for s in seeds for t in T[s]["stored"])
    part_a = {42: 3, 43: 2, 44: 6, 100: 6, 101: 6, 102: 6}
    p0 = p0 and all(cnt(T[s], "told") == part_a.get(s) for s in seeds)
    p1 = m(L, "told") - m(T, "told") >= 1 and sum(cnt(L[s], "told") >= cnt(T[s], "told") for s in seeds) >= 5
    p2 = m(L, "never") <= m(T, "never") + Fraction(1, 2)
    p3 = m(L, "self") >= m(T, "self") - Fraction(1, 2)
    v = "VOID" if not p0 else ("GO" if (p1 and p2 and p3) else "NO-GO")
    return {"P0": p0, "P1": p1, "P2": p2, "P3": p3, "verdict": v, "m": m, "seeds": seeds}


def main():
    d = sys.argv[1]
    other = sys.argv[2] if len(sys.argv) > 2 else "lived"   # lived | lived_key (file prefix)
    T, L = load(d, "tell"), load(d, other)
    v = verdict(T, L)
    print("seed   told tell/lived   para   false   self   echoes(told+para)")
    for s in v["seeds"]:
        f = lambda kind: f"{cnt(T[s], kind)}/{cnt(L[s], kind)}"
        print(f"{s:<6} {f('told'):<17}{f('para'):<7}{f('never'):<8}{f('self'):<7}{echoes(T[s])}/{echoes(L[s])}")
    m = v["m"]
    for name, R in (("tell", T), ("lived", L)):
        print(f"mean {name:<5}: told {float(m(R, 'told')):.2f}  para {float(m(R, 'para')):.2f}  false "
              f"{float(m(R, 'never')):.2f}  self {float(m(R, 'self')):.2f}")
    for k, t in (("P0", "formats recorded"), ("P1", "told lived - tell >= 1, >= 5/6"), ("P2", "false lived <= tell + 0.5"),
                 ("P3", "self lived >= tell - 0.5")):
        print(f"{k} ({t}): {'holds' if v[k] else 'FAILS'}")
    print(f"VERDICT: {v['verdict']}")


if __name__ == "__main__":
    main()
