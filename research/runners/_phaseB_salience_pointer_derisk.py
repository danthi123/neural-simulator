"""Salience-pointer de-risk — the mechanism the multi-referent NEGATIVE
(2026-06-17-multireferent-disambiguation-NEGATIVE.md) named: when the WM holds several referents, an attentional
SALIENCE BOOST on the most-recent (foregrounded) referent should make a bare pronoun bind IT.

The plain loop holds referents as a set with NO usable recency (which dominates is seed-dependent attractor
competition). The fix biology uses is attentional selection: the salient referent is driven harder (a transient
gain boost), winning the competition. This probe tests whether a strong-enough boost makes the boosted referent
RELIABLY dominate the read (3 seeds), with an order-control proving it tracks the BOOST, not a fixed concept.

ARMS (per boost factor f in {1.0 control, 2.0, 4.0}; the boost = drive_pA x f + proportionally more stim steps):
  * write cat (normal), then bird (boosted f) -> does bird dominate? (the recent/attended referent)
  * order-control: write bird (normal), then cat (boosted f) -> does cat dominate? (proves it's the boost)

VERDICT: GO = at some boost, the boosted referent dominates by margin in BOTH the natural and order-control
arms, all 3 seeds (a bare pronoun binds the salient referent). BOUNDARY = a boost helps but is seed-fragile.
NEGATIVE = even a 4x boost cannot reliably win the attractor competition (needs a different mechanism, e.g.
explicit inhibition of the non-salient referent).

Run: SIM_BACKEND=numpy python -m research.runners._phaseB_salience_pointer_derisk --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.content_selection_spiking import SpikingLoopContextBuffer

CONCEPTS = ["cat", "bird", "fish", "worm", "dog", "fox"]
BOOSTS = [1.0, 2.0, 4.0]


def _boosted_update(wm, concept, f):
    """Write `concept` with an attentional salience boost f: higher drive + proportionally more stim steps."""
    wm.update([concept], drive_pA=2500.0 * f, stim=int(40 * f), settle=15)


def _read_pair(wm, a, b, window=20):
    rates = wm.read(window=window)
    return rates.get(a, 0.0), rates.get(b, 0.0)


def run_seed(seed):
    def wm():
        return SpikingLoopContextBuffer(CONCEPTS, n=600, pattern_size=40, seed=seed, enable_ou=False)

    rows = {}
    for f in BOOSTS:
        # natural: cat normal, bird boosted (bird = the attended/recent referent)
        w = wm(); w.update(["cat"]); _boosted_update(w, "bird", f)
        nat_cat, nat_bird = _read_pair(w, "cat", "bird")
        nat = nat_bird > 1.2 * nat_cat
        # order-control: bird normal, cat boosted (cat attended) -> cat should win (tracks the BOOST)
        w2 = wm(); w2.update(["bird"]); _boosted_update(w2, "cat", f)
        ord_cat, ord_bird = _read_pair(w2, "cat", "bird")
        ordc = ord_cat > 1.2 * ord_bird
        rows[f] = {"nat_cat": round(nat_cat, 4), "nat_bird": round(nat_bird, 4), "nat_boost_wins": bool(nat),
                   "ord_cat": round(ord_cat, 4), "ord_bird": round(ord_bird, 4), "ord_boost_wins": bool(ordc),
                   "both": bool(nat and ordc)}
    return {"seed": seed, "boosts": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="research/findings/raw/_phaseB_salience_pointer.json")
    a = ap.parse_args()

    print("[salience-pointer de-risk] does an attentional BOOST make a bare pronoun bind the salient referent?\n"
          "  GO = the boosted referent dominates in BOTH natural + order-control, all seeds, at some boost.\n",
          flush=True)
    results = []
    for seed in a.seeds:
        r = run_seed(seed)
        results.append(r)
        for f in BOOSTS:
            b = r["boosts"][f]
            print(f"  [seed {seed} boost {f:.0f}x] natural bird {b['nat_bird']} vs cat {b['nat_cat']} -> "
                  f"{b['nat_boost_wins']} | order cat {b['ord_cat']} vs bird {b['ord_bird']} -> {b['ord_boost_wins']} "
                  f"|| both {b['both']}", flush=True)

    # best boost = the smallest f where ALL seeds pass BOTH arms
    go_boost = None
    for f in BOOSTS:
        if all(r["boosts"][f]["both"] for r in results):
            go_boost = f
            break
    any_help = any(r["boosts"][f]["both"] for r in results for f in BOOSTS)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results, "go_boost": go_boost}, fh, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    if go_boost is not None:
        print(f"  GO: an attentional salience boost ({go_boost:.0f}x drive) makes the boosted referent reliably "
              "dominate the WM read in BOTH the natural and order-control arms, all 3 seeds. ⇒ multi-referent "
              "disambiguation works with a salience pointer: a bare pronoun binds the attended/most-recent referent. "
              "The boundary is converted — extend MultiTurnAgent to boost the salient referent on write.", flush=True)
    elif any_help:
        print("  BOUNDARY: a boost helps on some seeds/arms but does not RELIABLY win the attractor competition "
              "across all 3 seeds + both arms — seed-fragile. A stronger mechanism (explicit inhibition of the "
              "non-salient referent, or a separate spotlight population) is the next step.", flush=True)
    else:
        print("  NEGATIVE: even a 4x boost cannot win the attractor competition — the loop's set-hold is robust to "
              "drive asymmetry; multi-referent disambiguation needs a different mechanism (winner-take-all "
              "inhibition between referent attractors, not just a salience boost).", flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
