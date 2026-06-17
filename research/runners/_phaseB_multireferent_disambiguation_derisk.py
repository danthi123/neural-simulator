"""Multi-referent disambiguation de-risk — when the working memory holds SEVERAL discourse referents, which one
does a bare pronoun ("it") bind? This is the honest next stress on multi-turn dialogue
(2026-06-17-multiturn-anaphora-derisk-GO.md flagged it): the single-referent case is GO; the realistic case has
several candidates and a pronoun must pick one.

LINGUISTIC DEFAULT. A bare pronoun usually binds the MOST RECENT salient referent ("the dog saw the cat. it ran."
-> it = the cat, the recent one). So the testable question is: does the spiking WM loop carry a RECENCY gradient
(the most-recently-written referent dominates the read), or does it hold all referents as an equal SET (the
validated >=3-set hold) -- in which case a bare pronoun is genuinely ambiguous and disambiguation needs an added
salience signal?

ARMS (3 seeds):
  * NATURAL: write A then B (B most recent), read -> does B dominate A? Order-control: write B then A -> A should
    dominate (proves RECENCY, not a fixed concept bias).
  * REFRESH (a cheap salience mechanism if NATURAL is a set-hold): re-drive the recent referent once more before
    the read -> does that create the gradient?

VERDICT: GO = the recent referent dominates the read (margin) in NATURAL, and the order-control flips it (so a
bare "it" binds the recent referent). BOUNDARY = NATURAL is a near-equal set-hold (no usable recency) but REFRESH
creates the gradient (disambiguation needs an explicit salience refresh -- a precise, buildable next step).
NEGATIVE = neither produces a reliable recency gradient (multi-referent disambiguation needs a richer mechanism).

Run: SIM_BACKEND=numpy python -m research.runners._phaseB_multireferent_disambiguation_derisk --seeds 42 43 44
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


def _read_pair(wm, a, b, window=20):
    """Return (rate_a, rate_b) from a WM read."""
    rates = wm.read(window=window)
    return rates.get(a, 0.0), rates.get(b, 0.0)


def run_seed(seed):
    def wm():
        return SpikingLoopContextBuffer(CONCEPTS, n=600, pattern_size=40, seed=seed, enable_ou=False)

    # NATURAL: write cat then bird (bird most recent). Does bird dominate?
    w = wm(); w.update(["cat"]); w.update(["bird"])
    nat_cat, nat_bird = _read_pair(w, "cat", "bird")
    nat_recent_wins = nat_bird > 1.2 * nat_cat

    # ORDER-CONTROL: write bird then cat (cat most recent). Does cat dominate now? (recency, not concept bias)
    w2 = wm(); w2.update(["bird"]); w2.update(["cat"])
    ord_cat, ord_bird = _read_pair(w2, "cat", "bird")
    ord_recent_wins = ord_cat > 1.2 * ord_bird

    # REFRESH: write cat then bird, then refresh bird once more -> stronger recency?
    w3 = wm(); w3.update(["cat"]); w3.update(["bird"]); w3.update(["bird"])
    ref_cat, ref_bird = _read_pair(w3, "cat", "bird")
    ref_recent_wins = ref_bird > 1.2 * ref_cat

    out = {
        "seed": seed,
        "natural": {"cat": round(nat_cat, 4), "bird_recent": round(nat_bird, 4), "recent_wins": bool(nat_recent_wins)},
        "order_ctrl": {"cat_recent": round(ord_cat, 4), "bird": round(ord_bird, 4), "recent_wins": bool(ord_recent_wins)},
        "refresh": {"cat": round(ref_cat, 4), "bird_recent": round(ref_bird, 4), "recent_wins": bool(ref_recent_wins)},
    }
    out["natural_go"] = bool(nat_recent_wins and ord_recent_wins)
    out["refresh_go"] = bool(ref_recent_wins)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="research/findings/raw/_phaseB_multireferent_disambiguation.json")
    a = ap.parse_args()

    print("[multi-referent disambiguation] when the WM holds 2 referents, does a bare 'it' bind the RECENT one?\n"
          "  GO = recency gradient in NATURAL (order-control flips it); BOUNDARY = needs the REFRESH salience signal.\n",
          flush=True)
    results = []
    for seed in a.seeds:
        r = run_seed(seed)
        results.append(r)
        n, o, f = r["natural"], r["order_ctrl"], r["refresh"]
        print(f"  [seed {seed}] NATURAL cat {n['cat']} vs bird(recent) {n['bird_recent']} -> recent_wins "
              f"{n['recent_wins']} | ORDER cat(recent) {o['cat_recent']} vs bird {o['bird']} -> {o['recent_wins']} | "
              f"REFRESH bird(recent) {f['bird_recent']} vs cat {f['cat']} -> {f['recent_wins']}", flush=True)

    nat = sum(r["natural_go"] for r in results)
    ref = sum(r["refresh_go"] for r in results)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results, "natural_go_seeds": nat, "refresh_go_seeds": ref}, fh, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    if nat == len(results):
        print(f"  GO ({nat}/{len(results)}): the spiking WM loop carries a RECENCY gradient — the most-recently-"
              "written referent dominates the read and the order-control flips it. A bare pronoun binds the recent "
              "referent (the linguistic default) with no added machinery. Multi-referent disambiguation by recency "
              "is spiking-native.", flush=True)
    elif ref == len(results):
        print(f"  BOUNDARY ({nat}/{len(results)} natural, {ref}/{len(results)} with refresh): the plain loop holds "
              "both referents as a near-equal SET (no usable recency — consistent with the validated multi-concept "
              "hold), but an explicit RECENCY REFRESH (re-driving the salient referent) creates the gradient. ⇒ "
              "multi-referent disambiguation needs a salience-refresh signal — a precise, buildable next step "
              "(biology: attentional re-activation / the most-recent referent kept salient). Honest deliverable.",
              flush=True)
    else:
        print(f"  NEGATIVE ({nat}/{len(results)} natural, {ref}/{len(results)} refresh): neither produces a reliable "
              "recency gradient at this config — bare-pronoun multi-referent disambiguation needs a richer salience "
              "mechanism than recency-refresh (e.g. a separate attention pointer). Maps the boundary.", flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
