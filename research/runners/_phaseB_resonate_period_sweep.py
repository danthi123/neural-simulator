"""Cheap orchestration lever (the resonator-paper bonus, 2026-06-17): does a SHORTER resonate window preserve the
composer's phase read? Our `RFPhasorComposer` runs `period+8` (=208) resonate steps per op; the Renner/Frady Loihi
resonator represents a phasor with a 16-step cycle. If a much shorter `period` still answers who/what correctly,
it is a free ~(208/period)x on EVERY op (fewer steps to launch/fuse/scan), no sim/ edit -- just a constructor knob.

Sweeps `period` and reports per-period who/what accuracy (vs ground truth) + ms/query, multi-seed. The smallest
period that preserves accuracy is the free speedup. Pairs with the batched scan (CYCLE 152) and the graph refactor.

Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_resonate_period_sweep [--seeds 42,43,44]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")

from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402

VOCAB = ["dog", "cat", "bird", "fish", "elephant", "horse", "lion", "wolf",
         "go", "run", "fly", "swim", "eat", "see", "chase", "hunt",
         "north", "south", "east", "west", "river", "tree", "mouse", "deer"]
FACTS = [("dog", "go", "north"), ("cat", "run", "south"), ("bird", "fly", "east"),
         ("fish", "swim", "west"), ("elephant", "eat", "tree"), ("horse", "see", "river"),
         ("lion", "chase", "deer"), ("wolf", "hunt", "mouse")]
PERIODS = [8, 16, 24, 32, 48, 64, 100, 200]


def run_seed(seed, period):
    c = RFPhasorComposer(seed=seed, D=128, vocab=VOCAB, period=period)
    for a, ac, p in FACTS:
        c.store(a, ac, p)
    what = sum(int(c.query_patient(a, ac) == p) for a, ac, p in FACTS)
    who = sum(int(c.query_agent(ac, p) == a) for a, ac, p in FACTS)
    # abstention must still hold (a never-stored cue -> None)
    moat = int(c.query_patient("lion", "fly") is None)
    a, ac, _p = FACTS[-1]
    c.query_patient(a, ac)
    t = time.time()
    for _ in range(8):
        c.query_patient(a, ac)
    ms = (time.time() - t) / 8 * 1000
    return what / len(FACTS), who / len(FACTS), moat, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[resonate period sweep] does a shorter resonate window keep who/what correct? seeds={seeds}\n", flush=True)
    rows = {}
    for period in PERIODS:
        whats, whos, moats, mss = [], [], [], []
        for s in seeds:
            w, h, m, ms = run_seed(s, period)
            whats.append(w); whos.append(h); moats.append(m); mss.append(ms)
        rows[period] = {"what": float(np.mean(whats)), "who": float(np.mean(whos)),
                        "moat": int(min(moats)), "ms": float(np.mean(mss))}
        print(f"  period={period:>3} (steps={period+8:>3}): what {np.mean(whats):.3f} | who {np.mean(whos):.3f} | "
              f"moat {min(moats)}/1 | {np.mean(mss):6.1f} ms/query", flush=True)

    base_ms = rows[200]["ms"]
    # smallest period with full who/what accuracy + moat intact, all seeds
    ok = [p for p in PERIODS if rows[p]["what"] >= 0.999 and rows[p]["who"] >= 0.999 and rows[p]["moat"] == 1]
    print(f"\n{'='*88}", flush=True)
    if ok:
        pmin = min(ok)
        print(f"  GO: period {pmin} (steps {pmin+8}) preserves who/what + moat at full accuracy -> "
              f"{200/pmin:.1f}x fewer steps than 200, {base_ms/max(rows[pmin]['ms'],1e-9):.1f}x faster/query "
              f"({base_ms:.0f}->{rows[pmin]['ms']:.0f} ms). FREE (constructor knob, no sim/ edit); stacks with the "
              f"batched scan + the graph refactor. Verify against test_rf_* goldens before changing the default.",
              flush=True)
    else:
        print(f"  BOUNDARY: no period < 200 preserves full accuracy -> the 208-step window is load-bearing for the "
              f"phase read; the speedup must come from the graph/fusion refactor, not a shorter window.", flush=True)
    print(f"{'='*88}", flush=True)
    out = {"seeds": seeds, "periods": {str(p): rows[p] for p in PERIODS},
           "smallest_full_accuracy_period": (min(ok) if ok else None)}
    path = os.path.join(_REPO, "research", "findings", "raw", "_resonate_period_sweep.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
