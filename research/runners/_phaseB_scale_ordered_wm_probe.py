"""Scale validation (quick roadmap win): does the order-encoded WM's ordered-sequence recall hold to the full
n_slots=7 Lisman-Idiart span when the phasor dimension is lifted D=128 -> 256?

The multi-sentence (CYCLE 137) and coherence (138) de-risks ran at D=128 (for byte-parity with the
BrainConversationalAgent composer) and reported an honest ceiling at K~4 (bundle cross-talk erodes longer
sequences). The ordered-WM foundation (CYCLE 135) already ran at D=256 and was clean to load 5. This probe
confirms the obvious-but-unverified next step: at D=256 the ordered recall holds across the full 7-slot span,
so longer discourse/context is a dimension question, not a wall. Reuse-by-import of the production
`OrderedPositionWM`; NO sim/ edit; CPU/numpy.

Run: SIM_BACKEND=numpy python -u -m research.runners._phaseB_scale_ordered_wm_probe --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.ordered_position_wm import OrderedPositionWM

LOADS = [3, 5, 7]
DIMS = [128, 256]
N_TRIALS = 50


def ordered_recall_acc(wm, K, seed):
    """Fraction of trials where the full ordered K-tuple is recovered EXACTLY (every slot's item correct)."""
    rng = np.random.default_rng(seed * 100003 + K)
    vocab = wm.words
    ok = 0
    for _ in range(N_TRIALS):
        items = [vocab[i] for i in rng.choice(len(vocab), size=K, replace=False)]
        comp = wm.encode_sequence(items)
        rec = [wm.read_slot(comp, f"pos{k}", gate=False)[0] for k in range(K)]
        ok += int(rec == items)
    return ok / N_TRIALS


def moat_clean(wm, seed):
    """At the calibrated threshold, an empty/scrambled slot must abstain on a load-5 hold."""
    rng = np.random.default_rng(seed * 7 + 5)
    vocab = wm.words
    abst = 0
    n = 30
    for _ in range(n):
        items = [vocab[i] for i in rng.choice(len(vocab), size=5, replace=False)]
        comp = wm.encode_sequence(items)
        for probe in ("emptyslot", "scrambled"):
            w, _m = wm.read_slot(comp, probe, gate=True)
            abst += int(w is None)
    return abst / (2 * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="research/findings/raw/_phaseB_scale_ordered_wm.json")
    a = ap.parse_args()

    print("[scale probe] ordered recall vs phasor dim D, across loads {3,5,7}; does D=256 hold the full 7-slot span?\n",
          flush=True)
    results = {}
    for D in DIMS:
        results[D] = {}
        for K in LOADS:
            accs = []
            for seed in a.seeds:
                wm = OrderedPositionWM(seed=seed, D=D, n_slots=7)
                accs.append(ordered_recall_acc(wm, K, seed))
            results[D][K] = {"mean": float(np.mean(accs)), "per_seed": [round(v, 3) for v in accs]}
            print(f"  D={D:>3}  K={K}: recall mean={np.mean(accs):.3f}  per-seed={[round(v,2) for v in accs]}",
                  flush=True)
    # Moat at D=256 with the worst-slot-calibrated threshold (the production fix).
    moat = []
    for seed in a.seeds:
        wm = OrderedPositionWM(seed=seed, D=256, n_slots=7)
        moat.append(moat_clean(wm, seed))
    moat_mean = float(np.mean(moat))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"results": {str(D): {str(K): results[D][K] for K in LOADS} for D in DIMS},
                   "moat_d256_abstain_mean": moat_mean, "seeds": a.seeds}, f, indent=2)

    print(f"\n  moat @ D=256 (empty/scrambled abstain, worst-slot-calibrated): {moat_mean:.3f}", flush=True)
    d128_7 = results[128][7]["mean"]
    d256_7 = results[256][7]["mean"]
    d256_5 = results[256][5]["mean"]
    print("\n=== VERDICT ===", flush=True)
    if d256_7 >= 0.80 and d256_5 >= 0.80 and moat_mean >= 0.99:
        print(f"  GO: at D=256 ordered recall holds the FULL 7-slot span (K=7 {d256_7:.3f}, K=5 {d256_5:.3f} >= 0.80)"
              f" with the moat intact ({moat_mean:.3f}); D=128 K=7 was {d128_7:.3f}. Longer discourse is a"
              " DIMENSION knob, not a wall -- the sequence ceiling lifts ~4 -> 7 by doubling D.", flush=True)
    elif d256_7 >= 0.80:
        print(f"  PARTIAL: D=256 K=7 recall {d256_7:.3f} >= 0.80 but the moat or K=5 is marginal"
              f" (moat {moat_mean:.3f}, K5 {d256_5:.3f}).", flush=True)
    else:
        print(f"  BOUNDARY: D=256 does NOT cleanly reach K=7 ({d256_7:.3f}) -- the span needs D>256 or a"
              " different code (the 7-slot Lisman-Idiart ceiling is dimension-limited beyond 256).", flush=True)
    print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
