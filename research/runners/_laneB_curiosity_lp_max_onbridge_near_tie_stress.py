"""Supplementary characterization of `_laneB_curiosity_lp_max_onbridge_derisk.py`'s NAMED failure mode #1
(design SS4: "WTA gives no clean winner / multiple fire when LP-slopes are close ... the de-risk uses
SYNTHETIC well-separated LP first"). The main de-risk sweep deliberately used a comfortable margin (the true
max in [0.55,0.95] vs the runner-up in [0.10,0.40]), so it never stress-tested near-ties -- this is NOT one
of the four formal GO gates, it is an honest characterization of the predicted boundary case.

Forces the top-two options to sit within a small, swept margin and reports the WTA's evaluable (clean dead-
margin winner) rate and accuracy-when-evaluable, per margin, across the same 6 seeds. Reuses `LPMaxWTA`
unmodified (no reinvention).

Run: env SIM_BACKEND=numpy .venv/bin/python -u -m \
  research.runners._laneB_curiosity_lp_max_onbridge_near_tie_stress \
  --out research/findings/raw/lanes/curiosity/lp_max_onbridge_near_tie_characterization.json
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

from research.runners._laneB_curiosity_lp_max_onbridge_derisk import LPMaxWTA, N_OPTIONS  # noqa: E402

SEEDS_DEFAULT = (42, 43, 44, 100, 101, 102)
MARGINS_DEFAULT = (0.20, 0.10, 0.05, 0.02, 0.01)
N_TRIALS_PER_SEED = 20
BASE_LP = 0.60          # the runner-up sits at BASE_LP; the true max sits at BASE_LP + margin


def sweep(seeds=SEEDS_DEFAULT, margins=MARGINS_DEFAULT, n_trials=N_TRIALS_PER_SEED,
          n_options=N_OPTIONS, base=BASE_LP):
    rows = []
    for margin in margins:
        tot_evaluable = tot_correct = tot_trials = 0
        for seed in seeds:
            rng = np.random.default_rng(seed * 5003 + 7)
            wta = LPMaxWTA(seed, n_options)
            for _ in range(n_trials):
                idx = rng.permutation(n_options)
                max_idx, second_idx = int(idx[0]), int(idx[1])
                rest_idx = idx[2:]
                lp = np.zeros(n_options, dtype=np.float64)
                lp[max_idx] = base + margin
                lp[second_idx] = base
                if len(rest_idx):
                    lp[rest_idx] = rng.uniform(0.0, 0.30, size=len(rest_idx))
                r = wta.select(lp)
                tot_trials += 1
                if r.winner is not None:
                    tot_evaluable += 1
                    if r.winner == max_idx:
                        tot_correct += 1
        eval_rate = tot_evaluable / tot_trials
        acc = (tot_correct / tot_evaluable) if tot_evaluable else None
        rows.append({"margin": margin, "n_trials": tot_trials, "n_evaluable": tot_evaluable,
                     "evaluable_rate": eval_rate, "accuracy_when_evaluable": acc})
        acc_str = f"{acc:.1%}" if acc is not None else "n/a"
        print(f"  margin={margin:5.2f}  evaluable={tot_evaluable}/{tot_trials} ({eval_rate:.1%})  "
              f"accuracy-when-evaluable={acc_str}", flush=True)
    return rows


def main() -> None:
    os.environ.setdefault("SIM_BACKEND", "numpy")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    ap.add_argument("--margins", type=float, nargs="+", default=list(MARGINS_DEFAULT))
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_PER_SEED)
    ap.add_argument("--out", default="research/findings/raw/lanes/curiosity/"
                                      "lp_max_onbridge_near_tie_characterization.json")
    args = ap.parse_args()

    print(
        "[lane-B LP-max ON-BRIDGE near-tie stress] characterizing design failure-mode #1 (WTA indecision "
        "under close LP-slope margins) -- NOT a formal GO gate, an honest boundary characterization.\n",
        flush=True,
    )
    rows = sweep(seeds=tuple(args.seeds), margins=tuple(args.margins), n_trials=args.n_trials)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "purpose": "supplementary characterization of design failure-mode #1 (WTA indecision under "
                       "close LP-slope margins); NOT one of the four formal GO gates in the de-risk sweep.",
            "seeds": list(args.seeds), "n_trials_per_margin_total": args.n_trials * len(args.seeds),
            "base_lp": BASE_LP, "rows": rows,
        }, fh, indent=2)
    print(f"\n  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
