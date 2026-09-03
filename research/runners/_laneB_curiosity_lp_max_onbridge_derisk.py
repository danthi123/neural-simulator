"""ON-BRIDGE SPIKING realization of learning-progress(LP)-MAXIMIZING curiosity SELECTION -- the neural-max
half of the DESIGN doc `research/findings/2026-09-03-on-bridge-spiking-LP-max-curiosity-DESIGN.md` (built
from branch `research/onbridge-lpmax-design`, not yet merged to `main` at the time this runner was written).

WHAT THIS BUILDS (design SS3b/SS3d, "the un-built core"). Every existing on-bridge curiosity runner
(`_curiosity_seek_learn_onbridge_derisk.py`) reads a SPIKING per-concept learning-progress signal
(`deliver_reward` -> `snc_B - snc_A`) but then picks the ask with a HOST `max(want[c] for c in cands)` (that
runner's lines ~527-531) -- the SELECTION over options has never been neural here. This module builds that
selector: an N-channel FSI lateral-inhibition WTA (design Primitive 1), reused VERBATIM (not re-derived) from
the already-6/6-GO `_affect_marker_wta_derisk.py` affective-marker circuit -- `_build_bridge`/`_pool_rates`
are imported unmodified; the only thing that changes is WHAT drives the channels (felt-mood Gaussian tuning
-> a direct per-option LP-slope current, design SS3b: "drive_pa[i] proportional to LP-slope(i)").

THE CHEAP DE-RISK THIS RUNNER EXECUTES BY DEFAULT (design SS4, "the cheapest CPU de-risk" -- THE deliverable
of this build). Build ONLY the selection layer; drive it with a SYNTHETIC, KNOWN LP-slope vector (one clear
max-learnable option, several mid options, one option pinned at LP=0) so the (known-fragile) LP ESTIMATE
never confounds the read on the (novel, untested) neural MAX -- exactly the isolation the design calls for.
A separate, NEVER-WIRED "would-be novelty" scalar is recorded at near-max for the same zero-LP option on
every trial and never enters any option pool's `cp_external_input_current` -- proving BY CONSTRUCTION that
the selection cannot be captured by novelty even though the zero-LP option is, on paper, maximally
"interesting" (design failure-mode #3). Sweep many random LP vectors x 6 seeds; four gates mirror the CPU
proxy's g1-g5 bar (`_laneB_curiosity_lp_max_selection_derisk.py`, 2026-08-07 6-seed GO):

  g_select      neural WTA winner == host argmax(LP-slope), >=90% of evaluable trials, EVERY seed.
  g_noisy       the zero-LP/max-novelty option's win COUNT is 0 (raw count reported, not a ratio alone --
                docs/TERMS.md "selective").
  g_loadbearing swap the drive from the true LP-slope vector to the (unwired, would-be) novelty vector: the
                SAME circuit must then track novelty (win-rate on the zero-LP option rises) and STOP tracking
                the true max-LP option. The swap is a static per-trial config change, so "persistence" across
                the whole sweep is the gate itself -- no separate probe needed.
  g_specificity mis-route the LP-slope per a fixed seeded permutation (assembly i driven by option perm(i)'s
                LP -- `_affect_marker_wta_derisk`'s own shuffle anti-cheat, generalized): the reported winner
                (read off WHICH PHYSICAL ASSEMBLY won, never translated back through perm) must stop tracking
                the TRUE max-LP option -- proof the winner is genuinely read from the spiking race, not
                silently re-derived from the raw LP vector by a host formula blind to the mis-routing.

FULL BUILD (design SS3c) -- NOT implemented here, and deliberately so. The design's ONE-bridge composition
(this WTA fed by `_curiosity_seek_learn_onbridge_derisk.build_curiosity_bridge` + `deliver_reward`'s spiking
`reward_read`, in place of the synthetic vector above) is the NAMED next lever once this de-risk is GO: it
would reintroduce the fragile LP ESTIMATE the de-risk above deliberately isolates away from, so building it
before this layer's own correctness is established would confound two untested things at once (exactly the
failure the design's SS4 opening paragraph warns against). The two reuse points are named in the module
docstring above and in DESIGN SS2's asset table; wiring them together is mechanical once this GO stands.

REUSE, no `sim/` edit, no existing runner modified:
  * selection primitive  -- `research.runners._affect_marker_wta_derisk._build_bridge` / `_pool_rates`
  * discipline helpers    -- `tools.lab.undefined_if_empty` / `attributable_to`

Run (CPU only, no GPU, no sim import needed for the default synthetic de-risk beyond `_build_bridge`'s own):
  env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    .venv/bin/python -u -m research.runners._laneB_curiosity_lp_max_onbridge_derisk \
    --smoke --out research/findings/raw/lanes/curiosity/lp_max_onbridge_smoke.json

  env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    .venv/bin/python -u -m research.runners._laneB_curiosity_lp_max_onbridge_derisk \
    --seeds 42 43 44 100 101 102 --out research/findings/raw/lanes/curiosity/lp_max_onbridge_derisk.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# -- the selection PRIMITIVE, reused verbatim (see module docstring; no reinvention). ----------------------
from research.runners._affect_marker_wta_derisk import (  # noqa: E402
    _build_bridge, _pool_rates, WARMUP_STEPS, WASHOUT_STEPS, RUN_STEPS, DEAD_MARGIN,
)
from tools.lab import undefined_if_empty, attributable_to  # noqa: E402


# -- the selection-layer drive (design SS3b/SS3d): a DIRECT proportional current, not the affect-marker's
# Gaussian population tuning -- the input here already IS the per-option scalar the design specifies
# (`drive_pa[i] proportional to LP-slope(i)`), so no tuning-curve computation is needed. -------------------
DRIVE_BASE_PA = 150.0        # baseline every option pool gets regardless of LP-slope (affect-marker's regime)
DRIVE_GAIN_PA = 1400.0       # pA per unit LP-slope (LP-slope clipped to [0, ~1] before scaling)

N_OPTIONS = 6                 # == N_VALENCE_POOLS: reuses the ALREADY-6/6-GO 6-channel topology/weights
                               # (TO_FSI_WEIGHT/CROSS_INHIB_WEIGHT/DEAD_MARGIN, all imported unchanged) as-is.
N_TRIALS = 50
SEEDS_DEFAULT = (42, 43, 44, 100, 101, 102)

# gate bars (mirror the CPU proxy's g1-g5 rigor, `_laneB_curiosity_lp_max_selection_derisk.py`)
SELECT_ACC_BAR = 0.90
LOADBEARING_ACC_BAR = 0.90
SPEC_MAX_ACC = 0.50            # permuted accuracy must fall at/under this AND at/under half the intact accuracy


def lp_to_drive(lp_slope) -> np.ndarray:
    lp = np.clip(np.asarray(lp_slope, dtype=np.float64), 0.0, None)
    return DRIVE_BASE_PA + DRIVE_GAIN_PA * lp


@dataclass
class SelectResult:
    winner: Optional[int]
    rates: np.ndarray
    margin: float


class LPMaxWTA:
    """One process-warm N-channel FSI lateral-inhibition WTA (`_affect_marker_wta_derisk._build_bridge`,
    prefix `lp_opt`), driven directly by a per-option LP-slope array instead of a Gaussian-tuned mood value."""

    def __init__(self, seed: int, n_options: int = N_OPTIONS):
        if n_options < 3:
            raise ValueError("n_options must be >= 3 (need a max, a mid, and a noisy option)")
        self.seed = int(seed)
        self.n_options = int(n_options)
        self.bridge, self.opt_idx, self.fsi_idx = _build_bridge(self.seed, self.n_options, "lp_opt")

    def select(self, drive_values: Sequence[float], *, perm: Optional[np.ndarray] = None,
               dead_margin: float = DEAD_MARGIN) -> SelectResult:
        values = np.asarray(drive_values, dtype=np.float64)
        if perm is not None:
            # g_specificity mis-routing (design SS4): physical assembly i is driven by option perm[i]'s
            # value. `winner` below is still the PHYSICAL/canonical assembly identity -- read off which
            # assembly actually won, never translated back through perm.
            values = values[perm]
        drive_pa = lp_to_drive(values)
        rates = _pool_rates(self.bridge, self.opt_idx, drive_pa,
                             warmup=WARMUP_STEPS, washout=WASHOUT_STEPS, run=RUN_STEPS)
        order = np.argsort(rates)[::-1]
        top, second = int(order[0]), int(order[1])
        margin = float(rates[top] - rates[second])
        winner = top if margin > dead_margin else None
        return SelectResult(winner=winner, rates=rates, margin=margin)


def gen_trial(rng: np.random.Generator, n_options: int):
    """One synthetic trial: a KNOWN LP-slope vector (one clear max-learnable option, several mid options, one
    option pinned at LP=0) plus a separate, NEVER-WIRED 'would-be novelty' scalar that peaks on that SAME
    zero-LP option (design SS4: "a HIGH novelty current to the noisy option, to prove novelty cannot leak
    into the LP-driven race"). The three LP bands never overlap (max > mid > noisy=0 by construction), so the
    host argmax is unambiguous -- what is under test is only whether the SPIKING circuit finds the same max."""
    idx = rng.permutation(n_options)
    max_idx, noisy_idx = int(idx[0]), int(idx[1])
    mid_idx = idx[2:]
    lp = np.zeros(n_options, dtype=np.float64)
    lp[max_idx] = rng.uniform(0.55, 0.95)
    lp[noisy_idx] = 0.0
    if len(mid_idx):
        lp[mid_idx] = rng.uniform(0.10, 0.40, size=len(mid_idx))
    novelty = rng.uniform(0.0, 0.25, size=n_options)
    novelty[noisy_idx] = rng.uniform(0.85, 1.0)
    return lp, novelty, max_idx, noisy_idx


def evaluate(seed: int, n_options: int = N_OPTIONS, n_trials: int = N_TRIALS,
             *, verbose: bool = False) -> Dict[str, object]:
    rng = np.random.default_rng(seed * 911 + 3)
    wta = LPMaxWTA(seed, n_options)
    perm = rng.permutation(n_options)
    while np.any(perm == np.arange(n_options)):        # avoid an accidental fixed point diluting g_specificity
        perm = rng.permutation(n_options)

    trials = []
    for t in range(n_trials):
        lp, novelty, max_idx, noisy_idx = gen_trial(rng, n_options)
        intact = wta.select(lp)
        lesion = wta.select(novelty)                     # g_loadbearing: LP drive -> (unwired) novelty drive
        permd = wta.select(lp, perm=perm)                 # g_specificity: mis-routed LP

        trials.append({
            "trial": t, "lp": lp.tolist(), "novelty": novelty.tolist(),
            "max_idx": max_idx, "noisy_idx": noisy_idx,
            "intact_winner": intact.winner, "intact_margin": intact.margin,
            "lesion_winner": lesion.winner, "lesion_margin": lesion.margin,
            "perm_winner": permd.winner, "perm_margin": permd.margin,
        })
        if verbose:
            print(f"    [seed {seed} trial {t:02d}] max={max_idx} noisy={noisy_idx} "
                  f"intact_win={intact.winner}(m={intact.margin:.4f}) "
                  f"lesion_win={lesion.winner}(m={lesion.margin:.4f}) "
                  f"perm_win={permd.winner}(m={permd.margin:.4f})", flush=True)

    n = len(trials)
    intact_eval = [r for r in trials if r["intact_winner"] is not None]
    lesion_eval = [r for r in trials if r["lesion_winner"] is not None]
    perm_eval = [r for r in trials if r["perm_winner"] is not None]

    intact_correct = sum(1 for r in intact_eval if r["intact_winner"] == r["max_idx"])
    intact_noisy_wins = sum(1 for r in intact_eval if r["intact_winner"] == r["noisy_idx"])
    lesion_track_novelty = sum(1 for r in lesion_eval if r["lesion_winner"] == r["noisy_idx"])
    lesion_still_lp = sum(1 for r in lesion_eval if r["lesion_winner"] == r["max_idx"])
    perm_correct = sum(1 for r in perm_eval if r["perm_winner"] == r["max_idx"])

    raw_select = undefined_if_empty(f"seed {seed} g_select intact_correct", len(intact_eval), intact_correct, n)
    raw_load = undefined_if_empty(f"seed {seed} g_loadbearing lesion_track_novelty", len(lesion_eval),
                                   lesion_track_novelty, n)
    raw_spec = undefined_if_empty(f"seed {seed} g_specificity perm_correct", len(perm_eval), perm_correct, n)

    acc_select = None if raw_select is None else raw_select / len(intact_eval)
    acc_load = None if raw_load is None else raw_load / len(lesion_eval)
    acc_spec = None if raw_spec is None else raw_spec / len(perm_eval)
    lesion_lp_frac = None if not lesion_eval else lesion_still_lp / len(lesion_eval)

    g_select = bool(acc_select is not None and acc_select >= SELECT_ACC_BAR)
    g_noisy = bool(intact_noisy_wins == 0)
    g_loadbearing = bool(acc_load is not None and acc_load >= LOADBEARING_ACC_BAR
                          and lesion_lp_frac is not None and lesion_lp_frac <= 0.10)
    g_specificity = bool(acc_spec is not None and acc_select is not None
                          and acc_spec <= SPEC_MAX_ACC and acc_spec <= 0.5 * acc_select)

    go = bool(g_select and g_noisy and g_loadbearing and g_specificity)

    if acc_select is not None and lesion_lp_frac is not None:
        attributable_to(f"seed {seed}: winner==max-LP option, intact-LP-drive vs novelty-drive lesion",
                         acc_select, lesion_lp_frac)

    return {
        "seed": int(seed), "n_trials": n,
        "n_evaluable_intact": len(intact_eval), "n_evaluable_lesion": len(lesion_eval),
        "n_evaluable_perm": len(perm_eval),
        "acc_select": acc_select, "acc_loadbearing_novelty_track": acc_load,
        "lesion_still_tracks_lp_frac": lesion_lp_frac, "acc_specificity_permuted": acc_spec,
        "intact_correct": intact_correct, "intact_noisy_wins": intact_noisy_wins,
        "lesion_track_novelty": lesion_track_novelty, "lesion_still_tracks_lp": lesion_still_lp,
        "perm_correct": perm_correct,
        "g_select": g_select, "g_noisy": g_noisy, "g_loadbearing": g_loadbearing, "g_specificity": g_specificity,
        "GO": go,
        "trials": trials,
    }


def main() -> None:
    os.environ.setdefault("SIM_BACKEND", "numpy")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    ap.add_argument("--smoke", action="store_true", help="tiny 1-seed CPU smoke")
    ap.add_argument("--selection", choices=("wta", "bg"), default="wta")
    ap.add_argument(
        "--fast-tonic", action="store_true",
        help="opt-in two-pool phasic-minus-tonic LP realization (design SS3a) -- accepted for CLI-surface "
             "completeness; a documented NO-OP in this synthetic de-risk (see module docstring's FULL BUILD "
             "note -- the split only matters once real progress_read pulses, not already-differenced slope "
             "scalars, are the input, and the design itself says start simple and add it only if a mastered "
             "case leaks, which this de-risk does not model).")
    ap.add_argument("--n-options", type=int, default=N_OPTIONS)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if args.out is None:
        args.out = "research/findings/raw/lanes/curiosity/lp_max_onbridge_derisk.json"
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.selection == "bg":
        print(
            "[lane-B LP-max ON-BRIDGE] --selection bg (Primitive 2, BG selection-by-disinhibition) is "
            "DEFERRED by the design itself (\"make it the second build, gated on Primitive 1 passing\", "
            "DESIGN SS3b). NOT implemented in this de-risk -- run --selection wta (the default).",
            flush=True,
        )
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"selection": "bg", "status": "deferred", "GO": None,
                       "reason": "Primitive 2 gated on Primitive 1 passing; not built in this de-risk."},
                      fh, indent=2)
        print(f"  [saved] {args.out}", flush=True)
        return

    if args.fast_tonic:
        print(
            "[lane-B LP-max ON-BRIDGE] --fast-tonic accepted but is a documented NO-OP in this synthetic "
            "de-risk (see module docstring). Proceeding with the direct LP-slope -> current drive.",
            flush=True,
        )

    seeds = args.seeds[:1] if args.smoke else args.seeds
    n_trials = 8 if args.smoke else args.n_trials

    print(
        "[lane-B LP-max ON-BRIDGE de-risk] SYNTHETIC LP-slope sweep over an N-channel FSI lateral-inhibition "
        "WTA (reused from `_affect_marker_wta_derisk`, felt-mood drive -> LP-slope drive). Isolates the "
        "NEURAL MAX from the (known-fragile) LP estimate per DESIGN SS4.\n",
        flush=True,
    )

    results = []
    for seed in seeds:
        r = evaluate(seed, n_options=args.n_options, n_trials=n_trials, verbose=args.verbose or args.smoke)
        results.append(r)
        print(
            f"  [seed {seed}] g_select(acc={r['acc_select']}) g_noisy(noisy_wins={r['intact_noisy_wins']}) "
            f"g_loadbearing(track_novelty={r['acc_loadbearing_novelty_track']}, "
            f"still_lp={r['lesion_still_tracks_lp_frac']}) "
            f"g_specificity(perm_acc={r['acc_specificity_permuted']})\n"
            f"            evaluable intact={r['n_evaluable_intact']}/{r['n_trials']} "
            f"lesion={r['n_evaluable_lesion']}/{r['n_trials']} perm={r['n_evaluable_perm']}/{r['n_trials']}  "
            f"==>  {'GO' if r['GO'] else 'NO'}\n",
            flush=True,
        )

    n_go = sum(1 for r in results if r["GO"])
    payload = {
        "selection": "wta",
        "n_options": args.n_options,
        "drive_base_pa": DRIVE_BASE_PA, "drive_gain_pa": DRIVE_GAIN_PA, "dead_margin": DEAD_MARGIN,
        "select_acc_bar": SELECT_ACC_BAR, "loadbearing_acc_bar": LOADBEARING_ACC_BAR,
        "spec_max_acc": SPEC_MAX_ACC,
        "smoke": bool(args.smoke),
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    print("=" * 100, flush=True)
    print(
        f"  LP-MAX ON-BRIDGE SELECTION: {n_go}/{len(results)} seeds GO "
        f"({'ALL GO' if n_go == len(results) else 'partial/negative - inspect per-seed flags'})",
        flush=True,
    )
    print(f"  [saved] {args.out}\n" + "=" * 100, flush=True)


if __name__ == "__main__":
    main()
