"""Board #135 (#75a): joint (s2_norm x s2_gain x ridge) OPERATING-POINT SWEEP for the #75 signed-linear
spiking readout, so common-mode rejection can move ENTIRELY to the READOUT instead of a pre-readout
z-norm CONSTANT.

WHY THIS SWEEP (named next-rung, #75 + #75b). The #75 signed-linear-discriminant spiking readout SOLVED
the spike-port quantization wall (gap collapses to +0.0243) but did not reach a usable capability
(capability_go 0/6): the RATE linear-separability ceiling of the z-normalized C2 feature code is only
~0.4653, ~0.09 below config-B's raw-rate MAX+centroid ceiling (0.56). #75's own diagnosis: "the z-norm
lateral inhibition -- REQUIRED to keep the near-threshold LIF graded -- removes the common-mode MAGNITUDE
the raw-rate centroid exploited." Its named lever #1: "Move common-mode rejection ENTIRELY to the
READOUT. Feed the S2 LIF a LIGHTER-normalized (or raw) cosine drive at a LOWER s2_gain so it stays in the
graded (non-saturating) regime and preserves magnitude, and let the signed readout + its FF inhibition do
the common-mode subtraction (its job)." #75b (the OTHER named lever, a nonlinear 2-layer granule
expansion) was tried and came back a NULL (dNONLIN -0.0087, ties the #75 1-layer baseline) -- its own
decomposition INDEPENDENTLY CONFIRMED #75's magnitude/common-mode diagnosis (the RATE ceiling does not
move with a nonlinear expansion of the SAME z-normed code), which is why #75a (this rung) is now "the more
promising open lever" (board #136's own note).

THE MECHANISM (built here; NO new architecture -- a pure OPERATING-POINT search over an EXISTING knob).
The #75 runner (`_vision_lindiscrim_readout_derisk.py`, REUSED BY IMPORT, not modified) already exposes
`--s2-norm {none,submean,z}` and `--s2-gain` and `--ridge` as CLI knobs -- the pre-readout normalization
this arc's own diagnosis names as the culprit. This runner is a joint grid search over those three knobs
(the readout's own common-mode rejection -- FF inhibition inside `_spiking_class_read` -- is UNCHANGED;
what varies is how much of that job the pre-readout S2 normalization does FOR it), following the exact
2D-then-decisive convention #75/#75b established: EXPLORE on a 3-seed split {42,43,100}, pick the best
operating point by mean held accuracy, then CONFIRM on the full 6-seed decisive run (including the 3
seeds {44,101,102} the exploration never saw -- the anti-cheat against op-point-fitting-to-few-seeds).

GRID (guided by the named lever: LOWER gain -> non-saturating; drop z's per-image over-normalization):
  s2_norm in {none, submean, z}  (z is the #75 baseline/control)
  s2_gain in {0.5, 1.0, 1.5, 2.0}  (2.0 is the #75 baseline/control; the rest are LOWER, per the lever)
  ridge   in {0.1, 0.5, 1.0}       (0.5 is the #75 baseline/control)
The (z, 2.0, 0.5) cell IS the #75 baseline -- included in the grid, not run separately, so the sweep's own
numbers reproduce #75's published result as an internal consistency check.

GO GATE. task_go (this rung) = the CHOSEN op-point's capability_go (>=5/6 on the full decisive 6 seeds,
the SAME strict bar #72/#75/#75b use: beats V1-direct AND flat-pool by margin, learning load-bearing,
position pooled out, scramble/label-shuffle at chance) -- not merely "improves LEARNED_spkwta_held". If
no cell reaches that bar, this is reported as an HONEST NEGATIVE/PARTIAL (an operating-point search that
found no improving direction, or one that improves but does not cross the capability line) -- the wall
NAME changes (from "z-norm constant" to whatever the sweep shows dominates instead), it is not force-GO'd.

ANTI-CHEATS:
  1. The (z, 2.0, 0.5) grid cell must reproduce #75's own 6-seed numbers (checked against the cited
     artifact) -- verifies this sweep's harness, not a new pipeline.
  2. Exploration (42/43/100) is NEVER used to report the final verdict; the decisive run is the FULL 6
     seeds, and the held-out-from-exploration seeds (44/101/102) are reported SEPARATELY so an op-point
     that only wins on the seeds it was chosen on is caught, not hidden in a 6-seed mean.
  3. Every per-seed verdict field already built into `run_seed` (capability_go, learning_load_bearing,
     beats_config_c_nogo, position_pooled_out, label_shuffle_null, scramble_centroid_held) is reused
     UNCHANGED -- this sweep adds no new pass/fail logic, only a search over an existing knob.

No `sim/` edit. The #75 runner (`run_seed`) is imported and called UNCHANGED; only the `s2_norm`/`s2_gain`/
`ridge` fields of its argparse Namespace are varied.

Sources: same as #75/#75b (Carandini & Heeger 2012, divisive normalisation as the common-mode-rejection
computation this sweep is choosing WHERE to place).

Smoke (1 seed, tiny grid):
  SIM_BACKEND=numpy python -u -m research.runners._vision_lindiscrim_opsweep_derisk \
      --explore-seeds 42 --decisive-seeds 42 --gains 1.0 2.0 --norms z --ridges 0.5 \
      --out research/findings/raw/lanes/perception/vlin_opsweep_smoke.json

Decisive (6-seed, full grid):
  SIM_BACKEND=numpy OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u -m \
      research.runners._vision_lindiscrim_opsweep_derisk \
      --out research/findings/raw/lanes/perception/vision_lindiscrim_opsweep_6seed.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._vision_lindiscrim_readout_derisk import run_seed  # noqa: E402

OUT = Path("research/findings/raw/lanes/perception/vision_lindiscrim_opsweep_6seed.json")

# #75's own published baseline (6-seed, s2_norm=z, s2_gain=2.0, ridge=0.5, n_s2=96, n_glimpses=2) --
# the internal-consistency target for this sweep's (z, 2.0, 0.5) grid cell.
BASELINE_75 = {
    "LEARNED_spkwta_held": 0.4375, "RATE_lin_ceiling_held": 0.4653, "A_v1_direct_held": 0.4184,
}


def _default_args(**overrides):
    """The #75 runner's own argparse defaults, replicated here (NOT imported from argparse.parse_args,
    since that consumes sys.argv) so this sweep can build one Namespace per grid cell. Every default
    below is copied verbatim from `_vision_lindiscrim_readout_derisk.main()`."""
    d = dict(
        code="count", c1_code="count", n_classes=4, n_slots=3, n_pos_total=8, pos_span=8.0, n_ex=6,
        image_size=56, slot_offset=10.0, stroke_len=7.0, stroke_tk=1.8, pixel_noise=0.03,
        n_orientations=8, n_frequencies=2, n_pos=24, rf_radius=3, orient_norm="z", c1_gate=0.15,
        c1_win=6, c1_stride=3, s2_p=3, n_s2=96, ridge=0.5, n_glimpses=2, class_pop=24, read_gain=2.5,
        read_bias=1.0, T_read=48, s1_mode="spiking", s2_norm="z", T1=64, T2=48, tau=8.0, v_thresh=1.0,
        t_ref=2, noise=0.06, s1_gain=1.2, s2_gain=2.0, kwta_frac=0.15, decode_margin=0.15,
        beat_margin=0.10, pos_decode_margin=0.15, nogo_floor=0.34,
    )
    d.update(overrides)
    return argparse.Namespace(**d)


def _run_grid_cell(s2_norm, s2_gain, ridge, seeds, code="count"):
    a = _default_args(s2_norm=s2_norm, s2_gain=s2_gain, ridge=ridge)
    rows = [run_seed(s, a, code) for s in seeds]
    learned = float(np.mean([r["decode"]["LEARNED_spkwta_held"] for r in rows]))
    rate_ceil = float(np.mean([r["decode"]["RATE_lin_ceiling_held"] for r in rows]))
    v1 = float(np.mean([r["decode"]["A_v1_direct_held"] for r in rows]))
    n_go = sum(1 for r in rows if r["verdicts"]["capability_go"])
    n_lb = sum(1 for r in rows if r["verdicts"]["learning_load_bearing"])
    return {
        "s2_norm": s2_norm, "s2_gain": s2_gain, "ridge": ridge, "seeds": seeds,
        "learned_spkwta_held_mean": round(learned, 4),
        "rate_lin_ceiling_held_mean": round(rate_ceil, 4),
        "a_v1_direct_held_mean": round(v1, 4),
        "learned_minus_v1_mean": round(learned - v1, 4),
        "n_capability_go": n_go, "n_learning_load_bearing": n_lb, "n_seeds": len(rows),
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--explore-seeds", type=int, nargs="+", default=[42, 43, 100])
    p.add_argument("--decisive-seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    p.add_argument("--norms", nargs="+", default=["none", "submean", "z"])
    p.add_argument("--gains", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--ridges", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    p.add_argument("--code", default="count")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    t0 = time.time()
    grid = list(itertools.product(args.norms, args.gains, args.ridges))
    print(f"[vision-lindiscrim-opsweep] {len(grid)} cells x {len(args.explore_seeds)} explore seeds "
          f"{args.explore_seeds}; decisive seeds {args.decisive_seeds}", flush=True)

    explore = []
    for (norm, gain, ridge) in grid:
        cell = _run_grid_cell(norm, gain, ridge, args.explore_seeds, args.code)
        explore.append(cell)
        print(f"  explore norm={norm:<8} gain={gain:<4} ridge={ridge:<4} "
              f"LEARNED={cell['learned_spkwta_held_mean']:.4f} RATEceil={cell['rate_lin_ceiling_held_mean']:.4f} "
              f"V1={cell['a_v1_direct_held_mean']:.4f} dGO={cell['n_capability_go']}/{cell['n_seeds']}",
              flush=True)

    # Rank by mean held accuracy on the exploration split (the primary target quantity).
    ranked = sorted(explore, key=lambda c: c["learned_spkwta_held_mean"], reverse=True)
    best = ranked[0]
    baseline_cell = next((c for c in explore if c["s2_norm"] == "z" and c["s2_gain"] == 2.0
                           and c["ridge"] == 0.5), None)

    print(f"\n[best on exploration] norm={best['s2_norm']} gain={best['s2_gain']} ridge={best['ridge']} "
          f"LEARNED={best['learned_spkwta_held_mean']:.4f}", flush=True)
    if baseline_cell:
        print(f"[baseline (#75 z/2.0/0.5) on exploration] LEARNED={baseline_cell['learned_spkwta_held_mean']:.4f} "
              f"RATEceil={baseline_cell['rate_lin_ceiling_held_mean']:.4f}", flush=True)

    # ---- DECISIVE: full 6-seed run of the chosen best op-point AND the #75 baseline op-point ----
    print("\n[decisive] running chosen best + baseline on the FULL 6-seed set...", flush=True)
    decisive_best = _run_grid_cell(best["s2_norm"], best["s2_gain"], best["ridge"],
                                    args.decisive_seeds, args.code)
    decisive_baseline = _run_grid_cell("z", 2.0, 0.5, args.decisive_seeds, args.code)

    # Held-out-from-exploration check: seeds NOT in explore-seeds, evaluated separately (anti-cheat 2).
    holdout_seeds = [s for s in args.decisive_seeds if s not in args.explore_seeds]
    best_holdout_rows = [r for r in decisive_best["rows"] if r["seed"] in holdout_seeds]
    baseline_holdout_rows = [r for r in decisive_baseline["rows"] if r["seed"] in holdout_seeds]
    best_holdout_learned = float(np.mean([r["decode"]["LEARNED_spkwta_held"] for r in best_holdout_rows])) \
        if best_holdout_rows else float("nan")
    baseline_holdout_learned = float(np.mean([r["decode"]["LEARNED_spkwta_held"] for r in baseline_holdout_rows])) \
        if baseline_holdout_rows else float("nan")

    n_go_best = decisive_best["n_capability_go"]
    n_go_base = decisive_baseline["n_capability_go"]
    n_seeds = decisive_best["n_seeds"]
    task_go = bool(n_go_best >= 5)  # the same strict capability_go>=5/6 bar as #72/#75/#75b
    lifts_baseline = bool(decisive_best["learned_spkwta_held_mean"] - decisive_baseline["learned_spkwta_held_mean"]
                           >= 0.02)
    overall = (
        "VISION-OPSWEEP-GO" if task_go
        else "VISION-OPSWEEP-LIFTS-BASELINE-PARTIAL" if lifts_baseline
        else "VISION-OPSWEEP-NULL-NO-LEVER-FOUND"
    )

    top = {
        "probe": "vision_lindiscrim_opsweep", "board": "135 (#75a)", "overall_verdict": overall,
        "grid": {"norms": args.norms, "gains": args.gains, "ridges": args.ridges, "n_cells": len(grid)},
        "explore_seeds": args.explore_seeds, "decisive_seeds": args.decisive_seeds,
        "holdout_seeds_from_exploration": holdout_seeds,
        "exploration_ranked": [
            {k: c[k] for k in ("s2_norm", "s2_gain", "ridge", "learned_spkwta_held_mean",
                                "rate_lin_ceiling_held_mean", "a_v1_direct_held_mean",
                                "learned_minus_v1_mean", "n_capability_go", "n_seeds")}
            for c in ranked
        ],
        "chosen_op_point": {"s2_norm": best["s2_norm"], "s2_gain": best["s2_gain"], "ridge": best["ridge"]},
        "decisive": {
            "chosen": {
                "learned_spkwta_held_mean": decisive_best["learned_spkwta_held_mean"],
                "rate_lin_ceiling_held_mean": decisive_best["rate_lin_ceiling_held_mean"],
                "a_v1_direct_held_mean": decisive_best["a_v1_direct_held_mean"],
                "n_capability_go": n_go_best, "n_learning_load_bearing": decisive_best["n_learning_load_bearing"],
                "n_seeds": n_seeds,
                "per_seed_capability_go": [r["verdicts"]["capability_go"] for r in decisive_best["rows"]],
                "per_seed_learned_spkwta_held": [r["decode"]["LEARNED_spkwta_held"] for r in decisive_best["rows"]],
                "holdout_from_exploration_learned_mean": round(best_holdout_learned, 4),
            },
            "baseline_75_z_2p0_0p5": {
                "learned_spkwta_held_mean": decisive_baseline["learned_spkwta_held_mean"],
                "rate_lin_ceiling_held_mean": decisive_baseline["rate_lin_ceiling_held_mean"],
                "a_v1_direct_held_mean": decisive_baseline["a_v1_direct_held_mean"],
                "n_capability_go": n_go_base, "n_seeds": n_seeds,
                "per_seed_capability_go": [r["verdicts"]["capability_go"] for r in decisive_baseline["rows"]],
                "per_seed_learned_spkwta_held": [r["decode"]["LEARNED_spkwta_held"] for r in decisive_baseline["rows"]],
                "holdout_from_exploration_learned_mean": round(baseline_holdout_learned, 4),
                "published_reference_75": BASELINE_75,
                "reproduces_published_75": (
                    abs(decisive_baseline["learned_spkwta_held_mean"] - BASELINE_75["LEARNED_spkwta_held"]) < 0.01
                ),
            },
        },
        "verdicts": {
            "task_go_capability_5of6": task_go,
            "lifts_baseline_by_ge_0p02": lifts_baseline,
            "chosen_op_point_is_baseline": bool(best["s2_norm"] == "z" and best["s2_gain"] == 2.0
                                                 and best["ridge"] == 0.5),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(top, indent=2, default=str))

    print("\n" + "=" * 100, flush=True)
    print(f"[{overall}] chosen={top['chosen_op_point']} "
          f"LEARNED(chosen)={decisive_best['learned_spkwta_held_mean']:.4f} "
          f"LEARNED(baseline)={decisive_baseline['learned_spkwta_held_mean']:.4f} "
          f"GO(chosen)={n_go_best}/{n_seeds} GO(baseline)={n_go_base}/{n_seeds} "
          f"holdout(chosen)={best_holdout_learned:.4f} holdout(baseline)={baseline_holdout_learned:.4f}",
          flush=True)
    print(f"[reproduces #75 published baseline] {top['decisive']['baseline_75_z_2p0_0p5']['reproduces_published_75']}",
          flush=True)
    print(f"[written] {out_path}", flush=True)
    print("=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
