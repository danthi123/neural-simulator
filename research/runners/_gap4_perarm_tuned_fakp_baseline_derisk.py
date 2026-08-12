"""gap#4 REDIRECT #1 -- is the 2026-08-02 "chained FA/KP wall" REAL, or a per-arm LEARNING-RATE artifact?

THE QUESTION (from the wave-1 verification, workflow wrufiei6u -> the finding
`2026-08-11-gap4-wave1-verification-corrected-the-FA-KP-wall-is-partly-an-lr-artifact.md`):
  The wall findings (`2026-08-02-gap4-depth-rescue-untestable-on-spikes-...`) reported that the CHAINED transport-free
  FA and its KP repair COLLAPSE to majority-class (single-class output) at N>=3 on XOR -- "the located wall". BUT they
  ran FA/KP at the SAME shared learning rate as the output arm (lr 0.05). A verification skeptic re-ran them at a FAIR
  per-arm lr (0.01-0.02) and found they TOO leave majority-class and beat the reservoir at N=3/4 -- i.e. the "collapse"
  was an lr-DIVERGENCE artifact, not a property of the chained transport-free rule.

  This runner turns that 1-2-seed skeptic re-run into a proper 6-seed, multi-depth, per-arm-lr SWEEP, so the wall's
  status is banked as a reproducible finding: for FA and KP separately, at each depth N in {2,3,4}, sweep the HIDDEN
  transport-free lr over a grid and take the BEST per-arm lr per seed. Then:
    WALL_IS_ARTIFACT  if FA and/or KP ENTER the regime (leave majority-class + beat the OPTIMAL-ridge reservoir) at
                      some fair lr, at N>=3, at >=5/6 seeds  -> the located wall does NOT survive fair per-arm tuning.
    WALL_SURVIVES     if FA and KP FAIL to enter at EVERY lr in the grid, at N>=3, at >=5/6 seeds -> the wall is real,
                      not a tuning miss (and the enter-the-regime GOs are then genuinely distinctive).

  Reuses run_seed() from the wall runner UNCHANGED (same forward init, same arms, same task, same reservoir floor) --
  the ONLY thing swept is lr_fa (FA) and kp_lr (KP). Output arm lr stays 0.05 (matched to the wall/DECOLLE runs).
  NO sim/ edit. numpy/CPU. Config matched to the DECOLLE 6-seed run (hidden 32, T 24, epochs 200, subsample 2000,
  bptt-hidden 128, bptt-epochs 400) so the comparison is apples-to-apples.

Run (fan one process per seed across cores):
    for S in 42 43 44 100 101 102; do SIM_BACKEND=numpy .venv/bin/python -m \
      research.runners._gap4_perarm_tuned_fakp_baseline_derisk --seeds $S --n-list 2 3 4 \
      --lr-grid 0.005 0.01 0.02 0.05 \
      --out research/findings/raw/_gap4_perarm_fakp/perarm_s${S}.json & done; wait
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap4_perarm_tuned_fakp_baseline_derisk \
      --aggregate "research/findings/raw/_gap4_perarm_fakp/perarm_s*.json"
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import glob
import json

import numpy as np

from research.runners._gap4_bptt_snn_chained_fa_transport_free_derisk import run_seed

ENTER_MARGIN = 0.03  # matches the wall runner's own go_fixed / go_kp criteria


def _one(seed, n_layers, lr_fa_kp, hidden, T, epochs, subsample, out_lr, bptt_hidden, bptt_epochs):
    """One run at a given per-arm hidden lr (applied to BOTH FA lr_fa AND KP kp_lr). Output arm lr = out_lr."""
    r = run_seed(seed, hidden, T, epochs, out_lr, lr_fa_kp, 1.0, subsample, {},
                 n_hidden_layers=n_layers, kp_lr=lr_fa_kp, check_depth=False,
                 task_xor=True, bptt_hidden=bptt_hidden, bptt_epochs=bptt_epochs)
    return {
        "lr": lr_fa_kp,
        "fa": r["chained_fa_inherit"],
        "kp": r["chained_fa_kp_inherit"],
        "frozen_opt": r["frozen_optimal_matched_inherit"],
        "bptt": r["bptt_inherit"],
        "chance": r["chance"],
    }


def run_one_seed(seed, n_list, lr_grid, hidden, T, epochs, subsample, out_lr, bptt_hidden, bptt_epochs):
    per_depth = {}
    for N in n_list:
        sweeps = [_one(seed, N, lr, hidden, T, epochs, subsample, out_lr, bptt_hidden, bptt_epochs)
                  for lr in lr_grid]
        chance = sweeps[0]["chance"]
        frozen_opt = float(np.mean([s["frozen_opt"] for s in sweeps]))  # ~lr-independent
        # per arm: best-lr held-out + whether that best enters (leaves majority AND beats optimal reservoir)
        fa_best = max(sweeps, key=lambda s: s["fa"])
        kp_best = max(sweeps, key=lambda s: s["kp"])
        fa_enters = bool(fa_best["fa"] > chance + ENTER_MARGIN and fa_best["fa"] > fa_best["frozen_opt"] + ENTER_MARGIN)
        kp_enters = bool(kp_best["kp"] > chance + ENTER_MARGIN and kp_best["kp"] > kp_best["frozen_opt"] + ENTER_MARGIN)
        per_depth[str(N)] = {
            "chance": chance, "frozen_opt": frozen_opt, "bptt": float(np.mean([s["bptt"] for s in sweeps])),
            "fa_best": fa_best["fa"], "fa_best_lr": fa_best["lr"], "fa_enters": fa_enters,
            "kp_best": kp_best["kp"], "kp_best_lr": kp_best["lr"], "kp_enters": kp_enters,
            "fa_at_shared_lr005": next((s["fa"] for s in sweeps if abs(s["lr"] - 0.05) < 1e-9), None),
            "kp_at_shared_lr005": next((s["kp"] for s in sweeps if abs(s["lr"] - 0.05) < 1e-9), None),
            "sweep": sweeps,
        }
    return {"seed": seed, "per_depth": per_depth}


def aggregate(paths):
    rows = []
    for p in sorted(glob.glob(paths)):
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        print("no result files matched", paths)
        return
    depths = sorted({d for r in rows for d in r["per_depth"]}, key=int)
    print(f"\n=== per-arm-tuned FA/KP baseline -- {len(rows)} seeds, XOR ===")
    print(f"{'N':>2} | {'maj':>5} {'frzOpt':>6} {'bptt':>5} | {'FAbest':>6} {'lr':>5} {'ent':>3} {'/6':>3} | "
          f"{'KPbest':>6} {'lr':>5} {'ent':>3} {'/6':>3} | {'FA@.05':>6} {'KP@.05':>6}")
    verdict_lines = []
    for d in depths:
        fa_best = np.mean([r["per_depth"][d]["fa_best"] for r in rows])
        kp_best = np.mean([r["per_depth"][d]["kp_best"] for r in rows])
        fa_ent = sum(1 for r in rows if r["per_depth"][d]["fa_enters"])
        kp_ent = sum(1 for r in rows if r["per_depth"][d]["kp_enters"])
        maj = np.mean([r["per_depth"][d]["chance"] for r in rows])
        frz = np.mean([r["per_depth"][d]["frozen_opt"] for r in rows])
        bptt = np.mean([r["per_depth"][d]["bptt"] for r in rows])
        fa_lr = np.median([r["per_depth"][d]["fa_best_lr"] for r in rows])
        kp_lr = np.median([r["per_depth"][d]["kp_best_lr"] for r in rows])
        fa05 = np.mean([r["per_depth"][d]["fa_at_shared_lr005"] or np.nan for r in rows])
        kp05 = np.mean([r["per_depth"][d]["kp_at_shared_lr005"] or np.nan for r in rows])
        print(f"{d:>2} | {maj:5.3f} {frz:6.3f} {bptt:5.3f} | {fa_best:6.3f} {fa_lr:5.3f} "
              f"{'Y' if fa_ent>=5 else 'n':>3} {fa_ent:>2}/6 | {kp_best:6.3f} {kp_lr:5.3f} "
              f"{'Y' if kp_ent>=5 else 'n':>3} {kp_ent:>2}/6 | {fa05:6.3f} {kp05:6.3f}")
        if int(d) >= 3:
            survives = fa_ent < 5 and kp_ent < 5
            verdict_lines.append((d, survives, fa_ent, kp_ent))
    print("\n--- VERDICT (N>=3, the wall depths) ---")
    any_enter = any((not sv) for _, sv, _, _ in verdict_lines)
    for d, sv, fe, ke in verdict_lines:
        print(f"  N={d}: {'WALL SURVIVES fair tuning (FA<5/6 AND KP<5/6 enter)' if sv else 'WALL IS AN LR ARTIFACT — FA or KP ENTER at a fair per-arm lr'} (FA {fe}/6, KP {ke}/6)")
    print(f"\n  OVERALL: {'⛔ the 2026-08-02 chained-FA/KP wall is (at least partly) an lr-tuning ARTIFACT on XOR' if any_enter else '✅ the wall SURVIVES fair per-arm tuning — it is real, not a tuning miss'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-list", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--lr-grid", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--timesteps", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--subsample", type=int, default=2000)
    ap.add_argument("--out-lr", type=float, default=0.05, help="output-arm lr (matched to the wall/DECOLLE runs)")
    ap.add_argument("--bptt-hidden", type=int, default=128)
    ap.add_argument("--bptt-epochs", type=int, default=400)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--aggregate", type=str, default=None)
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.aggregate)
        return

    results = [run_one_seed(sd, args.n_list, args.lr_grid, args.hidden, args.timesteps, args.epochs,
                            args.subsample, args.out_lr, args.bptt_hidden, args.bptt_epochs)
               for sd in args.seeds]
    out = {"config": {"seeds": args.seeds, "n_list": args.n_list, "lr_grid": args.lr_grid,
                      "hidden": args.hidden, "T": args.timesteps, "epochs": args.epochs,
                      "subsample": args.subsample, "out_lr": args.out_lr,
                      "bptt_hidden": args.bptt_hidden, "bptt_epochs": args.bptt_epochs},
           "results": results}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print("wrote", args.out)
    for r in results:
        for d, v in r["per_depth"].items():
            print(f"seed {r['seed']} N={d}: FA_best {v['fa_best']:.3f}@{v['fa_best_lr']} enter={v['fa_enters']} | "
                  f"KP_best {v['kp_best']:.3f}@{v['kp_best_lr']} enter={v['kp_enters']} | "
                  f"maj {v['chance']:.3f} frzOpt {v['frozen_opt']:.3f} | FA@.05 {v['fa_at_shared_lr005']}")


if __name__ == "__main__":
    main()
