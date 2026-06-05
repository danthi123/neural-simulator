"""(de-risk A, Task 1 sweep) Efficient joint sweep for the divisive-normalization cleanup. Captures the
composer's REAL est ONCE (V=320 production codes) then evaluates many (w_match, bias, w_cfs, w_fs, einh,
run_steps) operating points -- WITH divnorm and the matched no-divnorm baseline -- reusing the same est.
Reports the best divnorm operating point, the matched no-divnorm baseline at that point, and the
no-divnorm GLOBAL best (the honest plateau to beat). numpy oracle re-confirmed on the same est.

  python -m research.findings.raw._divnorm_sweep --seed 42
"""
from __future__ import annotations
import argparse
import json
import itertools

import numpy as np

from research.findings.raw._spiking_cleanup_divnorm_probe import (
    capture_real_est, build_divnorm_bridge, evaluate)


def run_point(items, code_mat, widx, words, seed, w_match, bias, w_cfs, w_fs, n_fs, einh, run_steps,
              divnorm):
    bridge, idx = build_divnorm_bridge(seed, code_mat, w_match, w_cfs, w_fs, n_fs, einh,
                                       enable_divnorm=divnorm)
    M = len(words); D = code_mat.shape[1]
    np_rec, sp_rec, cue_cos = evaluate(items, code_mat, widx, words, bridge, idx, D, M, bias, run_steps)
    return np_rec, sp_rec, cue_cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab", type=int, default=320)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-flat", type=int, default=10)
    ap.add_argument("--n-attr", type=int, default=5)
    ap.add_argument("--n-fs", type=int, default=40)
    ap.add_argument("--run-steps", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items, code_mat, widx, words = capture_real_est(args.seed, args.vocab, args.proj_dim,
                                                    args.n_flat, args.n_attr)
    print(f"[sweep] seed={args.seed} captured {len(items)} real-est items (V={args.vocab})", flush=True)

    # numpy oracle (once)
    np_ok = 0
    for est, true, _ in items:
        np_ok += int(words[int(np.argmax(code_mat @ est))] == true)
    np_rec = np_ok / len(items)
    print(f"[sweep] numpy oracle = {np_rec:.3f}", flush=True)

    # ---- no-divnorm baseline sweep (the honest plateau to beat): matched filter + bias threshold ----
    nodiv_grid = [(40, -500), (40, -400), (60, -600), (100, -600), (100, -800), (40, -300)]
    nodiv_best = (-1.0, None)
    for w_match, bias in nodiv_grid:
        _, sp, _ = run_point(items, code_mat, widx, words, args.seed, w_match, bias, 0, 0,
                             args.n_fs, -75, args.run_steps, divnorm=False)
        print(f"[nodiv] w_match={w_match} bias={bias} -> spiking={sp:.3f}", flush=True)
        if sp > nodiv_best[0]:
            nodiv_best = (sp, {"w_match": w_match, "bias": bias})

    # ---- divnorm sweep ----
    # bias scales with w_match (need a threshold proportional to drive); w_cfs/w_fs gentle (320 sources pool).
    div_grid = list(itertools.product(
        [(60, -400), (60, -600), (100, -600), (100, -800), (150, -900)],   # (w_match, bias)
        [8, 15, 25],                                                        # w_cfs
        [8, 15],                                                            # w_fs
        [-75, -90],                                                         # einh (sigma surrogate)
    ))
    div_best = (-1.0, None)
    div_rows = []
    for (w_match, bias), w_cfs, w_fs, einh in div_grid:
        _, sp, cue = run_point(items, code_mat, widx, words, args.seed, w_match, bias, w_cfs, w_fs,
                              args.n_fs, einh, args.run_steps, divnorm=True)
        row = {"w_match": w_match, "bias": bias, "w_cfs": w_cfs, "w_fs": w_fs, "einh": einh, "spiking": sp}
        div_rows.append(row)
        print(f"[div] w_match={w_match} bias={bias} w_cfs={w_cfs} w_fs={w_fs} einh={einh} -> spiking={sp:.3f}",
              flush=True)
        if sp > div_best[0]:
            div_best = (sp, row)

    # matched no-divnorm baseline AT the divnorm-best (w_match, bias) -> the clean with-vs-without contrast
    bb = div_best[1]
    _, nodiv_at_best, _ = run_point(items, code_mat, widx, words, args.seed, bb["w_match"], bb["bias"],
                                    0, 0, args.n_fs, -75, args.run_steps, divnorm=False)

    res = {"seed": args.seed, "vocab": args.vocab, "n_items": len(items), "run_steps": args.run_steps,
           "numpy": np_rec, "nodiv_global_best": {"spiking": nodiv_best[0], **(nodiv_best[1] or {})},
           "divnorm_best": {"spiking": div_best[0], **(div_best[1] or {})},
           "nodiv_at_divbest_point": nodiv_at_best, "div_rows": div_rows}
    print("\n[summary] " + json.dumps(res, indent=2), flush=True)
    print(f"\n[VERDICT seed {args.seed}] numpy={np_rec:.3f}  nodiv_best={nodiv_best[0]:.3f}  "
          f"divnorm_best={div_best[0]:.3f}  (nodiv at div-best point={nodiv_at_best:.3f})", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
