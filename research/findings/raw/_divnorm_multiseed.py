"""(de-risk A, Task 1 multi-seed) Run the divisive-normalization cleanup at a FIXED operating point across
seeds 42/43/44 and report the recovery table: numpy oracle vs spiking-WITH-divnorm vs spiking-WITHOUT
(matched-filter-only baseline at the same w_match+bias). This is the decisive multi-seed measurement for the
GO/NEGATIVE decision. The operating point is taken from the seed-42 sweep's best.

  python -u -m research.findings.raw._divnorm_multiseed \
      --w-match 100 --bias -600 --w-cfs 15 --w-fs 10 --einh -75 --run-steps 400 \
      --seeds 42 43 44 --out research/findings/raw/_divnorm_multiseed.json
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from research.findings.raw._spiking_cleanup_divnorm_probe import (
    capture_real_est, build_divnorm_bridge, evaluate)


def one_seed(seed, vocab, proj_dim, n_flat, n_attr, w_match, bias, w_cfs, w_fs, n_fs, einh,
             run_steps, input_drive, ou_std):
    items, code_mat, widx, words = capture_real_est(seed, vocab, proj_dim, n_flat, n_attr)
    M = len(words); D = code_mat.shape[1]
    # numpy oracle
    np_ok = sum(int(words[int(np.argmax(code_mat @ est))] == t) for est, t, _ in items)
    np_rec = np_ok / len(items)
    # spiking WITH divnorm
    b1, i1 = build_divnorm_bridge(seed, code_mat, w_match, w_cfs, w_fs, n_fs, einh,
                                  enable_divnorm=True, ou_std=ou_std)
    _, sp_div, cue = evaluate(items, code_mat, widx, words, b1, i1, D, M, bias, run_steps,
                              input_drive=input_drive)
    # spiking WITHOUT divnorm (matched filter only, same w_match+bias) -- the with-vs-without contrast
    b0, i0 = build_divnorm_bridge(seed, code_mat, w_match, 0, 0, n_fs, einh,
                                  enable_divnorm=False, ou_std=ou_std)
    _, sp_nodiv, _ = evaluate(items, code_mat, widx, words, b0, i0, D, M, bias, run_steps,
                              input_drive=input_drive)
    return {"seed": seed, "n_items": len(items), "cue_cos": cue,
            "numpy": np_rec, "spiking_divnorm": sp_div, "spiking_nodivnorm": sp_nodiv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--vocab", type=int, default=320)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-flat", type=int, default=15)
    ap.add_argument("--n-attr", type=int, default=8)
    ap.add_argument("--w-match", type=float, default=100.0)
    ap.add_argument("--bias", type=float, default=-600.0)
    ap.add_argument("--w-cfs", type=float, default=15.0)
    ap.add_argument("--w-fs", type=float, default=10.0)
    ap.add_argument("--n-fs", type=int, default=40)
    ap.add_argument("--einh", type=float, default=-75.0)
    ap.add_argument("--run-steps", type=int, default=400)
    ap.add_argument("--input-drive", type=float, default=2500.0)
    ap.add_argument("--ou-std", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    op = {"w_match": args.w_match, "bias": args.bias, "w_cfs": args.w_cfs, "w_fs": args.w_fs,
          "n_fs": args.n_fs, "einh": args.einh, "run_steps": args.run_steps,
          "input_drive": args.input_drive, "ou_std": args.ou_std}
    print(f"[multiseed] operating point: {json.dumps(op)}", flush=True)
    rows = []
    for seed in args.seeds:
        r = one_seed(seed, args.vocab, args.proj_dim, args.n_flat, args.n_attr, args.w_match,
                     args.bias, args.w_cfs, args.w_fs, args.n_fs, args.einh, args.run_steps,
                     args.input_drive, args.ou_std)
        rows.append(r)
        print(f"[multiseed] seed={r['seed']} cue_cos={r['cue_cos']:.3f}  numpy={r['numpy']:.3f}  "
              f"divnorm={r['spiking_divnorm']:.3f}  nodivnorm={r['spiking_nodivnorm']:.3f}", flush=True)

    np_mean = float(np.mean([r["numpy"] for r in rows]))
    div_mean = float(np.mean([r["spiking_divnorm"] for r in rows]))
    nodiv_mean = float(np.mean([r["spiking_nodivnorm"] for r in rows]))
    res = {"operating_point": op, "rows": rows,
           "numpy_mean": np_mean, "divnorm_mean": div_mean, "nodivnorm_mean": nodiv_mean,
           "margin_to_numpy": div_mean - np_mean, "divnorm_lift_over_nodivnorm": div_mean - nodiv_mean}
    print("\n[summary] " + json.dumps(res, indent=2), flush=True)
    print(f"\n[VERDICT] numpy_mean={np_mean:.3f}  divnorm_mean={div_mean:.3f}  nodivnorm_mean={nodiv_mean:.3f}  "
          f"margin_to_numpy={div_mean - np_mean:+.3f}  divnorm_lift={div_mean - nodiv_mean:+.3f}", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
