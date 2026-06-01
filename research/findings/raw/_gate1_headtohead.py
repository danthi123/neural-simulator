"""Reusable separability head-to-head on a captured-activity npz (Gate 1 / Gate 2 comparator).

Fair head-to-head at matched averaging level k: pool-argmax (the documented readout) vs full-code
leave-one-out nearest-centroid NN. Also clean between-concept cosine. Used to compare bridges (e.g. Gate 2:
baseline topographic prior vs strong) -- a stronger prior PASSES if it lifts the clean-code separability
notably above the ~0.64 baseline.

Numeric arrays only. Run: python -m research.findings.raw._gate1_headtohead --npz <file.npz>
"""
from __future__ import annotations
import argparse
import os
import numpy as np


def _mc(A):
    A = A - A.mean(axis=1, keepdims=True)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    a = ap.parse_args()
    if not os.path.exists(a.npz):
        print(f"CANNOT-CONCLUDE: {a.npz} not found", flush=True); return
    d = np.load(a.npz)
    X, y, pw = d["X"].astype(np.float64), d["y"].astype(np.int64), d["pool_of_word"].astype(np.int64)
    nw = len(np.unique(y)); M = int(d["m_samples"]); P = 200
    print(f"=== head-to-head {os.path.basename(a.npz)} (V={nw}, M={M}) ===", flush=True)

    # clean (M-avg) codes: pool-argmax + between-cos
    clean = np.stack([X[y == c].mean(0) for c in range(nw)])
    pa_clean = float(np.mean(clean.reshape(nw, nw, P).mean(2).argmax(1) == pw))
    cn = _mc(clean); btw = float((cn @ cn.T)[~np.eye(nw, dtype=bool)].mean())
    print(f"  clean {M}-avg: pool-argmax {pa_clean:.3f}   between-concept cos {btw:.3f}", flush=True)

    best_nn = 0.0
    for k in [1, 4, 8]:
        ng = M // k
        if ng < 2:
            continue
        codes = []; lab = []
        for c in range(nw):
            xs = X[y == c]
            for g in range(ng):
                codes.append(xs[g*k:(g+1)*k].mean(0)); lab.append(c)
        codes = np.array(codes); lab = np.array(lab)
        pa = float(np.mean(codes.reshape(len(codes), nw, P).mean(2).argmax(1) == pw[lab]))
        cc = _mc(codes); ok = 0
        for i in range(len(codes)):
            cents = np.stack([cc[(lab == c) & (np.arange(len(codes)) != i)].mean(0) for c in range(nw)])
            cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
            ok += int((cc[i] @ cents.T).argmax() == lab[i])
        nn = ok / len(codes); best_nn = max(best_nn, nn)
        print(f"  k={k:2d}-avg: pool-argmax {pa:.3f}  full-code NN {nn:.3f}", flush=True)
    print(f"\nSUMMARY[{os.path.basename(a.npz)}]: clean-pool-argmax {pa_clean:.3f}  best-NN {best_nn:.3f}  "
          f"between-cos {btw:.3f}", flush=True)


if __name__ == "__main__":
    main()
