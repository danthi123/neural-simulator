"""FRESH-GATE cheap-first (spiking-SSM class → the emergence-compatible extract): does a FIXED HiPPO-structured
multi-timescale recurrence hold information LONGER than a random reservoir — from the FORWARD structure alone, no BPTT?
Direct memory-horizon test: present a CUE token, then N random distractor tokens, decode the cue from the reservoir
state after N distractors (a LOCAL linear read-out). Sweep N (the horizon). If the structured recurrence decodes the cue
at larger N than the random reservoir, the fading-memory ceiling (that blocks the whole generation ladder) is surpassed
by STRUCTURING the fixed recurrence (multi-timescale time constants = the SSM/HiPPO forward long-range = biology's
diverse dendritic/synaptic time constants) — consistent with the R3 reframe (fixed reservoir wins) + the multi-timescale
reservoir GO. Gate: `2026-07-13-fresh-gate-spiking-SSM-...md`.

ARMS: (1) RANDOM reservoir (ESN: tanh recurrence, spectral radius ~0.95); (2) MULTI-TIMESCALE diagonal (the SSM extract:
per-unit leaky integrators `x <- a_i*x + W_in@u`, `a_i` spanning a LOG range of time constants ≈ HiPPO-LegS init);
(3) PERMUTED time-constants (anti-cheat: same a-values, shuffled — the diagonal structure without the principled range;
should NOT change a diagonal reservoir, a sanity control) + a within-window sanity (both decode at N=0). numpy-CPU.
Reuse-by-import: none needed (self-contained rate reservoirs); NO `sim/` edit.

Run: SIM_BACKEND=numpy python -m research.runners._ssm_fixed_structured_reservoir_derisk --seed 42
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import time

import numpy as np

_V = 12          # cue vocabulary (the cue is one of V tokens)
_N_DISTRACT = 20  # distractor vocabulary (disjoint from cues so the read can't just count cue tokens)
_N = 300         # reservoir units


def _win(rng, n_in):
    return (rng.standard_normal((_N, n_in)) * (1.0 / np.sqrt(n_in))).astype(np.float64)


def _run_seq(kind, A, W_in, U):
    """Drive a reservoir with the token sequence U (list of one-hot input vecs) -> final state x."""
    x = np.zeros(_N)
    for u in U:
        if kind == "random":
            x = np.tanh(A @ x + W_in @ u)                     # ESN: mixing tanh recurrence
        else:
            x = A * x + W_in @ u                              # diagonal leaky integrators (linear, the SSM extract)
    return x


def _build_A(kind, rng):
    if kind == "random":
        W = rng.standard_normal((_N, _N)); ev = np.max(np.abs(np.linalg.eigvals(W)))
        return (0.95 / ev) * W                                # spectral radius 0.95 (echo-state)
    # multi-timescale diagonal: per-unit leak spanning a LOG range of time constants tau in [1.5, 1000]
    if kind == "fastonly":
        tau = np.exp(np.linspace(np.log(1.5), np.log(4.0), _N))       # NO slow units -> should fade (anti-cheat)
    else:
        tau = np.exp(np.linspace(np.log(1.5), np.log(1000.0), _N))
    a = np.exp(-1.0 / tau)                                    # a_i in (~0.49 .. ~0.999); slow units hold info long
    if kind == "permuted":
        a = rng.permutation(a)                                # anti-cheat: shuffle which dim gets which tau
    return a


def _decode_acc(kind, rng, gaps, n_train=1500, n_test=400):
    n_in = _V + _N_DISTRACT
    A = _build_A(kind, rng); W_in = _win(rng, n_in)

    def seq(cue, N):
        U = [np.eye(n_in)[cue]]                               # cue token (ids 0.._V-1)
        for _ in range(N):
            U.append(np.eye(n_in)[_V + rng.integers(_N_DISTRACT)])   # distractor tokens (disjoint ids)
        return U

    accs = {}
    for N in gaps:
        Xtr = []; ytr = []
        for _ in range(n_train):
            c = int(rng.integers(_V)); Xtr.append(_run_seq(kind, A, W_in, seq(c, N))); ytr.append(c)
        Xtr = np.asarray(Xtr); ytr = np.asarray(ytr)
        m, s = Xtr.mean(0), Xtr.std(0) + 1e-6; Xn = (Xtr - m) / s
        Wd = np.linalg.solve(Xn.T @ Xn + 1.0 * np.eye(_N), Xn.T @ np.eye(_V)[ytr])
        Xte = []; yte = []
        for _ in range(n_test):
            c = int(rng.integers(_V)); Xte.append(_run_seq(kind, A, W_in, seq(c, N))); yte.append(c)
        Xte = (np.asarray(Xte) - m) / s
        accs[N] = float(np.mean(np.argmax(Xte @ Wd, 1) == np.asarray(yte)))
    return accs


def run(seed):
    gaps = [0, 5, 15, 40, 80, 150]
    out = {}
    for kind in ("random", "multitimescale", "fastonly", "permuted"):
        out[kind] = _decode_acc(kind, np.random.default_rng(seed * 17 + hash(kind) % 1000), gaps)
    print(f"[ssm-reservoir seed={seed}] cue-decode accuracy vs distractor gap (chance {1/_V:.3f}):")
    print(f"    gap:            {gaps}")
    for kind in ("random", "multitimescale", "fastonly", "permuted"):
        print(f"    {kind:14s}: {[round(out[kind][g],3) for g in gaps]}")
    # GATE: does the multi-timescale hold the cue at LONG gaps where the random reservoir has faded to chance?
    deep = [80, 150]
    mt_deep = np.mean([out["multitimescale"][g] for g in deep])
    rand_deep = np.mean([out["random"][g] for g in deep])
    fast_deep = np.mean([out["fastonly"][g] for g in deep])
    go = (mt_deep > rand_deep + 0.15) and (mt_deep > fast_deep + 0.15) and (mt_deep > 1.0 / _V + 0.15)
    print(f"    DEEP (gap 80-150): multitimescale={mt_deep:.3f} vs random={rand_deep:.3f} "
          f"| fastonly={fast_deep:.3f} -> {'GO (SLOW time-constants load-bearing)' if go else 'no'}")
    return dict(seed=seed, gaps=gaps, accs=out, mt_deep=round(mt_deep, 3), rand_deep=round(rand_deep, 3), go=bool(go))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = a.seeds if a.seeds else [a.seed]
    t0 = time.time()
    res = [run(s) for s in seeds]
    if len(res) > 1:
        print(f"[ssm-reservoir] {sum(1 for r in res if r['go'])}/{len(res)} seeds GO")
    if a.out:
        json.dump(dict(results=res, elapsed_s=round(time.time() - t0, 1)), open(a.out, "w"))


if __name__ == "__main__":
    main()
