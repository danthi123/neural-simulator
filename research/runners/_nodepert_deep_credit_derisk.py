"""FRESH-MECHANISM-CLASS de-risk (the fresh-spiking-deep-credit-class gate's TOP PICK): NODE PERTURBATION as a
zeroth-order, reward-modulated, three-factor deep-credit rule — does it train a depth-2 HIDDEN layer where the entire
feedback-alignment / burst-multiplexed family FAILED (2026-07-13 dense-redundant NEGATIVE, coupled, 6-seed)?

WHY IT MIGHT WORK WHERE FA FAILED (the load-bearing difference): the exhausted family carries an error signal BACKWARD
across hidden layers on spikes, and the boundary is that signal's credit-QUALITY (burst-multiplexing SNR). NODE
PERTURBATION injects NOTHING backward — no feedback matrix, no weight transport, no burst channel. Each hidden neuron
DISCOVERS its credit by correlating its OWN injected noise ξ with a GLOBAL scalar loss-difference dL = L(perturbed) −
L(clean): an UNBIASED zeroth-order estimate of ∂L/∂z (Fiete-Seung 2006; Williams REINFORCE). Emergence-fit: the
perturbation IS the intrinsic spiking noise; the update `ΔW = −η·(dL/σ²)·ξ ⊗ x` is the three-factor / neuromodulator-
gated Hebbian the sim already runs shallowly (the BG reward-modulated cascade). Single-phase (two forward EVALUATIONS,
no settling), local, no weight transport.

DE-RISK (single-variable swap on the D1 depth-2 tasks; numpy off-bridge FIRST, then on-bridge only after a GO):
reuse `_load_task` (emerge1 XOR-inheritance + dense-redundant — the exact tasks the FA/BDSP family sat at-or-below
chance on). Train a 2-hidden MLP: OUTPUT layer by the clean delta rule (it has a target); HIDDEN layers by NODE
PERTURBATION only. Compare held-out acc to the oracle (backprop ceiling) + the single-layer-linear floor + chance.
GO = the NP-trained depth-2 HIDDEN layers train ABOVE chance (where FA sat at/below) on >=5/6 seeds.
ANTI-CHEATS (single-variable, decisive): (1) SHUFFLED-dL (the loss-difference from a DIFFERENT example) MUST collapse to
the drift floor — proves the credit rides the REAL global signal, not the noise; (2) HIDDEN-FROZEN (train only the output
delta, hidden random) = the single-layer floor (depth must HELP); (3) WRONG-SIGN dL must anti-learn.

Run: python -m research.runners._nodepert_deep_credit_derisk --seeds 42 43 44 100 101 102 --task dense
     python -m research.runners._nodepert_deep_credit_derisk --seeds 42 --task emerge1 --smoke
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners._d1_onbridge_learn_to_accuracy_derisk import _load_task  # reuse the EXACT tasks the FA family failed


def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def _forward(Ws, x, noise=None):
    """2-hidden MLP: hidden tanh, output linear (logits). noise = per-hidden-layer pre-activation perturbation or None."""
    a = x; acts = [a]
    for i, W in enumerate(Ws):
        z = W @ a
        if noise is not None and i < len(noise) and noise[i] is not None:
            z = z + noise[i]
        a = np.tanh(z) if i < len(Ws) - 1 else z
        acts.append(a)
    return acts, acts[-1]                                        # acts[-1] = logits


def _ce(logits, y):
    p = _softmax(logits); return float(-np.log(max(p[y], 1e-12))), p


def _acc(Ws, X, y):
    return float(np.mean([int(np.argmax(_forward(Ws, X[i])[1]) == y[i]) for i in range(len(X))]))


def train_nodepert(Xtr, ytr, sizes, epochs, lr_hid, lr_out, sigma, seed, mode="np", k=1):
    """OUTPUT layer: clean delta rule. HIDDEN layers: NODE PERTURBATION (unless mode says otherwise).
    mode: 'np' | 'shuffle_dl' (dL from a permuted example — anti-cheat) | 'wrong_sign' | 'hidden_frozen' (floor).
    k = number of ANTITHETIC noise resamples averaged per example (variance reduction — the verdict's named lever; each
    resample uses +ξ and −ξ antithetically -> the linear-in-ξ term cancels the estimator's bias floor)."""
    rng = np.random.default_rng(seed)
    Ws = [rng.standard_normal((sizes[i + 1], sizes[i])) * (1.0 / np.sqrt(sizes[i])) for i in range(len(sizes) - 1)]
    nH = len(Ws) - 1
    idx = np.arange(len(Xtr))
    for _ep in range(epochs):
        rng.shuffle(idx)
        perm = rng.permutation(len(Xtr))
        for t, j in enumerate(idx):
            x, yj = Xtr[j], ytr[j]
            acts, logits = _forward(Ws, x)
            L, p = _ce(logits, yj)
            grad = [np.zeros_like(Ws[i]) for i in range(nH)]     # averaged hidden gradient estimate
            for _r in range(k):
                noise = [rng.standard_normal(sizes[i + 1]) * sigma for i in range(nH)] + [None]
                Lp = _ce(_forward(Ws, x, [n if n is None else n for n in noise])[1], yj)[0]
                Lm = _ce(_forward(Ws, x, [None if n is None else -n for n in noise])[1], yj)[0]  # antithetic −ξ
                dL = 0.5 * (Lp - Lm)                             # antithetic central estimate (cancels the even-order bias)
                if mode == "shuffle_dl":                        # anti-cheat: credit from a DIFFERENT example's loss-diff
                    jk = perm[t]; xk, yk = Xtr[jk], ytr[jk]
                    Lpk = _ce(_forward(Ws, xk, noise)[1], yk)[0]
                    Lmk = _ce(_forward(Ws, xk, [None if n is None else -n for n in noise])[1], yk)[0]
                    dL = 0.5 * (Lpk - Lmk)
                if mode == "wrong_sign":
                    dL = -dL
                for i in range(nH):
                    grad[i] += (dL / (sigma * sigma)) * np.outer(noise[i], acts[i])
            if mode != "hidden_frozen":
                for i in range(nH):
                    Ws[i] += -lr_hid * grad[i] / k
            # OUTPUT layer: clean delta rule (it HAS a target)
            delta = p.copy(); delta[yj] -= 1.0
            Ws[-1] += -lr_out * np.outer(delta, acts[-2])
    return Ws


def run(seed, task, parity_bits, epochs, lr_hid, lr_out, sigma, hidden, k=1, depth=2):
    (Xtr, ytr), (Xte, yte), n_bits = _load_task(task, seed, parity_bits)
    Xtr = np.asarray(Xtr, np.float64); Xte = np.asarray(Xte, np.float64)
    n_cls = int(max(ytr.max(), yte.max())) + 1
    sizes = [n_bits] + [hidden] * depth + [n_cls]               # `depth` hidden layers (does NP's credit hold as depth GROWS?)
    chance = float(np.bincount(yte, minlength=n_cls).max() / len(yte))
    out = {"seed": seed, "task": task, "depth": depth, "chance": round(chance, 3)}
    for mode in ("np", "shuffle_dl", "wrong_sign", "hidden_frozen"):
        Ws = train_nodepert(Xtr, ytr, sizes, epochs, lr_hid, lr_out, sigma, seed, mode=mode, k=k)
        out[mode] = round(_acc(Ws, Xte, yte), 3)
    # oracle/floor for the ceiling + the "depth must help" reference
    out["margin_over_chance"] = round(out["np"] - chance, 3)
    out["np_beats_chance"] = bool(out["np"] > chance + 0.05)
    out["shuffle_collapses"] = bool(out["shuffle_dl"] <= chance + 0.05)      # anti-cheat: shuffled-dL at chance
    out["depth_helps"] = bool(out["np"] > out["hidden_frozen"] + 0.05)      # NP hidden > frozen-hidden floor
    out["GO"] = bool(out["np_beats_chance"] and out["shuffle_collapses"] and out["depth_helps"])
    print(f"[nodepert seed={seed} {task}] chance={chance:.3f} | NP={out['np']:.3f} shuffle_dl={out['shuffle_dl']:.3f} "
          f"wrong_sign={out['wrong_sign']:.3f} hidden_frozen={out['hidden_frozen']:.3f} "
          f"-> {'GO' if out['GO'] else 'no'} (beats_chance={out['np_beats_chance']} shuffle_collapses={out['shuffle_collapses']} depth_helps={out['depth_helps']})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--task", choices=["emerge1", "parity", "dense"], default="dense")
    ap.add_argument("--parity-bits", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr-hid", type=float, default=0.02); ap.add_argument("--lr-out", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=0.5); ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--k", type=int, default=1, help="antithetic node-perturbation resamples per example (variance reduction)")
    ap.add_argument("--depth", type=int, default=2, help="number of hidden layers (does NP's credit hold as depth grows?)")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    if a.smoke:
        a.epochs = 30
    res = [run(s, a.task, a.parity_bits, a.epochs, a.lr_hid, a.lr_out, a.sigma, a.hidden, k=a.k, depth=a.depth) for s in a.seeds]
    if len(res) > 1:
        ng = sum(1 for r in res if r["GO"])
        print(f"[nodepert] {ng}/{len(res)} seeds GO | mean NP margin over chance "
              f"{np.mean([r['margin_over_chance'] for r in res]):+.3f}", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"))


if __name__ == "__main__":
    main()
