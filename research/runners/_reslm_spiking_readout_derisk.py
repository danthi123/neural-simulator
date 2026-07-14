"""SPIKING READ-OUT for the emergent generator (closing the generator's last non-spiking piece): the reslm reservoir is
already spiking (EMERGE-82 OnBridgeLSM), but its next-token READ-OUT is a numpy linear-softmax argmax. This converts the
read-out to SPIKES: the linear read-out's per-token scores drive a one-of-K FS-WTA Izhikevich bridge (the validated
`build_fswta_score_bridge`/`fswta_drive`, the same spiking WTA the D3 register uses), and the SPIKING winner is the
predicted next token. NO `sim/` edit (reuse-by-import).

The read-out WEIGHTS are still learned by the committed LOCAL one-step delta rule (Widrow-Hoff on the clean next-token
error, no BPTT); only the argmax SELECTION is moved onto spikes. Cheap-first: a self-contained small reservoir + a trained
linear read-out on a toy next-token task, then the spiking-WTA read matched against the numpy argmax.

GO (6-seed 42/43/44/100/101/102 + FRESH 7/8/9/10/11/12): the spiking-WTA next-token prediction AGREES with the numpy
argmax read (parity) AND both beat chance, with a SHUFFLED-score control (FS-WTA on permuted scores) disagreeing. ⇒ the
generator's read-out is realizable on spikes with no accuracy loss.

Run: SIM_BACKEND=numpy python -m research.runners._reslm_spiking_readout_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("SIM_BACKEND", "numpy")
import argparse
import json
import numpy as np

from research.runners._d3_spiking_attractor_derisk import build_fswta_score_bridge, fswta_drive

V = 12                       # toy vocabulary (a K=V FS-WTA bridge); --vocab overrides (e.g. 200 = real-scale WTA check)
N_POOL = 150                 # reservoir size
N_SEQ = 400                  # training sequences
SEQ_LEN = 6


def _reservoir(seed):
    rng = np.random.default_rng(seed * 7 + 1)
    Win = rng.standard_normal((N_POOL, V)) * 0.7
    W = rng.standard_normal((N_POOL, N_POOL))
    W *= 0.9 / (np.max(np.abs(np.linalg.eigvals(W))) + 1e-9)
    return Win, W


def _states_and_targets(seed, Win, W, easy=False):
    """A toy next-token task: next = f(last two tokens) (2nd-order; --easy = 1st-order = f(prev1), learnable to high
    accuracy at large V -> discriminable read-out scores that isolate the FS-WTA's discrimination from near-uniform
    scores). Returns per-step reservoir states + the next-token targets."""
    rng = np.random.default_rng(seed * 11 + 3)
    rule = rng.integers(0, V, (V, V))                            # next = rule[prev2, prev1]
    rule1 = rng.integers(0, V, V)                               # next = rule1[prev1] (1st-order, easy)
    X, Y = [], []
    for _ in range(N_SEQ):
        toks = list(rng.integers(0, V, 2))
        x = np.zeros(N_POOL)
        for t in range(2):
            e = np.zeros(V); e[toks[t]] = 1.0
            x = np.tanh(W @ x + Win @ e)
        for _ in range(SEQ_LEN):
            nxt = int(rule1[toks[-1]]) if easy else int(rule[toks[-2], toks[-1]])
            X.append(x.copy()); Y.append(nxt)
            e = np.zeros(V); e[nxt] = 1.0
            x = np.tanh(W @ x + Win @ e)
            toks.append(nxt)
    return np.array(X), np.array(Y)


def _train_readout(X, Y):
    """The committed LOCAL one-step delta rule (Widrow-Hoff on the clean next-token error)."""
    Wro = np.zeros((V, N_POOL)); lr = 0.02
    for _ in range(6):
        for i in range(len(X)):
            z = Wro @ X[i]; p = np.exp(z - z.max()); p /= p.sum()
            t = np.zeros(V); t[Y[i]] = 1.0
            Wro += lr * np.outer(t - p, X[i])
    return Wro


def run(seed):
    Win, W = _reservoir(seed)
    X, Y = _states_and_targets(seed, Win, W)
    ntr = int(0.7 * len(X))
    Wro = _train_readout(X[:ntr], Y[:ntr])
    Xte, Yte = X[ntr:], Y[ntr:]
    sb = build_fswta_score_bridge(seed=seed, K=V)
    numpy_pred, spk_pred, shuf_pred = [], [], []
    rng = np.random.default_rng(seed * 5 + 9)
    for i in range(len(Xte)):
        scores = Wro @ Xte[i]                                   # the linear read-out's per-token scores
        numpy_pred.append(int(np.argmax(scores)))
        _, acc = fswta_drive(sb, V, scores, settle=25)         # drive the spiking one-of-K WTA with the scores
        spk_pred.append(int(np.argmax(acc)) if acc.max() > 0 else -1)
        sh = scores[rng.permutation(V)]                        # anti-cheat: permuted scores
        _, acc2 = fswta_drive(sb, V, sh, settle=25)
        shuf_pred.append(int(np.argmax(acc2)) if acc2.max() > 0 else -1)
    numpy_pred = np.array(numpy_pred); spk_pred = np.array(spk_pred); shuf_pred = np.array(shuf_pred)
    agree = float(np.mean(spk_pred == numpy_pred))             # spiking-WTA vs numpy argmax parity
    np_acc = float(np.mean(numpy_pred == Yte))
    spk_acc = float(np.mean(spk_pred == Yte))
    shuf_agree = float(np.mean(shuf_pred == numpy_pred))
    go = bool(agree > 0.9 and spk_acc > 1.5 / V and abs(spk_acc - np_acc) < 0.05 and shuf_agree < 0.5)
    print(f"[spk-readout seed={seed}] parity(spk==numpy)={agree:.3f} numpy_acc={np_acc:.3f} spk_acc={spk_acc:.3f} "
          f"shuffle_parity={shuf_agree:.3f} -> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, "parity": agree, "numpy_acc": np_acc, "spk_acc": spk_acc, "shuffle_parity": shuf_agree, "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--vocab", type=int, default=None, help="override V (e.g. 200 = real-scale FS-WTA discrimination)")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    if a.vocab is not None:
        global V
        V = a.vocab
    res = [run(s) for s in a.seeds]
    print(f"[spk-readout] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
