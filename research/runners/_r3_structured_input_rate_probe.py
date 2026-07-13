"""CHEAP RATE PROBE (before touching the 678-line spiking runner): is a distal-decode task a VALID regime for the R3
"learn W_in beats fixed W_in" claim — and does INPUT STRUCTURE create the headroom? The `2026-07-13-R3-spiking-Win-...`
finding showed the K-cue ONE-HOT distal-decode has NO headroom (a fixed random W_in separates orthogonal cues even at
dist=32; the rate reference confirmed learn≈fixed) because orthogonal cues have nothing to LEARN. Hypothesis: with
OVERLAPPING (structured) cue codes — where a random W_in scrambles exploitable structure but a learned W_in maps them to
a separable subspace — a learned W_in beats a fixed one (the R3 property, as on the LANGUAGE task).

DESIGN: a FIXED random ESN reservoir (W_rec frozen, spectral 0.95 — the R3 reframe: training W_rec hurts). Task =
[cue_k code] . filler×d . [query]; decode k from the reservoir state at the query. cue codes over D_in features:
  ORTHOGONAL: one-hot (cue k -> feature k).                 (the current spiking-runner regime -> no headroom)
  STRUCTURED: each cue = a sparse SUM of shared feature atoms (codes OVERLAP) -> a random W_in collides them.
ARMS x {orthogonal, structured}: fixed_win (random W_in, train read-out only) vs learn_win (train W_in + read-out by
BPTT through the SHORT unroll with W_rec FROZEN = the R3 BPTT_frozen_wrec CEILING = upper bound on "does learning W_in
help"). GATE: learn_win - fixed_win >= +0.10 with STRUCTURED codes AND ~0 with ORTHOGONAL codes -> input structure is the
headroom, distal-decode IS a valid R3 regime once the input is structured (-> modify the spiking runner's cue codes).
If learn~=fixed even with structured codes -> distal-decode is NOT the R3 regime; the faithful spiking test is the
LANGUAGE task. numpy-CPU; self-contained; NO `sim/` edit.

Run: python -m research.runners._r3_structured_input_rate_probe --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json

import numpy as np

_N = 200          # reservoir units (overridable via --n-pool)
_D_IN = 24        # input feature dimension (overridable via --d-in)
_N_ATOMS = 8      # structured codes = sums of a SMALL shared atom pool -> heavy overlap -> a random W_in COLLIDES them
_ACTIVE = 3       # atoms combined per cue


def _cue_codes(K, kind, rng):
    """K cue codes over _D_IN features. orthogonal: one-hot (needs _D_IN>=K). structured: each cue = a sum of _ACTIVE
    atoms from a SMALL shared pool of _N_ATOMS random dense atoms -> the codes OVERLAP HEAVILY (a random W_in collides
    them; a learned W_in that de-mixes the shared atoms could separate them)."""
    if kind == "orthogonal":
        assert _D_IN >= K, "orthogonal needs D_in >= K"
        C = np.eye(_D_IN)[:K]
    else:                                                    # structured: overlapping combinations of shared atoms
        atoms = rng.standard_normal((_N_ATOMS, _D_IN))       # shared dense atoms
        C = np.zeros((K, _D_IN))
        for k in range(K):
            sel = rng.choice(_N_ATOMS, size=_ACTIVE, replace=False)
            C[k] = atoms[sel].sum(0)
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    return C


def _build_reservoir(rng):
    W = rng.standard_normal((_N, _N)); ev = np.max(np.abs(np.linalg.eigvals(W)))
    return (0.95 / ev) * W                                   # frozen ESN recurrence


def _forward(W_rec, W_in, code_seq):
    """Run the ESN over the input-code sequence; return states [T, _N] (h_t after each token) — for BPTT."""
    h = np.zeros(_N); H = []
    pre = []
    for x in code_seq:
        z = W_rec @ h + W_in @ x
        h = np.tanh(z)
        pre.append(z); H.append(h)
    return np.asarray(H), np.asarray(pre)


def _make_examples(K, d, C, n_per, rng):
    """[cue_k] filler×d [query]; filler + query are ZERO input codes (silent) -> the reservoir must HOLD the cue. Returns
    list of (code_seq [T, _D_IN], label k). T = 1 + d + 1."""
    zero = np.zeros(_D_IN)
    ex = []
    for _ in range(n_per):
        for k in range(K):
            seq = [C[k]] + [zero] * d + [zero]               # query = zero code (decode from the HELD state)
            ex.append((np.asarray(seq), k))
    rng.shuffle(ex)
    return ex


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)


def _fixed_acc(W_rec, W_in, tr, ev, K, epochs=200, lr=0.5):
    """fixed W_in: collect query states, train a softmax read-out only, decode acc on eval."""
    def qstate(ex):
        H, _ = _forward(W_rec, W_in, ex[0]); return H[-1]
    Xtr = np.asarray([qstate(e) for e in tr]); ytr = np.asarray([e[1] for e in tr])
    Xev = np.asarray([qstate(e) for e in ev]); yev = np.asarray([e[1] for e in ev])
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr = np.concatenate([(Xtr - m) / s, np.ones((len(Xtr), 1))], 1)
    Zev = np.concatenate([(Xev - m) / s, np.ones((len(Xev), 1))], 1)
    W = np.zeros((_N + 1, K)); OH = np.zeros((len(ytr), K)); OH[np.arange(len(ytr)), ytr] = 1
    for _ in range(epochs):
        P = _softmax(Ztr @ W); W -= lr * (Ztr.T @ (P - OH) / len(ytr) + 1e-4 * W)
    return float(np.mean(np.argmax(Zev @ W, 1) == yev))


def _learn_acc(W_rec, W_in0, tr, ev, K, epochs=120, lr_in=0.05, lr_out=0.5):
    """learn W_in + read-out by BPTT through the SHORT unroll with W_rec FROZEN (the R3 BPTT_frozen_wrec ceiling)."""
    W_in = W_in0.copy(); Wout = np.zeros((_N + 1, K))
    m = np.zeros(_N); s = np.ones(_N)
    for ep in range(epochs):
        # refit standardization on current W_in each few epochs
        if ep % 20 == 0:
            Q = np.asarray([_forward(W_rec, W_in, e[0])[0][-1] for e in tr])
            m, s = Q.mean(0), Q.std(0) + 1e-6
        gWin = np.zeros_like(W_in); gWout = np.zeros_like(Wout)
        for seq, lab in tr:
            H, pre = _forward(W_rec, W_in, seq)               # H[T, N]
            hq = (H[-1] - m) / s
            z = np.concatenate([hq, [1.0]]) @ Wout
            p = _softmax(z[None])[0]; p[lab] -= 1.0           # dCE/dlogits
            gWout += np.outer(np.concatenate([hq, [1.0]]), p)
            # backprop into h_query only (dominant path; the cue enters h_query via the frozen recurrence over the unroll)
            dh = (Wout[:-1] @ p) / s                          # dL/dH[-1]
            for t in range(len(seq) - 1, -1, -1):
                dz = dh * (1.0 - np.tanh(pre[t]) ** 2)        # through tanh
                gWin += np.outer(dz, seq[t])                  # dL/dW_in from this step's input
                dh = W_rec.T @ dz                             # propagate to h_{t-1} (W_rec frozen -> no grad)
        W_in -= lr_in * gWin / len(tr)
        Wout -= lr_out * gWout / len(tr)
    # eval
    Q = np.asarray([_forward(W_rec, W_in, e[0])[0][-1] for e in tr]); m, s = Q.mean(0), Q.std(0) + 1e-6
    def acc(exs):
        X = np.asarray([_forward(W_rec, W_in, e[0])[0][-1] for e in exs]); y = np.asarray([e[1] for e in exs])
        Z = np.concatenate([(X - m) / s, np.ones((len(X), 1))], 1)
        return float(np.mean(np.argmax(Z @ Wout, 1) == y))
    return acc(ev)


def run(seed, K, d, n_per):
    rng = np.random.default_rng(seed)
    W_rec = _build_reservoir(np.random.default_rng(seed * 31 + 5))
    W_in0 = np.random.default_rng(seed * 7919 + 3).standard_normal((_N, _D_IN)) * (1.0 / np.sqrt(_D_IN))
    out = {"seed": seed, "K": K, "d": d}
    for kind in ("orthogonal", "structured"):
        C = _cue_codes(K, kind, np.random.default_rng(seed * 101 + 7))
        tr = _make_examples(K, d, C, n_per, rng); ev = _make_examples(K, d, C, 1, np.random.default_rng(seed * 3 + 1))
        fx = _fixed_acc(W_rec, W_in0, tr, ev, K)
        ln = _learn_acc(W_rec, W_in0, tr, ev, K)
        out[kind] = {"fixed": round(fx, 3), "learn": round(ln, 3), "margin": round(ln - fx, 3)}
    print(f"[r3-struct seed={seed} K={K} d={d}] chance={1/K:.3f}", flush=True)
    for kind in ("orthogonal", "structured"):
        o = out[kind]
        print(f"    {kind:11s}: fixed={o['fixed']:.3f}  learn={o['learn']:.3f}  margin={o['margin']:+.3f}", flush=True)
    out["structure_is_headroom"] = bool(out["structured"]["margin"] >= 0.10 and out["orthogonal"]["margin"] < 0.10)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--K", type=int, default=16); ap.add_argument("--d", type=int, default=6)
    ap.add_argument("--n-per", type=int, default=8); ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--n-pool", type=int, default=200); ap.add_argument("--d-in", type=int, default=24)
    ap.add_argument("--n-atoms", type=int, default=8); ap.add_argument("--active", type=int, default=3)
    a = ap.parse_args()
    global _N, _D_IN, _N_ATOMS, _ACTIVE
    _N = a.n_pool; _D_IN = a.d_in; _N_ATOMS = a.n_atoms; _ACTIVE = a.active
    res = [run(s, a.K, a.d, a.n_per) for s in a.seeds]
    if len(res) > 1:
        ns = sum(1 for r in res if r["structure_is_headroom"])
        sm = np.mean([r["structured"]["margin"] for r in res]); om = np.mean([r["orthogonal"]["margin"] for r in res])
        print(f"[r3-struct] structure-is-headroom {ns}/{len(res)} | mean margin structured={sm:+.3f} orthogonal={om:+.3f}", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"))


if __name__ == "__main__":
    main()
