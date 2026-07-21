"""gap#2 (learned binder) Rank-1 CEILING — fixed-FHRR + iterative resonator cleanup, multi-bind retrieve@P=1..6 over
the 788 CORRELATED stream-cortex codes (phasor). Per the 2026-07-21 research gate: bound the whole binder arc with the
best-achievable-with-ideal-algebra separation curve BEFORE building the AKOrN local-Hebbian-J energy binder.

Task (the multi-bind SVO fact = a SUPERPOSITION of P bindings):
  codes  Z_c = exp(i 2pi theta_c)   (788 concept phasors, D=128, mean|phasor-cos| ~ the correlated 787-scale codes)
  roles  R_r = exp(i 2pi phi_r)      (P near-orthogonal developmental role keys)
  bind   b(r,f) = R_r (.) Z_f        (elementwise complex product = phase sum)
  fact   s = sum_i b(role_i, filler_i)              [P bindings superposed]
  unbind role_i:  u = s (.) conj(R_i) = Z_{f_i} + crosstalk    -> cleanup = argmax_c |<u, Z_c>|
RESONATOR (iterative cleanup, Frady-Sommer): estimate all fillers, then refine by SUBTRACTING the other estimated
bindings (residual = s - sum_{j!=i} b(role_j, fhat_j)) and re-unbinding -> removes crosstalk, the ceiling read.

Anti-cheats: permuted-role (wrong key -> chance); decorrelated-code control (op works on decorrelated codes too, so
the win is the OP not the correlation); the P=1 baseline. CPU/numpy, multi-seed. `--iters`, `--n-facts`, `--pmax`.
"""
import argparse
import glob
import json
import os
import numpy as np


def load_phasor_codes(pattern="bridges/developed/scale787/day_*/grounded_codes.npz", cap=None):
    path = sorted(glob.glob(pattern))[-1]                 # latest day = most concepts
    z = np.load(path, allow_pickle=True)
    keys = [k for k in z.keys() if k.startswith("g:")]
    theta = np.stack([np.asarray(z[k], np.float64) for k in keys])   # (N, D) phases in [0,1)
    if cap:
        theta = theta[:cap]
    return np.exp(2j * np.pi * theta), keys, path             # (N, D) complex unit phasors


def cleanup(u, Z):
    # argmax over the codebook of |<u, Z_c>| (complex matched filter) -> concept index
    score = np.abs(Z.conj() @ u)                          # (N,)
    return int(np.argmax(score))


def retrieve_at_P(Z, seed, P, n_facts, iters, permute=False, decorrelate=False):
    rng = np.random.default_rng(seed * 131 + P)
    N, D = Z.shape
    if decorrelate:
        th = rng.random((N, D)); ZB = np.exp(2j * np.pi * th)    # decorrelated phasor codebook, same size
    else:
        ZB = Z
    roles = np.exp(2j * np.pi * rng.random((P, D)))       # P developmental role phasors
    n_ok = n = 0
    for _ in range(n_facts):
        fids = rng.choice(N, P, replace=False)
        s = np.zeros(D, complex)
        for i in range(P):
            s = s + roles[i] * ZB[fids[i]]                # bind = phase-sum; fact = superposition
        # first-pass estimates
        est = [cleanup(s * roles[i].conj(), ZB) for i in range(P)]
        # resonator refinement: subtract the other estimated bindings, re-unbind
        for _it in range(iters):
            new = []
            for i in range(P):
                resid = s.copy()
                for j in range(P):
                    if j != i:
                        resid = resid - roles[j] * ZB[est[j]]
                key = roles[(i + 1) % P] if permute else roles[i]
                new.append(cleanup(resid * key.conj(), ZB))
            if new == est:
                est = new; break
            est = new
        for i in range(P):
            n_ok += int(est[i] == fids[i]); n += 1
    return n_ok / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--pmax", type=int, default=6)
    ap.add_argument("--n-facts", type=int, default=150)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--cap", type=int, default=None)
    args = ap.parse_args()

    Z, keys, path = load_phasor_codes(cap=args.cap)
    N, D = Z.shape
    Zn = Z / (np.abs(Z) + 1e-12)
    off = np.abs(Zn.conj() @ Zn.T)[~np.eye(N, dtype=bool)] / D
    print(f"[gap2 resonator CEILING] codes={N} D={D} from {os.path.basename(os.path.dirname(path))} | "
          f"mean phasor|cos|={np.mean(off):.3f} (correlated) | chance=1/{N}={1.0/N:.4f}")

    for P in range(1, args.pmax + 1):
        r = np.mean([retrieve_at_P(Z, s, P, args.n_facts, args.iters) for s in args.seeds])
        perm = np.mean([retrieve_at_P(Z, s, P, args.n_facts, args.iters, permute=True) for s in args.seeds]) if P > 1 else 0.0
        deco = np.mean([retrieve_at_P(Z, s, P, args.n_facts, args.iters, decorrelate=True) for s in args.seeds])
        print(f"  P={P}: retrieve {r:.3f} | permuted-role {perm:.3f} | decorrelated-ctrl {deco:.3f}"
              f"{'  <-- ceiling target >=0.80' if P >= 3 else ''}")
    print("  => this is the best-achievable-with-fixed-FHRR ceiling; the AKOrN local-Hebbian-J energy binder "
          "(ON vs this OFF baseline) must REACH it with a locally-written J (emergence bar).")


if __name__ == "__main__":
    main()
