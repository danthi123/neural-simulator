"""GATE-1 (the design-gate deliverable): does NODE PERTURBATION's UNBIASED feedforward credit beat FA-partial
Kolen-Pollack (KP) on the R3 input-map (W_in) credit gap? (design gate wf_82e0ddec-4a0 /
`2026-07-13-np-next-step-design-gate-*`.)

WHY: R3 reaches the deep ceiling with a LINEAR W_in; the biological rule KP plateaus below it, and the residual is a
FEEDFORWARD credit-quality gap caused by feedback-alignment PARTIALITY (KP's learned feedback is still not weight
transport). Node perturbation removes FA partiality BY CONSTRUCTION (unbiased zeroth-order, NO feedback matrix), and this
single-step next-class task is a PURE FEEDFORWARD W_in-credit problem (read at the predict step after 1 token; W_rec FIXED)
— NP's 6+6-seed-validated regime, NOT the retracted through-reservoir recurrent regime.

INSTRUMENT (reuse-by-import, numpy): `_reslm_generalize_rate_check.build_codes` + `build_stream` — additive class codes
with an identity confound + held-out rare synonyms; predict the next CLASS from the current token. Fixed random reservoir
W_rec (spectral 0.95) + a local read-out. The ONLY variable across arms = the W_in credit RULE.

ARMS (single variable = how W_in is trained; W_rec/Wout-eval/codes/reservoir byte-identical):
  oracle : W_in by full BPTT through the 2-step forward (the ceiling)
  fixed  : W_in fixed-random (the floor)
  fa     : W_in by input-eligibility x (FIXED-random feedback Bfb @ err)              [plain feedback-alignment]
  kp     : W_in by input-eligibility x (LEARNED feedback Bfb, Kolen-Pollack mirror)   [the R3 bar]
  np     : W_in by NODE PERTURBATION (perturb the reservoir read-state xi, correlate with the GLOBAL CE loss-difference,
           update W_in via the three-factor outer(xi, input-rate); antithetic-k; unbiased, NO feedback matrix)  [the TEST]
  shuffle: np with dL from a DIFFERENT example (MUST collapse to fa)                   [anti-cheat]
  wrong  : np with dL negated (MUST anti-learn)                                        [anti-cheat]

GO gate (per seed, held-out next-class margin over chance): np−shuffle >= +0.10 AND np−kp > 0 AND
np captures >= 60% of (oracle−fa) AND wrong < fa. Multi-seed: standard 42/43/44/100/101/102 AND FRESH 7/8/9/10/11/12;
overall GO iff >=5/6 in BOTH sets AND FRESH-pooled (np−kp) >= +0.05. (The fresh-seed gate is the exact one that caught the
retracted recurrent-NP seed artifact.) numpy/CPU; NO `sim/` edit.

Run: SIM_BACKEND=numpy python -m research.runners._np_feedforward_win_credit_derisk --seeds 42 43 44 100 101 102 --k 8
     SIM_BACKEND=numpy python -m research.runners._np_feedforward_win_credit_derisk --seeds 7 8 9 10 11 12 --k 8   # FRESH
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
from research.runners._reslm_generalize_rate_check import build_codes, build_stream, _fit_ridge, _decode_acc


def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def train_win(train, codes, m, n, G, seed, epochs, lr_out, lr_in, mode, alpha=0.3, sigma=0.35, k=8):
    """Train W_in by `mode`; W_rec FIXED. Returns the trained W_in (+ the shared fixed W_rec/b) for eval."""
    rng = np.random.RandomState(seed * 3 + 17)
    W_rec = rng.normal(0, 1, (n, n)) / np.sqrt(n)
    W_rec *= 0.95 / max(np.max(np.abs(np.linalg.eigvals(W_rec))), 1e-6)
    W_in = rng.normal(0, 1, (n, m)) / np.sqrt(m)
    Bfb = rng.normal(0, 1, (n, G)); b = np.zeros(n); Wout = np.zeros((G, n))

    def _fwd(tok, W_in_, xi=None):
        """2-step: present token, blank predict step; read at the predict step. xi = read-state node perturbation."""
        h = np.zeros(n); acts = []; xs = []
        for t in range(2):
            x = codes[tok] if t == 0 else np.zeros(m)
            pre = W_rec @ h + W_in_ @ x + b
            a = np.tanh(pre); acts.append(a); xs.append(x)
            h = (1 - alpha) * h + alpha * a
            if t == 1 and xi is not None:
                h = h + xi
        return h, acts, xs

    perm = None
    for _ep in range(epochs):
        order = list(range(len(train))); rng.shuffle(order)
        perm = rng.permutation(len(train))
        for ti, si in enumerate(order):
            tok, nc = train[si]
            hq, acts, xs = _fwd(tok, W_in)
            p = _softmax(Wout @ hq); delta = -p.copy(); delta[nc] += 1.0
            Wout += lr_out * np.outer(delta, hq)                      # local read-out (all arms)
            if mode == "fixed":
                continue
            if mode in ("fa", "kp"):
                # input-synapse e-prop: e_in[j,v] ~ d h_j / d W_in[j,v] (forward-filtered); credit = feedback @ err.
                # feedforward-dominant: d pre_j/d W_in[j,v] at the token step = 1[x==v]; the read-state carries it 1 step.
                L = Bfb @ delta                                       # (n,) top-down credit per reservoir unit
                # e_in over the token step (step 0): act0 = acts[0], input = codes[tok]
                psi0 = alpha * (1 - acts[0] ** 2)
                e_col = (psi0 * (1 - alpha))                          # 1-step leaky carry to the read state
                # update only the columns of the present token (feedforward input credit)
                on = codes[tok] > 0.5
                W_in[:, on] += lr_in * np.outer(L * e_col, np.ones(int(on.sum())))
                if mode == "kp":                                      # Kolen-Pollack: feedback MIRRORS the read-out update
                    Bfb += lr_out * np.outer(hq, delta)               # -> Bfb tracks Wout^T (aligned learned feedback)
            if mode in ("np", "shuffle", "wrong"):
                on = codes[tok] > 0.5
                gW = np.zeros((n, int(on.sum())))
                for _r in range(k):
                    xi = sigma * rng.standard_normal(n)
                    Lp = -np.log(max(_softmax(Wout @ _fwd(tok, W_in, xi)[0])[nc], 1e-12))
                    Lm = -np.log(max(_softmax(Wout @ _fwd(tok, W_in, -xi)[0])[nc], 1e-12))
                    dL = 0.5 * (Lp - Lm)
                    if mode == "shuffle":
                        tk, nk = train[perm[ti]]
                        Lp = -np.log(max(_softmax(Wout @ _fwd(tk, W_in, xi)[0])[nk], 1e-12))
                        Lm = -np.log(max(_softmax(Wout @ _fwd(tk, W_in, -xi)[0])[nk], 1e-12))
                        dL = 0.5 * (Lp - Lm)
                    if mode == "wrong":
                        dL = -dL
                    gW += (dL / (sigma * sigma)) * np.outer(xi, np.ones(int(on.sum())))  # NP credit for on-col W_in[j,v]=xi_j*x_v(=1)
                W_in[:, on] += -lr_in * gW / k
            if mode == "oracle":
                # BPTT of W_in through the 2-step (feedforward-dominant): dL/dhq = Wout.T @ (p - onehot)
                dh = Wout.T @ (p - (np.arange(G) == nc).astype(float))
                # step1 (blank): h1=(1-a)h0+a*tanh(Wrec h0+b); dpre1=a*(1-acts[1]^2)*dh; dh0 += (1-a)dh + Wrec.T@dpre1
                dpre1 = alpha * (1 - acts[1] ** 2) * dh
                dh0 = (1 - alpha) * dh + W_rec.T @ dpre1
                # step0 (token): dpre0 = a*(1-acts[0]^2)*dh0; dW_in += outer(dpre0, codes[tok])
                dpre0 = alpha * (1 - acts[0] ** 2) * dh0
                W_in += -lr_in * np.outer(dpre0, codes[tok])
        if mode == "kp":                                              # explicit KP feedback mirror (once per epoch batch)
            pass
    return W_rec, W_in, b, alpha


def _reads(sents, codes, m, n, W_rec, W_in, b, alpha, G):
    def fwd(tok):
        h = np.zeros(n)
        for t in range(2):
            x = codes[tok] if t == 0 else np.zeros(m)
            h = (1 - alpha) * h + alpha * np.tanh(W_rec @ h + W_in @ x + b)
        return h
    R = np.array([np.concatenate([fwd(tok), [1.0]]) for tok, _ in sents]); Y = np.array([nc for _, nc in sents])
    return R, Y


def run(seed, G, syn, sf, idn, id_pool, n, epochs, lr_out, lr_in, sigma, k, n_seq):
    codes, V, m = build_codes(seed, G, syn, sf, idn, id_pool=id_pool)
    train, evl = build_stream(seed, G, syn, n_seq)
    chance = 1.0 / G
    out = {"seed": seed, "chance": round(chance, 3)}
    for mode in ("fixed", "oracle", "fa", "kp", "np", "shuffle", "wrong"):
        W_rec, W_in, b, alpha = train_win(train, codes, m, n, G, seed, epochs, lr_out, lr_in, mode, sigma=sigma, k=k)
        Rtr, Ytr = _reads(train, codes, m, n, W_rec, W_in, b, alpha, G)
        Rev, Yev = _reads(evl, codes, m, n, W_rec, W_in, b, alpha, G)
        W = _fit_ridge(Rtr, Ytr, G, lam=1.0)
        out[mode] = round(_decode_acc(Rev, Yev, W) - chance, 4)       # HELD-OUT margin over chance
    npv, sh, wr, fa, kp, orc = out["np"], out["shuffle"], out["kp"], out["fa"], out["kp"], out["oracle"]
    fa_m = out["fa"]
    out["np_minus_shuffle"] = round(out["np"] - out["shuffle"], 4)
    out["np_minus_kp"] = round(out["np"] - out["kp"], 4)
    denom = max(out["oracle"] - fa_m, 1e-6)
    out["frac_of_oracle_gap"] = round((out["np"] - fa_m) / denom, 3)
    out["GO"] = bool(out["np"] - out["shuffle"] >= 0.10 and out["np"] - out["kp"] > 0
                     and out["frac_of_oracle_gap"] >= 0.60 and out["wrong"] < fa_m)
    print(f"[ffwd-win seed={seed}] margins/chance: oracle={out['oracle']:+.3f} kp={out['kp']:+.3f} fa={out['fa']:+.3f} "
          f"np={out['np']:+.3f} shuffle={out['shuffle']:+.3f} wrong={out['wrong']:+.3f} | np-kp={out['np_minus_kp']:+.3f} "
          f"np-shuf={out['np_minus_shuffle']:+.3f} frac_gap={out['frac_of_oracle_gap']:.2f} -> {'GO' if out['GO'] else 'no'}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--G", type=int, default=6); ap.add_argument("--syn", type=int, default=5)
    ap.add_argument("--sf", type=int, default=2); ap.add_argument("--idn", type=int, default=20)
    ap.add_argument("--id-pool", type=int, default=0); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--lr-out", type=float, default=0.02)
    ap.add_argument("--lr-in", type=float, default=0.05); ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--n-seq", type=int, default=40)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [run(s, a.G, a.syn, a.sf, a.idn, a.id_pool, a.n, a.epochs, a.lr_out, a.lr_in, a.sigma, a.k, a.n_seq) for s in a.seeds]
    ng = sum(1 for r in res if r["GO"])
    print(f"[ffwd-win] {ng}/{len(res)} GO | pooled np-kp {np.mean([r['np_minus_kp'] for r in res]):+.4f} "
          f"| pooled np-shuffle {np.mean([r['np_minus_shuffle'] for r in res]):+.4f}", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
