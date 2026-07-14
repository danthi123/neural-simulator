"""PAST-RESERVOIR Rung 2 (from the 2026-07-13 fresh-mechanism-class gate + the Zucchet source-read): a per-neuron
SELECTIVE (input-dependent) DIAGONAL SSM, trained by an EXACT forward-mode ELIGIBILITY TRACE (no BPTT, no weight
transport), beats a fixed-lambda reservoir on a long-range CONJUNCTION task that the reservoir cannot do — carrying the
Rung-1 conjunction ingredient RECURRENTLY across distance. NO `sim/` edit; self-contained numpy.

MECHANISM (derived from Zucchet arXiv:2305.15947, confirmed to survive per-neuron input-dependence — see
`2026-07-13-Zucchet-diagonal-RTRL-SURVIVES-...`):
  gated leaky integrator, per neuron i:   h_{t,i} = lam_{t,i} * h_{t-1,i} + (1 - lam_{t,i}) * inj_{t,i}
  input-dependent (selective) gate:        lam_{t,i} = sigmoid( w_i . u_t + c_i )        (theta_i = w_i, c_i -- PER NEURON)
  injection:                               inj_{t,i} = ( W_in u_t )_i                    (W_in FIXED random)
  EXACT forward-mode eligibility (local, O(n*d_in)):
     e^w_{i,t} = lam_{t,i} * e^w_{i,t-1} + (dlam/dw_i)*(h_{t-1,i} - inj_{t,i}) ,  dlam/dw_i = lam(1-lam) u_t
  read-out (linear, delta-rule) at the read step; local error delta_i = sum_v (p_v - tgt_v) W_ro[v,i] (spatial backprop
  through the read-out only); gate update  Δtheta_i ∝ delta_i * e^theta_{i,read}. Forward-mode, transport-free.

TASK (needs a long-range conjunction): [KEY, filler×D, QUERY] -> target = rule[KEY, QUERY]. The distal KEY must be HELD
(the reservoir fades it) AND conjoined with the recent QUERY (the Rung-1 product). A learned input-dependent gate can
hold the key (lam≈1 during filler) and route/conjoin at the query; a fixed-lambda reservoir cannot.

ARMS (single variable = the gate):
  - selective:  input-dependent lam, gate params (w,c) TRAINED by the eligibility trace + read-out delta-rule
  - fixed_res:  FIXED per-neuron lambda (a leaky ESN, Rung-1 baseline), only the read-out trained
  - detached:   input-dependent lam but gate params FIXED random (not trained) -> tests that LEARNING the gate matters
  - permgate:   gate reads a PERMUTED input -> selectivity uninformative (anti-cheat)

GO (6-seed 42/43/44/100/101/102 + FRESH 7/8/9/10/11/12): selective accuracy > fixed_res + margin AND > detached + margin
AND > permgate + margin AND > chance. ⇒ a locally-trained (transport-free) selective diagonal SSM captures the recurrent
conjunction the fixed reservoir cannot — the honest path past the reservoir bound.

Run: python -m research.runners._reslm_rung2_selective_ssm_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import numpy as np

K = 6                        # number of KEY/QUERY symbols (target = rule[key, query], K*K classes collapsed to K)
D_IN = 10                    # token embedding dim
N_HID = 64                   # SSM units
DEPTH = 12                   # filler length between KEY and QUERY (the distal gap the reservoir fades)
N_SEQ = 900
EPOCHS = 6
LR_RO = 0.05                 # read-out delta-rule lr
LR_GATE = 0.05               # gate eligibility-trained lr
FORGET_BIAS = 2.5            # gate bias init (Jozefowicz 2015): start lam~0.9 so the eligibility trace survives the
                             # filler and the gate can LEARN when to hold/release (else lam~0.5 fades the key by lam^D)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _embed(seed):
    rng = np.random.default_rng(seed * 3 + 1)
    # tokens: 0..K-1 = symbols (used as both key and query), K = filler
    return rng.standard_normal((K + 1, D_IN)) * 0.8


def _make_seqs(seed):
    rng = np.random.default_rng(seed * 11 + 5)
    rule = rng.integers(0, K, (K, K))                          # target class = rule[key, query]
    seqs = []
    for _ in range(N_SEQ):
        key = int(rng.integers(0, K)); query = int(rng.integers(0, K))
        toks = [key] + [K] * DEPTH + [query]                   # KEY, filler..., QUERY
        seqs.append((toks, int(rule[key, query])))
    return seqs, rule


def _params(seed):
    rng = np.random.default_rng(seed * 7 + 2)
    Win = rng.standard_normal((N_HID, D_IN)) / np.sqrt(D_IN)   # fixed injection
    w = rng.standard_normal((N_HID, D_IN)) / np.sqrt(D_IN)     # gate input weights (trained in 'selective')
    c = np.full(N_HID, FORGET_BIAS)                            # gate bias -> lam starts ~0.9 (holds; trace survives)
    fixed_lam = _sigmoid(rng.standard_normal(N_HID) * 0.5 + FORGET_BIAS)  # fixed leaky ESN baseline (same init scale)
    return Win, w, c, fixed_lam


def _forward(toks, E, Win, w, c, fixed_lam, arm, permidx):
    """Run the SSM over a token sequence; return final h + (for the selective arm) the eligibility traces e^w,e^c at the
    final (read) step. Gate: input-dependent (selective/detached/permgate) or fixed (fixed_res)."""
    h = np.zeros(N_HID)
    ew = np.zeros((N_HID, D_IN)); ec = np.zeros(N_HID)
    for t, tok in enumerate(toks):
        u = E[tok]
        inj = Win @ u
        if arm == "fixed_res":
            lam = fixed_lam
        else:
            ug = u[permidx] if arm == "permgate" else u
            lam = _sigmoid(w @ ug + c)
        h_prev = h
        h = lam * h_prev + (1.0 - lam) * inj
        if arm == "selective":                                 # exact forward-mode eligibility (gate params only)
            ug = u
            dlam = lam * (1.0 - lam)                            # d sigmoid
            base = (h_prev - inj)                               # d h / d lam
            ew = lam[:, None] * ew + (dlam * base)[:, None] * ug[None, :]
            ec = lam * ec + dlam * base
    return h, ew, ec


def _run_arm(seed, arm):
    E = _embed(seed); Win, w0, c0, fixed_lam = _params(seed)
    seqs, rule = _make_seqs(seed)
    permidx = np.random.default_rng(seed * 13 + 4).permutation(D_IN)
    w = w0.copy(); c = c0.copy()
    Wro = np.zeros((K, N_HID))
    ntr = int(0.7 * len(seqs))
    for _ep in range(EPOCHS):
        for (toks, y) in seqs[:ntr]:
            h, ew, ec = _forward(toks, E, Win, w, c, fixed_lam, arm, permidx)
            z = Wro @ h; z -= z.max(); p = np.exp(z); p /= p.sum()
            err = p.copy(); err[y] -= 1.0                       # p - onehot
            delta = Wro.T @ err                                # spatial backprop through the read-out (local)
            Wro -= LR_RO * np.outer(err, h)
            if arm == "selective":                             # eligibility-trained gate (forward-mode, transport-free)
                # gradient DESCENT on L: theta -= LR * (dL/dh) ⊙ (dh/dtheta) = LR * delta ⊙ e   (delta = dL/dh)
                w -= LR_GATE * (delta[:, None] * ew)
                c -= LR_GATE * (delta * ec)
    # eval
    cor = 0; tot = 0
    for (toks, y) in seqs[ntr:]:
        h, _, _ = _forward(toks, E, Win, w, c, fixed_lam, arm, permidx)
        z = Wro @ h
        cor += int(np.argmax(z) == y); tot += 1
    return cor / tot


def run(seed):
    acc = {a: _run_arm(seed, a) for a in ("selective", "fixed_res", "detached", "permgate")}
    chance = 1.0 / K
    go = bool(acc["selective"] > acc["fixed_res"] + 0.10 and acc["selective"] > acc["detached"] + 0.08
              and acc["selective"] > acc["permgate"] + 0.08 and acc["selective"] > chance + 0.15)
    print(f"[rung2 seed={seed}] selective={acc['selective']:.3f} fixed_res={acc['fixed_res']:.3f} "
          f"detached={acc['detached']:.3f} permgate={acc['permgate']:.3f} (chance={chance:.3f}) "
          f"-> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, **acc, "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    if a.depth is not None:
        global DEPTH
        DEPTH = a.depth
    res = [run(s) for s in a.seeds]
    print(f"[rung2] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
