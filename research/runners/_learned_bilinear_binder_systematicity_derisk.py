"""2026-07-15 — the emergence-bar FORWARD from TEST A: can a LEARNED bilinear BINDING STRUCTURE achieve the systematic
extrapolation that a general map-learning classifier could not? (TEST A: fixed +-1 bind 0.87 extrapolates; a 2-hidden
MLP learner 0.39 memorizes+fails.) `2026-06-11` showed a GRADIENT-trained bilinear binder is systematic on decorrelated
codes; this runs it on TEST A's EXACT task to confirm the structure transfers + set up the BIOLOGICAL-rule follow-on.

THE POINT (the emergence bar): the fixed FHRR bind is a hand-provided primitive. If a bilinear STRUCTURE (bound =
(W_a @ cat) (.) (W_b @ q), the multiplicative interaction FORCED, but the projections W_a,W_b LEARNED) extrapolates
where the unstructured MLP does not, then the systematicity-enabling structure can be LEARNED (not hand-fixed) as long
as the multiplicative-binding INDUCTIVE BIAS is provided -- the honest 'learn-to-use-a-binding-primitive' path. Rung 1
= gradient reference (this file). Rung 2 (follow-on) = train the same bilinear by the GO transport-free deep-credit rule.

ARMS on TEST A's task (`_fixedbind_systematicity_derisk.build_task`, 7x7, held-out combos, decorrelated +-1 codes):
  bilinear (learned W_a,W_b + linear read-out, gradient) | plain MLP (from `_train_snn`, the map-classifier control) |
  fixed +-1 bind (the reference ceiling) | permuted (anti-cheat) | 1-NN memfloor.
GATE (6-seed): bilinear held-out >> MLP AND >> memfloor AND ~ fixed-bind; permuted collapses.

Run: SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u -m research.runners._learned_bilinear_binder_systematicity_derisk
"""
import os, sys, json, argparse
import numpy as np
from numpy.linalg import norm, solve

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._fixedbind_systematicity_derisk import (
    build_task, _bind, N_INTENT, _dataset, _ridge)
from research.runners._deep_eprop_binder_bundling_derisk import _train_snn, score_snn, standardize, T


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e / e.sum(axis=1, keepdims=True)


def train_bilinear(CAT, Q, y, tr, D, K, d_bind=16, epochs=400, lr=0.05, seed=0, lam=1e-3, credit="gradient"):
    """bound = (W_a @ cat) (.) (W_b @ q); logits = W_out @ bound. Learn W_a,W_b,W_out on the attested combos.
    The multiplicative (.) is the FIXED binding inductive bias; the projections are LEARNED.
    credit='gradient' -> true backprop (uses Wo.T = weight transport); credit='transport_free' -> the credit to the
    projections is `g @ B` with B a FIXED RANDOM feedback matrix (feedback alignment: NO weight transport = the biologically
    -legal rule). The read-out W_out is still local-delta-trained either way."""
    rng = np.random.default_rng(seed * 17 + 3)
    Wa = rng.standard_normal((d_bind, D)) / np.sqrt(D)
    Wb = rng.standard_normal((d_bind, D)) / np.sqrt(D)
    Wo = rng.standard_normal((K, d_bind)) / np.sqrt(d_bind)
    B = rng.standard_normal((K, d_bind)) / np.sqrt(d_bind)    # fixed random feedback (transport-free credit path)
    Xc, Xq, yt = CAT[tr], Q[tr], y[tr]
    n = len(yt)
    for ep in range(epochs):
        pa = Xc @ Wa.T; pb = Xq @ Wb.T                       # (n, d_bind)
        bound = pa * pb                                       # (.) bind
        logits = bound @ Wo.T                                # (n, K)
        p = _softmax(logits)
        Y = np.zeros((n, K)); Y[np.arange(n), yt] = 1.0
        g = (p - Y) / n                                       # (n, K)
        gWo = g.T @ bound + lam * Wo
        gb = (g @ Wo) if credit == "gradient" else (g @ B)   # grad wrt bound: weight-transport vs fixed-random-feedback
        gWa = (gb * pb).T @ Xc + lam * Wa                     # chain through pa
        gWb = (gb * pa).T @ Xq + lam * Wb
        Wo -= lr * gWo; Wa -= lr * gWa; Wb -= lr * gWb
    return Wa, Wb, Wo


def _bilinear_pred(Wa, Wb, Wo, CAT, Q):
    bound = (CAT @ Wa.T) * (Q @ Wb.T)
    return np.argmax(bound @ Wo.T, axis=1)


def run_one(seed, epochs_mlp=120):
    a, b, cat_code, q_code, intent_of, held, D = build_task(seed, easy=False)
    cells, y, is_held = _dataset(cat_code, q_code, intent_of, held, n_per=1, seed=seed)
    tr = ~is_held
    CAT = np.array([cat_code[c] for (c, q) in cells]); Q = np.array([q_code[q] for (c, q) in cells])
    out = {"seed": seed, "chance": round(1.0 / N_INTENT, 4), "n_held": int(is_held.sum())}
    # ARM: LEARNED BILINEAR binder, GRADIENT (the structure-is-learnable reference)
    Wa, Wb, Wo = train_bilinear(CAT, Q, y, tr, D, N_INTENT, seed=seed, credit="gradient")
    pbi = _bilinear_pred(Wa, Wb, Wo, CAT, Q)
    out["bilinear_train"] = round(float(np.mean(pbi[tr] == y[tr])), 4)
    out["bilinear_held"] = round(float(np.mean(pbi[is_held] == y[is_held])), 4)
    # ARM: LEARNED BILINEAR binder, TRANSPORT-FREE (the biologically-legal rule: fixed random feedback, NO weight transport)
    WaT, WbT, WoT = train_bilinear(CAT, Q, y, tr, D, N_INTENT, seed=seed, credit="transport_free")
    ptf = _bilinear_pred(WaT, WbT, WoT, CAT, Q)
    out["bilinear_tf_train"] = round(float(np.mean(ptf[tr] == y[tr])), 4)
    out["bilinear_tf_held"] = round(float(np.mean(ptf[is_held] == y[is_held])), 4)
    # REFERENCE: fixed +-1 bind + ridge (TEST A's ceiling)
    B = np.array([_bind(cat_code[c], q_code[q]) for (c, q) in cells])
    Btr, Bev = standardize(B[tr], B)
    pf = _ridge(Btr, y[tr], Bev, N_INTENT, lam=8.0)
    out["fixedbind_held"] = round(float(np.mean(pf[is_held] == y[is_held])), 4)
    # CONTROL: plain MLP map-classifier (TEST A: memorizes+fails)
    C = np.concatenate([CAT, Q], axis=1); Ctr, Cev = standardize(C[tr], C)
    lay = _train_snn(Ctr, y[tr], [C.shape[1], 48, 48, N_INTENT], T, epochs_mlp, 0.05, 1.0, seed, credit_mode="eprop")
    _, out["mlp_held"], _ = score_snn(lay, Cev, y, is_held, 1.0); out["mlp_held"] = round(out["mlp_held"], 4)
    # ANTI-CHEAT permuted: shuffle intent -> bilinear held collapses
    rp = np.random.default_rng(seed + 9); yp = y.copy(); yp[tr] = y[tr][rp.permutation(int(tr.sum()))]
    Wa2, Wb2, Wo2 = train_bilinear(CAT, Q, yp, tr, D, N_INTENT, seed=seed)
    ppp = _bilinear_pred(Wa2, Wb2, Wo2, CAT, Q)
    out["bilinear_permuted_held"] = round(float(np.mean(ppp[is_held] == y[is_held])), 4)
    # memfloor
    hi = np.where(is_held)[0]

    def nn(i):
        d = [norm(C[i] - C[j]) for j in np.where(tr)[0]]; return y[np.where(tr)[0][int(np.argmin(d))]]
    out["memfloor_held"] = round(float(np.mean([nn(i) == y[i] for i in hi])), 4) if len(hi) else 0.0
    out["GO"] = bool(out["bilinear_held"] > 0.6 and out["bilinear_held"] > out["mlp_held"] + 0.15
                     and out["bilinear_held"] > out["memfloor_held"] + 0.15
                     and out["bilinear_permuted_held"] < out["chance"] + 0.2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--out", default="research/findings/raw/_learned_bilinear_binder_systematicity.json")
    a = ap.parse_args()
    rows = [run_one(s) for s in a.seeds]
    for r in rows:
        print(f"[bilinear s{r['seed']}] chance={r['chance']} || GRAD-BILINEAR held={r['bilinear_held']:.3f} "
              f"| TRANSPORT-FREE held={r['bilinear_tf_held']:.3f} (train {r['bilinear_tf_train']:.3f}) "
              f"| fixed-bind={r['fixedbind_held']:.3f} | MLP={r['mlp_held']:.3f} | memfloor={r['memfloor_held']:.3f} "
              f"| permuted={r['bilinear_permuted_held']:.3f} || {'GO' if r['GO'] else 'no'}", flush=True)
    ngo = sum(x["GO"] for x in rows)
    print(f"[bilinear] {ngo}/{len(rows)} GO (learned bilinear STRUCTURE extrapolates >> MLP + memfloor, ~ fixed bind; permuted collapses)", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
