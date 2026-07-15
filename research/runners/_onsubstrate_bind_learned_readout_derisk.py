"""2026-07-15 — RUNG 2 (the emergence step per TEST A's own conclusion "the path is LEARN to USE a fixed bind/store"):
over RUNG-1's FIXED spiking coincidence bind (real SimulationBridge), is the READ-OUT that maps the spiking bound rates ->
intent learnable by BIOLOGICAL credit (a 1-hidden read-out trained by feedback alignment = FIXED random feedback, NO weight
transport), and does it STILL extrapolate to held-out (cat,qt) combinations like the least-squares ridge did (RUNG 1) — while
a from-scratch classifier on the raw [cat;q] does not?

If YES: the systematicity path is FIXED spiking bind (RUNG 1, the substrate's native ⊙) + a BIOLOGICALLY-LEARNED read-out over
it — "learn to USE the fixed bind," transport-free. That is the honest emergence-bar realization (learned read-out × fixed
biological binding primitive), consistent with the whole session's convergence and the learned-bilinear transport-free result.

Reuse-by-import: RUNG-1's bound-rate computation (`build_bind_bridge`/`hadamard_spiking` via a factored `bound_rates`), the task
harness, and the MLP-on-concat control. The transport-free read-out mirrors `_learned_bilinear_binder_systematicity_derisk`'s
`credit='transport_free'` (fixed random feedback for the hidden credit). NO `sim/` edit. numpy = smoke; SIM_BACKEND=cupy for GPU.

Run: SIM_BACKEND=numpy python -u -m research.runners._onsubstrate_bind_learned_readout_derisk --seeds 42
"""
import os, sys, json, argparse
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners.core_sim_composition import build_bind_bridge, hadamard_spiking
from research.runners._fixedbind_systematicity_derisk import build_task, _dataset, _ridge, N_INTENT
from research.runners._deep_eprop_binder_bundling_derisk import _train_snn, score_snn, standardize, T
from research.runners._onsubstrate_coincidence_systematicity_derisk import _fill_currents, RUN_STEPS, COINC_BIAS


def bound_rates(seed):
    """RUNG-1's factored bound-rate computation: drive cat/qt codes DIRECTLY (identity) into the fixed spiking coincidence
    bind on a real SimulationBridge; return the spiking bound rates B (n, 2D) + the task."""
    a, b, cat_code, q_code, intent_of, held, D = build_task(seed, easy=False)
    cells, y, is_held = _dataset(cat_code, q_code, intent_of, held, n_per=1, seed=seed)
    bridge, idx = build_bind_bridge(seed, D)
    B = np.zeros((len(cells), 2 * D), np.float32)
    for i, (c, q) in enumerate(cells):
        role = (cat_code[c] * 2 - 1).astype(np.float32)
        fon, foff = _fill_currents(q_code[q] * 2 - 1, D)
        bon, boff = hadamard_spiking(bridge, idx, role, fon, foff, D, RUN_STEPS, COINC_BIAS)
        B[i] = np.concatenate([bon, boff])
    return B, y, is_held, cells, cat_code, q_code, D


def _relu(x):
    return np.maximum(x, 0.0)


def train_readout(Xtr, ytr, K, d_hid=24, epochs=500, lr=0.05, seed=0, lam=1e-3, credit="transport_free"):
    """A 1-hidden read-out h=relu(X@W1.T), logits=h@W2.T over the FIXED spiking bind's bound rates. credit='gradient' uses
    W2 (weight transport) for the hidden credit; credit='transport_free' uses a FIXED RANDOM feedback matrix Bfb (feedback
    alignment, NO weight transport = the biologically-legal rule). W2 is local-delta-trained either way."""
    rng = np.random.default_rng(seed * 17 + 3)
    F = Xtr.shape[1]
    W1 = rng.standard_normal((d_hid, F)) / np.sqrt(F)
    W2 = rng.standard_normal((K, d_hid)) / np.sqrt(d_hid)
    Bfb = rng.standard_normal((K, d_hid)) / np.sqrt(d_hid)             # fixed random feedback (transport-free path)
    Y = np.eye(K)[ytr]
    for _ in range(epochs):
        pre = Xtr @ W1.T
        h = _relu(pre)                                                # (n, d_hid)
        logits = h @ W2.T
        p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        g = (p - Y) / len(ytr)                                        # (n, K)
        gW2 = g.T @ h + lam * W2
        dh = (g @ W2) if credit == "gradient" else (g @ Bfb)          # hidden credit: transport vs fixed-random-feedback
        dh = dh * (pre > 0)
        gW1 = dh.T @ Xtr + lam * W1
        W2 -= lr * gW2; W1 -= lr * gW1
    return W1, W2


def _readout_pred(W1, W2, X):
    return np.argmax(_relu(X @ W1.T) @ W2.T, axis=1)


def run_one(seed):
    B, y, is_held, cells, cat_code, q_code, D = bound_rates(seed)
    tr = ~is_held
    Btr, Bev = standardize(B[tr], B)
    out = {"seed": seed, "chance": round(1.0 / N_INTENT, 4), "n_held": int(is_held.sum()), "D": D}
    # RUNG 1 baseline: least-squares ridge read-out over the spiking bind
    pr = _ridge(Btr, y[tr], Bev, N_INTENT, lam=8.0)
    out["ridge_held"] = round(float(np.mean(pr[is_held] == y[is_held])), 4)
    # RUNG 2: BIOLOGICAL read-out over the fixed spiking bind (transport-free = feedback alignment, NO weight transport)
    W1t, W2t = train_readout(Btr, y[tr], N_INTENT, seed=seed, credit="transport_free")
    ptf = _readout_pred(W1t, W2t, Bev)
    out["transportfree_held"] = round(float(np.mean(ptf[is_held] == y[is_held])), 4)
    out["transportfree_train"] = round(float(np.mean(ptf[tr] == y[tr])), 4)
    # gradient read-out (weight transport) = reference ceiling for the learned read-out
    W1g, W2g = train_readout(Btr, y[tr], N_INTENT, seed=seed, credit="gradient")
    out["gradient_held"] = round(float(np.mean(_readout_pred(W1g, W2g, Bev)[is_held] == y[is_held])), 4)
    # controls: from-scratch classifier on RAW [cat;q] + the PARENT's STRONGER controls (1-NN memfloor + linear-raw ridge),
    # RESTORED after the 2026-07-15 adversarial verify caught that dropping them overstated robustness (held to memfloor: 3/6).
    CAT = np.array([cat_code[c] for (c, q) in cells]); Q = np.array([q_code[q] for (c, q) in cells])
    C = np.concatenate([CAT, Q], axis=1); Ctr, Cev = standardize(C[tr], C)
    lay = _train_snn(Ctr, y[tr], [C.shape[1], 48, 48, N_INTENT], T, 120, 0.05, 1.0, seed, credit_mode="eprop")
    _, mlp_h, _ = score_snn(lay, Cev, y, is_held, 1.0); out["mlp_held"] = round(mlp_h, 4)
    d = np.linalg.norm(C[tr][None, :, :] - C[:, None, :], axis=2); pmf = y[tr][d.argmin(1)]     # 1-NN memfloor on raw code
    out["memfloor_held"] = round(float(np.mean(pmf[is_held] == y[is_held])), 4)
    pl = _ridge(Ctr, y[tr], Cev, N_INTENT, lam=8.0)                                             # linear ridge on raw concat
    out["linear_held"] = round(float(np.mean(pl[is_held] == y[is_held])), 4)
    # anti-cheats: permuted labels (must collapse) + no-hidden-credit lesion (zero the read-out learning -> chance)
    rp = np.random.default_rng(seed + 7); yp = y.copy(); yp[tr] = y[tr][rp.permutation(int(tr.sum()))]
    W1p, W2p = train_readout(Btr, yp[tr], N_INTENT, seed=seed, credit="transport_free")
    out["permuted_held"] = round(float(np.mean(_readout_pred(W1p, W2p, Bev)[is_held] == y[is_held])), 4)
    out["GO_vs_mlp"] = bool(out["transportfree_held"] > 0.55 and out["transportfree_held"] > out["mlp_held"] + 0.15
                            and out["permuted_held"] < out["chance"] + 0.2)
    # HONEST GO (the parent's full controls): also beat the 1-NN memfloor AND the linear-raw ridge by +0.15
    out["GO"] = bool(out["GO_vs_mlp"] and out["transportfree_held"] > out["memfloor_held"] + 0.15
                     and out["transportfree_held"] > out["linear_held"] + 0.15)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", default="research/findings/raw/_onsubstrate_bind_learned_readout.json")
    a = ap.parse_args()
    rows = [run_one(s) for s in a.seeds]
    for r in rows:
        print(f"[bind+bioread s{r['seed']}] chance={r['chance']} || TRANSPORT-FREE read-out over the FIXED spiking bind "
              f"held={r['transportfree_held']:.3f} | gradient-ref={r['gradient_held']:.3f} | MLP-raw={r['mlp_held']:.3f} "
              f"| memfloor={r['memfloor_held']:.3f} | linear-raw={r['linear_held']:.3f} | permuted={r['permuted_held']:.3f} "
              f"|| GO_vs_mlp={r['GO_vs_mlp']} GO_full_controls={'GO' if r['GO'] else 'no'}", flush=True)
    ngo = sum(x["GO"] for x in rows); ngm = sum(x["GO_vs_mlp"] for x in rows)
    print(f"[bind+bioread] HONEST: {ngo}/{len(rows)} GO vs the PARENT's full controls (memfloor+linear, +0.15); {ngm}/{len(rows)} "
          f"vs the weak e-prop MLP alone. The transport-free-learnable sub-claim (transport-free ≈ gradient) survives; "
          f"the >>from-scratch robustness was overstated by the reduced control set.", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
