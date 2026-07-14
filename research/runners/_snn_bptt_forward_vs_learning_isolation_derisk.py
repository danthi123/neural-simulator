"""FORWARD-vs-LEARNING isolation (the SURPASS residual-isolation for the deep-credit-on-spikes wall): does a
surrogate-gradient BPTT-trained SPIKING (LIF) net -- the BEST-POSSIBLE on-spike credit signal -- TRAIN the depth-2
compositional-inheritance task, where the OnBridgeBDSPNet's LOCAL burst-dependent-plasticity credit does NOT (0/6 via
every cheap credit-side lever: capacity, graded read-variance, DECOLLE per-layer-local, population-cleaning)?

WHY (2026-07-14). Four converging negatives bounded the on-spike deep-credit wall as NOT credit-side. The precise
residual is (harder compositional task) x (cheap spiking net scale): either the spiking FORWARD cannot represent the
depth-2 function at this scale (a representational wall), OR it can and the on-spike LOCAL weight-finding (BDSP) is the
wall. Surrogate-BPTT (Neftci-Mostafa-Zenke; the field's workhorse that trains deep multi-hidden-layer SNNs) is the
best-possible weight-finder for a spiking net -- it ISOLATES the two:
  * If a 2-hidden-layer LIF SNN trained by surrogate-BPTT TRAINS this task (inherit >> chance, generalizes) => a SPIKING
    net CAN represent + learn the depth-2 compositional function. The forward is NOT the wall; the OnBridgeBDSPNet's
    failure is the LOCAL BDSP credit at cheap scale (=> the lever is a scaled/richer local on-spike credit, not the
    substrate's representational capacity). The emergence engine's learning substrate is viable; the local rule needs work.
  * If surrogate-BPTT ALSO fails => the spiking forward genuinely cannot represent the depth-2 task at this net scale
    (a representational/scale wall) => scale is the lever.

THE NET: input(features, rate-coded over T steps) -> LIF H1 -> LIF H2 -> LIF out(k). Trained by surrogate-BPTT
(reuse-by-import sim/bptt_snn_gpu). Loss = cross-entropy on the summed output spikes. ANTI-CHEATS: permuted-label ->
~chance (no leakage); 1-hidden-layer floor -> ~chance (the task needs depth). Reports vs chance + the rate oracle (1.0).
NO sim/ edit."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    make_task_semantic_inheritance, stage0_depth_genuineness, _train_oracle, _acc_on)
from sim.dendritic_mlp import DendriticMLP  # noqa: E402
from sim.bptt_snn_gpu import LIFLayerXP, forward_unroll_xp, backward_unroll_xp, atan_surrogate  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_snn_bptt_isolation.json"


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _build_layers(sizes, T, rng, w_scales):
    """LIF layers input->H1->...->out. W_in shape (n_pre, n_post). First layer needs a stronger init to fire from the
    continuous rate-coded input (like the char-SNN's std=2.0 for sparse drive)."""
    layers = []
    for i in range(len(sizes) - 1):
        n_pre, n_post = sizes[i], sizes[i + 1]
        W = rng.normal(0.0, w_scales[i] / np.sqrt(n_pre), (n_pre, n_post)).astype(np.float64)
        layers.append(LIFLayerXP(W_in=W, n_post=n_post, threshold=1.0, leak=0.9))
    return layers


def _forward_logits(X, layers, T, in_gain):
    """Rate-code each feature row as a CONSTANT input current over T steps -> summed output spikes = class logits."""
    B = X.shape[0]
    inp = np.repeat((in_gain * X)[None, :, :], T, axis=0).astype(np.float64)   # (T, B, n_in)
    fs = forward_unroll_xp(inp, layers, xp=np)
    out_spikes = fs["spikes"][-1]                                             # (T, B, k)
    logits = out_spikes.sum(axis=0)                                          # (B, k)
    return logits, fs, inp


def _accuracy(X, y, layers, T, in_gain, sub=None):
    if sub is not None:
        X = X[sub]; y = y[sub]
    if len(X) == 0:
        return float("nan")
    logits, _, _ = _forward_logits(X, layers, T, in_gain)
    return float(np.mean(np.argmax(logits, axis=1) == y))


def _spatial_backward(inputs, layers, fs, output_grad, alpha=2.0):
    """SPATIAL-ONLY surrogate gradient: the per-timestep membrane surrogate derivative BUT NO through-time recurrence
    (BPTT's recurrent_dv/recurrent_ds zeroed). Isolates whether a LOCAL one-step surrogate rule suffices vs needing
    BPTT's temporal credit-through-time. Runner-side (NO sim/ edit)."""
    T, B, V_in = inputs.shape
    L = len(layers)
    spikes = fs["spikes"]; v_per = fs["v"]
    weight_grads = [np.zeros_like(l.W_in) for l in layers]
    dv_grads = [np.zeros((T, B, l.n_post), dtype=l.W_in.dtype) for l in layers]
    for li in range(L - 1, -1, -1):
        layer = layers[li]; v_layer = v_per[li]
        if li == L - 1:
            ds_grad = output_grad
        else:
            nxt = layers[li + 1]
            ds_grad = np.zeros((T, B, layer.n_post), dtype=layer.W_in.dtype)
            for t in range(T):
                ds_grad[t] = dv_grads[li + 1][t] @ nxt.W_in.T
        for t in range(T):
            surrogate_t = atan_surrogate(v_layer[t] - layer.threshold, alpha=alpha, xp=np)
            dv_grads[li][t] = ds_grad[t] * surrogate_t          # spatial only: NO recurrent_dv / recurrent_ds
        x_pre = inputs if li == 0 else spikes[li - 1]
        for t in range(T):
            weight_grads[li] += x_pre[t].T @ dv_grads[li][t]
    return weight_grads


def _eprop_grads(inputs, layers, fs, output_grad, B_direct, alpha_leak, alpha_surr=2.0):
    """e-prop (Bellec 2020) with DIRECT feedback alignment (Nokland) -- a LOCAL, FORWARD-mode, TRANSPORT-FREE rule (no
    BPTT, no W^T). Per weight: dw_ji = sum_t L_j(t) * psi_j(t) * eps_i(t), where eps_i(t)=alpha*eps_i(t-1)+z_pre_i(t) is
    the FORWARD eligibility (captures the leak-recurrence exactly for a diagonal/feedforward net), psi_j(t) is the
    membrane surrogate, and L_j(t) is the learning signal = the OUTPUT error projected to layer j by a FIXED-RANDOM
    B_direct (DFA; output layer uses the error directly). NO sim/ edit; NO weight transport (B_direct is a separate
    fixed-random stream)."""
    T, Bn, _ = inputs.shape
    L = len(layers)
    spikes = fs["spikes"]; v_per = fs["v"]
    weight_grads = [np.zeros_like(l.W_in) for l in layers]
    eps = [np.zeros((Bn, l.W_in.shape[0]), dtype=l.W_in.dtype) for l in layers]   # (B, n_pre) running eligibility
    for t in range(T):
        for li in range(L):
            pre = inputs[t] if li == 0 else spikes[li - 1][t]                     # (B, n_pre)
            eps[li] = alpha_leak * eps[li] + pre                                  # forward eligibility trace
            psi = atan_surrogate(v_per[li][t] - layers[li].threshold, alpha=alpha_surr, xp=np)  # (B, n_post)
            if li == L - 1:
                Lsig = output_grad[t]                                             # output error directly (B, k)
            else:
                Lsig = output_grad[t] @ B_direct[li]                             # DFA: fixed-random projection (B, n_post)
            g = Lsig * psi                                                        # (B, n_post)
            weight_grads[li] += eps[li].T @ g                                     # (n_pre, n_post)
    return weight_grads


def _train_snn(Xtr, ytr, sizes, T, epochs, lr, in_gain, seed, batch=32, credit_mode="bptt"):
    rng = np.random.default_rng(seed + 1)
    w_scales = [2.5] + [1.0] * (len(sizes) - 2)          # strong first layer, moderate deeper
    layers = _build_layers(sizes, T, rng, w_scales)
    k = sizes[-1]
    n = len(Xtr)
    # e-prop DFA feedback: fixed-random (k, n_post_li) per HIDDEN layer, SEPARATE seed stream => no weight transport
    B_direct = None
    if credit_mode == "eprop":
        frng = np.random.default_rng(seed + 8888)
        B_direct = [frng.normal(0.0, 1.0 / np.sqrt(k), (k, sizes[li + 1])).astype(np.float64)
                    for li in range(len(layers) - 1)]     # only hidden layers use DFA; output uses the error directly
    for ep in range(epochs):
        perm = rng.permutation(n)
        for b0 in range(0, n, batch):
            bi = perm[b0:b0 + batch]
            Xb, yb = Xtr[bi], ytr[bi]
            logits, fs, inp = _forward_logits(Xb, layers, T, in_gain)
            p = _softmax(logits)
            delta = p.copy(); delta[np.arange(len(yb)), yb] -= 1.0          # (B, k) dL/d(sum spikes)
            # distribute the summed-spike gradient equally over the T timesteps
            og = np.repeat((delta / T)[None, :, :], T, axis=0).astype(np.float64)  # (T, B, k)
            if credit_mode == "spatial":
                wg = _spatial_backward(inp, layers, fs, og, alpha=2.0)
            elif credit_mode == "eprop":
                wg = _eprop_grads(inp, layers, fs, og, B_direct, alpha_leak=layers[0].leak, alpha_surr=2.0)
            else:
                wg, _ = backward_unroll_xp(inp, layers, fs, og, alpha=2.0, xp=np)
            for li in range(len(layers)):
                layers[li].W_in -= lr * (wg[li] / len(bi))
    return layers


def run_seed(seed, hidden, T, epochs, lr, in_gain, subsample, task_kwargs, n_hidden_layers=2,
             credit_mode="bptt"):
    task_full = make_task_semantic_inheritance(seed, **task_kwargs)
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = task_full
    k = meta["k_classes"]; n_in = Xtr.shape[1]
    inh_idx = idx["inh_idx"]
    s0 = stage0_depth_genuineness(((Xtr, ytr, Ltr), (Xte, yte, Lte)), idx, k, hidden=96, epochs=250,
                                  lr=0.3, batch=128, seed=seed)
    if len(inh_idx):
        yv = yte[inh_idx]; chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    else:
        chance = float("nan")
    onet = DendriticMLP([n_in] + [96] * n_hidden_layers + [k], seed=seed)
    _train_oracle(onet, Xtr, ytr, 250, 0.3, 128, seed)
    oracle_inh = _acc_on(onet, Xte, yte, inh_idx)

    if subsample is not None and len(Xtr) > subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:subsample]
        Xtr, ytr = Xtr[keep], ytr[keep]

    sizes = [n_in] + [hidden] * n_hidden_layers + [k]
    layers = _train_snn(Xtr, ytr, sizes, T, epochs, lr, in_gain, seed, credit_mode=credit_mode)
    train_acc = _accuracy(Xtr, ytr, layers, T, in_gain)
    inh_acc = _accuracy(Xte, yte, layers, T, in_gain, sub=inh_idx)

    # anti-cheat 1: 1-hidden-layer floor (task needs depth -> should be ~chance)
    fl_sizes = [n_in] + [hidden] + [k]
    fl_layers = _train_snn(Xtr, ytr, fl_sizes, T, epochs, lr, in_gain, seed, credit_mode=credit_mode)
    floor_inh = _accuracy(Xte, yte, fl_layers, T, in_gain, sub=inh_idx)

    # anti-cheat 2: permuted-label (no leakage -> ~chance)
    prng = np.random.default_rng(seed + 555)
    yperm = ytr[prng.permutation(len(ytr))]
    pm_layers = _train_snn(Xtr, yperm, sizes, T, epochs, lr, in_gain, seed, credit_mode=credit_mode)
    perm_inh = _accuracy(Xte, yte, pm_layers, T, in_gain, sub=inh_idx)

    trains = bool((not np.isnan(inh_acc)) and inh_acc > floor_inh + 0.03 and inh_acc > chance + 0.03)
    return {"seed": seed, "chance": chance, "stage0_depth_separating": bool(s0.get("depth_separating")),
            "oracle_inherit": oracle_inh, "snn_train_acc": train_acc, "snn_inherit_heldout": inh_acc,
            "floor_inherit_heldout": floor_inh, "permuted_inherit": perm_inh, "trains_at_all": trains}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-hidden-layers", type=int, default=2)
    ap.add_argument("--timesteps", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--in-gain", type=float, default=1.0)
    ap.add_argument("--credit-mode", type=str, default="bptt", choices=["bptt", "spatial", "eprop"])
    ap.add_argument("--train-subsample", type=int, default=400)
    ap.add_argument("--n-super", type=int, default=12)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=2)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    task_kwargs = {"n_super": args.n_super, "n_members": args.n_members,
                   "held_per_super": args.held_per_super, "n_prop": args.n_prop,
                   "member_id_dim": args.member_id_dim, "n_obs": args.n_obs,
                   "noise": args.noise, "feature_seed": args.feature_seed}
    t0 = time.time()
    try:
        r = run_seed(args.seed, args.hidden, args.timesteps, args.epochs, args.lr, args.in_gain,
                     args.train_subsample, task_kwargs, n_hidden_layers=args.n_hidden_layers,
                     credit_mode=args.credit_mode)
    except Exception as e:
        r = {"seed": args.seed, "error": repr(e), "traceback": traceback.format_exc()}

    out = {"probe": "snn_bptt_forward_vs_learning_isolation", "seed": args.seed,
           "config": {"hidden": args.hidden, "n_hidden_layers": args.n_hidden_layers, "T": args.timesteps,
                      "epochs": args.epochs, "lr": args.lr, "in_gain": args.in_gain,
                      "train_subsample": args.train_subsample, "credit_mode": args.credit_mode, "task": task_kwargs},
           "elapsed_seconds": round(time.time() - t0, 1), "result": r}
    if "snn_inherit_heldout" in r:
        ch = r["chance"]
        out["verdict"] = (
            f"surrogate-BPTT SNN {'TRAINS' if r['trains_at_all'] else 'fails'} the depth-2 task "
            f"(train {r['snn_train_acc']:.3f}, inherit {r['snn_inherit_heldout']:.3f}; chance {ch:.3f}, "
            f"floor {r['floor_inherit_heldout']:.3f}, permuted {r['permuted_inherit']:.3f}, oracle {r['oracle_inherit']:.3f}). "
            + ("=> the SPIKING FORWARD CAN represent+learn it with a good credit signal -> the OnBridgeBDSPNet failure is the LOCAL BDSP credit at cheap scale, NOT the forward (lever: scaled/richer local on-spike credit)."
               if r["trains_at_all"] else
               "=> even the best-possible on-spike credit (surrogate-BPTT) fails -> a representational/scale wall of the spiking forward at this net size (lever: genuine scale)."))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(out.get("verdict", json.dumps(r)[:300]))


if __name__ == "__main__":
    main()
