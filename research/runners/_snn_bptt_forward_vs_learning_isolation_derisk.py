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
    if credit_mode in ("eprop", "eprop_shuffle"):
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
            elif credit_mode == "eprop_shuffle":
                # ANTI-CHEAT: scramble the learning signal across the batch (eligibility intact, credit MISMATCHED to
                # the example) -> must collapse to chance if the DFA credit channel is genuinely load-bearing.
                ogs = og[:, rng.permutation(len(bi)), :]
                wg = _eprop_grads(inp, layers, fs, ogs, B_direct, alpha_leak=layers[0].leak, alpha_surr=2.0)
            else:
                wg, _ = backward_unroll_xp(inp, layers, fs, og, alpha=2.0, xp=np)
            for li in range(len(layers)):
                layers[li].W_in -= lr * (wg[li] / len(bi))
    return layers


# ============================================================================================================
# FA-CONVERGENCE REFERENCE (`--measure-fa-convergence`, ADDITIVE, default OFF). The "FA converges" fingerprint to
# compare the Izhikevich substrate against (gap#4 finding 2026-08-02, Update 4). Same e-prop DFA rule (credit_mode
# =eprop, fixed-random B_direct) on the LIF net, where it TRAINS the depth-2 task (inherit ~0.895). Logs the SAME
# per-epoch reads as the on-bridge runner's --measure-fa-convergence: (1) cos(forward-chain(li), B_direct[li]^T)
# per hidden pathway [RISING => FA converges]; (2) cos(delta_k@B_direct[li], delta_k@chain(li)^T) [delivered DFA
# credit vs transport gradient]; (3) inherit-heldout per epoch. If the LIF cos RISES and the Izhikevich cos stays
# FLAT => FA-convergence fails on the point-neuron Izhikevich substrate specifically. NO sim/ edit.
def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 1e-12 else 0.0


def _fa_chain_lif(layers, li):
    """Downstream forward map from hidden pathway li's post layer to output = layers[li+1].W_in @ ... @ layers[-1].W_in.
    For the LAST hidden pathway this is exactly the H_last->out readout weight (the classic Lillicrap FA signature)."""
    M = layers[li + 1].W_in
    for p in range(li + 2, len(layers)):
        M = M @ layers[p].W_in
    return M                                       # (n_post_li, k)


def measure_fa_convergence_lif(seed, hidden, T, epochs, lr, in_gain, subsample, task_kwargs, n_hidden_layers,
                               fd_batch, fa_eval_every):
    """FA-convergence reference on the LIF net (credit_mode=eprop). Per-epoch cos(forward, B_direct^T) as the SAME
    e-prop DFA rule trains -- the 'FA converges' fingerprint. Mirrors _train_snn's eprop branch, adding per-epoch
    logging. NO change to _train_snn / run_seed (both stay byte-identical)."""
    t0 = time.time()
    task_full = make_task_semantic_inheritance(seed, **task_kwargs)
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = task_full
    k = meta["k_classes"]; n_in = Xtr.shape[1]; inh_idx = idx["inh_idx"]
    yv = yte[inh_idx]; chance = float(max(np.mean(yv == c) for c in np.unique(yv))) if len(inh_idx) else float("nan")
    if subsample is not None and len(Xtr) > subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:subsample]
        Xtr, ytr = Xtr[keep], ytr[keep]
    sizes = [n_in] + [hidden] * n_hidden_layers + [k]
    # ---- build layers + B_direct EXACTLY as _train_snn(credit_mode='eprop') does (faithful training dynamics) ----
    rng = np.random.default_rng(seed + 1)
    w_scales = [2.5] + [1.0] * (len(sizes) - 2)
    layers = _build_layers(sizes, T, rng, w_scales)
    n = len(Xtr)
    frng = np.random.default_rng(seed + 8888)
    B_direct = [frng.normal(0.0, 1.0 / np.sqrt(k), (k, sizes[li + 1])).astype(np.float64)
                for li in range(len(layers) - 1)]
    n_hp = len(B_direct)
    mb = min(int(fd_batch), n)
    mrng = np.random.default_rng(seed + 909)
    m_idx = mrng.permutation(n)[:mb]

    def _fa_cos_now():
        return [_cos(_fa_chain_lif(layers, li), B_direct[li].T) for li in range(n_hp)]

    def _credit_align_now():
        chains = [_fa_chain_lif(layers, li) for li in range(n_hp)]
        Xm, ym = Xtr[m_idx], ytr[m_idx]
        logits, _, _ = _forward_logits(Xm, layers, T, in_gain)
        p = _softmax(logits); dk = p.copy(); dk[np.arange(len(ym)), ym] -= 1.0   # (mb, k) softmax error
        cols = [[] for _ in range(n_hp)]
        for r in range(len(ym)):
            for li in range(n_hp):
                cols[li].append(_cos(dk[r] @ B_direct[li], dk[r] @ chains[li].T))
        return [float(np.nanmean(c)) if c else float("nan") for c in cols]

    def _rec(ep, heavy):
        rec = {"epoch": ep, "fa_cos": _fa_cos_now()}
        if heavy:
            rec["credit_align"] = _credit_align_now()
            rec["inherit"] = _accuracy(Xte, yte, layers, T, in_gain, sub=inh_idx)
        else:
            rec["credit_align"] = None; rec["inherit"] = None
        return rec

    traj = [_rec(0, True)]                          # init (pre-training) read
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        for b0 in range(0, n, 32):
            bi = perm[b0:b0 + 32]
            Xb, yb = Xtr[bi], ytr[bi]
            logits, fs, inp = _forward_logits(Xb, layers, T, in_gain)
            p = _softmax(logits); delta = p.copy(); delta[np.arange(len(yb)), yb] -= 1.0
            og = np.repeat((delta / T)[None, :, :], T, axis=0).astype(np.float64)
            wg = _eprop_grads(inp, layers, fs, og, B_direct, alpha_leak=layers[0].leak, alpha_surr=2.0)
            for li in range(len(layers)):
                layers[li].W_in -= lr * (wg[li] / len(bi))
        heavy = ((ep % max(1, fa_eval_every) == 0) or ep == epochs)
        traj.append(_rec(ep, heavy))

    fa_top = [rec["fa_cos"][-1] for rec in traj]           # LAST hidden pathway = the readout FA signature
    inh = [rec["inherit"] for rec in traj if rec["inherit"] is not None]
    init_c = fa_top[0]; final_c = fa_top[-1]
    peak_c = max(fa_top, key=lambda v: abs(v)); rise = final_c - init_c
    converges = bool(abs(peak_c) - abs(init_c) > 0.05 and final_c > init_c)
    read = (f"FA {'CONVERGES' if converges else 'does NOT converge (flat/near-0)'}: top-hidden cos(W,B^T) "
            f"init {init_c:+.3f} -> final {final_c:+.3f} (peak {peak_c:+.3f}, rise {rise:+.3f}); "
            f"inherit {(inh[0] if inh else float('nan')):.3f} -> {(inh[-1] if inh else float('nan')):.3f} "
            f"(chance {chance:.3f}).")
    return {"seed": seed, "substrate": "lif", "credit": "fixed_dfa", "k_classes": int(k), "chance": chance,
            "n_hidden_pathways": n_hp, "epochs": epochs, "fa_eval_every": fa_eval_every,
            "fa_cos_top_init": init_c, "fa_cos_top_final": final_c, "fa_cos_top_peak": peak_c,
            "fa_cos_top_rise": rise, "fa_converges": converges,
            "inherit_init": (inh[0] if inh else float("nan")), "inherit_final": (inh[-1] if inh else float("nan")),
            "trajectory": traj, "READ": read, "elapsed_seconds": round(time.time() - t0, 1)}


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
    ap.add_argument("--credit-mode", type=str, default="bptt", choices=["bptt", "spatial", "eprop", "eprop_shuffle"])
    ap.add_argument("--train-subsample", type=int, default=400)
    ap.add_argument("--n-super", type=int, default=12)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=2)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--measure-fa-convergence", action="store_true",
                    help="FA-CONVERGENCE reference (additive, default off): per-epoch cos(forward, B_direct^T) for the "
                         "eprop DFA rule on the LIF net (where it TRAINS, inherit ~0.895) -- the 'FA converges' "
                         "fingerprint. Forces credit_mode=eprop. Compare to the on-bridge runner's same flag.")
    ap.add_argument("--fd-batch", type=int, default=16,
                    help="examples for the item-(2) credit-align cross-check (only under --measure-fa-convergence).")
    ap.add_argument("--fa-eval-every", type=int, default=1,
                    help="epoch cadence for the HEAVY per-epoch reads (credit_align + inherit) under "
                         "--measure-fa-convergence; cos(W,B^T) is logged EVERY epoch regardless.")
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    task_kwargs = {"n_super": args.n_super, "n_members": args.n_members,
                   "held_per_super": args.held_per_super, "n_prop": args.n_prop,
                   "member_id_dim": args.member_id_dim, "n_obs": args.n_obs,
                   "noise": args.noise, "feature_seed": args.feature_seed}

    # ---- FA-CONVERGENCE REFERENCE BRANCH (additive; the 'FA converges' fingerprint, no train/floor/permuted arms) ----
    if args.measure_fa_convergence:
        t0 = time.time()
        try:
            r = measure_fa_convergence_lif(args.seed, args.hidden, args.timesteps, args.epochs, args.lr, args.in_gain,
                                           args.train_subsample, task_kwargs, args.n_hidden_layers,
                                           args.fd_batch, args.fa_eval_every)
        except Exception as e:
            r = {"seed": args.seed, "error": repr(e), "traceback": traceback.format_exc()}
        out = {"probe": "snn_bptt_fa_convergence_reference", "seed": args.seed,
               "config": {"hidden": args.hidden, "n_hidden_layers": args.n_hidden_layers, "T": args.timesteps,
                          "epochs": args.epochs, "lr": args.lr, "in_gain": args.in_gain,
                          "train_subsample": args.train_subsample, "credit_mode": "eprop", "task": task_kwargs,
                          "fd_batch": args.fd_batch, "fa_eval_every": args.fa_eval_every},
               "elapsed_seconds": round(time.time() - t0, 1), "result": r}
        out["verdict"] = r.get("READ", r.get("error", "no result"))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(out["verdict"])
        print(f"[fa-convergence-lif] wrote {args.out}")
        return

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
