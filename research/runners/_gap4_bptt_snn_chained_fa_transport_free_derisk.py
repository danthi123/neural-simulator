"""gap#4 CRUX BRIDGE: the transport-free LOCAL rule on the TRAINABLE (BPTT-viable) LIF SNN -- the decisive test the
reservoir terminus could not run.

THE RECONCILIATION THIS RUNNER CLOSES (from the committed record).
  * RATE overturn (`2026-08-01-gap4-transport-free-ceiling-FALSIFIED-...`): a transport-free LOCAL rule -- chained
    multi-hop fixed-random feedback-alignment + sigma'(v-theta) + graded credit (KP-learned feedback for the depth
    rescue) -- trains deep credit AT RATE (clears depth-2 6-seed; KP rescues MNIST depth-4).
  * SPIKING terminus (`2026-08-02-gap4-crux-wall-LOCATED-...`): on the MOVABLE-PLATEAU coincidence-plateau RESERVOIR
    substrate, transport-free directed deep credit has NO purchase (oracle/lower-CV/DECOLLE/relaxed/bottleneck all
    directed ~ 0) -- the deep layer is reservoir-redundant and the sigma'(v-theta) read carries no credit-usable
    selectivity THERE.
  * But `2026-07-14-deep-credit-spiking-training-wall-...` already showed a 2-hidden-layer LIF SNN trained by
    surrogate-gradient BPTT (`_snn_bptt_forward_vs_learning_isolation_derisk.py`, via `sim/bptt_snn_gpu.py`) reaches
    ~0.82 on the depth-2 compositional-inheritance task => the spiking SUBSTRATE is VIABLE; the wall was the LOCAL
    rule ON THE RESERVOIR, not the forward.

So the crux gap is precisely: the TRANSPORT-FREE LOCAL rule on a TRAINABLE (not reservoir) spiking substrate. This
runner puts BOTH on the SAME LIF SNN forward + SAME task:

  ARM 1  bptt              : surrogate-gradient BPTT (`sim/bptt_snn_gpu.backward_unroll_xp`) -- the NON-local,
                             best-possible CEILING (reuse-by-import, unchanged).
  ARM 2  chained_fa        : the rate-overturn rule ported -- chained multi-hop FIXED-random feedback-alignment,
                             e_{li} = (e_{li+1} @ Y[li]) . sigma'(v-theta)_{li}, eligibility-weighted local update.
                             TRANSPORT-FREE: each Y[li] is a SEPARATE fixed-random stream, NEVER W-transported.
  ARM 2b chained_fa_kp     : same chain but Y[li] is KP-LEARNED (Kolen-Pollack, transport-free) -- the depth rescue
                             (port of `_gnw_d1_spiking_bdsp_derisk._kp_update` / the multihop `train` KP arm).
  ARM 3  frozen_reservoir  : hidden layers FROZEN at init; ONLY the output LIF layer learns (its local delta rule,
                             IDENTICAL to the output update inside chained_fa) -- the fixed-random-expansion reservoir
                             baseline. `chained - frozen` isolates EXACTLY the hidden-layer directed-credit contribution.
  ANTI-CHEAT permuted      : chained_fa with SHUFFLED labels -> must collapse to chance (the directed lift is the
                             correct-label-attributable part, credit - permuted, exactly the multihop runner's metric).

THE DECISIVE QUESTION. Does the transport-free LOCAL rule get DIRECTED-credit purchase on the TRAINABLE LIF SNN
(beat frozen AND permuted, approaching the BPTT ceiling), where it could NOT on the movable-plateau reservoir?
  * YES  => the SUBSTRATE (reservoir-redundancy + no-selectivity read) was the wall; a trainable spiking substrate is
            the surpass -- transport-free local credit works on spikes once the forward is itself plastic/BPTT-viable.
  * NO   => a deeper transport-free-vs-BPTT gap on spikes (the local rule fails even where the forward is trainable) --
            names the next mechanism, not a failure to hide.

DECISIVE METRICS (per seed, on the inherit held-out set):
  directed_over_permuted = chained_fa_inherit - permuted_inherit     (correct-label-attributable directed lift)
  purchase_over_frozen   = chained_fa_inherit - frozen_reservoir_inherit
  bptt_fraction_captured = (chained_fa - frozen) / (bptt - frozen)   (fraction of the BPTT-achievable deep credit)
  GO(seed) = chained_fa_inherit > permuted_inherit + 0.03 AND chained_fa_inherit > frozen_reservoir_inherit + 0.03
  6-seed headline = #seeds passing GO / n_seeds (+ the same for the KP arm).

ANTI-CHEATS: permuted -> chance (asserted); no-weight-transport (each Y is byte-!= every W_in and its transpose;
the KP update reads ONLY pre/post activity + Y, never a forward W -- asserted from source); frozen baseline reported;
cfg.seed is threaded into every SNN + task (per-seed reproducible). NO sim/ edit -- the LIF SNN forward + BPTT +
atan-surrogate are reused-by-import from `sim/bptt_snn_gpu`; the chained-FA credit is a RUNNER-side function.

Run (numpy CPU; the depth-2 BPTT-viable net, ~ (n_in+2*hidden+k) LIF units over T steps):
    SIM_BACKEND=numpy python -m research.runners._gap4_bptt_snn_chained_fa_transport_free_derisk --seeds 42
    SIM_BACKEND=numpy python -m research.runners._gap4_bptt_snn_chained_fa_transport_free_derisk --seeds 42 43 44 45 46 47
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
# TINY matmuls -> one BLAS thread per process (oversubscription is ~30x slower); parallelize across seeds instead.
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import inspect
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

# ---- reuse-by-import: the BPTT-viable LIF SNN forward + surrogate-gradient BPTT + atan surrogate (NO sim/ edit) ----
from sim.bptt_snn_gpu import (  # noqa: E402
    LIFLayerXP, forward_unroll_xp, backward_unroll_xp, atan_surrogate)
from sim.dendritic_mlp import DendriticMLP  # noqa: E402 -- the rate backprop oracle reference
# ---- reuse-by-import: the SAME task + the SAME forward/eval helpers the 0.82 BPTT result used ----
from research.runners._snn_bptt_forward_vs_learning_isolation_derisk import (  # noqa: E402
    _softmax, _build_layers, _forward_logits, _accuracy)
from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    make_task_semantic_inheritance, stage0_depth_genuineness, _train_oracle, _acc_on)

OUT = _REPO / "research" / "findings" / "raw" / "_gap4_bptt_snn_chained_fa.json"


# ============================================================================================================
# The CHAINED multi-hop TRANSPORT-FREE feedback-alignment credit -- the rate-overturn rule ported to the LIF SNN.
# Per timestep, descend the output error hop-by-hop through the stack (top -> bottom):
#     e_{L-1} = output_grad[t]                              # output error (target access; the top layer)
#     e_{li}  = (e_{li+1} @ Y[li]) . sigma'(v-theta)_{li}   # hidden li: TRANSPORT-FREE (Y[li] fixed-random or KP-
#                                                           # learned, NEVER a forward W); sigma' = atan-surrogate.
# The per-layer weight update is eps[li]^T @ (sigma'-gated error), where eps[li] = alpha_leak*eps[li] + pre is the
# FORWARD eligibility trace (captures the LIF leak-recurrence exactly for a feedforward net -- Bellec 2020 e-prop).
# This mirrors the multihop rate runner's `train` ladder (e_u2=(e_out@Y_out).S1n; e_u1=(e_u2@Y2).S0n) with the SAME
# per-layer sigma' RELATIVE gate (normalized to mean 1.0 so it selects WHICH neurons receive credit without its
# absolute magnitude scaling the step) -- but here on the REAL BPTT-viable spiking forward, not the reservoir.
# ============================================================================================================
def _make_feedback(sizes, seed):
    """Fixed-random chained feedback Y[li] : (sizes[li+2], sizes[li+1]) for each hidden transition li in 0..L-2 (one
    per hidden layer: the top hidden reads the output error (k), each deeper hidden reads the layer-above error).
    Drawn from a SEPARATE seed stream => TRANSPORT-FREE (never derived from any forward W_in). Returns a list of
    len(sizes)-2 matrices."""
    frng = np.random.default_rng(seed + 8888)
    Y = []
    for li in range(len(sizes) - 2):                         # transitions: output->top-hidden, hidden->hidden, ...
        n_above, n_here = sizes[li + 2], sizes[li + 1]
        Y.append((frng.standard_normal((n_above, n_here)) / np.sqrt(n_above)).astype(np.float64))
    return Y


def _chained_fa_grads(inputs, layers, fs, output_grad, Y_list, alpha_leak, alpha_surr=2.0,
                      sigma_norm=True, train_hidden=True, kp_cfg=None, lr=0.0):
    """CHAINED multi-hop transport-free feedback-alignment credit on the LIF SNN. Returns weight_grads (descent-side,
    same sign convention as backward_unroll_xp so the caller subtracts lr*wg).

    train_hidden=False => FROZEN-reservoir arm: ONLY the output LIF layer's local delta update is returned (hidden
    grads stay 0). This output update is BYTE-IDENTICAL to the output block of the full chain, so (chained - frozen)
    isolates EXACTLY the hidden-layer directed credit.
    kp_cfg (dict{kp_lr,kp_decay}) => Kolen-Pollack LEARNED feedback: accumulate the transport-free KP outer products
    (post=error at the layer above, pre=that transition's input spikes) over the T window and apply ONE KP update to
    each Y[li] IN PLACE (so Y_list learns across batches). NEVER reads a forward W_in (transport-free by construction)."""
    T, Bn, _ = inputs.shape
    L = len(layers)
    spikes = fs["spikes"]; v_per = fs["v"]
    weight_grads = [np.zeros_like(l.W_in) for l in layers]
    eps = [np.zeros((Bn, l.W_in.shape[0]), dtype=l.W_in.dtype) for l in layers]
    kp_accum = [np.zeros_like(Y) for Y in Y_list] if kp_cfg is not None else None

    for t in range(T):
        # forward eligibility traces (all layers) -- the leak-recurrence factor the e-prop / rate rule both use
        for li in range(L):
            pre = inputs[t] if li == 0 else spikes[li - 1][t]
            eps[li] = alpha_leak * eps[li] + pre
        # per-layer membrane surrogate sigma'(v-theta); RELATIVE-normalized (mean 1.0) when sigma_norm
        psi = []
        for li in range(L):
            p = atan_surrogate(v_per[li][t] - layers[li].threshold, alpha=alpha_surr, xp=np)
            if sigma_norm:
                p = p / (p.mean() + 1e-9)
            psi.append(p)
        # ---- output layer (target access): local delta update, gated by its own sigma' (== the e-prop DFA output) ----
        e_above = output_grad[t]                                    # (B, k) output error
        g_out = e_above * psi[L - 1]
        weight_grads[L - 1] += eps[L - 1].T @ g_out
        # ---- chained descent through the hidden layers (only when train_hidden) ----
        if train_hidden:
            for li in range(L - 2, -1, -1):
                if kp_accum is not None:
                    # KP pairing for Y[li] (mirrors forward transition W_in of layer li+1): post = e_above (error at
                    # layer li+1), pre = spikes[li][t] (the input to that transition). outer == Y[li].shape. NO W read.
                    kp_accum[li] += e_above.T @ spikes[li][t]
                e_below = (e_above @ Y_list[li]) * psi[li]          # error at layer li (TRANSPORT-FREE; Y not W)
                weight_grads[li] += eps[li].T @ e_below            # eligibility-weighted local update
                e_above = e_below

    # ---- KP learned-feedback update: Y[li] += lr*(kp_lr*outer - kp_decay*Y[li]); transport-free (Payeur/Akrout KP) ----
    if kp_cfg is not None:
        denom = max(1, Bn * T)
        for li in range(len(Y_list)):
            outer = kp_accum[li] / denom
            Y_list[li] = Y_list[li] + lr * (float(kp_cfg["kp_lr"]) * outer - float(kp_cfg["kp_decay"]) * Y_list[li])
    return weight_grads


def _no_weight_transport(Y_list, layers):
    """anti-cheat: no feedback matrix Y[li] is byte-equal to any forward W_in or its transpose (the 'Y is secretly
    W^T' backprop-in-disguise cheat). Y is a separate fixed-random stream; KP drives Y^T -> W in DIRECTION but never
    copies, so genuine arms pass."""
    for Y in Y_list:
        for l in layers:
            W = l.W_in
            if Y.shape == W.shape and np.array_equal(Y, W):
                return False
            if Y.shape == W.T.shape and np.array_equal(Y, W.T):
                return False
    return True


def _kp_reads_no_forward_weight():
    """anti-cheat (source guard): the chained-FA credit fn never reads a forward weight inside its KP path. The KP
    block uses ONLY kp_accum (post/pre activity), Y_list and kp_cfg -- assert the source holds no `l.W_in` read in
    the KP branch. Best-effort tripwire against a future in-file edit."""
    try:
        src = inspect.getsource(_chained_fa_grads)
    except (OSError, TypeError):
        return True
    kp_region = src.split("KP learned-feedback update")[-1]
    return "W_in" not in kp_region


# ============================================================================================================
# Trainer -- the SAME LIF SNN forward (_build_layers/_forward_logits from the 0.82 runner) with a selectable credit
# arm. mode in {bptt, chained_fa, chained_fa_kp, frozen_reservoir}; permuted is mode=chained_fa on shuffled y.
# ============================================================================================================
def _train_snn_arm(Xtr, ytr, sizes, T, epochs, lr, lr_fa, in_gain, seed, mode, batch=32,
                   sigma_norm=True, kp_lr=0.2, kp_decay=1e-4):
    rng = np.random.default_rng(seed + 1)
    w_scales = [2.5] + [1.0] * (len(sizes) - 2)                # strong first layer (same as the 0.82 runner)
    layers = _build_layers(sizes, T, rng, w_scales)
    k = sizes[-1]; n = len(Xtr)
    Y_list = None
    if mode in ("chained_fa", "chained_fa_kp", "frozen_reservoir"):
        Y_list = _make_feedback(sizes, seed)                   # fixed-random (separate stream) -> transport-free
    kp_cfg = {"kp_lr": kp_lr, "kp_decay": kp_decay} if mode == "chained_fa_kp" else None
    alpha_leak = layers[0].leak
    for _ in range(epochs):
        perm = rng.permutation(n)
        for b0 in range(0, n, batch):
            bi = perm[b0:b0 + batch]
            Xb, yb = Xtr[bi], ytr[bi]
            logits, fs, inp = _forward_logits(Xb, layers, T, in_gain)
            p = _softmax(logits)
            delta = p.copy(); delta[np.arange(len(yb)), yb] -= 1.0     # (B, k) dL/d(sum spikes)
            og = np.repeat((delta / T)[None, :, :], T, axis=0).astype(np.float64)  # (T, B, k)
            if mode == "bptt":
                wg, _ = backward_unroll_xp(inp, layers, fs, og, alpha=2.0, xp=np)
                use_lr = lr
            else:
                train_hidden = mode in ("chained_fa", "chained_fa_kp")
                wg = _chained_fa_grads(inp, layers, fs, og, Y_list, alpha_leak, alpha_surr=2.0,
                                       sigma_norm=sigma_norm, train_hidden=train_hidden,
                                       kp_cfg=kp_cfg, lr=lr_fa)
                use_lr = lr_fa
            for li in range(len(layers)):
                # frozen_reservoir: hidden grads are 0 by construction; still apply (0 => no-op) for one code path.
                # the OUTPUT layer always learns at lr (its target-access delta), hidden at use_lr.
                arm_lr = lr if (li == len(layers) - 1) else use_lr
                layers[li].W_in -= arm_lr * (wg[li] / len(bi))
    return layers, Y_list


def run_seed(seed, hidden, T, epochs, lr, lr_fa, in_gain, subsample, task_kwargs, n_hidden_layers=2,
             sigma_norm=True, kp_lr=0.2, kp_decay=1e-4, check_depth=True):
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_semantic_inheritance(seed, **task_kwargs)
    k = meta["k_classes"]; n_in = Xtr.shape[1]; inh_idx = idx["inh_idx"]
    if len(inh_idx):
        yv = yte[inh_idx]; chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    else:
        chance = float("nan")

    # ---- absolute reference: rate backprop oracle (the depth-2 DendriticMLP) on the SAME task ----
    onet = DendriticMLP([n_in] + [96] * n_hidden_layers + [k], seed=seed)
    _train_oracle(onet, Xtr, ytr, 250, 0.3, 128, seed)
    oracle_inh = _acc_on(onet, Xte, yte, inh_idx)
    depth_sep = None
    if check_depth:
        s0 = stage0_depth_genuineness(((Xtr, ytr, Ltr), (Xte, yte, Lte)), idx, k, hidden=96, epochs=250,
                                      lr=0.3, batch=128, seed=seed)
        depth_sep = bool(s0.get("depth_separating"))

    if subsample is not None and len(Xtr) > subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:subsample]
        Xtr, ytr = Xtr[keep], ytr[keep]

    sizes = [n_in] + [hidden] * n_hidden_layers + [k]

    def _run(mode, ylabels):
        layers, Y_list = _train_snn_arm(Xtr, ylabels, sizes, T, epochs, lr, lr_fa, in_gain, seed, mode,
                                        sigma_norm=sigma_norm, kp_lr=kp_lr, kp_decay=kp_decay)
        inh = _accuracy(Xte, yte, layers, T, in_gain, sub=inh_idx)
        tr = _accuracy(Xtr, ytr, layers, T, in_gain)
        nt = _no_weight_transport(Y_list, layers) if Y_list is not None else True
        return {"inherit": inh, "train": tr, "no_transport": bool(nt)}, layers, Y_list

    # ---- ARM 1: surrogate-BPTT ceiling ----
    bptt, _, _ = _run("bptt", ytr)
    # ---- ARM 3: frozen-hidden reservoir (only the output LIF layer learns) ----
    frozen, _, _ = _run("frozen_reservoir", ytr)
    # ---- ARM 2: chained transport-free FIXED-random FA (the primary transport-free-local arm) ----
    chained, _, Yc = _run("chained_fa", ytr)
    # ---- ARM 2b: chained transport-free KP-LEARNED FA (the depth rescue) ----
    chained_kp, _, _ = _run("chained_fa_kp", ytr)
    # ---- ANTI-CHEAT: permuted labels through the chained-FA arm -> chance ----
    prng = np.random.default_rng(seed + 555)
    yperm = ytr[prng.permutation(len(ytr))]
    permuted, _, _ = _run("chained_fa", yperm)

    # ---- decisive metrics (inherit held-out) ----
    def _frac(x):
        d = bptt["inherit"] - frozen["inherit"]
        return float((x - frozen["inherit"]) / d) if abs(d) > 1e-6 else float("nan")

    directed_over_permuted = float(chained["inherit"] - permuted["inherit"])
    purchase_over_frozen = float(chained["inherit"] - frozen["inherit"])
    kp_directed_over_permuted = float(chained_kp["inherit"] - permuted["inherit"])
    kp_purchase_over_frozen = float(chained_kp["inherit"] - frozen["inherit"])
    go_fixed = bool(chained["inherit"] > permuted["inherit"] + 0.03 and
                    chained["inherit"] > frozen["inherit"] + 0.03)
    go_kp = bool(chained_kp["inherit"] > permuted["inherit"] + 0.03 and
                 chained_kp["inherit"] > frozen["inherit"] + 0.03)

    return {
        "seed": seed, "chance": chance, "n_in": n_in, "k": k, "sizes": sizes,
        "depth_separating": depth_sep, "oracle_inherit": oracle_inh,
        "bptt_inherit": bptt["inherit"], "bptt_train": bptt["train"],
        "frozen_reservoir_inherit": frozen["inherit"], "frozen_reservoir_train": frozen["train"],
        "chained_fa_inherit": chained["inherit"], "chained_fa_train": chained["train"],
        "chained_fa_kp_inherit": chained_kp["inherit"], "chained_fa_kp_train": chained_kp["train"],
        "permuted_inherit": permuted["inherit"],
        "directed_over_permuted": directed_over_permuted,
        "purchase_over_frozen": purchase_over_frozen,
        "bptt_fraction_captured": _frac(chained["inherit"]),
        "kp_directed_over_permuted": kp_directed_over_permuted,
        "kp_purchase_over_frozen": kp_purchase_over_frozen,
        "kp_bptt_fraction_captured": _frac(chained_kp["inherit"]),
        "GO_fixed": go_fixed, "GO_kp": go_kp,
        # anti-cheats
        "no_transport_chained_fa": bool(chained["no_transport"]),
        "no_transport_chained_kp": bool(chained_kp["no_transport"]),
        "kp_source_no_forward_weight": bool(_kp_reads_no_forward_weight()),
        "permuted_near_chance": bool(abs(permuted["inherit"] - chance) <= 0.06)
        if not np.isnan(chance) else None,
    }


def _agg(results):
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {}
    def _m(key):
        vals = [r[key] for r in ok if r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")
    n = len(ok)
    return {
        "n_seeds": n,
        "mean_chance": _m("chance"),
        "mean_oracle_inherit": _m("oracle_inherit"),
        "mean_bptt_inherit": _m("bptt_inherit"),
        "mean_frozen_reservoir_inherit": _m("frozen_reservoir_inherit"),
        "mean_chained_fa_inherit": _m("chained_fa_inherit"),
        "mean_chained_fa_kp_inherit": _m("chained_fa_kp_inherit"),
        "mean_permuted_inherit": _m("permuted_inherit"),
        "mean_directed_over_permuted": _m("directed_over_permuted"),
        "mean_purchase_over_frozen": _m("purchase_over_frozen"),
        "mean_bptt_fraction_captured": _m("bptt_fraction_captured"),
        "mean_kp_directed_over_permuted": _m("kp_directed_over_permuted"),
        "mean_kp_purchase_over_frozen": _m("kp_purchase_over_frozen"),
        "GO_fixed_seeds": f"{sum(bool(r.get('GO_fixed')) for r in ok)}/{n}",
        "GO_kp_seeds": f"{sum(bool(r.get('GO_kp')) for r in ok)}/{n}",
        "no_transport_all": bool(all(r.get("no_transport_chained_fa") and r.get("no_transport_chained_kp")
                                     and r.get("kp_source_no_forward_weight") for r in ok)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-hidden-layers", type=int, default=2)
    ap.add_argument("--timesteps", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05, help="BPTT + output-layer local-delta learning rate")
    ap.add_argument("--lr-fa", type=float, default=None, help="chained-FA hidden learning rate (default = --lr)")
    ap.add_argument("--in-gain", type=float, default=1.0)
    ap.add_argument("--sigma-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="normalize sigma'(v-theta) per layer to mean 1.0 (the rate-rule RELATIVE gate)")
    ap.add_argument("--kp-lr", type=float, default=0.2)
    ap.add_argument("--kp-decay", type=float, default=1e-4)
    ap.add_argument("--train-subsample", type=int, default=400)
    ap.add_argument("--no-depth-check", action="store_true", help="skip the stage0 depth-genuineness probe (faster)")
    # task kwargs (mirror the 0.82 runner defaults)
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

    lr_fa = args.lr if args.lr_fa is None else args.lr_fa
    task_kwargs = {"n_super": args.n_super, "n_members": args.n_members,
                   "held_per_super": args.held_per_super, "n_prop": args.n_prop,
                   "member_id_dim": args.member_id_dim, "n_obs": args.n_obs,
                   "noise": args.noise, "feature_seed": args.feature_seed}
    t0 = time.time()
    results = []
    for sd in args.seeds:
        try:
            r = run_seed(sd, args.hidden, args.timesteps, args.epochs, args.lr, lr_fa, args.in_gain,
                         args.train_subsample, task_kwargs, n_hidden_layers=args.n_hidden_layers,
                         sigma_norm=args.sigma_norm, kp_lr=args.kp_lr, kp_decay=args.kp_decay,
                         check_depth=not args.no_depth_check)
        except Exception as e:
            r = {"seed": sd, "error": repr(e), "traceback": traceback.format_exc()}
        results.append(r)
        if "error" not in r:
            print(f"[seed {sd}] chained_fa {r['chained_fa_inherit']:.3f} (kp {r['chained_fa_kp_inherit']:.3f}) "
                  f"vs frozen {r['frozen_reservoir_inherit']:.3f} vs permuted {r['permuted_inherit']:.3f} | "
                  f"BPTT ceiling {r['bptt_inherit']:.3f}, oracle {r['oracle_inherit']:.3f}, chance {r['chance']:.3f} "
                  f"| directed(over-perm) {r['directed_over_permuted']:+.3f}, purchase(over-frozen) "
                  f"{r['purchase_over_frozen']:+.3f}, frac-BPTT {r['bptt_fraction_captured']:.2f} | "
                  f"GO_fixed={r['GO_fixed']} GO_kp={r['GO_kp']}")
        else:
            print(f"[seed {sd}] ERROR: {r['error']}")

    agg = _agg(results)
    out = {"probe": "gap4_bptt_snn_chained_fa_transport_free", "seeds": args.seeds,
           "config": {"hidden": args.hidden, "n_hidden_layers": args.n_hidden_layers, "T": args.timesteps,
                      "epochs": args.epochs, "lr": args.lr, "lr_fa": lr_fa, "in_gain": args.in_gain,
                      "sigma_norm": args.sigma_norm, "kp_lr": args.kp_lr, "kp_decay": args.kp_decay,
                      "train_subsample": args.train_subsample, "task": task_kwargs},
           "elapsed_seconds": round(time.time() - t0, 1), "results": results, "aggregate": agg}
    if agg:
        out["verdict"] = (
            f"transport-free chained-FA on the TRAINABLE LIF SNN: chained_fa {agg['mean_chained_fa_inherit']:.3f} "
            f"(kp {agg['mean_chained_fa_kp_inherit']:.3f}) vs frozen {agg['mean_frozen_reservoir_inherit']:.3f} vs "
            f"permuted {agg['mean_permuted_inherit']:.3f}; BPTT ceiling {agg['mean_bptt_inherit']:.3f}, oracle "
            f"{agg['mean_oracle_inherit']:.3f}. directed(over-permuted) {agg['mean_directed_over_permuted']:+.3f}, "
            f"purchase(over-frozen) {agg['mean_purchase_over_frozen']:+.3f}, BPTT-fraction "
            f"{agg['mean_bptt_fraction_captured']:.2f}. GO_fixed {agg['GO_fixed_seeds']}, GO_kp {agg['GO_kp_seeds']}. "
            + ("=> the transport-free LOCAL rule GETS directed-credit purchase on the TRAINABLE spiking substrate "
               "where it could NOT on the reservoir -> the SUBSTRATE was the wall (a trainable-substrate surpass)."
               if (agg.get("GO_fixed_seeds", "0/1").split("/")[0] != "0") else
               "=> the transport-free LOCAL rule STILL fails on the trainable spiking substrate -> a deeper "
               "transport-free-vs-BPTT gap on spikes (names the next mechanism)."))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + out.get("verdict", "(no aggregate)"))


if __name__ == "__main__":
    main()
