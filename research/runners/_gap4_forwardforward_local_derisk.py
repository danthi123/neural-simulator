"""gap#4 SURPASS candidate: a SPIKING FORWARD-FORWARD (per-layer LOCAL contrastive-goodness) rule on a TRAINABLE LIF
stack -- the ONE genuinely-new mechanism class for the located wall (Hinton 2022 arXiv:2212.13345; spike-native traces
per Traces-Propagation arXiv:2509.13053; Kohan Signal-Propagation contrastive idea). NO sim/ edit.

THE LOCATED WALL (do NOT re-derive -- read the two 2026-08-02 findings):
  * DOC1 (crux wall LOCATED): on the movable-plateau RESERVOIR substrate, even a PERFECT W^T oracle (feedback-alignment
    ~0.999) gives NO directed credit through the finite-spike sigma'(v-theta) read (oracle == permuted, 5 controls).
  * DOC2 (depth-rescue untestable): on the TRAINABLE LIF SNN, the CHAINED multi-hop transport-free rule (fixed-FA AND
    KP-learned) does NOT leave majority-class at N>=3 -- FA/KP held acc 0.45-0.54 == chance at N=3,4 on the depth-2 XOR
    task, byte-identical for FA and KP (the degenerate-dynamics fingerprint). Only N=2 enters the learning regime.

WHY FORWARD-FORWARD IS THE NEW MECHANISM CLASS (not another FA/dendrite/BDSP variant):
  Every prior arm routes a TOP-DOWN error hop-by-hop and re-gates it by sigma'(v-theta) at each hop -- exactly what the
  wall kills. FORWARD-FORWARD has NO top-down credit path AT ALL. Each hidden layer trains from its OWN LOCAL contrastive
  "goodness" objective on its OWN forward spike-rate (positive = real-label-paired input, negative = wrong-label-paired):
      goodness g_l(x) = mean_j r_{l,j}^2         r_{l,j} = (1/T) sum_t s_{l,j}[t]   (per-neuron spike RATE)
      per-layer loss  L_l = -log sigma(g_l^pos - theta)  - log(1 - sigma(g_l^neg - theta))   (push pos UP, neg DOWN)
      local update    dW_l = pre_l^T @ ( dg_l  (.)  psi_l )   psi_l = mean_t sigma'(v_l[t]-theta_v)  (surrogate elig.)
  There is NO chain, NO feedback matrix, NO delivered top-down error -- so the finite-spike-read wall has NOTHING to fail
  through. The sigma' here gates a LOCAL, strongly label-dependent goodness error (g^pos vs g^neg), not a weak error
  routed through misaligned feedback. Between layers the rate vector is LAYER-NORMALIZED (only orientation passes) so a
  deeper layer cannot free-ride on the goodness magnitude below it -- it must find NEW discriminative features.
  BRAIN-BASED: local pre*post*surrogate three-factor update (STDP-modulated eligibility), no weight transport, no
  cross-layer credit. The BPTT arm is a labelled CEILING ONLY (never shipped).

THE ENTER-THE-REGIME CHECK (decisive, the DOC2 wall's own metric): does the DEEP net LEAVE majority-class at N=3 AND
N=4?  Report per-layer goodness-classification accuracy + accuracy-above-majority at every depth. GO if the full-net FF
held-out >= chance+0.20 AND min-over-seeds clearly above majority AND beats the frozen reservoir by >=0.10 with the
permuted-label arm collapsing to chance and the BPTT ceiling confirming the target is learnable -- precisely where the
top-down FA/KP arms collapsed. HONEST-NEGATIVE (first-class) if FF also fails to leave majority-class: report whether it
is a TOTAL failure (never leaves majority at any depth) or a WEAK-COUPLING failure (shallow layers discriminate, the
DEEP layer is trained-but-not-obligatory -- its per-layer accuracy ~ chance while the full net rides the shallow layers).

ARMS (all on the SAME LIF forward + SAME task):
  ff              : FORWARD-FORWARD, all N hidden layers trained by the local contrastive-goodness rule (the candidate).
  ff_reservoir    : hidden layers FROZEN at init; FF-goodness inference only (random-projection floor -- should ~chance).
  reservoir_ridge : the STRONGEST reservoir control -- frozen random LIF reservoir read by an OPTIMAL 5-fold-CV ridge on
                    the concatenated summed-spikes (reuse-by-import). On XOR (not linearly decodable from a fixed random
                    projection) this must sit at chance; ff >> it == the categorical unlock.
  bptt_ceiling    : surrogate-gradient BPTT SNN with a supervised output layer (reuse-by-import) -- the labelled UPPER
                    bound / target-exists check (NON-local; ceiling only, never shipped).
  ff_permuted     : FF trained on SHUFFLED labels -> must collapse to chance (isolates the correct-label-attributable lift).

Run (numpy CPU; fan across seeds as parallel processes -- launch-bound):
    # NOTE: the GO operating point needs --lr 1.0 --label-gain 3.0 (+ a tuned BPTT ceiling). The default lr
    #       reproduces CHANCE (a GO=False landmine) -- always pass these flags. See the finding's Reproduce block.
    # smoke (one seed, N=3):
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap4_forwardforward_local_derisk \
        --task-xor --seed 42 --n-list 3 --epochs 350 --lr 1.0 --label-gain 3.0 \
        --bptt-hidden 128 --bptt-epochs 200 --bptt-lr 0.2 \
        --out research/findings/raw/_gap4_ff/ff_smoke_seed42.json
    # one seed, all depths N=2,3,4 (one process per seed; fan 6):
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap4_forwardforward_local_derisk \
        --task-xor --seed 42 --n-list 2 3 4 --epochs 350 --lr 1.0 --label-gain 3.0 \
        --bptt-hidden 128 --bptt-epochs 200 --bptt-lr 0.2 \
        --out research/findings/raw/_gap4_ff/ff_xor_seed42.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

# ---- reuse-by-import: the BPTT-viable LIF forward + atan surrogate (NO sim/ edit) ----
from sim.bptt_snn_gpu import forward_unroll_xp, atan_surrogate  # noqa: E402
# ---- reuse-by-import: the SAME layer builder + BPTT ceiling trainer + eval the 0.82 result used ----
from research.runners._snn_bptt_forward_vs_learning_isolation_derisk import (  # noqa: E402
    _build_layers, _train_snn, _accuracy)
# ---- reuse-by-import: the EXACT tasks the DOC2 wall was measured on ----
from research.runners._gap4_bptt_snn_chained_fa_transport_free_derisk import (  # noqa: E402
    make_task_xor, make_task_hier3, _frozen_reservoir_optimal)

OUT = _REPO / "research" / "findings" / "raw" / "_gap4_ff" / "_gap4_ff.json"
ALPHA_SURR = 2.0


# ============================================================================================================
# Label overlay: append a BIPOLAR one-hot (-1 everywhere, +1 at class c) scaled by label_gain (Hinton's label-in-input,
# adapted so the label drives current on the SAME +/-1 scale as the XOR features). Positive = correct c, negative = a
# wrong c. Inference tries every c and accumulates goodness. NO leakage: both pos and neg carry a label, so the net
# cannot detect "is there a label" -- it must correlate the overlaid label with the input-derived class.
# ============================================================================================================
def _overlay(X, c, k, label_gain):
    B = X.shape[0]
    oh = -np.ones((B, k), dtype=X.dtype)
    oh[:, c] = 1.0
    return np.concatenate([X, label_gain * oh], axis=1)


def _forward_ff(x_current, layers, T, in_gain):
    """Per-layer LIF forward with FF LAYER-NORM between layers (only the orientation of the rate vector passes on).
    Returns per-layer (rate r_l (B,H), surrogate psi_l (B,H) = mean_t sigma'(v-theta), pre-current pre_l (B,n_pre)).
    Every layer receives a CONSTANT input current over T (rate-coded, exactly as the task features are). The layer-norm
    is RMS (unit ROOT-MEAN-SQUARE, i.e. L2/sqrt(H)) so the normalized rate vector has entries O(1) -- matching the +/-1
    feature scale that keeps layer 0 firing -- so EVERY deeper layer stays in its firing regime instead of going silent
    (an L2-unit norm gives entries ~1/sqrt(H) -> deep layers barely fire -> no goodness -> no local gradient)."""
    cur = x_current
    rates, psis, pres = [], [], []
    for layer in layers:
        pre = (in_gain * cur).astype(np.float64)                 # (B, n_pre) constant current -> the gradient's `pre`
        inp = np.repeat(pre[None, :, :], T, axis=0)              # (T, B, n_pre)
        fs = forward_unroll_xp(inp, [layer], xp=np)
        s = fs["spikes"][0]                                      # (T, B, H)
        v = fs["v"][0]                                           # (T, B, H)
        r = s.sum(axis=0) / T                                    # (B, H) per-neuron spike RATE in [0,1]
        psi = atan_surrogate(v - layer.threshold, alpha=ALPHA_SURR, xp=np).mean(axis=0)   # (B, H) time-avg surrogate
        rates.append(r); psis.append(psi); pres.append(pre)
        rms = np.sqrt((r ** 2).mean(axis=1, keepdims=True))     # RMS layer-norm: ORIENTATION only, entries O(1)
        cur = r / (rms + 1e-8)
    return rates, psis, pres


def _goodness(r):
    """FF goodness = mean-of-squares of the per-neuron rate (width-invariant, in [0,1])."""
    return (r ** 2).mean(axis=1)                                 # (B,)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _ff_local_grads(Xpos, Xneg, layers, T, in_gain, theta, objective="paired"):
    """The FORWARD-FORWARD local update. For EACH layer independently, using ONLY its own rate/voltage/input current --
    NO cross-layer credit path, NO feedback matrix, NO delivered top-down error (this is exactly why the finite-spike
    read wall has nothing to fail through). dg/dr_j = 2 r_j / H; local weight grad dW_l = pre_l^T @ ((dL/dr) (.) psi_l).

    objective='paired' (default, theta-FREE, robust -- SymBa/contrastive-FF): per EXAMPLE d = g_pos - g_neg, push the
        pos-vs-neg goodness DIFFERENCE up: L = -log sigma(d). e_pos = -(1 - sigma(d)), e_neg = +(1 - sigma(d)). No
        absolute threshold to mis-set; the two passes are the SAME features with correct vs wrong overlaid label.
    objective='absolute' (Hinton): sigma(g - theta) toward target 1 (pos) / 0 (neg); sensitive to theta at the operating
        point."""
    r_pos, psi_pos, pre_pos = _forward_ff(Xpos, layers, T, in_gain)
    r_neg, psi_neg, pre_neg = _forward_ff(Xneg, layers, T, in_gain)
    grads = []
    for li, layer in enumerate(layers):
        H = layer.n_post
        gp = _goodness(r_pos[li]); gn = _goodness(r_neg[li])
        if objective == "absolute":
            e_pos = (_sigmoid(gp - theta) - 1.0)                 # (B,) push positive goodness ABOVE theta
            e_neg = (_sigmoid(gn - theta) - 0.0)                 # (B,) push negative goodness BELOW theta
        else:
            w = 1.0 - _sigmoid(gp - gn)                          # (B,) paired: how far the correct pairing is from winning
            e_pos = -w                                           # raise g_pos
            e_neg = +w                                           # lower g_neg
        dr_pos = (e_pos[:, None] * (2.0 / H)) * r_pos[li]        # (B,H) dL/dr on the positive pass
        dr_neg = (e_neg[:, None] * (2.0 / H)) * r_neg[li]
        gW = pre_pos[li].T @ (dr_pos * psi_pos[li]) + pre_neg[li].T @ (dr_neg * psi_neg[li])   # (n_pre,H)
        grads.append(gW)
    return grads


def _neg_labels(y, k, rng):
    """A wrong label per example (uniform over the k-1 classes != y)."""
    off = rng.integers(1, k, size=len(y))
    return (y + off) % k


def _train_ff(Xtr, ytr, sizes_ff, T, epochs, lr, in_gain, seed, k, theta, label_gain,
              batch=32, train_hidden=True, objective="paired"):
    """Train the FF stack. train_hidden=False => FROZEN reservoir (weights never move; FF-goodness inference only)."""
    rng = np.random.default_rng(seed + 1)
    w_scales = [2.5] + [1.0] * (len(sizes_ff) - 2)              # strong first layer (same init as every SNN arm here)
    layers = _build_layers(sizes_ff, T, rng, w_scales)
    n = len(Xtr)
    nrng = np.random.default_rng(seed + 202)
    for _ in range(epochs):
        if not train_hidden:
            break
        perm = rng.permutation(n)
        for b0 in range(0, n, batch):
            bi = perm[b0:b0 + batch]
            Xb, yb = Xtr[bi], ytr[bi]
            yneg = _neg_labels(yb, k, nrng)
            # positive = correct-label overlay; negative = wrong-label overlay (SAME input features)
            Xpos = np.concatenate([_overlay(Xb[j:j + 1], int(yb[j]), k, label_gain) for j in range(len(bi))], axis=0)
            Xneg = np.concatenate([_overlay(Xb[j:j + 1], int(yneg[j]), k, label_gain) for j in range(len(bi))], axis=0)
            grads = _ff_local_grads(Xpos, Xneg, layers, T, in_gain, theta, objective=objective)
            for li in range(len(layers)):
                layers[li].W_in -= lr * (grads[li] / len(bi))
    return layers


def _ff_goodness_table(X, layers, T, in_gain, k, label_gain):
    """G[b, c, l] = goodness of example b with candidate label c at hidden layer l. One forward per candidate label."""
    B = len(X); L = len(layers)
    G = np.zeros((B, k, L), dtype=np.float64)
    for c in range(k):
        Xc = _overlay(X, c, k, label_gain)
        rates, _, _ = _forward_ff(Xc, layers, T, in_gain)
        for li in range(L):
            G[:, c, li] = _goodness(rates[li])
    return G


def _majority_fraction(y):
    if len(y) == 0:
        return float("nan")
    return float(max(np.mean(y == c) for c in np.unique(y)))


def _ff_eval(X, y, layers, T, in_gain, k, label_gain, exclude_first=True):
    """Full-net FF accuracy (argmax_c of accumulated goodness over the USED hidden layers) + per-layer accuracy
    (argmax_c g_l) so weak-coupling is visible: does the DEEP layer discriminate, or only the shallow ones?"""
    if len(X) == 0:
        return {"full_acc": float("nan"), "per_layer_acc": [], "used_layers": []}
    G = _ff_goodness_table(X, layers, T, in_gain, k, label_gain)      # (B, k, L)
    L = len(layers)
    per_layer = [float(np.mean(np.argmax(G[:, :, li], axis=1) == y)) for li in range(L)]
    used = list(range(1, L)) if (exclude_first and L > 1) else list(range(L))
    Gacc = G[:, :, used].sum(axis=2)                                 # accumulate goodness across used layers
    full_acc = float(np.mean(np.argmax(Gacc, axis=1) == y))
    return {"full_acc": full_acc, "per_layer_acc": per_layer, "used_layers": used}


def run_seed(seed, n_hidden_layers, hidden, T, epochs, lr, in_gain, theta, label_gain, subsample,
             task_xor, task_hier3, bptt_hidden, bptt_epochs, bptt_lr, wide_hidden, objective="paired"):
    if task_hier3:
        (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_hier3(seed)
    else:
        (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_xor(seed)
    k = meta["k_classes"]; n_in = Xtr.shape[1]; inh_idx = idx["inh_idx"]
    Xinh = Xte[inh_idx]; yinh = yte[inh_idx]
    chance = _majority_fraction(yinh)
    majority = chance

    if subsample is not None and len(Xtr) > subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:subsample]
        Xtr, ytr = Xtr[keep], ytr[keep]

    sizes_ff = [n_in + k] + [hidden] * n_hidden_layers            # ALL N layers are trainable hidden (no output layer)

    # ---- ARM ff: FORWARD-FORWARD, all hidden layers trained by the local contrastive-goodness rule ----
    ff_layers = _train_ff(Xtr, ytr, sizes_ff, T, epochs, lr, in_gain, seed, k, theta, label_gain, train_hidden=True, objective=objective)
    ff_tr = _ff_eval(Xtr, ytr, ff_layers, T, in_gain, k, label_gain)
    ff_inh = _ff_eval(Xinh, yinh, ff_layers, T, in_gain, k, label_gain)

    # ---- ARM ff_reservoir: FROZEN-at-init hidden, FF-goodness inference (random-projection floor) ----
    res_layers = _train_ff(Xtr, ytr, sizes_ff, T, epochs, lr, in_gain, seed, k, theta, label_gain, train_hidden=False, objective=objective)
    res_inh = _ff_eval(Xinh, yinh, res_layers, T, in_gain, k, label_gain)

    # ---- ARM ff_permuted: FF trained on SHUFFLED labels -> must collapse to chance ----
    prng = np.random.default_rng(seed + 555)
    yperm = ytr[prng.permutation(len(ytr))]
    perm_layers = _train_ff(Xtr, yperm, sizes_ff, T, epochs, lr, in_gain, seed, k, theta, label_gain, train_hidden=True, objective=objective)
    perm_inh = _ff_eval(Xinh, yinh, perm_layers, T, in_gain, k, label_gain)

    # ---- ARM reservoir_ridge: OPTIMAL 5-fold-CV ridge readout of a frozen random LIF reservoir (strongest floor) ----
    res_ridge, res_lam = _frozen_reservoir_optimal(Xtr, ytr, Xinh, yinh, n_in, k, hidden,
                                                   n_hidden_layers, T, in_gain, seed)

    # ---- ARM bptt_ceiling: surrogate-gradient BPTT SNN with supervised output layer (labelled UPPER bound) ----
    bh = bptt_hidden if bptt_hidden is not None else hidden
    be = bptt_epochs if bptt_epochs is not None else epochs
    bl = bptt_lr if bptt_lr is not None else lr
    bptt_sizes = [n_in] + [bh] * n_hidden_layers + [k]
    bptt_layers = _train_snn(Xtr, ytr, bptt_sizes, T, be, bl, in_gain, seed, credit_mode="bptt")
    bptt_inh = _accuracy(Xte, yte, bptt_layers, T, in_gain, sub=inh_idx)

    # ---- decisive metrics on the inherit held-out ----
    ff_acc = ff_inh["full_acc"]
    enters_regime = bool((not np.isnan(ff_acc)) and ff_acc > majority + 0.05)
    per_layer = ff_inh["per_layer_acc"]
    deepest_above_majority = float(per_layer[0] - majority) if per_layer else float("nan")  # layer 0 = deepest-from-out? no: index 0 = FIRST hidden
    # For a FEEDFORWARD FF stack layer index 0 is the FIRST hidden (closest to input); the LAST index is the top hidden.
    top_layer_above_majority = float(per_layer[-1] - majority) if per_layer else float("nan")
    beats_reservoir = float(ff_acc - res_ridge)
    beats_res_ff = float(ff_acc - res_inh["full_acc"])
    directed_over_permuted = float(ff_acc - perm_inh["full_acc"])
    bptt_confirms = bool((not np.isnan(bptt_inh)) and bptt_inh > majority + 0.15)

    go = bool(
        ff_acc >= majority + 0.20 and
        beats_reservoir >= 0.10 and
        beats_res_ff >= 0.10 and
        directed_over_permuted >= 0.10 and
        perm_inh["full_acc"] <= majority + 0.08 and
        bptt_confirms
    )
    # weak-coupling detector: full net rides shallow layers while a deeper layer is trained-but-not-obligatory
    max_single_layer = float(max(per_layer)) if per_layer else float("nan")
    depth_contributes = bool(per_layer and ff_acc > max_single_layer + 0.02)   # accumulation beats best single layer
    weak_coupling = bool(per_layer and enters_regime and (top_layer_above_majority <= 0.05))

    return {
        "seed": int(seed), "N": int(n_hidden_layers), "k_classes": int(k), "n_in": int(n_in),
        "chance": chance, "majority": majority,
        "ff_inherit": ff_acc, "ff_train": ff_tr["full_acc"], "ff_per_layer_acc": per_layer,
        "ff_used_layers": ff_inh["used_layers"],
        "ff_reservoir_inherit": res_inh["full_acc"], "ff_reservoir_per_layer_acc": res_inh["per_layer_acc"],
        "reservoir_ridge_inherit": float(res_ridge), "reservoir_ridge_lambda": float(res_lam),
        "bptt_ceiling_inherit": bptt_inh,
        "permuted_inherit": perm_inh["full_acc"],
        "enters_learning_regime": enters_regime,
        "acc_above_majority": float(ff_acc - majority),
        "top_layer_above_majority": top_layer_above_majority,
        "beats_reservoir_ridge_by": beats_reservoir, "beats_reservoir_ff_by": beats_res_ff,
        "directed_over_permuted": directed_over_permuted,
        "bptt_confirms_target": bptt_confirms,
        "max_single_layer_acc": max_single_layer, "depth_contributes": depth_contributes,
        "weak_coupling_suspected": weak_coupling,
        "GO": go,
        "task": meta.get("task"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="if set, loop these seeds in-process")
    ap.add_argument("--n-list", type=int, nargs="*", default=[2, 3, 4], help="hidden-layer depths to sweep")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--timesteps", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--in-gain", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=0.05, help="FF goodness threshold (absolute objective only)")
    ap.add_argument("--objective", type=str, default="paired", choices=["paired", "absolute"])
    ap.add_argument("--label-gain", type=float, default=3.0)
    ap.add_argument("--train-subsample", type=int, default=2000)
    ap.add_argument("--task-xor", action="store_true")
    ap.add_argument("--task-hier3", action="store_true")
    ap.add_argument("--bptt-hidden", type=int, default=None)
    ap.add_argument("--bptt-epochs", type=int, default=None)
    ap.add_argument("--bptt-lr", type=float, default=None)
    ap.add_argument("--wide-hidden", type=int, default=256)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    if not (args.task_xor or args.task_hier3):
        args.task_xor = True                                    # default to the DOC2 wall's own task

    seeds = args.seeds if args.seeds else [args.seed]
    t0 = time.time()
    results = []
    for sd in seeds:
        for N in args.n_list:
            try:
                r = run_seed(sd, N, args.hidden, args.timesteps, args.epochs, args.lr, args.in_gain,
                             args.theta, args.label_gain, args.train_subsample,
                             args.task_xor, args.task_hier3, args.bptt_hidden, args.bptt_epochs,
                             args.bptt_lr, args.wide_hidden, objective=args.objective)
            except Exception as e:
                r = {"seed": int(sd), "N": int(N), "error": repr(e), "traceback": traceback.format_exc()}
            results.append(r)
            tag = (f"seed {sd} N={N}: ff_inh={r.get('ff_inherit')} maj={r.get('majority')} "
                   f"res_ridge={r.get('reservoir_ridge_inherit')} bptt={r.get('bptt_ceiling_inherit')} "
                   f"perm={r.get('permuted_inherit')} per_layer={r.get('ff_per_layer_acc')} GO={r.get('GO')}")
            print(tag, flush=True)

    out = {"probe": "gap4_forwardforward_local", "backend": os.environ.get("SIM_BACKEND"),
           "config": {"n_list": args.n_list, "hidden": args.hidden, "T": args.timesteps, "epochs": args.epochs,
                      "lr": args.lr, "in_gain": args.in_gain, "theta": args.theta, "label_gain": args.label_gain,
                      "train_subsample": args.train_subsample,
                      "task": "hier3" if args.task_hier3 else "xor",
                      "bptt_hidden": args.bptt_hidden, "bptt_epochs": args.bptt_epochs, "bptt_lr": args.bptt_lr},
           "seeds": seeds, "elapsed_seconds": round(time.time() - t0, 1), "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[wrote {args.out}] elapsed {out['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
