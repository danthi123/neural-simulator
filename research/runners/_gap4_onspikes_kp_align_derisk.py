"""gap#4 deep-credit-ON-SPIKES -- port the RATE-level KP learned-feedback surpass to the SPIKING substrate, and MEASURE
the mechanism the existing spiking runner never read: cos(Y_l, W_{l+1}^T) -- the transport-free feedback-ALIGNMENT.

WHAT THIS PORTS. The rate de-risk (`_gap4_learned_feedback_derisk.py`, finding 2026-08-11) showed transport-free
Kolen-Pollack LEARNED feedback REACHES the 3rd hidden layer where FIXED-random DFA cannot, and its CENTRAL SIGNATURE is
cos(G_l, W_l^T) RISING from ~0 to ~0.83 through training (co-adapted, never copied). The spiking substrate ALREADY has
the KP rule wired (`_gap4_bptt_snn_chained_fa_transport_free_derisk._chained_fa_grads`, the `chained_fa_kp` arm on the
BPTT-viable LIF SNN `sim/bptt_snn_gpu`), but it NEVER emits cos(Y_l, W_{l+1}^T) -- so whether KP actually ALIGNS its
feedback on spikes (the mechanism) was inferred, never measured. This runner measures it.

THE MAPPING (rate G  <->  spiking Y). Both replace W^T in a SEQUENTIAL backward pass with a SEPARATE fixed-random matrix
that KP co-adapts by a LOCAL matched delta, transport-free:
  * rate  : e_l = (e_{l+1} @ G_l) . phi'(a_l) ;  KP: G_l -= step_l^T   (the SAME Adam step applied to W_l, transposed).
  * spikes: e_l = (e_{l+1} @ Y_l) . sigma'(v-theta)_l ;  KP: Y_l += kp_sign*lr_kp*(sum_t e_{l+1}^T @ z_l) / (B T)
    where z_l = the spikes of layer l (pre of the l->l+1 transition), sigma' = the atan surrogate, and the forward
    weight is updated W_{l+1} -= lr*wg_{l+1}. The eligibility trace eps_l = alpha_leak*eps_l + pre captures the LIF
    leak-recurrence (Bellec 2020 e-prop). TRANSPORT-FREE: Y_l is a separate random stream, the credit path computes
    e @ Y (never a forward W^T), and the KP update reads only post/pre activity + Y (never a forward W).

THE SIGN QUESTION (why kp_sign is a knob, not hard-coded). The rate rule DECREMENTS G by the same step W is decremented
by (`G -= step^T`), so G tracks W^T with the SAME sign -> cos rises. The committed spiking runner ADDS the KP outer
(`Y += lr*kp_lr*outer`) while the forward SUBTRACTS (`W -= lr*wg`), i.e. the OPPOSITE sign, which would make Y ANTI-track
W^T. This runner measures cos(Y,W^T) init->final for BOTH signs (kp_sign in {+1 = the committed runner, -1 = the
rate-matched decrement}) so the mechanism is read from the substrate, not assumed. A `kp_sign=+1` arm reproduces the
committed runner; `kp_sign=-1` tests the rate-matched sign.

THE ARMS (matched budget within a config; ALL on the SAME LIF SNN forward + task).
  * bptt        : surrogate-gradient BPTT ceiling (reuse `sim/bptt_snn_gpu.backward_unroll_xp`; uses W^T -- the
                  non-local reference the transport-free arms may NOT use).
  * frozen      : hidden layers FROZEN; only the output LIF layer learns (its local delta) -- the reservoir baseline.
  * fixed_fa    : chained transport-free FIXED-random FA (Y frozen at init) == the FREEZE-G LEVER ENDPOINT.
  * kp_plus     : chained transport-free KP-learned feedback, kp_sign=+1 (the committed runner's sign).
  * kp_minus    : chained transport-free KP-learned feedback, kp_sign=-1 (the rate-matched decrement sign).
  * permuted    : fixed_fa on SHUFFLED labels (anti-cheat floor -> must collapse to ~chance).

THE DECISIVE READS (per seed, on the inherit held-out set).
  Q1 mechanism/sign : does KP ALIGN its feedback on spikes? cos(Y_l, W_{l+1}^T) init -> final, per layer, per sign.
  Q2 credit         : does the ALIGNING KP beat fixed-DFA deep credit on spikes?  kp_acc - fixed_fa_acc.
  Q3 budget         : does more alignment budget (epochs, kp_lr) GROW the alignment / the KP-over-fixed margin?
  Q4 depth          : does KP revive deep credit at N=3 (redundant depth), where the committed runner read it DEAD
                      (both fixed_fa and kp collapsed to the permuted floor at N>=3)?
The freeze-G lever is intrinsic: fixed_fa == KP with feedback-learning OFF; (kp - fixed_fa) is the learned-feedback win.

ANTI-CHEATS (executed via tools.lab / Verdict, not asserted in prose):
  transport-free : max |cos(Y_l, W_{l+1}^T)| at INIT < 0.8 (separate random stream, not a W^T copy); Y never byte-equal
                   any forward W or its transpose (reuse `_no_weight_transport`); the KP update reads no forward weight.
  lever          : the aligning KP arm beats fixed_fa (freeze-G collapses it); KP moves Y, fixed_fa leaves Y frozen.
  permuted       : shuffled-label FA -> ~chance (no fit from a signal-free target).
  ceiling        : BPTT solves the task (a trainable-substrate ceiling exists), else the seed carries no depth info.

This is a SMOKE de-risk (seed 42 by default; --seeds for the self-aggregating sweep). NO sim/ edit -- the LIF SNN
forward + BPTT + atan surrogate are reuse-by-import from `sim/bptt_snn_gpu`; the task + eval helpers from the committed
runners; the chained credit + the cos(Y,W^T) read-out are RUNNER-side. SIM_BACKEND=numpy (tiny matmuls -> CPU).
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

# reuse-by-import: the BPTT-viable LIF SNN forward + surrogate BPTT + atan surrogate (NO sim/ edit)
from sim.bptt_snn_gpu import backward_unroll_xp, atan_surrogate  # noqa: E402
# reuse-by-import: the SAME LIF build/forward/eval helpers the 0.82 BPTT result used
from research.runners._snn_bptt_forward_vs_learning_isolation_derisk import (  # noqa: E402
    _softmax, _build_layers, _forward_logits, _accuracy)
# reuse-by-import: the depth-2 XOR->threshold RATE-OVERTURN task (NOT linearly reservoir-decodable) + the fixed-random
# feedback maker + the no-weight-transport anti-cheat (transport-free by construction)
from research.runners._gap4_bptt_snn_chained_fa_transport_free_derisk import (  # noqa: E402
    make_task_xor, _make_feedback, _no_weight_transport)

from tools.lab import lever, attributable_to, assert_backend, LeverError  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_gap4_onspikes_kp_align.json"


# ------------------------------------------------------------------------- the chained transport-free credit (+cos read)
def _chained_credit(inputs, layers, fs, output_grad, Y_list, alpha_leak, alpha_surr=2.0, sigma_norm=True,
                    train_hidden=True, learn_feedback=False, kp_sign=+1.0, kp_lr=0.2, kp_decay=1e-4, lr_fa=0.0):
    """CHAINED multi-hop transport-free feedback-alignment credit on the LIF SNN (reproduces the committed runner's
    `_chained_fa_grads` with an explicit kp_sign knob). Returns weight_grads (descent-side; caller subtracts lr*wg).

    train_hidden=False  -> FROZEN reservoir (only the output layer's local delta; hidden grads 0).
    learn_feedback=True -> KP: accumulate the transport-free outer (post = error at the layer above, pre = that
                           transition's input spikes) over T and apply ONE KP step to each Y_l IN PLACE:
                               Y_l += kp_sign*lr_fa*(kp_lr*outer - kp_decay*Y_l)
                           kp_sign=+1 reproduces the committed runner; kp_sign=-1 is the rate-matched decrement (Y
                           tracks W^T with the SAME sign W is updated with). NEVER reads a forward W (transport-free)."""
    T, Bn, _ = inputs.shape
    L = len(layers)
    spikes = fs["spikes"]; v_per = fs["v"]
    weight_grads = [np.zeros_like(l.W_in) for l in layers]
    eps = [np.zeros((Bn, l.W_in.shape[0]), dtype=l.W_in.dtype) for l in layers]
    kp_accum = [np.zeros_like(Y) for Y in Y_list] if learn_feedback else None
    for t in range(T):
        for li in range(L):
            pre = inputs[t] if li == 0 else spikes[li - 1][t]
            eps[li] = alpha_leak * eps[li] + pre
        psi = []
        for li in range(L):
            p = atan_surrogate(v_per[li][t] - layers[li].threshold, alpha=alpha_surr, xp=np)
            if sigma_norm:
                p = p / (p.mean() + 1e-9)
            psi.append(p)
        e_above = output_grad[t]
        weight_grads[L - 1] += eps[L - 1].T @ (e_above * psi[L - 1])
        if train_hidden:
            for li in range(L - 2, -1, -1):
                if kp_accum is not None:
                    kp_accum[li] += e_above.T @ spikes[li][t]     # post = error at li+1, pre = input spikes; NO W read
                e_below = (e_above @ Y_list[li]) * psi[li]        # TRANSPORT-FREE (Y not W)
                weight_grads[li] += eps[li].T @ e_below
                e_above = e_below
    if kp_accum is not None:
        denom = max(1, Bn * T)
        for li in range(len(Y_list)):
            outer = kp_accum[li] / denom
            Y_list[li] = Y_list[li] + kp_sign * lr_fa * (float(kp_lr) * outer - float(kp_decay) * Y_list[li])
    return weight_grads


def _bw_cos(Y_list, layers):
    """cos(Y_l, W_{l+1}^T) per hidden transition -- the transport-free feedback-alignment signature (the rate finding's
    central read). Y_list[li] (shape (sizes[li+2], sizes[li+1])) replaces W_{li+1}^T (layers[li+1].W_in.T, same shape)."""
    out = []
    for li in range(len(Y_list)):
        y = Y_list[li].ravel()
        wt = layers[li + 1].W_in.T.ravel()
        ny, nw = np.linalg.norm(y), np.linalg.norm(wt)
        out.append(float(y @ wt / (ny * nw)) if (ny > 1e-12 and nw > 1e-12) else 0.0)
    return out


# ------------------------------------------------------------------------------------------------ one arm
def _train_arm(Xtr, ytr, sizes, T, epochs, lr, lr_fa, in_gain, seed, mode, batch=32, sigma_norm=True,
               kp_sign=+1.0, kp_lr=0.2, kp_decay=1e-4):
    """mode in {bptt, frozen, fixed_fa, kp}. Returns (layers, Y_list, Y0, W0_hidden, bw_init, bw_final)."""
    rng = np.random.default_rng(seed + 1)
    w_scales = [2.5] + [1.0] * (len(sizes) - 2)
    layers = _build_layers(sizes, T, rng, w_scales)
    k = sizes[-1]; n = len(Xtr)
    Y_list = _make_feedback(sizes, seed) if mode in ("frozen", "fixed_fa", "kp") else None
    bw_init = _bw_cos(Y_list, layers) if Y_list is not None else []
    Y0 = [Y.copy() for Y in Y_list] if Y_list is not None else None
    alpha_leak = layers[0].leak
    for _ in range(epochs):
        perm = rng.permutation(n)
        for b0 in range(0, n, batch):
            bi = perm[b0:b0 + batch]
            Xb, yb = Xtr[bi], ytr[bi]
            logits, fs, inp = _forward_logits(Xb, layers, T, in_gain)
            p = _softmax(logits)
            delta = p.copy(); delta[np.arange(len(yb)), yb] -= 1.0
            og = np.repeat((delta / T)[None, :, :], T, axis=0).astype(np.float64)
            if mode == "bptt":
                wg, _ = backward_unroll_xp(inp, layers, fs, og, alpha=2.0, xp=np)
                use_lr = lr
            else:
                train_hidden = mode in ("fixed_fa", "kp")
                learn_feedback = (mode == "kp")
                wg = _chained_credit(inp, layers, fs, og, Y_list, alpha_leak, alpha_surr=2.0, sigma_norm=sigma_norm,
                                     train_hidden=train_hidden, learn_feedback=learn_feedback, kp_sign=kp_sign,
                                     kp_lr=kp_lr, kp_decay=kp_decay, lr_fa=lr_fa)
                use_lr = lr_fa
            for li in range(len(layers)):
                arm_lr = lr if (li == len(layers) - 1) else use_lr
                layers[li].W_in -= arm_lr * (wg[li] / len(bi))
    bw_final = _bw_cos(Y_list, layers) if Y_list is not None else []
    return layers, Y_list, Y0, bw_init, bw_final


def _feedback_moved(Y_list, Y0):
    if Y_list is None or Y0 is None:
        return None
    return any(not np.array_equal(Y_list[i], Y0[i]) for i in range(len(Y_list)))


# ------------------------------------------------------------------------------------------------ one seed
def run_seed(seed, hidden, T, epochs, lr, lr_fa, in_gain, subsample, n_hidden_layers, sigma_norm, kp_lr, kp_decay,
             bptt_epochs, bptt_hidden):
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_xor(seed)
    k = meta["k_classes"]; n_in = Xtr.shape[1]; inh_idx = idx["inh_idx"]
    yv = yte[inh_idx]
    chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    if subsample is not None and len(Xtr) > subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:subsample]
        Xtr, ytr = Xtr[keep], ytr[keep]
    sizes = [n_in] + [hidden] * n_hidden_layers + [k]
    bptt_sizes = [n_in] + [bptt_hidden] * n_hidden_layers + [k]     # BPTT ceiling gets its own (wider) capacity so the
    #                                                                 ceiling is VALID (XOR needs a wider net than the
    #                                                                 chained arms' hidden=32; the committed runner used 128)

    def acc(layers):
        return _accuracy(Xte, yte, layers, T, in_gain, sub=inh_idx)

    # ---- BPTT ceiling (own capacity) ----
    bl, _, _, _, _ = _train_arm(Xtr, ytr, bptt_sizes, T, bptt_epochs, lr, lr_fa, in_gain, seed, "bptt",
                                sigma_norm=sigma_norm)
    bptt_acc = acc(bl)
    # ---- frozen reservoir ----
    fr, _, _, _, _ = _train_arm(Xtr, ytr, sizes, T, epochs, lr, lr_fa, in_gain, seed, "frozen", sigma_norm=sigma_norm)
    frozen_acc = acc(fr)
    # ---- fixed_fa (frozen Y = the freeze-G lever endpoint) ----
    fal, faY, faY0, fa_bw_i, fa_bw_f = _train_arm(Xtr, ytr, sizes, T, epochs, lr, lr_fa, in_gain, seed, "fixed_fa",
                                                  sigma_norm=sigma_norm)
    fixed_fa_acc = acc(fal)
    fa_moved = _feedback_moved(faY, faY0)
    # ---- kp_plus (committed runner's sign) ----
    kpl_p, kpY_p, kpY0_p, kp_bw_i_p, kp_bw_f_p = _train_arm(Xtr, ytr, sizes, T, epochs, lr, lr_fa, in_gain, seed, "kp",
                                                            sigma_norm=sigma_norm, kp_sign=+1.0, kp_lr=kp_lr,
                                                            kp_decay=kp_decay)
    kp_plus_acc = acc(kpl_p); kp_plus_moved = _feedback_moved(kpY_p, kpY0_p)
    # ---- kp_minus (rate-matched decrement sign) ----
    kpl_m, kpY_m, kpY0_m, kp_bw_i_m, kp_bw_f_m = _train_arm(Xtr, ytr, sizes, T, epochs, lr, lr_fa, in_gain, seed, "kp",
                                                            sigma_norm=sigma_norm, kp_sign=-1.0, kp_lr=kp_lr,
                                                            kp_decay=kp_decay)
    kp_minus_acc = acc(kpl_m); kp_minus_moved = _feedback_moved(kpY_m, kpY0_m)
    # ---- permuted (anti-cheat floor) ----
    prng = np.random.default_rng(seed + 555)
    yperm = ytr[prng.permutation(len(ytr))]
    pel, _, _, _, _ = _train_arm(Xtr, yperm, sizes, T, epochs, lr, lr_fa, in_gain, seed, "fixed_fa",
                                 sigma_norm=sigma_norm)
    permuted_acc = acc(pel)

    # deepest transition index (a1-equivalent = the deepest-from-output hidden feedback), and the output-adjacent one
    deep = 0; adj = len(kp_bw_f_p) - 1
    # transport-free init check: max |cos| at init across arms (all use the SAME _make_feedback stream)
    bw_init_max = float(max(abs(c) for c in kp_bw_i_p)) if kp_bw_i_p else float("nan")
    no_transport = bool(_no_weight_transport(kpY_p, kpl_p) and _no_weight_transport(kpY_m, kpl_m)
                        and _no_weight_transport(faY, fal))

    def _delta_align(bw_i, bw_f):
        return [float(bw_f[i] - bw_i[i]) for i in range(len(bw_f))] if bw_f else []

    return {
        "seed": int(seed), "n_hidden_layers": int(n_hidden_layers), "epochs": int(epochs), "kp_lr": float(kp_lr),
        "sizes": sizes, "chance": chance,
        "bptt_acc": bptt_acc, "frozen_acc": frozen_acc, "fixed_fa_acc": fixed_fa_acc,
        "kp_plus_acc": kp_plus_acc, "kp_minus_acc": kp_minus_acc, "permuted_acc": permuted_acc,
        # the KP-over-fixed-DFA margin per sign (the learned>fixed win, on spikes)
        "kp_plus_over_fixed": float(kp_plus_acc - fixed_fa_acc),
        "kp_minus_over_fixed": float(kp_minus_acc - fixed_fa_acc),
        "kp_plus_over_frozen": float(kp_plus_acc - frozen_acc),
        "kp_minus_over_frozen": float(kp_minus_acc - frozen_acc),
        # the transport-free ALIGNMENT signature cos(Y_l, W_{l+1}^T) init -> final, per sign, per layer
        "fixed_fa_bw_init": fa_bw_i, "fixed_fa_bw_final": fa_bw_f,
        "kp_plus_bw_init": kp_bw_i_p, "kp_plus_bw_final": kp_bw_f_p,
        "kp_minus_bw_init": kp_bw_i_m, "kp_minus_bw_final": kp_bw_f_m,
        "kp_plus_bw_delta": _delta_align(kp_bw_i_p, kp_bw_f_p),
        "kp_minus_bw_delta": _delta_align(kp_bw_i_m, kp_bw_f_m),
        # deep-layer (a1) and output-adjacent alignment finals (the rate finding's headline layers)
        "kp_plus_bw_deep_init": (kp_bw_i_p[deep] if kp_bw_i_p else float("nan")),
        "kp_plus_bw_deep_final": (kp_bw_f_p[deep] if kp_bw_f_p else float("nan")),
        "kp_minus_bw_deep_init": (kp_bw_i_m[deep] if kp_bw_i_m else float("nan")),
        "kp_minus_bw_deep_final": (kp_bw_f_m[deep] if kp_bw_f_m else float("nan")),
        "kp_plus_bw_adj_final": (kp_bw_f_p[adj] if kp_bw_f_p else float("nan")),
        "kp_minus_bw_adj_final": (kp_bw_f_m[adj] if kp_bw_f_m else float("nan")),
        "bw_init_max_abs": bw_init_max,
        # levers
        "fa_moved": fa_moved, "kp_plus_moved": kp_plus_moved, "kp_minus_moved": kp_minus_moved,
        "no_transport": no_transport,
        # permuted must NOT EXCEED chance (label-shuffle carries no signal). BELOW-chance is FINE and even stronger --
        # "chance" here is the majority-class rate, so a uniform-guess permuted net sits slightly below it.
        "permuted_near_chance": bool(permuted_acc <= chance + 0.06),
        "bptt_solves": bool(bptt_acc > chance + 0.15),
    }


def _agg(rows):
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {}

    def m(key):
        vals = [r[key] for r in ok if isinstance(r.get(key), (int, float)) and not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")
    n = len(ok)
    # which sign aligns: the one whose deep-layer cos RISES the most (final - init)
    kp_plus_deep_delta = m("kp_plus_bw_deep_final") - m("kp_plus_bw_deep_init")
    kp_minus_deep_delta = m("kp_minus_bw_deep_final") - m("kp_minus_bw_deep_init")
    aligning = "kp_minus" if kp_minus_deep_delta > kp_plus_deep_delta else "kp_plus"
    return {
        "n_seeds": n, "mean_chance": m("chance"),
        "mean_bptt_acc": m("bptt_acc"), "mean_frozen_acc": m("frozen_acc"),
        "mean_fixed_fa_acc": m("fixed_fa_acc"),
        "mean_kp_plus_acc": m("kp_plus_acc"), "mean_kp_minus_acc": m("kp_minus_acc"),
        "mean_permuted_acc": m("permuted_acc"),
        "mean_kp_plus_over_fixed": m("kp_plus_over_fixed"), "mean_kp_minus_over_fixed": m("kp_minus_over_fixed"),
        "mean_kp_plus_over_frozen": m("kp_plus_over_frozen"), "mean_kp_minus_over_frozen": m("kp_minus_over_frozen"),
        # alignment signature (deep layer a1)
        "mean_kp_plus_bw_deep_init": m("kp_plus_bw_deep_init"), "mean_kp_plus_bw_deep_final": m("kp_plus_bw_deep_final"),
        "mean_kp_minus_bw_deep_init": m("kp_minus_bw_deep_init"),
        "mean_kp_minus_bw_deep_final": m("kp_minus_bw_deep_final"),
        "kp_plus_deep_align_delta": float(kp_plus_deep_delta), "kp_minus_deep_align_delta": float(kp_minus_deep_delta),
        "mean_kp_plus_bw_adj_final": m("kp_plus_bw_adj_final"), "mean_kp_minus_bw_adj_final": m("kp_minus_bw_adj_final"),
        "aligning_sign": aligning, "mean_bw_init_max_abs": m("bw_init_max_abs"),
        "bptt_solves_seeds": f"{sum(bool(r.get('bptt_solves')) for r in ok)}/{n}",
        "permuted_near_chance_seeds": f"{sum(bool(r.get('permuted_near_chance')) for r in ok)}/{n}",
        "no_transport_all": bool(all(r.get("no_transport") for r in ok)),
        "fa_frozen_all": bool(all(r.get("fa_moved") is False for r in ok)),
        "kp_plus_moved_all": bool(all(r.get("kp_plus_moved") for r in ok)),
        "kp_minus_moved_all": bool(all(r.get("kp_minus_moved") for r in ok)),
    }


def _evaluate(agg):
    """Verdict on the DEPLOYED KP rule (kp_plus, the committed spiking runner's sign): does it both ALIGN (cos rises)
    AND beat fixed-DFA on spikes, transport-free, with the anti-cheats holding? kp_plus is gated as PRIMARY (it is the
    deployed rule, NOT selected for winning); kp_minus (the rate-matched decrement sign) is reported as a sign-probe.
    A NO-GO here is an honest gap#4 deliverable (maps the residual)."""
    if not agg:
        return {"status": "ERROR"}
    aligning = "kp_plus"                                            # the DEPLOYED sign, gated as primary (not cherry-picked)
    kp_over_fixed = agg[f"mean_{aligning}_over_fixed"]
    deep_delta = agg[f"{aligning}_deep_align_delta"]
    other = "kp_minus"
    # lever: the aligning KP vs fixed_fa (freeze-G endpoint). If the arms are IDENTICAL (KP == fixed_fa), the lever does
    # NOT move -> LeverError. That is NOT a crash to hide: at a REDUNDANT depth the deep credit is dead so KP == fixed ==
    # the permuted floor, and the unmoved lever IS the residual (KP cannot revive dead deep credit). Record it as a NULL
    # lever -> the verdict becomes UNDEFINED (the manipulation had no measurable effect), never a fabricated GO/NO-GO.
    lever_moved = True
    try:
        lever("KP learned feedback ON (aligning sign) vs fixed_fa (freeze-G) -- spiking deep-credit accuracy",
              round(agg["mean_fixed_fa_acc"], 4), round(agg[f"mean_{aligning}_acc"], 4),
              continuous="acc: KP(%s) %.3f vs fixed_fa %.3f | deep cos(Y,W^T) init %.3f -> final %.3f (delta %+.3f)"
              % (aligning, agg[f"mean_{aligning}_acc"], agg["mean_fixed_fa_acc"],
                 agg[f"mean_{aligning}_bw_deep_init"], agg[f"mean_{aligning}_bw_deep_final"], deep_delta))
        attributable_to("spiking deep-credit accuracy attributable to LEARNING the feedback (aligning KP vs fixed_fa)",
                        treatment_value=agg[f"mean_{aligning}_acc"], control_value=agg["mean_fixed_fa_acc"])
    except LeverError:
        lever_moved = False

    v = Verdict("gap4_onspikes_kp_aligns_and_beats_fixed")
    v.require("lever_kp_changes_accuracy", bool(lever_moved), expect=True,
              note="the KP-vs-fixed_fa accuracy lever MOVED. If it did NOT (KP == fixed_fa == the permuted floor), the "
                   "deep credit is DEAD at this depth and no verdict is earned -> UNDEFINED, the honest residual "
                   "(KP cannot revive dead deep credit; e.g. a REDUNDANT-depth-3 XOR).")
    v.require("bptt_ceiling_exists", agg["bptt_solves_seeds"].split("/")[0] == agg["bptt_solves_seeds"].split("/")[1],
              expect=True, note="BPTT solves the task on all seeds (%s): a trainable-substrate ceiling exists"
              % agg["bptt_solves_seeds"])
    v.require("permuted_at_chance", agg["permuted_near_chance_seeds"].split("/")[0]
              == agg["permuted_near_chance_seeds"].split("/")[1], expect=True,
              note="permuted-label FA sits at chance on all seeds (%s): the signal is label-attributable"
              % agg["permuted_near_chance_seeds"])
    v.require("transport_free_not_a_copy", bool(agg["mean_bw_init_max_abs"] < 0.8), expect=True,
              note="max |cos(Y_l, W_{l+1}^T)| at INIT = %.3f (< 0.8 -> separate random stream, not a W^T copy)"
              % agg["mean_bw_init_max_abs"])
    v.require("transport_free_no_byte_copy", bool(agg["no_transport_all"]), expect=True,
              note="no Y_l is byte-equal any forward W or its transpose, all seeds")
    v.require("lever_kp_moves_feedback", bool(agg["%s_moved_all" % aligning]) and bool(agg["fa_frozen_all"]),
              expect=True, note="the aligning KP moved Y every seed; fixed_fa left Y frozen (freeze-G lever)")
    v.require("kp_aligns_on_spikes", bool(deep_delta > 0.05), expect=True,
              note="the aligning KP's deep-layer cos(Y,W^T) RISES by %+.3f init->final (the transport-free alignment "
                   "signature on spikes; the other sign delta = %+.3f)" % (deep_delta, agg["%s_deep_align_delta" % other]))
    v.require("kp_beats_fixed_dfa", bool(kp_over_fixed > 0.02), expect=True,
              note="the aligning KP beats fixed-DFA by %+.3f accuracy on spikes (learned feedback improves deep credit)"
              % kp_over_fixed)
    v.control("kp_acc_differs_from_fixed", treatment=agg[f"mean_{aligning}_acc"], control=agg["mean_fixed_fa_acc"],
              min_separation=1e-6, note="the aligning-KP accuracy must differ from fixed-DFA (the manipulation landed)")

    decided = v.decide(bool(deep_delta > 0.05 and kp_over_fixed > 0.02))
    return {"status": decided["status"], "go": bool(decided["status"] == "GO"),
            "preconditions": decided["preconditions"], "undefined_reasons": decided["undefined_reasons"],
            "aligning_sign": aligning, "kp_over_fixed": float(kp_over_fixed), "deep_align_delta": float(deep_delta)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-hidden-layers", type=int, default=2)
    ap.add_argument("--timesteps", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=200, help="matched budget for the transport-free arms. The rate "
                    "finding showed KP feedback-alignment converges SLOWER than backprop (under-trains at low budget); "
                    "raise this to test whether more alignment budget grows the KP-over-fixed margin on spikes.")
    ap.add_argument("--bptt-epochs", type=int, default=200, help="BPTT ceiling arm epochs.")
    ap.add_argument("--bptt-hidden", type=int, default=96, help="BPTT ceiling arm width (own capacity, wider than the "
                    "chained arms so the ceiling is VALID -- XOR needs a wider net than hidden=32).")
    ap.add_argument("--lr", type=float, default=0.05, help="BPTT + output-layer local-delta lr")
    ap.add_argument("--lr-fa", type=float, default=None, help="chained-FA hidden lr (default = --lr)")
    ap.add_argument("--in-gain", type=float, default=1.0)
    ap.add_argument("--sigma-norm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--kp-lr", type=float, default=0.2)
    ap.add_argument("--kp-decay", type=float, default=1e-4)
    ap.add_argument("--train-subsample", type=int, default=800)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    lr_fa = args.lr if args.lr_fa is None else args.lr_fa
    backend = assert_backend(os.environ.get("SIM_BACKEND", "numpy"))
    device = "cpu" if backend == "numpy" else "cuda"

    t0 = time.time()
    rows = []
    for sd in args.seeds:
        try:
            r = run_seed(sd, args.hidden, args.timesteps, args.epochs, args.lr, lr_fa, args.in_gain,
                         args.train_subsample, args.n_hidden_layers, args.sigma_norm, args.kp_lr, args.kp_decay,
                         args.bptt_epochs, args.bptt_hidden)
        except Exception as e:
            r = {"seed": sd, "error": repr(e), "traceback": traceback.format_exc()}
        rows.append(r)
        if "error" not in r:
            print("[seed %d N=%dh ep=%d kp_lr=%.2f] fixed_fa=%.3f kp+=%.3f kp-=%.3f | frozen=%.3f perm=%.3f bptt=%.3f "
                  "chance=%.3f | deep cos(Y,W^T) kp+ %.3f->%.3f  kp- %.3f->%.3f | kp+_over_fixed %+.3f kp-_over_fixed "
                  "%+.3f | no_transport=%s" % (
                      r["seed"], r["n_hidden_layers"], r["epochs"], r["kp_lr"], r["fixed_fa_acc"], r["kp_plus_acc"],
                      r["kp_minus_acc"], r["frozen_acc"], r["permuted_acc"], r["bptt_acc"], r["chance"],
                      r["kp_plus_bw_deep_init"], r["kp_plus_bw_deep_final"], r["kp_minus_bw_deep_init"],
                      r["kp_minus_bw_deep_final"], r["kp_plus_over_fixed"], r["kp_minus_over_fixed"], r["no_transport"]))
        else:
            print("[seed %d] ERROR %s" % (sd, r["error"]))

    agg = _agg(rows)
    ev = _evaluate(agg)
    out = {"probe": "gap4_onspikes_kp_align", "task": "depth2_xor_threshold", "seeds": args.seeds,
           "backend": backend, "device": device,
           "config": {"hidden": args.hidden, "n_hidden_layers": args.n_hidden_layers, "T": args.timesteps,
                      "epochs": args.epochs, "bptt_epochs": args.bptt_epochs, "bptt_hidden": args.bptt_hidden,
                      "lr": args.lr, "lr_fa": lr_fa,
                      "in_gain": args.in_gain, "sigma_norm": args.sigma_norm, "kp_lr": args.kp_lr,
                      "kp_decay": args.kp_decay, "train_subsample": args.train_subsample},
           "elapsed_seconds": round(time.time() - t0, 1), "rows": rows, "aggregate": agg, "result": ev}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    if agg:
        print("\n  AGG (%d seeds): fixed_fa=%.3f  kp+=%.3f  kp-=%.3f | frozen=%.3f perm=%.3f bptt=%.3f chance=%.3f"
              % (agg["n_seeds"], agg["mean_fixed_fa_acc"], agg["mean_kp_plus_acc"], agg["mean_kp_minus_acc"],
                 agg["mean_frozen_acc"], agg["mean_permuted_acc"], agg["mean_bptt_acc"], agg["mean_chance"]))
        print("  ALIGNMENT deep cos(Y,W^T): kp+ %.3f->%.3f (delta %+.3f)  kp- %.3f->%.3f (delta %+.3f) => ALIGNING SIGN = %s"
              % (agg["mean_kp_plus_bw_deep_init"], agg["mean_kp_plus_bw_deep_final"], agg["kp_plus_deep_align_delta"],
                 agg["mean_kp_minus_bw_deep_init"], agg["mean_kp_minus_bw_deep_final"], agg["kp_minus_deep_align_delta"],
                 agg["aligning_sign"]))
        print("  KP-over-fixed-DFA: kp+ %+.3f  kp- %+.3f  | transport-free init max|cos| %.3f (<0.8), no_transport=%s"
              % (agg["mean_kp_plus_over_fixed"], agg["mean_kp_minus_over_fixed"], agg["mean_bw_init_max_abs"],
                 agg["no_transport_all"]))
    print("\n[onspikes-kp-align] status=%s  aligning_sign=%s  kp_over_fixed=%+.3f  deep_align_delta=%+.3f  wrote %s"
          % (ev.get("status"), ev.get("aligning_sign"), ev.get("kp_over_fixed", float("nan")),
             ev.get("deep_align_delta", float("nan")), args.out))


if __name__ == "__main__":
    main()
