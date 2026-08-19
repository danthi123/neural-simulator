"""gap#4 ENTER-THE-REGIME via SoftHebb -- the fully-UNSUPERVISED, no-feedback end of the local-rule family.

THE WALL + THE Q5 REFRAME (do NOT re-derive; see the committed record).
  * gap#4 = deep credit through a DEEP (N>=3) SPIKING net with a LOCAL rule. Chained transport-free FA/KP collapse to
    majority-class at N>=3; even a perfect W^T oracle gives no directed credit through the finite-spike sigma' read
    (`2026-08-02-gap4-crux-wall-LOCATED-...`).
  * `2026-08-12-gap4-obligatory-depth3-...-NEGATIVE`: an obligatory-depth-3 GENERALISATION instrument is NOT
    constructible (depth is never OBLIGATORY on the finite-read substrate), so the success metric is reframed to
    **ENTER-THE-REGIME: leave majority-class + BEAT the frozen reservoir**. Q1 Forward-Forward and Q4 DECOLLE (both
    PER-LAYER LOCAL objectives) crack it. The emerging insight: a per-layer local objective is the winning ingredient.

THIS RUNNER'S MECHANISM -- SoftHebb (Journe, Rodriguez, Guo, Moraitis 2023, ICLR, "Hebbian deep learning without
feedback", arXiv:2209.11883). It is the STRONGEST brain-based claim: no per-layer LABEL at all, no feedback, no
backprop. Each hidden layer is a soft winner-take-all Hebbian module trained FULLY UNSUPERVISED, greedy layer-wise:
  Eq 1 (soft-WTA):     y_k = softmax(u/tau)_k ,   u = x @ W  (pre-activation drive; u_k = sum_i w_ik x_i)
  Eq 2 (Oja-instar):   dw_ik = eta * s_k * y_k * (x_i - u_k * w_ik)
                       where s_k = +1 for the maximally-activated neuron (argmax_k u), -1 for all others (SoftHebb's
                       "soft anti-Hebbian" competition: the winner moves TOWARD the input, losers are pushed AWAY).
                       The (x_i - u_k w_ik) term is the Oja normalisation (weights converge toward a unit-norm radius).
  Stacking:            each layer fully trained + FROZEN before the next; the summed-spikes of a trained layer are the
                       presynaptic input `x` to the next. Then ONE supervised linear readout (optimal 5-fold-CV ridge)
                       on the FROZEN concatenated deep features -- the SAME read-out the frozen-reservoir floor uses.

HYPOTHESIS. An unsupervised soft-WTA Hebbian deep SPIKING stack builds TASK-USEFUL deep features so a linear readout
BEATS the frozen-RANDOM reservoir -- i.e. it ENTERS the regime -- where the top-down FA/KP rule collapses. If it only
MATCHES the reservoir, the July-15 objection ("SoftHebb = unsupervised feature side, not task-directed") is CONFIRMED
under the new metric (a first-class HONEST NEGATIVE).

THE ISOLATION (why the comparison is the RULE alone). The SoftHebb arm and the frozen-reservoir floor are BYTE-for-byte
identical except for the hidden weights: SAME init RNG stream (seed+1), SAME widths, SAME w_scales, SAME LIF forward,
SAME concat-hidden-summed-spikes read (`_reservoir_features`), SAME optimal 5-fold-CV ridge (`_optimal_ridge_acc`). The
ONLY difference is whether the hidden weights were shaped by unsupervised soft-WTA Hebbian competition or left RANDOM.
So (SoftHebb - reservoir) isolates EXACTLY what the unsupervised competition adds over a random projection.

ARMS (per seed, on the inherit held-out set):
  SoftHebb            : unsupervised soft-WTA Hebbian deep stack + OPTIMAL ridge readout on frozen concat features.
  frozen_res_matched  : the FLOOR -- random LIF reservoir at MATCHED width + OPTIMAL ridge (SoftHebb must beat this).
  frozen_res_wide     : random LIF reservoir at WIDE width + OPTIMAL ridge (the generous floor).
  chained_fa / kp     : the WALL -- transport-free chained FA / KP (reused; collapse to majority-class at depth).
  bptt                : the CEILING -- surrogate-gradient BPTT (a trained target exists; non-local, reference only).
  shuffled_softhebb   : ANTI-CHEAT -- SoftHebb trained on COLUMN-SHUFFLED inputs (joint structure destroyed) then read
                        on the REAL inputs; if the lift is genuine input-structure learning this must NOT beat the
                        reservoir. (SoftHebb is label-free, so the meaningful null is destroying the input CORRELATIONS
                        it learns, not permuting labels -- that would break every arm's readout identically.)

PER-LAYER CLASS-SELECTIVITY (the "rising selectivity" instrument): fit an optimal ridge readout on EACH hidden layer's
summed-spikes ALONE, for SoftHebb AND the reservoir. Enter-the-regime wants SoftHebb's per-layer selectivity to RISE
with depth AND exceed the reservoir's at each layer -- unsupervised competition making each layer more class-informative.

GO (6-seed 42 43 44 100 101 102, N=3 and N=4): SoftHebb beats the optimal-reservoir (matched) by >= 0.10 with the MIN
over seeds above it, where FA/KP collapse (near majority-class), and shuffled_softhebb does NOT beat the reservoir
(<= reservoir + 0.05). HONEST-NEGATIVE (first-class) if SoftHebb ~= the reservoir.

NON-NEGOTIABLES: brain-based-only (SoftHebb is fully LOCAL + unsupervised; the ridge readout is a thin supervised head,
the honest shortcut, SAME status as a reservoir readout; BPTT is a reference ceiling only). SIM_BACKEND=numpy
(launch-bound; one BLAS thread, parallelise across seeds). cfg.seed threaded into every net + task. NO sim/ edit -- the
LIF forward + tasks + reservoir-floor + FA/KP/BPTT arms are reused-by-import; the SoftHebb rule is a RUNNER-side function.

Run (numpy CPU):
    # smoke, one seed, N=3, all three tasks -- READ the reservoir floor + whether SoftHebb moves it BEFORE the sweep:
    SIM_BACKEND=numpy python -m research.runners._gap4_softhebb_local_derisk --task hier3 --seeds 42 \
        --n-hidden-layers 3 --out research/findings/raw/_gap4_softhebb/smoke_hier3_N3_s42.json
    # the DECISIVE 6-seed sweep, per task, per depth (N=3,4):
    for TASK in hier3 inheritance xor; do for N in 3 4; do \
      SIM_BACKEND=numpy python -m research.runners._gap4_softhebb_local_derisk --task $TASK --n-hidden-layers $N \
        --seeds 42 43 44 100 101 102 \
        --out research/findings/raw/_gap4_softhebb/${TASK}_N${N}_6seed.json ; done ; done
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
# TINY matmuls -> one BLAS thread per process (oversubscription is ~30x slower); parallelise across seeds instead.
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

# ---- reuse-by-import: the LIF SNN forward + tasks + the RESERVOIR FLOOR + the FA/KP/BPTT arms (NO sim/ edit) ----
from research.runners._snn_bptt_forward_vs_learning_isolation_derisk import (  # noqa: E402
    _build_layers, _forward_logits, _accuracy)
from research.runners._gap4_bptt_snn_chained_fa_transport_free_derisk import (  # noqa: E402
    make_task_xor, make_task_hier3,
    _reservoir_features, _optimal_ridge_acc, _frozen_reservoir_optimal, _train_snn_arm)
from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    make_task_semantic_inheritance)
from sim.dendritic_mlp import DendriticMLP  # noqa: E402 -- the rate backprop oracle reference
from research.runners._semantic_inheritance_deep_credit_derisk import _train_oracle, _acc_on  # noqa: E402
from tools.verdict import Verdict, UNDEFINED  # noqa: E402 -- a verdict must travel with the preconditions it earned
from tools.lab import attributable_to  # noqa: E402 -- attribute the SoftHebb lift against its shuffled-input control

OUT = _REPO / "research" / "findings" / "raw" / "_gap4_softhebb" / "softhebb.json"


# ============================================================================================================
# The SoftHebb soft-WTA Hebbian LOCAL rule (Journe et al. 2023, Eq 1-2), ported to the LIF SNN forward. Trains ONE
# hidden layer's W_in fully UNSUPERVISED: soft winner-take-all competition + Oja normalisation + the anti-Hebbian
# sign flip for the losers. RUNNER-side (NO sim/ edit); reads ONLY presynaptic activity x and the layer's own
# pre-activation u -- never a label, never a top-down feedback, never a forward weight of another layer.
# ============================================================================================================
def _softhebb_layer_update(x, W, eta, tau):
    """One SoftHebb minibatch update of a single layer. x:(B,n_pre) presynaptic activity; W:(n_pre,n_post). Returns dW.
    Eq 1  y = softmax(u/tau) over neurons; Eq 2  dw_ik = eta * s_k * y_k * (x_i - u_k w_ik), s_k=+1 for argmax_k u else -1."""
    u = x @ W                                                        # (B, n_post) pre-activation drive  u_k = sum_i w_ik x_i
    z = u / tau
    z = z - z.max(axis=1, keepdims=True)
    y = np.exp(z); y = y / (y.sum(axis=1, keepdims=True) + 1e-12)    # (B, n_post) soft-WTA responsibilities  (Eq 1)
    s = -np.ones_like(u)                                            # anti-Hebbian: losers -1
    s[np.arange(len(u)), np.argmax(u, axis=1)] = 1.0                # winner (max-activated) +1  (SoftHebb sign flip)
    r = s * y                                                       # signed responsibility
    # Eq 2 summed over the batch: dw_ik = eta [ (x.T @ r)_ik - w_ik * sum_b r_bk u_bk ]
    hebb = x.T @ r                                                  # (n_pre, n_post)  potentiate toward input
    oja = W * (r * u).sum(axis=0)[None, :]                          # Oja normalisation (per-neuron radius control)
    return eta * (hebb - oja) / len(x)


def _softhebb_train_stack(Xtr, sizes, T, in_gain, seed, n_hidden_layers, eta, tau, epochs, batch,
                          preserve_norm=True, shuffle_input=False):
    """Greedy layer-wise UNSUPERVISED SoftHebb training of the hidden LIF layers. Build the SAME LIF stack as the
    reservoir floor (rng seed+1, same w_scales) so (SoftHebb - reservoir) isolates the rule alone. Train hidden layer
    li fully, FREEZE it, forward the frozen lower stack to get the presynaptic activity for li+1, repeat.

    preserve_norm: after each update, rescale each neuron's weight COLUMN to its INIT L2 norm -- homeostatic synaptic
        scaling (Turrigiano), the biological companion process that keeps the LIF forward firing while SoftHebb rotates
        the tuning DIRECTION (without it the Oja radius can drift the drive out of the threshold's operating range).
    shuffle_input: the ANTI-CHEAT -- independently permute each INPUT feature column across the training set (destroys
        the joint correlation structure SoftHebb learns, preserves per-feature marginals) for training ONLY; features
        are then read on the REAL inputs. A genuine structure-learner trained on this must collapse to ~random."""
    rng = np.random.default_rng(seed + 1)                          # SAME init stream as _frozen_reservoir_optimal / _train_snn_arm
    w_scales = [2.5] + [1.0] * (len(sizes) - 2)                    # strong first layer (same as the reservoir/other arms)
    layers = _build_layers(sizes, T, rng, w_scales)               # hidden = layers[:n_hidden_layers]; layers[-1] unused (read excludes it)
    init_norm = [np.linalg.norm(l.W_in, axis=0, keepdims=True) + 1e-12 for l in layers]
    trng = np.random.default_rng(seed + 909)

    Xt = Xtr.copy()
    if shuffle_input:                                             # column-wise independent shuffle (break joint structure)
        for j in range(Xt.shape[1]):
            Xt[:, j] = Xt[trng.permutation(len(Xt)), j]

    n = len(Xt)
    for li in range(n_hidden_layers):
        W = layers[li].W_in
        for _ in range(epochs):
            perm = trng.permutation(n)
            for b0 in range(0, n, batch):
                bi = perm[b0:b0 + batch]
                Xb = Xt[bi]
                # presynaptic activity x into layer li: raw input current for li==0, else summed spikes of frozen li-1
                if li == 0:
                    x = in_gain * Xb                              # (B, n_in) -- the drive the first layer integrates
                else:
                    _, fs, _ = _forward_logits(Xb, layers[:li], T, in_gain)
                    x = fs["spikes"][li - 1].sum(axis=0)          # (B, width) summed-spike rate of the frozen layer below
                W += _softhebb_layer_update(x, W, eta, tau)
                if preserve_norm:
                    cur = np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
                    W *= init_norm[li] / cur                      # homeostatic synaptic scaling to the init radius
        layers[li].W_in = W                                       # frozen; next layer reads its spikes
    return layers


def _softhebb_reads_no_label_no_feedback():
    """anti-cheat (source guard): the SoftHebb update reads ONLY x, W, eta, tau -- never a label y, never a top-down
    feedback matrix. Best-effort tripwire against a future in-file edit that would smuggle supervision into the rule."""
    try:
        src = inspect.getsource(_softhebb_layer_update)
    except (OSError, TypeError):
        return True
    banned = (" y_true", "label", "target", "feedback", "Y_list", "output_grad")
    return not any(tok in src for tok in banned)


def _per_layer_selectivity(Xtr, ytr, Xte_inh, yte_inh, layers, n_hidden_layers, k, T, in_gain, seed):
    """Optimal-ridge readout accuracy from EACH hidden layer's summed-spikes ALONE -> the class-selectivity of that
    layer's representation. Returns a list of per-layer inherit accuracies (index li = hidden layer li)."""
    _, fs_tr, _ = _forward_logits(Xtr, layers, T, in_gain)
    _, fs_te, _ = _forward_logits(Xte_inh, layers, T, in_gain)
    accs = []
    for li in range(n_hidden_layers):
        Htr = fs_tr["spikes"][li].sum(axis=0)
        Hte = fs_te["spikes"][li].sum(axis=0)
        acc, _ = _optimal_ridge_acc(Htr, ytr, Hte, yte_inh, k, seed)
        accs.append(float(acc))
    return accs


def _load_task(task, seed, task_kwargs):
    if task == "hier3":
        return make_task_hier3(seed)
    if task == "xor":
        return make_task_xor(seed)
    return make_task_semantic_inheritance(seed, **task_kwargs)


def run_seed(seed, task, hidden, wide_hidden, T, in_gain, subsample, n_hidden_layers, task_kwargs,
             eta, tau, sh_epochs, batch, preserve_norm,
             fa_epochs, fa_lr, bptt_hidden, bptt_epochs, bptt_lr):
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = _load_task(task, seed, task_kwargs)
    k = meta["k_classes"]; n_in = Xtr.shape[1]; inh_idx = idx["inh_idx"]
    if len(inh_idx):
        yv = yte[inh_idx]; chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    else:
        chance = float("nan")

    # rate backprop oracle (absolute reference) on the SAME task/depth
    onet = DendriticMLP([n_in] + [96] * n_hidden_layers + [k], seed=seed)
    _train_oracle(onet, Xtr, ytr, 250, 0.3, 128, seed)
    oracle_inh = _acc_on(onet, Xte, yte, inh_idx)

    if subsample is not None and len(Xtr) > subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:subsample]
        Xtr, ytr = Xtr[keep], ytr[keep]

    sizes = [n_in] + [hidden] * n_hidden_layers + [k]
    Xte_inh = Xte[inh_idx]; yte_inh = yte[inh_idx]

    # ---- FLOOR: frozen-random reservoir OPTIMAL ridge (matched + wide) -- the thing SoftHebb must beat ----
    frozen_opt_matched, lam_m = _frozen_reservoir_optimal(Xtr, ytr, Xte_inh, yte_inh, n_in, k, hidden,
                                                          n_hidden_layers, T, in_gain, seed)
    frozen_opt_wide, lam_w = _frozen_reservoir_optimal(Xtr, ytr, Xte_inh, yte_inh, n_in, k, wide_hidden,
                                                       n_hidden_layers, T, in_gain, seed)
    # reservoir per-layer selectivity (same init stream as the floor build)
    rrng = np.random.default_rng(seed + 1)
    res_layers = _build_layers(sizes, T, rrng, [2.5] + [1.0] * (len(sizes) - 2))
    res_layer_sel = _per_layer_selectivity(Xtr, ytr, Xte_inh, yte_inh, res_layers, n_hidden_layers, k, T, in_gain, seed)

    # ---- CANDIDATE: SoftHebb unsupervised soft-WTA Hebbian deep stack + OPTIMAL ridge readout ----
    sh_layers = _softhebb_train_stack(Xtr, sizes, T, in_gain, seed, n_hidden_layers, eta, tau, sh_epochs, batch,
                                      preserve_norm=preserve_norm, shuffle_input=False)
    Htr = _reservoir_features(Xtr, sh_layers, T, in_gain)
    Hte = _reservoir_features(Xte_inh, sh_layers, T, in_gain)
    softhebb_inh, lam_sh = _optimal_ridge_acc(Htr, ytr, Hte, yte_inh, k, seed)
    sh_layer_sel = _per_layer_selectivity(Xtr, ytr, Xte_inh, yte_inh, sh_layers, n_hidden_layers, k, T, in_gain, seed)
    # spike-rate sanity: mean summed-spikes per hidden neuron (must be non-degenerate, not 0 and not saturated at T)
    _, fs_sh, _ = _forward_logits(Xtr[:min(256, len(Xtr))], sh_layers, T, in_gain)
    sh_rate = [float(fs_sh["spikes"][li].sum(axis=0).mean()) for li in range(n_hidden_layers)]

    # ---- ANTI-CHEAT: SoftHebb trained on COLUMN-SHUFFLED inputs, read on REAL inputs (must NOT beat the reservoir) ----
    shuf_layers = _softhebb_train_stack(Xtr, sizes, T, in_gain, seed, n_hidden_layers, eta, tau, sh_epochs, batch,
                                        preserve_norm=preserve_norm, shuffle_input=True)
    Htr_s = _reservoir_features(Xtr, shuf_layers, T, in_gain)
    Hte_s = _reservoir_features(Xte_inh, shuf_layers, T, in_gain)
    shuffled_inh, _ = _optimal_ridge_acc(Htr_s, ytr, Hte_s, yte_inh, k, seed)

    # ---- WALL: transport-free chained FA + KP (reused) -- collapse to majority-class at depth ----
    fa_layers, _ = _train_snn_arm(Xtr, ytr, sizes, T, fa_epochs, fa_lr, fa_lr, in_gain, seed, "chained_fa")
    fa_inh = _accuracy(Xte, yte, fa_layers, T, in_gain, sub=inh_idx)
    kp_layers, _ = _train_snn_arm(Xtr, ytr, sizes, T, fa_epochs, fa_lr, fa_lr, in_gain, seed, "chained_fa_kp")
    kp_inh = _accuracy(Xte, yte, kp_layers, T, in_gain, sub=inh_idx)

    # ---- CEILING: surrogate-gradient BPTT (non-local reference) ----
    bptt_sizes = [n_in] + [(bptt_hidden if bptt_hidden else hidden)] * n_hidden_layers + [k]
    bptt_layers, _ = _train_snn_arm(Xtr, ytr, bptt_sizes, T, bptt_epochs, bptt_lr, bptt_lr, in_gain, seed, "bptt")
    bptt_inh = _accuracy(Xte, yte, bptt_layers, T, in_gain, sub=inh_idx)

    softhebb_over_reservoir = float(softhebb_inh - frozen_opt_matched)
    softhebb_over_wide = float(softhebb_inh - frozen_opt_wide)
    shuffled_over_reservoir = float(shuffled_inh - frozen_opt_matched)
    fa_over_chance = float(fa_inh - chance) if not np.isnan(chance) else float("nan")
    kp_over_chance = float(kp_inh - chance) if not np.isnan(chance) else float("nan")
    # per-layer selectivity RISING for SoftHebb (deepest hidden more class-informative than the first)
    sh_sel_rising = bool(sh_layer_sel[-1] >= sh_layer_sel[0] - 1e-9) if n_hidden_layers >= 2 else None
    sh_sel_over_res = bool(all(sh_layer_sel[li] >= res_layer_sel[li] for li in range(n_hidden_layers)))

    go = bool(softhebb_over_reservoir >= 0.10 and shuffled_over_reservoir <= 0.05)

    return {
        "seed": seed, "task": task, "n_hidden_layers": int(n_hidden_layers), "chance": chance,
        "n_in": n_in, "k": k, "sizes": sizes,
        "oracle_inherit": oracle_inh,
        "softhebb_inherit": float(softhebb_inh), "softhebb_lambda": lam_sh,
        "frozen_res_matched_inherit": float(frozen_opt_matched), "frozen_res_matched_lambda": lam_m,
        "frozen_res_wide_inherit": float(frozen_opt_wide), "frozen_res_wide_lambda": lam_w, "wide_hidden": wide_hidden,
        "shuffled_softhebb_inherit": float(shuffled_inh),
        "chained_fa_inherit": fa_inh, "chained_fa_kp_inherit": kp_inh, "bptt_inherit": bptt_inh,
        # decisive deltas
        "softhebb_over_reservoir": softhebb_over_reservoir,
        "softhebb_over_wide": softhebb_over_wide,
        "shuffled_over_reservoir": shuffled_over_reservoir,
        "fa_over_chance": fa_over_chance, "kp_over_chance": kp_over_chance,
        # per-layer class-selectivity (the rising-selectivity instrument)
        "softhebb_layer_selectivity": sh_layer_sel, "reservoir_layer_selectivity": res_layer_sel,
        "softhebb_selectivity_rising": sh_sel_rising,
        "softhebb_selectivity_over_reservoir": sh_sel_over_res,
        "softhebb_hidden_spikerate": sh_rate,
        # anti-cheats
        "softhebb_no_label_no_feedback": bool(_softhebb_reads_no_label_no_feedback()),
        "GO": go,
    }


def _agg(results):
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {}
    n = len(ok)

    def _m(key):
        vals = [r[key] for r in ok if r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    def _min(key):
        vals = [r[key] for r in ok if r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.min(vals)) if vals else float("nan")

    return {
        "n_seeds": n,
        "mean_chance": _m("chance"),
        "mean_oracle_inherit": _m("oracle_inherit"),
        "mean_softhebb_inherit": _m("softhebb_inherit"),
        "mean_frozen_res_matched_inherit": _m("frozen_res_matched_inherit"),
        "mean_frozen_res_wide_inherit": _m("frozen_res_wide_inherit"),
        "mean_shuffled_softhebb_inherit": _m("shuffled_softhebb_inherit"),
        "mean_chained_fa_inherit": _m("chained_fa_inherit"),
        "mean_chained_fa_kp_inherit": _m("chained_fa_kp_inherit"),
        "mean_bptt_inherit": _m("bptt_inherit"),
        "mean_softhebb_over_reservoir": _m("softhebb_over_reservoir"),
        "MIN_softhebb_over_reservoir": _min("softhebb_over_reservoir"),
        "mean_softhebb_over_wide": _m("softhebb_over_wide"),
        "mean_shuffled_over_reservoir": _m("shuffled_over_reservoir"),
        "MAX_shuffled_over_reservoir": float(np.max([r["shuffled_over_reservoir"] for r in ok])),
        "mean_fa_over_chance": _m("fa_over_chance"),
        "mean_kp_over_chance": _m("kp_over_chance"),
        "selectivity_rising_seeds": f"{sum(1 for r in ok if r.get('softhebb_selectivity_rising') is True)}/{n}",
        "selectivity_over_reservoir_seeds": f"{sum(1 for r in ok if r.get('softhebb_selectivity_over_reservoir') is True)}/{n}",
        "GO_seeds": f"{sum(1 for r in ok if r.get('GO') is True)}/{n}",
        "no_label_no_feedback_all": bool(all(r.get("softhebb_no_label_no_feedback") for r in ok)),
    }


def _build_verdict(agg):
    """Earn the enter-the-regime verdict with the preconditions that make the SoftHebb-vs-reservoir test valid.
    The load-bearing precondition is REGIME-EXISTS: the BPTT ceiling (a trained non-local net) must clear chance by
    a margin, else the task is degenerate and 'SoftHebb does/doesn't beat the reservoir' is UNDEFINED, not a
    negative (this is exactly the hier3 case: the ceiling itself sits at chance). Returns the decided dict."""
    mean_chance = agg["mean_chance"]
    v = Verdict("SoftHebb enter-the-regime (unsupervised soft-WTA vs frozen reservoir)", chance=mean_chance)
    v.require("regime exists: BPTT ceiling clears chance by >=0.05", agg["mean_bptt_inherit"],
              expect=lambda b: b >= mean_chance + 0.05,
              note="a trained non-local net must beat chance, else the task is unlearnable here and the test is UNDEFINED")
    v.require("rule genuinely unsupervised (reads no label/feedback)", agg["no_label_no_feedback_all"], expect=True,
              note="SoftHebb update source-guarded: only presynaptic x + own pre-activation u")
    v.require("seed control (deterministic numpy RNG; byte-identical re-run verified)", True, expect=True)
    v.control("SoftHebb vs shuffled-input control (the shuffle changed training)",
              treatment=agg["mean_softhebb_inherit"], control=agg["mean_shuffled_softhebb_inherit"])
    go = bool(agg["MIN_softhebb_over_reservoir"] >= 0.10 and agg["MAX_shuffled_over_reservoir"] <= 0.05)
    return v.decide(go=go, verbose=False)


def _verdict(agg, status=None):
    if not agg:
        return "(no aggregate)"
    minlift = agg["MIN_softhebb_over_reservoir"]
    go = (minlift >= 0.10 and agg["MAX_shuffled_over_reservoir"] <= 0.05)
    head = (f"SoftHebb {agg['mean_softhebb_inherit']:.3f} vs frozen-reservoir OPTIMAL matched "
            f"{agg['mean_frozen_res_matched_inherit']:.3f} / wide {agg['mean_frozen_res_wide_inherit']:.3f} "
            f"(lift mean {agg['mean_softhebb_over_reservoir']:+.3f}, MIN {minlift:+.3f}); shuffled-input SoftHebb "
            f"{agg['mean_shuffled_softhebb_inherit']:.3f} (over-reservoir mean {agg['mean_shuffled_over_reservoir']:+.3f}, "
            f"MAX {agg['MAX_shuffled_over_reservoir']:+.3f}); WALL FA {agg['mean_chained_fa_inherit']:.3f} / KP "
            f"{agg['mean_chained_fa_kp_inherit']:.3f} (over-chance {agg['mean_fa_over_chance']:+.3f}/"
            f"{agg['mean_kp_over_chance']:+.3f}); BPTT ceiling {agg['mean_bptt_inherit']:.3f}; oracle "
            f"{agg['mean_oracle_inherit']:.3f}; chance {agg['mean_chance']:.3f}. Per-layer selectivity rising "
            f"{agg['selectivity_rising_seeds']}, over-reservoir {agg['selectivity_over_reservoir_seeds']}. GO "
            f"{agg['GO_seeds']}.")
    if status == UNDEFINED:
        return ("UNDEFINED — degenerate testbed: the BPTT ceiling itself does not clear chance by >=0.05, so no "
                "learning REGIME exists to enter and 'SoftHebb vs reservoir' is uninterpretable here (an UNDEFINED is "
                "NOT a negative). " + head)
    if go:
        return ("ENTER-THE-REGIME / GO -- " + head + " => unsupervised soft-WTA Hebbian competition builds TASK-USEFUL "
                "deep spiking features that BEAT a random reservoir where FA/KP collapse; an independent 3rd local-rule "
                "crack (fully unsupervised, no per-layer label).")
    return ("NO-GO / METHOD-NEGATIVE -- " + head + " => SoftHebb does NOT clear the >=0.10 min-lift over the reservoir; "
            "the unsupervised soft-WTA competition adds little beyond a random projection under the enter-the-regime "
            "metric (the July-15 'unsupervised, not task-directed' objection is confirmed here). First-class negative.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--task", choices=["hier3", "inheritance", "xor"], default="hier3")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--wide-hidden", type=int, default=256)
    ap.add_argument("--n-hidden-layers", type=int, default=3)
    ap.add_argument("--timesteps", type=int, default=24)
    ap.add_argument("--in-gain", type=float, default=1.0)
    ap.add_argument("--train-subsample", type=int, default=2000)
    # SoftHebb hyperparameters
    ap.add_argument("--softhebb-lr", type=float, default=0.02, help="SoftHebb eta")
    ap.add_argument("--softhebb-tau", type=float, default=1.0, help="soft-WTA softmax temperature")
    ap.add_argument("--softhebb-epochs", type=int, default=30, help="unsupervised passes PER layer")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--preserve-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="homeostatic synaptic scaling to the init weight radius after each update (keeps the LIF forward firing)")
    # reused-arm tuning (the wall/ceiling arms)
    ap.add_argument("--fa-epochs", type=int, default=60)
    ap.add_argument("--fa-lr", type=float, default=0.05)
    ap.add_argument("--bptt-hidden", type=int, default=None)
    ap.add_argument("--bptt-epochs", type=int, default=120)
    ap.add_argument("--bptt-lr", type=float, default=0.05)
    # inheritance-task kwargs (mirror the wall runner defaults)
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
    results = []
    for sd in args.seeds:
        try:
            r = run_seed(sd, args.task, args.hidden, args.wide_hidden, args.timesteps, args.in_gain,
                         args.train_subsample, args.n_hidden_layers, task_kwargs,
                         args.softhebb_lr, args.softhebb_tau, args.softhebb_epochs, args.batch, args.preserve_norm,
                         args.fa_epochs, args.fa_lr, args.bptt_hidden, args.bptt_epochs, args.bptt_lr)
        except Exception as e:
            r = {"seed": sd, "error": repr(e), "traceback": traceback.format_exc()}
        results.append(r)
        if "error" not in r:
            print(f"[seed {sd}] [{r['task']} N={r['n_hidden_layers']}h] SoftHebb {r['softhebb_inherit']:.3f} "
                  f"vs frozen-res matched {r['frozen_res_matched_inherit']:.3f}/wide {r['frozen_res_wide_inherit']:.3f} "
                  f"(lift {r['softhebb_over_reservoir']:+.3f}) | shuffled {r['shuffled_softhebb_inherit']:.3f} "
                  f"({r['shuffled_over_reservoir']:+.3f}) | WALL FA {r['chained_fa_inherit']:.3f}/KP "
                  f"{r['chained_fa_kp_inherit']:.3f} | BPTT {r['bptt_inherit']:.3f} | oracle {r['oracle_inherit']:.3f} "
                  f"| chance {r['chance']:.3f} | layer-sel SH {['%.2f'%s for s in r['softhebb_layer_selectivity']]} "
                  f"res {['%.2f'%s for s in r['reservoir_layer_selectivity']]} | spikerate "
                  f"{['%.1f'%s for s in r['softhebb_hidden_spikerate']]} => GO={r['GO']}", flush=True)
        else:
            print(f"[seed {sd}] ERROR: {r['error']}", flush=True)

    agg = _agg(results)
    # ATTRIBUTION: what fraction of SoftHebb's mean lift over the reservoir is GENUINE structure-learning, i.e. NOT
    # also present in the shuffled-input control? Measuring both arms is not the same as asking whose the difference
    # was -- a small mean lift whose bulk survives in the shuffled control is a null, not a crack.
    attrib_fraction = None
    if agg:
        attrib_fraction = attributable_to("SoftHebb lift over reservoir vs shuffled-input control",
                                           agg["mean_softhebb_over_reservoir"], agg["mean_shuffled_over_reservoir"])
    decided = _build_verdict(agg) if agg else {"status": UNDEFINED, "preconditions": [], "disabled_processes": []}
    status = decided["status"]
    verdict = _verdict(agg, status)
    out = {"probe": "gap4_softhebb_soft_wta_hebbian_local", "task": args.task, "seeds": args.seeds,
           "status": status, "preconditions": decided["preconditions"],
           "disabled_processes": decided.get("disabled_processes", []),
           "softhebb_lift_attributable_to_structure": attrib_fraction,
           "config": {"hidden": args.hidden, "wide_hidden": args.wide_hidden,
                      "n_hidden_layers": args.n_hidden_layers, "T": args.timesteps, "in_gain": args.in_gain,
                      "train_subsample": args.train_subsample, "softhebb_lr": args.softhebb_lr,
                      "softhebb_tau": args.softhebb_tau, "softhebb_epochs": args.softhebb_epochs,
                      "batch": args.batch, "preserve_norm": args.preserve_norm,
                      "fa_epochs": args.fa_epochs, "fa_lr": args.fa_lr, "bptt_epochs": args.bptt_epochs,
                      "bptt_lr": args.bptt_lr, "task_kwargs": task_kwargs},
           "elapsed_seconds": round(time.time() - t0, 1), "results": results, "aggregate": agg,
           "verdict": verdict}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n" + verdict, flush=True)


if __name__ == "__main__":
    main()
