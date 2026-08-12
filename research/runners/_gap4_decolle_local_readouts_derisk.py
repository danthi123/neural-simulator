"""gap#4 CRUX SURPASS: DECOLLE per-layer LOCAL readouts on the TRAINABLE (surrogate-gradient) LIF SNN -- does a DEEP
(N>=3) spiking net ENTER the learning regime where the top-down CHAINED FA/KP rule collapses to majority-class?

THE WALL (located this month, do NOT re-derive):
  * 2026-08-02-gap4-depth-rescue-untestable-...: on this TRAINABLE LIF SNN the transport-free CHAINED multi-hop FA (and
    its KP repair) does NOT get a deep (N>=3) spiking net into the learning regime AT ALL -- both FA and KP collapse to
    BYTE-IDENTICAL majority-class output at N=3 and N=4. Credit-alignment (a LEARNED quantity, Lillicrap 2016) is then
    undefined, so the FA-vs-KP feedback question is premature: the binding wall sits UPSTREAM of the feedback -- the
    top-down credit path does not enter the deep-spiking learning regime.
  * 2026-08-02-gap4-crux-wall-LOCATED-... Update 4: the exhaustively-earned NAMED SURPASS is "a genuinely TRAINABLE
    spiking substrate ... as the field's working deep-spiking trainers do (e-prop, DECOLLE, SuperSpike)". (An earlier
    "DECOLLE smoke" in that finding bolted a local readout onto the FROZEN movable-plateau RESERVOIR -- a different,
    non-trainable substrate; this runner puts DECOLLE on the TRAINABLE surrogate-gradient substrate the wall lives on.)

THE HYPOTHESIS (the #1 named surpass): DECOLLE (Kaiser, Mostafa, Neftci 2020, "Synaptic Plasticity Dynamics for Deep
Continuous Local Learning", Front. Neurosci. 14:424). Each layer has its OWN FIXED-RANDOM local readout B_l (k, n_l);
each layer's weights are trained by that layer's OWN local classification error -- "errors do NOT propagate through
neurons and across layers" (paper, verbatim). The finite-spike sigma'(v-theta) top-down credit path the wall lives in
is BYPASSED: no descending credit at all. Hypothesis: a DEEP (N>=3) spiking net trained with DECOLLE per-layer local
readouts ENTERS the learning regime (leaves majority-class) and learns on REAL spikes, where the top-down FA/KP rule
could not.

DECOLLE weight update (paper Eq. 8, ported to this LIF SNN, per hidden layer li, per timestep t):
  local logits    y_l = spikes_li . B_l^T                       (B_l fixed-random, transport-free)
  local error     d_l = (softmax(y_l) - onehot(target)) / T     (the runner's softmax-CE convention; DECOLLE uses MSE,
                                                                  softmax-CE is the classification analogue, same shape)
  error at li     e_l = (d_l . B_l) * sigma'(v_li - theta)       (backproject through the FIXED readout x surrogate)
  weight grad     dW_li += eps_li^T . e_l                        (eps_li = alpha_leak*eps_li + pre = e-prop eligibility)
NO chain: e_l depends ONLY on layer li's own activity + its own B_l + the target -- never on any layer-above error.
The OUTPUT LIF layer is trained by the TRUE output error (target access), IDENTICALLY to the frozen/chained/bptt arms,
so (decolle - frozen) isolates EXACTLY the DECOLLE hidden-layer contribution (the runner's own frozen-isolation
convention). Final accuracy is read from the OUTPUT layer's summed spikes -- the SAME read every arm uses (fair,
apples-to-apples); the DECOLLE-native per-layer-readout accuracy is ALSO reported.

ARMS (all on the SAME LIF SNN forward + SAME task + SAME 6-seed harness -- reuse-by-import, NO edit to the base runner):
  decolle           : the candidate (per-layer local readouts, this file's _decolle_grads).
  chained_fa        : the WALL baseline (chained multi-hop fixed-random FA)          -- reuse _train_snn_arm.
  chained_fa_kp     : the WALL baseline (chained multi-hop KP-learned feedback)      -- reuse _train_snn_arm.
  dfa_eprop         : reference (e-prop with DIRECT feedback = OUTPUT error projected to each hidden by fixed-random
                      B_direct; NOT DECOLLE -- it still uses the output error). The depth-rescue Update named "does DFA
                      scale to N>=3?" as the open question; included to place DECOLLE vs DFA.  -- reuse isolation _train_snn.
  bptt              : the labelled CEILING (surrogate-gradient BPTT; scaffold, NOT brain-based) -- reuse _train_snn_arm.
  frozen_reservoir  : the FLOOR (fixed random hidden + trained output readout only; the R3 reservoir reframe) -- _train_snn_arm.
  decolle_shuffled  : ANTI-CHEAT (DECOLLE on SHUFFLED labels -> must collapse; the local error must carry signal).

ENTER-THE-REGIME metric (the wall's own metric, DECISIVE): does the deep net LEAVE majority-class? Per arm we report
held-out accuracy, accuracy-above-majority (majority = chance = modal class freq), the PREDICTION modal-fraction
(=1.0 means the arm predicts ONE class = the majority-class collapse signature), per-layer mean spike rate (REAL
spikes, not rate), and per-layer LINEAR (ridge) class-decodability (the deep layers becoming class-selective is the
enter-the-regime signature: DECOLLE should make them selective where the frozen reservoir leaves them random).

GO GATE (per seed, then 6-seed): decolle held-out >= chance + 0.20 AND leaves majority-class AND beats frozen_reservoir
by >= 0.10 AND chained_fa AND chained_fa_kp collapse to majority-class AND decolle_shuffled collapses AND bptt confirms
a target exists (> chance + 0.15). 6-seed headline = #seeds GO / n, with min-over-seeds decolle clearly above majority.
If DECOLLE ALSO fails to enter the regime -> a first-class HONEST NEGATIVE (report: does it leave majority-class at all?
at what depth does it break?).

Brain-based: DECOLLE local readouts are a LOCAL rule (fixed-random projections + local loss, no global backprop, no
weight transport). The BPTT arm is a labelled CEILING only (scaffold). ONE spiking substrate (the LIF SNN). NO sim/ edit
-- forward + surrogate + BPTT reused-by-import from sim/bptt_snn_gpu; every credit rule is a RUNNER-side function.

Run (numpy CPU; fan seeds across processes):
    # smoke, N=3, seed 42:
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap4_decolle_local_readouts_derisk \
        --task-xor --seeds 42 --n-hidden-layers 3 --hidden 32 --epochs 200 --lr 0.05 --train-subsample 2000 \
        --out research/findings/raw/_gap4_decolle/smoke_N3_s42.json
    # 6-seed, per depth (fan across processes; N=3 then N=4):
    for N in 3 4; do for S in 42 43 44 100 101 102; do SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._gap4_decolle_local_readouts_derisk --task-xor --seeds $S --n-hidden-layers $N \
        --hidden 32 --epochs 200 --lr 0.05 --train-subsample 2000 \
        --out research/findings/raw/_gap4_decolle/decolle_N${N}_s${S}.json & done; wait; done
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
# TINY matmuls -> one BLAS thread per process (oversubscription is ~30x slower); parallelize ACROSS seeds instead.
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

# ---- reuse-by-import (NO edit): the tasks + the WALL arms' trainer + the frozen-optimal control ----
from research.runners._gap4_bptt_snn_chained_fa_transport_free_derisk import (  # noqa: E402
    make_task_xor, make_task_hier3, make_task_nestedxor, _train_snn_arm, _frozen_reservoir_optimal)
# ---- reuse-by-import (NO edit): the SAME forward/eval helpers + build every arm's forward IDENTICALLY ----
from research.runners._snn_bptt_forward_vs_learning_isolation_derisk import (  # noqa: E402
    _softmax, _build_layers, _forward_logits, _accuracy, _train_snn as _train_snn_isolation)
# ---- reuse-by-import (NO edit): the LIF surrogate gradient ----
from sim.bptt_snn_gpu import atan_surrogate  # noqa: E402
# ---- workflow gates: earn the verdict (preconditions travel with it) + attribute the treatment/control ----
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402


# ============================================================================================================
# DECOLLE per-layer LOCAL readout credit (Kaiser/Mostafa/Neftci 2020) -- the named surpass, ON the trainable LIF SNN.
# ============================================================================================================
def _make_decolle_readouts(sizes, k, seed):
    """Fixed-random per-HIDDEN-layer local readout B_l : (k, n_l). Drawn from a SEPARATE seed stream (seed+4444) =>
    transport-free (never derived from any forward W_in, and a DIFFERENT stream than the chained-FA feedback seed+8888).
    One per hidden layer (the output layer has no local readout -- it IS the classifier, trained by the true error)."""
    frng = np.random.default_rng(seed + 4444)
    return [(frng.standard_normal((k, sizes[li + 1])) / np.sqrt(sizes[li + 1])).astype(np.float64)
            for li in range(len(sizes) - 2)]        # hidden layers only (transitions li=0..L-2)


def _decolle_grads(inputs, layers, fs, output_grad, ytrain, B_list, alpha_leak, alpha_surr=2.0,
                   sigma_norm=True, train_hidden=True):
    """DECOLLE per-layer LOCAL readout credit on the LIF SNN. Returns descent-side weight_grads (same sign convention
    as backward_unroll_xp / _chained_fa_grads, so the caller subtracts lr*wg).

    Each HIDDEN layer li is trained by its OWN local classification error via its OWN fixed-random readout B_list[li]:
        y_l = spikes_li . B_list[li]^T ; d_l = (softmax(y_l) - onehot(ytrain)) / T ;
        e_l = (d_l . B_list[li]) * sigma'(v_li - theta) ; dW_li += eps_li^T . e_l .
    NO descending credit -- e_l uses ONLY layer li's own spikes, its own B_list[li], and the target (never a
    layer-above error). train_hidden=False => only the OUTPUT layer's local delta is returned (the frozen-reservoir
    path, byte-identical output block), so (decolle - frozen) isolates EXACTLY the DECOLLE hidden-layer credit.

    The OUTPUT LIF layer is trained by the TRUE output error output_grad[t] (target access) -- IDENTICAL to the output
    block of _chained_fa_grads / the frozen arm -- so the classifier read is fair across arms."""
    T, Bn, _ = inputs.shape
    L = len(layers)
    spikes = fs["spikes"]; v_per = fs["v"]
    weight_grads = [np.zeros_like(l.W_in) for l in layers]
    eps = [np.zeros((Bn, l.W_in.shape[0]), dtype=l.W_in.dtype) for l in layers]
    ridx = np.arange(Bn)
    for t in range(T):
        # forward eligibility traces (all layers) -- the leak-recurrence factor (e-prop; == the chained/DFA arms use)
        for li in range(L):
            pre = inputs[t] if li == 0 else spikes[li - 1][t]
            eps[li] = alpha_leak * eps[li] + pre
        # per-layer membrane surrogate sigma'(v-theta); RELATIVE-normalized (mean 1.0) when sigma_norm (== the arms)
        psi = []
        for li in range(L):
            p = atan_surrogate(v_per[li][t] - layers[li].threshold, alpha=alpha_surr, xp=np)
            if sigma_norm:
                p = p / (p.mean() + 1e-9)
            psi.append(p)
        # ---- OUTPUT layer (target access): local delta gated by its own sigma' -- IDENTICAL to the chained/frozen out ----
        g_out = output_grad[t] * psi[L - 1]
        weight_grads[L - 1] += eps[L - 1].T @ g_out
        # ---- HIDDEN layers: EACH trained by its OWN local readout error (NO chain, NO output-error descent) ----
        if train_hidden:
            for li in range(L - 1):
                r_li = spikes[li][t]                          # (B, n_li) REAL spikes at t
                local_logits = r_li @ B_list[li].T           # (B, k) the layer's own local prediction
                p_l = _softmax(local_logits)
                d_l = p_l.copy(); d_l[ridx, ytrain] -= 1.0    # (B, k) local softmax-CE error
                d_l /= T                                      # match the output 1/T timestep-averaging
                e_l = (d_l @ B_list[li]) * psi[li]            # (B, n_li) local error at li x surrogate (transport-free)
                weight_grads[li] += eps[li].T @ e_l          # eligibility-weighted local update
    return weight_grads


def _no_readout_transport(B_list, layers):
    """anti-cheat: no DECOLLE readout B_l is byte-equal to any forward W_in or its transpose (the 'readout is secretly
    W^T' backprop-in-disguise cheat). B_l is a separate fixed-random stream, never a forward W."""
    for B in B_list:
        for l in layers:
            W = l.W_in
            if B.shape == W.shape and np.array_equal(B, W):
                return False
            if B.shape == W.T.shape and np.array_equal(B, W.T):
                return False
    return True


def _train_snn_decolle(Xtr, ytr, sizes, T, epochs, lr, in_gain, seed, batch=32, sigma_norm=True):
    """Train the LIF SNN with DECOLLE per-layer local readouts. Forward init is BYTE-IDENTICAL to _train_snn_arm
    (rng seed+1, w_scales [2.5]+[1.0]*...), so every arm shares the SAME forward -- the comparison is on the CREDIT
    RULE alone. Returns (layers, B_list)."""
    rng = np.random.default_rng(seed + 1)
    w_scales = [2.5] + [1.0] * (len(sizes) - 2)                 # strong first layer (same as every other arm)
    layers = _build_layers(sizes, T, rng, w_scales)
    k = sizes[-1]; n = len(Xtr)
    B_list = _make_decolle_readouts(sizes, k, seed)
    alpha_leak = layers[0].leak
    for _ in range(epochs):
        perm = rng.permutation(n)
        for b0 in range(0, n, batch):
            bi = perm[b0:b0 + batch]
            Xb, yb = Xtr[bi], ytr[bi]
            logits, fs, inp = _forward_logits(Xb, layers, T, in_gain)
            p = _softmax(logits)
            delta = p.copy(); delta[np.arange(len(yb)), yb] -= 1.0        # (B, k) output error
            og = np.repeat((delta / T)[None, :, :], T, axis=0).astype(np.float64)  # (T, B, k)
            wg = _decolle_grads(inp, layers, fs, og, yb, B_list, alpha_leak, alpha_surr=2.0,
                                sigma_norm=sigma_norm, train_hidden=True)
            for li in range(len(layers)):
                layers[li].W_in -= lr * (wg[li] / len(bi))
    return layers, B_list


# ============================================================================================================
# ENTER-THE-REGIME instrumentation -- the wall's own metric.
# ============================================================================================================
def _pred_modal_frac(X, layers, T, in_gain):
    """Fraction of predictions in the MOST-COMMON predicted class. ~1.0 == the arm predicts ONE class = the
    majority-class collapse signature (the wall's fingerprint: FA/KP give byte-identical majority-class output)."""
    if len(X) == 0:
        return float("nan")
    logits, _, _ = _forward_logits(X, layers, T, in_gain)
    preds = np.argmax(logits, axis=1)
    _, counts = np.unique(preds, return_counts=True)
    return float(counts.max() / len(preds))


def _layer_spike_rates(X, layers, T, in_gain, max_rows=512):
    """Per-layer mean spike rate (spikes per neuron per timestep) on a sample -- confirms REAL, healthy spiking
    (not silent, not saturated) at every layer. Uses the forward's fs['spikes'] (binary)."""
    if len(X) == 0:
        return []
    Xs = X[:max_rows]
    _, fs, _ = _forward_logits(Xs, layers, T, in_gain)
    return [float(np.asarray(sp).mean()) for sp in fs["spikes"]]      # includes output layer (last)


def _ridge_decode_acc(Htr, ytr, Hte, yte, k, seed):
    """Optimal linear (ridge, 5-fold-CV lambda) class-decode accuracy from a feature matrix. Per-layer selectivity read:
    how linearly class-decodable the layer's summed-spike rate is. Standalone (no import of the base runner's private
    ridge helpers needed)."""
    lams = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    Htr = np.concatenate([Htr, np.ones((len(Htr), 1))], axis=1)
    Hte = np.concatenate([Hte, np.ones((len(Hte), 1))], axis=1)
    Ytr_oh = np.eye(k)[ytr]
    d = Htr.shape[1]

    def _fit_pred(A, Yoh, B, lam):
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Yoh)
        return B @ W

    idxp = np.random.default_rng(seed + 31).permutation(len(Htr))
    folds = np.array_split(idxp, 5)
    best_lam, best_cv = float(lams[0]), -1.0
    for lam in lams:
        accs = []
        for f in folds:
            m = np.ones(len(Htr), dtype=bool); m[f] = False
            if not m.any() or len(f) == 0:
                continue
            pred = _fit_pred(Htr[m], Ytr_oh[m], Htr[f], lam)
            accs.append(float(np.mean(np.argmax(pred, axis=1) == ytr[f])))
        cv = float(np.mean(accs)) if accs else -1.0
        if cv > best_cv:
            best_cv, best_lam = cv, float(lam)
    pred = _fit_pred(Htr, Ytr_oh, Hte, best_lam)
    return float(np.mean(np.argmax(pred, axis=1) == yte))


def _per_layer_selectivity(layers, Xtr, ytr, Xte_inh, yte_inh, T, in_gain, k, seed, max_tr=1200):
    """Per-HIDDEN-layer linear class-decodability of the TRAINED layers' summed spikes (train-fit, held-eval).
    The enter-the-regime signature: DECOLLE should make the DEEP hidden layers class-selective (decodability rises
    from the frozen-reservoir random baseline)."""
    if len(Xte_inh) == 0:
        return []
    Xt = Xtr[:max_tr]; yt = ytr[:max_tr]
    _, fstr, _ = _forward_logits(Xt, layers, T, in_gain)
    _, fste, _ = _forward_logits(Xte_inh, layers, T, in_gain)
    out = []
    for li in range(len(layers) - 1):                            # hidden layers only
        Htr = np.asarray(fstr["spikes"][li]).sum(axis=0)         # (B, n_li) summed spikes
        Hte = np.asarray(fste["spikes"][li]).sum(axis=0)
        out.append(_ridge_decode_acc(Htr, yt, Hte, yte_inh, k, seed))
    return out


def _decolle_local_readout_acc(layers, B_list, Xte_inh, yte_inh, T, in_gain):
    """The DECOLLE-NATIVE prediction: each hidden layer's OWN fixed-random local readout, argmax of the summed-spike
    projection. Reports per-hidden-layer local-readout held-out accuracy (the signal DECOLLE actually trains toward)."""
    if len(Xte_inh) == 0:
        return []
    _, fs, _ = _forward_logits(Xte_inh, layers, T, in_gain)
    out = []
    for li in range(len(layers) - 1):
        R = np.asarray(fs["spikes"][li]).sum(axis=0)             # (B, n_li)
        logits = R @ B_list[li].T                                # (B, k)
        out.append(float(np.mean(np.argmax(logits, axis=1) == yte_inh)))
    return out


def _arm_report(name, layers, Xtr, ytr, Xte, yte, inh_idx, T, in_gain, chance, k):
    """Held-out accuracy + enter-the-regime reads for one trained arm."""
    acc = _accuracy(Xte, yte, layers, T, in_gain, sub=inh_idx)
    modal = _pred_modal_frac(Xte[inh_idx] if len(inh_idx) else Xte, layers, T, in_gain)
    rates = _layer_spike_rates(Xtr, layers, T, in_gain)
    above = float(acc - chance) if not np.isnan(chance) else float("nan")
    # leaves-majority: clearly above chance AND not predicting a single class (modal < 0.95)
    leaves = bool((not np.isnan(above)) and above > 0.05 and (not np.isnan(modal)) and modal < 0.95)
    return {"arm": name, "held_acc": acc, "above_majority": above, "pred_modal_frac": modal,
            "layer_spike_rates": rates, "leaves_majority": leaves}


# ============================================================================================================
# Per-seed driver.
# ============================================================================================================
def _make_task(seed, task, hier3_kwargs):
    if task == "hier3":
        return make_task_hier3(seed, **hier3_kwargs)
    if task == "nestedxor":
        return make_task_nestedxor(seed)
    return make_task_xor(seed)


def run_seed(seed, task, n_hidden_layers, hidden, T, epochs, lr, in_gain, subsample, sigma_norm,
             bptt_hidden, bptt_epochs, bptt_lr, hier3_kwargs):
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = _make_task(seed, task, hier3_kwargs)
    k = meta["k_classes"]; n_in = Xtr.shape[1]; inh_idx = idx["inh_idx"]
    if len(inh_idx):
        yv = yte[inh_idx]; chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    else:
        chance = float("nan")

    if subsample is not None and len(Xtr) > subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:subsample]
        Xtr, ytr = Xtr[keep], ytr[keep]

    sizes = [n_in] + [hidden] * n_hidden_layers + [k]

    # ---- DETERMINISM self-check: build the forward twice at this seed -> W_in must be byte-identical (the substrate
    #      IS seeded by `seed`, per CLAUDE.md's actual_seed_used trap -- here seeding is pure-numpy rng(seed+1)). ----
    _r1 = _build_layers(sizes, T, np.random.default_rng(seed + 1), [2.5] + [1.0] * (len(sizes) - 2))
    _r2 = _build_layers(sizes, T, np.random.default_rng(seed + 1), [2.5] + [1.0] * (len(sizes) - 2))
    seeded_ok = bool(all(np.array_equal(a.W_in, b.W_in) for a, b in zip(_r1, _r2)))

    # ===== ARMS (every arm shares the SAME forward init: rng seed+1, w_scales [2.5,1.0,...]) =====
    # candidate: DECOLLE per-layer local readouts
    dec_layers, B_list = _train_snn_decolle(Xtr, ytr, sizes, T, epochs, lr, in_gain, seed, sigma_norm=sigma_norm)
    # WALL baselines: chained fixed-FA and chained KP-learned FA
    fa_layers, _ = _train_snn_arm(Xtr, ytr, sizes, T, epochs, lr, lr, in_gain, seed, "chained_fa", sigma_norm=sigma_norm)
    kp_layers, _ = _train_snn_arm(Xtr, ytr, sizes, T, epochs, lr, lr, in_gain, seed, "chained_fa_kp", sigma_norm=sigma_norm)
    # reference: DFA e-prop (OUTPUT error projected directly to each hidden by fixed-random B_direct -- NOT DECOLLE)
    dfa_layers = _train_snn_isolation(Xtr, ytr, sizes, T, epochs, lr, in_gain, seed, credit_mode="eprop")
    # CEILING: surrogate-gradient BPTT (labelled scaffold), properly tuned
    b_sizes = [n_in] + [(bptt_hidden if bptt_hidden else hidden)] * n_hidden_layers + [k]
    bptt_layers, _ = _train_snn_arm(Xtr, ytr, b_sizes, T, (bptt_epochs if bptt_epochs else epochs),
                                    (bptt_lr if bptt_lr else lr), lr, in_gain, seed, "bptt", sigma_norm=sigma_norm)
    # FLOOR: frozen reservoir (fixed random hidden, trained output readout only)
    fr_layers, _ = _train_snn_arm(Xtr, ytr, sizes, T, epochs, lr, lr, in_gain, seed, "frozen_reservoir", sigma_norm=sigma_norm)
    # ANTI-CHEAT: DECOLLE on SHUFFLED labels -> must collapse (local error must carry signal)
    prng = np.random.default_rng(seed + 555)
    yperm = ytr[prng.permutation(len(ytr))]
    shuf_layers, _ = _train_snn_decolle(Xtr, yperm, sizes, T, epochs, lr, in_gain, seed, sigma_norm=sigma_norm)

    def _rep(name, ly):
        return _arm_report(name, ly, Xtr, ytr, Xte, yte, inh_idx, T, in_gain, chance, k)

    arms = {
        "decolle": _rep("decolle", dec_layers),
        "chained_fa": _rep("chained_fa", fa_layers),
        "chained_fa_kp": _rep("chained_fa_kp", kp_layers),
        "dfa_eprop": _rep("dfa_eprop", dfa_layers),
        "bptt_ceiling": _rep("bptt_ceiling", bptt_layers),
        "frozen_reservoir": _rep("frozen_reservoir", fr_layers),
        "decolle_shuffled": _rep("decolle_shuffled", shuf_layers),
    }

    # ---- enter-the-regime deep-dives: per-layer selectivity (decolle vs frozen vs chained_fa) + DECOLLE-native read ----
    Xte_inh = Xte[inh_idx] if len(inh_idx) else Xte[:0]
    yte_inh = yte[inh_idx] if len(inh_idx) else yte[:0]
    selectivity = {
        "decolle": _per_layer_selectivity(dec_layers, Xtr, ytr, Xte_inh, yte_inh, T, in_gain, k, seed),
        "frozen_reservoir": _per_layer_selectivity(fr_layers, Xtr, ytr, Xte_inh, yte_inh, T, in_gain, k, seed),
        "chained_fa": _per_layer_selectivity(fa_layers, Xtr, ytr, Xte_inh, yte_inh, T, in_gain, k, seed),
    }
    decolle_local_readout_acc = _decolle_local_readout_acc(dec_layers, B_list, Xte_inh, yte_inh, T, in_gain)

    # ---- frozen-reservoir OPTIMAL ridge readout (the STRONGER reservoir baseline, matched width) ----
    frozen_opt_matched, _lam = _frozen_reservoir_optimal(Xtr, ytr, Xte_inh, yte_inh, n_in, k, hidden,
                                                         n_hidden_layers, T, in_gain, seed) if len(inh_idx) else (float("nan"), None)

    # ===== decisive GO (per seed) =====
    dec = arms["decolle"]; fr = arms["frozen_reservoir"]; bp = arms["bptt_ceiling"]
    fa = arms["chained_fa"]; kp = arms["chained_fa_kp"]; shuf = arms["decolle_shuffled"]
    decolle_beats_frozen = float(dec["held_acc"] - fr["held_acc"])
    fa_collapsed = bool((not fa["leaves_majority"]))
    kp_collapsed = bool((not kp["leaves_majority"]))
    shuffled_collapses = bool((not np.isnan(chance)) and shuf["held_acc"] <= chance + 0.06)
    bptt_confirms_target = bool((not np.isnan(chance)) and bp["held_acc"] > chance + 0.15)
    decolle_enters = bool(dec["leaves_majority"] and (not np.isnan(chance)) and dec["above_majority"] >= 0.20)
    GO = bool(decolle_enters and decolle_beats_frozen >= 0.10 and fa_collapsed and kp_collapsed
              and shuffled_collapses and bptt_confirms_target)

    # ---- ATTRIBUTION (tools.lab): whose is the DECOLLE lift? Fraction of DECOLLE's held-out accuracy NOT present
    #      in each control -- the shuffled-label DECOLLE (correct-label local error) and the frozen reservoir. ----
    attr_vs_shuffled = attributable_to(f"DECOLLE held-out lift vs shuffled-label local error (s{seed} N{n_hidden_layers})",
                                       dec["held_acc"], shuf["held_acc"])
    attr_vs_frozen = attributable_to(f"DECOLLE held-out vs frozen reservoir (s{seed} N{n_hidden_layers})",
                                     dec["held_acc"], fr["held_acc"])

    # ---- EARN THE VERDICT (tools.verdict.Verdict): preconditions travel with the verdict; UNDEFINED (never a NO-GO)
    #      whenever the instrument itself fails -- e.g. the labelled BPTT ceiling below chance (the hier3 task-fit
    #      wall) means NO target exists, so the DECOLLE-vs-wall comparison measures nothing. ----
    chance_art = {"chance": chance}
    v = Verdict(f"gap4 DECOLLE local readouts {task} N{n_hidden_layers}h s{seed}", chance=chance)
    v.floor("labelled BPTT ceiling above chance (a target exists)", bp["held_acc"], artifact=chance_art, key="chance",
            note="if the ceiling cannot beat chance no target exists -> UNDEFINED, not a NO-GO (hier3 task-fit wall)")
    v.floor("DECOLLE above chance", dec["held_acc"], artifact=chance_art, key="chance")
    v.require("BPTT confirms a target (> chance+0.15)", bptt_confirms_target, expect=True)
    v.require("DECOLLE enters the learning regime (above_majority>=0.20 AND leaves majority-class)",
              decolle_enters, expect=True)
    v.control("DECOLLE vs frozen-reservoir floor", dec["held_acc"], fr["held_acc"], min_separation=0.10,
              note="DECOLLE must beat the fixed-hidden reservoir (trained readout only) by >=0.10")
    v.control("DECOLLE vs shuffled-label DECOLLE (anti-cheat)", dec["held_acc"], shuf["held_acc"], min_separation=0.10,
              note="the local error must carry correct-label signal")
    v.require("chained fixed-FA collapses to majority-class (the located wall)", fa_collapsed, expect=True)
    v.require("chained KP collapses to majority-class (the located wall)", kp_collapsed, expect=True)
    v.require("no readout weight transport (B_l != W and != W^T)",
              bool(_no_readout_transport(B_list, dec_layers)), expect=True)
    v.require("substrate genuinely seeded (build-twice identical)", seeded_ok, expect=True)
    decided = v.decide(go=GO, verbose=False)
    below_chance = bool((not np.isnan(chance)) and bp["held_acc"] <= chance)   # instrument (ceiling) failed -> UNDEFINED

    return {
        "seed": seed, "task": task, "n_hidden_layers": int(n_hidden_layers), "sizes": sizes,
        "k": k, "chance": chance, "seeded_substrate_ok": seeded_ok,
        "arms": arms,
        "per_layer_selectivity_ridge": selectivity,
        "decolle_local_readout_acc_per_hidden": decolle_local_readout_acc,
        "frozen_optimal_matched_inherit": frozen_opt_matched,
        "no_readout_transport": bool(_no_readout_transport(B_list, dec_layers)),
        # decisive
        "decolle_beats_frozen": decolle_beats_frozen,
        "decolle_enters_regime": decolle_enters,
        "fa_collapsed": fa_collapsed, "kp_collapsed": kp_collapsed,
        "shuffled_collapses": shuffled_collapses, "bptt_confirms_target": bptt_confirms_target,
        "GO": GO,
        # ---- attribution (tools.lab) + earned verdict (tools.verdict) + instrument-failure declaration ----
        "attribution_decolle_vs_shuffled": attr_vs_shuffled,
        "attribution_decolle_vs_frozen": attr_vs_frozen,
        "verdict": decided, "verdict_status": decided["status"], "below_chance": below_chance,
    }


def _agg(results, seeds):
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {}
    n = len(ok)

    def _m(path):
        vals = []
        for r in ok:
            v = r
            for p in path:
                v = v[p] if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float("nan")

    def _minmax(path, which="min"):
        vals = []
        for r in ok:
            v = r
            for p in path:
                v = v[p] if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                vals.append(float(v))
        if not vals:
            return float("nan")
        return float(min(vals)) if which == "min" else float(max(vals))

    def _cnt(key):
        return sum(1 for r in ok if r.get(key) is True)

    return {
        "n_seeds": n, "mean_chance": _m(["chance"]),
        "mean_decolle_acc": _m(["arms", "decolle", "held_acc"]),
        "min_decolle_acc": _minmax(["arms", "decolle", "held_acc"], "min"),
        "mean_decolle_above_majority": _m(["arms", "decolle", "above_majority"]),
        "min_decolle_above_majority": _minmax(["arms", "decolle", "above_majority"], "min"),
        "mean_chained_fa_acc": _m(["arms", "chained_fa", "held_acc"]),
        "mean_chained_fa_kp_acc": _m(["arms", "chained_fa_kp", "held_acc"]),
        "mean_dfa_eprop_acc": _m(["arms", "dfa_eprop", "held_acc"]),
        "mean_bptt_acc": _m(["arms", "bptt_ceiling", "held_acc"]),
        "mean_frozen_reservoir_acc": _m(["arms", "frozen_reservoir", "held_acc"]),
        "mean_frozen_optimal_matched": _m(["frozen_optimal_matched_inherit"]),
        "mean_decolle_shuffled_acc": _m(["arms", "decolle_shuffled", "held_acc"]),
        "mean_decolle_beats_frozen": _m(["decolle_beats_frozen"]),
        "decolle_enters_regime_seeds": f"{_cnt('decolle_enters_regime')}/{n}",
        "fa_collapsed_seeds": f"{_cnt('fa_collapsed')}/{n}",
        "kp_collapsed_seeds": f"{_cnt('kp_collapsed')}/{n}",
        "shuffled_collapses_seeds": f"{_cnt('shuffled_collapses')}/{n}",
        "bptt_confirms_target_seeds": f"{_cnt('bptt_confirms_target')}/{n}",
        "no_readout_transport_all": bool(all(r.get("no_readout_transport") for r in ok)),
        "seeded_substrate_all": bool(all(r.get("seeded_substrate_ok") for r in ok)),
        "GO_seeds": f"{_cnt('GO')}/{n}",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-hidden-layers", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--timesteps", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--in-gain", type=float, default=1.0)
    ap.add_argument("--sigma-norm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--train-subsample", type=int, default=2000)
    ap.add_argument("--task-xor", action="store_true", help="depth-2 XOR->threshold task (the alignment-probe task; DIRECT comparison to the wall).")
    ap.add_argument("--task-hier3", action="store_true", help="obligatory-depth-3 compositional task.")
    ap.add_argument("--task-nestedxor", action="store_true")
    ap.add_argument("--bptt-hidden", type=int, default=None)
    ap.add_argument("--bptt-epochs", type=int, default=None)
    ap.add_argument("--bptt-lr", type=float, default=None)
    ap.add_argument("--out", type=str, default=str(_REPO / "research" / "findings" / "raw" / "_gap4_decolle" / "decolle.json"))
    args = ap.parse_args()

    task = "hier3" if args.task_hier3 else ("nestedxor" if args.task_nestedxor else "xor")
    hier3_kwargs = {}

    t0 = time.time()
    results = []
    for sd in args.seeds:
        try:
            r = run_seed(sd, task, args.n_hidden_layers, args.hidden, args.timesteps, args.epochs, args.lr,
                         args.in_gain, args.train_subsample, args.sigma_norm,
                         args.bptt_hidden, args.bptt_epochs, args.bptt_lr, hier3_kwargs)
        except Exception as e:
            r = {"seed": sd, "error": repr(e), "traceback": traceback.format_exc()}
        results.append(r)
        if "error" not in r:
            a = r["arms"]
            print(f"[seed {sd} N={r['n_hidden_layers']}h task={task} chance={r['chance']:.3f}] "
                  f"DECOLLE {a['decolle']['held_acc']:.3f} (above_maj {a['decolle']['above_majority']:+.3f}, "
                  f"modal {a['decolle']['pred_modal_frac']:.2f}, enters={r['decolle_enters_regime']}) | "
                  f"FA {a['chained_fa']['held_acc']:.3f}(modal {a['chained_fa']['pred_modal_frac']:.2f}) "
                  f"KP {a['chained_fa_kp']['held_acc']:.3f}(modal {a['chained_fa_kp']['pred_modal_frac']:.2f}) "
                  f"DFA {a['dfa_eprop']['held_acc']:.3f} | BPTT {a['bptt_ceiling']['held_acc']:.3f} | "
                  f"frozen {a['frozen_reservoir']['held_acc']:.3f} (opt {r['frozen_optimal_matched_inherit']:.3f}) | "
                  f"shuf {a['decolle_shuffled']['held_acc']:.3f} | "
                  f"beats_frozen {r['decolle_beats_frozen']:+.3f} => GO={r['GO']}", flush=True)
            print(f"    selectivity(ridge) decolle {['%.2f'%x for x in r['per_layer_selectivity_ridge']['decolle']]} "
                  f"vs frozen {['%.2f'%x for x in r['per_layer_selectivity_ridge']['frozen_reservoir']]} "
                  f"vs FA {['%.2f'%x for x in r['per_layer_selectivity_ridge']['chained_fa']]} | "
                  f"decolle-local-readout/hidden {['%.2f'%x for x in r['decolle_local_readout_acc_per_hidden']]}", flush=True)
        else:
            print(f"[seed {sd}] ERROR: {r['error']}", flush=True)

    agg = _agg(results, args.seeds)
    out = {"probe": "gap4_decolle_local_readouts", "task": task, "seeds": args.seeds,
           "config": {"n_hidden_layers": args.n_hidden_layers, "hidden": args.hidden, "T": args.timesteps,
                      "epochs": args.epochs, "lr": args.lr, "in_gain": args.in_gain, "sigma_norm": args.sigma_norm,
                      "train_subsample": args.train_subsample, "bptt_hidden": args.bptt_hidden,
                      "bptt_epochs": args.bptt_epochs, "bptt_lr": args.bptt_lr},
           "elapsed_seconds": round(time.time() - t0, 1), "results": results, "aggregate": agg}
    # ---- the earned verdict travels with its preconditions (tools.verdict). Merge every seed's preconditions to the
    #      top level so `verdict_preconditions` can enforce them; the status is UNDEFINED if ANY seed's instrument
    #      failed (below-chance ceiling), never a fabricated NO-GO. `below_chance` declared for the below_chance gate.
    ok_res = [r for r in results if "error" not in r and isinstance(r.get("verdict"), dict)]
    statuses = [r["verdict"]["status"] for r in ok_res]
    merged_pre = [c for r in ok_res for c in r["verdict"]["preconditions"]]
    any_undef = any(s == "UNDEFINED" for s in statuses)
    all_go = bool(statuses) and all(s == "GO" for s in statuses)
    status_word = "UNDEFINED" if any_undef else ("GO" if all_go else "NO-GO")
    out["preconditions"] = merged_pre
    out["verdict_status"] = status_word
    out["below_chance"] = bool(any(r.get("below_chance") for r in ok_res))
    if agg:
        out["verdict"] = (
            f"{status_word} — [{task} N={args.n_hidden_layers}h] DECOLLE {agg['mean_decolle_acc']:.3f} "
            f"(min {agg['min_decolle_acc']:.3f}, above-maj mean {agg['mean_decolle_above_majority']:+.3f}) | "
            f"WALL: chained-FA {agg['mean_chained_fa_acc']:.3f} (collapsed {agg['fa_collapsed_seeds']}), "
            f"KP {agg['mean_chained_fa_kp_acc']:.3f} (collapsed {agg['kp_collapsed_seeds']}) | "
            f"DFA {agg['mean_dfa_eprop_acc']:.3f} | BPTT ceiling {agg['mean_bptt_acc']:.3f} "
            f"({agg['bptt_confirms_target_seeds']} confirm target) | frozen {agg['mean_frozen_reservoir_acc']:.3f} "
            f"(opt {agg['mean_frozen_optimal_matched']:.3f}) | shuffled {agg['mean_decolle_shuffled_acc']:.3f} "
            f"({agg['shuffled_collapses_seeds']} collapse) | decolle-beats-frozen {agg['mean_decolle_beats_frozen']:+.3f} "
            f"| chance {agg['mean_chance']:.3f}. GO {agg['GO_seeds']} "
            f"(decolle enters regime {agg['decolle_enters_regime_seeds']}).")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + out.get("verdict", "(no aggregate)"), flush=True)
    print(f"[decolle] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
