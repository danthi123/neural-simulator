"""gap#4 crux de-risk -- TWO-STAGE BIRDSONG TEACHING-SIGNAL DECOMPOSITION on a DEEP (N>=2/3) spiking LIF stack.

THE WALL (do NOT re-derive; located 2026-08-02):
  Deep credit through a DEEP spiking net with a LOCAL rule is blocked because the finite-spike sigma'(v-theta) read
  carries NO directed credit (even a perfect W^T oracle == a label-shuffle; `2026-08-02-gap4-crux-wall-LOCATED-*`),
  and at N>=3 the transport-free local rules (chained-FA/KP) do not enter the learning regime at all
  (`2026-08-02-gap4-depth-rescue-untestable-*`). DFA e-prop trains at N=2 (inherit ~0.895) but routes the OUTPUT
  ERROR through the surrogate sigma' at every hidden neuron -- the exact read that carries no directed credit.

THE MECHANISM (the biology's variance-tractable answer to "credit for a produced sequence"):
  TWO-STAGE learning (Tesileanu, Olveczky, Balasubramanian 2017, eLife 6:e20944 -- READ). Decompose credit into:
    STAGE A -- a LOW-DIM LMAN-analogue TUTOR learns a corrective teaching signal by REWARD-modulated node
      perturbation CONFINED to a few units + a reward BASELINE (Eq 6: df_j = eta*(R - Rbar)*xi_j; Rbar = running
      mean reward). Because the perturbed dimension is LOW (u in R^k, not the high-dim deep read-state), the
      zeroth-order estimator variance (variance ~ perturbed-dim) is TRACTABLE -- the regime the refuted naive-NP on
      the high-dim deep read-state VIOLATED (`2026-07-13-NP-vs-KP-REFUTED`).
    STAGE B -- the DEEP HVC->RA motor stack trains by a REWARD-INDEPENDENT LOCAL Hebbian rule that FOLLOWS the
      tutor's per-neuron target (Eq 1: dW_ij = eta * c_pre_j * (teach_i - theta)). There is NO top-down error routed
      through sigma' -- deep credit becomes LOCAL TARGET-FOLLOWING. The finite-spike-read wall is SIDESTEPPED: RL
      supplies the target (low-dim, variance-tractable); Hebbian follows it locally (no sigma').
  DEEP extension (documented, honest): the birdsong circuit is SHALLOW (one HVC->RA synapse; LMAN biases RA). To
  make the student DEEP, the tutor's low-dim latent u is BROADCAST to every layer by a FIXED-RANDOM matrix M_l ->
  per-neuron teaching c_l = u @ M_l; each layer follows it by Eq 1. This differs from DFA e-prop: DFA broadcasts the
  supervised OUTPUT ERROR (delta=p-y, routed through sigma', vanishes as it learns); the tutor broadcasts an
  RL-LEARNED TARGET the layers FOLLOW (no sigma', persists), so it does not need the deep net's forward to be
  organized first (no chicken-and-egg alignment bootstrap). Timescale-matching (Eq 4): student + tutor operate on
  the SAME per-trial teaching by construction.

HOW I DIFFER FROM THE PRIOR 2026-05-16 SONGBIRD NEGATIVE (`2026-05-16-generator-G1-songbird-NEGATIVE`):
  That attempt trained a SongHVC argmax controller by SELF-COMPREHENSION of babbled productions over the G.20
  recognition substrate; it failed because the ORDER-READOUT JUDGE could not discriminate order -> reward was
  identically 0 -> the controller never moved (single-stage, no two-stage decomposition, no reward baseline, no
  low-dim-tutor confinement, a broken reward). Here: (1) a WELL-DEFINED environmental reward (produced-output
  correctness, the auditory-error analogue), (2) the TWO-STAGE decomposition (low-dim RL tutor + reward-independent
  deep Hebbian), (3) a reward BASELINE (R-Rbar), (4) a DEEP spiking LIF student on the today-validated
  compositional-inheritance instrument -- none of which existed in 2026-05-16.

ARMS: tutor_teach (candidate) | reservoir (frozen-hidden floor) | bptt (surrogate-BPTT ceiling, a target exists) |
  shuffle_tutor (scramble the per-example teaching -> must collapse: the tutor target must carry signal) |
  permuted_reward (permute reward in the tutor NP -> tutor cannot learn -> collapse) | naive_np (GENUINE high-dim
  node perturbation on the deep read-state via a perturbed forward pass, no low-dim confinement/no baseline ->
  reproduce its collapse; shows the low-dim+baseline is what works).

ENTER-THE-REGIME: does the deep student LEAVE majority-class (varied predictions AND inherit > majority + margin)?
GO: 6-seed held-out inherit >= chance+0.20, min > majority-class, BEATS reservoir by >=0.10, on depth-2/3 LIF, with
  shuffle_tutor + permuted_reward collapsing (to ~majority). HONEST-NEGATIVE otherwise (name the failing stage:
  STAGE A = tutor cannot learn a useful target; STAGE B = Hebbian-follow cannot track it).

BRAIN-BASED: the tutor RL (node perturbation + baseline) and the student Hebbian (Eq 1) are BOTH local/brain-based;
  reward is environmental (the world scores the motor output). BPTT is a CEILING only (non-biological, labeled).
NO sim/ edit (reuse-by-import). SIM_BACKEND=numpy. Verify cfg.seed n/a (no CoreSimConfig; np.random.default_rng seeded).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    make_task_semantic_inheritance, stage0_depth_genuineness, _train_oracle, _acc_on)
from research.runners._snn_bptt_forward_vs_learning_isolation_derisk import (  # noqa: E402
    _build_layers, _forward_logits, _accuracy, _softmax, _train_snn)
from sim.dendritic_mlp import DendriticMLP  # noqa: E402
from sim.bptt_snn_gpu import LIFLayerXP, forward_step_xp  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_gap4_birdsong" / "tutor_teach.json"


# ------------------------------------------------------------------------------------------------------------
# Forward helpers -- reuse _forward_logits for the clean pass; a small perturbed forward for the naive-NP arm.
# ------------------------------------------------------------------------------------------------------------
def _layer_pre_activities(X, layers, T, in_gain):
    """Clean forward. Returns (logits, pre) where pre[l] = TIME-MEAN presynaptic activity INTO layer l
    (pre[0]=mean input current = in_gain*X; pre[l>0]=mean spike-rate of layer l-1), each (B, n_pre_l)."""
    logits, fs, inp = _forward_logits(X, layers, T, in_gain)   # inp (T,B,n_in); fs["spikes"][l] (T,B,n_l)
    pre = [inp.mean(axis=0)]                                    # (B, n_in)
    for l in range(len(layers) - 1):
        pre.append(fs["spikes"][l].mean(axis=0))               # mean spike-rate of layer l -> input to l+1
    return logits, pre


def _forward_logits_perturbed(X, layers, T, in_gain, perts):
    """Forward unroll with a CONSTANT per-neuron activity perturbation `perts[l]` (B, n_l) added to layer l's spike
    output every timestep (node perturbation injected INTO the forward, so it propagates to the output). Only hidden
    layers are perturbed (perts[-1] for the output layer is unused). Returns summed-output-spike logits (B, k)."""
    B = X.shape[0]
    x0 = (in_gain * X).astype(np.float64)                      # constant input current
    L = len(layers)
    states = [layers[l].init_state(B, xp=np) for l in range(L)]
    out_sum = np.zeros((B, layers[-1].n_post), dtype=np.float64)
    for _t in range(T):
        x_in = x0
        for l, layer in enumerate(layers):
            states[l], s = forward_step_xp(states[l], x_in, layer, xp=np)
            if l < L - 1 and perts is not None and perts[l] is not None:
                s = s + perts[l]                                # inject the node perturbation post-spike
            x_in = s
            if l == L - 1:
                out_sum += s
    return out_sum


def _reward_negerr(produced, y, k):
    """Environmental reward = NEGATIVE auditory-template error = -mean_k (produced_k - onehot(y)_k)^2 -- the songbird's
    own reward (the bird compares its produced song to the memorized tutor TEMPLATE; the mismatch is the scalar reward,
    computed by the world/sensory system, NOT by the brain). GRADED + NON-SATURATING (unlike softmax p_correct, which
    saturates as the tutor drive grows and kills the node-perturbation signal). The tutor thus learns the RESIDUAL
    correction (u -> onehot(y) - logits/beta): LMAN corrects what RA gets wrong and FADES as RA improves. Per-ex (B,)."""
    oh = np.zeros((len(y), k), dtype=np.float64)
    oh[np.arange(len(y)), y] = 1.0
    return -((produced - oh) ** 2).mean(axis=1)


# ------------------------------------------------------------------------------------------------------------
# STAGE-B student: reward-INDEPENDENT local Hebbian following the tutor target (Eq 1).  NO sigma'.
# ------------------------------------------------------------------------------------------------------------
def _hebbian_follow(layers, pre, teach, lr, out_teach, out_pre, lr_out, renorm_init_norms,
                    out_rate=None, delta_readout=False):
    """Eq 1 (centered covariance form): for each HIDDEN layer, dW = lr * (pre - mean_pre)^T @ (teach - mean_teach) / B.
    `teach[l]` (B, n_post_l) is the tutor's per-neuron target for layer l's postsynaptic neurons; `pre[l]` (B,
    n_pre_l) the presynaptic activity. Centering both = the covariance/BCM-stable form; theta (Eq 1) = the batch-mean
    teaching. The OUTPUT layer either FOLLOWS `out_teach` Hebbian-style (Eq 1) OR, if delta_readout, uses a stronger
    LOCAL error-correcting delta rule toward the target (error = out_teach - out_rate) -- still reward-independent,
    still local; this is the STRONGEST readout, to prove the deep-credit negative is not a weak-readout artifact.
    Optional homeostatic renorm keeps each layer's column L2 at init (RA homeostasis). Reward-INDEPENDENT."""
    B = pre[0].shape[0]
    L = len(layers)
    for l in range(L - 1):                                      # hidden layers follow their broadcast teaching
        pc = pre[l] - pre[l].mean(axis=0, keepdims=True)
        tc = teach[l] - teach[l].mean(axis=0, keepdims=True)
        layers[l].W_in += lr * (pc.T @ tc) / B
    pc = out_pre - out_pre.mean(axis=0, keepdims=True)
    if delta_readout and out_rate is not None:                 # strong local delta rule toward the target
        err = out_teach - out_rate
        layers[L - 1].W_in += lr_out * (pc.T @ err) / B
    else:                                                      # Eq-1 Hebbian follow of the tutor target
        tc = out_teach - out_teach.mean(axis=0, keepdims=True)
        layers[L - 1].W_in += lr_out * (pc.T @ tc) / B
    if renorm_init_norms is not None:
        for l in range(L):
            cn = np.linalg.norm(layers[l].W_in, axis=0, keepdims=True)
            cn = np.where(cn < 1e-8, 1.0, cn)
            layers[l].W_in *= (renorm_init_norms[l] / cn)


def train_tutor_teach(Xtr, ytr, sizes, T, epochs, lr, lr_out, lr_tutor, in_gain, seed, k,
                      sigma=0.5, beta=1.0, baseline_lambda=0.05, batch=32, antithetic_k=1,
                      shuffle_tutor=False, permuted_reward=False, renorm=True, diag_y=None, oracle_tutor=False,
                      delta_readout=False):
    """Two-stage: low-dim tutor (u in R^k) learned by reward-NP+baseline (Eq 6); deep stack Hebbian-follows the
    fixed-random broadcast of u (Eq 1). Returns (layers, diag). `layers` = the student ALONE (tutor is a faded
    scaffold). diag reports whether STAGE A learned a useful target (tutor argmax accuracy on train)."""
    rng = np.random.default_rng(seed + 1)
    w_scales = [2.5] + [1.0] * (len(sizes) - 2)
    layers = _build_layers(sizes, T, rng, w_scales)
    n_in = sizes[0]
    L = len(layers)
    init_norms = [np.linalg.norm(l.W_in, axis=0, keepdims=True) if renorm else None for l in layers] if renorm else None
    # tutor: u = X @ U  (U low-dim -> class-space k). Node perturbation perturbs u (dim k), credits U via the input.
    trng = np.random.default_rng(seed + 4242)
    U = trng.normal(0.0, 0.01, (n_in, k)).astype(np.float64)
    # FIXED-RANDOM broadcast of the low-dim tutor to each HIDDEN layer's neurons (the deep extension).
    brng = np.random.default_rng(seed + 7777)
    M = [brng.normal(0.0, 1.0 / np.sqrt(k), (k, sizes[l + 1])).astype(np.float64) for l in range(L - 1)]
    Rbar = 0.0; Rstd = 1.0                                     # EMA reward baseline + scale (advantage normalization)
    tutor_wd = 1e-3                                            # weight decay -> bounded tutor (stable NP fixed point)
    n = len(Xtr)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for b0 in range(0, n, batch):
            bi = perm[b0:b0 + batch]
            Xb, yb = Xtr[bi], ytr[bi]
            Bn = len(bi)
            # ---- STAGE A: tutor forward + explore + reward + NP update (Eq 6) ----
            logits, pre = _layer_pre_activities(Xb, layers, T, in_gain)
            out_pre = pre[-1] if L > 1 else pre[0]              # activity into the output layer
            rate = logits / T                                  # mean output spike-RATE (~onehot scale; argmax-invariant)
            if not oracle_tutor:
                u = Xb @ U                                      # (B, k) tutor mean teaching
                gU = np.zeros_like(U)
                for _kk in range(antithetic_k):                 # antithetic node perturbation (variance reduction)
                    xi = rng.normal(0.0, sigma, (Bn, k))
                    Rp = _reward_negerr(rate + beta * (u + xi), yb, k)
                    Rm = _reward_negerr(rate + beta * (u - xi), yb, k)
                    adv = np.clip((Rp - Rm) / 2.0 / (Rstd + 1e-3), -5.0, 5.0)
                    if permuted_reward:                         # ANTI-CHEAT: reward mismatched to the example
                        adv = adv[rng.permutation(Bn)]
                    gU += Xb.T @ (adv[:, None] * xi) / (sigma * sigma * Bn)
                mR = float(_reward_negerr(rate + beta * u, yb, k).mean())
                Rbar = (1.0 - baseline_lambda) * Rbar + baseline_lambda * mR
                Rstd = (1.0 - baseline_lambda) * Rstd + baseline_lambda * float(
                    _reward_negerr(rate + beta * u, yb, k).std() + 1e-6)
                U += lr_tutor * (gU / antithetic_k) - tutor_wd * U
            # ---- STAGE B: student Hebbian-follows the tutor target (reward-INDEPENDENT, Eq 1, NO sigma') ----
            if oracle_tutor:                                    # STAGE-B ISOLATION: perfect target on train (a ceiling)
                u_now = np.zeros((Bn, k), dtype=np.float64); u_now[np.arange(Bn), yb] = 1.0
            else:
                u_now = Xb @ U                                 # tutor's CURRENT mean teaching (matched timescale)
            teach = [u_now @ M[l] for l in range(L - 1)]       # per-hidden-neuron broadcast target
            out_teach = u_now                                  # output follows the tutor class-space target
            if shuffle_tutor:                                  # ANTI-CHEAT: teaching mismatched to the example
                sp = rng.permutation(Bn)
                teach = [t[sp] for t in teach]
                out_teach = out_teach[sp]
            _hebbian_follow(layers, pre[:L - 1] if L > 1 else pre, teach, lr,
                            out_teach, out_pre, lr_out, init_norms,
                            out_rate=(logits / T), delta_readout=delta_readout)
    # STAGE-A diagnostic: did the tutor learn a useful target? (tutor argmax vs true label on TRAIN)
    diag = {"tutor_Rbar_final": float(Rbar), "tutor_u_absmean": float(np.abs(Xtr @ U).mean())}
    u_all = Xtr @ U
    diag["tutor_train_acc"] = float(np.mean(np.argmax(u_all, axis=1) == ytr))
    if diag_y is not None:
        Xhe, yhe = diag_y
        diag["tutor_heldout_acc"] = float(np.mean(np.argmax(Xhe @ U, axis=1) == yhe))
    return layers, diag


# ------------------------------------------------------------------------------------------------------------
# naive high-dim NODE PERTURBATION on the deep read-state (the REFUTED regime; reproduce its collapse).
# Genuine NP: a perturbed forward pass propagates the hidden perturbation to the output; credit by (R_pert-R_clean).
# Output layer trained by the clean delta-rule (it has a target y), hidden layers by NP only (as 2026-07-13).
# ------------------------------------------------------------------------------------------------------------
def train_naive_np(Xtr, ytr, sizes, T, epochs, lr, lr_out, in_gain, seed, k, sigma=0.5, batch=32, renorm=True):
    rng = np.random.default_rng(seed + 1)
    w_scales = [2.5] + [1.0] * (len(sizes) - 2)
    layers = _build_layers(sizes, T, rng, w_scales)
    L = len(layers)
    init_norms = [np.linalg.norm(l.W_in, axis=0, keepdims=True) for l in layers] if renorm else None
    n = len(Xtr)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for b0 in range(0, n, batch):
            bi = perm[b0:b0 + batch]
            Xb, yb = Xtr[bi], ytr[bi]
            Bn = len(bi)
            logits, pre = _layer_pre_activities(Xb, layers, T, in_gain)
            # perturb ALL hidden layers' activities (high-dim); propagate through a perturbed forward pass
            perts = [rng.normal(0.0, sigma, (Bn, sizes[l + 1])) for l in range(L - 1)] + [None]
            logits_p = _forward_logits_perturbed(Xb, layers, T, in_gain, perts)
            R_c = _reward_negerr(logits / T, yb, k)
            R_p = _reward_negerr(logits_p / T, yb, k)
            adv = (R_p - R_c)                                   # antithetic baseline = the clean rendition
            for l in range(L - 1):                              # hidden layers: NP credit (high-dim, variance ~ n_l)
                g = (adv[:, None] * perts[l]) / (sigma * sigma) # NP gradient estimate for layer l's activity
                pc = pre[l] - pre[l].mean(axis=0, keepdims=True)
                layers[l].W_in += lr * (pc.T @ g) / Bn
            # output layer: clean supervised delta rule on summed spikes (it HAS a target)
            p = _softmax(logits); delta = p.copy(); delta[np.arange(Bn), yb] -= 1.0
            out_pre = pre[-1]
            pc = out_pre - out_pre.mean(axis=0, keepdims=True)
            layers[L - 1].W_in -= lr_out * (pc.T @ delta) / Bn
            if init_norms is not None:
                for l in range(L):
                    cn = np.linalg.norm(layers[l].W_in, axis=0, keepdims=True)
                    cn = np.where(cn < 1e-8, 1.0, cn); layers[l].W_in *= (init_norms[l] / cn)
    return layers


def train_reservoir(Xtr, ytr, sizes, T, epochs, lr_out, in_gain, seed, renorm=True):
    """Frozen-reservoir FLOOR: hidden LIF layers FIXED at init; only the OUTPUT readout trained (supervised delta on
    summed spikes). Deep layers get NO credit -> the reservoir baseline the candidate must beat by >=0.10."""
    rng = np.random.default_rng(seed + 1)
    w_scales = [2.5] + [1.0] * (len(sizes) - 2)
    layers = _build_layers(sizes, T, rng, w_scales)
    L = len(layers)
    init_norm_out = np.linalg.norm(layers[-1].W_in, axis=0, keepdims=True) if renorm else None
    n = len(Xtr)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for b0 in range(0, n, 32):
            bi = perm[b0:b0 + 32]
            Xb, yb = Xtr[bi], ytr[bi]
            logits, pre = _layer_pre_activities(Xb, layers, T, in_gain)
            p = _softmax(logits); delta = p.copy(); delta[np.arange(len(bi)), yb] -= 1.0
            out_pre = pre[-1]
            pc = out_pre - out_pre.mean(axis=0, keepdims=True)
            layers[L - 1].W_in -= lr_out * (pc.T @ delta) / len(bi)
            if init_norm_out is not None:
                cn = np.linalg.norm(layers[-1].W_in, axis=0, keepdims=True)
                cn = np.where(cn < 1e-8, 1.0, cn); layers[-1].W_in *= (init_norm_out / cn)
    return layers


def _enter_regime(X, y, layers, T, in_gain, sub):
    """ENTER-THE-REGIME read: does the student leave majority-class? Returns (inherit_acc, majority_pred_frac,
    n_distinct_pred). majority_pred_frac<0.9 => varied predictions (not collapsed to one class)."""
    if sub is not None:
        X = X[sub]; y = y[sub]
    if len(X) == 0:
        return float("nan"), float("nan"), 0
    logits, _, _ = _forward_logits(X, layers, T, in_gain)
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == y))
    vals, cnts = np.unique(pred, return_counts=True)
    return acc, float(cnts.max() / len(pred)), int(len(vals))


def run_seed(seed, hidden, T, epochs, lr, lr_out, lr_tutor, in_gain, subsample, task_kwargs,
             n_hidden_layers, sigma, beta, arms, renorm=True, antithetic_k=1, delta_readout=False):
    task_full = make_task_semantic_inheritance(seed, **task_kwargs)
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = task_full
    k = meta["k_classes"]; n_in = Xtr.shape[1]; inh_idx = idx["inh_idx"]
    s0 = stage0_depth_genuineness(((Xtr, ytr, Ltr), (Xte, yte, Lte)), idx, k, hidden=96, epochs=250,
                                  lr=0.3, batch=128, seed=seed)
    if len(inh_idx):
        yv = yte[inh_idx]; chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    else:
        chance = float("nan")
    majority = chance  # majority-class == max class frequency on the inheritance held-out subset
    onet = DendriticMLP([n_in] + [96] * n_hidden_layers + [k], seed=seed)
    _train_oracle(onet, Xtr, ytr, 250, 0.3, 128, seed)
    oracle_inh = _acc_on(onet, Xte, yte, inh_idx)

    if subsample is not None and len(Xtr) > subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:subsample]
        Xtr, ytr = Xtr[keep], ytr[keep]

    sizes = [n_in] + [hidden] * n_hidden_layers + [k]
    res = {"seed": seed, "chance": chance, "majority": majority, "k_classes": int(k),
           "stage0_depth_separating": bool(s0.get("depth_separating")), "oracle_inherit": oracle_inh,
           "n_hidden_layers": n_hidden_layers}

    def _eval(layers, name):
        acc, mpf, ndp = _enter_regime(Xte, yte, layers, T, in_gain, inh_idx)
        tr = _accuracy(Xtr, ytr, layers, T, in_gain)
        res[f"{name}_inherit"] = acc; res[f"{name}_train"] = tr
        res[f"{name}_majority_pred_frac"] = mpf; res[f"{name}_n_distinct_pred"] = ndp
        return acc, mpf

    diag_y = (Xte[inh_idx], yte[inh_idx]) if len(inh_idx) else None
    if "reservoir" in arms:
        _eval(train_reservoir(Xtr, ytr, sizes, T, epochs, lr_out, in_gain, seed, renorm=renorm), "reservoir")
    if "bptt" in arms:
        _eval(_train_snn(Xtr, ytr, sizes, T, epochs, lr, in_gain, seed, credit_mode="bptt"), "bptt")
    if "oracle_tutor" in arms:   # STAGE-B ISOLATION ceiling: perfect onehot target on train, deep Hebbian-follow, no sigma'
        ly, _ = train_tutor_teach(Xtr, ytr, sizes, T, epochs, lr, lr_out, lr_tutor, in_gain, seed, k,
                                  sigma=sigma, beta=beta, renorm=renorm, oracle_tutor=True, delta_readout=delta_readout)
        _eval(ly, "oracle_tutor")
    if "tutor_teach" in arms:
        ly, dg = train_tutor_teach(Xtr, ytr, sizes, T, epochs, lr, lr_out, lr_tutor, in_gain, seed, k,
                                   sigma=sigma, beta=beta, renorm=renorm, diag_y=diag_y, antithetic_k=antithetic_k,
                                   delta_readout=delta_readout)
        _eval(ly, "tutor_teach"); res["tutor_diag"] = dg
    if "shuffle_tutor" in arms:
        ly, _ = train_tutor_teach(Xtr, ytr, sizes, T, epochs, lr, lr_out, lr_tutor, in_gain, seed, k,
                                  sigma=sigma, beta=beta, shuffle_tutor=True, renorm=renorm, antithetic_k=antithetic_k)
        _eval(ly, "shuffle_tutor")
    if "permuted_reward" in arms:
        ly, dg = train_tutor_teach(Xtr, ytr, sizes, T, epochs, lr, lr_out, lr_tutor, in_gain, seed, k,
                                   sigma=sigma, beta=beta, permuted_reward=True, renorm=renorm, diag_y=diag_y,
                                   antithetic_k=antithetic_k)
        _eval(ly, "permuted_reward"); res["permuted_reward_tutor_diag"] = dg
    if "naive_np" in arms:
        _eval(train_naive_np(Xtr, ytr, sizes, T, epochs, lr, lr_out, in_gain, seed, k, sigma=sigma, renorm=renorm),
              "naive_np")

    # ENTER-THE-REGIME + GO verdict for the candidate
    tt = res.get("tutor_teach_inherit", float("nan"))
    rv = res.get("reservoir_inherit", float("nan"))
    res["enters_regime"] = bool((not np.isnan(tt)) and tt > majority + 0.03
                                and res.get("tutor_teach_majority_pred_frac", 1.0) < 0.9)
    res["beats_reservoir_by"] = float(tt - rv) if not (np.isnan(tt) or np.isnan(rv)) else float("nan")
    res["above_chance_by"] = float(tt - chance) if not np.isnan(tt) else float("nan")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-hidden-layers", type=int, default=2)
    ap.add_argument("--timesteps", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.05, help="student HIDDEN Hebbian lr (also the BPTT ceiling lr)")
    ap.add_argument("--lr-out", type=float, default=0.05, help="student OUTPUT readout lr (tutor_teach/reservoir/naive)")
    ap.add_argument("--lr-tutor", type=float, default=0.2, help="tutor node-perturbation lr (Eq 6)")
    ap.add_argument("--sigma", type=float, default=0.5, help="tutor/naive-NP exploration std")
    ap.add_argument("--beta", type=float, default=2.0, help="tutor->output bias gain (LMAN drive on RA)")
    ap.add_argument("--antithetic-k", type=int, default=1, help="antithetic node-perturbation samples (variance lever)")
    ap.add_argument("--in-gain", type=float, default=1.0)
    ap.add_argument("--train-subsample", type=int, default=400)
    ap.add_argument("--arms", type=str, nargs="+",
                    default=["reservoir", "bptt", "oracle_tutor", "tutor_teach", "shuffle_tutor",
                             "permuted_reward", "naive_np"])
    ap.add_argument("--n-super", type=int, default=12)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=2)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--no-renorm", action="store_true", help="disable the homeostatic column-L2 renorm-to-init")
    ap.add_argument("--delta-readout", action="store_true",
                    help="STRONG local error-correcting delta readout toward the target (still reward-independent/local)")
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    task_kwargs = {"n_super": args.n_super, "n_members": args.n_members, "held_per_super": args.held_per_super,
                   "n_prop": args.n_prop, "member_id_dim": args.member_id_dim, "n_obs": args.n_obs,
                   "noise": args.noise, "feature_seed": args.feature_seed}
    t0 = time.time()
    results = []
    for sd in args.seeds:
        try:
            r = run_seed(sd, args.hidden, args.timesteps, args.epochs, args.lr, args.lr_out, args.lr_tutor,
                         args.in_gain, args.train_subsample, task_kwargs, args.n_hidden_layers,
                         args.sigma, args.beta, args.arms, renorm=(not args.no_renorm),
                         antithetic_k=args.antithetic_k, delta_readout=args.delta_readout)
        except Exception as e:
            r = {"seed": sd, "error": repr(e), "traceback": traceback.format_exc()}
        results.append(r)
        if "error" in r:
            print(f"[seed {sd}] ERROR {r['error']}")
        else:
            print(f"[seed {sd}] N={r['n_hidden_layers']} chance={r['chance']:.3f} oracle={r['oracle_inherit']:.3f} "
                  f"| tutor={r.get('tutor_teach_inherit', float('nan')):.3f} "
                  f"oracleT={r.get('oracle_tutor_inherit', float('nan')):.3f} "
                  f"resv={r.get('reservoir_inherit', float('nan')):.3f} bptt={r.get('bptt_inherit', float('nan')):.3f} "
                  f"| shuf={r.get('shuffle_tutor_inherit', float('nan')):.3f} "
                  f"permR={r.get('permuted_reward_inherit', float('nan')):.3f} "
                  f"naiveNP={r.get('naive_np_inherit', float('nan')):.3f} "
                  f"| enters={r.get('enters_regime')} beats_resv={r.get('beats_reservoir_by', float('nan')):+.3f}")
            dg = r.get("tutor_diag")
            if dg:
                print(f"         STAGE-A tutor: Rbar={dg['tutor_Rbar_final']:.3f} |u|={dg['tutor_u_absmean']:.3f} "
                      f"tutor_train_acc={dg['tutor_train_acc']:.3f} "
                      f"tutor_heldout_acc={dg.get('tutor_heldout_acc', float('nan')):.3f}")

    out = {"probe": "gap4_birdsong_two_stage_tutor_teach", "seeds": args.seeds,
           "config": {"hidden": args.hidden, "n_hidden_layers": args.n_hidden_layers, "T": args.timesteps,
                      "epochs": args.epochs, "lr": args.lr, "lr_out": args.lr_out, "lr_tutor": args.lr_tutor,
                      "sigma": args.sigma, "beta": args.beta, "in_gain": args.in_gain,
                      "train_subsample": args.train_subsample, "task": task_kwargs, "arms": args.arms},
           "elapsed_seconds": round(time.time() - t0, 1), "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[tutor_teach] wrote {args.out} ({out['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
