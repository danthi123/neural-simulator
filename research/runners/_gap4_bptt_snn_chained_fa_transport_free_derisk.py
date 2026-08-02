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

THE CATEGORICAL-UNLOCK TEST (--task-xor, ADDITIVE, default OFF). The inheritance task above is linearly reservoir-
decodable GIVEN WIDTH (a wide-256 optimally-read frozen reservoir reaches 0.840 > chained_fa 0.778), so the matched-
width purchase is partly "denied width", NOT proof that directed credit does what NO reservoir can. `--task-xor` swaps
in the depth-2 XOR->threshold RATE-OVERTURN task (reuse-by-import of emerge1's `make_task`, the SAME task the rate
overturn used): XOR is NOT linearly separable, so a FIXED random projection + linear readout PROVABLY cannot decode it
at ANY width -> a WIDE frozen reservoir must FAIL where a TRAINED hidden can compute it. Two categorical controls are
added EVERY run: the frozen-reservoir OPTIMAL 5-fold-CV ridge readout at MATCHED width (--hidden) AND WIDE width
(--wide-hidden). The DECISIVE read: chained_fa >> WIDE-frozen-optimal AND WIDE-frozen-optimal ~ chance => CATEGORICAL
UNLOCK (directed credit does what no reservoir can); if WIDE-frozen-optimal still solves it => still reservoir-
decodable (pick a harder task). Headroom: a PROPERLY-TUNED BPTT arm (--bptt-hidden/--bptt-epochs = the 0.82-source
config) confirms the task IS learnable by a trained net (the crux run under-tuned BPTT -> its ceiling was invalid).

Run (numpy CPU; the depth-2 BPTT-viable net, ~ (n_in+2*hidden+k) LIF units over T steps):
    # inheritance task (byte-identical to the crux 6-seed run):
    SIM_BACKEND=numpy python -m research.runners._gap4_bptt_snn_chained_fa_transport_free_derisk --seeds 42
    # THE CATEGORICAL-UNLOCK XOR test, seed 42 (matched width 32, wide 256, BPTT ceiling at the 0.82-source config):
    SIM_BACKEND=numpy python -m research.runners._gap4_bptt_snn_chained_fa_transport_free_derisk --task-xor \
        --seeds 42 --hidden 32 --wide-hidden 256 --epochs 200 --lr 0.05 --bptt-hidden 128 --train-subsample 2000 \
        --out research/findings/raw/gap4/realspikes/bptt_snn_chained_fa_XOR_seed42.json
    # 6-seed:
    SIM_BACKEND=numpy python -m research.runners._gap4_bptt_snn_chained_fa_transport_free_derisk --task-xor \
        --seeds 42 43 44 100 101 102 --hidden 32 --wide-hidden 256 --epochs 200 --lr 0.05 --bptt-hidden 128 \
        --train-subsample 2000 --out research/findings/raw/gap4/realspikes/bptt_snn_chained_fa_XOR_6seed.json
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
# ---- reuse-by-import: the EXACT depth-2 XOR->threshold RATE-OVERTURN task (--task-xor). `make_task` here is the SAME
#      function `_gap4_depth2_bdsp_credit_derisk` imports (it re-exports emerge1's), so the ported task is byte-identical
#      to the rate overturn's `make_task`: pair-XOR (level-1 latents, XOR is NOT linearly separable) -> threshold-over-
#      XORs (level-2). A fixed random projection + linear readout PROVABLY cannot decode XOR => a WIDE frozen reservoir
#      must FAIL where a TRAINED hidden can compute it. Held-out = UNSEEN bit patterns (systematic generalization). ----
from research.runners._emerge1_deep_dendritic_representation_derisk import (  # noqa: E402
    make_task as _make_task_xor_raw, N_BITS as _XOR_N_BITS, N_PAIRS as _XOR_N_PAIRS)

OUT = _REPO / "research" / "findings" / "raw" / "_gap4_bptt_snn_chained_fa.json"


def make_task_xor(seed):
    """Wrap the imported depth-2 XOR->threshold rate-overturn task into the 4-tuple interface run_seed consumes
    (the SAME shape as make_task_semantic_inheritance). X is +/-1 over `_XOR_N_BITS` bits (rate-coded as constant
    current over T by _forward_logits, exactly as the inheritance task's real-valued X is); y in {0,1} (k=2). The
    WHOLE held-out set is the 'inherit' generalization set (UNSEEN bit patterns) -> inh_idx = all held rows; there
    is no memorization-control split for this task (memctrl_idx empty). Latents = the level-1 pair-XORs (reported,
    not gating)."""
    (Xtr, ytr, Ltr), (Xte, yte, Lte) = _make_task_xor_raw(seed)
    inh_idx = np.arange(len(Xte), dtype=np.int64)
    meta = {"task": "depth2_xor_threshold", "k_classes": 2, "n_bits": int(_XOR_N_BITS),
            "n_pairs": int(_XOR_N_PAIRS), "n_features": int(Xtr.shape[1]),
            "n_train": int(len(Xtr)), "n_heldout": int(len(Xte)), "n_inherit_heldout": int(len(inh_idx))}
    idx = {"inh_idx": inh_idx, "memctrl_idx": np.array([], dtype=np.int64)}
    return (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx


# ============================================================================================================
# THE DEPTH-3 OBLIGATORY task (--task-nestedxor, ADDITIVE, default OFF) -- the CLEAN KP-depth-rescue test the XOR
# depth-sweep could NOT do. XOR only REQUIRES depth-2, so at --n-hidden-layers 3/4 layers 3-4 are REDUNDANT (credit
# degrades through redundant hops; NOT a test of KP's obligatory-depth rescue). This task genuinely REQUIRES 3
# STACKED nonlinear layers so that at N>=3 the depth is OBLIGATORY, not redundant:
#     L1  p_j    = XOR(b_{2j}, b_{2j+1})                          j in 0..NX_N_PAIRS-1     (nonlinear: pair-XOR)
#     L2  g_m    = MAJORITY(p over disjoint group m of NX_GROUP)  m in 0..NX_N_GROUPS-1    (nonlinear: threshold)
#     L3  label  = XOR over the NX_N_GROUPS group-majorities g_m                            (nonlinear: top XOR)
# WHY IT DOES NOT COLLAPSE (the trap a pure nested-XOR falls into): XOR(XOR(...)) folds to a WIDE parity, which a
# single hidden layer represents -> a nested-XOR tree is depth-1, NOT depth-3. Inserting a MAJORITY between the XOR
# levels breaks the parity algebra (MAJ != parity), so the three operations cannot be folded: computing g_m needs
# XOR (L1) THEN threshold (L2); XORing the g_m needs a THIRD nonlinearity. Expectation on the stage0 probe: the
# depth-2 oracle UNDERFITS held-out (l2 ~ chance) while the depth-3 oracle CLEARS it (l3 >= 0.80) -> depth3_requiring.
# ============================================================================================================
NX_N_BITS = 12                         # 2^12 = 4096 patterns (enumerable; same order as the emerge1 XOR task's 2^10)
NX_N_PAIRS = NX_N_BITS // 2            # = 6  level-1 pair-XORs (XOR of adjacent bit pairs)
NX_GROUP = 3                          # level-2 MAJORITY over disjoint groups of 3 pair-XORs (odd -> non-degenerate)
NX_N_GROUPS = NX_N_PAIRS // NX_GROUP  # = 2  group-majorities; level-3 = XOR over these


def make_task_nestedxor(seed):
    """DEPTH-3 nested task: bits -> pair-XORs (L1) -> group-MAJORITIES (L2) -> XOR-over-groups (L3 = label). Enumerate
    all 2^NX_N_BITS patterns; three STACKED non-collapsing nonlinear operations (a MAJORITY between the two XOR levels
    prevents the parity fold that would make a nested-XOR tree depth-1). A depth-2 net should be UNABLE to fit it while
    a depth-3 net can -- confirm via stage0 (l2 ~ chance, l3 solves). Same 4-tuple interface as make_task_xor: X in
    +/-1 (rate-coded as constant current over T, exactly as the XOR task's X is), y in {0,1} (k=2); the WHOLE held-out
    set (UNSEEN bit patterns) is the 'inherit' generalization set; latents reported = the level-1 pair-XORs (non-gating)."""
    rng = np.random.default_rng(seed)
    n = 1 << NX_N_BITS
    bits = ((np.arange(n)[:, None] >> np.arange(NX_N_BITS)[None, :]) & 1).astype(np.float64)     # (n, N_BITS) {0,1}
    pair_xor = np.logical_xor(bits[:, 0::2].astype(bool), bits[:, 1::2].astype(bool))            # (n, N_PAIRS) bool
    thr = (NX_GROUP + 1) // 2                                                                    # majority >= ceil(G/2)
    groups = pair_xor.reshape(n, NX_N_GROUPS, NX_GROUP)                                          # (n, M, G)
    g_maj = (groups.sum(axis=2) >= thr)                                                          # (n, M) bool -- L2
    label = np.bitwise_xor.reduce(g_maj.astype(np.int64), axis=1).astype(np.int64)              # (n,) {0,1} -- L3
    X = bits * 2.0 - 1.0                                                                         # +/-1
    idx = rng.permutation(n)
    cut = int(0.65 * n)
    tr, te = idx[:cut], idx[cut:]
    Ltr = pair_xor[tr].astype(np.float64); Lte = pair_xor[te].astype(np.float64)                 # level-1 latents
    inh_idx = np.arange(len(te), dtype=np.int64)                                                 # whole held-out set
    meta = {"task": "nestedxor_depth3", "k_classes": 2, "n_bits": int(NX_N_BITS),
            "n_pairs": int(NX_N_PAIRS), "n_groups": int(NX_N_GROUPS), "group_size": int(NX_GROUP),
            "n_features": int(X.shape[1]), "n_train": int(len(tr)), "n_heldout": int(len(te)),
            "n_inherit_heldout": int(len(inh_idx))}
    idxd = {"inh_idx": inh_idx, "memctrl_idx": np.array([], dtype=np.int64)}
    return (X[tr], label[tr], Ltr), (X[te], label[te], Lte), meta, idxd


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
# CATEGORICAL controls -- the frozen-reservoir OPTIMAL ridge readout at MATCHED width AND WIDE width. The frozen_reservoir
# ARM reads its frozen hidden with a WEAK local-delta; these controls read the SAME frozen hidden OPTIMALLY (5-fold-CV
# ridge over the CONCATENATED hidden-layer summed-spikes = the most generous read: every hidden rep, as the verification
# did). ON A LINEARLY RESERVOIR-DECODABLE TASK a WIDE (256) frozen reservoir, optimally read, can MATCH/EXCEED directed
# credit (the inheritance caveat: wide-256 reached 0.840 > chained_fa 0.778 -> "denied width", NOT a categorical unlock).
# ON A TASK NOT LINEARLY DECODABLE FROM A FIXED RANDOM PROJECTION (XOR) a fixed random reservoir provably CANNOT decode
# it at ANY width -> wide-frozen-optimal should sit at CHANCE, and chained_fa beating it == the CATEGORICAL unlock.
# ============================================================================================================
def _reservoir_features(X, layers, T, in_gain):
    """Forward X through the FROZEN LIF layers; return the CONCATENATED summed-spike rate vector of ALL hidden layers
    (excludes the output layer) -- the most generous reservoir read-out (reads every hidden rep, matching the
    verification's 'reading BOTH hidden reps optimally')."""
    _, fs, _ = _forward_logits(X, layers, T, in_gain)
    feats = [fs["spikes"][li].sum(axis=0) for li in range(len(layers) - 1)]     # hidden layers only (exclude output)
    return np.concatenate(feats, axis=1)                                        # (B, sum of hidden widths)


def _ridge_predict(Htr, Ytr_oh, Hte, lam):
    d = Htr.shape[1]
    W = np.linalg.solve(Htr.T @ Htr + lam * np.eye(d), Htr.T @ Ytr_oh)
    return Hte @ W


def _optimal_ridge_acc(Htr, ytr, Hte, yte, k, seed, n_folds=5,
                       lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """OPTIMAL linear readout of the frozen reservoir: one-hot ridge regression with 5-fold-CV lambda selection on the
    TRAIN reservoir features, refit at the best lambda, argmax accuracy on the held-out reservoir features. The
    reservoir's BEST-possible linear decode (the FAIR categorical control the verification used)."""
    Htr = np.concatenate([Htr, np.ones((len(Htr), 1))], axis=1)                 # bias column
    Hte = np.concatenate([Hte, np.ones((len(Hte), 1))], axis=1)
    Ytr_oh = np.eye(k)[ytr]
    idxp = np.random.default_rng(seed + 31).permutation(len(Htr))
    folds = np.array_split(idxp, n_folds)
    best_lam, best_cv = float(lams[0]), -1.0
    for lam in lams:
        accs = []
        for f in folds:
            m = np.ones(len(Htr), dtype=bool); m[f] = False
            if not m.any() or len(f) == 0:
                continue
            pred = _ridge_predict(Htr[m], Ytr_oh[m], Htr[f], lam)
            accs.append(float(np.mean(np.argmax(pred, axis=1) == ytr[f])))
        cv = float(np.mean(accs)) if accs else -1.0
        if cv > best_cv:
            best_cv, best_lam = cv, float(lam)
    pred = _ridge_predict(Htr, Ytr_oh, Hte, best_lam)
    return float(np.mean(np.argmax(pred, axis=1) == yte)), best_lam


def _frozen_reservoir_optimal(Xtr, ytr, Xte_inh, yte_inh, n_in, k, res_width, n_hidden_layers, T, in_gain, seed):
    """Build a FROZEN random LIF reservoir of width `res_width`, forward TRAIN + held-out INHERIT through it, fit the
    OPTIMAL 5-fold-CV ridge readout. Returns (inherit accuracy, best lambda). Uses the SAME init RNG stream (seed+1) as
    the frozen_reservoir arm, so at res_width==hidden it reads LITERALLY that arm's frozen hidden read optimally."""
    rrng = np.random.default_rng(seed + 1)                                      # same init stream as _train_snn_arm
    res_sizes = [n_in] + [res_width] * n_hidden_layers + [k]
    w_scales = [2.5] + [1.0] * (len(res_sizes) - 2)
    res_layers = _build_layers(res_sizes, T, rrng, w_scales)                    # FROZEN at init (never trained)
    Htr = _reservoir_features(Xtr, res_layers, T, in_gain)
    Hte = _reservoir_features(Xte_inh, res_layers, T, in_gain)
    return _optimal_ridge_acc(Htr, ytr, Hte, yte_inh, k, seed)


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
             sigma_norm=True, kp_lr=0.2, kp_decay=1e-4, check_depth=True,
             task_xor=False, task_nestedxor=False, bptt_hidden=None, bptt_epochs=None, bptt_lr=None,
             wide_hidden=256):
    if task_nestedxor:
        (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_nestedxor(seed)
    elif task_xor:
        (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_xor(seed)
    else:
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
    s0_l0 = s0_l1 = s0_l2 = s0_l3 = s0_gap = None
    depth3_requiring = None
    if check_depth:
        s0 = stage0_depth_genuineness(((Xtr, ytr, Ltr), (Xte, yte, Lte)), idx, k, hidden=96, epochs=250,
                                      lr=0.3, batch=128, seed=seed)
        depth_sep = bool(s0.get("depth_separating"))
        s0_l0 = s0.get("linear_inherit_heldout"); s0_l1 = s0.get("l1_inherit_heldout")
        s0_l2 = s0.get("l2_inherit_heldout"); s0_l3 = s0.get("l3_inherit_heldout")
        s0_gap = s0.get("depth_gap")
        # DEPTH-3 GATE (the confirm read the XOR sweep could not make): depth-2 oracle ~ chance AND depth-3 oracle
        # solves it AND a clear 2->3 jump. This is STRICTER than `depth_separating` (which fires on max(l2,l3), so it
        # is True even for a depth-2 task); depth3_requiring is what validates the task as OBLIGATORY-depth-3.
        if not np.isnan(chance) and s0_l2 is not None and s0_l3 is not None:
            depth3_requiring = bool(s0_l2 <= chance + 0.06 and s0_l3 >= 0.80 and (s0_l3 - s0_l2) >= 0.15)

    if subsample is not None and len(Xtr) > subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:subsample]
        Xtr, ytr = Xtr[keep], ytr[keep]

    sizes = [n_in] + [hidden] * n_hidden_layers + [k]

    def _run(mode, ylabels, sizes_=None, epochs_=None, lr_=None):
        layers, Y_list = _train_snn_arm(Xtr, ylabels, sizes_ if sizes_ is not None else sizes, T,
                                        epochs_ if epochs_ is not None else epochs,
                                        lr_ if lr_ is not None else lr, lr_fa, in_gain, seed, mode,
                                        sigma_norm=sigma_norm, kp_lr=kp_lr, kp_decay=kp_decay)
        inh = _accuracy(Xte, yte, layers, T, in_gain, sub=inh_idx)
        tr = _accuracy(Xtr, ytr, layers, T, in_gain)
        nt = _no_weight_transport(Y_list, layers) if Y_list is not None else True
        return {"inherit": inh, "train": tr, "no_transport": bool(nt)}, layers, Y_list

    # ---- ARM 1: surrogate-BPTT ceiling -- PROPERLY TUNED (its own hidden/epochs/lr so the headroom/ceiling is VALID;
    #      the crux run under-tuned BPTT (train 0.665) -> invalid ceiling. Defaults None => matched config (byte-identical
    #      to the prior run); the XOR command passes the 0.82-source epochs/lr). ----
    bptt_sizes = [n_in] + [(bptt_hidden if bptt_hidden is not None else hidden)] * n_hidden_layers + [k]
    bptt, _, _ = _run("bptt", ytr, sizes_=bptt_sizes, epochs_=bptt_epochs, lr_=bptt_lr)
    # ---- ARM 3: frozen-hidden reservoir (only the output LIF layer learns, weak local delta) ----
    frozen, _, _ = _run("frozen_reservoir", ytr)
    # ---- ARM 2: chained transport-free FIXED-random FA (the primary transport-free-local arm) ----
    chained, _, Yc = _run("chained_fa", ytr)
    # ---- ARM 2b: chained transport-free KP-LEARNED FA (the depth rescue) ----
    chained_kp, _, _ = _run("chained_fa_kp", ytr)
    # ---- ANTI-CHEAT: permuted labels through the chained-FA arm -> chance ----
    prng = np.random.default_rng(seed + 555)
    yperm = ytr[prng.permutation(len(ytr))]
    permuted, _, _ = _run("chained_fa", yperm)

    # ---- CATEGORICAL controls: the frozen-reservoir OPTIMAL ridge readout at MATCHED width AND WIDE width. The
    #      decisive question is whether chained_fa beats the WIDE frozen reservoir OPTIMALLY read -- if the wide
    #      reservoir now FAILS (XOR is not linearly decodable from a fixed random projection) while chained_fa wins,
    #      that is the categorical unlock; if the wide reservoir still solves it, the task is still reservoir-decodable.
    Xte_inh = Xte[inh_idx]; yte_inh = yte[inh_idx]
    frozen_opt_matched, lam_m = _frozen_reservoir_optimal(Xtr, ytr, Xte_inh, yte_inh, n_in, k, hidden,
                                                          n_hidden_layers, T, in_gain, seed)
    frozen_opt_wide, lam_w = _frozen_reservoir_optimal(Xtr, ytr, Xte_inh, yte_inh, n_in, k, wide_hidden,
                                                       n_hidden_layers, T, in_gain, seed)

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

    # THE CATEGORICAL-UNLOCK read: chained_fa >> wide-frozen-optimal AND wide-frozen-optimal ~ chance.
    chained_over_wide_optimal = float(chained["inherit"] - frozen_opt_wide)
    chained_over_matched_optimal = float(chained["inherit"] - frozen_opt_matched)
    if not np.isnan(chance):
        wide_optimal_over_chance = float(frozen_opt_wide - chance)
        wide_optimal_at_chance = bool(abs(frozen_opt_wide - chance) <= 0.06)
        bptt_solves_task = bool(bptt["inherit"] > chance + 0.15)          # headroom: XOR IS learnable by a trained net
        categorical_unlock = bool(chained["inherit"] > frozen_opt_wide + 0.10 and wide_optimal_at_chance)
    else:
        wide_optimal_over_chance = float("nan"); wide_optimal_at_chance = None
        bptt_solves_task = None; categorical_unlock = None

    return {
        "seed": seed, "chance": chance, "n_in": n_in, "k": k, "sizes": sizes,
        "task": ("nestedxor" if task_nestedxor else ("xor" if task_xor else "inheritance")),
        "n_hidden_layers": int(n_hidden_layers),
        "depth_separating": depth_sep, "oracle_inherit": oracle_inh,
        # ---- stage0 depth-genuineness surfaced (the DEPTH-3 confirm read) ----
        "stage0_linear_inherit": s0_l0, "stage0_l1_inherit": s0_l1,
        "stage0_l2_inherit": s0_l2, "stage0_l3_inherit": s0_l3, "stage0_depth_gap": s0_gap,
        "depth3_requiring": depth3_requiring,
        "bptt_inherit": bptt["inherit"], "bptt_train": bptt["train"], "bptt_sizes": bptt_sizes,
        "frozen_reservoir_inherit": frozen["inherit"], "frozen_reservoir_train": frozen["train"],
        "chained_fa_inherit": chained["inherit"], "chained_fa_train": chained["train"],
        "chained_fa_kp_inherit": chained_kp["inherit"], "chained_fa_kp_train": chained_kp["train"],
        "permuted_inherit": permuted["inherit"],
        # ---- the CATEGORICAL controls ----
        "frozen_optimal_matched_inherit": frozen_opt_matched, "frozen_optimal_matched_lambda": lam_m,
        "frozen_optimal_wide_inherit": frozen_opt_wide, "frozen_optimal_wide_lambda": lam_w,
        "wide_hidden": wide_hidden,
        "chained_over_wide_optimal": chained_over_wide_optimal,
        "chained_over_matched_optimal": chained_over_matched_optimal,
        "wide_optimal_over_chance": wide_optimal_over_chance,
        "wide_optimal_at_chance": wide_optimal_at_chance,
        "bptt_solves_task": bptt_solves_task,
        "CATEGORICAL_UNLOCK": categorical_unlock,
        "directed_over_permuted": directed_over_permuted,
        "purchase_over_frozen": purchase_over_frozen,
        "bptt_fraction_captured": _frac(chained["inherit"]),
        "kp_directed_over_permuted": kp_directed_over_permuted,
        "kp_purchase_over_frozen": kp_purchase_over_frozen,
        "kp_bptt_fraction_captured": _frac(chained_kp["inherit"]),
        # THE DEPTH-SWEEP DECISIVE metric: the KP-LEARNED-over-FIXED-random-FA margin. The rate depth-rescue signature
        # is this margin GROWING with n_hidden_layers (fixed FA degrades faster than KP as depth grows); read it at
        # N=2,3,4 across runs. Derivable from chained_fa_kp_inherit - chained_fa_inherit; emitted explicitly here so the
        # cross-depth read needs no per-file arithmetic.
        "kp_over_fixed_fa": float(chained_kp["inherit"] - chained["inherit"]),
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
    def _count(key):
        return sum(1 for r in ok if r.get(key) is True)
    return {
        "n_seeds": n,
        "mean_chance": _m("chance"),
        # ---- stage0 depth-3 confirm (the gate for the OBLIGATORY-depth-3 task) ----
        "mean_stage0_l2_inherit": _m("stage0_l2_inherit"),
        "mean_stage0_l3_inherit": _m("stage0_l3_inherit"),
        "depth3_requiring_seeds": f"{_count('depth3_requiring')}/{n}",
        "mean_oracle_inherit": _m("oracle_inherit"),
        "mean_bptt_inherit": _m("bptt_inherit"),
        "mean_frozen_reservoir_inherit": _m("frozen_reservoir_inherit"),
        "mean_chained_fa_inherit": _m("chained_fa_inherit"),
        "mean_chained_fa_kp_inherit": _m("chained_fa_kp_inherit"),
        "mean_permuted_inherit": _m("permuted_inherit"),
        # ---- the CATEGORICAL controls (the decisive read) ----
        "mean_frozen_optimal_matched_inherit": _m("frozen_optimal_matched_inherit"),
        "mean_frozen_optimal_wide_inherit": _m("frozen_optimal_wide_inherit"),
        "mean_chained_over_wide_optimal": _m("chained_over_wide_optimal"),
        "mean_chained_over_matched_optimal": _m("chained_over_matched_optimal"),
        "mean_wide_optimal_over_chance": _m("wide_optimal_over_chance"),
        "wide_optimal_at_chance_seeds": f"{_count('wide_optimal_at_chance')}/{n}",
        "bptt_solves_task_seeds": f"{_count('bptt_solves_task')}/{n}",
        "CATEGORICAL_UNLOCK_seeds": f"{_count('CATEGORICAL_UNLOCK')}/{n}",
        "mean_directed_over_permuted": _m("directed_over_permuted"),
        "mean_purchase_over_frozen": _m("purchase_over_frozen"),
        "mean_bptt_fraction_captured": _m("bptt_fraction_captured"),
        "mean_kp_directed_over_permuted": _m("kp_directed_over_permuted"),
        "mean_kp_purchase_over_frozen": _m("kp_purchase_over_frozen"),
        # the depth-sweep decisive metric aggregated: mean KP-over-fixed-FA margin (compare across N=2,3,4).
        "mean_kp_over_fixed_fa": _m("kp_over_fixed_fa"),
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
    # ---- ADDITIVE (default-OFF): the CATEGORICAL-unlock test on the depth-2 XOR->threshold task ----
    ap.add_argument("--task-xor", action="store_true",
                    help="use the depth-2 XOR->threshold RATE-OVERTURN task (NOT linearly reservoir-decodable) instead "
                         "of the semantic-inheritance task; default OFF => inheritance task byte-identical to the crux run.")
    # ---- ADDITIVE (default-OFF): the DEPTH-3 OBLIGATORY task -- the clean KP-depth-rescue test the XOR sweep could not do ----
    ap.add_argument("--task-nestedxor", action="store_true",
                    help="use the DEPTH-3 nested task (bits->pair-XORs->group-MAJORITIES->XOR-over-groups; genuinely "
                         "REQUIRES 3 nonlinear layers so at --n-hidden-layers>=3 the depth is OBLIGATORY, not redundant) "
                         "instead of the inheritance/xor task; default OFF. CONFIRM the depth-3 requirement via the stage0 "
                         "probe (l2 ~ chance, l3 solves => depth3_requiring) BEFORE reading the depth sweep.")
    ap.add_argument("--stage0-only", action="store_true",
                    help="run ONLY the stage0 depth-genuineness probe (0/1/2/3-hidden oracles) + the depth3 gate and exit "
                         "-- the cheap CONFIRM path (no SNN arms). Use to validate a task gates at depth-3 before sweeping.")
    ap.add_argument("--wide-hidden", type=int, default=256,
                    help="width of the WIDE frozen-reservoir optimal-readout control (the categorical control: on XOR a "
                         "fixed random reservoir CANNOT decode at ANY width, so wide-frozen-optimal should sit at chance).")
    # BPTT ceiling tuning (default None => matched --hidden/--epochs/--lr, byte-identical; the XOR command passes the
    # 0.82-source config so the headroom/ceiling is VALID -- the crux run under-tuned BPTT, invalidating its ceiling).
    ap.add_argument("--bptt-hidden", type=int, default=None, help="BPTT-arm hidden width (default = --hidden).")
    ap.add_argument("--bptt-epochs", type=int, default=None, help="BPTT-arm epochs (default = --epochs).")
    ap.add_argument("--bptt-lr", type=float, default=None, help="BPTT-arm learning rate (default = --lr).")
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
    task_name = "nestedxor" if args.task_nestedxor else ("xor" if args.task_xor else "inheritance")

    # ---- CHEAP CONFIRM PATH: the stage0 depth-genuineness probe ALONE (no SNN arms). Validates a task's OBLIGATORY
    #      depth: depth3_requiring = (l2 ~ chance) AND (l3 >= 0.80) AND (l3 - l2 >= 0.15). Run this BEFORE the sweep. ----
    if args.stage0_only:
        s0_results = []
        for sd in args.seeds:
            if args.task_nestedxor:
                (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_nestedxor(sd)
            elif args.task_xor:
                (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_xor(sd)
            else:
                (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_semantic_inheritance(sd, **task_kwargs)
            k = meta["k_classes"]; inh = idx["inh_idx"]
            chance = (float(max(np.mean(yte[inh] == c) for c in np.unique(yte[inh])))
                      if len(inh) else float("nan"))
            s0 = stage0_depth_genuineness(((Xtr, ytr, Ltr), (Xte, yte, Lte)), idx, k, hidden=96, epochs=250,
                                          lr=0.3, batch=128, seed=sd)
            l2 = s0["l2_inherit_heldout"]; l3 = s0["l3_inherit_heldout"]
            depth3 = bool((not np.isnan(chance)) and l2 <= chance + 0.06 and l3 >= 0.80 and (l3 - l2) >= 0.15)
            s0["seed"] = sd; s0["depth3_requiring"] = depth3
            s0_results.append(s0)
            print(f"[seed {sd}] chance {chance:.3f} | l0 {s0['linear_inherit_heldout']:.3f} "
                  f"l1 {s0['l1_inherit_heldout']:.3f} l2 {l2:.3f} l3 {l3:.3f} => depth3_requiring={depth3} "
                  f"(depth_separating={s0['depth_separating']})", flush=True)
        n = len(s0_results)
        # Below-chance / UNDEFINED self-declaration (silent-failure discipline: UNDEFINED, not a NO-GO). A stage0
        # confirm whose DEEPEST oracle cannot fit TRAIN measures NOTHING about depth-genuineness -- the task is
        # UNFITTABLE, so no depth ceiling exists to gate a depth-rescue test against (e.g. a parity task: backprop
        # cannot fit deep XOR -- Shalev-Shwartz 2017). The artifact must OWN that status so downstream reads never
        # mistake "0/N depth3_requiring" (which could also mean "fittable but shallow-solvable") for a real NO-GO.
        def _mean(key):
            vals = [r[key] for r in s0_results if isinstance(r.get(key), (int, float))]
            return sum(vals) / len(vals) if vals else float("nan")
        n_go = sum(bool(r['depth3_requiring']) for r in s0_results)
        mean_l3_train = _mean("l3_train"); mean_chance = _mean("chance")
        # "unfittable" = the confirm did not gate AND the depth-3 oracle sits at/below chance ON TRAIN (cannot even
        # memorize) -> the deepest ceiling does not exist. Robust to per-seed noise (uses the cross-seed mean).
        unfittable = bool(n_go == 0 and not (mean_l3_train != mean_l3_train)  # not NaN
                          and mean_l3_train <= mean_chance + 0.05)
        verdict = ("UNDEFINED_task_unfittable (depth-3 oracle cannot fit TRAIN: mean l3_train "
                   f"{mean_l3_train:.3f} <= chance {mean_chance:.3f} -> NO depth-3 ceiling exists; a boolean-"
                   "obligatory-depth task on stacked parity is not backprop-optimizable (Shalev-Shwartz 2017), so it "
                   "cannot gate a depth-rescue test)" if unfittable
                   else (f"DEPTH3_REQUIRING ({n_go}/{n})" if n_go else
                         f"NOT_DEPTH3 ({n_go}/{n}: task fittable but does not obligate depth-3)"))
        out = {"probe": "gap4_bptt_snn_chained_fa_transport_free__STAGE0_ONLY", "task": task_name,
               "seeds": args.seeds, "stage0": s0_results,
               "depth3_requiring_seeds": f"{n_go}/{n}", "below_chance": unfittable, "verdict": verdict,
               "mean_l3_train": mean_l3_train, "mean_chance": mean_chance,
               "gate": "depth3_requiring = (l2 <= chance+0.06) AND (l3 >= 0.80) AND (l3 - l2 >= 0.15)"}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nstage0-only [{task_name}]: depth3_requiring {out['depth3_requiring_seeds']}. "
              f"Gate: l2<=chance+0.06 AND l3>=0.80 AND l3-l2>=0.15.")
        return

    t0 = time.time()
    results = []
    for sd in args.seeds:
        try:
            r = run_seed(sd, args.hidden, args.timesteps, args.epochs, args.lr, lr_fa, args.in_gain,
                         args.train_subsample, task_kwargs, n_hidden_layers=args.n_hidden_layers,
                         sigma_norm=args.sigma_norm, kp_lr=args.kp_lr, kp_decay=args.kp_decay,
                         check_depth=not args.no_depth_check, task_xor=args.task_xor,
                         task_nestedxor=args.task_nestedxor,
                         bptt_hidden=args.bptt_hidden, bptt_epochs=args.bptt_epochs, bptt_lr=args.bptt_lr,
                         wide_hidden=args.wide_hidden)
        except Exception as e:
            r = {"seed": sd, "error": repr(e), "traceback": traceback.format_exc()}
        results.append(r)
        if "error" not in r:
            print(f"[seed {sd}] [N={r.get('n_hidden_layers')}h depth3_req={r.get('depth3_requiring')} "
                  f"l2={r.get('stage0_l2_inherit')} l3={r.get('stage0_l3_inherit')}] "
                  f"chained_fa {r['chained_fa_inherit']:.3f} (kp {r['chained_fa_kp_inherit']:.3f}, "
                  f"kp-over-fa {r['kp_over_fixed_fa']:+.3f}) "
                  f"vs frozen {r['frozen_reservoir_inherit']:.3f} vs permuted {r['permuted_inherit']:.3f} | "
                  f"BPTT ceiling {r['bptt_inherit']:.3f} (train {r['bptt_train']:.3f}), oracle {r['oracle_inherit']:.3f}, "
                  f"chance {r['chance']:.3f} | frozen-OPTIMAL matched {r['frozen_optimal_matched_inherit']:.3f} / "
                  f"WIDE-{r['wide_hidden']} {r['frozen_optimal_wide_inherit']:.3f} | chained-over-wide-optimal "
                  f"{r['chained_over_wide_optimal']:+.3f}, wide-at-chance={r['wide_optimal_at_chance']}, "
                  f"bptt-solves={r['bptt_solves_task']} => CATEGORICAL_UNLOCK={r['CATEGORICAL_UNLOCK']} | "
                  f"GO_fixed={r['GO_fixed']} GO_kp={r['GO_kp']}")
        else:
            print(f"[seed {sd}] ERROR: {r['error']}")

    agg = _agg(results)
    out = {"probe": "gap4_bptt_snn_chained_fa_transport_free",
           "task": task_name, "seeds": args.seeds,
           "config": {"hidden": args.hidden, "n_hidden_layers": args.n_hidden_layers, "T": args.timesteps,
                      "epochs": args.epochs, "lr": args.lr, "lr_fa": lr_fa, "in_gain": args.in_gain,
                      "sigma_norm": args.sigma_norm, "kp_lr": args.kp_lr, "kp_decay": args.kp_decay,
                      "train_subsample": args.train_subsample, "task_xor": bool(args.task_xor),
                      "task_nestedxor": bool(args.task_nestedxor),
                      "wide_hidden": args.wide_hidden, "bptt_hidden": args.bptt_hidden,
                      "bptt_epochs": args.bptt_epochs, "bptt_lr": args.bptt_lr, "task": task_kwargs},
           "elapsed_seconds": round(time.time() - t0, 1), "results": results, "aggregate": agg}
    if agg:
        _cat = agg.get("CATEGORICAL_UNLOCK_seeds", "0/0")
        # DEPTH-3 SWEEP read (nestedxor): depth3_requiring confirms the task gates at depth-3; the KP-depth-rescue
        # signature is kp_over_fixed_fa GROWING with --n-hidden-layers while fixed-FA purchase_over_frozen collapses
        # toward 0 at N>=3 and KP holds. Compare these across the N=2,3,4 runs.
        out["depth3_sweep_read"] = (
            f"[{out['task']}] N={args.n_hidden_layers}h: depth3_requiring {agg.get('depth3_requiring_seeds')} "
            f"(stage0 l2 {agg.get('mean_stage0_l2_inherit')} vs l3 {agg.get('mean_stage0_l3_inherit')}, chance "
            f"{agg['mean_chance']:.3f}); kp_over_fixed_fa {agg.get('mean_kp_over_fixed_fa')}; fixed-FA "
            f"purchase_over_frozen {agg.get('mean_purchase_over_frozen')} vs KP purchase_over_frozen "
            f"{agg.get('mean_kp_purchase_over_frozen')}; GO_fixed {agg['GO_fixed_seeds']} GO_kp {agg['GO_kp_seeds']}. "
            f"Sweep signature (read across N=2,3,4): kp_over_fixed_fa GROWS + fixed-FA purchase collapses to ~0 at "
            f"N>=3 while KP holds => KP rescues OBLIGATORY depth on spikes.")
        out["verdict"] = (
            f"[{out['task']}] transport-free chained-FA on the TRAINABLE LIF SNN: chained_fa "
            f"{agg['mean_chained_fa_inherit']:.3f} (kp {agg['mean_chained_fa_kp_inherit']:.3f}) vs frozen-local "
            f"{agg['mean_frozen_reservoir_inherit']:.3f} vs permuted {agg['mean_permuted_inherit']:.3f}; BPTT ceiling "
            f"{agg['mean_bptt_inherit']:.3f}, oracle {agg['mean_oracle_inherit']:.3f}, chance {agg['mean_chance']:.3f}. "
            f"CATEGORICAL controls: frozen-OPTIMAL matched {agg['mean_frozen_optimal_matched_inherit']:.3f} / "
            f"WIDE-{args.wide_hidden} {agg['mean_frozen_optimal_wide_inherit']:.3f}; chained-over-wide-optimal "
            f"{agg['mean_chained_over_wide_optimal']:+.3f}; wide-at-chance {agg['wide_optimal_at_chance_seeds']}, "
            f"bptt-solves {agg['bptt_solves_task_seeds']}, CATEGORICAL_UNLOCK {_cat}. GO_fixed {agg['GO_fixed_seeds']}, "
            f"GO_kp {agg['GO_kp_seeds']}. "
            + ("=> CATEGORICAL UNLOCK: chained_fa BEATS a WIDE frozen reservoir that FAILS (the task is NOT linearly "
               "reservoir-decodable) -> directed credit does what NO reservoir can, at any width."
               if _cat.split("/")[0] not in ("0", "") and _cat.split("/")[0] == _cat.split("/")[1] else
               ("=> PARTIAL: chained_fa beats wide-frozen-optimal on some seeds; read wide-at-chance + bptt-solves per "
                "seed (if wide-frozen still solves it, the task remains reservoir-decodable -> pick a harder task)."
                if _cat.split("/")[0] not in ("0", "") else
                "=> NOT a categorical unlock: the WIDE frozen reservoir (optimally read) still matches/exceeds "
                "chained_fa -> the task is still reservoir-decodable OR chained_fa also struggles; see the per-seed table.")))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + out.get("verdict", "(no aggregate)"))


if __name__ == "__main__":
    main()
