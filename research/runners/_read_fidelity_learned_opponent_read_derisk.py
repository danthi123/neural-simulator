"""READ-FIDELITY de-risk, ITERATION 6 -- the RANK-1 residual iteration 5 (opponent/push-pull) earned: a
LEARNED / REGULARIZED opponent read, on the surprise->source_provenance F2 crux.

WHY THIS RUN EXISTS (do NOT re-derive -- the opponent read is BANKED and NO-GO):
  iteration 4 (`_read_fidelity_popvec_template_derisk.py`,
    2026-08-28-read-fidelity-popvec-template-biological-read-NOGO-power-gap-not-signal-gap.md): a RECTIFIED
    single-channel matched-filter TEMPLATE + LIF readout, NO-GO 0/6 (z=[0.05,0.18,0.04,1.13,0.09,1.32]).
  iteration 5 (`_read_fidelity_opponent_pushpull_derisk.py`,
    2026-08-28-read-fidelity-opponent-pushpull-NOGO-...md): the OPPONENT / PUSH-PULL read (recover the sign a
    rectified channel discards, via a Dale's-law excitatory-push + inhibitory-pull pair), NO-GO 0/6 and, on
    average, WORSE than the single channel (mean z 0.213 vs 0.468). Its OWN diagnosis + ranked-#1 next lever
    (verbatim): "a LEARNED opponent gain, not a fixed 1:1 subtraction ... a variance-weighted combination akin
    to how Salinas & Abbott's optimal linear estimator itself down-weights noisy channels". The popvec NO-GO's
    OWN constraint #3 (verbatim): "The template (mean-difference) is a WEAKER ESTIMATOR than logistic
    regression. A biologically-plausible local learning rule on the readout weights (e.g. a delta/perceptron
    rule the substrate can run) would close part of the estimator gap without leaving spiking."

THE DECISIVE OBSERVATION THIS RUN IS BUILT ON. The read that PROVABLY recovers the signal is iteration 3's
DECODER -- and the LINEAR arm of it (L2 LOGISTIC) already separates 6/6, shuffle-clean. So the signal is
LINEARLY separable; the missing ingredient is NOT nonlinearity (that is the dendritic lever, and our own
2026-08-25 vision-2layer NO-GO already showed nonlinear expansion does not reliably lift a linear ceiling) --
it is ESTIMATOR QUALITY. iterations 4/5's template is the DIAGONAL mean-difference `(mu_gen-mu_perc)/pooled_sd`:
it normalizes each of the 10 time-bins by its OWN std and IGNORES cross-bin covariance. L2 logistic (and Fisher
LDA) instead account for the temporal correlation BETWEEN bins. The 10 bins ARE temporally correlated (a neuron
firing in one bin tends to fire in the next), so the covariance-aware direction can differ substantially from
the diagonal one -- and that difference is exactly what the working decoder exploits and the failing template
does not. This run replaces the diagonal template with a covariance-aware LEARNED direction, still realized as
a biological opponent (Dale's-law E/I) read + LIF readout.

THE MECHANISM -- three template-fitting methods, each producing a SIGNED 10-bin direction `w` then RECTIFIED
into the SAME opponent (push-pull) E/I pair iteration 5 used (template_E=clip(w,0,None),
template_I=clip(-w,0,None)); everything downstream (LIF readout, CV, null, gate) is iteration 5's VERBATIM:
  - `meandiff` = iteration 5's EXACT direction (diagonal mean-difference). Run IN THIS FILE so the z-shift vs
    the learned methods is a same-process, same-fold, same-null A/B (not a cross-file number lift).
  - `lda` (PRIMARY / gating) = shrinkage-regularized FISHER LDA: w = (Sigma + lam*I)^{-1} (mu_gen - mu_perc),
    Sigma = pooled within-class covariance across the 10 bins on the TRAIN fold. This is the covariance-whitened
    matched filter -- the DIRECT fix for the diagonal-vs-full gap above. Biologically it is a matched filter
    preceded by DECORRELATING (whitening) inhibition -- a decades-established cortical/cerebellar circuit motif
    (recurrent/feedforward inhibition decorrelates population codes: Cayco-Gajic, Clopath & Silver 2017, Nat
    Commun 8:1116; King, Zylberberg & DeWeese 2013, J Neurosci 33:5475; the whitening role of inhibition in
    Pehlevan & Chklovskii 2015, NeurIPS). The shrinkage lam is a homeostatic regularizer (prevents the
    near-singular small-sample covariance from amplifying noise) -- exactly the "down-weight noisy channels"
    the opponent NO-GO's rank-1 residual asked for, realized as covariance shrinkage rather than a per-bin gain.
  - `logistic` (diagnostic) = L2-regularized logistic direction fit by a local error-corrective DELTA rule
    (batch gradient descent on the cross-entropy = the logistic delta/Widrow-Hoff-Rescorla-Wagner update:
    dw ∝ presyn_activity * (target - output)) with weight-decay (homeostatic synaptic scaling). This is the
    EXACT estimator family that gets iteration 3's linear-decoder 6/6, now realized through the SAME opponent-
    LIF read -- so if `logistic` ALSO fails through this read while its per-neuron classification stays 6/6
    (diagnostic below), the residual is DECISIVELY the READ ARCHITECTURE (population-contrast collapse), not the
    estimator, and the next lever is a richer read (per-neuron matched-filter), not a better direction.

THE DIAGNOSTICS (non-gating -- they isolate WHERE any residual lives, per the NO-DEFER "quantify the residual"
discipline):
  D1 `dir_holdout_acc`: for each of {meandiff, lda, logistic}, the HELD-OUT stratified-CV accuracy of
     classifying gen-vs-perc neurons from their 10-bin delta_held_base profile by projecting onto `w` and
     thresholding at the train-projected class midpoint. Reproduces/extends iteration 3's decoder question at
     the DIRECTION level (no LIF): does the learned direction separate held-out NEURONS better than the diagonal
     one? If yes but the LIF read still fails -> residual is the read architecture.
  D2 z-shift of each method's opponent-LIF read (intact delta_held_base) vs iteration 5's meandiff, in-run.
  D3 `lda` at 3 shrinkage lam values (SENSITIVITY, non-gating) so a NO-GO cannot be dismissed as a single-lam
     artifact.

LEAKAGE / ANTI-CHEAT (identical bars to iterations 3/4/5, unchanged): the direction and any standardization are
fit on the TRAIN fold's neurons ONLY; the readout is evaluated on the HELD-OUT test fold's own raw spikes; pool
membership (+1 gen / -1 perc) is a STRUCTURAL wiring fact used as each read-synapse's fixed sign, identical to
every prior iteration (not label leakage -- same convention iteration 4/5 justify). Neuron-identity permutation
null (K_SHUF draws, Z_FLOOR=2.0, SHUF_COLLAPSE_MAX_RATE=0.15) is the instrument-validity bar. The ONLY changed
variable vs iteration 5 is the template-fitting METHOD (diagonal mean-difference -> covariance-whitened
LDA / logistic); pool build/train/lesion, raster capture, binned features, CV fold machinery, LIF readout model,
threshold calibration, anti-cheat shape, and the delta_held_base gate are reused VERBATIM.

THE GATE (pre-registered before any seed ran, UNCHANGED from iteration 4/5): PRIMARY = `lda` on
`delta_held_base` (the cross-edge-attributable feature): intact real_margin_mean > 0 AND z >= Z_FLOOR AND
shuffle_collapses AND the margin is lesion-attributable (|lesion mean| < F2_LESION_RATIO*|intact mean|). GO
requires the PRIMARY gate PASS on every one of 6 seeds. `meandiff`/`logistic`/`raw_held`/the lam-sensitivity are
reported but do NOT gate.

DE-RISK ONLY -- no production wiring, no `sim/` edit, no default flip. numpy CPU throughout; pool-runnable
(cost-routing: numpy reanalysis of already-captured rasters, 0 GPU, 0 Claude tokens -- the shortlist's own
prescription for this lever).

Run:
  SIM_BACKEND=numpy python -m research.runners._read_fidelity_learned_opponent_read_derisk --seeds 42 --smoke
  SIM_BACKEND=numpy python -m research.runners._read_fidelity_learned_opponent_read_derisk \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_read_fidelity_learned_opponent_read_derisk_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only -- never touch the GPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sim.backend import to_host
from tools.lab import attributable_to
from research.runners._read_fidelity_nonrate_latency_derisk import (
    ReadFidelityPool, RECALL_STEPS, N_READS, PRE_STEPS, EPISODE_DRIVE_PA,
)
from research.runners._read_fidelity_nonrate_latency_dispersion_derisk import _capture_reads
from research.runners._read_fidelity_decoder_separability_derisk import (
    _binned_features, _avg_binned, _stratified_folds, _seed_trap_check,
)
from research.runners._onebrain_integration_surprise_episodic_crossedge import (
    F2_LESION_RATIO, CROSS_EDGE_LR, N_EPISODES, HMAX, CUE_PA, CTX_DRIVE_PA, _build_pool,
)
from research.runners._read_fidelity_popvec_template_derisk import (
    _expand_template_to_steps, _lif_batch_spike_counts, _calibrate_v_th, _arm_verdict,
)

# ---- this run's own pre-registered constants (declared BEFORE any measurement) ----
# ALL identical to iteration 4/5 -- the only changed variable is the template-fitting METHOD (below).
N_BINS = 10
Z_FLOOR = 2.0
SHUF_COLLAPSE_MAX_RATE = 0.15
K_FOLDS = 5
R_REPEATS = 5
K_SHUF = 20
R_REPEATS_SMOKE = 2
K_SHUF_SMOKE = 5
TAU_MEM_STEPS = 5.0
V_RESET = 0.0
CALIB_TARGET_FRAC = 0.5
V_TH_FLOOR = 1e-6
EPS = 1e-9
WEIGHT_CONDS = ("intact", "lesion")
FEAT_KINDS = ("raw_held", "delta_held_base")

# ---- learned-direction hyperparameters (host knobs, pre-registered, NOT tuned on the test gate) ----
# LDA shrinkage: the covariance is regularized as Sigma + LDA_SHRINK * mean(diag(Sigma)) * I -- a trace-scaled
# ridge so the same fraction works regardless of the feature scale. LDA_SHRINK=0.30 is a moderate,
# field-standard shrinkage (Ledoit-Wolf estimates typically land 0.1-0.5 at this sample/feature ratio); it is
# the PRIMARY gating value. The sensitivity sweep reports 0.10/0.30/1.00 so a NO-GO is not a single-lam artifact.
LDA_SHRINK = 0.30
LDA_SHRINK_SWEEP = (0.10, 0.30, 1.00)
# Logistic (delta-rule) diagnostic: batch-gradient-descent L2 logistic on train-fold-standardized features.
LOGIT_EPOCHS = 400
LOGIT_LR = 0.5
LOGIT_L2 = 1.0


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  The LEARNED directions -- each returns a SIGNED (n_bins,) direction `w` in RAW per-bin units
#  (so it drops into the SAME opponent rectification + LIF read iteration 5 uses)
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _dir_meandiff(prof_train, labels_bool):
    """iteration 5's EXACT direction: diagonal mean-difference, (mu_gen-mu_perc)/pooled_per_bin_std. Reproduced
    here so the learned-vs-diagonal comparison is a same-process, same-fold A/B."""
    gen = prof_train[labels_bool]; perc = prof_train[~labels_bool]
    assert gen.shape[0] >= 2 and perc.shape[0] >= 2, "need >=2 train neurons per class"
    mu_gen, mu_perc = gen.mean(axis=0), perc.mean(axis=0)
    pooled_sd = 0.5 * (gen.std(axis=0, ddof=1) + perc.std(axis=0, ddof=1))
    return (mu_gen - mu_perc) / (pooled_sd + EPS)


def _dir_lda(prof_train, labels_bool, shrink=LDA_SHRINK):
    """Shrinkage Fisher LDA: w = (Sigma_pooled + shrink*mean_diag*I)^{-1} (mu_gen - mu_perc). Sigma_pooled is the
    pooled within-class covariance across the 10 bins on the TRAIN fold. Covariance-whitened matched filter --
    the covariance-aware direction the diagonal mean-difference cannot form. Shrinkage prevents the small-sample
    near-singular covariance from amplifying noise (the homeostatic regularizer the opponent NO-GO asked for)."""
    gen = prof_train[labels_bool]; perc = prof_train[~labels_bool]
    assert gen.shape[0] >= 2 and perc.shape[0] >= 2, "need >=2 train neurons per class"
    mu_gen, mu_perc = gen.mean(axis=0), perc.mean(axis=0)
    # pooled within-class covariance (ddof via (n-k) normalization)
    d = prof_train.shape[1]
    xc = np.vstack([gen - mu_gen, perc - mu_perc])
    n_eff = max(1, xc.shape[0] - 2)
    sigma = (xc.T @ xc) / n_eff                      # (d, d)
    ridge = shrink * float(np.mean(np.diag(sigma)) + EPS)
    sigma_reg = sigma + ridge * np.eye(d)
    w = np.linalg.solve(sigma_reg, (mu_gen - mu_perc))
    return w


def _dir_logistic(prof_train, labels_bool, epochs=LOGIT_EPOCHS, lr=LOGIT_LR, l2=LOGIT_L2):
    """L2-regularized logistic direction fit by a local error-corrective DELTA rule (batch gradient descent on
    the cross-entropy: dw ∝ X^T (target - sigmoid(Xw)) - l2*w -- the logistic Widrow-Hoff/Rescorla-Wagner update
    with homeostatic weight-decay). Fit on TRAIN-fold-standardized features; the standardization is folded back
    into `w` so the returned direction is in RAW per-bin units (drops into the same read as the others). This is
    the estimator family that gets iteration 3's linear decoder 6/6."""
    X = prof_train.astype(np.float64)
    y = labels_bool.astype(np.float64)               # gen=1, perc=0
    mu = X.mean(axis=0); sd = X.std(axis=0, ddof=0) + EPS
    Xs = (X - mu) / sd
    n, d = Xs.shape
    w = np.zeros(d); b = 0.0
    for _ in range(epochs):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = y - p                                   # (n,)
        gw = Xs.T @ err / n - l2 * w                  # + weight decay
        gb = err.mean()
        w += lr * gw
        b += lr * gb
    return w / sd                                     # fold standardization back into RAW per-bin units


_DIRECTIONS = {"meandiff": _dir_meandiff, "lda": _dir_lda, "logistic": _dir_logistic}


def _opponent_from_direction(w):
    """Rectify a signed direction into the SAME Dale's-law opponent (push-pull) E/I pair iteration 5 uses."""
    return np.clip(w, 0.0, None), np.clip(-w, 0.0, None)


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  One fold: fit the direction on TRAIN neurons, drive the opponent-LIF readout with TEST spikes
#  (identical read to iteration 5 -- net excitatory-minus-inhibitory synaptic current into one LIF)
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _fold_margin(raster_pairs, tE_bins, tI_bins, train_idx, test_idx, labels_bool, steps, n_bins, feat_kind):
    tE_steps = _expand_template_to_steps(tE_bins, steps, n_bins)
    tI_steps = _expand_template_to_steps(tI_bins, steps, n_bins)
    n_union = labels_bool.size
    test_gen = np.zeros(n_union, dtype=bool); test_gen[test_idx] = labels_bool[test_idx]
    test_perc = np.zeros(n_union, dtype=bool); test_perc[test_idx] = ~labels_bool[test_idx]
    train_gen = np.zeros(n_union, dtype=bool); train_gen[train_idx] = labels_bool[train_idx]
    train_perc = np.zeros(n_union, dtype=bool); train_perc[train_idx] = ~labels_bool[train_idx]

    def _net_current(raster, gen_mask, perc_mask):
        sig = (raster[:, gen_mask].sum(axis=1).astype(np.float64)
               - raster[:, perc_mask].sum(axis=1).astype(np.float64))
        return tE_steps * sig - tI_steps * sig       # net (push - pull) into the LIF membrane

    train_I = np.stack([_net_current(rh, train_gen, train_perc) for (_rb, rh) in raster_pairs])
    v_th = _calibrate_v_th(train_I, TAU_MEM_STEPS)
    held_I = np.stack([_net_current(rh, test_gen, test_perc) for (_rb, rh) in raster_pairs])
    base_I = np.stack([_net_current(rb, test_gen, test_perc) for (rb, _rh) in raster_pairs])
    n_reads = held_I.shape[0]
    counts = _lif_batch_spike_counts(np.concatenate([held_I, base_I], axis=0), TAU_MEM_STEPS, v_th)
    held_counts, base_counts = counts[:n_reads], counts[n_reads:]
    if feat_kind == "raw_held":
        return float(held_counts.mean()), float(v_th)
    return float((held_counts - base_counts).mean()), float(v_th)


def _one_cv_pass(labels_bool, rng, profiles_held, profiles_base, raster_pairs, feat_kind, method, shrink):
    prof = profiles_held if feat_kind == "raw_held" else (profiles_held - profiles_base)
    dir_fn = _DIRECTIONS[method]
    margins = []
    for train_idx, test_idx in _stratified_folds(labels_bool, K_FOLDS, rng):
        if method == "lda":
            w = dir_fn(prof[train_idx], labels_bool[train_idx], shrink=shrink)
        else:
            w = dir_fn(prof[train_idx], labels_bool[train_idx])
        tE, tI = _opponent_from_direction(w)
        m, _v = _fold_margin(raster_pairs, tE, tI, train_idx, test_idx, labels_bool,
                             RECALL_STEPS, N_BINS, feat_kind)
        margins.append(m)
    return float(np.mean(margins))


def _combo_stats(raster_pairs, profiles_held, profiles_base, n_gen, n_perc, feat_kind, method, rng,
                 repeats, k_shuf, shrink=LDA_SHRINK):
    """Real (REPEATS-repeated CV) vs a K_SHUF-draw neuron-identity permutation null -- IDENTICAL statistical
    shape to iterations 3/4/5 (same Z_FLOOR, same SHUF_COLLAPSE_MAX_RATE, same anti-cheat definition)."""
    n_union = n_gen + n_perc
    real_label = np.zeros(n_union, dtype=bool); real_label[:n_gen] = True
    real_vals = np.array([_one_cv_pass(real_label, rng, profiles_held, profiles_base, raster_pairs,
                                       feat_kind, method, shrink) for _ in range(repeats)])
    null_vals = []
    for _ in range(k_shuf):
        perm = rng.permutation(n_union)
        y_shuf = np.zeros(n_union, dtype=bool); y_shuf[perm[:n_gen]] = True
        null_vals.append(_one_cv_pass(y_shuf, rng, profiles_held, profiles_base, raster_pairs,
                                      feat_kind, method, shrink))
    null_vals = np.asarray(null_vals, dtype=np.float64)
    real_mean = float(real_vals.mean())
    real_std = float(real_vals.std(ddof=1)) if real_vals.size > 1 else 0.0
    null_mean = float(null_vals.mean())
    null_std = float(null_vals.std(ddof=1)) if null_vals.size > 1 else 0.0
    z = (real_mean - null_mean) / null_std if null_std > 0 else (float("inf") if real_mean != null_mean else 0.0)
    frac_null_clears = (float(np.mean(np.abs((null_vals - null_mean) / null_std) >= Z_FLOOR))
                        if null_std > 0 else float("nan"))
    shuffle_collapses = bool(frac_null_clears <= SHUF_COLLAPSE_MAX_RATE) if not np.isnan(frac_null_clears) else False
    return {
        "feat_kind": feat_kind, "method": method, "shrink": (shrink if method == "lda" else None),
        "real_margin_mean": real_mean, "real_margin_std": real_std,
        "real_margin_all": [float(x) for x in real_vals],
        "null_margin_mean": null_mean, "null_margin_std": null_std,
        "z": float(z), "n_repeats": int(repeats), "n_shuf": int(k_shuf),
        "frac_null_clears_floor": frac_null_clears, "shuffle_collapses": shuffle_collapses,
    }


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  D1 diagnostic: held-out per-neuron classification accuracy of each DIRECTION (no LIF) -- does the
#  learned direction separate held-out NEURONS better than the diagonal one? (reproduces iteration 3)
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _direction_holdout_acc(profiles_held, profiles_base, n_gen, n_perc, method, rng, repeats):
    prof = profiles_held - profiles_base            # delta_held_base profile (the gating feature)
    n_union = n_gen + n_perc
    labels_bool = np.zeros(n_union, dtype=bool); labels_bool[:n_gen] = True
    dir_fn = _DIRECTIONS[method]
    accs = []
    for _ in range(repeats):
        fold_acc = []
        for train_idx, test_idx in _stratified_folds(labels_bool, K_FOLDS, rng):
            w = dir_fn(prof[train_idx], labels_bool[train_idx])
            proj_tr = prof[train_idx] @ w
            thr = 0.5 * (proj_tr[labels_bool[train_idx]].mean() + proj_tr[~labels_bool[train_idx]].mean())
            proj_te = prof[test_idx] @ w
            pred_gen = proj_te >= thr
            fold_acc.append(float(np.mean(pred_gen == labels_bool[test_idx])))
        accs.append(float(np.mean(fold_acc)))
    return float(np.mean(accs)), float(np.std(accs))


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Per-seed run
# ─────────────────────────────────────────────────────────────────────────────────────────────
def run_seed(seed, repeats, k_shuf):
    t0 = time.time()
    pool = ReadFidelityPool(seed)
    traj = pool.train()
    emg_grew = bool(traj[-1]["w"] > 5 * 0.05)
    emg_specific = bool(abs(traj[-1]["w_other"] - 0.05) < 0.03)

    ix = pool.ix
    n_gen = int(ix["prov_generated"].size)
    n_perc = int(ix["prov_perceived"].size)
    union = np.concatenate([ix["prov_generated"], ix["prov_perceived"]])
    read_dict = {"gen": ix["prov_generated"], "perc": ix["prov_perceived"]}

    # ---- INTACT ----
    pairs_intact = _capture_reads(pool, read_dict, union)
    held_i = _avg_binned([h for (_b, h) in pairs_intact], N_BINS)
    base_i = _avg_binned([b for (b, _h) in pairs_intact], N_BINS)

    # ---- LESIONED (same event as iterations 1-5: zero surprise->provgen, in place) ----
    data = np.asarray(to_host(pool.b.cp_connections.data)).copy()
    data[pool.masks["surprise->provgen"]] = 0.0
    pool.b.cp_connections.data = pool.xp.asarray(data, dtype=pool.b.cp_connections.data.dtype)
    pairs_lesion = _capture_reads(pool, read_dict, union)
    held_l = _avg_binned([h for (_b, h) in pairs_lesion], N_BINS)
    base_l = _avg_binned([b for (b, _h) in pairs_lesion], N_BINS)

    feats = {"intact": (pairs_intact, held_i, base_i), "lesion": (pairs_lesion, held_l, base_l)}
    base_off = int(seed) * 32452843 + 613     # this module's own distinct RNG offset family (siblings in the
                                              # file family: *104729+17, *65599+41, *7919+101, *997+3,
                                              # *15485863+271, *50331653+191, *179424673+337 -- none collide)
    combo_i = 0

    def _rng_next():
        nonlocal combo_i
        r = np.random.default_rng(base_off + combo_i * 6151)
        combo_i += 1
        return r

    # ---- GATING method = lda: full 4 combos (intact/lesion x raw/delta), full null ----
    combos = {}
    for cond in WEIGHT_CONDS:
        raster_pairs, held_p, base_p = feats[cond]
        for feat_kind in FEAT_KINDS:
            combos[f"lda__{cond}__{feat_kind}"] = _combo_stats(
                raster_pairs, held_p, base_p, n_gen, n_perc, feat_kind, "lda", _rng_next(),
                repeats, k_shuf, shrink=LDA_SHRINK)

    # ---- DIAGNOSTIC methods meandiff/logistic: intact+lesion on delta_held_base (for a full primary verdict) ----
    for method in ("meandiff", "logistic"):
        for cond in WEIGHT_CONDS:
            raster_pairs, held_p, base_p = feats[cond]
            combos[f"{method}__{cond}__delta_held_base"] = _combo_stats(
                raster_pairs, held_p, base_p, n_gen, n_perc, "delta_held_base", method, _rng_next(),
                repeats, k_shuf)

    # ---- D3 lam-sensitivity for lda (intact delta_held_base only), non-gating ----
    lam_sweep = {}
    raster_pairs, held_p, base_p = feats["intact"]
    for lam in LDA_SHRINK_SWEEP:
        if abs(lam - LDA_SHRINK) < 1e-12:
            lam_sweep[f"lam_{lam}"] = combos["lda__intact__delta_held_base"]  # already computed at the gating lam
            continue
        lam_sweep[f"lam_{lam}"] = _combo_stats(
            raster_pairs, held_p, base_p, n_gen, n_perc, "delta_held_base", "lda", _rng_next(),
            repeats, k_shuf, shrink=lam)

    # ---- verdicts ----
    attributable_to(
        "F2 learned(lda) opponent read margin (delta_held_base) -- intact vs lesioned cross-edge",
        combos["lda__intact__delta_held_base"]["real_margin_mean"],
        combos["lda__lesion__delta_held_base"]["real_margin_mean"])
    primary = _arm_verdict(combos["lda__intact__delta_held_base"], combos["lda__lesion__delta_held_base"],
                           "F2 learned(lda) opponent read margin (delta_held_base, cross-edge attributable)")
    secondary = _arm_verdict(combos["lda__intact__raw_held"], combos["lda__lesion__raw_held"],
                             "F2 learned(lda) opponent read margin (raw_held, NOT gating)")
    md_verd = _arm_verdict(combos["meandiff__intact__delta_held_base"],
                           combos["meandiff__lesion__delta_held_base"], "meandiff (iteration-5 reproduction)")
    lg_verd = _arm_verdict(combos["logistic__intact__delta_held_base"],
                           combos["logistic__lesion__delta_held_base"], "logistic (best-linear diagnostic)")

    # ---- D1 direction holdout accuracy (per method) ----
    dir_acc = {}
    for method in ("meandiff", "lda", "logistic"):
        m, s = _direction_holdout_acc(held_i, base_i, n_gen, n_perc, method, _rng_next(), repeats)
        dir_acc[method] = {"acc_mean": m, "acc_std": s}

    n_combos_shuffle_collapse = sum(c["shuffle_collapses"] for c in combos.values())

    return {
        "seed": int(seed), "elapsed_s": round(time.time() - t0, 1),
        "cue_concept": pool.cue_c, "assert_concept": pool.assert_cp,
        "final_weight_trained_block": float(traj[-1]["w"]), "final_weight_other_blocks": float(traj[-1]["w_other"]),
        "emergence_grew_from_near_zero": emg_grew, "emergence_other_blocks_stayed_near_seed": emg_specific,
        "n_gen": n_gen, "n_perc": n_perc,
        "combos": combos,
        "lam_sensitivity_lda_intact_delta": {k: {"z": v["z"], "real": v["real_margin_mean"],
                                                  "shuffle_collapses": v["shuffle_collapses"]}
                                             for k, v in lam_sweep.items()},
        "direction_holdout_acc": dir_acc,
        "PRIMARY_lda_delta_held_base": primary,
        "SECONDARY_lda_raw_held_not_gating": secondary,
        "DIAG_meandiff_delta_held_base": md_verd,
        "DIAG_logistic_delta_held_base": lg_verd,
        "n_combos_shuffle_collapse": int(n_combos_shuffle_collapse), "n_combos": len(combos),
        "PASS": bool(primary["PASS"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true", help="1 seed, lighter CV/shuffle budget")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]
    repeats = R_REPEATS_SMOKE if args.smoke else R_REPEATS
    k_shuf = K_SHUF_SMOKE if args.smoke else K_SHUF

    seed_trap = _seed_trap_check(seeds[0])
    print(f"[seed-trap] build-twice at seed={seeds[0]}: identical={seed_trap['identical']} "
          f"n_neurons={seed_trap['n_neurons']} hash={seed_trap['hash_build1']}", flush=True)

    runs = []
    for s in seeds:
        r = run_seed(s, repeats, k_shuf)
        runs.append(r)
        p = r["PRIMARY_lda_delta_held_base"]
        cl = r["combos"]["lda__intact__delta_held_base"]
        cm = r["combos"]["meandiff__intact__delta_held_base"]
        cg = r["combos"]["logistic__intact__delta_held_base"]
        da = r["direction_holdout_acc"]
        print(f"[seed {s}] ({r['elapsed_s']}s) block(c={r['cue_concept']},c'={r['assert_concept']}) "
              f"w={r['final_weight_trained_block']:.2f} | "
              f"LDA delta z={cl['z']:.2f} real={cl['real_margin_mean']:.3f} PASS={p['PASS']} "
              f"lesion_ok={p['lesion_ok']} | meandiff z={cm['z']:.2f} logistic z={cg['z']:.2f} | "
              f"holdout-acc md={da['meandiff']['acc_mean']:.3f} lda={da['lda']['acc_mean']:.3f} "
              f"lg={da['logistic']['acc_mean']:.3f} | n_shuf_ok={r['n_combos_shuffle_collapse']}/{r['n_combos']}",
              flush=True)

    n_pass = sum(r["PASS"] for r in runs)
    n_shuf_ok = sum(r["n_combos_shuffle_collapse"] == r["n_combos"] for r in runs)
    all_go_raw = bool(n_pass == len(runs)) and not args.smoke

    dec, preconditions = None, []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("read_fidelity_learned_opponent_read_derisk")
        Vd.require("shuffle_anticheat_collapses_on_every_combo",
                   1 if all(r["n_combos_shuffle_collapse"] == r["n_combos"] for r in runs) else 0,
                   expect=lambda x: x >= 1,
                   note="the neuron-identity permutation null must individually clear Z_FLOOR on <= "
                        f"{SHUF_COLLAPSE_MAX_RATE} of its own draws, on EVERY combo, before the learned "
                        "readout's verdict can be trusted -- same instrument-validity bar as iterations 2-5")
        Vd.require("emergence_grew_from_near_zero", 1 if all(r["emergence_grew_from_near_zero"] for r in runs) else 0,
                   expect=lambda x: x >= 1, note="the reused cross-edge trained normally (sanity on the shared substrate)")
        Vd.require("anti_cheat_random_assignment", 1 if len(set((r["cue_concept"], r["assert_concept"])
                   for r in runs)) > 1 else 0, expect=lambda x: x >= 1,
                   note="the per-seed block pair must actually vary (inherited from the parent runner)")
        dec = Vd.decide(all_go_raw, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    verdict_status = dec.get("status") if dec else None
    go = all_go_raw if dec is None else bool(dec.get("go"))
    if verdict_status == "UNDEFINED":
        tag = "UNDEFINED"
    elif args.smoke:
        tag = f"SMOKE ({'PASS' if runs[0]['PASS'] else 'NO-GO'}, 1-seed indicator)"
    else:
        tag = ("GO -- the LEARNED (covariance-whitened LDA) opponent read CLEARS the crux" if all_go_raw
               else "NO-GO/PARTIAL")

    # per-method mean/peak z across seeds (intact delta_held_base) -- the decisive comparison
    def _zs(method):
        return [r["combos"][f"{method}__intact__delta_held_base"]["z"] for r in runs]
    z_summary = {m: {"per_seed": _zs(m), "mean": float(np.mean(_zs(m))), "peak": float(np.max(_zs(m)))}
                 for m in ("meandiff", "lda", "logistic")}
    acc_summary = {m: {"per_seed": [r["direction_holdout_acc"][m]["acc_mean"] for r in runs],
                       "mean": float(np.mean([r["direction_holdout_acc"][m]["acc_mean"] for r in runs]))}
                   for m in ("meandiff", "lda", "logistic")}

    verdict = (f"{tag}. Replaces iterations 4/5's DIAGONAL mean-difference template with a covariance-aware "
               f"LEARNED direction (shrinkage Fisher LDA, shrink={LDA_SHRINK}; the covariance-whitened matched "
               f"filter -- a matched filter preceded by decorrelating inhibition), rectified into the SAME "
               f"Dale's-law opponent (push-pull) E/I pair + LIF readout iteration 5 used, on the SAME trained "
               f"cross-edge and SAME captured rasters iterations 1-5 used (no retraining confound). PRIMARY gate "
               f"= lda on delta_held_base (cross-edge-attributable): {n_pass}/{len(runs)} seeds PASS "
               f"(z>=Z_FLOOR={Z_FLOOR} AND lesion-attributable AND shuffle anti-cheat collapses). Per-method "
               f"intact delta_held_base z (mean/peak): meandiff={z_summary['meandiff']['mean']:.3f}/"
               f"{z_summary['meandiff']['peak']:.3f}, lda={z_summary['lda']['mean']:.3f}/"
               f"{z_summary['lda']['peak']:.3f}, logistic={z_summary['logistic']['mean']:.3f}/"
               f"{z_summary['logistic']['peak']:.3f}. Direction holdout-accuracy (per-neuron gen/perc, mean over "
               f"seeds): meandiff={acc_summary['meandiff']['mean']:.3f}, lda={acc_summary['lda']['mean']:.3f}, "
               f"logistic={acc_summary['logistic']['mean']:.3f} (chance=0.5). Anti-cheat: K_SHUF={k_shuf}-draw "
               f"neuron-identity permutation null collapses on "
               f"{sum(r['n_combos_shuffle_collapse'] for r in runs)}/{sum(r['n_combos'] for r in runs)} "
               f"combo-seed pairs."
               + (f" UNDEFINED, NOT a validated verdict either way: {len(dec.get('undefined_reasons', []))} "
                  f"precondition(s) unmet -- {'; '.join(dec.get('undefined_reasons', []))}."
                  if verdict_status == "UNDEFINED" else ""))

    payload = {
        "probe": "read_fidelity_learned_opponent_read_derisk", "verdict": verdict, "GO": go,
        "n_seeds": len(runs), "n_seeds_pass_primary": n_pass, "n_seeds_shuffle_ok": n_shuf_ok,
        "seeds": seeds, "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
        "preconditions": preconditions,
        "seed_trap_build_twice": seed_trap,
        "z_summary_intact_delta_held_base": z_summary,
        "direction_holdout_acc_summary": acc_summary,
        "config": {
            "n_bins": N_BINS, "z_floor": Z_FLOOR, "shuf_collapse_max_rate": SHUF_COLLAPSE_MAX_RATE,
            "k_folds": K_FOLDS, "r_repeats": repeats, "k_shuf": k_shuf,
            "tau_mem_steps": TAU_MEM_STEPS, "v_reset": V_RESET, "calib_target_frac": CALIB_TARGET_FRAC,
            "v_th_floor": V_TH_FLOOR, "weight_conds": list(WEIGHT_CONDS), "feat_kinds": list(FEAT_KINDS),
            "lda_shrink": LDA_SHRINK, "lda_shrink_sweep": list(LDA_SHRINK_SWEEP),
            "logit_epochs": LOGIT_EPOCHS, "logit_lr": LOGIT_LR, "logit_l2": LOGIT_L2,
            "recall_steps": RECALL_STEPS, "n_reads": N_READS, "pre_steps": PRE_STEPS,
            "episode_drive_pa": EPISODE_DRIVE_PA, "f2_lesion_ratio": F2_LESION_RATIO,
            "cross_edge_hebbian_lr": CROSS_EDGE_LR, "n_episodes": N_EPISODES,
            "hebbian_max_weight": HMAX, "cue_pa": CUE_PA, "ctx_drive_pa": CTX_DRIVE_PA,
            "rng_formula": "seed*32452843+613, +combo_index*6151 per combo (this module's own distinct offset "
                           "family; does not collide with the six sibling offsets in the read-fidelity file "
                           "family)",
        },
        "mechanism": ("Reuses ReadFidelityPool (build/train/lesion + `_drive2`) VERBATIM from iteration-1, "
                      "`_capture_reads` VERBATIM from iteration-2, `_binned_features`/`_avg_binned`/"
                      "`_stratified_folds`/`_seed_trap_check` VERBATIM from iteration-3, and "
                      "`_expand_template_to_steps`/`_lif_batch_spike_counts`/`_calibrate_v_th`/`_arm_verdict` "
                      "VERBATIM from iteration-4. Changes ONLY the template-fitting METHOD: iterations 4/5's "
                      "DIAGONAL mean-difference `(mu_gen-mu_perc)/pooled_per_bin_std` (which ignores cross-bin "
                      "covariance) becomes a covariance-aware LEARNED direction -- primary/gating: shrinkage "
                      "Fisher LDA `w=(Sigma_pooled+shrink*mean_diag*I)^{-1}(mu_gen-mu_perc)` (the covariance-"
                      "whitened matched filter); diagnostic: L2-logistic fit by a local delta/Widrow-Hoff rule "
                      "with weight-decay (the estimator family that gets iteration-3's linear decoder 6/6). Each "
                      "signed direction is rectified into the SAME Dale's-law opponent (push-pull) E/I pair "
                      "iteration-5 used (template_E=clip(w,0,None), template_I=clip(-w,0,None)) driving ONE LIF "
                      "readout via net excitatory-minus-inhibitory current, on the TEST fold's own raw spikes, "
                      "signed by structural pool membership -- SAME read, SAME CV, SAME null, SAME gate."),
        "biology": ("The opponent (push-pull) sign realization is unchanged from iteration-5 (Hirsch, Alonso, "
                    "Reid & Martinez 1998, J Neurosci 18(22):9517-9528, PMID 9801388: cat V1 simple-cell "
                    "push-pull -- excitation minus inhibition, never a sign-flipping synapse). This run's "
                    "addition is the covariance-WHITENED matched filter: a matched filter preceded by "
                    "DECORRELATING inhibition, a decades-established cortical/cerebellar circuit motif -- "
                    "recurrent/feedforward inhibition decorrelates and whitens population codes (Cayco-Gajic, "
                    "Clopath & Silver 2017, Nat Commun 8:1116, 'Sparse synaptic connectivity is required for "
                    "decorrelation and pattern separation'; King, Zylberberg & DeWeese 2013, J Neurosci "
                    "33(13):5475-5485, PMID 23536063, doi:10.1523/JNEUROSCI.4188-12.2013 -- a SEPARATE "
                    "inhibitory population (Dale's law) actively DECORRELATES the excitatory population via "
                    "LOCAL synaptic plasticity rules that measure stimulus-dependent between-neuron "
                    "correlations, the exact whitening-by-inhibition motif this read approximates). The "
                    "shrinkage "
                    "term is a homeostatic regularizer preventing the small-sample near-singular covariance from "
                    "amplifying noise -- the 'down-weight noisy channels' the opponent NO-GO's rank-1 residual "
                    "asked for, realized as covariance shrinkage. The logistic diagnostic direction is fit by a "
                    "local error-corrective delta/Widrow-Hoff-Rescorla-Wagner rule with homeostatic weight-"
                    "decay (synaptic scaling)."),
        "scaffold_residuals": [
            "the covariance Sigma and the direction w are fit by a HOST linear solve / gradient descent, not by "
            "an actual spiking decorrelating microcircuit -- the whitening is biologically MOTIVATED (inhibitory "
            "decorrelation) but host-COMPUTED here; a spiking anti-Hebbian whitening layer is the on-substrate "
            "realization if this read direction proves to be the missing power",
            "the opponent read still COLLAPSES the test population into ONE signed pop-contrast per step (fixed "
            "+1/-1 by structural pool membership) then weights it by the 10-bin direction -- it does NOT give "
            "each neuron an individually-learned (matched-filter) synaptic weight; if the LEARNED DIRECTION "
            "separates held-out neurons well (D1) but the LIF read still fails, the residual is THIS read "
            "architecture (population collapse), and the next lever is a per-neuron matched-filter opponent read",
            "each opponent channel is individually rectified but the readout membrane sums their NET drive with "
            "unconstrained current subtraction (current-based LIF, not conductance-based E/I) -- unchanged from "
            "iteration 4/5",
            "LDA_SHRINK/LOGIT_* are host-chosen regularizer knobs (a sensitivity sweep over shrink is reported "
            "so a NO-GO is not a single-lam artifact); TAU_MEM_STEPS/CALIB_TARGET_FRAC/K_FOLDS/R_REPEATS/K_SHUF "
            "unchanged from iteration 4/5",
            "the readout is trained/tested on NEURON IDENTITY folds, not independent TRIALS (this pool family "
            "has no independent trials) -- unchanged, inherited constraint",
            "same host-curated training schedule / topology as the parent crossedge runner (declared there, "
            "unchanged)",
        ],
        "runs": runs,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[READ-FIDELITY LEARNED OPPONENT] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (payload["GO"] or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
