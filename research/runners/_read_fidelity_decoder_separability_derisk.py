"""READ-FIDELITY de-risk, ITERATION 3 -- repoints the crux from "which SCALAR read separates the pools" to
"is there ANY separable signal at all", after all three scalar reads tried so far came up empty on the
surprise->source_provenance F2 crux:
  iteration 1 (`_read_fidelity_nonrate_latency_derisk.py`,
    research/findings/2026-08-28-read-fidelity-nonrate-latency-UNDEFINED.md): first-spike LATENCY, UNDEFINED
    (0/6 PASS, shuffle-identity anti-cheat ambiguous 3/6 -- an instrument problem, not a validated null).
  iteration 2 (`_read_fidelity_nonrate_latency_dispersion_derisk.py`,
    research/findings/2026-08-28-read-fidelity-dispersion-instrument-fixed-still-NO-read-beats-rate.md):
    fixed the instrument (permutation-over-neuron-identity, shuffle collapses 6/6 on BOTH read-kinds), then
    re-ran LATENCY (still 0/6 PASS) and added ISI-CV DISPERSION (1/6 PASS). Three read-kinds now tried
    (rate, latency, dispersion); none clears the floor on more than 1/6 seeds with a validated instrument.

THE REPOINTED QUESTION (this file). Three read PRIMITIVES failing is consistent with two very different
underlying situations, and nothing tried so far distinguishes them:
  (a) READ-FIDELITY LIMIT: the trained cross-edge DOES induce a separable generated-vs-perceived population
      difference, but rate/latency/dispersion are each too coarse a SCALAR summary to detect it (each reduces
      an entire per-neuron spike train to one number, throwing away everything else).
  (b) WIRING/CREDIT PROBLEM: the trained cross-edge does NOT induce any separable population difference at
      all -- the "floor-miss" is an ABSENCE OF SIGNAL, not an instrument limit, and no read of any
      sophistication will find what is not there.
These have OPPOSITE next actions: (a) says keep searching read primitives; (b) says stop reading and go
upstream (the cross-edge's own training/topology). This run tests the two apart DIRECTLY: instead of reducing
each pool to one number, hand the RICHEST available readout (the FULL per-neuron binned spike-count profile)
to a DECODER and ask whether ANY function of it -- linear or nonlinear -- can tell generated from perceived,
on the SAME trained cross-edge and the SAME captured rasters iteration 1/2 already produced. If even a
flexible decoder cannot beat a label-shuffled null, no scalar read ever will either: that is (b), not (a).

THE FEATURE (richer than any of the three prior reads, one honest step, from the SAME raster). Each already-
committed read collapses an entire (RECALL_STEPS=100-step, N_READS=8-read) per-neuron spike train to ONE
number (mean rate; first-spike step; ISI-CV). Here every neuron instead keeps a BINNED spike-COUNT PROFILE:
RECALL_STEPS is split into N_BINS=10 equal windows and each neuron's spike count in each window is recorded,
giving a 10-dimensional per-neuron feature vector -- rate, onset timing (which bins first go nonzero) AND
coarse-grained dispersion (how counts vary bin-to-bin) are all still recoverable from this vector, so it
strictly subsumes what rate/latency/dispersion each measured separately without throwing away the shape a
scalar discards. The N_READS=8 reads are averaged into that one vector (denoising only, NOT treated as 8
independent samples): iteration 2's own diagnosis (this pool family runs with `enable_ou_process=False` AND
`enable_short_term_plasticity=False`, so a fixed reset + fixed drive is a deterministic function with only
~2 distinct achievable raster states) applies identically here, and is why this run's genuine sample axis is
NEURON IDENTITY (n_gen=n_perc=32, n_union=64 per seed), exactly as iteration 2's permutation-over-identity
fix already established -- not read-repetition.

TWO FEATURE VARIANTS, because a RAW per-neuron profile in the held (surprise) condition could trivially
decode pool identity from pre-existing STRUCTURAL asymmetry between the two hard-wired regions (source_
provenance's own fixed pathways into prov_generated/prov_perceived are density-matched by design -- see
`_laneC_source_provenance_opponent_derisk.py` -- but per-neuron heterogeneous thresholds are drawn per-neuron
regardless of pool, so a decoder COULD in principle key off idiosyncratic wiring rather than anything the
cross-edge trained). Reporting only the raw-held decode would conflate "the pools are distinguishable AT ALL"
with "the TRAINED EDGE made them distinguishable" -- the actual crux question. Both are measured:
  `raw_held`        -- the per-neuron profile from the HELD (surprise-driven) read alone. Answers the LITERAL
                        ask ("can any decoder separate generated-vs-perceived from the full raster") and is
                        the more permissive/optimistic reading.
  `delta_held_base` -- (held profile) - (base, no-surprise-hold profile), per neuron. Isolates the surprise-
                        hold-SPECIFIC contribution, net of any static baseline asymmetry between the pools --
                        the same "held margin minus base margin" quantity every prior read targeted, just
                        computed per-neuron instead of per-pool-mean before any statistic is taken.
Both are decoded under BOTH the INTACT (trained) weights and the LESIONED (surprise->provgen zeroed, same
event as iteration 1/2) weights, so decodability that survives lesion (structural) can be told apart from
decodability that vanishes under lesion (cross-edge-attributable) -- 2 weight-conditions x 2 feature variants
x 2 decoders = 8 combos per seed.

TWO DECODERS (numpy only -- no sklearn/torch in this environment; both hand-rolled, Adam-optimized):
  (a) LINEAR  -- L2-regularized logistic regression (`_logreg_fit`/`_logreg_predict`).
  (b) NONLINEAR -- a shallow 1-hidden-layer MLP (H=8 tanh units, sigmoid output, `_mlp_fit`/`_mlp_predict`),
      the task's "small nonlinear decoder" option (in place of an RBF-SVM, which would need a dual-QP solver
      this environment has no library for).
Both are evaluated by REPEATS=5 independent stratified K_FOLDS=5-fold cross-validations (fold membership
re-shuffled each repeat to average out small-N fold-assignment variance), features z-scored on the TRAIN fold
only (no leakage), pooled held-out accuracy per repeat.

ANTI-CHEAT (critical, matches iteration 2's own fix in SHAPE): K_SHUF=20 independent NEURON-IDENTITY
relabelings (a fresh random 32/32 split of which neuron counts as generated/perceived, sizes preserved,
reused from iteration 2's own `_permutation_stats` convention) -- the identical CV pipeline (same features,
same fold machinery, same decoder) is re-run on each shuffled labeling, giving a null accuracy distribution.
`z = (real_mean - null_mean) / null_std`; `shuffle_collapses` requires the FRACTION of null draws that
individually clear Z_FLOOR against the null's own mean/std to stay <= SHUF_COLLAPSE_MAX_RATE (both constants
reused verbatim from iteration 2, same bar, no read-kind gets an easier floor). A combo counts as
`separable` only if real accuracy > 0.5 AND z >= Z_FLOOR AND the shuffle anti-cheat collapses.

INTERPRETATION THE CONTROLLER SHOULD APPLY (pre-registered before any seed was run):
  * real >> shuffled (>=1 combo `separable` on all 6 seeds) => a genuine population signal EXISTS that
    rate/latency/dispersion each missed => the crux is a READ-FIDELITY limit; keep searching read primitives.
  * real ~= shuffled ~= chance on EVERY combo, EVERY seed => no decoder of the two tried, given the richest
    per-neuron readout available, can separate the pools => the crux is a WIRING/CREDIT problem, not a read
    problem; move upstream to the cross-edge's own training/topology instead of trying a 4th read primitive.
  * a mixed picture (e.g. `delta_held_base` separable but `raw_held` at chance, or `intact` separable but
    `lesion` also separable) is itself informative and is reported per-combo, not collapsed into one verdict.

DE-RISK ONLY -- no production wiring, no `sim/` edit, no default flip. Subclasses NOTHING new: reuses
`ReadFidelityPool` (build/train/lesion + the raster-capturing `_drive2`) VERBATIM from the committed
iteration-1 file, and `_capture_reads` (the N_READS (base,held) raster-pair capture) VERBATIM from the
committed iteration-2 file -- SAME trained cross-edge, SAME captured rasters, no retraining confound. numpy
CPU throughout; pool-runnable.

Run:
  SIM_BACKEND=numpy python -m research.runners._read_fidelity_decoder_separability_derisk --seeds 42 --smoke
  SIM_BACKEND=numpy python -m research.runners._read_fidelity_decoder_separability_derisk \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_read_fidelity_decoder_separability_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only -- never touch the GPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
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
from research.runners._onebrain_integration_surprise_episodic_crossedge import (
    F2_LESION_RATIO, CROSS_EDGE_LR, N_EPISODES, HMAX, CUE_PA, CTX_DRIVE_PA, _build_pool,
)

# ---- this run's own pre-registered constants (declared BEFORE any measurement) ----
N_BINS = 10                      # RECALL_STEPS=100 / N_BINS -> 10-step bins; the richest per-neuron feature
Z_FLOOR = 2.0                     # UNCHANGED from iteration 1/2 -- same scale-free significance floor
SHUF_COLLAPSE_MAX_RATE = 0.15     # UNCHANGED from iteration 2 -- same anti-cheat bar, no read-kind gets easier
K_FOLDS = 5                       # stratified k-fold (n_gen=n_perc=32 -> ~6-7 held out per class per fold)
R_REPEATS = 5                     # independent CV repeats (fresh fold split each) for the REAL-label decode
K_SHUF = 20                       # independent neuron-identity relabelings for the null distribution
R_REPEATS_SMOKE = 2               # lighter smoke-mode budget (--smoke), same pipeline, fewer draws
K_SHUF_SMOKE = 5
MLP_HIDDEN = 8                    # shallow -- one hidden layer, H units
L2_REG = 0.01
ADAM_LR = 0.05
LOGREG_EPOCHS = 250
MLP_EPOCHS = 250
WEIGHT_CONDS = ("intact", "lesion")
FEAT_KINDS = ("raw_held", "delta_held_base")
DECODERS = ("linear", "mlp")


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Feature extraction: per-neuron BINNED spike-count profile (subsumes rate/latency/dispersion)
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _binned_features(raster, n_bins):
    """raster: (steps, n_neurons) bool -> (n_neurons, n_bins) float64 per-neuron per-bin spike COUNTS."""
    steps, n = raster.shape
    edges = np.linspace(0, steps, n_bins + 1).astype(int)
    out = np.zeros((n, n_bins), dtype=np.float64)
    for bi in range(n_bins):
        lo, hi = edges[bi], edges[bi + 1]
        out[:, bi] = raster[lo:hi].sum(axis=0)
    return out


def _avg_binned(rasters, n_bins):
    """Mean binned-feature vector over the (near-duplicated, see module docstring) N_READS rasters --
    DENOISING only, not extra independent samples; the genuine sample axis is neuron identity."""
    return np.mean([_binned_features(r, n_bins) for r in rasters], axis=0)


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Hand-rolled decoders (no sklearn/torch in this environment) -- Adam-optimized, numpy only
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _adam_fit(grad_fn, params, epochs, lr=ADAM_LR, b1=0.9, b2=0.999, eps=1e-8):
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    for t in range(1, epochs + 1):
        grads = grad_fn(params)
        for k in params:
            m[k] = b1 * m[k] + (1 - b1) * grads[k]
            v[k] = b2 * v[k] + (1 - b2) * (grads[k] ** 2)
            mhat = m[k] / (1 - b1 ** t)
            vhat = v[k] / (1 - b2 ** t)
            params[k] = params[k] - lr * mhat / (np.sqrt(vhat) + eps)
    return params


def _logreg_grads(params, X, y, l2):
    n = X.shape[0]
    z = X @ params["w"] + params["b"]
    p = _sigmoid(z)
    diff = p - y
    gw = X.T @ diff / n + 2 * l2 * params["w"]
    gb = np.array([diff.mean()])
    return {"w": gw, "b": gb}


def _logreg_fit(X, y, rng):
    d = X.shape[1]
    params = {"w": np.zeros(d), "b": np.zeros(1)}
    return _adam_fit(lambda p: _logreg_grads(p, X, y, L2_REG), params, LOGREG_EPOCHS)


def _logreg_predict(params, X):
    p = _sigmoid(X @ params["w"] + params["b"])
    return p, (p >= 0.5).astype(int)


def _mlp_forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = _sigmoid(z2).ravel()
    return a2, a1, z1


def _mlp_grads(params, X, y, l2):
    n = X.shape[0]
    a2, a1, z1 = _mlp_forward(params, X)
    dz2 = (a2 - y).reshape(-1, 1) / n
    gW2 = a1.T @ dz2 + 2 * l2 * params["W2"]
    gb2 = dz2.sum(axis=0)
    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (1.0 - np.tanh(z1) ** 2)
    gW1 = X.T @ dz1 + 2 * l2 * params["W1"]
    gb1 = dz1.sum(axis=0)
    return {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}


def _mlp_fit(X, y, rng):
    d = X.shape[1]
    params = {
        "W1": rng.normal(0, np.sqrt(2.0 / d), size=(d, MLP_HIDDEN)),
        "b1": np.zeros(MLP_HIDDEN),
        "W2": rng.normal(0, np.sqrt(2.0 / MLP_HIDDEN), size=(MLP_HIDDEN, 1)),
        "b2": np.zeros(1),
    }
    return _adam_fit(lambda p: _mlp_grads(p, X, y, L2_REG), params, MLP_EPOCHS)


def _mlp_predict(params, X):
    a2, _, _ = _mlp_forward(params, X)
    return a2, (a2 >= 0.5).astype(int)


DECODER_FNS = {"linear": (_logreg_fit, _logreg_predict), "mlp": (_mlp_fit, _mlp_predict)}


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Cross-validation + permutation-null (same statistical shape as iteration 2's `_permutation_stats`)
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _stratified_folds(y_bool, k, rng):
    """y_bool: (n,) bool. Returns k (train_idx, test_idx) folds, each preserving class balance -- works for
    ANY labeling (real or shuffled), not just the canonical [gen..., perc...] ordering."""
    pos = np.flatnonzero(y_bool).copy()
    neg = np.flatnonzero(~y_bool).copy()
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_chunks = np.array_split(pos, k)
    neg_chunks = np.array_split(neg, k)
    n = y_bool.size
    folds = []
    for i in range(k):
        test = np.concatenate([pos_chunks[i], neg_chunks[i]])
        train = np.setdiff1d(np.arange(n), test, assume_unique=False)
        folds.append((train, test))
    return folds


def _cv_oof_accuracy(X, y_bool, k, rng, fit_fn, predict_fn):
    """ONE stratified k-fold CV pass. Standardizes each fold's features on the TRAIN split ONLY (no leakage).
    Returns pooled out-of-fold accuracy over all n samples."""
    n = X.shape[0]
    preds = np.full(n, -1, dtype=int)
    y_num = y_bool.astype(np.float64)
    for train_idx, test_idx in _stratified_folds(y_bool, k, rng):
        mu = X[train_idx].mean(axis=0)
        sd = X[train_idx].std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        Xtr = (X[train_idx] - mu) / sd
        Xte = (X[test_idx] - mu) / sd
        params = fit_fn(Xtr, y_num[train_idx], rng)
        _, cls = predict_fn(params, Xte)
        preds[test_idx] = cls
    assert (preds >= 0).all(), "every sample must be predicted exactly once out-of-fold"
    return float((preds == y_bool.astype(int)).mean())


def _combo_stats(X, y_real, n_gen, n_perc, decoder_name, rng, repeats, k_shuf):
    """Real (REPEATS-repeated stratified CV) vs a K_SHUF-draw neuron-identity permutation null, on the
    IDENTICAL feature matrix X and CV machinery -- the anti-cheat. z-score + shuffle-collapse fraction mirror
    iteration 2's `_permutation_stats` exactly (same Z_FLOOR, same SHUF_COLLAPSE_MAX_RATE)."""
    fit_fn, predict_fn = DECODER_FNS[decoder_name]
    real_accs = np.array([_cv_oof_accuracy(X, y_real, K_FOLDS, rng, fit_fn, predict_fn) for _ in range(repeats)])
    n_union = n_gen + n_perc
    null_accs = []
    for _ in range(k_shuf):
        perm = rng.permutation(n_union)
        y_shuf = np.zeros(n_union, dtype=bool)
        y_shuf[perm[:n_gen]] = True
        null_accs.append(_cv_oof_accuracy(X, y_shuf, K_FOLDS, rng, fit_fn, predict_fn))
    null_accs = np.asarray(null_accs, dtype=np.float64)

    real_mean = float(real_accs.mean())
    real_std = float(real_accs.std(ddof=1)) if real_accs.size > 1 else 0.0
    null_mean = float(null_accs.mean())
    null_std = float(null_accs.std(ddof=1)) if null_accs.size > 1 else 0.0
    z = (real_mean - null_mean) / null_std if null_std > 0 else (float("inf") if real_mean != null_mean else 0.0)
    frac_null_clears = (float(np.mean(np.abs((null_accs - null_mean) / null_std) >= Z_FLOOR))
                         if null_std > 0 else float("nan"))
    shuffle_collapses = bool(frac_null_clears <= SHUF_COLLAPSE_MAX_RATE) if not np.isnan(frac_null_clears) else False
    separable = bool(real_mean > 0.5 and z >= Z_FLOOR and shuffle_collapses)
    return {
        "decoder": decoder_name, "real_acc_mean": real_mean, "real_acc_std": real_std,
        "real_acc_all": [float(x) for x in real_accs],
        "null_acc_mean": null_mean, "null_acc_std": null_std,
        "z": float(z), "n_repeats": int(repeats), "n_shuf": int(k_shuf),
        "frac_null_clears_floor": frac_null_clears, "shuffle_collapses": shuffle_collapses,
        "separable": separable,
    }


def _seed_trap_check(seed):
    """CLAUDE.md's own recipe: build the pool TWICE at the same seed and hash a per-neuron array that is
    seeded from `cfg.seed` at build time (`cp_neuron_firing_thresholds`) -- identical => genuinely seeded,
    not silently drawing from the unseeded global RNG. Uses the bare (untrained) pool, not the full
    ReadFidelityPool (which additionally runs the ambiguous-pattern encode) -- cheaper, and the claim under
    test (substrate seeding) is settled at BUILD time, before any training."""
    p1 = _build_pool(seed)
    p1.ensure_built()
    p2 = _build_pool(seed)
    p2.ensure_built()
    t1 = np.asarray(to_host(p1.bridge.cp_neuron_firing_thresholds))
    t2 = np.asarray(to_host(p2.bridge.cp_neuron_firing_thresholds))
    identical = bool(np.array_equal(t1, t2))
    return {"identical": identical, "n_neurons": int(t1.size),
            "hash_build1": hashlib.sha256(t1.tobytes()).hexdigest()[:16],
            "hash_build2": hashlib.sha256(t2.tobytes()).hexdigest()[:16]}


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
    y_real = np.zeros(union.size, dtype=bool)
    y_real[:n_gen] = True
    read_dict = {"gen": ix["prov_generated"], "perc": ix["prov_perceived"]}

    # ---- INTACT ----
    pairs_intact = _capture_reads(pool, read_dict, union)
    held_feat_i = _avg_binned([h for (_b, h) in pairs_intact], N_BINS)
    base_feat_i = _avg_binned([b for (b, _h) in pairs_intact], N_BINS)

    # ---- LESIONED (same event as iteration 1/2: zero surprise->provgen, in place) ----
    data = np.asarray(to_host(pool.b.cp_connections.data)).copy()
    data[pool.masks["surprise->provgen"]] = 0.0
    pool.b.cp_connections.data = pool.xp.asarray(data, dtype=pool.b.cp_connections.data.dtype)
    pairs_lesion = _capture_reads(pool, read_dict, union)
    held_feat_l = _avg_binned([h for (_b, h) in pairs_lesion], N_BINS)
    base_feat_l = _avg_binned([b for (b, _h) in pairs_lesion], N_BINS)

    feats = {
        "intact": {"raw_held": held_feat_i, "delta_held_base": held_feat_i - base_feat_i},
        "lesion": {"raw_held": held_feat_l, "delta_held_base": held_feat_l - base_feat_l},
    }

    base_off = int(seed) * 15485863 + 271   # this module's own distinct RNG offset family (see other module
                                              # docstrings for the sibling offsets already in use: *104729+17,
                                              # *65599+41, *7919+101, *997+3 -- none collide with this one)
    combos = {}
    combo_i = 0
    for cond in WEIGHT_CONDS:
        for feat_kind in FEAT_KINDS:
            X = feats[cond][feat_kind]
            for decoder_name in DECODERS:
                rng = np.random.default_rng(base_off + combo_i * 104651)
                combo_i += 1
                combos[f"{cond}__{feat_kind}__{decoder_name}"] = _combo_stats(
                    X, y_real, n_gen, n_perc, decoder_name, rng, repeats, k_shuf)

    any_separable = any(c["separable"] for c in combos.values())
    all_at_chance = all((not c["separable"]) and abs(c["real_acc_mean"] - 0.5) < 0.08
                         for c in combos.values())
    null_means = [c["null_acc_mean"] for c in combos.values()]
    max_null_dev_from_chance = float(max(abs(m - 0.5) for m in null_means))
    n_shuffle_collapse = sum(c["shuffle_collapses"] for c in combos.values())

    # intact-vs-lesion attributability on delta_held_base (the crux-matched feature), per decoder --
    # (accuracy - 0.5) as the effect size so attributable_to's "effect from zero" semantics apply.
    attributability = {}
    for decoder_name in DECODERS:
        ti = combos[f"intact__delta_held_base__{decoder_name}"]["real_acc_mean"] - 0.5
        tl = combos[f"lesion__delta_held_base__{decoder_name}"]["real_acc_mean"] - 0.5
        frac = attributable_to(f"decoder({decoder_name}) delta_held_base separability, intact vs lesion", ti, tl)
        attributability[decoder_name] = None if frac is None else float(frac)

    return {
        "seed": int(seed), "elapsed_s": round(time.time() - t0, 1),
        "cue_concept": pool.cue_c, "assert_concept": pool.assert_cp,
        "final_weight_trained_block": float(traj[-1]["w"]), "final_weight_other_blocks": float(traj[-1]["w_other"]),
        "emergence_grew_from_near_zero": emg_grew, "emergence_other_blocks_stayed_near_seed": emg_specific,
        "n_gen": n_gen, "n_perc": n_perc,
        "combos": combos,
        "any_combo_separable": bool(any_separable),
        "all_combos_at_chance": bool(all_at_chance),
        "n_combos_shuffle_collapse": int(n_shuffle_collapse), "n_combos": len(combos),
        "max_null_deviation_from_chance": max_null_dev_from_chance,
        "attributable_to_crossedge_delta_held_base": attributability,
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
        best = max(r["combos"].items(), key=lambda kv: kv[1]["z"] if not np.isnan(kv[1]["z"]) else -1e9)
        print(f"[seed {s}] ({r['elapsed_s']}s) block(c={r['cue_concept']},c'={r['assert_concept']}) "
              f"w={r['final_weight_trained_block']:.2f} w_other={r['final_weight_other_blocks']:.3f} | "
              f"any_separable={r['any_combo_separable']} all_at_chance={r['all_combos_at_chance']} "
              f"shuf_collapse={r['n_combos_shuffle_collapse']}/{r['n_combos']} | "
              f"best_combo={best[0]} acc={best[1]['real_acc_mean']:.3f} "
              f"null={best[1]['null_acc_mean']:.3f} z={best[1]['z']:.2f} sep={best[1]['separable']}",
              flush=True)

    n_any_separable = sum(r["any_combo_separable"] for r in runs)
    n_all_chance = sum(r["all_combos_at_chance"] for r in runs)
    n_shuf_ok = sum(r["n_combos_shuffle_collapse"] == r["n_combos"] for r in runs)

    dec, preconditions = None, []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("read_fidelity_decoder_separability_derisk")
        Vd.require("shuffle_anticheat_collapses_on_every_combo",
                   1 if all(r["n_combos_shuffle_collapse"] == r["n_combos"] for r in runs) else 0,
                   expect=lambda x: x >= 1,
                   note="the neuron-identity permutation null must individually clear Z_FLOOR on <= "
                        f"{SHUF_COLLAPSE_MAX_RATE} of its own draws, on EVERY combo, before any real-vs-null "
                        "comparison here can be trusted -- same instrument-validity bar as iteration 2")
        Vd.require("null_pipeline_centers_near_chance",
                   1 if all(r["max_null_deviation_from_chance"] < 0.15 for r in runs) else 0,
                   expect=lambda x: x >= 1,
                   note="the shuffled-label null's own mean accuracy must sit near 0.5 -- a biased null (e.g. "
                        "CV leakage) would inflate BOTH real and null accuracy together and hide a real signal "
                        "behind an equally-inflated floor")
        Vd.require("emergence_grew_from_near_zero", 1 if all(r["emergence_grew_from_near_zero"] for r in runs) else 0,
                   expect=lambda x: x >= 1, note="the reused cross-edge trained normally (sanity on the shared substrate)")
        Vd.require("anti_cheat_random_assignment", 1 if len(set((r["cue_concept"], r["assert_concept"])
                   for r in runs)) > 1 else 0, expect=lambda x: x >= 1,
                   note="the per-seed block pair must actually vary (inherited from the parent runner)")
        # `go` here means "a genuine, reproducible signal was found" -- the OPTIMISTIC reading (a). The
        # PESSIMISTIC reading (b, wiring problem) is reported in the verdict text regardless of this tag,
        # because both outcomes are equally valuable answers to the question this run asks.
        go_signal_found = bool(n_any_separable == len(runs)) and not args.smoke
        dec = Vd.decide(go_signal_found, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    verdict_status = dec.get("status") if dec else None
    if verdict_status == "UNDEFINED":
        tag = "UNDEFINED"
    elif args.smoke:
        tag = f"SMOKE ({'signal-found' if runs[0]['any_combo_separable'] else 'chance-level'}, 1-seed indicator)"
    elif n_any_separable == len(runs):
        tag = "SIGNAL-FOUND (read-fidelity limit) -- reads are inadequate, keep searching read primitives"
    elif n_all_chance == len(runs):
        tag = "NO-SIGNAL (wiring/credit problem) -- move upstream, not another read primitive"
    else:
        tag = f"MIXED -- {n_any_separable}/{len(runs)} seeds show a decodable signal, {n_all_chance}/{len(runs)} at chance on every combo"

    verdict = (f"{tag}. Richest available per-neuron readout (10-bin spike-count profile, subsumes rate/"
               f"latency/dispersion) fed to a LINEAR (logistic regression) and a NONLINEAR (shallow MLP) "
               f"decoder, cross-validated (K_FOLDS={K_FOLDS} x R_REPEATS={repeats} repeats), on the SAME "
               f"trained cross-edge and SAME captured rasters iteration 1/2 used (no retraining confound). "
               f"2 weight-conditions (intact/lesion) x 2 feature variants (raw_held/delta_held_base) x 2 "
               f"decoders = 8 combos/seed. Anti-cheat: K_SHUF={k_shuf}-draw neuron-identity permutation null, "
               f"shuffle collapses on {sum(r['n_combos_shuffle_collapse'] for r in runs)}/"
               f"{sum(r['n_combos'] for r in runs)} combo-seed pairs. "
               f"{n_any_separable}/{len(runs)} seeds have >=1 separable combo; {n_all_chance}/{len(runs)} seeds "
               f"are at chance on EVERY combo. "
               + (f" UNDEFINED, NOT a validated verdict either way: {len(dec.get('undefined_reasons', []))} "
                  f"precondition(s) unmet -- {'; '.join(dec.get('undefined_reasons', []))}."
                  if verdict_status == "UNDEFINED" else ""))

    payload = {
        "probe": "read_fidelity_decoder_separability_derisk", "verdict": verdict,
        "GO_signal_found": bool(n_any_separable == len(runs)) and not args.smoke,
        "ALL_CHANCE_wiring_problem": bool(n_all_chance == len(runs)) and not args.smoke,
        "n_seeds": len(runs), "n_seeds_any_separable": n_any_separable, "n_seeds_all_chance": n_all_chance,
        "n_seeds_shuffle_ok_on_every_combo": n_shuf_ok,
        "seeds": seeds, "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
        "preconditions": preconditions,
        "seed_trap_build_twice": seed_trap,
        "config": {
            "n_bins": N_BINS, "z_floor": Z_FLOOR, "shuf_collapse_max_rate": SHUF_COLLAPSE_MAX_RATE,
            "k_folds": K_FOLDS, "r_repeats": repeats, "k_shuf": k_shuf,
            "mlp_hidden": MLP_HIDDEN, "l2_reg": L2_REG, "adam_lr": ADAM_LR,
            "logreg_epochs": LOGREG_EPOCHS, "mlp_epochs": MLP_EPOCHS,
            "weight_conds": list(WEIGHT_CONDS), "feat_kinds": list(FEAT_KINDS), "decoders": list(DECODERS),
            "recall_steps": RECALL_STEPS, "n_reads": N_READS, "pre_steps": PRE_STEPS,
            "episode_drive_pa": EPISODE_DRIVE_PA, "f2_lesion_ratio": F2_LESION_RATIO,
            "cross_edge_hebbian_lr": CROSS_EDGE_LR, "n_episodes": N_EPISODES,
            "hebbian_max_weight": HMAX, "cue_pa": CUE_PA, "ctx_drive_pa": CTX_DRIVE_PA,
            "rng_formula": "seed*15485863+271, +combo_index*104651 per combo (this module's own distinct "
                            "offset family; does not collide with _assign_blocks(*104729+17), "
                            "_shuffle_mask(*65599+41), iteration-2's permutation rng(*7919+101), or "
                            "_make_ambiguous_pattern(*997+3))",
        },
        "mechanism": ("Reuses ReadFidelityPool (build/train/lesion + the raster-capturing `_drive2`) VERBATIM "
                      "from the committed iteration-1 file, and `_capture_reads` (N_READS (base,held) raster-"
                      "pair capture) VERBATIM from the committed iteration-2 file. Adds ONLY: a per-neuron "
                      "10-bin spike-count feature (richer than any prior scalar read), two hand-rolled "
                      "decoders (L2 logistic regression; a shallow 1-hidden-layer MLP, both Adam-optimized, "
                      "numpy only -- no sklearn/torch available in this environment), stratified k-fold "
                      "cross-validation with train-fold-only standardization (no leakage), and a neuron-"
                      "identity permutation-null anti-cheat in the same statistical shape as iteration 2's own "
                      "fix (K_SHUF draws, z-score vs the null, a shuffle-collapse fraction bar)."),
        "biology": ("Population/pattern DECODING as an analysis method (not a claimed biological readout "
                    "mechanism): Averbeck, Latham & Pouget 2006 (Nat Rev Neurosci 7:358-366, 'Neural "
                    "correlations, population coding and computation') and Quiroga & Panzeri 2009 (Nat Rev "
                    "Neurosci 10:173-185, 'Extracting information from neuronal populations: information "
                    "theory and decoding approaches') establish that a population code's information content "
                    "is a property of the FULL joint response, not any single summary statistic -- a linear or "
                    "nonlinear decoder trained on the full per-neuron response can detect separations that a "
                    "reduction to one scalar (rate, latency, or dispersion) provably cannot. This run is that "
                    "check applied directly: if even a flexible decoder on the richest available per-neuron "
                    "readout cannot beat a label-shuffled null, the absence is in the SUBSTRATE's population "
                    "response, not in any one read primitive's sensitivity."),
        "scaffold_residuals": [
            "N_BINS=10 binning of RECALL_STEPS is a host-chosen resolution, not a computed feature -- a coarser "
            "or finer binning could in principle shift a borderline combo's z-score",
            "the decoders (L2 logistic regression; an 8-unit-hidden MLP) are two points in a much larger "
            "decoder-family space; a genuine absence-of-signal verdict from THESE two is evidence, not proof, "
            "that NO decoder could ever separate the pools",
            "K_FOLDS=5/R_REPEATS/K_SHUF are host-chosen statistical-power/tolerance knobs (same class as "
            "iteration 2's K_PERM/SHUF_COLLAPSE_MAX_RATE), not computed features",
            "N_READS=8 reads averaged into ONE per-neuron feature vector per condition (denoising only, per "
            "this pool family's deterministic dynamics -- see module docstring); the genuine per-seed sample "
            "count is n_union=64 neurons, not 64*8",
            "same host-curated training schedule / topology as the parent crossedge runner (declared there, "
            "unchanged)",
        ],
        "runs": runs,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[READ-FIDELITY DECODER] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (payload["GO_signal_found"] or payload["ALL_CHANCE_wiring_problem"] or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
