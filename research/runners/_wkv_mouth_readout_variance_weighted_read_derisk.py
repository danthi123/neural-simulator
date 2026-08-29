"""RANK-2 de-risk (`research/findings/2026-08-28-mouth-read-power-wall-deep-research-ranked-shortlist.md` sec.3
item 2) for the mouth read-power wall (#80 / gap#4): does a VARIANCE-WEIGHTED / regularized read that down-weights
noise-dominated input channels BEAT the FIXED (uniform, 1:1) read the batched-substrate read-out uses today, on
the SAME captured substrate data?

THE WALL (do not re-derive): `_wkv_mouth_readout_eprop_batched_substrate_derisk.py`'s learning-forward margin ->
logit conversion is `logits = margin_sub / gain + head_b`, where `gain` is ONE SCALAR shared by all V=1000 word
channels, and every host-feature INPUT dimension d (of D=128) contributes to every word's margin with the SAME
relative weight `head_w[v,d]` the checkpoint assigned it -- the substrate's OWN read noise per input channel is
never accounted for. The sibling read-fidelity F2 crux's opponent/push-pull NO-GO
(`research/findings/2026-08-28-read-fidelity-opponent-pushpull-NOGO-sign-recovery-does-not-lift-and-net-hurts-read-power.md`)
diagnosed that an UNREGULARIZED, uniform (1:1) combination of noisy channels adds variance without matching signal;
its named next lever is "a LEARNED opponent gain ... a variance-weighted combination akin to how Salinas & Abbott's
optimal linear estimator itself down-weights noisy channels" (Salinas & Abbott, J Comput Neurosci 1:89-107, 1994).
THIS runner transplants that lever to the mouth arc, as a clean single-variable A/B.

DESIGN ITERATION (read before the method -- the first attempt was WRONG and is kept as a documented negative,
not silently deleted): iteration 1 fit an INDEPENDENT gain per OUTPUT WORD channel (v in 0..999) from repeated
substrate reads of a small TRAIN stimulus batch. It collapsed catastrophically (held-out var_recov 0.0001-0.44
against a 0.95 fixed baseline, and the SAME collapse in-sample) because (a) with only B~4-8 stimuli, most words'
`host_ideal[b,v]` values are small for EVERY sampled stimulus (a word is only strongly implicated by whichever
stimulus actually targets it), so a per-word OLS slope is fit from an ill-conditioned handful of points and is
dominated by which stimuli happened to be sampled, not by genuine repeat-noise; (b) worse, this is a WINNER-TAKE-
ALL argmax over V INDEPENDENTLY-calibrated channels (unlike Salinas-Abbott's population VECTOR decode, which
COMBINES channels into one continuous estimate) -- any single channel's gain landing near zero from small-sample
noise makes `margin/gain` explode and that channel wins every argmax regardless of true evidence. A trust-region
clip contained the explosion but the underlying estimate was still built on too little, badly-conditioned data
(the smoke run's James-Stein prior variance was itself estimated from those same unstable per-channel points, so
it inflated confidence in obviously-noisy channels: w_v_mean 0.999 -- essentially zero shrinkage). This is filed
because it is itself informative: naive per-OUTPUT-channel gain fitting from a modest capture is ill-posed here.

ITERATION 2 (this file's actual method) moves the variance-weighting to the INPUT side, where it is well-
conditioned: reweight the D=128 host-feature CHANNELS (not the V=1000 output words) by how reliably the substrate
transmits EACH ONE, using a one-hot PROBE SWEEP (drive exactly one input dimension at a time, at a fixed
magnitude, and read the resulting V-word margin pattern R times). Every probe's statistics pool over all V=1000
words x R repeats (thousands of samples per input channel, instead of a handful of samples per output channel) --
the same OLE structure Salinas & Abbott describe (many noisy "cells", here input channels, combined into ONE
downstream estimate), not V independent single-shot calibrations:
  FIXED    (today's method): read with the checkpoint's own head_w, unmodified; ONE global gain (pooled
            least-squares over a [B,V] TRAIN capture, the SAME formula `_calibrate_gain` uses).
  VARIANCE-WEIGHTED (rank-2): for each input dim d, drive a UNIT probe (scale = the RMS host-feature magnitude)
            along d ONLY, read the resulting margin pattern[r,v] R_PROBE times (with head_w already set -- so
            this measures "if only dim d were active, how faithfully does the substrate reproduce head_w[:,d]'s
            V-word pattern"). Fit gain_d (pooled OLS over v, exactly `_calibrate_gain`'s formula but restricted to
            probe d's V-length response) and noise_var_d (residual variance pooled over v AND repeats -- V*R_PROBE
            samples per channel, well-powered). SNR_d = Var_v(gain_d*head_w[:,d]) / max(eps,noise_var_d).
            reliability_d = SNR_d / (SNR_d + c), c = median_d(SNR_d) (TRAIN-only, pre-registered before any TEST
            number is read). Build `head_w_reweighted[v,d] = head_w[v,d] * reliability_d` (down-weight noisy
            input channels toward silence, keep reliable ones near full strength -- exactly the OLE-style down-
            weighting, realized as a per-channel SYNAPTIC GAIN a local Hebbian/BCM-style rule could converge to,
            computed directly here because that converged value is what is being tested). Read THIS weight matrix
            on the substrate (its own gain, fit the SAME pooled way on TRAIN) and score it identically to the
            fixed arm. Reweighting head_w's COLUMNS (shared by every output word) rather than fitting each row/
            output-word independently is what keeps this well-conditioned: 128 channels, not 1000, and every
            TRAIN stimulus informs every channel (a stimulus drives all 128 dims at once), not just the words it
            happens to target.
  BOTH arms add the SAME head_b and are scored against the SAME PF/target ground truth (from the ORIGINAL,
  un-reweighted head_w's softmax) on the SAME held-out stimuli; the read weight matrix (fixed head_w vs
  reliability-reweighted head_w) is the only thing that differs -- a clean single-variable A/B.

DATA: NOT already-recorded on disk (checked: `research/findings/raw/_wkv_mouth_readout_snr_ensemble/`,
`_mouth_readout_tuning/`, `_wkv_mouth_hid_correlation_diagnostic/` hold only aggregate summary JSON, no raw
per-(repeat,block,word) margin traces). This runner does a SMALL FIXED-WEIGHT capture: head_w (ground truth,
never trained here) and head_w_reweighted are each set via `set_weights` and read repeatedly; no gradient step,
no training loop. Reduced scale (B=8, read_window=45, not the decisive 48/120) keeps the whole 6-seed capture
to CPU/numpy in well under an hour. `_reset()` (called every `batch_margin`) clears v/u/firing/conductance state
but NOT the OU noise process or neuron RNG stream, so repeated calls at fixed weights/fixed drive give genuinely
different noise draws each time.

FOLD / NULL DESIGN: "TRAIN" = the probe sweep (dimension reliability, no sentence data at all -- a pure
substrate calibration, exactly as `_calibrate_gain` is a physical measurement, not training) + a TRAIN sentence
capture (each arm's own global gain). "HELD-OUT" = a disjoint TEST sentence/position batch, never used for the
probe sweep or either gain fit -- scored ONCE, after both estimators are frozen. "Channel-identity permutation
null" = re-score the variance-weighted arm on the SAME held-out capture with `reliability_d` PERMUTED across
input-dimension index d (K_PERM draws): if the reweighting exploits real per-dimension reliability structure,
scrambling which dimension gets which weight should destroy the lift; if the lift survives, it was not
dimension-specific.

GO-GATE (pre-registered, from the task prompt): held-out variance-weighted recov EXCEEDS the fixed-estimator
baseline (delta_observed > 0) BEYOND the permutation null (z = (delta_observed - mean(delta_perm)) / std(delta_perm)
>= Z_FLOOR=2.0) at >=5/6 seeds, AND both gains + the reliability weights are fit purely on TRAIN/probe data and
scored purely on HELD-OUT (structural, not a post-hoc check) -- reported per-seed, honestly, including in-sample
numbers for comparison.

ANTI-CHEATS: head_w is NEVER trained/updated by gradient (this is a read-calibration comparison, not a learning
run); the shrinkage constant `c` is a TRAIN/probe-only rule (median TRAIN-probe SNR) fixed BEFORE any TEST number
is read; determinism via cfg.seed (build-twice threshold hash, seed 42 only, to hold cost down); `lever()` checks
the two arms' weight matrices and recov actually differ (the A/B is not accidentally void).

Run (smoke, ~1 seed, few repeats): SIM_BACKEND=numpy .venv/bin/python \
    -m research.runners._wkv_mouth_readout_variance_weighted_read_derisk --smoke --seeds 42
Run (6-seed de-risk): SIM_BACKEND=numpy .venv/bin/python \
    -m research.runners._wkv_mouth_readout_variance_weighted_read_derisk \
    --seeds 42,43,44,100,101,102 \
    --json research/findings/raw/_wkv_mouth_readout_variance_weighted_read_6seed.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["SIM_BACKEND"] = "numpy"                     # CPU ONLY for this de-risk -- set BEFORE any sim import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from sim.backend import to_host  # noqa: E402
from tools.lab import lever, void_if, undefined_if_empty, assert_backend, project_cost  # noqa: E402

from research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk import BatchedSubstrateReadout  # noqa: E402
from research.runners._wkv_mouth_readout_eprop_learn_derisk import _positions  # noqa: E402
from research.runners._wkv_fewspike_read_derisk import WKVReadout, _native, _load_eval  # noqa: E402

Z_FLOOR = 2.0
EPS = 1e-9


def _thr_hash(seed, ro, B, hid_pop, pop, ou_std, read_window, hid_gain, ratio, n_bias, bias_drive_pA):
    s = BatchedSubstrateReadout(ro, seed, B, hid_pop=hid_pop, pop=pop, ou_std=ou_std, read_window=read_window,
                                hid_gain=hid_gain, ratio=ratio, n_bias=n_bias, bias_drive_pA=bias_drive_pA)
    thr = np.asarray(to_host(s._b.cp_neuron_firing_thresholds)).astype(np.float64)
    del s
    return hashlib.sha1(thr.tobytes()).hexdigest()[:16]


def _capture(s_batch, Hb, n_repeats):
    """R fixed-weight repeated substrate reads of the SAME B stimuli/probes. Returns margin[R,B,V]."""
    out = []
    for _ in range(n_repeats):
        out.append(s_batch.batch_margin(Hb, silence_bias=True))
    return np.stack(out, axis=0)                                            # [R, B, V]


def _pooled_gain(margin, host_ideal):
    """ONE pooled least-squares gain over every sample -- `_calibrate_gain`'s own formula, applied here to
    whatever capture is passed in (a TRAIN stimulus batch for the fixed/reweighted global gain, or a probe batch
    for the per-dimension calibration)."""
    mean_r = margin.mean(axis=0)
    num = float((mean_r * host_ideal).sum()); den = float((host_ideal ** 2).sum())
    return num / max(EPS, den)


def _probe_reliability(s_batch, ro, hw, D, B, R_probe, scale, seed):
    """One-hot probe sweep over the D host-feature input channels (head_w already set on s_batch). Returns
    (reliability[D] in (0,1), diag). Well-conditioned: each channel's statistics pool over V*R_PROBE samples."""
    V = ro.V
    dims = np.arange(D)
    n_batches = int(np.ceil(D / B))
    gain_d = np.zeros(D); noise_var_d = np.zeros(D); signal_var_d = np.zeros(D)
    for bi in range(n_batches):
        d_batch = dims[bi * B:(bi + 1) * B]
        nb = len(d_batch)
        feats = np.zeros((B, D))
        for i, d in enumerate(d_batch):
            feats[i, d] = scale
        margin_probe = _capture(s_batch, feats, R_probe)                     # [R,B,V]
        expected = scale * hw[:, d_batch].T                                  # [nb, V] ideal noise-free pattern
        for i, d in enumerate(d_batch):
            m = margin_probe[:, i, :]                                       # [R,V]
            g = _pooled_gain(m[:, None, :], expected[i][None, :])           # reuse pooled-gain formula, B'=1
            resid = m - g * expected[i][None, :]                            # [R,V]
            gain_d[d] = g
            noise_var_d[d] = float(resid.var(ddof=1))
            signal_var_d[d] = float((g * expected[i]).var())
    snr_d = signal_var_d / np.maximum(EPS, noise_var_d)
    c = max(float(np.median(snr_d)), EPS)
    w_d = snr_d / (snr_d + c)
    diag = {"c_shrink": round(c, 6), "snr_d_mean": round(float(snr_d.mean()), 4),
            "snr_d_median": round(float(np.median(snr_d)), 4),
            "w_d_mean": round(float(w_d.mean()), 4), "w_d_min": round(float(w_d.min()), 4),
            "w_d_max": round(float(w_d.max()), 4), "gain_d_mean": round(float(gain_d.mean()), 4)}
    return w_d, diag


def _recov(margin, gain, head_b, Y, PF, unk):
    """margin [R,B,V] raw; gain a SCALAR divisor (both arms use one pooled global gain -- only the WEIGHT MATRIX
    that produced `margin` differs between arms). Returns (recov_argmax, argmax_agree), matching the established
    `recov_argmax = mean_pred_mass / mean_true_mass` convention (`_eval_hostlinear`/`_eval_substrate`)."""
    logits = margin / gain + head_b[None, None, :]                          # [R,B,V]
    if unk >= 0:
        logits = logits.copy(); logits[:, :, unk] = -1e30
    win = logits.argmax(axis=2)                                             # [R,B]
    R, B = win.shape
    mass_read = PF[np.arange(B)[None, :].repeat(R, axis=0), win].mean()
    mass_ax = PF[np.arange(B), Y].mean()
    agree = float((win == Y[None, :]).mean())
    return float(mass_read / max(EPS, mass_ax)), agree


def run_seed(seed, ro, args, seed_hash_check):
    ev_ids, _ = _load_eval(ro, args.corpus, args.n_sentences, seed, args.n_sentences)
    usable = [ids for ids in ev_ids if len(ids) >= args.warmup + 2]
    cut = int(args.frac_train * len(usable))
    train_ids, eval_ids = usable[:cut], usable[cut:]
    Htr, Ytr, PFtr = _positions(ro, train_ids, args.warmup, args.n_train_pos)
    Hte, Yte, PFte = _positions(ro, eval_ids, args.warmup, args.n_test_pos)
    void_if(len(Htr) < args.batch or len(Hte) < args.batch, "insufficient train/test positions")
    head_b = ro.head_b.astype(np.float64)
    hw = ro.head_w
    D, V = ro.D, ro.V

    rngp = np.random.default_rng(seed * 331 + 5)
    tr_idx = rngp.choice(len(Htr), size=args.batch, replace=False)
    te_idx = rngp.choice(len(Hte), size=args.batch, replace=False)
    Hb_tr, Yb_tr, PFb_tr = Htr[tr_idx], Ytr[tr_idx], PFtr[tr_idx]
    Hb_te, Yb_te, PFb_te = Hte[te_idx], Yte[te_idx], PFte[te_idx]
    feat_scale = float(np.sqrt((Htr ** 2).mean()))                          # RMS host-feature magnitude -> probe scale

    s_batch = BatchedSubstrateReadout(ro, seed, args.batch, hid_pop=args.sub_hid_pop, pop=args.sub_pop,
                                      ou_std=args.ou_std, read_window=args.read_window, hid_gain=args.hid_gain,
                                      ratio=args.ratio, settle_frac=args.settle_frac, n_bias=args.n_bias,
                                      bias_drive_pA=args.bias_drive_pA)

    t0 = time.time()
    # -- TRAIN/PROBE: calibrate per-input-channel reliability with head_w already set (the checkpoint's own read) --
    s_batch.set_weights(hw)
    w_d, probe_diag = _probe_reliability(s_batch, ro, hw, D, args.batch, args.repeats_probe, feat_scale, seed)
    hw_reweighted = hw * w_d[None, :]                                       # [V,D] down-weight noisy input dims

    # -- FIXED arm: TRAIN-gain-fit + HELD-OUT score, weights = hw (unmodified) --
    s_batch.set_weights(hw)
    margin_tr_fixed = _capture(s_batch, Hb_tr, args.repeats_train)
    margin_te_fixed = _capture(s_batch, Hb_te, args.repeats_test)
    host_ideal_tr = Hb_tr @ hw.T
    gain_fixed = _pooled_gain(margin_tr_fixed, host_ideal_tr)

    # -- VARIANCE-WEIGHTED arm: TRAIN-gain-fit + HELD-OUT score, weights = hw_reweighted --
    s_batch.set_weights(hw_reweighted)
    margin_tr_var = _capture(s_batch, Hb_tr, args.repeats_train)
    margin_te_var = _capture(s_batch, Hb_te, args.repeats_test)
    host_ideal_tr_var = Hb_tr @ hw_reweighted.T
    gain_var = _pooled_gain(margin_tr_var, host_ideal_tr_var)
    capture_secs = round(time.time() - t0, 1)

    lever(f"weight_matrix_seed{seed}", before=round(float(np.linalg.norm(hw)), 4),
          after=round(float(np.linalg.norm(hw_reweighted)), 4), required=False,
          continuous=round(float(np.abs(hw - hw_reweighted).mean()), 6))

    unk = ro.unk_idx
    fixed_recov_te, fixed_agree_te = _recov(margin_te_fixed, gain_fixed, head_b, Yb_te, PFb_te, unk)
    var_recov_te, var_agree_te = _recov(margin_te_var, gain_var, head_b, Yb_te, PFb_te, unk)
    delta_observed = var_recov_te - fixed_recov_te

    fixed_recov_tr, _ = _recov(margin_tr_fixed, gain_fixed, head_b, Yb_tr, PFb_tr, unk)
    var_recov_tr, _ = _recov(margin_tr_var, gain_var, head_b, Yb_tr, PFb_tr, unk)

    # -- channel(dimension)-identity permutation null: re-read weights = hw * w_d[perm], its OWN pooled gain --
    rng = np.random.default_rng(seed * 991 + 13)
    null_deltas = []
    for _ in range(args.k_perm):
        perm = rng.permutation(D)
        hw_perm = hw * w_d[perm][None, :]
        s_batch.set_weights(hw_perm)
        margin_te_perm = _capture(s_batch, Hb_te, args.repeats_test)
        margin_tr_perm = _capture(s_batch, Hb_tr, max(2, args.repeats_train // 4))   # cheap TRAIN gain re-fit
        gain_perm = _pooled_gain(margin_tr_perm, Hb_tr @ hw_perm.T)
        var_recov_perm, _ = _recov(margin_te_perm, gain_perm, head_b, Yb_te, PFb_te, unk)
        null_deltas.append(var_recov_perm - fixed_recov_te)
    null_deltas = np.asarray(null_deltas)
    null_mean = float(null_deltas.mean()); null_std = float(null_deltas.std(ddof=1)) if len(null_deltas) > 1 else 0.0
    z = (delta_observed - null_mean) / max(EPS, null_std)

    go_seed = bool(delta_observed > 0 and z >= Z_FLOOR)

    m = {
        "seed": seed, "V": ro.V, "D": ro.D, "B": args.batch,
        "repeats_probe": args.repeats_probe, "repeats_train": args.repeats_train, "repeats_test": args.repeats_test,
        "read_window": args.read_window, "feat_scale": round(feat_scale, 4), "capture_secs": capture_secs,
        "seed_hash_check": seed_hash_check,
        "probe_diag": probe_diag,
        "gain_fixed": round(gain_fixed, 6), "gain_var": round(gain_var, 6),
        "held_out": {"fixed_recov": round(fixed_recov_te, 4), "fixed_argmax_agree": round(fixed_agree_te, 4),
                     "var_recov": round(var_recov_te, 4), "var_argmax_agree": round(var_agree_te, 4),
                     "delta": round(delta_observed, 4)},
        "in_sample": {"fixed_recov": round(fixed_recov_tr, 4), "var_recov": round(var_recov_tr, 4),
                     "delta": round(var_recov_tr - fixed_recov_tr, 4)},
        "permutation_null": {"k_perm": args.k_perm, "null_mean": round(null_mean, 4), "null_std": round(null_std, 4),
                              "z": round(z, 4)},
        "go_seed": go_seed,
    }
    print(f"[seed {seed}] HELD-OUT fixed_recov={fixed_recov_te:.4f} var_recov={var_recov_te:.4f} "
          f"delta={delta_observed:+.4f} | null mean={null_mean:+.4f} std={null_std:.4f} z={z:+.3f} | "
          f"in-sample delta={m['in_sample']['delta']:+.4f} | w_d mean={probe_diag['w_d_mean']} | "
          f"GO={go_seed} ({capture_secs}s)", flush=True)
    del s_batch
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=6000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--batch", type=int, default=8)                        # B block-diagonal substrate copies
    ap.add_argument("--n-train-pos", type=int, default=320)
    ap.add_argument("--n-test-pos", type=int, default=320)
    ap.add_argument("--frac-train", type=float, default=0.8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats-probe", type=int, default=3)                # repeats per one-hot input-dim probe
    ap.add_argument("--repeats-train", type=int, default=5)
    ap.add_argument("--repeats-test", type=int, default=5)
    ap.add_argument("--k-perm", type=int, default=20)
    # substrate operating point (the batched-substrate runner's OWN defaults; only B/read_window reduced for CPU)
    ap.add_argument("--sub-pop", type=int, default=1)
    ap.add_argument("--sub-hid-pop", type=int, default=4)
    ap.add_argument("--read-window", type=int, default=45)                 # decisive default is 120; cut for CPU
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_wkv_mouth_readout_variance_weighted_read.json")
    args = ap.parse_args()

    if args.smoke:
        args.n_sentences = min(args.n_sentences, 4000)
        args.batch = min(args.batch, 4)
        args.n_train_pos = min(args.n_train_pos, 80)
        args.n_test_pos = min(args.n_test_pos, 80)
        args.repeats_probe = min(args.repeats_probe, 2)
        args.repeats_train = min(args.repeats_train, 3)
        args.repeats_test = min(args.repeats_test, 3)
        args.k_perm = min(args.k_perm, 3)
        args.read_window = min(args.read_window, 25)

    assert_backend("numpy", note="(this de-risk is CPU-only by design -- no GPU touched)")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    results = []
    seed_hash_check = None
    t_all = time.time()
    for si, seed in enumerate(seeds):
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        if not Path(ckpt).exists():
            print(f"[skip] seed {seed}: checkpoint {ckpt} missing", flush=True)
            continue
        ro = WKVReadout(ckpt)
        if seed_hash_check is None:
            h1 = _thr_hash(seed, ro, args.batch, args.sub_hid_pop, args.sub_pop, args.ou_std, args.read_window,
                           args.hid_gain, args.ratio, args.n_bias, args.bias_drive_pA)
            h2 = _thr_hash(seed, ro, args.batch, args.sub_hid_pop, args.sub_pop, args.ou_std, args.read_window,
                           args.hid_gain, args.ratio, args.n_bias, args.bias_drive_pA)
            seed_hash_check = {"seed": seed, "thr_hash_1": h1, "thr_hash_2": h2, "seeded": bool(h1 == h2)}
            print(f"[seed-trap] thr hash {h1} == {h2} -> {'SEEDED' if h1 == h2 else 'NOT SEEDED'}", flush=True)
        m = run_seed(seed, ro, args, seed_hash_check)
        results.append(m)
        project_cost("variance-weighted-read 6-seed", si + 1, len(seeds), time.time() - t_all, warn_hours=1.5)

    go_n = int(sum(1 for r in results if r["go_seed"]))
    undefined_if_empty("variance_weighted_read_GO_seeds", len(results), go_n, len(results))
    summary = {}
    if results:
        summary = {
            "n_seeds": len(results), "go_count": go_n, "go_5of6": bool(go_n >= 5),
            "fixed_recov_mean": round(float(np.mean([r["held_out"]["fixed_recov"] for r in results])), 4),
            "var_recov_mean": round(float(np.mean([r["held_out"]["var_recov"] for r in results])), 4),
            "delta_mean": round(float(np.mean([r["held_out"]["delta"] for r in results])), 4),
            "z_mean": round(float(np.mean([r["permutation_null"]["z"] for r in results])), 4),
            "n_seeds_delta_positive": int(sum(1 for r in results if r["held_out"]["delta"] > 0)),
            "n_seeds_z_ge_floor": int(sum(1 for r in results if r["permutation_null"]["z"] >= Z_FLOOR)),
        }
    out = {"results": _native(results), "summary": _native(summary), "seeds": seeds,
           "z_floor": Z_FLOOR, "seed_hash_check": seed_hash_check,
           "backend": os.environ.get("SIM_BACKEND", "numpy"), "device": "cpu",
           "provenance": "reanalysis+small-capture: NO raw traces existed on disk (checked _wkv_mouth_readout_"
                         "snr_ensemble/, _mouth_readout_tuning/, _wkv_mouth_hid_correlation_diagnostic/ -- summary "
                         "JSON only), so this runner does a SMALL fixed-weight (head_w, head_w*reliability -- "
                         "neither ever gradient-trained) capture on a reduced-scale (B/read_window cut from the "
                         "decisive 48/120) numpy CPU substrate. No GPU. Iteration 1 (per-output-word gain, "
                         "ill-conditioned) is documented and superseded in-module by iteration 2 (per-input-"
                         "channel reliability reweighting) -- see the module docstring.",
           "elapsed_s": round(time.time() - t_all, 1), "argv": sys.argv}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    if summary:
        print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(results)} rows -> {args.json} ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
