"""gap#4 / #80 mouth read-SNR -- DIAGNOSE the ||W||->cap runaway (candidate: read-CONFIDENCE / gain miscalibration).

Two prior findings (2026-08-27) established: (1) at FULL data scale the substrate-forward learned readout hits the
||W||->cap (40) with weight_cosine 0.136 / recov 0.371, vs the host-proxy (exact linear map W@h+head_b) which
converges NATURALLY at ||W||~24 with wcos 0.40 / recov 0.86 -- i.e. the substrate read costs ~0.26 wcos and ~0.49
recov via a weight-norm runaway, NOT via read noise (averaging K reads is flat) and NOT via a per-word direction
artifact (the substrate gradient is cos~0.9928 aligned with the ideal map's gradient). (2) the ONE per-seed GAIN
scalar (`_calibrate_gain` in the eprop_batched runner) is measured ONCE with a RANDOM-DIRECTION probe
(0.12*randn(V,D), a diffuse/isotropic weight pattern) and used for the WHOLE run via `margin_sub/gain`.

THIS SCRIPT asks: is that ONE random-direction gain scalar actually the right SCALE for the STRUCTURED direction
the learning rule is trying to recover (head_w, or any correlated/low-entropy weight pattern), or does the graded
conductance readout compress (under-read) STRUCTURED drive relative to what an isotropic random probe of the SAME
Frobenius norm predicts? A structured direction concentrates correlated excitatory/inhibitory drive onto the SAME
postsynaptic pools; if the driving-force term (`df_e*g_e + df_i*g_i`, both g's are population FIRING-RATE-driven
conductance sums) is not perfectly linear in total synaptic drive (e.g. the driving hid/hidinh pools' firing rate
saturates), then a HIGH-KURTOSIS (structured) weight pattern will read out SMALLER, relative to its ideal linear
margin, than a LOW-KURTOSIS (random/diffuse) pattern of the same norm -- exactly a magnitude/confidence
miscalibration, not a direction (gradient) or noise artifact.

Probes (all NO TARGET LEAK for candidate fixes: (D)/(E) use head_w only for THIS DIAGNOSIS, to characterize the
compression; the proposed fix -- self-referential adaptive recalibration against the CURRENT W_hat -- never reads
head_w or the labels):
  (A) the CURRENT calibration: random probe, scale 0.12 (the eprop runner's fixed init-scale probe).
  (B) random probe RESCALED to ||head_w|| (~37.5) -- isolates NORM alone (random direction).
  (C) random probe RESCALED to the w_target cap (40) -- isolates NORM alone at the cap.
  (D) head_w ITSELF (natural norm ~37.5) -- the real structured target direction.
  (E) head_w-DIRECTION rescaled DOWN to probe (A)'s norm (~43) -- isolates STRUCTURE at a MATCHED norm vs (A).
  (F) DIRECT test: margin_sub(head_w) / gain_A vs the exact host-linear margin of head_w -- is the CURRENT
      calibration's gain under-reading the actual target direction?

Then tests the FIX: an ADAPTIVE gain that periodically RE-CALIBRATES using the CURRENT W_hat as its own probe
(self-referential AGC -- Turrigiano/Carandini-Heeger-style operating-point gain recalibration; no target/label
leak, `host_matmul` used only as a calibration reference for W_hat that already has NO gradient dependency, exactly
like the existing `_calibrate_gain` probe) -- confirms whether this makes a SHORT training run converge ||W|| near
~24 (not the 40 cap) instead of running away.

ANTI-CHEATS: same substrate forward as the production runner (0 host matmul on the actual learning error signal);
the calibration matmul is a DECLARED residual (identical in kind to the existing `_calibrate_gain`); determinism via
cfg.seed (build-twice hash, reused from the eprop runner's `_thr_hash`). Runner-only, additive, no sim/ edit.

Run (diagnosis only, ~1-2 min):
  SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_readout_softmax_confidence_derisk --seed 42 \
      --json research/findings/raw/_wkv_softmax_confidence/diag_s42.json

Run (diagnosis + short adaptive-gain training probe, ~5-10 min):
  SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_readout_softmax_confidence_derisk --seed 42 \
      --short-train --json research/findings/raw/_wkv_softmax_confidence/diag_train_s42.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from tools.lab import lever, assert_backend  # noqa: E402

from research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk import (  # noqa: E402
    BatchedSubstrateReadout, _sub_logits, _softmax_rows, _wcos, _native, _thr_hash,
)
from research.runners._wkv_mouth_readout_eprop_learn_derisk import _positions  # noqa: E402
from research.runners._wkv_fewspike_read_derisk import WKVReadout, _load_eval  # noqa: E402


def _measure_gain(s_batch, W_probe, feats_signed):
    """Same regression `_calibrate_gain` does, but on a CALLER-SUPPLIED weight matrix (not always random) so we can
    probe different SCALES and DIRECTIONS. host_probe is a CALIBRATION-only matmul (never the learning forward).

    Reports BOTH a GLOBAL correlation (over the flattened [B,V] array -- conflates between-ROW/position variance
    with within-row/across-vocab structure) and a ROW-CENTERED correlation (each of the B positions' V-length row
    mean-subtracted BEFORE flattening). The actual learning signal is `softmax(logits)` computed PER ROW -- softmax
    is invariant to a per-row additive shift (softmax(x + c*1) = softmax(x)) -- so a between-row baseline that the
    GLOBAL correlation is sensitive to is INVISIBLE to the real error signal. Row-centered corr is the metric that
    actually predicts what the softmax-onehot delta rule sees; a global-vs-row-centered gap indicates the raw
    margin carries a large per-position (across-V-roughly-constant) baseline that training never has to fight."""
    Wfull = np.concatenate([W_probe, -W_probe], axis=1)
    featF = np.concatenate([np.maximum(feats_signed, 0.0), np.maximum(-feats_signed, 0.0)], axis=1)
    host_probe = featF @ Wfull.T                                          # [B, V] IDEAL linear margin (CALIB only)
    s_batch.set_weights(W_probe)
    margin_sub = s_batch.batch_margin(feats_signed, silence_bias=True)    # [B, V] SUBSTRATE read
    hp = host_probe.reshape(-1); ms = margin_sub.reshape(-1)
    num = float((ms * hp).sum()); den = float((hp * hp).sum())
    gain = num / max(1e-12, den)
    corr = float(np.corrcoef(hp, ms)[0, 1]) if hp.std() > 1e-12 and ms.std() > 1e-12 else 0.0
    # row-centered: subtract EACH ROW's own mean (across V) before flattening -- isolates within-row structure.
    hp_rc = (host_probe - host_probe.mean(axis=1, keepdims=True)).reshape(-1)
    ms_rc = (margin_sub - margin_sub.mean(axis=1, keepdims=True)).reshape(-1)
    corr_rc = float(np.corrcoef(hp_rc, ms_rc)[0, 1]) if hp_rc.std() > 1e-12 and ms_rc.std() > 1e-12 else 0.0
    # magnitude ratio: ||margin_sub|| / ||gain * host_probe(gain=1 scale, i.e. host_probe itself)|| tells us, at
    # gain=1, how much the substrate UNDER/OVER-reads a unit-gain ideal map (informative alongside the fitted gain).
    resid = float(np.linalg.norm(ms - gain * hp) / (np.linalg.norm(gain * hp) + 1e-12))
    # per-row baseline magnitude vs within-row (signal) magnitude -- quantifies how much of margin_sub's variance is
    # a between-row constant (invisible to softmax) vs real across-V structure (visible to softmax).
    row_mean_norm = float(np.linalg.norm(margin_sub.mean(axis=1)))
    within_row_norm = float(np.linalg.norm(ms_rc))
    return {"gain": round(gain, 6), "corr": round(corr, 4), "corr_row_centered": round(corr_rc, 4),
            "resid_ratio": round(resid, 4), "probe_norm": round(float(np.linalg.norm(W_probe)), 3),
            "margin_sub_norm": round(float(np.linalg.norm(ms)), 3),
            "host_probe_norm": round(float(np.linalg.norm(hp)), 3),
            "margin_row_baseline_norm": round(row_mean_norm, 3),
            "margin_within_row_norm": round(within_row_norm, 3)}, margin_sub, host_probe


def run_diagnosis(seed, args):
    ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
    ro = WKVReadout(ckpt)

    # seed-trap: build-twice hash (CLAUDE.md cfg.seed rule)
    h1 = _thr_hash(seed, ro, args.sub_hid_pop, args.sub_pop, args.ou_std, args.sub_read_window,
                   args.hid_gain, args.ratio, args.n_bias, args.bias_drive_pA)
    h2 = _thr_hash(seed, ro, args.sub_hid_pop, args.sub_pop, args.ou_std, args.sub_read_window,
                   args.hid_gain, args.ratio, args.n_bias, args.bias_drive_pA)
    seed_hash_check = {"seed": seed, "thr_hash_1": h1, "thr_hash_2": h2, "seeded": bool(h1 == h2)}
    print(f"[seed-trap] {h1} == {h2} -> {'SEEDED' if h1 == h2 else 'NOT SEEDED'}", flush=True)

    ev_ids, _ = _load_eval(ro, args.corpus, args.n_sentences, seed, args.n_sentences)
    usable = [ids for ids in ev_ids if len(ids) >= args.warmup + 2]
    cut = int(0.8 * len(usable))
    train_ids = usable[:cut]
    H, Y, _ = _positions(ro, train_ids, args.warmup, max(args.batch, 64))

    s_batch = BatchedSubstrateReadout(ro, seed, args.batch, hid_pop=args.sub_hid_pop, pop=args.sub_pop,
                                      ou_std=args.ou_std, read_window=args.sub_read_window, hid_gain=args.hid_gain,
                                      ratio=args.ratio, settle_frac=args.settle_frac, n_bias=args.n_bias,
                                      bias_drive_pA=args.bias_drive_pA)
    feats = H[:args.batch]
    hw = ro.head_w.astype(np.float64)
    hw_norm = float(np.linalg.norm(hw))

    rng = np.random.default_rng(seed * 7 + 3)
    Wa = 0.12 * rng.standard_normal((ro.V, ro.D))                          # (A) current calibration probe
    a_norm = float(np.linalg.norm(Wa))
    Wb = Wa * (hw_norm / a_norm)                                           # (B) random, rescaled to ||head_w||
    Wc = Wa * (args.w_target / a_norm)                                     # (C) random, rescaled to the cap
    We = hw * (a_norm / hw_norm)                                           # (E) head_w-direction, rescaled to (A)'s norm

    resA, marginA, hostA = _measure_gain(s_batch, Wa, feats)
    resB, marginB, hostB = _measure_gain(s_batch, Wb, feats)
    resC, marginC, hostC = _measure_gain(s_batch, Wc, feats)
    resD, marginD, hostD = _measure_gain(s_batch, hw, feats)               # (D) head_w itself, natural norm
    resE, marginE, hostE = _measure_gain(s_batch, We, feats)

    # (G) SCALE SWEEP along the head_w DIRECTION (0.1x..1.0x): reuses the SAME built s_batch (cheap, no rebuild) --
    # localizes WHERE the structured-direction correlation collapses (a smooth saturation-with-scale vs an abrupt
    # structure-only collapse independent of scale).
    scale_sweep = []
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        Wg = hw * frac
        resG, _, _ = _measure_gain(s_batch, Wg, feats)
        resG["frac_of_headw"] = frac
        scale_sweep.append(resG)
        print(f"[seed {seed}] scale-sweep frac={frac} norm={resG['probe_norm']} gain={resG['gain']} "
              f"corr={resG['corr']} corr_rc={resG['corr_row_centered']} margin_sub_norm={resG['margin_sub_norm']} "
              f"host_probe_norm={resG['host_probe_norm']} row_baseline={resG['margin_row_baseline_norm']} "
              f"within_row={resG['margin_within_row_norm']}", flush=True)

    # (F) DIRECT test: is head_w's substrate margin, calibrated by the CURRENT gain_A, under-scaled vs the exact
    # host-linear margin of head_w? ratio < 1 => under-read at the target direction using today's calibration.
    logit_via_gA = marginD / resA["gain"]
    ratio_direct = float(np.linalg.norm(logit_via_gA) / (np.linalg.norm(hostD) + 1e-12))

    # ORDER-CONFOUND CONTROL: re-test the SAME probe A (random) at the END of the sequence (after 10 head_w-related
    # reads) -- if corr stays ~0.95 (matching the FIRST A read), the head_w-direction result is NOT an artifact of
    # read-order/accumulated-state; if it has degraded toward 0 too, the whole sequence is order-confounded.
    resA_repeat, _, _ = _measure_gain(s_batch, Wa, feats)
    print(f"[seed {seed}] ORDER-CONTROL: A repeated at END corr={resA_repeat['corr']}/"
          f"{resA_repeat['corr_row_centered']}rc (first A was corr={resA['corr']}/{resA['corr_row_centered']}rc) "
          f"-> {'NOT an order artifact' if resA_repeat['corr'] > 0.7 else 'ORDER-CONFOUNDED, RESULT SUSPECT'}",
          flush=True)

    # RECONCILIATION with the prior gradalign finding (2026-08-27, cos(g_sub,g_host)~0.975 at W=head_w): that test
    # compares POST-SOFTMAX ERROR gradients (P-onehot)^T@h, which share an identical, LABEL-DRIVEN "-onehot^T@h"
    # term between g_sub and g_host regardless of how informative the read is -- a shared component that can inflate
    # cosine alignment even when the read's raw margin carries near-zero information (this test's corr~0.03 finding).
    # Reproduce that EXACT quantity here (same head_w, same gain, same batch) to confirm both are correct facts about
    # DIFFERENT quantities, not a contradiction.
    Yb = Y[:args.batch]
    head_b = ro.head_b.astype(np.float64)
    def _sgrad(margin_or_host, gain_val, Hb_, Yb_):
        logits = margin_or_host / gain_val + head_b[None, :]
        if unk >= 0:
            logits = logits.copy(); logits[:, unk] = -1e30
        P = _softmax_rows(logits)
        P[np.arange(len(Yb_)), Yb_] -= 1.0
        return (P.T @ Hb_) / len(Yb_)
    unk = ro.unk_idx
    g_host = _sgrad(hostD, 1.0, feats, Yb)               # host EXACT map: logits = host_ideal + head_b (gain=1)
    g_sub = _sgrad(marginD, resA["gain"], feats, Yb)      # substrate: logits = margin_sub(head_w)/gain_A + head_b
    cos_grad_headw = float((g_host.reshape(-1) @ g_sub.reshape(-1))
                           / (np.linalg.norm(g_host) * np.linalg.norm(g_sub) + 1e-12))
    Yoh = np.zeros((args.batch, ro.V)); Yoh[np.arange(args.batch), Yb] = 1.0
    shared_label_term = (Yoh.T @ feats) / args.batch      # the "-onehot" component SHARED by g_host and g_sub
    frac_shared_in_ghost = float(np.linalg.norm(shared_label_term) / (np.linalg.norm(g_host) + 1e-12))
    frac_shared_in_gsub = float(np.linalg.norm(shared_label_term) / (np.linalg.norm(g_sub) + 1e-12))
    print(f"[seed {seed}] RECONCILE: cos(g_sub,g_host)@headw={cos_grad_headw:.4f} (gradalign reported ~0.975) | "
          f"||shared_label_term||/||g_host||={frac_shared_in_ghost:.3f} "
          f"||shared_label_term||/||g_sub||={frac_shared_in_gsub:.3f}", flush=True)

    print(f"[seed {seed}] gain: A(rand,0.12,norm={resA['probe_norm']})={resA['gain']} corr={resA['corr']}/"
          f"{resA['corr_row_centered']}rc | B(rand@||hw||={resB['probe_norm']})={resB['gain']} "
          f"corr={resB['corr']}/{resB['corr_row_centered']}rc | C(rand@cap={resC['probe_norm']})={resC['gain']} "
          f"corr={resC['corr']}/{resC['corr_row_centered']}rc | D(head_w,norm={resD['probe_norm']})={resD['gain']} "
          f"corr={resD['corr']}/{resD['corr_row_centered']}rc | E(hw-dir@A-norm={resE['probe_norm']})={resE['gain']} "
          f"corr={resE['corr']}/{resE['corr_row_centered']}rc", flush=True)
    print(f"[seed {seed}] DIRECT: ||margin_sub(head_w)/gain_A|| / ||host_ideal(head_w)|| = {ratio_direct:.4f} "
          f"({'UNDER-READ' if ratio_direct < 0.9 else ('OVER-READ' if ratio_direct > 1.1 else 'MATCHED')})",
          flush=True)

    # structure-vs-magnitude decomposition (the falsifiable claims):
    structure_effect_at_hw_norm = round(resD["gain"] / max(1e-9, resB["gain"]), 4)   # D vs B, SAME norm
    structure_effect_at_a_norm = round(resE["gain"] / max(1e-9, resA["gain"]), 4)    # E vs A, SAME norm
    magnitude_effect_random = round(resC["gain"] / max(1e-9, resA["gain"]), 4)       # C vs A, random dir, diff norm

    lever(f"gain_structure_effect_at_hwnorm_seed{seed}", before=1.0, after=structure_effect_at_hw_norm,
          required=False, continuous=structure_effect_at_hw_norm - 1.0)
    lever(f"gain_direct_underread_ratio_seed{seed}", before=1.0, after=ratio_direct, required=False,
          continuous=1.0 - ratio_direct)

    out = {
        "seed": seed, "V": ro.V, "D": ro.D, "head_w_norm": round(hw_norm, 3), "w_target": args.w_target,
        "batch": args.batch, "sub_read_window": args.sub_read_window,
        "probe_A_random_0.12": resA, "probe_B_random_at_hwnorm": resB, "probe_C_random_at_cap": resC,
        "probe_D_headw_itself": resD, "probe_E_headwdir_at_Anorm": resE,
        "headw_direction_scale_sweep": scale_sweep,
        "direct_underread_ratio_headw_via_gainA": round(ratio_direct, 4),
        "structure_effect_at_hwnorm_D_over_B": structure_effect_at_hw_norm,
        "structure_effect_at_Anorm_E_over_A": structure_effect_at_a_norm,
        "magnitude_effect_random_C_over_A": magnitude_effect_random,
        "seed_hash_check": seed_hash_check,
        "verdict_structure_compression": bool(structure_effect_at_hw_norm < 0.85 or structure_effect_at_a_norm < 0.85),
        "verdict_magnitude_compression_random_dir": bool(magnitude_effect_random < 0.85),
        "verdict_direct_underread": bool(ratio_direct < 0.9),
        # ROW-CENTERED reconciliation: does within-row (across-vocab) correlation recover for head_w once the
        # between-row (per-position) baseline -- invisible to softmax -- is removed?
        "verdict_row_centered_recovers_headw": bool(resD["corr_row_centered"] > 0.5),
        "corr_row_centered_summary": {"A_random": resA["corr_row_centered"], "D_headw": resD["corr_row_centered"],
                                      "scale_sweep_rc": [g["corr_row_centered"] for g in scale_sweep]},
        "order_control_A_repeat": resA_repeat,
        "verdict_order_confounded": bool(resA_repeat["corr"] < 0.7),
        "reconcile_cos_grad_sub_host_at_headw": round(cos_grad_headw, 4),
        "reconcile_shared_label_term_frac_of_ghost": round(frac_shared_in_ghost, 4),
        "reconcile_shared_label_term_frac_of_gsub": round(frac_shared_in_gsub, 4),
    }
    del s_batch
    return out


def run_short_train_probe(seed, args):
    """Reduced-scale training run comparing FIXED gain (current) vs ADAPTIVE gain (re-measured periodically against
    the CURRENT W_hat, self-referential, no target/label leak) -- does adaptive gain keep ||W|| near ~24 instead of
    running to the w_target cap, over a SHORT budget (not the full 1660-step production run)?"""
    ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
    ro = WKVReadout(ckpt)
    ev_ids, _ = _load_eval(ro, args.corpus, args.n_sentences, seed, args.n_sentences)
    usable = [ids for ids in ev_ids if len(ids) >= args.warmup + 2]
    cut = int(0.8 * len(usable))
    train_ids, eval_ids = usable[:cut], usable[cut:]
    H, Y, _ = _positions(ro, train_ids, args.warmup, args.n_train_pos)
    He, Ye, PFe = _positions(ro, eval_ids, args.warmup, args.n_eval_pos)
    hw = ro.head_w.astype(np.float64); head_b = ro.head_b.astype(np.float64)
    unk = ro.unk_idx

    s_batch = BatchedSubstrateReadout(ro, seed, args.batch, hid_pop=args.sub_hid_pop, pop=args.sub_pop,
                                      ou_std=args.ou_std, read_window=args.sub_read_window, hid_gain=args.hid_gain,
                                      ratio=args.ratio, settle_frac=args.settle_frac, n_bias=args.n_bias,
                                      bias_drive_pA=args.bias_drive_pA)

    def calibrate(W_probe_source_feats):
        rng2 = np.random.default_rng(seed * 7 + 3)
        Wp = 0.12 * rng2.standard_normal((ro.V, ro.D))
        res, _, _ = _measure_gain(s_batch, Wp, W_probe_source_feats)
        return res["gain"]

    def calibrate_adaptive(W_current, feats_batch):
        """Self-referential AGC: re-measure gain using the CURRENT W_hat as its own probe (no head_w/labels)."""
        res, _, _ = _measure_gain(s_batch, W_current, feats_batch)
        g = res["gain"]
        return g if abs(g) > 1e-6 else None

    def train(mode, regain_every):
        rng = np.random.default_rng(seed * 991 + 7)
        W = 0.01 * rng.standard_normal((ro.V, ro.D))
        gain = calibrate(H[:args.batch])
        idx = np.arange(len(H))
        n_full = (len(idx) // args.batch) * args.batch
        n_grad = 0
        traj = []
        for ep in range(args.epochs):
            rng.shuffle(idx)
            for start in range(0, n_full, args.batch):
                bi = idx[start:start + args.batch]
                Hb = H[bi]
                if mode == "adaptive" and regain_every > 0 and n_grad > 0 and n_grad % regain_every == 0:
                    g_new = calibrate_adaptive(W, Hb)
                    if g_new is not None:
                        gain = g_new
                s_batch.set_weights(W)
                margin_sub = s_batch.batch_margin(Hb, silence_bias=True)
                logits = _sub_logits(margin_sub, gain, head_b, unk)
                P = _softmax_rows(logits)
                P[np.arange(args.batch), Y[bi]] -= 1.0
                W = W - args.lr * (P.T @ Hb) / args.batch - args.weight_decay * W
                if args.w_target > 0:
                    nrm = float(np.linalg.norm(W))
                    if nrm > args.w_target:
                        W *= args.w_target / nrm
                n_grad += 1
            wc = _wcos(W, hw)
            nrm = float(np.linalg.norm(W))
            traj.append({"epoch": ep + 1, "n_grad": n_grad, "w_norm": round(nrm, 3), "weight_cosine": wc,
                        "gain": round(gain, 5)})
            print(f"[{mode} seed {seed} ep {ep + 1}/{args.epochs}] ||W||={nrm:.2f} wcos={wc} gain={gain:.4g} "
                  f"n_grad={n_grad}", flush=True)
        return W, traj, n_grad

    t0 = time.time()
    W_fixed, traj_fixed, ng1 = train("fixed", 0)
    t1 = time.time()
    W_adapt, traj_adapt, ng2 = train("adaptive", args.regain_every)
    t2 = time.time()

    from research.runners._wkv_mouth_readout_eprop_learn_derisk import _eval_hostlinear
    hl_fixed = _eval_hostlinear(ro, W_fixed, He, Ye, PFe)
    hl_adapt = _eval_hostlinear(ro, W_adapt, He, Ye, PFe)

    out = {
        "seed": seed, "epochs": args.epochs, "n_train_pos": len(H), "n_eval_pos": len(He),
        "regain_every": args.regain_every, "w_target": args.w_target,
        "fixed": {"w_norm_final": round(float(np.linalg.norm(W_fixed)), 3), "weight_cosine": _wcos(W_fixed, hw),
                  "hostlinear_recov": round(hl_fixed["recov_argmax"], 4), "trajectory": traj_fixed,
                  "secs": round(t1 - t0, 1)},
        "adaptive": {"w_norm_final": round(float(np.linalg.norm(W_adapt)), 3), "weight_cosine": _wcos(W_adapt, hw),
                     "hostlinear_recov": round(hl_adapt["recov_argmax"], 4), "trajectory": traj_adapt,
                     "secs": round(t2 - t1, 1)},
        "head_w_norm": round(float(np.linalg.norm(hw)), 3),
    }
    lever(f"adaptive_gain_wnorm_seed{seed}", before=out["fixed"]["w_norm_final"],
          after=out["adaptive"]["w_norm_final"], required=False,
          continuous=out["fixed"]["w_norm_final"] - out["adaptive"]["w_norm_final"])
    lever(f"adaptive_gain_wcos_seed{seed}", before=out["fixed"]["weight_cosine"],
          after=out["adaptive"]["weight_cosine"], required=False,
          continuous=out["adaptive"]["weight_cosine"] - out["fixed"]["weight_cosine"])
    del s_batch
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=8)           # memory-safe default (shared machine, 2026-08-27 OOM)
    ap.add_argument("--w-target", type=float, default=40.0)
    ap.add_argument("--sub-hid-pop", type=int, default=4)
    ap.add_argument("--sub-pop", type=int, default=1)
    ap.add_argument("--sub-read-window", type=int, default=64)  # memory-safe default (matches known-safe reference)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--short-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=8e-4)
    ap.add_argument("--n-train-pos", type=int, default=1200)
    ap.add_argument("--n-eval-pos", type=int, default=400)
    ap.add_argument("--regain-every", type=int, default=20)
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_softmax_confidence/diag.json")
    args = ap.parse_args()

    assert_backend(os.environ.get("SIM_BACKEND", "numpy"),
                   note="(softmax-confidence diagnosis is GPU-bound; numpy only for tiny smoke)")

    t0 = time.time()
    diag = run_diagnosis(args.seed, args)
    train = run_short_train_probe(args.seed, args) if args.short_train else None
    out = {"diagnosis": _native(diag), "short_train": _native(train), "argv": sys.argv,
           "backend": os.environ.get("SIM_BACKEND", "numpy"), "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"[done] -> {args.json} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
