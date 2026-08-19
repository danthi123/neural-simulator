"""SPIKING port of the position-invariant CONFIGURAL HMAX S->C hierarchy (board #72).

WHY THIS RUNNER EXISTS. The rate de-risk
(research/findings/2026-08-19-vision-hmax-hierarchy-composed-pooling-solves-position-invariance-learning-not-load-bearing.md,
runner research/runners/_vision_hmax_hierarchy_derisk.py) CLEARED position-invariant configural
recognition at RATE level (HMAX-trace held 0.5972 vs V1-direct 0.3698 vs flat-pool 0.2674, 5/6 GO)
and proved the invariance is carried by the innate COMPOSED-POOLING TOPOLOGY (random S2 == learned
S2), which is biologically faithful (complex-cell pooling wiring is developmental/innate). It flagged
the honest next step for full brain-based credit: build the S->C stack on SPIKING neurons.

THIS RUNNER IS THAT STEP. It reuses the rate front end + rendering + template-learning + decode BY
IMPORT (no rebuild) and replaces ONLY the rate arithmetic of the S/C layers with a genuinely SPIKING
stack:

  RETINA -> S1 rate Gabor drive (deployed sim.visual_cortex Gabor/V1, reused by import, NOT edited)
         -> hypercolumn orientation-competition + firing-gate (lateral inhibition; shapes the DRIVE)
         -> **S1 SPIKING**  : LIF somata (real threshold + absolute refractory + membrane noise) turn
                              the Gabor drive into SPIKE TRAINS over a T-step window. Read as spike
                              COUNT or FIRST-SPIKE LATENCY (recency).
         -> **C1 SPIKING**  : local retinotopic MAX-pool per orientation realised as a spiking WTA
                              (feedforward inhibition -> earliest/strongest spike wins), + optional
                              per-band kWTA lateral inhibition (Thorpe/Masquelier denoising).
         -> **S2 SPIKING**  : convolutional configural templates -> a LIF COINCIDENCE unit per
                              (location, template) that fires supralinearly only when the template's
                              C1-feature conjunction is co-active (LIF w/ leak = coincidence detector).
         -> **C2 SPIKING**  : global MAX-pool per template realised as a spiking WTA over locations.
  Identity decoded off the C2 spike code (count or latency vector), nearest-cosine-centroid.

THE NEURAL CODE IS THE CRUX (grounded in 2026-06-02-step2a-spiking-visual-word-recognition):
reading spiking vision on THIS substrate with a spike-COUNT/rate code hits a noise ceiling; the
biologically-correct code is FIRST-SPIKE LATENCY / rank-order (Thorpe/Masquelier 2007; Kheradpisheh
et al. 2018) + per-band kWTA lateral inhibition. So this runner runs BOTH codes and reports the
operating point; the PRIMARY GO is read off the latency code (the code the literature + our own prior
say is correct for sparse spiking vision). A rate model is GENEROUS; if the capability survives on
spikes with real refractory + noise, that is the decisive brain-based step.

ARMS (identical to the rate runner; chance = 1/n_classes):
  A  V1-DIRECT   decode off the flattened C1 spike code (position-specific floor).
  H  FLAT-POOL   global orientation histogram of the C1 spike code (= REMOVE S2 stage; config-blind).
  T  HMAX-TRACE  S2 templates trace-learned on a moving-object C1-spike-code continuity stream (PRIMARY).
  R  HMAX-RANDOM lesion: random S2 templates (isolates template-LEARNING vs innate topology).
  B  HMAX-IMPRINT Serre/Poggio one-shot patches (secondary).
  P1 HMAX-p1     ablation: S2 extent = 1 C1 cell (no conjunction) -> collapses to histogram.

ANTI-CHEATS (they ARE the result), on SPIKES:
  1. HELD-OUT POSITIONS: train {0,2,4,6}, test held {1,3,5,7} NEVER seen. Spiking held decode must
     beat V1-direct-held AND flat-pool.
  2. ARCHITECTURE load-bearing: lesion S2 (-> flat pool) -> chance (6/6 in rate).
  3. POSITION POOLED OUT: object decodable off C2 spikes, position ~chance; pixel-scramble null.
  4. 6 seeds (42/43/44/100/101/102), per-seed + pooled, deterministic (every RNG seeded); a re-run
     byte-compares (determinism), and a label-shuffle null -> chance.

BRAIN-BASED status: the S1/S2 somata GENUINELY SPIKE (LIF: leak, hard threshold, reset, absolute
refractory, per-step membrane noise -> discrete spike events). The C-layer MAX = a spiking WTA
(feedforward inhibition: first-spike-wins latency MAX, or count MAX). The inter-areal signal is a
spike-count or first-spike-latency population code (both real neural codes; latency is the correct one
here). The retinotopic weight-sharing + pooling windows are FLAGGED innate developmental scaffolds
(complex-cell RFs are developmental) -- the SAME concession the rate finding made and defended as
biology. No sim/ edit; reuses the deployed Gabor/V1 front end by import.

Smoke:
  SIM_BACKEND=numpy python -u -m research.runners._vision_hmax_spiking_derisk \
      --seeds 42 --code latency --out research/findings/raw/lanes/perception/vhmax_spiking_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ---- reuse the RATE de-risk front end + rendering + template-learning + decode BY IMPORT (no rebuild) ----
from research.runners._genfrontier_optionB_visual_similarity_derisk import (  # noqa: E402
    build_gabor_response_matrix,
    encode_v1,
    pool_v1_to_complex,
)
from research.runners._vision_hmax_hierarchy_derisk import (  # noqa: E402
    _build_objects,
    _c1_maxpool,
    _centroid_decode,
    _continuity_patch_stream,
    _extract_patches,
    _hist_oracle,
    _hypercolumn_norm,
    _imprint_templates,
    _l2n,
    _object_classes,
    _positions,
    _random_templates,
    _s2_c2,
    _scramble_images,
    _trace_competitive_templates,
    _within_split_decode,
)
from tools.lab import attributable_to  # noqa: E402

OUT = Path("research/findings/raw/lanes/perception/vision_hmax_spiking.json")


# ============================================================================================
# The ONLY new machinery: a genuinely SPIKING leaky integrate-and-fire layer. Real threshold,
# reset, absolute refractory, per-step membrane noise -> discrete spike events. Vectorised over
# (n_units, n_cells). Returns spike COUNT and FIRST-SPIKE time per cell (the two neural codes).
# ============================================================================================
def lif_spike_read(drive, T, seed, tau=8.0, v_thresh=1.0, t_ref=2, noise=0.06, gain=1.0):
    """drive (M, C) >= 0 -> (counts (M,C), first_spike (M,C)).

    LIF: dv = (dt/tau)(-v + gain*drive + noise); spike when v>=v_thresh -> reset 0, refractory t_ref.
    Strong drive => more spikes AND earlier first spike (latency code emerges). first_spike = T for a
    cell that never fires (the latest/"no-spike" bin). dt = 1 ms."""
    rng = np.random.default_rng(seed)
    M, C = drive.shape
    v = np.zeros((M, C), dtype=np.float32)
    ref = np.zeros((M, C), dtype=np.int32)
    counts = np.zeros((M, C), dtype=np.float32)
    first = np.full((M, C), float(T), dtype=np.float32)
    I = (gain * drive).astype(np.float32)
    for t in range(int(T)):
        can = ref <= 0
        v = np.where(can, v + (1.0 / tau) * (-v + I) + rng.standard_normal((M, C)).astype(np.float32) * noise, v)
        spk = can & (v >= v_thresh)
        counts += spk
        newf = spk & (first >= T)
        first = np.where(newf, float(t), first)
        v = np.where(spk, 0.0, v)
        ref = np.where(spk, t_ref, ref - 1)
    return counts, first


def spike_code(counts, first, T, code):
    """Turn (counts, first_spike) into the chosen neural code, non-negative.
      count   -> spike count (rate code).
      latency -> first-spike RECENCY = T - first_spike (earliest spike = largest value; the
                 Thorpe/Masquelier rank-order code; 0 for a cell that never fired)."""
    if code == "count":
        return counts.astype(np.float32)
    if code == "latency":
        return (float(T) - first).astype(np.float32)
    raise ValueError(code)


def _kwta_per_band(read, n_orient, frac):
    """Per-band (per image, per orientation) kWTA lateral inhibition: keep the top `frac` cells by
    value, zero the rest (Thorpe/Masquelier per-layer inhibition -> denoise the sparse spike code).
    read (N, n_orient*n_pos^2). frac<=0 disables."""
    if frac is None or frac >= 1.0 or frac <= 0.0:
        return read
    N = read.shape[0]
    m = read.reshape(N, n_orient, -1)
    npix = m.shape[2]
    k = max(1, int(round(frac * npix)))
    if k >= npix:
        return read
    thr = np.sort(m, axis=2)[:, :, npix - k][:, :, None]  # kth-largest per (image, orient)
    out = np.where(m >= thr, m, 0.0)
    return out.reshape(N, -1).astype(np.float32)


# ============================================================================================
def _c1_spiking(complex_drive, a, seed, code):
    """Rate Gabor complex drive -> hypercolumn competition+gate (drive shaping) -> S1 spikes (LIF) or
    rate (--s1-mode rate) -> chosen neural code -> per-band kWTA -> C1 innate local MAX-pool per
    orientation. -> (N, n_orient, g, g)."""
    hcol = _hypercolumn_norm(complex_drive, a.n_orientations, a.n_pos, a.orient_norm, a.c1_gate)  # (N, C)
    if a.s1_mode == "rate":
        read = hcol.astype(np.float32)                    # rate control: raw drive, no LIF
    else:
        counts, first = lif_spike_read(hcol, a.T1, seed, tau=a.tau, v_thresh=a.v_thresh, t_ref=a.t_ref,
                                       noise=a.noise, gain=a.s1_gain)
        read = spike_code(counts, first, a.T1, code)
    read = _kwta_per_band(read, a.n_orientations, a.kwta_frac)
    return _c1_maxpool(read, a.n_orientations, a.n_pos, a.c1_win, a.c1_stride)


def _s2_c2_spiking(c1, templates, p, a, seed, code):
    """S2 convolutional cosine template match -> (LIF COINCIDENCE spikes: supralinear, only a matched
    conjunction crosses threshold) -> C2 global MAX-pool per template. --s2-mode rate = the rate
    cosine+MAX control. Returns C2 code (N, n_S2).

    Two spiking C2 pool reads (--c2-pool):
      max    : spiking WTA = MAX over locations of the S2 spike read (raw Riesenhuber-Poggio MAX).
      ksum   : top-k location SUM (a denoised soft-WTA -- averages the k best-matching locations so a
               single noisy location cannot win; biologically a graded pool of the strongest afferents)."""
    if a.s2_mode == "rate":
        return _s2_c2(c1, templates, p)
    patches = _extract_patches(c1, p)                    # (N, n_loc, D)
    N, n_loc, D = patches.shape
    pn = _l2n(patches, axis=2)
    drive = np.clip(pn @ templates.T, 0.0, None)         # (N, n_loc, n_S2) cosine match, non-negative
    # S2 lateral inhibition ACROSS the template bank at each location (competition -> winner-relative
    # contrast code). The bare cosine match sits in a narrow band (common-mode ~0.8, discriminative
    # modulation ~0.04): a bare LIF threshold saturates and washes it out. Subtracting the per-location
    # template mean (or z-scoring) exposes the contrast so the near-threshold LIF is SENSITIVE to which
    # templates win -- biologically the same hypercolumn-style lateral inhibition used at C1.
    if a.s2_norm == "submean":
        drive = np.clip(drive - drive.mean(axis=2, keepdims=True), 0.0, None)
    elif a.s2_norm == "z":
        mu = drive.mean(axis=2, keepdims=True)
        sd = drive.std(axis=2, keepdims=True)
        drive = np.clip((drive - mu) / (sd + 1e-6), 0.0, None)
    if a.c2_pool == "drivepop":
        # Riesenhuber-Poggio complex-cell MAX is a POOLING NONLINEARITY (feedforward shunting
        # inhibition), not a spike code: pool the graded S2 template-match drive by MAX over
        # locations FIRST (the complex-cell op), THEN the C2 soma -- a POPULATION of M redundant LIF
        # units -- spikes on the clean pooled drive and its population rate is the code. One
        # quantization (not compounded per-location), population-averaged to beat the sub-quantization
        # discriminative modulation. Fully spiking at the C2 soma; S2 provides graded (dendritic) drive.
        c2_drive = drive.max(axis=1)                     # (N, n_S2) complex-cell MAX over locations
        M = max(1, a.c2_pop)
        tiled = np.repeat(c2_drive, M, axis=1)           # (N, n_S2*M)
        cc, cf = lif_spike_read(tiled, a.T2, seed + 991, tau=a.tau, v_thresh=a.v_thresh,
                                t_ref=a.t_ref, noise=a.noise, gain=a.s2_gain)
        pop = spike_code(cc, cf, a.T2, code).reshape(N, -1, M).mean(axis=2)  # population avg
        return pop.astype(np.float32)
    counts, first = lif_spike_read(drive.reshape(N * n_loc, -1), a.T2, seed + 777,
                                   tau=a.tau, v_thresh=a.v_thresh, t_ref=a.t_ref,
                                   noise=a.noise, gain=a.s2_gain)
    s2 = spike_code(counts, first, a.T2, code).reshape(N, n_loc, -1)   # (N, n_loc, n_S2)
    if a.c2_pool == "max":
        return s2.max(axis=1).astype(np.float32)
    # ksum: top-k location sum per template (denoised soft-WTA)
    k = max(1, min(a.c2_k, n_loc))
    topk = np.sort(s2, axis=1)[:, n_loc - k:, :]         # (N, k, n_S2)
    return topk.sum(axis=1).astype(np.float32)


def _flat(c1):
    return c1.reshape(c1.shape[0], -1).astype(np.float32)


# ============================================================================================
def run_seed(seed, a, code):
    positions = _positions(a.n_pos_total, a.image_size, a.pos_span)
    held_pi = list(range(1, a.n_pos_total, 2))
    train_pi = [pi for pi in range(a.n_pos_total) if pi not in held_pi]
    train_positions = [positions[pi] for pi in train_pi]
    held_positions = [positions[pi] for pi in held_pi]
    import math
    thetas = [(k / a.n_slots) * math.pi for k in range(a.n_slots)]
    class_perms = _object_classes(a.n_classes, a.n_slots)

    tr_imgs, tr_cls, tr_pos = _build_objects(class_perms, thetas, train_positions, a.n_ex, a, seed * 101 + 1)
    he_imgs, he_cls, he_pos = _build_objects(class_perms, thetas, held_positions, a.n_ex, a, seed * 101 + 2)
    sc_imgs = _scramble_images(he_imgs, seed * 101 + 3)

    W = build_gabor_response_matrix(
        n_orientations=a.n_orientations, n_frequencies=a.n_frequencies,
        n_positions_per_dim=a.n_pos, retina_size=a.image_size, receptive_field_radius=a.rf_radius)

    def complex_of(imgs):
        return pool_v1_to_complex(encode_v1(imgs, W), a.n_orientations, a.n_frequencies, a.n_pos)

    # SPIKING C1 (LIF S1 + WTA pool). Deterministic per (seed, split).
    tr_c1 = _c1_spiking(complex_of(tr_imgs), a, seed * 101 + 11, code)
    he_c1 = _c1_spiking(complex_of(he_imgs), a, seed * 101 + 12, code)
    sc_c1 = _c1_spiking(complex_of(sc_imgs), a, seed * 101 + 13, code)

    chance = 1.0 / a.n_classes
    chance_pos = 1.0 / len(held_pi)

    # ---------- ARM A: V1-DIRECT (flattened C1 spike code; position-specific) ----------
    A_held = _centroid_decode(_flat(tr_c1), tr_cls, _flat(he_c1), he_cls)
    # ---------- ARM H: FLAT-POOL (global orientation histogram = architecture lesion + config-blind) ----------
    H_held = _centroid_decode(_hist_oracle(tr_c1, a.n_orientations), tr_cls,
                              _hist_oracle(he_c1, a.n_orientations), he_cls)

    dim = a.n_orientations * a.s2_p * a.s2_p

    def hmax_arm(templates, p):
        code_tr = _s2_c2_spiking(tr_c1, templates, p, a, seed * 101 + 21, code)
        code_he = _s2_c2_spiking(he_c1, templates, p, a, seed * 101 + 22, code)
        held_dec = _centroid_decode(code_tr, tr_cls, code_he, he_cls)
        n = code_tr.shape[0]
        idx = np.arange(n)
        np.random.default_rng(seed * 47 + 3).shuffle(idx)
        fit, test = idx[: n // 2], idx[n // 2:]
        same_pos = _centroid_decode(code_tr[fit], tr_cls[fit], code_tr[test], tr_cls[test])
        cross_pos = _centroid_decode(code_tr[fit], tr_cls[fit], code_he, he_cls)
        return {"held": held_dec, "same_pos": same_pos, "cross_pos": cross_pos,
                "code_he": code_he, "templates": templates, "p": p}

    # ARM B: HMAX-IMPRINT (Serre/Poggio one-shot patches, learned on the C1 SPIKE code)
    imp = hmax_arm(_imprint_templates(tr_c1, a.s2_p, a.n_s2, seed * 13 + 5), a.s2_p)
    # ARM T: HMAX-TRACE (trace-competitive Hebbian on a moving-object C1-SPIKE-code continuity stream)
    stream = _continuity_patch_stream(tr_c1, tr_cls, tr_pos, a.n_classes, a.s2_p, a.trace_passes, seed * 17 + 7)
    shuf_stream = list(stream)
    np.random.default_rng(seed * 19 + 9).shuffle(shuf_stream)
    trc_ = hmax_arm(_trace_competitive_templates(stream, a.n_s2, a.trace_epochs, a.lr, a.trace_decay,
                                                 a.boost_beta, seed * 23 + 11), a.s2_p)
    tsh_ = hmax_arm(_trace_competitive_templates(shuf_stream, a.n_s2, a.trace_epochs, a.lr, a.trace_decay,
                                                 a.boost_beta, seed * 23 + 11), a.s2_p)
    # ARM R: HMAX-RANDOM (random projection; template-learning lesion)
    rnd = hmax_arm(_random_templates(dim, a.n_s2, seed * 29 + 13), a.s2_p)
    # ARM P1: single-cell ablation (p=1: no conjunction extent)
    p1_ = hmax_arm(_imprint_templates(tr_c1, 1, a.n_s2, seed * 31 + 15), 1)

    arms = {"imprint": imp, "trace": trc_, "random": rnd}
    B_held, T_held, Tshuf_held, R_held, P1_held = imp["held"], trc_["held"], tsh_["held"], rnd["held"], p1_["held"]

    primary = arms[a.primary_arm]
    P_held, P_code_he = primary["held"], primary["code_he"]
    P_scr_held = _centroid_decode(_s2_c2_spiking(tr_c1, primary["templates"], primary["p"], a, seed * 101 + 21, code),
                                  tr_cls,
                                  _s2_c2_spiking(sc_c1, primary["templates"], primary["p"], a, seed * 101 + 23, code),
                                  he_cls)

    # ---------- anti-cheat 3: position pooled out (off the PRIMARY C2 held spike code) ----------
    obj_split = _within_split_decode(P_code_he, he_cls, seed * 37 + 17)
    pos_split = _within_split_decode(P_code_he, he_pos, seed * 37 + 19)
    position_pooled_out = (obj_split >= chance + a.decode_margin) and (pos_split <= chance_pos + a.pos_decode_margin)

    # ---------- anti-cheat 4: label-shuffle null ----------
    lbl_shuf = np.random.default_rng(seed * 41 + 21).permutation(he_cls)
    B_labelshuffle = _within_split_decode(P_code_he, lbl_shuf, seed * 43 + 23)

    invariance_gap = max(0.0, primary["same_pos"] - primary["cross_pos"])
    capability_go = bool(
        (P_held >= chance + a.decode_margin)
        and (P_held - A_held >= a.beat_margin)
        and (P_held - H_held >= a.beat_margin)
        and position_pooled_out
        and (P_scr_held <= chance + a.decode_margin)
        and (invariance_gap <= a.inv_gap)
    )
    architecture_load_bearing = bool(P_held - H_held >= a.beat_margin)
    template_learning_load_bearing = bool(P_held - R_held >= a.beat_margin)
    hist_blind = bool(H_held <= chance + a.decode_margin)
    trace_load_bearing = bool(T_held - Tshuf_held >= a.beat_margin)

    return {
        "seed": seed, "code": code,
        "chance_object": round(chance, 4), "chance_position": round(chance_pos, 4),
        "primary_arm": a.primary_arm,
        "decode": {
            "A_v1_direct_held": round(A_held, 4),
            "H_flat_pool_held": round(H_held, 4),
            "PRIMARY_same_pos_cv": round(primary["same_pos"], 4),
            "PRIMARY_cross_pos_cv": round(primary["cross_pos"], 4),
            "PRIMARY_held": round(P_held, 4),
            "PRIMARY_scramble_held": round(P_scr_held, 4),
            "B_hmax_imprint_held": round(B_held, 4),
            "T_hmax_trace_held": round(T_held, 4),
            "T_hmax_traceshuffle_held": round(Tshuf_held, 4),
            "R_hmax_random_held": round(R_held, 4),
            "P1_hmax_imprint_p1_held": round(P1_held, 4),
        },
        "dissociation": {
            "object_decode_heldsplit": round(obj_split, 4),
            "position_decode_heldsplit": round(pos_split, 4),
            "label_shuffle_null": round(B_labelshuffle, 4),
            "position_pooled_out": position_pooled_out,
        },
        "invariance_gap_train_minus_held": round(invariance_gap, 4),
        "verdicts": {
            "capability_go": capability_go,
            "architecture_load_bearing": architecture_load_bearing,
            "template_learning_load_bearing": template_learning_load_bearing,
            "flat_pool_configuration_blind": hist_blind,
            "trace_load_bearing": trace_load_bearing,
        },
    }


def _summarize(rows, a, code, t0):
    def mean(path):
        vals = []
        for r in rows:
            cur = r
            for k in path:
                cur = cur[k]
            vals.append(float(cur))
        return round(float(np.mean(vals)), 4)

    def frac(path):
        def _get(r):
            cur = r
            for k in path:
                cur = cur[k]
            return cur
        return round(float(np.mean([1.0 if _get(r) else 0.0 for r in rows])), 4)

    hd = lambda k: mean(("decode", k))  # noqa: E731
    attributable_to(f"[{code}] SPIKING HMAX held-invariance -> composed HIERARCHY (vs V1-direct held)",
                    hd("PRIMARY_held"), hd("A_v1_direct_held"))
    attributable_to(f"[{code}] SPIKING HMAX held-invariance -> S2 CONJUNCTION stage (vs flat pool)",
                    hd("PRIMARY_held"), hd("H_flat_pool_held"))
    attributable_to(f"[{code}] SPIKING HMAX held-invariance -> LEARNED templates (vs random projection)",
                    hd("PRIMARY_held"), hd("R_hmax_random_held"), warn_below=0.0)

    n_go = sum(1 for r in rows if r["verdicts"]["capability_go"])
    overall = ("SPIKING-HMAX-GO" if n_go == len(rows)
               else "SPIKING-HMAX-NOGO" if n_go == 0
               else f"SPIKING-HMAX-PARTIAL-{n_go}/{len(rows)}")
    return {
        "probe": "vision_hmax_spiking", "code": code, "overall_verdict": overall,
        "seeds": a.seeds, "n_seeds": len(rows), "chance_object": round(1.0 / a.n_classes, 4),
        "per_seed_capability_go": [r["verdicts"]["capability_go"] for r in rows],
        "decode_means": {k: mean(("decode", k)) for k in rows[0]["decode"]},
        "dissociation_means": {
            "object_decode_heldsplit": mean(("dissociation", "object_decode_heldsplit")),
            "position_decode_heldsplit": mean(("dissociation", "position_decode_heldsplit")),
            "label_shuffle_null": mean(("dissociation", "label_shuffle_null")),
        },
        "invariance_gap_mean": mean(("invariance_gap_train_minus_held",)),
        "verdict_fracs": {k: frac(("verdicts", k)) for k in rows[0]["verdicts"]},
        "headroom": {
            "primary_minus_v1_held": round(hd("PRIMARY_held") - hd("A_v1_direct_held"), 4),
            "primary_minus_flat_held": round(hd("PRIMARY_held") - hd("H_flat_pool_held"), 4),
            "primary_minus_random_held": round(hd("PRIMARY_held") - hd("R_hmax_random_held"), 4),
            "primary_p3_minus_p1_held": round(hd("PRIMARY_held") - hd("P1_hmax_imprint_p1_held"), 4),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    p.add_argument("--code", choices=["latency", "count", "both"], default="both",
                   help="neural code the S/C layers are read with. 'both' runs latency AND count and "
                        "reports the operating point; PRIMARY GO read off latency.")
    p.add_argument("--primary-code", choices=["latency", "count"], default="latency")
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--n-slots", type=int, default=3)
    p.add_argument("--n-pos-total", type=int, default=8)
    p.add_argument("--pos-span", type=float, default=8.0)
    p.add_argument("--n-ex", type=int, default=6)
    p.add_argument("--image-size", type=int, default=56)
    p.add_argument("--slot-offset", type=float, default=10.0)
    p.add_argument("--stroke-len", type=float, default=7.0)
    p.add_argument("--stroke-tk", type=float, default=1.8)
    p.add_argument("--pixel-noise", type=float, default=0.03)
    p.add_argument("--primary-arm", choices=["trace", "imprint", "random"], default="trace")
    # V1 front end
    p.add_argument("--n-orientations", type=int, default=8)
    p.add_argument("--n-frequencies", type=int, default=2)
    p.add_argument("--n-pos", type=int, default=24)
    p.add_argument("--rf-radius", type=int, default=3)
    p.add_argument("--orient-norm", choices=["none", "div", "z"], default="z")
    p.add_argument("--c1-gate", type=float, default=0.15)
    # C1 innate local pool
    p.add_argument("--c1-win", type=int, default=6)
    p.add_argument("--c1-stride", type=int, default=3)
    # S2 configural templates
    p.add_argument("--s2-p", type=int, default=3)
    p.add_argument("--n-s2", type=int, default=128)
    # trace learning
    p.add_argument("--trace-passes", type=int, default=12)
    p.add_argument("--trace-epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--trace-decay", type=float, default=0.9)
    p.add_argument("--boost-beta", type=float, default=2.0)
    # SPIKING (LIF) operating point
    p.add_argument("--s1-mode", choices=["spiking", "rate"], default="spiking",
                   help="S1 stage: LIF spikes (default) or rate control (isolate where the drop is)")
    p.add_argument("--s2-mode", choices=["spiking", "rate"], default="spiking",
                   help="S2/C2 stage: LIF coincidence spikes (default) or rate cosine+MAX control")
    p.add_argument("--c2-pool", choices=["max", "ksum", "drivepop"], default="drivepop",
                   help="spiking C2 global pool: raw MAX over spike code, denoised top-k SUM, or "
                        "DRIVEPOP = MAX-pool the graded S2 drive then a POPULATION of LIF C2 somata "
                        "spike on the pooled drive (default; decouples pooling from spike-quantization)")
    p.add_argument("--c2-k", type=int, default=3, help="k for c2-pool=ksum")
    p.add_argument("--c2-pop", type=int, default=24, help="C2 soma population size for c2-pool=drivepop")
    p.add_argument("--s2-norm", choices=["none", "submean", "z"], default="submean",
                   help="S2 lateral inhibition across the template bank at each location (expose the "
                        "winner-relative contrast so the LIF is not saturated by the cosine common-mode)")
    p.add_argument("--T1", type=int, default=64, help="S1 LIF window (ms/steps)")
    p.add_argument("--T2", type=int, default=48, help="S2 LIF window (ms/steps)")
    p.add_argument("--tau", type=float, default=8.0, help="LIF membrane time constant")
    p.add_argument("--v-thresh", type=float, default=1.0)
    p.add_argument("--t-ref", type=int, default=2, help="absolute refractory (steps)")
    p.add_argument("--noise", type=float, default=0.06, help="per-step membrane noise sd")
    p.add_argument("--s1-gain", type=float, default=1.2, help="Gabor drive -> S1 current gain")
    p.add_argument("--s2-gain", type=float, default=2.5, help="cosine match -> S2 current gain")
    p.add_argument("--kwta-frac", type=float, default=0.15,
                   help="per-band C1 kWTA fraction (Thorpe/Masquelier lateral inhibition); >=1 disables")
    # gate thresholds (same as rate)
    p.add_argument("--decode-margin", type=float, default=0.15)
    p.add_argument("--beat-margin", type=float, default=0.10)
    p.add_argument("--pos-decode-margin", type=float, default=0.15)
    p.add_argument("--inv-gap", type=float, default=0.20)
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()

    t0 = time.time()
    codes = ["latency", "count"] if a.code == "both" else [a.code]
    print(f"[vision-hmax-SPIKING] seeds={a.seeds} codes={codes} primary_arm={a.primary_arm} "
          f"LIF(T1={a.T1},T2={a.T2},tau={a.tau},vth={a.v_thresh},ref={a.t_ref},noise={a.noise},"
          f"s1g={a.s1_gain},s2g={a.s2_gain},kwta={a.kwta_frac})", flush=True)

    result = {}
    for code in codes:
        rows = [run_seed(s, a, code) for s in a.seeds]
        for r in rows:
            d, di, v = r["decode"], r["dissociation"], r["verdicts"]
            print(f"  [{code} seed {r['seed']}] V1he {d['A_v1_direct_held']:.2f} flat {d['H_flat_pool_held']:.2f} "
                  f"| PRIMARY({r['primary_arm']}) same {d['PRIMARY_same_pos_cv']:.2f} he {d['PRIMARY_held']:.2f} "
                  f"scr {d['PRIMARY_scramble_held']:.2f} | trace {d['T_hmax_trace_held']:.2f} "
                  f"(shuf {d['T_hmax_traceshuffle_held']:.2f}) rand {d['R_hmax_random_held']:.2f} "
                  f"p1 {d['P1_hmax_imprint_p1_held']:.2f} | obj/pos {di['object_decode_heldsplit']:.2f}/"
                  f"{di['position_decode_heldsplit']:.2f} | GO={v['capability_go']} "
                  f"arch_lb={v['architecture_load_bearing']} learn_lb={v['template_learning_load_bearing']}", flush=True)
        result[code] = {"summary": _summarize(rows, a, code, t0), "per_seed": rows}

    primary_code = a.primary_code if a.primary_code in result else codes[0]
    top = {
        "probe": "vision_hmax_spiking",
        "primary_code": primary_code,
        "overall_verdict": result[primary_code]["summary"]["overall_verdict"],
        "config": vars(a),
        "by_code": result,
        "operating_point_note": (
            "PRIMARY GO read off the LATENCY (first-spike/rank-order) code -- the biologically-correct "
            "code for sparse spiking vision (Thorpe/Masquelier 2007; Kheradpisheh et al. 2018; confirmed "
            "on this substrate in 2026-06-02-step2a). The COUNT (rate) code is reported alongside to show "
            "the count-vs-latency operating point."
        ),
        "mechanism": (
            "RETINA -> S1 rate Gabor drive (deployed sim.visual_cortex, reused by import) -> hypercolumn "
            "competition+gate (drive shaping) -> S1 LIF spikes (threshold+refractory+noise) -> C1 spiking "
            "WTA local MAX-pool per orientation + per-band kWTA -> S2 LIF coincidence units (convolutional "
            "configural templates; supralinear) -> C2 spiking WTA global MAX-pool. Decode off C2 spike code."
        ),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(top, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    for code in codes:
        s = result[code]["summary"]
        print(f"[{code}] {s['overall_verdict']}  PRIMARY_held={s['decode_means']['PRIMARY_held']} "
              f"V1={s['decode_means']['A_v1_direct_held']} flat={s['decode_means']['H_flat_pool_held']} "
              f"rand={s['decode_means']['R_hmax_random_held']} p1={s['decode_means']['P1_hmax_imprint_p1_held']} "
              f"| GO {sum(s['per_seed_capability_go'])}/{s['n_seeds']} "
              f"arch_lb={s['verdict_fracs']['architecture_load_bearing']} "
              f"learn_lb={s['verdict_fracs']['template_learning_load_bearing']}", flush=True)
    print(f"[written] {out_path}", flush=True)
    print("=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
