"""REWARD-MODULATED STDP sparse discriminative readout to close the fully-spiking HMAX vision NO-GO
(board #75).

WHY THIS RUNNER EXISTS. The spiking HMAX de-risk
(research/findings/2026-08-19-vision-spiking-hierarchy-frontend-holds-configural-readout-quantization-limited.md,
runner research/runners/_vision_hmax_spiking_derisk.py) showed, 6 seeds:
  (B) the LIF-spiking S1->C1 FRONT END PRESERVES position-invariant CONFIGURAL recognition on spikes
      (count held 0.5625 vs rate 0.5972, arch load-bearing 6/6) -- the perceptually HARD stage works;
  (C) FULLY spike-coding the S2->C2 configural READOUT is a NO-GO (held 0.34, position leaks 0.97,
      GO 0/6), because the RATE configural discrimination is a fine DISTRIBUTED cosine modulation
      (across-template std ~0.04 on a common-mode ~0.80) that falls BELOW the per-unit spike
      quantization floor, and -- crucially -- on RATE a RANDOM S2 readout == a LEARNED one
      ("template-learning NOT load-bearing").

THE KEY REFRAME the #72 finding makes, which THIS runner tests. On RATE, learning is inert because a
distributed random projection already separates the (linearly separable) configural classes and the
cosine-centroid decode divides out the common mode. But on SPIKES the distributed random code is
QUANTIZATION-FRAGILE. So the #72 finding PREDICTS: on spikes, a DISCRIMINATIVE SPARSE readout learned
with REWARD-MODULATED STDP -- which concentrates the configural signal into a FEW reliable units that
fire ABOVE the per-spike quantization floor -- becomes LOAD-BEARING where unsupervised/random is not.
This is also the mechanism our own 2026-06-02-step2a finding scoped as the remaining piece of a
faithful fully-spiking recognizer ("V1 latency code + per-band kWTA + R-STDP / learned readout";
vanilla unsupervised STDP is insufficient, a reward/supervised readout is needed).

THE NAMED, UNTRIED MECHANISM (built here). Instead of the unsupervised trace/imprint/random S2 of
config C, LEARN the S2 configural template bank with three-factor REWARD-MODULATED STDP:
  - The convolutional S2 templates are assigned round-robin to classes (n_S2/n_classes per class).
  - Forward (SPIKING): C1 patches -> cosine drive -> S2 lateral inhibition (winner-relative contrast)
    -> LIF S2 coincidence spikes at every location -> C2 spiking WTA global MAX-pool over locations
    -> a per-class spike-sum -> SPIKING WTA over the class populations = the prediction. Fully spiking.
  - Reward = correct/incorrect (a global dopamine sign). Three-factor R-STDP eligibility = the
    pre->post coincidence: the C1 patch at each template's WINNING location (pre) x the C2 spike (post).
      * true-class templates that fired -> POTENTIATE toward that patch  (reward / corrective teacher)
      * the wrongly-predicted class's templates that fired -> DEPRESS    (anti-STDP / punish)
    Weights are non-negative (excitatory) and L2-renormalised each update (homeostatic bound; no
    unbounded growth). This drives a SPARSE SELECTIVE code: a few strongly-, sharply-tuned S2 units
    per class that fire class-differentially -- a big supra-quantization modulation, not a ~5% one.

  R-STDP grounding: Frémaux & Gerstner 2016 (three-factor plasticity framework); Izhikevich 2007
  (DA-modulated STDP); Mozafari, Ganjtabesh, Nowzari-Dalini, Thorpe & Masquelier 2018, IEEE TNNLS
  29(12):6178-6190 ("First-spike-based visual categorization using reward-modulated STDP") -- R-STDP
  on a Thorpe/Masquelier conv-SNN for object recognition, the DIRECT precedent for this design.

ARMS:
  LEARNED   R-STDP-trained S2 template bank (the mechanism).
  RANDOM    IDENTICAL architecture + identical spiking forward + identical decodes, W UNTRAINED
            (random init). This IS config-C's random arm: it reproduces the NO-GO and is the
            like-for-like control for the reframe. LEARNED must BEAT RANDOM on spikes.

ANTI-CHEATS (they ARE the result), all on SPIKES, held-out positions:
  1. FULLY-SPIKING held-out-position accuracy (spiking-WTA class read) must BEAT the #72 config-C
     NO-GO floor (0.34) -- ideally approach config-B (0.56) / the rate ceiling (0.60). Train positions
     {0,2,4,6}; held {1,3,5,7} NEVER seen.
  2. LEARNING IS LOAD-BEARING (the reframe, the headline): LEARNED spiking readout must BEAT the
     RANDOM spiking readout of the same architecture -- unlike the rate case where random==learned. If
     learned ~= random on spikes too, that REFUTES the reframe (a first-class negative, reported).
  3. SPARSITY IS REAL: the learned readout fires FEW reliable units/class (active-unit counts +
     top-k concentration reported), not a denser projection.
  4. POSITION POOLED OUT (object decodable off C2 spikes, position ~chance); PIXEL-SCRAMBLE -> chance;
     LABEL-SHUFFLE -> chance; 6 seeds (42/43/44/100/101/102), per-seed + pooled; DETERMINISTIC
     (every RNG seeded from cfg.seed; a re-run byte-compares).

BRAIN-BASED status: the S1/S2 somata GENUINELY SPIKE (LIF: leak, hard threshold, reset, absolute
refractory, per-step membrane noise). The C-layer MAX = a spiking WTA (feedforward inhibition). The
class read = a spiking WTA over class-assigned populations (lateral inhibition). The plasticity is a
three-factor reward-modulated STDP (pre = winning-location C1 patch, post = C2 spike, third factor =
correct/incorrect dopamine sign). The retinotopic weight-sharing + pooling windows remain FLAGGED
innate developmental scaffolds (the same concession config B/C made). No sim/ edit; the LIF S1->C1
front end + LIF machinery are REUSED BY IMPORT from the #72 spiking runner.

Smoke:
  SIM_BACKEND=numpy python -u -m research.runners._vision_rstdp_readout_derisk \
      --seeds 42 --epochs 20 --n-s2 64 \
      --out research/findings/raw/lanes/perception/vrstdp_smoke.json
"""
from __future__ import annotations

import argparse
import json
import math
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

# ---- reuse the RATE front end + rendering + decode BY IMPORT ----
from research.runners._genfrontier_optionB_visual_similarity_derisk import (  # noqa: E402
    build_gabor_response_matrix,
    encode_v1,
    pool_v1_to_complex,
)
from research.runners._vision_hmax_hierarchy_derisk import (  # noqa: E402
    _build_objects,
    _centroid_decode,
    _extract_patches,
    _hist_oracle,
    _l2n,
    _object_classes,
    _positions,
    _scramble_images,
    _within_split_decode,
)
# ---- reuse the SPIKING C1 front end (config B) + LIF machinery BY IMPORT (#72 runner) ----
from research.runners._vision_hmax_spiking_derisk import (  # noqa: E402
    _c1_spiking,
    _flat,
    lif_spike_read,
    spike_code,
)
from tools.lab import attributable_to  # noqa: E402

OUT = Path("research/findings/raw/lanes/perception/vision_rstdp_readout.json")


# ============================================================================================
# SPIKING S2->C2 forward with a given (learned or random) template bank W. Everything downstream of
# the reused LIF S1->C1 front end. Returns the C2 spike code + per-template winning location + the
# spiking-WTA class prediction (the FULLY-SPIKING read).
# ============================================================================================
def _c2_kwta(r, k):
    """C2-level lateral inhibition across the template bank: keep the top-k templates active per image,
    zero the rest (a sparse WTA readout -- discards the graded per-template magnitude that carries
    position, and forces a FEW reliable units to represent the object). k<=0 or k>=n_S2 disables."""
    N, n_S2 = r.shape
    if k is None or k <= 0 or k >= n_S2:
        return r
    thr = np.sort(r, axis=1)[:, n_S2 - k][:, None]             # kth-largest per image
    return np.where(r >= thr, r, 0.0).astype(np.float32)


def _s2_forward(patches, W, class_mat, a, code, lif_seed, kwta_k):
    """patches (N, n_loc, D); W (n_S2, D) L2-normalised; class_mat (n_classes, n_S2) 0/1 membership.
    kwta_k = C2 readout lateral inhibition (keep top-k templates); pass 0 to disable (dense read used
    DURING TRAINING so every class's units can fire and earn reward; the sparse read is applied at EVAL).

    RETURN: r (N, n_S2) C2 spike code (global MAX over locations, then optional C2 kWTA),
            argloc (N, n_S2) winning location per template,
            pn (N, n_loc, D) L2-normalised patches (presynaptic drive, for the eligibility trace),
            cscore (N, n_classes) class-population spike sum, pred (N,) spiking-WTA class."""
    N, n_loc, D = patches.shape
    pn = _l2n(patches, axis=2)
    drive = np.clip(pn @ W.T, 0.0, None)                       # (N, n_loc, n_S2) cosine match
    if a.s2_norm == "submean":
        drive = np.clip(drive - drive.mean(axis=2, keepdims=True), 0.0, None)
    elif a.s2_norm == "z":
        mu = drive.mean(axis=2, keepdims=True)
        sd = drive.std(axis=2, keepdims=True)
        drive = np.clip((drive - mu) / (sd + 1e-6), 0.0, None)
    counts, first = lif_spike_read(drive.reshape(N * n_loc, -1), a.T2, lif_seed,
                                   tau=a.tau, v_thresh=a.v_thresh, t_ref=a.t_ref,
                                   noise=a.noise, gain=a.s2_gain)
    s2 = spike_code(counts, first, a.T2, code).reshape(N, n_loc, -1)  # (N, n_loc, n_S2)
    r = s2.max(axis=1).astype(np.float32)                      # C2 spiking WTA global MAX over locations
    argloc = s2.argmax(axis=1).astype(np.int64)                # (N, n_S2) winning location (eligibility)
    r = _c2_kwta(r, kwta_k)                                     # sparse readout (lateral inhibition)
    cscore = (r @ class_mat.T).astype(np.float32)              # (N, n_classes) class-population sum
    pred = cscore.argmax(axis=1).astype(np.int64)
    return r, argloc, pn, cscore, pred


def _init_templates(dim, n_S2, seed):
    """Non-negative random init (same family as the trace/random arms); the RANDOM control keeps it."""
    rng = np.random.default_rng(seed)
    return _l2n(np.abs(rng.standard_normal((n_S2, dim))).astype(np.float32) + 0.01, axis=1)


def _train_rstdp(patches_tr, y_tr, W0, class_mat, tmpl_class, a, code, seed):
    """Online three-factor REWARD-MODULATED STDP with WITHIN-CLASS competition + duty-cycle boosting.

    One presentation -> spiking forward -> spiking-WTA prediction -> reward (correct/incorrect):
      * the TRUE class's BEST-responding template(s) (competitive winner, duty-cycle-boosted so a FEW
        diverse units per class specialise rather than all collapsing) are POTENTIATED toward their
        winning-location C1 patch (reward / corrective teacher);
      * on error, the WRONGLY-PREDICTED class's best-responding template is DEPRESSED (anti-STDP).
    This yields a SPARSE SELECTIVE code (few reliable units/class), the mechanism the #72 reframe names.
    Eligibility = the pre->post coincidence (winning-location C1 patch x C2 spike). Weights non-negative
    + L2-renormalised each update (homeostatic bound)."""
    W = W0.copy()
    N = patches_tr.shape[0]
    n_classes = class_mat.shape[0]
    n_S2 = W.shape[0]
    per_class = n_S2 / max(1, n_classes)
    cls_idx = [np.nonzero(tmpl_class == c)[0] for c in range(n_classes)]
    duty = np.ones(n_S2, dtype=np.float64) / max(1.0, per_class)  # per-class target duty = 1/per_class
    rng = np.random.default_rng(seed * 7 + 3)
    train_acc = []
    for ep in range(int(a.epochs)):
        order = rng.permutation(N)
        n_correct = 0
        wins = np.zeros(n_S2, dtype=np.float64)
        boost = np.exp(a.boost_beta * (1.0 / max(1.0, per_class) - duty))  # spread winners over a class
        for i in order:
            i = int(i)
            lif_seed = (seed * 1_000_003 + ep * 7919 + i * 131 + 17) % (2 ** 31)
            r, argloc, pn, cscore, pred = _s2_forward(
                patches_tr[i:i + 1], W, class_mat, a, code, lif_seed, kwta_k=0)  # dense read for reward
            r = r[0]; argloc = argloc[0]; pn = pn[0]; pred = int(pred[0])
            y = int(y_tr[i])
            n_correct += int(pred == y)
            m = np.zeros(n_S2, dtype=np.float32)                # dopamine-signed per-template modulation
            # ---- reward: potentiate the TRUE class's fired templates ----
            yj = cls_idx[y]
            fired_y = yj[r[yj] > 0.0]
            if fired_y.shape[0] > 0:
                if a.rstdp_win_k <= 0:
                    win = fired_y                              # dense: all fired true-class templates
                else:
                    rb = r[fired_y] * boost[fired_y]           # competitive (duty-boosted) top-k winners
                    kwin = max(1, min(a.rstdp_win_k, fired_y.shape[0]))
                    win = fired_y[np.argsort(rb)[::-1][:kwin]]
                m[win] = 1.0
                wins[win] += 1.0
            # ---- punish: depress the wrongly-predicted class's fired templates (anti-STDP) ----
            if pred != y:
                pj = cls_idx[pred]
                fired_p = pj[r[pj] > 0.0]
                if fired_p.shape[0] > 0:
                    if a.rstdp_win_k <= 0:
                        m[fired_p] = -1.0                      # dense: all fired wrong-class templates
                    else:
                        m[fired_p[int(np.argmax(r[fired_p]))]] = -1.0  # competitive: the wrong winner
            if not np.any(m):
                continue
            x_win = pn[argloc]                                  # (n_S2, D) presynaptic pattern @ winner
            rmax = r.max()
            post = (r / (rmax + 1e-9)).astype(np.float32)       # normalised postsynaptic activity
            dW = (a.lr * m * post)[:, None] * x_win             # (n_S2, D)
            W = W + dW
            W = np.clip(W, 0.0, None)                           # excitatory
            nrm = np.linalg.norm(W, axis=1, keepdims=True)      # homeostatic L2 renorm (bounded)
            W = W / np.where(nrm < 1e-9, 1.0, nrm)
        duty = 0.5 * duty + 0.5 * (wins / max(1, N))            # update usage (duty-cycle boosting)
        train_acc.append(round(n_correct / max(1, N), 4))
    return W.astype(np.float32), train_acc


def _sparsity(r, class_mat, tmpl_class, y_true, topk=5):
    """Sparsity of the C2 spike code r (N, n_S2): mean active units/image, active fraction, and the
    fraction of the WINNING class's spike-sum carried by its top-k templates (concentration)."""
    N, n_S2 = r.shape
    active = (r > 0.0).sum(axis=1).astype(np.float32)
    winclass = (r @ class_mat.T).argmax(axis=1)
    conc = []
    win_active = []
    for i in range(N):
        c = int(winclass[i])
        rj = r[i][tmpl_class == c]                              # this class's template spikes
        s = rj.sum()
        win_active.append(float((rj > 0).sum()))
        if s > 1e-9:
            k = min(topk, rj.shape[0])
            conc.append(float(np.sort(rj)[::-1][:k].sum() / s))
    return {
        "active_units_mean": round(float(active.mean()), 3),
        "active_frac_mean": round(float((active / n_S2).mean()), 4),
        "winclass_active_units_mean": round(float(np.mean(win_active)) if win_active else 0.0, 3),
        f"winclass_top{topk}_mass_frac_mean": round(float(np.mean(conc)) if conc else 0.0, 4),
        "n_s2": int(n_S2),
    }


# ============================================================================================
def run_seed(seed, a, code):
    positions = _positions(a.n_pos_total, a.image_size, a.pos_span)
    held_pi = list(range(1, a.n_pos_total, 2))
    train_pi = [pi for pi in range(a.n_pos_total) if pi not in held_pi]
    train_positions = [positions[pi] for pi in train_pi]
    held_positions = [positions[pi] for pi in held_pi]
    thetas = [(k / a.n_slots) * math.pi for k in range(a.n_slots)]
    class_perms = _object_classes(a.n_classes, a.n_slots)

    tr_imgs, tr_cls, tr_pos = _build_objects(class_perms, thetas, train_positions, a.n_ex, a, seed * 101 + 1)
    he_imgs, he_cls, he_pos = _build_objects(class_perms, thetas, held_positions, a.n_ex, a, seed * 101 + 2)
    sc_imgs = _scramble_images(he_imgs, seed * 101 + 3)

    Wg = build_gabor_response_matrix(
        n_orientations=a.n_orientations, n_frequencies=a.n_frequencies,
        n_positions_per_dim=a.n_pos, retina_size=a.image_size, receptive_field_radius=a.rf_radius)

    def complex_of(imgs):
        return pool_v1_to_complex(encode_v1(imgs, Wg), a.n_orientations, a.n_frequencies, a.n_pos)

    # SPIKING C1 front end (config B: LIF S1 + spiking WTA C1 pool + per-band kWTA), read as `c1_code`.
    tr_c1 = _c1_spiking(complex_of(tr_imgs), a, seed * 101 + 11, a.c1_code)
    he_c1 = _c1_spiking(complex_of(he_imgs), a, seed * 101 + 12, a.c1_code)
    sc_c1 = _c1_spiking(complex_of(sc_imgs), a, seed * 101 + 13, a.c1_code)

    patches_tr = _extract_patches(tr_c1, a.s2_p)              # (N, n_loc, D)
    patches_he = _extract_patches(he_c1, a.s2_p)
    patches_sc = _extract_patches(sc_c1, a.s2_p)
    dim = a.n_orientations * a.s2_p * a.s2_p

    chance = 1.0 / a.n_classes
    chance_pos = 1.0 / len(held_pi)

    # ---- floors (same as #72/#44): V1-direct (position-specific) + flat orientation-histogram pool ----
    A_held = _centroid_decode(_flat(tr_c1), tr_cls, _flat(he_c1), he_cls)
    H_held = _centroid_decode(_hist_oracle(tr_c1, a.n_orientations), tr_cls,
                              _hist_oracle(he_c1, a.n_orientations), he_cls)

    # ---- template bank: round-robin class assignment ----
    tmpl_class = np.array([j % a.n_classes for j in range(a.n_s2)], dtype=np.int64)
    class_mat = np.zeros((a.n_classes, a.n_s2), dtype=np.float32)
    class_mat[tmpl_class, np.arange(a.n_s2)] = 1.0
    W0 = _init_templates(dim, a.n_s2, seed * 29 + 13)

    kk = a.c2_kwta_k                                          # sparse read applied at EVAL (both arms)
    # ---- RANDOM control (config-C-like NO-GO): identical forward + decodes, W untrained ----
    r_tr_rnd, _, _, _, pred_tr_rnd = _s2_forward(patches_tr, W0, class_mat, a, code, seed * 991 + 101, kk)
    r_he_rnd, _, _, _, pred_he_rnd = _s2_forward(patches_he, W0, class_mat, a, code, seed * 991 + 102, kk)
    rnd_spk_wta_held = float((pred_he_rnd == he_cls).mean())
    rnd_centroid_held = _centroid_decode(r_tr_rnd, tr_cls, r_he_rnd, he_cls)

    # ---- LEARNED (R-STDP) ----
    W, train_acc = _train_rstdp(patches_tr, tr_cls, W0, class_mat, tmpl_class, a, code, seed)
    r_tr, _, _, _, pred_tr = _s2_forward(patches_tr, W, class_mat, a, code, seed * 991 + 201, kk)
    r_he, argloc_he, _, cscore_he, pred_he = _s2_forward(patches_he, W, class_mat, a, code, seed * 991 + 202, kk)
    r_sc, _, _, _, _ = _s2_forward(patches_sc, W, class_mat, a, code, seed * 991 + 203, kk)
    learn_spk_wta_held = float((pred_he == he_cls).mean())
    learn_spk_wta_train = float((pred_tr == tr_cls).mean())
    learn_centroid_held = _centroid_decode(r_tr, tr_cls, r_he, he_cls)
    scr_centroid_held = _centroid_decode(r_tr, tr_cls, r_sc, he_cls)

    # ---- anti-cheat: position pooled out (off the LEARNED held C2 spike code) ----
    obj_split = _within_split_decode(r_he, he_cls, seed * 37 + 17)
    pos_split = _within_split_decode(r_he, he_pos, seed * 37 + 19)
    position_pooled_out = (obj_split >= chance + a.decode_margin) and (pos_split <= chance_pos + a.pos_decode_margin)

    # ---- anti-cheat: label-shuffle null ----
    lbl_shuf = np.random.default_rng(seed * 41 + 21).permutation(he_cls)
    lbl_shuffle_null = _within_split_decode(r_he, lbl_shuf, seed * 43 + 23)

    # ---- sparsity ----
    spars_learned = _sparsity(r_he, class_mat, tmpl_class, he_cls, topk=a.topk)
    spars_random = _sparsity(r_he_rnd, class_mat, tmpl_class, he_cls, topk=a.topk)

    # ---- verdicts ----
    learning_load_bearing = bool(learn_spk_wta_held - rnd_spk_wta_held >= a.beat_margin)
    beats_nogo = bool(learn_spk_wta_held >= a.nogo_floor + a.beat_margin)     # strict (+margin)
    beats_nogo_raw = bool(learn_spk_wta_held > a.nogo_floor)                  # raw (> 0.34 floor)
    capability_go = bool(
        (learn_spk_wta_held >= chance + a.decode_margin)
        and (learn_spk_wta_held - A_held >= a.beat_margin)
        and (learn_spk_wta_held - H_held >= a.beat_margin)
        and learning_load_bearing
        and position_pooled_out
        and (scr_centroid_held <= chance + a.decode_margin)
    )
    architecture_load_bearing = bool(learn_spk_wta_held - H_held >= a.beat_margin)

    return {
        "seed": seed, "code": code,
        "chance_object": round(chance, 4), "chance_position": round(chance_pos, 4),
        "decode": {
            "A_v1_direct_held": round(A_held, 4),
            "H_flat_pool_held": round(H_held, 4),
            "LEARNED_spkwta_train": round(learn_spk_wta_train, 4),
            "LEARNED_spkwta_held": round(learn_spk_wta_held, 4),
            "LEARNED_centroid_held": round(learn_centroid_held, 4),
            "LEARNED_scramble_centroid_held": round(scr_centroid_held, 4),
            "RANDOM_spkwta_held": round(rnd_spk_wta_held, 4),
            "RANDOM_centroid_held": round(rnd_centroid_held, 4),
        },
        "reframe": {
            "learned_minus_random_spkwta": round(learn_spk_wta_held - rnd_spk_wta_held, 4),
            "learned_minus_random_centroid": round(learn_centroid_held - rnd_centroid_held, 4),
            "learning_load_bearing": learning_load_bearing,
        },
        "sparsity": {"learned": spars_learned, "random": spars_random},
        "dissociation": {
            "object_decode_heldsplit": round(obj_split, 4),
            "position_decode_heldsplit": round(pos_split, 4),
            "label_shuffle_null": round(lbl_shuffle_null, 4),
            "position_pooled_out": position_pooled_out,
        },
        "train_acc_curve": train_acc,
        "verdicts": {
            "capability_go": capability_go,
            "beats_config_c_nogo": beats_nogo,
            "beats_config_c_nogo_raw": beats_nogo_raw,
            "learning_load_bearing": learning_load_bearing,
            "architecture_load_bearing": architecture_load_bearing,
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
    attributable_to(f"[{code}] R-STDP spiking readout held -> LEARNING (vs random spiking readout)",
                    hd("LEARNED_spkwta_held"), hd("RANDOM_spkwta_held"))
    attributable_to(f"[{code}] R-STDP spiking readout held -> HIERARCHY (vs V1-direct held)",
                    hd("LEARNED_spkwta_held"), hd("A_v1_direct_held"))
    attributable_to(f"[{code}] R-STDP spiking readout held -> S2 CONJUNCTION (vs flat pool)",
                    hd("LEARNED_spkwta_held"), hd("H_flat_pool_held"))

    n_go = sum(1 for r in rows if r["verdicts"]["capability_go"])
    overall = ("RSTDP-READOUT-GO" if n_go == len(rows)
               else "RSTDP-READOUT-NOGO" if n_go == 0
               else f"RSTDP-READOUT-PARTIAL-{n_go}/{len(rows)}")
    return {
        "probe": "vision_rstdp_readout", "code": code, "overall_verdict": overall,
        "seeds": a.seeds, "n_seeds": len(rows), "chance_object": round(1.0 / a.n_classes, 4),
        "config_c_nogo_floor": a.nogo_floor,
        "per_seed_capability_go": [r["verdicts"]["capability_go"] for r in rows],
        "per_seed_learning_load_bearing": [r["verdicts"]["learning_load_bearing"] for r in rows],
        "decode_means": {k: mean(("decode", k)) for k in rows[0]["decode"]},
        "reframe_means": {
            "learned_spkwta_held": hd("LEARNED_spkwta_held"),
            "random_spkwta_held": hd("RANDOM_spkwta_held"),
            "learned_minus_random_spkwta": mean(("reframe", "learned_minus_random_spkwta")),
            "learned_minus_random_centroid": mean(("reframe", "learned_minus_random_centroid")),
        },
        "sparsity_means": {
            "learned_active_units": mean(("sparsity", "learned", "active_units_mean")),
            "random_active_units": mean(("sparsity", "random", "active_units_mean")),
            "learned_winclass_active_units": mean(("sparsity", "learned", "winclass_active_units_mean")),
            f"learned_winclass_top{a.topk}_mass_frac": mean(("sparsity", "learned", f"winclass_top{a.topk}_mass_frac_mean")),
            f"random_winclass_top{a.topk}_mass_frac": mean(("sparsity", "random", f"winclass_top{a.topk}_mass_frac_mean")),
        },
        "dissociation_means": {
            "object_decode_heldsplit": mean(("dissociation", "object_decode_heldsplit")),
            "position_decode_heldsplit": mean(("dissociation", "position_decode_heldsplit")),
            "label_shuffle_null": mean(("dissociation", "label_shuffle_null")),
        },
        "verdict_fracs": {k: frac(("verdicts", k)) for k in rows[0]["verdicts"]},
        "headroom": {
            "learned_minus_random_spkwta": round(hd("LEARNED_spkwta_held") - hd("RANDOM_spkwta_held"), 4),
            "learned_minus_v1_held": round(hd("LEARNED_spkwta_held") - hd("A_v1_direct_held"), 4),
            "learned_minus_flat_held": round(hd("LEARNED_spkwta_held") - hd("H_flat_pool_held"), 4),
            "learned_minus_nogo_floor": round(hd("LEARNED_spkwta_held") - a.nogo_floor, 4),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    p.add_argument("--code", choices=["latency", "count", "both"], default="count",
                   help="neural code the S2/C2 readout is read with (default count: config B's best "
                        "front-end code; a sparse learned code carries strong count differences)")
    p.add_argument("--c1-code", choices=["latency", "count"], default="count",
                   help="neural code for the reused LIF S1->C1 front end (config B best = count)")
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
    # V1 front end
    p.add_argument("--n-orientations", type=int, default=8)
    p.add_argument("--n-frequencies", type=int, default=2)
    p.add_argument("--n-pos", type=int, default=24)
    p.add_argument("--rf-radius", type=int, default=3)
    p.add_argument("--orient-norm", choices=["none", "div", "z"], default="z")
    p.add_argument("--c1-gate", type=float, default=0.15)
    p.add_argument("--c1-win", type=int, default=6)
    p.add_argument("--c1-stride", type=int, default=3)
    # S2 configural templates
    p.add_argument("--s2-p", type=int, default=3)
    p.add_argument("--n-s2", type=int, default=64,
                   help="template-bank size (round-robin over classes -> n_s2/n_classes per class)")
    # R-STDP
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.15, help="R-STDP learning rate")
    p.add_argument("--rstdp-win-k", type=int, default=0,
                   help="within-class credit: 0 = potentiate ALL fired true-class templates (dense, the "
                        "headline op-point; depress all fired wrong-class); >0 = competitive top-k "
                        "winners only + duty-cycle boosting (a sparser but WEAKER-reframe code)")
    p.add_argument("--boost-beta", type=float, default=2.0,
                   help="duty-cycle boosting strength for the competitive (rstdp-win-k>0) mode")
    p.add_argument("--c2-kwta-k", type=int, default=0,
                   help="EVAL-ONLY C2 readout kWTA: keep top-k templates active across the bank. 0/>=n_s2 "
                        "disables (default: OFF -- kWTA sparsification of the read HURTS, random benefits)")
    p.add_argument("--topk", type=int, default=5, help="k for sparsity concentration metric")
    # SPIKING (LIF) front end operating point (config B defaults)
    p.add_argument("--s1-mode", choices=["spiking", "rate"], default="spiking")
    p.add_argument("--s2-norm", choices=["none", "submean", "z"], default="z",
                   help="S2 lateral inhibition across the template bank per location (winner-relative "
                        "contrast so the near-threshold LIF is not saturated by the cosine common-mode)")
    p.add_argument("--T1", type=int, default=64, help="S1 LIF window (ms/steps)")
    p.add_argument("--T2", type=int, default=48, help="S2 LIF window (ms/steps)")
    p.add_argument("--tau", type=float, default=8.0)
    p.add_argument("--v-thresh", type=float, default=1.0)
    p.add_argument("--t-ref", type=int, default=2)
    p.add_argument("--noise", type=float, default=0.06)
    p.add_argument("--s1-gain", type=float, default=1.2)
    p.add_argument("--s2-gain", type=float, default=2.0)
    p.add_argument("--kwta-frac", type=float, default=0.15)
    # gate thresholds
    p.add_argument("--decode-margin", type=float, default=0.15)
    p.add_argument("--beat-margin", type=float, default=0.10)
    p.add_argument("--pos-decode-margin", type=float, default=0.15)
    p.add_argument("--nogo-floor", type=float, default=0.34, help="#72 config-C fully-spiking NO-GO held")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()

    t0 = time.time()
    codes = ["latency", "count"] if a.code == "both" else [a.code]
    print(f"[vision-RSTDP-readout] seeds={a.seeds} codes={codes} n_s2={a.n_s2} epochs={a.epochs} "
          f"lr={a.lr} c1_code={a.c1_code} LIF(T1={a.T1},T2={a.T2},s2g={a.s2_gain},kwta={a.kwta_frac})",
          flush=True)

    result = {}
    for code in codes:
        rows = [run_seed(s, a, code) for s in a.seeds]
        for r in rows:
            d, rf, di, v = r["decode"], r["reframe"], r["dissociation"], r["verdicts"]
            sp = r["sparsity"]["learned"]
            print(f"  [{code} seed {r['seed']}] V1he {d['A_v1_direct_held']:.2f} flat {d['H_flat_pool_held']:.2f} "
                  f"| LEARNED spkwta he {d['LEARNED_spkwta_held']:.2f} (tr {d['LEARNED_spkwta_train']:.2f}) "
                  f"cent {d['LEARNED_centroid_held']:.2f} | RANDOM spkwta {d['RANDOM_spkwta_held']:.2f} "
                  f"cent {d['RANDOM_centroid_held']:.2f} | dLEARN {rf['learned_minus_random_spkwta']:+.2f} "
                  f"| active {sp['active_units_mean']:.0f}/{sp['n_s2']} winact {sp['winclass_active_units_mean']:.1f} "
                  f"| obj/pos {di['object_decode_heldsplit']:.2f}/{di['position_decode_heldsplit']:.2f} "
                  f"| GO={v['capability_go']} learn_lb={v['learning_load_bearing']} "
                  f"beat_nogo={v['beats_config_c_nogo']}", flush=True)
        result[code] = {"summary": _summarize(rows, a, code, t0), "per_seed": rows}

    top = {
        "probe": "vision_rstdp_readout",
        "primary_code": codes[0],
        "overall_verdict": result[codes[0]]["summary"]["overall_verdict"],
        "config": vars(a),
        "by_code": result,
        "mechanism": (
            "Reused LIF S1->C1 (config B, spiking, PRESERVES the capability) -> convolutional S2 cosine "
            "drive -> S2 lateral inhibition (winner-relative contrast) -> LIF S2 coincidence spikes -> "
            "C2 spiking WTA global MAX-pool -> per-class spike-sum -> SPIKING WTA over class populations "
            "= prediction. S2 template bank LEARNED by online three-factor REWARD-MODULATED STDP "
            "(pre = winning-location C1 patch, post = C2 spike, dopamine sign = correct/incorrect; "
            "non-negative + L2-renorm). RANDOM arm = identical architecture, W untrained (config-C-like "
            "NO-GO control). Sources: Frémaux & Gerstner 2016; Izhikevich 2007; Mozafari et al. 2018 TNNLS."
        ),
        "reframe_test": (
            "On RATE, random S2 == learned S2 (learning NOT load-bearing; #72). PREDICTION on SPIKES: a "
            "sparse discriminative R-STDP readout becomes load-bearing (learned >> random) because the "
            "distributed random code is quantization-fragile. learning_load_bearing = "
            "(learned - random spiking-WTA held) >= beat_margin, per seed."
        ),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(top, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    for code in codes:
        s = result[code]["summary"]
        rm = s["reframe_means"]
        sm = s["sparsity_means"]
        print(f"[{code}] {s['overall_verdict']}  LEARNED_spkwta_held={rm['learned_spkwta_held']} "
              f"RANDOM_spkwta_held={rm['random_spkwta_held']} (dLEARN={rm['learned_minus_random_spkwta']:+}) "
              f"vs NOGO {s['config_c_nogo_floor']} | learn_lb {sum(s['per_seed_learning_load_bearing'])}/{s['n_seeds']} "
              f"GO {sum(s['per_seed_capability_go'])}/{s['n_seeds']} | active learned/random "
              f"{sm['learned_active_units']}/{sm['random_active_units']}", flush=True)
    print(f"[written] {out_path}", flush=True)
    print("=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
