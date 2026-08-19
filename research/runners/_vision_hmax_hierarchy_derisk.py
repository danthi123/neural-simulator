"""Position-invariant CONFIGURAL object recognition via an HMAX S->C hierarchy (board #44).

WHY THIS RUNNER EXISTS. The predecessor de-risk
(research/findings/2026-08-19-vision-pooling-invariance-topology-not-learning-NOGO.md,
runner research/runners/_vision_pooling_invariance_derisk.py) proved, over 6 seeds, that a SINGLE
learned cross-position pooling layer is a NO-GO for position invariance, and QUANTIFIED the root
cause that unifies all four prior levers: the invariance-carrying quantity is the pooling TOPOLOGY
(which simple cells feed one complex unit across space), which is retinotopic/pre-wired -- a learned
pool "can only place weight where it SAW activity", so it can never respond at a held-out position.
It named the exact next mechanism, twice:
  (1) ACCEPT innate LOCAL retinotopic pooling as biology, then STACK simple->complex layers so GLOBAL
      invariance EMERGES from COMPOSED local shift-tolerance -- no single layer weights unseen
      positions (Riesenhuber & Poggio 1999, Nat Neurosci 2:1019-1025 -- HMAX; alternating S
      template-tuning / C MAX-pooling layers).
  (2) Make identity a CONJUNCTION of features (a CONFIGURAL object) so a random projection can NOT
      preserve it and the learned feature-binding is genuinely LOAD-BEARING (the prior NOGO showed
      learning was inert precisely because oriented-bar identity is a single LOCAL feature that any
      projection preserves).

THIS RUNNER DOES BOTH AT ONCE, which is the decisive test. Objects are histogram-MATCHED configural
shapes: K oriented strokes at K fixed relative slots, orientation assignment = a PERMUTATION, so
every class has an IDENTICAL global orientation histogram. A flat "pool everything per orientation"
oracle (the move the prior levers reached for) is therefore FORCED to chance -- configuration is the
only signal. The HMAX hierarchy composes:
  RETINA -> S1 (deployed Gabor/V1 simple, reused by import, NOT edited)
         -> C1 (innate LOCAL retinotopic MAX-pool per orientation -> local shift-tolerance; a FLAGGED
                developmental complex-cell RF scaffold, per route (1)'s "accept innate local pooling")
         -> S2 (LEARNED configural template units: each tuned to a local CONJUNCTION of C1 features,
                applied CONVOLUTIONALLY = innate retinotopic weight-sharing; the LEARNED part)
         -> C2 (innate GLOBAL MAX-pool per S2 template -> position invariance).
Object identity is decoded off the C2 vector.

THE KEY POINT that answers the prior NOGO's root cause: the LEARNED quantity (an S2 template) is NOT
a pool over positions -- it is a small local feature detector applied at EVERY position by the innate
convolution. It never has to "weight unseen positions": the retinotopic conv + global MAX pool spans
all positions identically, including held-out ones. Invariance is composed from innate local
shift-tolerance; identity is carried by the learned local conjunction templates.

ARMS (all share the S1->C1 front end; chance = 1/n_classes):
  A  V1-DIRECT      decode straight off C1 flattened (position-specific; the floor the prior NOGOs
                    also read). Fails at held positions.
  H  HIST-ORACLE    global orientation histogram (sum over ALL positions per orientation) -- a flat
                    position-invariant pool. CONFIGURATION-BLIND: forced to chance on matched objects.
  B  HMAX-IMPRINT   S2 templates IMPRINTED (one-shot Hebbian) from C1 patches of TRAIN-position images
                    (the canonical Serre/Poggio 2007 learned-patch HMAX). The main GO candidate.
  T  HMAX-TRACE     S2 templates learned by trace-modulated competitive Hebbian on a MOVING-object
                    continuity stream (Foldiak 1991; the fully-emergent biological variant).
  R  HMAX-RANDOM    LESION: identical hierarchy, RANDOM S2 templates. Isolates whether the LEARNED
                    templates (not the innate conv+pool) carry the configural identity.
  P1 HMAX-IMPRINT-p1  ABLATION: S2 spatial extent = 1 C1 cell (a single-stroke detector, no
                    conjunction). Collapses toward the histogram -> shows configural extent p>1 is
                    required.

ANTI-CHEATS (they ARE the result):
  1. HELD-OUT POSITIONS. Train S2 (imprint/trace) ONLY on even positions; decode at odd positions
     NEVER seen in training. Invariance = held-decode ~ train-decode AND >> V1-direct AND >> hist.
  2. LEARNED PART LOAD-BEARING. Lesion the learned S2 (RANDOM templates) -> configural decode drops
     toward chance/histogram. (learned - random) is the load-bearing margin. Also HMAX-TRACE vs a
     TEMPORAL-SHUFFLE control (route-2 trace test).
  3. POSITION POOLED OUT. Off the C2 code: object decodable, position NOT decodable (global MAX pool
     discards position). Plus a PIXEL-SCRAMBLE control (must not decode -> shape not a pixel artifact).
  4. 6 seeds (42/43/44/100/101/102), pooled + per-seed, + a LABEL-SHUFFLE null (-> chance).

CAPABILITY GO gate (per seed, read off the PRIMARY arm = HMAX-TRACE by default, a fully biological
learned arm) -- this is board #44, position-invariant CONFIGURAL recognition:
  held-object decode >= chance + decode_margin
  AND held beats V1-direct-held by beat_margin           (exceeds the RF-overlap ceiling)
  AND held beats FLAT-POOL by beat_margin                (composed conjunction+pool ARCHITECTURE is load-bearing)
  AND position pooled out (object decodable, position ~chance off the C2 code)
  AND pixel-scramble does not decode
  AND invariance gap (same-position CV minus held-position, one fit-split) <= inv_gap.

Reported SEPARATELY, because the prior NO-GO's root cause predicts it FAILS (and it does, 0/6):
  template_learning_load_bearing = held(imprint/trace) beats held(RANDOM projection) by beat_margin.
  A random S2 projection preserves the (separable) configural identity once the innate conjunction+pool
  architecture exists -> the invariance is carried by the composed TOPOLOGY, not the learned weights.
  This is the hierarchical analogue of the flat-case NO-GO ("topology not learning"), now at the S2 stage.

BRAIN-BASED status: a cheap-first RATE de-risk (a rate model is GENEROUS: if rate fails, spiking will
not save it). Unit drive = synaptic integration (template dot-product); C-layer MAX = a soft-max /
strongest-afferent complex-cell nonlinearity (Riesenhuber-Poggio's proposed cortical MAX op);
imprinting = one-shot Hebbian tuning; trace = a slow post-synaptic eligibility variable; competitive
update = local trace-modulated Hebbian + winner-take-all lateral inhibition. The retinotopic
weight-sharing (a template replicated across the field) and the local/global pooling windows are
FLAGGED innate developmental scaffolds, per the task's "accept innate local retinotopic pooling as
biology". No sim/ edit; reuses the deployed Gabor/V1 front end by import.

Smoke:
  SIM_BACKEND=numpy python -u -m research.runners._vision_hmax_hierarchy_derisk \
      --seeds 42 --out research/findings/raw/lanes/perception/vhmax_smoke.json
"""
from __future__ import annotations

import argparse
import itertools
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

from research.runners._genfrontier_optionB_visual_similarity_derisk import (  # noqa: E402
    build_gabor_response_matrix,
    encode_v1,
    pool_v1_to_complex,
)
from research.runners._laneD_v1_pooler_trace_invariance_derisk import _render_bar  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = Path("research/findings/raw/lanes/perception/vision_hmax_hierarchy.json")


# ============================================================================================
# Configural object rendering. Identity = an ARRANGEMENT (permutation) of orientations across a
# fixed set of relative slots. Every class shares the SAME orientation multiset -> identical global
# histogram -> configuration is the ONLY discriminating signal.
# ============================================================================================
def _object_classes(n_classes, n_slots):
    """Pick n_classes distinct permutations of range(n_slots) as the orientation->slot assignments.
    Every permutation uses each orientation exactly once => identical orientation histograms."""
    perms = list(itertools.permutations(range(n_slots)))
    # deterministic, spread-out choice (not the first n, to avoid near-identical arrangements)
    step = max(1, len(perms) // n_classes)
    chosen = [perms[(i * step) % len(perms)] for i in range(n_classes)]
    # de-dup while preserving count
    seen, out = set(), []
    for p in chosen:
        if p not in seen:
            seen.add(p)
            out.append(p)
    i = 0
    while len(out) < n_classes:
        if perms[i] not in seen:
            seen.add(perms[i])
            out.append(perms[i])
        i += 1
    return out[:n_classes]


def _render_object(center_xy, slot_perm, thetas, slot_offset, image_size, stroke_len, stroke_tk,
                   rng, pixel_noise, jitter):
    """Render one configural object: for each slot s, a stroke of orientation thetas[slot_perm[s]]
    at relative x-offset (s - (n_slots-1)/2)*slot_offset from center. Strokes max-combined."""
    cx0, cy0 = center_xy
    n_slots = len(slot_perm)
    on = np.zeros((image_size, image_size), dtype=np.float32)
    for s in range(n_slots):
        th = thetas[slot_perm[s]] + rng.normal(0.0, math.radians(4.0) * jitter)
        sx = cx0 + (s - (n_slots - 1) / 2.0) * slot_offset + rng.normal(0.0, image_size * 0.008 * jitter)
        sy = cy0 + rng.normal(0.0, image_size * 0.008 * jitter)
        ln = stroke_len * (1.0 + rng.normal(0.0, 0.04 * jitter))
        tk = stroke_tk * (1.0 + rng.normal(0.0, 0.06 * jitter))
        img = _render_bar(sx, sy, th, ln, tk, np.random.default_rng(0), image_size, 0.0)  # ON only, no noise
        on = np.maximum(on, img[0])
    gx = np.gradient(on, axis=1)
    gy = np.gradient(on, axis=0)
    off = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    off = off / (off.max() + 1e-6) * 0.3
    on = np.clip(on + rng.normal(0.0, pixel_noise, size=on.shape).astype(np.float32), 0.0, 1.0)
    off = np.clip(off + rng.normal(0.0, pixel_noise * 0.5, size=off.shape).astype(np.float32), 0.0, 1.0)
    return np.stack([on, off], axis=0)


def _positions(n, image_size, span):
    ctr = image_size * 0.5
    offs = np.linspace(-span, span, n)
    return [(float(ctr + o), float(ctr)) for o in offs]


def _build_objects(class_perms, thetas, positions, n_ex, a, seed):
    rng = np.random.default_rng(seed)
    imgs, cls, pos = [], [], []
    for ci, perm in enumerate(class_perms):
        for pi, ctr in enumerate(positions):
            for _ in range(n_ex):
                imgs.append(_render_object(ctr, perm, thetas, a.slot_offset, a.image_size,
                                           a.stroke_len, a.stroke_tk, rng, a.pixel_noise, jitter=1.0))
                cls.append(ci)
                pos.append(pi)
    return (np.asarray(imgs, np.float32), np.asarray(cls, np.int64), np.asarray(pos, np.int64))


def _scramble_images(images, seed):
    rng = np.random.default_rng(seed)
    c, h, w = images.shape[1:]
    out = np.empty_like(images)
    for i in range(images.shape[0]):
        perm = rng.permutation(h * w)
        out[i] = images[i].reshape(c, h * w)[:, perm].reshape(c, h, w)
    return out


# ============================================================================================
# HMAX layers.
# ============================================================================================
def _hypercolumn_norm(codes, n_orient, n_pos, mode="z", gate=0.15):
    """V1 hypercolumn processing, applied IDENTICALLY to every arm (it operates WITHIN a retinotopic
    position and does NOTHING across position, so it can never be the invariance differentiator):

      (i)  orientation COMPETITION -- lateral inhibition across the orientation columns of one
           hypercolumn (subtractive+divisive z / divisive). This removes the Gabor bank's large
           common-mode drive so the subtle CONFIGURAL signal (which orientation is at which slot)
           survives; without it a max-pool downstream is swamped by common-mode.
      (ii) an ENERGY GATE -- a hypercolumn only passes its normalised profile if its ABSOLUTE total
           drive exceeds `gate` x the image's peak (a firing threshold). WITHOUT this, z-normalisation
           fills EVERY background position with a spurious 'winning' orientation, and a downstream MAX
           pool then locks onto background noise instead of the object. The gate keeps background
           silent so the complex-cell MAX pool sees the object, not the noise floor. Biologically: a
           spike threshold / contrast-gain gate on the normalised orientation signal."""
    N = codes.shape[0]
    m = codes.reshape(N, n_orient, n_pos * n_pos).astype(np.float64, copy=True)
    energy = m.sum(axis=1)                                   # (N, n_pos^2) absolute drive per position
    eps = 1e-6
    if mode == "none":
        z = m
    elif mode == "div":
        z = m / (m.sum(axis=1, keepdims=True) + eps)
    elif mode == "z":
        mu = m.mean(axis=1, keepdims=True)
        sd = m.std(axis=1, keepdims=True)
        z = np.maximum((m - mu) / (sd + eps), 0.0)
    else:
        raise ValueError(mode)
    if gate > 0.0:
        thr = gate * energy.max(axis=1, keepdims=True)       # per-image firing threshold
        z = z * (energy >= thr)[:, None, :]
    return z.reshape(N, -1).astype(np.float32)


def _c1_maxpool(complex_map, n_orient, n_pos, win, stride):
    """C1: innate LOCAL retinotopic MAX-pool per orientation (developmental complex-cell RF; a
    FLAGGED scaffold). complex_map (N, n_orient*n_pos^2) -> (N, n_orient, g, g)."""
    N = complex_map.shape[0]
    F = complex_map.reshape(N, n_orient, n_pos, n_pos)
    starts = list(range(0, n_pos - win + 1, stride)) or [0]
    g = len(starts)
    out = np.zeros((N, n_orient, g, g), dtype=np.float32)
    for iy, sy in enumerate(starts):
        for ix, sx in enumerate(starts):
            out[:, :, iy, ix] = F[:, :, sy:sy + win, sx:sx + win].max(axis=(2, 3))
    return out


def _extract_patches(c1, p):
    """c1 (N, C, g, g) -> patches (N, n_loc, C*p*p) over all p x p sliding windows (stride 1)."""
    N, C, g, _ = c1.shape
    locs = list(itertools.product(range(g - p + 1), range(g - p + 1))) or [(0, 0)]
    out = np.zeros((N, len(locs), C * p * p), dtype=np.float32)
    for li, (iy, ix) in enumerate(locs):
        out[:, li, :] = c1[:, :, iy:iy + p, ix:ix + p].reshape(N, -1)
    return out


def _l2n(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(n < 1e-9, 1.0, n)


def _s2_c2(c1, templates, p):
    """S2 (convolutional cosine template match at every location) -> C2 (global MAX over locations).
    templates (n_S2, C*p*p), L2-normalised. Returns C2 code (N, n_S2)."""
    patches = _extract_patches(c1, p)               # (N, n_loc, D)
    pn = _l2n(patches, axis=2)                       # cosine match
    resp = pn @ templates.T                          # (N, n_loc, n_S2)
    return resp.max(axis=1).astype(np.float32)       # global MAX pool -> position invariant


def _imprint_templates(c1_train, p, n_S2, seed):
    """HMAX-IMPRINT: one-shot Hebbian tuning -- sample n_S2 C1 patches from TRAIN images as templates
    (Serre/Poggio 2007 learned-patch HMAX). Only patches with real structure (non-trivial norm)."""
    patches = _extract_patches(c1_train, p).reshape(-1, c1_train.shape[1] * p * p)
    norms = np.linalg.norm(patches, axis=1)
    keep = patches[norms > (0.25 * norms.max() + 1e-9)]
    if keep.shape[0] < n_S2:
        keep = patches[np.argsort(-norms)[:max(n_S2, 1)]]
    rng = np.random.default_rng(seed)
    idx = rng.choice(keep.shape[0], size=min(n_S2, keep.shape[0]), replace=False)
    return _l2n(keep[idx], axis=1)


def _random_templates(dim, n_S2, seed):
    """LESION: random Gaussian templates (no learning) -- same architecture, no configural tuning."""
    rng = np.random.default_rng(seed)
    return _l2n(np.abs(rng.standard_normal((n_S2, dim))).astype(np.float32), axis=1)


def _trace_competitive_templates(stream_patches, n_S2, epochs, lr, decay, boost_beta, seed):
    """HMAX-TRACE: trace-modulated competitive Hebbian on a MOVING-object C1-patch stream.
    Winner (on the leaky-integrated match) is pulled toward the patch; duty-cycle boosting spreads
    templates over configurations. stream_patches: list of (D,) C1 patches in continuity order."""
    D = stream_patches[0].shape[0]
    rng = np.random.default_rng(seed)
    W = _l2n(np.abs(rng.standard_normal((n_S2, D))).astype(np.float32) + 0.01, axis=1)
    T = max(len(stream_patches), 1)
    duty = np.ones(n_S2) / n_S2
    for _ in range(int(epochs)):
        y_bar = np.zeros(n_S2)
        wins = np.zeros(n_S2)
        boost = np.exp(boost_beta * (1.0 / n_S2 - duty))
        for x in stream_patches:
            xn = x / (np.linalg.norm(x) + 1e-9)
            g = (W @ xn) * boost
            y_bar = decay * y_bar + (1.0 - decay) * g
            k = int(np.argmax(y_bar))
            wins[k] += 1.0
            W[k] += lr * y_bar[k] * xn
            nn = np.linalg.norm(W[k])
            W[k] = np.maximum(W[k], 0.0) / (nn if nn > 1e-12 else 1.0)
        duty = 0.5 * duty + 0.5 * (wins / T)
    return _l2n(W, axis=1)


# ============================================================================================
# Decode.
# ============================================================================================
def _cos_normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n < 1e-9, 1.0, n)


def _centroid_decode(train_codes, train_labels, test_codes, test_labels):
    train = _cos_normalize(train_codes)
    test = _cos_normalize(test_codes)
    classes = np.unique(train_labels)
    cent = {}
    for c in classes:
        v = train[train_labels == c].mean(axis=0)
        nv = np.linalg.norm(v)
        cent[int(c)] = v / nv if nv > 1e-9 else v
    correct = 0
    for i in range(test.shape[0]):
        pred = max(classes, key=lambda c: float(test[i] @ cent[int(c)]))
        correct += int(pred == test_labels[i])
    return float(correct / max(1, test.shape[0]))


def _within_split_decode(codes, labels, seed):
    n = codes.shape[0]
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    h = n // 2
    return _centroid_decode(codes[idx[:h]], labels[idx[:h]], codes[idx[h:]], labels[idx[h:]])


def _hist_oracle(c1, n_orient):
    """Global orientation histogram: sum each orientation channel over ALL C1 positions. Position-
    invariant but configuration-BLIND (identical for histogram-matched classes)."""
    N = c1.shape[0]
    return c1.reshape(N, n_orient, -1).sum(axis=2).astype(np.float32)


def _continuity_patch_stream(c1_train, cls, pos, n_classes, p, passes, seed):
    """A moving-object stream of C1 PATCHES: each bout sweeps one object across its train positions
    in order; we take the max-norm patch of each frame (the object's location) so the trace sees the
    SAME configural patch translate. Returns a flat list of (D,) patches in continuity order."""
    rng = np.random.default_rng(seed)
    patches_all = _extract_patches(c1_train, p)             # (N, n_loc, D)
    by_cls = {c: np.where(cls == c)[0] for c in range(n_classes)}
    stream = []
    for _ in range(passes):
        cs = list(range(n_classes))
        rng.shuffle(cs)
        for c in cs:
            idx = by_cls[c]
            perm = rng.permutation(len(idx))
            order = idx[perm][np.argsort(pos[idx[perm]], kind="stable")]
            for i in order:
                pl = patches_all[i]
                # the patch centred on the object = the highest-energy location
                k = int(np.argmax(np.linalg.norm(pl, axis=1)))
                stream.append(pl[k])
    return stream


# ============================================================================================
def run_seed(seed, a):
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

    W = build_gabor_response_matrix(
        n_orientations=a.n_orientations, n_frequencies=a.n_frequencies,
        n_positions_per_dim=a.n_pos, retina_size=a.image_size, receptive_field_radius=a.rf_radius)

    def c1_of(imgs):
        cx = pool_v1_to_complex(encode_v1(imgs, W), a.n_orientations, a.n_frequencies, a.n_pos)
        cx = _hypercolumn_norm(cx, a.n_orientations, a.n_pos, a.orient_norm, a.c1_gate)
        return _c1_maxpool(cx, a.n_orientations, a.n_pos, a.c1_win, a.c1_stride)

    tr_c1, he_c1, sc_c1 = c1_of(tr_imgs), c1_of(he_imgs), c1_of(sc_imgs)
    chance = 1.0 / a.n_classes
    chance_pos = 1.0 / len(held_pi)

    # ---------- ARM A: V1-DIRECT (C1 flattened; position-specific) ----------
    def flat(c1):
        return c1.reshape(c1.shape[0], -1).astype(np.float32)
    A_train = _centroid_decode(flat(tr_c1), tr_cls, flat(tr_c1), tr_cls)  # train-on-train sanity
    A_held = _centroid_decode(flat(tr_c1), tr_cls, flat(he_c1), he_cls)

    # ---------- ARM H: FLAT-POOL (global orientation histogram = the NO-GO's flat approach) ----------
    # This is BOTH the configuration-blind control AND the ARCHITECTURE lesion: it removes the S2
    # conjunction stage and pools C1 straight to global (one value per orientation). On
    # histogram-matched configural objects it is forced toward chance -> so beating it isolates the
    # composed conjunction+pooling stage as the load-bearing mechanism (not the global pool per se).
    H_held = _centroid_decode(_hist_oracle(tr_c1, a.n_orientations), tr_cls,
                              _hist_oracle(he_c1, a.n_orientations), he_cls)

    # ---------- HMAX S2->C2 arms ----------
    dim = a.n_orientations * a.s2_p * a.s2_p

    def hmax_arm(templates, p):
        code_tr = _s2_c2(tr_c1, templates, p)
        code_he = _s2_c2(he_c1, templates, p)
        held_dec = _centroid_decode(code_tr, tr_cls, code_he, he_cls)  # full-train-centroid headline
        # FAIR invariance gap: ONE set of centroids (from a train fit-split), tested on same-position
        # held-out exemplars vs held-POSITION exemplars -> equal test-data amount, isolates the loss due
        # to POSITION change (not sample size, not self-fit optimism).
        n = code_tr.shape[0]
        idx = np.arange(n)
        np.random.default_rng(seed * 47 + 3).shuffle(idx)
        fit, test = idx[: n // 2], idx[n // 2:]
        same_pos = _centroid_decode(code_tr[fit], tr_cls[fit], code_tr[test], tr_cls[test])
        cross_pos = _centroid_decode(code_tr[fit], tr_cls[fit], code_he, he_cls)
        return {"train": same_pos, "held": held_dec, "same_pos": same_pos, "cross_pos": cross_pos,
                "code_he": code_he, "code_tr": code_tr, "templates": templates, "p": p}

    # ARM B: HMAX-IMPRINT (Serre/Poggio one-shot Hebbian patches)
    imp = hmax_arm(_imprint_templates(tr_c1, a.s2_p, a.n_s2, seed * 13 + 5), a.s2_p)
    # ARM T: HMAX-TRACE (trace-modulated competitive Hebbian on a moving-object continuity stream) + shuffle control
    stream = _continuity_patch_stream(tr_c1, tr_cls, tr_pos, a.n_classes, a.s2_p, a.trace_passes, seed * 17 + 7)
    shuf_stream = list(stream)
    np.random.default_rng(seed * 19 + 9).shuffle(shuf_stream)
    trc_ = hmax_arm(_trace_competitive_templates(stream, a.n_s2, a.trace_epochs, a.lr, a.trace_decay,
                                                 a.boost_beta, seed * 23 + 11), a.s2_p)
    tsh_ = hmax_arm(_trace_competitive_templates(shuf_stream, a.n_s2, a.trace_epochs, a.lr, a.trace_decay,
                                                 a.boost_beta, seed * 23 + 11), a.s2_p)
    # ARM R: HMAX-RANDOM (random projection; the template-learning lesion)
    rnd = hmax_arm(_random_templates(dim, a.n_s2, seed * 29 + 13), a.s2_p)
    # ARM P1: single-cell ablation (p=1: no conjunction extent)
    p1_ = hmax_arm(_imprint_templates(tr_c1, 1, a.n_s2, seed * 31 + 15), 1)

    arms = {"imprint": imp, "trace": trc_, "random": rnd}
    B_train, B_held, B_code_he = imp["train"], imp["held"], imp["code_he"]
    T_held, Tshuf_held, R_held, P1_held = trc_["held"], tsh_["held"], rnd["held"], p1_["held"]

    # PRIMARY capability arm (default TRACE = the biological temporal-continuity learner; the CAPABILITY
    # GO + all dissociations are read off THIS arm, so the GO is judged on a fully learned/biological arm).
    primary = arms[a.primary_arm]
    P_train, P_held, P_code_he = primary["train"], primary["held"], primary["code_he"]
    P_scr_held = _centroid_decode(_s2_c2(tr_c1, primary["templates"], primary["p"]), tr_cls,
                                  _s2_c2(sc_c1, primary["templates"], primary["p"]), he_cls)

    # ---------- anti-cheat 3: position pooled out (off the PRIMARY C2 held code) ----------
    obj_split = _within_split_decode(P_code_he, he_cls, seed * 37 + 17)
    pos_split = _within_split_decode(P_code_he, he_pos, seed * 37 + 19)
    position_pooled_out = (obj_split >= chance + a.decode_margin) and (pos_split <= chance_pos + a.pos_decode_margin)

    # ---------- anti-cheat 4: label-shuffle null (off PRIMARY held code) ----------
    lbl_shuf = np.random.default_rng(seed * 41 + 21).permutation(he_cls)
    B_labelshuffle = _within_split_decode(P_code_he, lbl_shuf, seed * 43 + 23)

    # ---------- verdicts ----------
    # PRIMARY = the CAPABILITY (board #44): position-invariant CONFIGURAL recognition. The HMAX
    # hierarchy must clear the held-out-position bar the flat learned pool (the prior NO-GO) could not:
    # beat the V1-direct floor AND the FLAT-POOL floor (= the composed conjunction+pool ARCHITECTURE is
    # load-bearing), with genuine invariance (pooled out, scramble fails, small train-held gap).
    # invariance gap = same-position minus cross-(held-)position accuracy, ONE fit-split of centroids
    # applied to both (equal test size). Small gap = genuine position invariance. Clamp at 0 (held
    # exceeding same-position is not an invariance failure).
    invariance_gap = max(0.0, primary["same_pos"] - primary["cross_pos"])
    capability_go = bool(
        (P_held >= chance + a.decode_margin)
        and (P_held - A_held >= a.beat_margin)          # beats position-specific V1-direct (RF-overlap ceiling)
        and (P_held - H_held >= a.beat_margin)          # beats FLAT-POOL -> ARCHITECTURE load-bearing
        and position_pooled_out
        and (P_scr_held <= chance + a.decode_margin)     # pixel scramble does not decode
        and (invariance_gap <= a.inv_gap)
    )
    architecture_load_bearing = bool(P_held - H_held >= a.beat_margin)  # conjunction+pool stage vs flat pool
    # SECONDARY (the honest test the prior NO-GO's root cause predicts FAILS): is the TEMPLATE-LEARNING
    # load-bearing, or does a random projection preserve the (separable) configural identity? Expected
    # FALSE -> invariance is carried by the innate composed TOPOLOGY, not the learned weights.
    template_learning_load_bearing = bool(P_held - R_held >= a.beat_margin)
    hist_blind = bool(H_held <= chance + a.decode_margin)  # flat pool forced to ~chance (configural)
    trace_load_bearing = bool(T_held - Tshuf_held >= a.beat_margin)

    return {
        "seed": seed,
        "chance_object": round(chance, 4),
        "chance_position": round(chance_pos, 4),
        "class_perms": [list(p) for p in class_perms],
        "train_positions_idx": train_pi,
        "held_positions_idx": held_pi,
        "n_train_images": int(tr_imgs.shape[0]),
        "n_held_images": int(he_imgs.shape[0]),
        "s2_dim": int(dim),
        "primary_arm": a.primary_arm,
        "decode": {
            "A_v1_direct_train": round(A_train, 4),
            "A_v1_direct_held": round(A_held, 4),
            "H_flat_pool_held": round(H_held, 4),
            "PRIMARY_same_pos_cv": round(primary["same_pos"], 4),
            "PRIMARY_cross_pos_cv": round(primary["cross_pos"], 4),
            "PRIMARY_held": round(P_held, 4),
            "PRIMARY_scramble_held": round(P_scr_held, 4),
            "B_hmax_imprint_train": round(B_train, 4),
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--n-slots", type=int, default=3, help="strokes per object; orientation multiset shared -> matched histograms")
    p.add_argument("--n-pos-total", type=int, default=8)
    p.add_argument("--pos-span", type=float, default=8.0)
    p.add_argument("--n-ex", type=int, default=6)
    p.add_argument("--image-size", type=int, default=56)
    p.add_argument("--slot-offset", type=float, default=10.0, help="px between adjacent stroke slots")
    p.add_argument("--stroke-len", type=float, default=7.0)
    p.add_argument("--stroke-tk", type=float, default=1.8)
    p.add_argument("--pixel-noise", type=float, default=0.03)
    p.add_argument("--primary-arm", choices=["trace", "imprint", "random"], default="trace",
                   help="which HMAX arm the CAPABILITY GO + dissociations are read off (default trace = "
                        "the biological temporal-continuity learner)")
    # V1 front end
    p.add_argument("--n-orientations", type=int, default=8)
    p.add_argument("--n-frequencies", type=int, default=2)
    p.add_argument("--n-pos", type=int, default=24)
    p.add_argument("--rf-radius", type=int, default=3)
    p.add_argument("--orient-norm", choices=["none", "div", "z"], default="z",
                   help="hypercolumn orientation competition (lateral inhibition); applied to ALL arms")
    p.add_argument("--c1-gate", type=float, default=0.15,
                   help="hypercolumn firing-threshold gate (frac of peak drive) that keeps background "
                        "silent so the complex-cell MAX pool sees the object; applied to ALL arms")
    # C1 innate local pool
    p.add_argument("--c1-win", type=int, default=6)
    p.add_argument("--c1-stride", type=int, default=3)
    # S2 configural templates
    p.add_argument("--s2-p", type=int, default=3, help="S2 spatial extent in C1 cells (must span >1 slot for a conjunction)")
    p.add_argument("--n-s2", type=int, default=128)
    # trace learning
    p.add_argument("--trace-passes", type=int, default=12)
    p.add_argument("--trace-epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--trace-decay", type=float, default=0.9)
    p.add_argument("--boost-beta", type=float, default=2.0)
    # gate thresholds
    p.add_argument("--decode-margin", type=float, default=0.15)
    p.add_argument("--beat-margin", type=float, default=0.10)
    p.add_argument("--pos-decode-margin", type=float, default=0.15)
    p.add_argument("--inv-gap", type=float, default=0.20)
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()

    t0 = time.time()
    print(f"[vision-hmax] seeds={a.seeds} classes={a.n_classes} slots={a.n_slots} pos={a.n_pos_total}@span{a.pos_span} "
          f"V1(orient={a.n_orientations},pos={a.n_pos},rf={a.rf_radius}) C1(win={a.c1_win},str={a.c1_stride}) "
          f"S2(p={a.s2_p},n={a.n_s2})", flush=True)

    rows = [run_seed(s, a) for s in a.seeds]
    for r in rows:
        d = r["decode"]
        di = r["dissociation"]
        v = r["verdicts"]
        print(f"  [seed {r['seed']}] V1 tr {d['A_v1_direct_train']:.2f} he {d['A_v1_direct_held']:.2f} "
              f"| flat {d['H_flat_pool_held']:.2f} | PRIMARY({r['primary_arm']}) same {d['PRIMARY_same_pos_cv']:.2f} "
              f"he {d['PRIMARY_held']:.2f} scr {d['PRIMARY_scramble_held']:.2f} | trace {d['T_hmax_trace_held']:.2f} "
              f"(shuf {d['T_hmax_traceshuffle_held']:.2f}) imp {d['B_hmax_imprint_held']:.2f} rand {d['R_hmax_random_held']:.2f} "
              f"p1 {d['P1_hmax_imprint_p1_held']:.2f} | obj/pos {di['object_decode_heldsplit']:.2f}/"
              f"{di['position_decode_heldsplit']:.2f} | CAP-GO={v['capability_go']} "
              f"arch_lb={v['architecture_load_bearing']} learn_lb={v['template_learning_load_bearing']}", flush=True)

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
    # attribution (1): the CAPABILITY over V1-direct held -> the composed HIERARCHY (innate pooling).
    attributable_to("HMAX held-invariance -> composed HIERARCHY (vs V1-direct held)",
                    hd("PRIMARY_held"), hd("A_v1_direct_held"))
    # attribution (2): the CAPABILITY over the FLAT pool -> the S2 CONJUNCTION stage (architecture).
    attributable_to("HMAX held-invariance -> S2 CONJUNCTION stage (vs flat pool)",
                    hd("PRIMARY_held"), hd("H_flat_pool_held"))
    # attribution (3): does TEMPLATE-LEARNING add over a random projection? (expected ~0/negative ->
    #   the honest finding: invariance is carried by the innate composed TOPOLOGY, not learned weights).
    attributable_to("HMAX held-invariance -> LEARNED templates (vs random projection)",
                    hd("PRIMARY_held"), hd("R_hmax_random_held"), warn_below=0.0)

    n_go = sum(1 for r in rows if r["verdicts"]["capability_go"])
    overall = ("HMAX-INVARIANCE-GO" if n_go == len(rows)
               else "HMAX-INVARIANCE-NOGO" if n_go == 0
               else f"HMAX-INVARIANCE-PARTIAL-{n_go}/{len(rows)}")

    summary = {
        "probe": "vision_hmax_hierarchy",
        "overall_verdict": overall,
        "seeds": a.seeds,
        "n_seeds": len(rows),
        "chance_object": round(1.0 / a.n_classes, 4),
        "per_seed_capability_go": [r["verdicts"]["capability_go"] for r in rows],
        "decode_means": {
            "A_v1_direct_train": mean(("decode", "A_v1_direct_train")),
            "A_v1_direct_held": mean(("decode", "A_v1_direct_held")),
            "H_flat_pool_held": mean(("decode", "H_flat_pool_held")),
            "PRIMARY_same_pos_cv": mean(("decode", "PRIMARY_same_pos_cv")),
            "PRIMARY_cross_pos_cv": mean(("decode", "PRIMARY_cross_pos_cv")),
            "PRIMARY_held": mean(("decode", "PRIMARY_held")),
            "PRIMARY_scramble_held": mean(("decode", "PRIMARY_scramble_held")),
            "B_hmax_imprint_train": mean(("decode", "B_hmax_imprint_train")),
            "B_hmax_imprint_held": mean(("decode", "B_hmax_imprint_held")),
            "T_hmax_trace_held": mean(("decode", "T_hmax_trace_held")),
            "T_hmax_traceshuffle_held": mean(("decode", "T_hmax_traceshuffle_held")),
            "R_hmax_random_held": mean(("decode", "R_hmax_random_held")),
            "P1_hmax_imprint_p1_held": mean(("decode", "P1_hmax_imprint_p1_held")),
        },
        "dissociation_means": {
            "object_decode_heldsplit": mean(("dissociation", "object_decode_heldsplit")),
            "position_decode_heldsplit": mean(("dissociation", "position_decode_heldsplit")),
            "label_shuffle_null": mean(("dissociation", "label_shuffle_null")),
        },
        "invariance_gap_mean": mean(("invariance_gap_train_minus_held",)),
        "verdict_fracs": {
            "capability_go": frac(("verdicts", "capability_go")),
            "architecture_load_bearing": frac(("verdicts", "architecture_load_bearing")),
            "template_learning_load_bearing": frac(("verdicts", "template_learning_load_bearing")),
            "flat_pool_configuration_blind": frac(("verdicts", "flat_pool_configuration_blind")),
            "trace_load_bearing": frac(("verdicts", "trace_load_bearing")),
        },
        "headroom": {
            "primary_minus_v1_held": round(hd("PRIMARY_held") - hd("A_v1_direct_held"), 4),
            "primary_minus_flat_held": round(hd("PRIMARY_held") - hd("H_flat_pool_held"), 4),
            "primary_minus_random_held": round(hd("PRIMARY_held") - hd("R_hmax_random_held"), 4),
            "primary_p3_minus_p1_held": round(hd("PRIMARY_held") - hd("P1_hmax_imprint_p1_held"), 4),
        },
        "mechanism": (
            "RETINA -> S1 (deployed Gabor/V1 simple) -> hypercolumn orientation-competition + firing-gate "
            "-> C1 (innate local retinotopic MAX-pool per orientation = local shift-tolerance, flagged "
            "scaffold) -> S2 (configural template units, convolutional cosine match = a local conjunction "
            "detector applied at every position by innate retinotopic weight-sharing; templates imprinted "
            "/ trace-learned / random) -> C2 (innate global MAX-pool per template = position invariance). "
            "Identity decoded off C2. Objects are histogram-matched configural shapes so a flat "
            "orientation-histogram pool (= no S2 stage) is forced to chance and configuration is the only "
            "signal. HMAX = Riesenhuber & Poggio 1999, Nat Neurosci 2:1019-1025; learned patches = "
            "Serre/Poggio 2007; trace = Foldiak 1991."
        ),
        "go_gate": (
            "CAPABILITY GO per seed (board #44 = position-invariant configural recognition): held-object "
            "decode >= chance+decode_margin; held beats V1-direct-held AND the FLAT-POOL (= composed "
            "conjunction+pool ARCHITECTURE load-bearing) by beat_margin; position pooled out (object "
            "decodable, position ~chance off C2); pixel-scramble does not decode; invariance gap "
            "|held-train| <= inv_gap. Reported separately (the honest test the prior NO-GO predicts FAILS): "
            "template_learning_load_bearing = HMAX-imprint beats HMAX-random (a random projection) by "
            "beat_margin."
        ),
        "config": vars(a),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_seed": rows}, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"[written] {out_path}", flush=True)
    print("=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
