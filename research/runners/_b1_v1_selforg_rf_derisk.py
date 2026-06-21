"""B1 de-risk: does a SELF-ORGANIZED V1 receptive-field bank discharge the
host-designed-Gabor residual?

CONTEXT (the scoping this implements):
  research/findings/2026-06-21-B1-v1-gabor-selforg-scoping.md  (0594b3b2)
  -> The V1 simple-cell RF weights are HOST-DESIGNED (a Gabor formula, 32
     oriented templates = 8 orient x 4 freq, sim/visual_cortex.py
     build_v1_simple_weights). The OPERATION (V1 filter -> spikes) runs
     on-substrate but the STRUCTURE (the Gabor weights) is host-computed ->
     a criterion-2 (neuromorphic-hardware-port) structure residual. A chip
     would need a host to compute + inject the Gabor bank.

THE DISCHARGE BAR (the honest framing from the scoping, NOT exact-Gabor recovery):
  The downstream pipeline (2026-06-16 generalization arc, Option B finding)
  uses V1 ONLY for SIMILARITY STRUCTURE -- the load-bearing output is a
  similarity-structured perception code (within>between margin), and that
  structure tracks the PIXELS not exact-Gabor-identity (RSA r=0.99 to pixels).
  So the GO bar is: a SELF-ORGANIZED RF bank (learned by a local rule from
  image input, OR developmentally-structured-random) that PRESERVES the
  pixel-similarity geometry -- measured by
    (a) RSA(self-org-RF codes vs host-Gabor-bank codes) HIGH, and
    (b) within>between category margin POSITIVE (reproduces Option B),
  with the discriminating controls:
    (c) a NO-LEARNING control collapses,
    (d) an UNSTRUCTURED-NOISE-input control collapses (catalog L.05 "wave/
        image content matters").
  Bonus faithfulness (nice-to-have, NOT required): Gabor-like orientation /
  frequency tuning of the learned RFs.

MECHANISMS (numpy/CPU, cheap-first):
  A (recommended) -- local-rule RF learning. SAILnet-spirit (Zylberberg-Murphy-
     DeWeese 2011): rate-Hebbian (Oja-normalized) feedforward + anti-Hebbian
     lateral inhibition + homeostatic per-unit threshold, on a STREAM of
     oriented-edge image patches (the V1-activating natural-image stimulus
     class; Olshausen-Field). RANDOM init -> learned RF bank. We use rate-
     Hebbian, NOT symmetric STDP (CYCLE-95 proved STDP is the wrong rule for
     symmetric correlation -- 656k events / 0 dW at dt~0).
  B (cheapest criterion-2 close) -- DEV-RANDOM structured-oriented-blob bank
     from rng(seed): a one-time genome-style random draw of localized oriented
     Gabor-like blobs (random orient/freq/phase/position), NOT the host Gabor
     formula. Moves the tag HOST-DESIGNED -> DEV-RANDOM (the accepted self-
     organized bar, like the role codes / feedback-alignment precedent).

ANTI-CHEAT CONTROLS:
  (c) NO-LEARNING  -- a fixed RANDOM RF bank (never trained): no oriented
      structure -> RSA-to-host LOW, margin LOW.
  (d) NOISE-INPUT  -- mechanism A trained on WHITE-NOISE patches instead of
      oriented-edge patches: oriented structure should NOT emerge (the
      structure comes from the input statistics + the rule, not the substrate).
  Plus: the LEARNED RFs and the TEST shapes are DISJOINT distributions -- the
  RF bank is trained on a broad oriented-edge patch stream; the test set is the
  Option-B 4/6 specific shape CATEGORIES. The bank never sees the test shapes
  (no leakage).

REUSE-BY-IMPORT (no sim/ edit):
  sim.visual_cortex.build_v1_simple_weights / gabor_kernel  -- the HOST Gabor
     bank = the SCORING REFERENCE the self-org bank is compared against (NOT
     deployed). The Option-B runner's shape rendering + similarity metrics are
     re-implemented here (small, self-contained) to keep this probe standalone.

Usage:
  python -m research.runners._b1_v1_selforg_rf_derisk \
      --seeds 42 43 44 \
      --out research/findings/raw/_b1_v1_selforg_rf_derisk.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# --- reuse-by-import: the project's REAL host Gabor front end (the REFERENCE) ---
from sim.visual_cortex import (
    build_v1_simple_weights,
    gabor_kernel,
    N_ORIENTATIONS,
    N_FREQUENCIES,
    V1_POSITIONS_PER_DIM,
    RETINA_SIZE,
    N_RETINA_CHANNELS,
)


# ============================================================================
# 0. The retina / RF geometry we self-organize in.
# ============================================================================
# To keep the learning tractable on CPU + interpretable, we learn RFs on a
# single retinotopic PATCH (a PATCH_SIZE x PATCH_SIZE window of the ON/OFF
# retina), then tile the learned filter bank translation-invariantly across the
# 16x16 position grid (exactly how the host bank reuses 32 templates across 256
# positions). This makes the self-org residual the SAME OBJECT as the host
# residual: a set of oriented filter TEMPLATES on a local patch.
PATCH = 9                 # local RF patch (>= host RF diameter 2*4+1 = 9)
N_FILTERS = 32            # learn 32 templates (= the host's 8 orient x 4 freq)
N_POS = V1_POSITIONS_PER_DIM   # 16
STRIDE = RETINA_SIZE // N_POS  # 2
PATCH_VEC = N_RETINA_CHANNELS * PATCH * PATCH   # ON/OFF patch = 2*9*9 = 162


# ============================================================================
# 1. Shape rendering -- the test set. Similarity lives in PIXELS only.
#    (re-implemented from _genfrontier_optionB_visual_similarity_derisk.py)
# ============================================================================

def _render_bar_image(cx, cy, theta, length, thickness, rng,
                      image_size=RETINA_SIZE, pixel_noise=0.04):
    """Oriented bar (line segment) -> (2,H,W) ON/OFF image (Option-B render)."""
    H = W = image_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    perp = np.abs(dx * math.sin(theta) - dy * math.cos(theta))
    along = dx * math.cos(theta) + dy * math.sin(theta)
    bar = np.exp(-(perp * perp) / (2.0 * thickness * thickness))
    bar = bar * (np.abs(along) <= (length / 2.0)).astype(np.float32)
    on = bar.astype(np.float32)
    gx = np.gradient(on, axis=1)
    gy = np.gradient(on, axis=0)
    off = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    off = off / (off.max() + 1e-6) * 0.3
    on = on + rng.normal(0.0, pixel_noise, size=on.shape).astype(np.float32)
    off = off + rng.normal(0.0, pixel_noise * 0.5, size=off.shape).astype(np.float32)
    on = np.clip(on, 0.0, 1.0)
    off = np.clip(off, 0.0, 1.0)
    return np.stack([on, off], axis=0)


def build_shape_set(n_categories, n_exemplars, rng, image_size=RETINA_SIZE):
    """n_categories x n_exemplars oriented-bar shapes + pixel-group labels
    (the Option-B test set: category = base orientation+position, exemplar =
    small visual jitter)."""
    images, labels, meta = [], [], []
    margin = image_size * 0.28
    for c in range(n_categories):
        base_theta = (c / n_categories) * math.pi
        ang = 2.0 * math.pi * (c / n_categories)
        base_cx = image_size / 2.0 + margin * math.cos(ang)
        base_cy = image_size / 2.0 + margin * math.sin(ang)
        base_len = image_size * 0.55
        base_thick = 1.6
        for e in range(n_exemplars):
            theta = base_theta + rng.normal(0.0, math.radians(7.0))
            cx = base_cx + rng.normal(0.0, image_size * 0.03)
            cy = base_cy + rng.normal(0.0, image_size * 0.03)
            length = base_len * (1.0 + rng.normal(0.0, 0.08))
            thick = base_thick * (1.0 + rng.normal(0.0, 0.10))
            img = _render_bar_image(cx, cy, theta, length, thick, rng,
                                    image_size=image_size)
            images.append(img)
            labels.append(c)
            meta.append(dict(category=c, exemplar=e,
                             base_theta_deg=round(math.degrees(base_theta), 1)))
    return (np.asarray(images, dtype=np.float32),
            np.asarray(labels, dtype=np.int64), meta)


# ============================================================================
# 2. Training-patch streams (the INPUT STATISTICS -- DISJOINT from the test set).
# ============================================================================

def _oriented_edge_patch(rng):
    """One oriented-edge / bar patch on the local ON/OFF patch grid.

    The V1-activating natural-image stimulus class (Olshausen-Field: natural
    images are dominated by oriented edges). RANDOM orientation/phase/frequency/
    position -- a BROAD distribution, NOT the 4/6 test categories. Returns a
    (PATCH_VEC,) ON/OFF patch vector (channel-first: [ON(81), OFF(81)]).
    """
    yy, xx = np.mgrid[0:PATCH, 0:PATCH].astype(np.float32)
    cx = (PATCH - 1) / 2.0 + rng.normal(0.0, 1.0)
    cy = (PATCH - 1) / 2.0 + rng.normal(0.0, 1.0)
    theta = rng.uniform(0.0, math.pi)           # any orientation
    # randomly an EDGE (step) or a BAR (line); both are oriented edge content
    if rng.random() < 0.5:
        # bar: gaussian ridge perpendicular to theta
        thick = rng.uniform(0.8, 2.2)
        perp = (xx - cx) * math.sin(theta) - (yy - cy) * math.cos(theta)
        sig = np.exp(-(perp * perp) / (2.0 * thick * thick)).astype(np.float32)
    else:
        # edge: smooth step across theta
        perp = (xx - cx) * math.sin(theta) - (yy - cy) * math.cos(theta)
        sig = (1.0 / (1.0 + np.exp(-perp / rng.uniform(0.5, 1.5)))).astype(np.float32)
        sig = sig - sig.mean()
    sig = sig + rng.normal(0.0, 0.05, size=sig.shape).astype(np.float32)
    # split bipolar signal into ON (positive) / OFF (negative magnitude),
    # matching the retina ON/OFF convention used by build_v1_simple_weights.
    on = np.clip(sig, 0.0, None)
    off = np.clip(-sig, 0.0, None)
    vec = np.concatenate([on.reshape(-1), off.reshape(-1)]).astype(np.float32)
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-9 else vec


def _noise_patch(rng):
    """UNSTRUCTURED white-noise patch (control d). No oriented content."""
    sig = rng.normal(0.0, 1.0, size=(PATCH, PATCH)).astype(np.float32)
    on = np.clip(sig, 0.0, None)
    off = np.clip(-sig, 0.0, None)
    vec = np.concatenate([on.reshape(-1), off.reshape(-1)]).astype(np.float32)
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-9 else vec


def make_patch_stream(n_patches, rng, kind="oriented"):
    gen = _oriented_edge_patch if kind == "oriented" else _noise_patch
    return np.stack([gen(rng) for _ in range(n_patches)], axis=0)  # (n_patches, PATCH_VEC)


# ============================================================================
# 3. Mechanism A -- SAILnet-spirit local-rule RF learning (rate-Hebbian).
# ============================================================================

def learn_rf_bank_sailnet(patches, n_filters=N_FILTERS, seed=0,
                          n_epochs=40, lr_W=0.05, lr_thresh=0.02,
                          lr_lateral=0.05, target_rate=0.05,
                          batch=64, verbose=False):
    """Learn a (n_filters, PATCH_VEC) RF bank from image patches by LOCAL rules.

    SAILnet (Zylberberg-Murphy-DeWeese 2011) Foldiak-style local learning:
      - feedforward Hebbian (Oja-normalized): dW_i = a_i * (x - a_i * W_i)
      - anti-Hebbian recurrent inhibition L between units (decorrelate):
            dL_ij = (a_i a_j - p^2)   (i != j), L >= 0
      - homeostatic per-unit threshold: dtheta_i = (a_i - p)
    Activity a is a rectified-linear membrane settled under lateral inhibition.
    RANDOM init. rate-Hebbian (NOT symmetric STDP; CYCLE-95).

    Returns: W (n_filters, PATCH_VEC) -- the learned RF templates.
    """
    rng = np.random.default_rng(seed)
    D = patches.shape[1]
    W = rng.normal(0.0, 1.0, size=(n_filters, D)).astype(np.float32)
    W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    L = np.zeros((n_filters, n_filters), dtype=np.float32)  # lateral inhibition (>=0)
    theta = np.full(n_filters, 0.5, dtype=np.float32)       # firing thresholds
    p = float(target_rate)

    n = patches.shape[0]
    for ep in range(n_epochs):
        order = rng.permutation(n)
        for bstart in range(0, n, batch):
            idx = order[bstart:bstart + batch]
            X = patches[idx]                       # (B, D)
            B = X.shape[0]
            # feedforward drive
            drive = X @ W.T                        # (B, n_filters)
            # settle activity under lateral inhibition (few fixed-point iters)
            a = np.maximum(drive - theta[None, :], 0.0)
            for _ in range(5):
                a = np.maximum(drive - theta[None, :] - a @ L.T, 0.0)
            # --- local updates (batch-averaged) ---
            # Oja feedforward: dW_i = mean_b a_bi (x_b - a_bi W_i)
            aT_x = a.T @ X                          # (n_filters, D)
            a2 = (a * a).sum(axis=0)                # (n_filters,)
            dW = (aT_x - a2[:, None] * W) / B
            W += lr_W * dW
            W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
            # anti-Hebbian lateral: dL_ij = mean(a_i a_j) - p^2, off-diagonal, >=0
            corr = (a.T @ a) / B                   # (n_filters, n_filters)
            dL = corr - p * p
            np.fill_diagonal(dL, 0.0)
            L += lr_lateral * dL
            np.maximum(L, 0.0, out=L)
            np.fill_diagonal(L, 0.0)
            # homeostatic threshold: dtheta_i = mean(a_i) - p
            dtheta = a.mean(axis=0) - p
            theta += lr_thresh * dtheta
        if verbose and (ep % 10 == 0 or ep == n_epochs - 1):
            print(f"  [A ep {ep}] |W| mean {np.linalg.norm(W,axis=1).mean():.3f} "
                  f"theta mean {theta.mean():.3f} L max {L.max():.3f}")
    return W


# ============================================================================
# 4. Mechanism B -- DEV-RANDOM structured oriented-blob bank (genome draw).
# ============================================================================

def devrandom_rf_bank(n_filters=N_FILTERS, seed=0):
    """A one-time genome-style random draw of localized oriented Gabor-like
    blobs on the local patch. RANDOM orientation/freq/phase/centre -- NOT the
    host Gabor FORMULA (which deterministically tiles 8 fixed orientations x 4
    fixed freqs). This moves the tag HOST-DESIGNED -> DEV-RANDOM (the accepted
    self-organized bar; the feedback-alignment precedent). The filter SHAPE is
    Gabor-like (biology: V1 RFs are oriented blobs) but the PARAMETERS are a
    random genome draw, not a host design.

    Returns (n_filters, PATCH_VEC) bank (ON/OFF channel-split, like the host).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:PATCH, 0:PATCH].astype(np.float32)
    cx0 = cy0 = (PATCH - 1) / 2.0
    W = np.zeros((n_filters, PATCH_VEC), dtype=np.float32)
    for i in range(n_filters):
        theta = rng.uniform(0.0, math.pi)          # random orientation
        freq = rng.uniform(0.08, 0.45)             # random spatial frequency
        phase = rng.uniform(0.0, 2 * math.pi)      # random phase
        sigma = rng.uniform(1.3, 3.0)              # random envelope
        cx = cx0 + rng.normal(0.0, 0.8)
        cy = cy0 + rng.normal(0.0, 0.8)
        kern = gabor_kernel(sigma, sigma, theta, freq, phase)
        g = np.array([[kern(x - cx, y - cy) for x in range(PATCH)]
                      for y in range(PATCH)], dtype=np.float32)
        on = np.clip(g, 0.0, None)
        off = np.clip(-g, 0.0, None)
        vec = np.concatenate([on.reshape(-1), off.reshape(-1)]).astype(np.float32)
        n = np.linalg.norm(vec)
        W[i] = vec / n if n > 1e-9 else vec
    return W


def random_rf_bank(n_filters=N_FILTERS, seed=0):
    """NO-LEARNING control (c): a fixed UNSTRUCTURED random RF bank (white noise
    weights). No oriented structure should emerge in the codes."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1.0, size=(n_filters, PATCH_VEC)).astype(np.float32)
    W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    return W


# ============================================================================
# 5. Encode the test shapes through a patch-template bank (tile over positions).
# ============================================================================

def _extract_patches(images):
    """Extract the local ON/OFF patch at each of the 16x16 retinotopic positions
    from each test image. Returns (N, N_POS*N_POS, PATCH_VEC).

    Mirrors build_v1_simple_weights: position (pos_x,pos_y) is centred at
    (pos_x*STRIDE + STRIDE//2, ...); a PATCH x PATCH window of the ON/OFF retina
    around it is the local input the filter bank reads.
    """
    N = images.shape[0]
    half = PATCH // 2
    # pad ON/OFF channels so edge positions have a full patch
    padded = np.pad(images, ((0, 0), (0, 0), (half, half), (half, half)),
                    mode="constant")
    out = np.empty((N, N_POS * N_POS, PATCH_VEC), dtype=np.float32)
    for pos_y in range(N_POS):
        for pos_x in range(N_POS):
            cy = pos_y * STRIDE + STRIDE // 2 + half   # +half for the pad offset
            cx = pos_x * STRIDE + STRIDE // 2 + half
            win = padded[:, :, cy - half:cy + half + 1, cx - half:cx + half + 1]
            # (N, 2, PATCH, PATCH) -> (N, PATCH_VEC) channel-first flatten
            vec = win.reshape(N, -1)
            out[:, pos_y * N_POS + pos_x, :] = vec
    return out


def encode_with_bank(images, W):
    """Encode test images through an RF template bank tiled over all positions.

    For each image: response[filter, position] = relu(W_filter . patch_position).
    Flatten to a (N, n_filters * N_POS^2) "V1-simple-like" code -- the same
    shape/role as the host V1-simple code (8192 = 32 templates x 256 positions),
    so the comparison to the host bank is apples-to-apples.
    """
    patches = _extract_patches(images)             # (N, P, PATCH_VEC)
    N, P, _ = patches.shape
    nf = W.shape[0]
    # (N, P, nf) = patches @ W.T  -> rectify -> (N, nf*P)
    resp = np.einsum("npd,fd->npf", patches, W)    # (N, P, nf)
    resp = np.maximum(resp, 0.0)
    # layout as (filter, position) to match host (orient/freq outer, position inner)
    code = np.transpose(resp, (0, 2, 1)).reshape(N, nf * P)
    return code.astype(np.float32)


# host reference code: the REAL Gabor V1-simple code (densified matmul)
def build_host_v1_matrix():
    pre, post, w = build_v1_simple_weights()
    n_v1 = N_ORIENTATIONS * N_FREQUENCIES * N_POS * N_POS
    n_retina = N_RETINA_CHANNELS * RETINA_SIZE * RETINA_SIZE
    W = np.zeros((n_v1, n_retina), dtype=np.float32)
    W[post, pre] = w
    return W


def encode_host_v1(images, Whost):
    N = images.shape[0]
    retina = images.reshape(N, -1).astype(np.float32)
    return np.maximum(retina @ Whost.T, 0.0)


# ============================================================================
# 6. Similarity metrics (Option-B definitions).
# ============================================================================

def _cos_matrix(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.where(norm < 1e-9, 1.0, norm)
    Xn = X / norm
    return Xn @ Xn.T


def within_between_margin(codes, labels):
    C = _cos_matrix(codes)
    N = codes.shape[0]
    same = labels[:, None] == labels[None, :]
    eye = np.eye(N, dtype=bool)
    within_mask = same & ~eye
    between_mask = ~same
    within = float(C[within_mask].mean()) if within_mask.any() else 0.0
    between = float(C[between_mask].mean()) if between_mask.any() else 0.0
    return within, between, within - between


def rsa_between_codes(codesA, codesB):
    """RSA: correlate off-diagonal of two code-sets' cosine matrices. LABEL-FREE.
    High r => codesA preserves the same pairwise-similarity geometry as codesB.
    Here codesB = the host Gabor code: does the self-org bank carry the SAME
    geometry the deployed host bank does?"""
    Ca = _cos_matrix(codesA)
    Cb = _cos_matrix(codesB)
    iu = np.triu_indices(Ca.shape[0], k=1)
    a, b = Ca[iu], Cb[iu]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rsa_pixel_provenance(images, codes):
    """RSA of codes vs raw PIXELS (the Option-B label-free provenance check)."""
    N = images.shape[0]
    pix = images.reshape(N, -1).astype(np.float32)
    return rsa_between_codes(codes, pix)


# ============================================================================
# 7. Optional faithfulness: Gabor-tuning of the learned RFs (nice-to-have).
# ============================================================================

def gabor_orientation_tuning(W):
    """Fraction of learned filters that are ORIENTED (vs blobby/unstructured).

    For each filter (recombine ON-OFF into a single bipolar patch), measure
    orientation selectivity via the structure tensor: oriented filters have an
    anisotropic gradient distribution (one dominant orientation). Report the
    mean orientation-selectivity index (0=isotropic, 1=perfectly oriented) and
    the fraction with OSI > 0.5. This is FAITHFULNESS (do Gabors emerge), NOT
    the discharge bar.
    """
    nf = W.shape[0]
    osis = []
    for i in range(nf):
        on = W[i, :PATCH * PATCH].reshape(PATCH, PATCH)
        off = W[i, PATCH * PATCH:].reshape(PATCH, PATCH)
        f = on - off                              # bipolar RF
        gx = np.gradient(f, axis=1)
        gy = np.gradient(f, axis=0)
        Jxx = float((gx * gx).sum())
        Jyy = float((gy * gy).sum())
        Jxy = float((gx * gy).sum())
        tr = Jxx + Jyy
        if tr < 1e-9:
            osis.append(0.0)
            continue
        # coherence of the structure tensor = orientation selectivity
        coh = math.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy * Jxy) / tr
        osis.append(coh)
    osis = np.asarray(osis)
    return float(osis.mean()), float((osis > 0.5).mean())


# ============================================================================
# 8. Per-seed run.
# ============================================================================

def run_seed(seed, n_categories, n_exemplars, n_patches, n_epochs):
    rng = np.random.default_rng(seed)

    # --- test set (Option-B shapes; similarity in PIXELS only) ---
    images, labels, meta = build_shape_set(n_categories, n_exemplars, rng)

    # --- HOST reference: the real Gabor V1-simple code (the scoring reference) ---
    Whost = build_host_v1_matrix()
    host_code = encode_host_v1(images, Whost)
    host_within, host_between, host_margin = within_between_margin(host_code, labels)
    host_rsa_pix = rsa_pixel_provenance(images, host_code)

    # --- training patch streams (DISJOINT from the test shapes) ---
    oriented_patches = make_patch_stream(n_patches, np.random.default_rng(seed + 1),
                                         kind="oriented")
    noise_patches = make_patch_stream(n_patches, np.random.default_rng(seed + 2),
                                      kind="noise")

    # === Mechanism A: SAILnet-spirit local-rule learning on oriented patches ===
    W_A = learn_rf_bank_sailnet(oriented_patches, seed=seed, n_epochs=n_epochs)
    code_A = encode_with_bank(images, W_A)
    A_within, A_between, A_margin = within_between_margin(code_A, labels)
    A_rsa_host = rsa_between_codes(code_A, host_code)
    A_rsa_pix = rsa_pixel_provenance(images, code_A)
    A_osi_mean, A_osi_frac = gabor_orientation_tuning(W_A)

    # === Mechanism B: DEV-RANDOM structured oriented-blob bank ===
    W_B = devrandom_rf_bank(seed=seed)
    code_B = encode_with_bank(images, W_B)
    B_within, B_between, B_margin = within_between_margin(code_B, labels)
    B_rsa_host = rsa_between_codes(code_B, host_code)
    B_rsa_pix = rsa_pixel_provenance(images, code_B)
    B_osi_mean, B_osi_frac = gabor_orientation_tuning(W_B)

    # === Control (c): NO-LEARNING random RF bank ===
    W_rand = random_rf_bank(seed=seed)
    code_rand = encode_with_bank(images, W_rand)
    rand_within, rand_between, rand_margin = within_between_margin(code_rand, labels)
    rand_rsa_host = rsa_between_codes(code_rand, host_code)
    rand_osi_mean, rand_osi_frac = gabor_orientation_tuning(W_rand)

    # === Control (d): NOISE-INPUT (mechanism A trained on white-noise patches) ===
    W_noise = learn_rf_bank_sailnet(noise_patches, seed=seed, n_epochs=n_epochs)
    code_noise = encode_with_bank(images, W_noise)
    noise_within, noise_between, noise_margin = within_between_margin(code_noise, labels)
    noise_rsa_host = rsa_between_codes(code_noise, host_code)
    noise_osi_mean, noise_osi_frac = gabor_orientation_tuning(W_noise)

    # --- per-seed verdict ---
    # GO bar: a self-org bank (A or B) PRESERVES the geometry:
    #   RSA-to-host >= 0.5  AND  margin positive (>= half the host margin, capped
    #   at the >=0.15 Option-B gate)  AND  both controls collapse.
    margin_gate = min(0.15, 0.5 * host_margin)
    A_ok = (A_rsa_host >= 0.5) and (A_margin >= margin_gate)
    B_ok = (B_rsa_host >= 0.5) and (B_margin >= margin_gate)
    # controls must collapse RELATIVE to the passing mechanism(s)
    best_margin = max(A_margin if A_ok else -1, B_margin if B_ok else -1)
    best_rsa = max(A_rsa_host if A_ok else -1, B_rsa_host if B_ok else -1)
    controls_collapse = (
        (rand_margin < 0.5 * best_margin or rand_rsa_host < 0.5 * best_rsa) and
        (noise_margin < 0.5 * best_margin or noise_rsa_host < 0.5 * best_rsa)
    ) if (A_ok or B_ok) else False

    if (A_ok or B_ok) and controls_collapse:
        verdict = "GO"
    elif (A_margin >= 0.05 or B_margin >= 0.05) and (A_rsa_host >= 0.3 or B_rsa_host >= 0.3):
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"

    def blk(within, between, margin, rsa_host=None, rsa_pix=None,
            osi_mean=None, osi_frac=None):
        d = dict(within=round(within, 4), between=round(between, 4),
                 margin=round(margin, 4))
        if rsa_host is not None:
            d["rsa_vs_host"] = round(rsa_host, 4)
        if rsa_pix is not None:
            d["rsa_vs_pixels"] = round(rsa_pix, 4)
        if osi_mean is not None:
            d["osi_mean"] = round(osi_mean, 4)
            d["osi_frac_gt0.5"] = round(osi_frac, 4)
        return d

    return dict(
        seed=seed, n_categories=n_categories, n_exemplars=n_exemplars, N=images.shape[0],
        margin_gate=round(margin_gate, 4),
        host_reference=blk(host_within, host_between, host_margin,
                           rsa_pix=host_rsa_pix),
        mechanism_A_learned=blk(A_within, A_between, A_margin,
                                A_rsa_host, A_rsa_pix, A_osi_mean, A_osi_frac),
        mechanism_B_devrandom=blk(B_within, B_between, B_margin,
                                  B_rsa_host, B_rsa_pix, B_osi_mean, B_osi_frac),
        control_c_no_learning=blk(rand_within, rand_between, rand_margin,
                                  rand_rsa_host, None, rand_osi_mean, rand_osi_frac),
        control_d_noise_input=blk(noise_within, noise_between, noise_margin,
                                  noise_rsa_host, None, noise_osi_mean, noise_osi_frac),
        A_ok=bool(A_ok), B_ok=bool(B_ok), controls_collapse=bool(controls_collapse),
        verdict=verdict,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-categories", type=int, default=4)
    ap.add_argument("--n-exemplars", type=int, default=4)
    ap.add_argument("--n-patches", type=int, default=4000)
    ap.add_argument("--n-epochs", type=int, default=40)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_b1_v1_selforg_rf_derisk.json")
    args = ap.parse_args()

    per_seed = [run_seed(s, args.n_categories, args.n_exemplars,
                         args.n_patches, args.n_epochs) for s in args.seeds]

    A_margins = [r["mechanism_A_learned"]["margin"] for r in per_seed]
    A_rsa = [r["mechanism_A_learned"]["rsa_vs_host"] for r in per_seed]
    A_osi = [r["mechanism_A_learned"]["osi_frac_gt0.5"] for r in per_seed]
    B_margins = [r["mechanism_B_devrandom"]["margin"] for r in per_seed]
    B_rsa = [r["mechanism_B_devrandom"]["rsa_vs_host"] for r in per_seed]
    host_margins = [r["host_reference"]["margin"] for r in per_seed]
    c_margins = [r["control_c_no_learning"]["margin"] for r in per_seed]
    c_rsa = [r["control_c_no_learning"]["rsa_vs_host"] for r in per_seed]
    d_margins = [r["control_d_noise_input"]["margin"] for r in per_seed]
    d_rsa = [r["control_d_noise_input"]["rsa_vs_host"] for r in per_seed]
    verdicts = [r["verdict"] for r in per_seed]

    all_go = all(v == "GO" for v in verdicts)
    A_pass_all = all(r["A_ok"] for r in per_seed)
    B_pass_all = all(r["B_ok"] for r in per_seed)
    controls_all = all(r["controls_collapse"] for r in per_seed)
    overall = "GO" if (all_go and controls_all and (A_pass_all or B_pass_all)) else (
        "PARTIAL" if all(v in ("GO", "PARTIAL") for v in verdicts) else "NEGATIVE")

    summary = dict(
        overall_verdict=overall,
        seeds=args.seeds,
        which_mechanism_passes=dict(A_all_seeds=bool(A_pass_all),
                                    B_all_seeds=bool(B_pass_all)),
        controls_collapse_all_seeds=bool(controls_all),
        host_reference_margin_mean=round(float(np.mean(host_margins)), 4),
        mechanism_A=dict(margin_mean=round(float(np.mean(A_margins)), 4),
                         margin_min=round(float(np.min(A_margins)), 4),
                         rsa_vs_host_mean=round(float(np.mean(A_rsa)), 4),
                         rsa_vs_host_min=round(float(np.min(A_rsa)), 4),
                         osi_frac_mean=round(float(np.mean(A_osi)), 4)),
        mechanism_B_devrandom=dict(margin_mean=round(float(np.mean(B_margins)), 4),
                                   margin_min=round(float(np.min(B_margins)), 4),
                                   rsa_vs_host_mean=round(float(np.mean(B_rsa)), 4),
                                   rsa_vs_host_min=round(float(np.min(B_rsa)), 4)),
        control_c_no_learning=dict(margin_mean=round(float(np.mean(c_margins)), 4),
                                   rsa_vs_host_mean=round(float(np.mean(c_rsa)), 4)),
        control_d_noise_input=dict(margin_mean=round(float(np.mean(d_margins)), 4),
                                   rsa_vs_host_mean=round(float(np.mean(d_rsa)), 4)),
        per_seed_verdicts=verdicts,
    )

    out = dict(summary=summary, per_seed=per_seed)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\n[written] {outp}")


if __name__ == "__main__":
    main()
