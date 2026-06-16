"""Option B de-risk: do SHARED VISUAL FEATURES (object shapes through the
project's Gabor/V1 front end) produce SIMILARITY-STRUCTURED perception codes?

Context (the cheap-first that just went GO):
  research/findings/2026-06-16-generalization-crossmodal-unify-cheap-first.md
  -> cross-modal Hebbian convergence transfers the conversation cortex's
     category-generalization to perception, BUT ONLY when the perception input
     itself carries similarity structure. "Option B is the PREREQUISITE."

THE LOAD-BEARING QUESTION (this probe):
  Does LEGITIMATE sensory rendering -- object shapes that share visual features
  within a category, passed through the EXISTING Gabor/V1 front end -- produce
  perception codes where same-category objects are SIMILAR and different-category
  objects are DISSIMILAR (the way IT does for visually-similar objects; Op de
  Beeck / Kriegeskorte RSA, Kiani 2007)?  The similarity must come from PIXELS
  (shared visual features), NOT from any injected semantic label.

WHAT IS REUSED (no sim/ edit):
  sim.visual_cortex.build_v1_simple_weights  -- the real retina->V1 Gabor RF bank
                                                (the project's V1 front end).
  sim.visual_cortex.gabor_kernel             -- (not directly needed; the bank is)
  We build the retina->V1 sparse Gabor weight matrix ONCE, then matmul each shape
  image through it to get a V1-simple response vector, then pool V1-simple->
  V1-complex (the runner's fixed phase/frequency pooling within orientation x
  position) for an "IT-like" pooled code.  These are EXACTLY the layers the
  deployed nav stack uses (g11_bg_runner lines ~2446-2566).

  The public API (render_gridworld_to_image) only renders the gridworld scene,
  so -- as the task permits -- we drive the Gabor RF bank directly on our own
  shape images (reuse-by-import).

THE FLAT-DISTINCT BASELINE (the discriminating gap):
  sim.text_embeddings.orthogonal_drive_pattern -- the current nav regime's codes
  (non-overlapping bands -> between-code cosine == 0 exactly).  Option B must beat
  this by producing within>between similarity from VISUAL features.

GATE (3 seeds 42/43/44):
  GO       margin (within-cat cos - between-cat cos) >= 0.15, AND the structure
           tracks VISUAL features not labels (anti-cheat 1), AND >> flat baseline.
  PARTIAL  0.05 <= margin < 0.15.
  NEGATIVE margin ~ 0 (Gabor front end gives no similarity structure -> Option B
           needs a learned similarity-preserving projection instead; a localized
           next step).

ANTI-CHEATS (mandatory):
  1. Feature-provenance / label-scramble: the within/between structure must follow
     the actual shared PIXELS, not the assigned category labels.  We verify two
     ways:
       (a) cluster the codes by their OWN code-space similarity (kmeans on the
           cosine geometry) and check the recovered clusters match the
           pixel-defined groups (adjusted purity), proving similarity == shape;
       (b) a randomly SCRAMBLED label assignment destroys the measured margin
           (the margin computed against scrambled labels collapses to ~0), while
           the TRUE pixel-group margin stays high -- so the signal is in the
           pixels, not a label we injected.
  2. Flat-distinct baseline (orthogonal_drive_pattern) scores margin ~ 0.
  3. No pre-seeded semantics: the category basis is a RENDERED pixel pattern (an
     oriented bar at angle theta_c at base position p_c), never a hand-set code
     vector added to the perception code.  (Construction documented below.)

Usage:
  python -m research.runners._genfrontier_optionB_visual_similarity_derisk \
      --seeds 42 43 44 --out research/findings/raw/_genfrontier_optionB_visual_similarity.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# --- reuse-by-import: the project's REAL V1 Gabor front end (no sim/ edit) ---
from sim.visual_cortex import (
    build_v1_simple_weights,
    N_ORIENTATIONS,
    N_FREQUENCIES,
    V1_POSITIONS_PER_DIM,
    RETINA_SIZE,
    N_RETINA_CHANNELS,
)
from sim.text_embeddings import orthogonal_drive_pattern


# ----------------------------------------------------------------------------
# 1. Shape rendering -- the similarity lives in PIXELS only.
# ----------------------------------------------------------------------------
# Each CATEGORY is a distinct VISUAL basis: an oriented bar (a line segment) at a
# base orientation theta_c, anchored at a base position (cx_c, cy_c).  An oriented
# bar is the canonical Gabor-activating stimulus (Hubel-Wiesel), so categories that
# differ in bar ORIENTATION + POSITION will drive DIFFERENT V1 orientation/position
# columns.  Each EXEMPLAR within a category = the category bar + small per-exemplar
# visual jitter (small angle wobble + small position shift + small length change +
# pixel noise).  NOTHING about the category is injected as a code; it is purely the
# rendered pixels.

def _render_bar_image(
    cx: float, cy: float, theta: float, length: float,
    thickness: float, rng: np.random.Generator,
    image_size: int = RETINA_SIZE, pixel_noise: float = 0.04,
) -> np.ndarray:
    """Render an oriented bar (line segment) into a (2, H, W) ON/OFF image.

    ON channel = bar intensity (a soft-edged line). OFF channel = a faint
    boundary/edge signal (gradient magnitude), matching the gridworld render's
    ON/OFF convention. The bar is the ONLY structured content; everything else
    is low-amplitude pixel noise.
    """
    H = W = image_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # signed distance from the (infinite) line through (cx,cy) at angle theta:
    #   normal direction is (sin theta, -cos theta) (perp to the bar direction).
    dx = xx - cx
    dy = yy - cy
    # distance perpendicular to the bar's long axis
    perp = np.abs(dx * math.sin(theta) - dy * math.cos(theta))
    # coordinate ALONG the bar's long axis (to clip to a finite segment)
    along = dx * math.cos(theta) + dy * math.sin(theta)
    # soft bar: Gaussian profile across thickness, hard clip along length
    bar = np.exp(-(perp * perp) / (2.0 * thickness * thickness))
    bar = bar * (np.abs(along) <= (length / 2.0)).astype(np.float32)

    on = bar.astype(np.float32)
    # OFF channel: faint edges of the bar (gradient magnitude), gives the front
    # end a bipolar signal like the gridworld render.
    gx = np.gradient(on, axis=1)
    gy = np.gradient(on, axis=0)
    off = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    off = off / (off.max() + 1e-6) * 0.3   # match gridworld OFF amplitude (~0.3)

    # low-amplitude pixel noise on BOTH channels (visual exemplar variation)
    on = on + rng.normal(0.0, pixel_noise, size=on.shape).astype(np.float32)
    off = off + rng.normal(0.0, pixel_noise * 0.5, size=off.shape).astype(np.float32)
    on = np.clip(on, 0.0, 1.0)
    off = np.clip(off, 0.0, 1.0)
    return np.stack([on, off], axis=0)


def build_shape_set(
    n_categories: int, n_exemplars: int, rng: np.random.Generator,
    image_size: int = RETINA_SIZE,
):
    """Build n_categories x n_exemplars shape images + the pixel-group labels.

    Category c gets a base orientation theta_c spread across [0, pi) and a base
    centre (cx_c, cy_c) spread across the image, so categories are visually
    distinct (different orientation columns + retinotopic positions).  Each
    exemplar perturbs angle / centre / length / thickness slightly + pixel noise,
    so same-category exemplars SHARE the bar (similar pixels) while differing in
    detail.

    Returns:
        images: (N, 2, H, W) float32  (N = n_categories * n_exemplars)
        labels: (N,) int   the TRUE pixel-defined category of each image
        meta:   list of per-image construction dicts (for honesty/audit)
    """
    images = []
    labels = []
    meta = []
    margin = image_size * 0.28
    for c in range(n_categories):
        # category visual basis -- a property of the PIXELS, not a code vector:
        base_theta = (c / n_categories) * math.pi            # 0, 45, 90, 135 deg for 4 cats
        # spread category centres around the image on a ring (distinct positions)
        ang = 2.0 * math.pi * (c / n_categories)
        base_cx = image_size / 2.0 + margin * math.cos(ang)
        base_cy = image_size / 2.0 + margin * math.sin(ang)
        base_len = image_size * 0.55
        base_thick = 1.6
        for e in range(n_exemplars):
            # small per-exemplar VISUAL jitter (shared bar, varied detail)
            theta = base_theta + rng.normal(0.0, math.radians(7.0))
            cx = base_cx + rng.normal(0.0, image_size * 0.03)
            cy = base_cy + rng.normal(0.0, image_size * 0.03)
            length = base_len * (1.0 + rng.normal(0.0, 0.08))
            thick = base_thick * (1.0 + rng.normal(0.0, 0.10))
            img = _render_bar_image(
                cx, cy, theta, length, thick, rng, image_size=image_size,
            )
            images.append(img)
            labels.append(c)
            meta.append(dict(
                category=c, exemplar=e,
                base_theta_deg=round(math.degrees(base_theta), 1),
                theta_deg=round(math.degrees(theta), 1),
                cx=round(float(cx), 1), cy=round(float(cy), 1),
                length=round(float(length), 1), thickness=round(float(thick), 2),
            ))
    return (np.asarray(images, dtype=np.float32),
            np.asarray(labels, dtype=np.int64), meta)


# ----------------------------------------------------------------------------
# 2. Encode each shape through the REAL Gabor/V1 front end.
# ----------------------------------------------------------------------------

def _retina_index(channel, py, px, retina_size=RETINA_SIZE):
    return channel * (retina_size * retina_size) + py * retina_size + px


def build_gabor_response_matrix(
    n_orientations=N_ORIENTATIONS, n_frequencies=N_FREQUENCIES,
    n_positions_per_dim=V1_POSITIONS_PER_DIM, retina_size=RETINA_SIZE,
    receptive_field_radius=4,
):
    """Build the dense (n_v1_simple, n_retina) Gabor weight matrix from the
    project's SPARSE retina->V1 weights.  V1_simple response = W @ retina_drive.

    This is the EXACT same Gabor RF bank the deployed nav stack installs
    (apply_v1_gabor_weights -> build_v1_simple_weights).
    """
    pre, post, w = build_v1_simple_weights(
        n_orientations=n_orientations, n_frequencies=n_frequencies,
        n_positions_per_dim=n_positions_per_dim, retina_size=retina_size,
        receptive_field_radius=receptive_field_radius,
    )
    n_v1 = n_orientations * n_frequencies * n_positions_per_dim * n_positions_per_dim
    n_retina = N_RETINA_CHANNELS * retina_size * retina_size
    W = np.zeros((n_v1, n_retina), dtype=np.float32)
    W[post, pre] = w   # the sparse Gabor weights, densified for a clean matmul
    return W


def encode_v1(images: np.ndarray, W: np.ndarray) -> np.ndarray:
    """images (N,2,H,W) -> V1-simple responses (N, n_v1_simple), rectified.

    Mirrors image_to_retina_drive (flatten channel-first) then the retina->V1
    matmul.  Rectified (relu) because V1 firing rates are non-negative.
    """
    N = images.shape[0]
    retina = images.reshape(N, -1).astype(np.float32)   # (N, n_retina), channel-first flatten
    v1 = retina @ W.T                                    # (N, n_v1_simple)
    return np.maximum(v1, 0.0)                           # rectify (non-negative rates)


def pool_v1_to_complex(
    v1: np.ndarray, n_orientations=N_ORIENTATIONS, n_frequencies=N_FREQUENCIES,
    n_positions_per_dim=V1_POSITIONS_PER_DIM,
) -> np.ndarray:
    """V1-simple -> V1-complex phase/frequency pooling (the runner's fixed
    pooling).  Complex index = orient*(n_pos^2) + pos_y*n_pos + pos_x; pools
    (sums) the n_frequencies simple cells that share orientation x position.

    Simple index layout (build_v1_simple_weights):
      orient*(n_freq*n_pos^2) + freq*(n_pos^2) + pos_y*n_pos + pos_x.
    """
    N = v1.shape[0]
    n_pos2 = n_positions_per_dim * n_positions_per_dim
    n_complex = n_orientations * n_pos2
    v1r = v1.reshape(N, n_orientations, n_frequencies, n_pos2)
    complex_resp = v1r.sum(axis=2)                       # pool over frequency axis
    return complex_resp.reshape(N, n_complex)


# ----------------------------------------------------------------------------
# 3. Similarity-preserving signature.
# ----------------------------------------------------------------------------

def _cos_matrix(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.where(norm < 1e-9, 1.0, norm)
    Xn = X / norm
    return Xn @ Xn.T


def within_between_margin(codes: np.ndarray, labels: np.ndarray):
    """within-cat mean cosine, between-cat mean cosine, margin (within - between).
    Diagonal (self-similarity) excluded."""
    C = _cos_matrix(codes)
    N = codes.shape[0]
    same = labels[:, None] == labels[None, :]
    eye = np.eye(N, dtype=bool)
    within_mask = same & ~eye
    between_mask = ~same
    within = float(C[within_mask].mean()) if within_mask.any() else 0.0
    between = float(C[between_mask].mean()) if between_mask.any() else 0.0
    return within, between, within - between


def _kmeans_cosine(codes: np.ndarray, k: int, rng: np.random.Generator, iters=50):
    """Tiny spherical k-means (cosine) -> cluster assignment, for anti-cheat 1a.
    No sklearn dependency."""
    norm = np.linalg.norm(codes, axis=1, keepdims=True)
    norm = np.where(norm < 1e-9, 1.0, norm)
    X = codes / norm
    N = X.shape[0]
    # k-means++ -ish init on the unit sphere
    idx0 = rng.integers(N)
    centers = [X[idx0]]
    for _ in range(1, k):
        d = 1.0 - np.max(np.stack([X @ c for c in centers], axis=1), axis=1)
        d = np.clip(d, 0, None)
        p = d / (d.sum() + 1e-12)
        centers.append(X[rng.choice(N, p=p)])
    C = np.stack(centers, axis=0)
    assign = np.zeros(N, dtype=np.int64)
    for _ in range(iters):
        sims = X @ C.T
        new = np.argmax(sims, axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
        for j in range(k):
            members = X[assign == j]
            if len(members) > 0:
                m = members.mean(axis=0)
                nrm = np.linalg.norm(m)
                C[j] = m / nrm if nrm > 1e-9 else C[j]
    return assign


def clustering_purity(assign: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of points whose cluster's majority TRUE label matches the point.
    1.0 = clusters perfectly recover the pixel-defined categories."""
    total = 0
    for cl in np.unique(assign):
        members = labels[assign == cl]
        if len(members) == 0:
            continue
        vals, counts = np.unique(members, return_counts=True)
        total += counts.max()
    return total / len(labels)


def rsa_pixel_provenance(images: np.ndarray, codes: np.ndarray) -> float:
    """Representational-similarity-analysis correlation (the IT/RSA signature):
    does the CODE similarity structure track the raw-PIXEL similarity structure?

    This is the strongest, LABEL-FREE form of anti-cheat 1: we never use the
    category labels at all.  We correlate the off-diagonal of the image-pixel
    cosine matrix with the off-diagonal of the perception-code cosine matrix.
    A high positive correlation means similar-looking shapes get similar codes
    -- i.e. the code's similarity structure comes from the VISUAL FEATURES,
    not from any injected category signal (Op de Beeck / Kriegeskorte RSA).
    """
    N = images.shape[0]
    pix = images.reshape(N, -1).astype(np.float32)
    Cpix = _cos_matrix(pix)
    Ccode = _cos_matrix(codes)
    iu = np.triu_indices(N, k=1)        # off-diagonal upper triangle
    a = Cpix[iu]
    b = Ccode[iu]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def random_partition_null(codes: np.ndarray, labels: np.ndarray,
                          rng: np.random.Generator, n_draws: int = 500):
    """Monte-Carlo label-independence null for the within>between margin.

    The within/between margin is a function of the GROUPING of objects.  If the
    code's structure followed an INJECTED LABEL rather than the pixels, then
    re-grouping the objects by a RANDOM partition (of the same block sizes,
    ignoring pixels) would preserve the margin.  If instead the structure is in
    the PIXELS (so the only high-margin grouping is the true pixel grouping),
    a random partition that MIXES same-pixel objects into different groups must
    DROP the margin toward 0.

    We draw `n_draws` random permutations of the labels (which shuffles which
    object is in which group while keeping group sizes) and report the null
    margin distribution.  (A single random shuffle has a documented small-N
    upward bias -- a few same-pixel items co-locate by chance; averaging over
    many draws quantifies + controls that bias.)  The decisive statistic is the
    GAP: true margin minus the null mean, and how many SDs above the null the
    true margin sits.
    """
    margins = np.empty(n_draws, dtype=np.float64)
    base = labels.copy()
    for i in range(n_draws):
        perm = rng.permutation(base)
        _, _, m = within_between_margin(codes, perm)
        margins[i] = m
    return dict(
        null_mean=float(margins.mean()),
        null_std=float(margins.std()),
        null_max=float(margins.max()),
        null_p95=float(np.percentile(margins, 95)),
    )


def run_seed(seed: int, n_categories: int, n_exemplars: int) -> dict:
    rng = np.random.default_rng(seed)

    # --- build the visual shapes (similarity in pixels only) ---
    images, labels, meta = build_shape_set(n_categories, n_exemplars, rng)
    N = images.shape[0]

    # --- the REAL Gabor/V1 front end ---
    W = build_gabor_response_matrix()
    v1 = encode_v1(images, W)               # (N, n_v1_simple) = (N, 8192)
    it_like = pool_v1_to_complex(v1)        # (N, n_v1_complex) = (N, 2048) "IT-like" pooled

    # --- similarity signature on the perception codes ---
    v1_within, v1_between, v1_margin = within_between_margin(v1, labels)
    it_within, it_between, it_margin = within_between_margin(it_like, labels)

    # --- ANTI-CHEAT 2: flat-distinct baseline (current nav regime) ---
    # Each object gets its OWN orthogonal band (no category sharing) -> the
    # honest flat regime.  margin must be ~0.
    # size the band layout so each object's non-overlapping band always fits:
    # n_neurons = N*64, n_active = 16 (< stride = 64) for any N.
    flat_n_neurons = N * 64
    flat_sparsity = 16.0 / flat_n_neurons
    flat_codes = np.stack([
        orthogonal_drive_pattern(i, n_cues=N, n_neurons=flat_n_neurons,
                                 sparsity=flat_sparsity)
        for i in range(N)
    ], axis=0).astype(np.float32)
    flat_within, flat_between, flat_margin = within_between_margin(flat_codes, labels)

    # --- ANTI-CHEAT 1a: cluster codes by their OWN geometry; do clusters
    #     recover the PIXEL groups? (structure follows pixels) ---
    assign_it = _kmeans_cosine(it_like, n_categories, np.random.default_rng(seed + 7))
    purity_it = clustering_purity(assign_it, labels)
    assign_v1 = _kmeans_cosine(v1, n_categories, np.random.default_rng(seed + 7))
    purity_v1 = clustering_purity(assign_v1, labels)

    # --- ANTI-CHEAT 1b: random-partition label-independence null. If the
    #     structure followed an injected LABEL, a random re-grouping (same group
    #     sizes) would keep the margin; since it lives in the PIXELS, mixing
    #     same-pixel objects into different groups DROPS the margin. Decisive
    #     statistic = true margin's gap above (and SDs above) the null.
    it_null = random_partition_null(it_like, labels,
                                    np.random.default_rng(seed + 13))
    it_null_gap = it_margin - it_null["null_mean"]
    it_null_sds = (it_margin - it_null["null_mean"]) / (it_null["null_std"] + 1e-9)

    # --- ANTI-CHEAT 1c: RSA pixel provenance (LABEL-FREE, the strongest form).
    #     Does code similarity track raw-pixel similarity? High corr => the
    #     code's structure comes from the VISUAL FEATURES (never uses labels).
    rsa_it = rsa_pixel_provenance(images, it_like)
    rsa_v1 = rsa_pixel_provenance(images, v1)

    # per-seed verdict (use the IT-like pooled code as the headline perception code)
    if it_margin >= 0.15:
        verdict = "GO"
    elif it_margin >= 0.05:
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"

    return dict(
        seed=seed, n_categories=n_categories, n_exemplars=n_exemplars, N=N,
        v1=dict(within=round(v1_within, 4), between=round(v1_between, 4),
                margin=round(v1_margin, 4),
                cluster_purity=round(purity_v1, 4),
                rsa_pixel_provenance=round(rsa_v1, 4)),
        it_like=dict(within=round(it_within, 4), between=round(it_between, 4),
                     margin=round(it_margin, 4),
                     null_mean=round(it_null["null_mean"], 4),
                     null_std=round(it_null["null_std"], 4),
                     null_p95=round(it_null["null_p95"], 4),
                     null_gap=round(it_null_gap, 4),
                     null_sds_above=round(it_null_sds, 2),
                     cluster_purity=round(purity_it, 4),
                     rsa_pixel_provenance=round(rsa_it, 4)),
        flat_baseline=dict(within=round(flat_within, 4),
                           between=round(flat_between, 4),
                           margin=round(flat_margin, 4)),
        verdict=verdict,
        sample_meta=meta[:n_exemplars + 1],   # a couple construction records
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-categories", type=int, default=4)
    ap.add_argument("--n-exemplars", type=int, default=4)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_genfrontier_optionB_visual_similarity.json")
    args = ap.parse_args()

    per_seed = [run_seed(s, args.n_categories, args.n_exemplars) for s in args.seeds]

    it_margins = [r["it_like"]["margin"] for r in per_seed]
    v1_margins = [r["v1"]["margin"] for r in per_seed]
    flat_margins = [r["flat_baseline"]["margin"] for r in per_seed]
    it_purities = [r["it_like"]["cluster_purity"] for r in per_seed]
    it_null_gap = [r["it_like"]["null_gap"] for r in per_seed]
    it_null_sds = [r["it_like"]["null_sds_above"] for r in per_seed]
    it_rsa = [r["it_like"]["rsa_pixel_provenance"] for r in per_seed]
    verdicts = [r["verdict"] for r in per_seed]

    # overall GATE: all seeds GO on the IT-like margin, AND the structure follows
    # pixels (cluster purity high + RSA pixel-provenance high + the true margin
    # sits well above the random-partition label-independence null), AND >> flat.
    all_go = all(v == "GO" for v in verdicts)
    purity_ok = min(it_purities) >= 0.75              # clusters recover pixel groups
    rsa_ok = min(it_rsa) >= 0.5                        # code-sim tracks pixel-sim (label-free)
    null_ok = (min(it_null_gap) >= 0.15) and (min(it_null_sds) >= 3.0)  # margin >> null
    flat_ok = max(flat_margins) <= 0.02               # flat baseline ~ 0
    if all_go and purity_ok and rsa_ok and null_ok and flat_ok:
        overall = "GO"
    elif all(v in ("GO", "PARTIAL") for v in verdicts) and min(it_margins) >= 0.05:
        overall = "PARTIAL"
    else:
        overall = "NEGATIVE"

    summary = dict(
        overall_verdict=overall,
        seeds=args.seeds,
        it_like_margin_mean=round(float(np.mean(it_margins)), 4),
        it_like_margin_min=round(float(np.min(it_margins)), 4),
        v1_margin_mean=round(float(np.mean(v1_margins)), 4),
        flat_baseline_margin_mean=round(float(np.mean(flat_margins)), 4),
        it_cluster_purity_min=round(float(np.min(it_purities)), 4),
        it_rsa_pixel_provenance_min=round(float(np.min(it_rsa)), 4),
        it_null_gap_min=round(float(np.min(it_null_gap)), 4),
        it_null_sds_above_min=round(float(np.min(it_null_sds)), 2),
        anti_cheats=dict(
            structure_follows_pixels_cluster_purity_ok=bool(purity_ok),
            structure_follows_pixels_rsa_provenance_ok=bool(rsa_ok),
            true_margin_above_label_null_ok=bool(null_ok),
            flat_baseline_near_zero_ok=bool(flat_ok),
        ),
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
