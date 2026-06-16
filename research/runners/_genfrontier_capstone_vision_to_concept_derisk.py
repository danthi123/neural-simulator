"""Generalization CAPSTONE -- STAGE 1 (vision -> concept): close Option B -> A end-to-end on the spiking substrate.

THE TWO GO PIECES THIS INTEGRATES (both 2026-06-16):
  * Option B (`_genfrontier_optionB_visual_similarity_derisk`, GO): object SHAPES with shared visual features,
    encoded through the project's REAL Gabor/V1 front end (`sim.visual_cortex.build_v1_simple_weights`), produce a
    similarity-STRUCTURED perception code (within-cat cos 0.86 vs between-cat 0.08, RSA-to-pixels 0.99). The
    similarity comes from PIXELS, never from an injected label.
  * Graded-propagation (`_genfrontier_graded_propagation_derisk`, GO): a perception->concept(NMDA) bridge where
    rate-Hebbian co-activation LEARNS the convergence and the NMDA concept assembly SPIKES (real cp_firing_states)
    category-correctly for a HELD-OUT cue (cat-acc 0.92, chance 0.25); flat-distinct ~chance, derangement
    collapses, no-confab moat intact.

WHAT THE PRIOR CONVERGENCE DE-RISKS USED (and what THIS replaces):
  The prior runners drove a SYNTHETIC structured perception ensemble (`structured_perception_sets`: a shared
  per-category core + a per-concept unique tail -- the same-category OVERLAP was MANUFACTURED by construction, a
  controlled given). THE CAPSTONE'S JOB: replace that synthetic input with the REAL Option-B vision-derived
  perception code, so the convergence + the NMDA concept-spiking run on GENUINE perception. This closes Option B
  -> A: "perceive a NOVEL object through real vision -> its concept neurons fire for the right category."

THE CONVERSION (the load-bearing new piece -- Gabor/V1 code -> a structure-preserving perception drive):
  Each shape -> render -> the REAL Gabor/V1 front end -> a V1-COMPLEX code (dim n_v1_complex = 8 orient x 16 x 16
  pos = 2048; non-negative real rates).  We set the convergence bridge's PERCEPTION REGION to be exactly those
  n_v1_complex cells (ONE perception neuron per V1-complex feature -- a faithful feature/retinotopic map, NO
  relabeling, NO injected category).  We then convert each shape's real-valued V1-complex code into a SPARSE
  perception DRIVE = its TOP-K most-active V1-complex features (the K strongest oriented-edge columns the shape
  excites).  Same-category shapes excite the SAME orientation/position columns (shared visual features) -> their
  top-K active SETS OVERLAP -> same-category VISUAL overlap becomes same-category PERCEPTION-ENSEMBLE overlap.
  Different categories excite different columns -> disjoint sets.  The conversion uses ONLY each image's own code
  (never the labels), so it cannot inject category structure; it can only PRESERVE the pixel-derived structure.
  We ASSERT this preservation directly: within-category active-set overlap > between-category (else the conversion
  destroyed the Gabor structure -> a localized PARTIAL/NEGATIVE).

  (Why top-K rather than a graded proportional drive: the convergence's rate-Hebbian + NMDA read-out machinery is
  validated on a SPARSE index-addressed perception ensemble -- `perc_sets[j]` = a set of active perception
  indices.  Top-K is the faithful, reuse-by-import conversion of a real graded code into that sparse-ensemble
  interface, and it is exactly the "threshold/rank the code -> the active perception neurons" option.  We report
  the active-set overlap margin so the structure-preservation is auditable.)

THE TEST (Option B -> A, end to end): for a HELD-OUT shape (its concept block NEVER co-activated during training)
-> render -> Gabor/V1 -> top-K perception drive -> run the bridge -> does the NMDA CONCEPT assembly SPIKE (real
cp_firing_states) in the correct semantic CATEGORY >> chance (1/n_cat)?  This is generalization from PIXELS to
SPIKING CONCEPTS through real vision.

GATE (3 seeds 42/43/44, GPU):
  GO       : held-out vision-derived concept-spike category accuracy >> chance, with: the FLAT-distinct perception
             baseline (orthogonal codes, no visual structure) at chance (structure load-bearing); the
             category-derangement control collapsing; the no-confab moat surviving (a visually-novel no-category
             shape -- an unseen orientation/position basis -- does NOT drive confident category concept-spikes).
  PARTIAL/NEGATIVE : the vision-derived perception is too noisy for the convergence (the Gabor code's structure
             does not survive the top-K conversion / the convergence) -- report honestly + localize (the
             conversion K, more n_per, the drive scale, more co-activation).

Reuse-by-import ONLY (sim.visual_cortex Gabor front end + Option B's shape construction/encoding + the
graded-propagation bridge/training/NMDA-read/anti-cheat).  NO sim/ edit.  GPU `SIM_BACKEND=cupy`.
Run:  SIM_BACKEND=cupy python -u -m research.runners._genfrontier_capstone_vision_to_concept_derisk --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# --- reuse-by-import: Option B's REAL Gabor/V1 shape rendering + encoding (no sim/ edit) ---
from research.runners._genfrontier_optionB_visual_similarity_derisk import (  # noqa: E402
    build_shape_set, build_gabor_response_matrix, encode_v1, pool_v1_to_complex,
    within_between_margin, _render_bar_image,
)
from sim.visual_cortex import (  # noqa: E402
    N_ORIENTATIONS, V1_POSITIONS_PER_DIM, RETINA_SIZE,
)
# --- reuse-by-import: the validated graded-propagation bridge + training + NMDA spike read + anti-cheats ---
from research.runners._genfrontier_graded_propagation_derisk import (  # noqa: E402
    build_propagation_bridge, train_convergence, evaluate_arm_spikes,
    read_heldout_spikes,
)
from research.runners._genfrontier_onsubstrate_convergence_derisk import (  # noqa: E402
    N_CAT, N_PER_CAT, F,
)
import math  # noqa: E402

# perception region size = the V1-complex feature dimension (one perception neuron per V1-complex cell).
N_V1_COMPLEX = N_ORIENTATIONS * V1_POSITIONS_PER_DIM * V1_POSITIONS_PER_DIM   # 8 * 16 * 16 = 2048


# ===========================================================================
# The conversion: real Gabor/V1 V1-complex code -> a sparse, structure-PRESERVING perception drive (top-K).
# ===========================================================================
def vision_to_perception_sets(it_like: np.ndarray, top_k: int):
    """Convert each shape's real-valued V1-complex code (it_like[i], length N_V1_COMPLEX, non-negative rates) into
    a SPARSE perception DRIVE = the indices of its TOP-K most-active V1-complex features.

    Returns a list of int index arrays (len F), each the active perception-neuron set for that shape.  ONLY each
    image's own code is used (the labels are never consulted) -- the conversion can only PRESERVE the pixel-derived
    structure, not inject category structure.
    """
    sets = []
    for i in range(it_like.shape[0]):
        code = it_like[i]
        # top-K strongest features (the oriented-edge columns the shape most excites).
        idx = np.argpartition(code, -top_k)[-top_k:]
        # keep only strictly-positive features (a shape with < top_k active cells should not pad with zeros, which
        # would be a constant background every shape shares -> a spurious common-mode overlap).
        idx = idx[code[idx] > 1e-6]
        sets.append(np.sort(idx).astype(np.int64))
    return sets


def _binary_active_matrix(sets, n_perc):
    """sets (list of index arrays) -> (F, n_perc) binary active-set matrix, for the active-set overlap measure."""
    M = np.zeros((len(sets), n_perc), np.float32)
    for i, s in enumerate(sets):
        M[i, s] = 1.0
    return M


def active_set_overlap_margin(sets, n_perc, cat_ids):
    """within-cat vs between-cat mean active-SET overlap (cosine of the binary active-set vectors).  This is THE
    structure-preservation assert: the conversion must keep same-category active sets MORE overlapping than
    between-category (else it destroyed the Gabor similarity).  Returns (within, between, margin)."""
    M = _binary_active_matrix(sets, n_perc)
    return within_between_margin(M, cat_ids)


# ===========================================================================
# Flat-distinct vision baseline: orthogonal (disjoint) perception sets, SAME sizes as the vision sets, but with NO
# visual structure (structure ablation).  Each shape gets its own disjoint block -> between-set overlap == 0.
# ===========================================================================
def flat_distinct_sets_like(sets, n_perc, seed):
    """Disjoint perception sets sized to match the per-shape vision active-set sizes, scattered across the region
    (no category sharing).  The structure-ablation baseline: every shape's perception ensemble is its own block."""
    rng = np.random.default_rng(seed)
    sizes = [len(s) for s in sets]
    total = int(np.sum(sizes))
    assert total <= n_perc, f"flat baseline needs {total} <= {n_perc} perception neurons"
    perm = rng.permutation(n_perc)[:total]
    out, off = [], 0
    for k in sizes:
        out.append(np.sort(perm[off:off + k]).astype(np.int64))
        off += k
    return out


# ===========================================================================
# A visually-novel NO-CATEGORY shape for the moat (an UNSEEN orientation/position basis, distinct from the trained
# categories) -> render -> Gabor/V1 -> top-K.  It should not strongly match any trained category's concept assembly.
# ===========================================================================
def novel_no_category_perc_set(W, top_k, n_categories, rng, image_size=RETINA_SIZE):
    """Render a bar at an orientation/position BETWEEN the trained category bases (an unseen visual basis), encode
    through the real Gabor/V1 front end, top-K.  Distinct from every trained category (whose bases are at
    c/n_categories * pi on a ring) -> drives a different column mixture -> low best-category familiarity."""
    # an orientation halfway between category 0 and 1's bases + a centre off the trained ring (a novel basis).
    base_theta = (0.5 / n_categories) * math.pi                 # between cat-0 (0) and cat-1 (pi/n_cat)
    margin = image_size * 0.10                                  # closer to centre than the trained ring (0.28)
    ang = math.pi * (0.5 / n_categories)
    cx = image_size / 2.0 + margin * math.cos(ang)
    cy = image_size / 2.0 + margin * math.sin(ang)
    theta = base_theta + rng.normal(0.0, math.radians(7.0))
    length = image_size * 0.55 * (1.0 + rng.normal(0.0, 0.08))
    thick = 1.6 * (1.0 + rng.normal(0.0, 0.10))
    img = _render_bar_image(cx, cy, theta, length, thick, rng, image_size=image_size)
    v1 = encode_v1(img[None, ...], W)
    it = pool_v1_to_complex(v1)[0]
    idx = np.argpartition(it, -top_k)[-top_k:]
    idx = idx[it[idx] > 1e-6]
    return np.sort(idx).astype(np.int64)


# ===========================================================================
# One seed, end-to-end: vision -> Gabor/V1 -> top-K perception drive -> convergence -> NMDA concept spikes.
# ===========================================================================
def run_seed(seed, a):
    a.seed_base = seed
    rng = np.random.default_rng(seed)
    cat_ids = np.repeat(np.arange(N_CAT), N_PER_CAT)

    # leakage-free split: hold out 1 concept per category (each held-out has same-cat TRAIN peers) -- mirrors the
    # graded-prop split exactly so the comparison holds.
    rng_split = np.random.default_rng(seed * 31 + 5)
    held_out = [int(rng_split.choice(np.where(cat_ids == c)[0])) for c in range(N_CAT)]
    train = [i for i in range(F) if i not in held_out]
    assert not (set(train) & set(held_out)), "leakage: train and held-out overlap"

    # ---- (1) render shapes (similarity in PIXELS only) + encode through the REAL Gabor/V1 front end ----
    images, labels, meta = build_shape_set(N_CAT, N_PER_CAT, rng, image_size=RETINA_SIZE)
    assert np.array_equal(labels, cat_ids), "shape labels must match the concept category layout"
    W = build_gabor_response_matrix()
    v1 = encode_v1(images, W)                       # (F, n_v1_simple)
    it_like = pool_v1_to_complex(v1)                # (F, N_V1_COMPLEX) -- the IT-like perception code
    assert it_like.shape[1] == N_V1_COMPLEX, (it_like.shape, N_V1_COMPLEX)

    # the Gabor code's OWN similarity structure (Option B's headline measure -- verify it holds for this seed)
    code_within, code_between, code_margin = within_between_margin(it_like, cat_ids)

    # ---- (2) the conversion: V1-complex code -> top-K perception drive; ASSERT structure preserved ----
    vis_sets = vision_to_perception_sets(it_like, a.top_k)
    set_within, set_between, set_margin = active_set_overlap_margin(vis_sets, N_V1_COMPLEX, cat_ids)
    structure_preserved = bool(set_margin > a.min_set_margin)
    print(f"  [seed {seed}] Gabor V1-complex code margin {code_margin:+.3f} (within {code_within:.3f} / between "
          f"{code_between:.3f}) -> top-{a.top_k} active-SET overlap margin {set_margin:+.3f} (within {set_within:.3f} "
          f"/ between {set_between:.3f})  [structure {'PRESERVED' if structure_preserved else 'LOST'}]", flush=True)

    # ---- (3) ARM 1: STRUCTURED vision-derived perception -> convergence -> NMDA concept spikes ----
    b1, pr, cr, rr, cb, rb = build_propagation_bridge(N_V1_COMPLEX, a.n_concept_per, a.n_readout_per, seed, a)
    xp = b1._cp if hasattr(b1, "_cp") else None
    diag = train_convergence(b1, xp, pr, cr, cb, vis_sets, train, a)
    print(f"  [seed {seed}] vision-derived train firing diag (first scene): perc {diag['perc']} conc {diag['conc']}",
          flush=True)
    S = evaluate_arm_spikes(b1, xp, pr, cr, rr, cb, rb, vis_sets, cat_ids, held_out, train, a)
    print(f"  [seed {seed}] VISION held-out: concept spikes/cue {S['concept_spikes_per_cue']:.0f}, readout "
          f"spikes/cue {S['readout_spikes_per_cue']:.0f}", flush=True)

    # ---- MOAT: a visually-novel NO-CATEGORY shape (unseen orientation/position) on the SAME trained bridge ----
    rngm = np.random.default_rng(seed * 41 + 9)
    novel_set = novel_no_category_perc_set(W, a.top_k, N_CAT, rngm)

    def _best_cat_spikes(perc_idx):
        cpb, _, _, _ = read_heldout_spikes(b1, xp, pr, cr, rr, cb, rb, perc_idx, a.perc_scale, a.read_steps)
        return float(np.max([cpb[cat_ids == c].mean() for c in range(N_CAT)]))

    ho_fam = float(np.mean([_best_cat_spikes(vis_sets[j]) for j in held_out]))
    novel_fam = _best_cat_spikes(novel_set)
    moat_ok = bool(ho_fam > novel_fam * 1.5 + 1e-9)        # a learned-category held-out shape is clearly more familiar
    del b1

    # ---- ARM 2: FLAT-distinct vision baseline (same set sizes, NO visual structure) ----
    flat_sets = flat_distinct_sets_like(vis_sets, N_V1_COMPLEX, seed * 19 + 3)
    b2, pr2, cr2, rr2, cb2, rb2 = build_propagation_bridge(N_V1_COMPLEX, a.n_concept_per, a.n_readout_per, seed, a)
    xp2 = b2._cp if hasattr(b2, "_cp") else None
    train_convergence(b2, xp2, pr2, cr2, cb2, flat_sets, train, a)
    Fl = evaluate_arm_spikes(b2, xp2, pr2, cr2, rr2, cb2, rb2, flat_sets, cat_ids, held_out, train, a)
    del b2

    # ---- ARM 3: category-DERANGEMENT control (vision perception, but co-activated with a WRONG-category concept
    # block).  If transfer is the LEARNED vision-category<->concept-category correspondence, held-out lands WRONG. ----
    derange = (np.arange(N_CAT) + 1) % N_CAT
    train_by_cat = {c: [t for t in train if cat_ids[t] == c] for c in range(N_CAT)}
    deranged_block = {}
    for t in train:
        c = int(cat_ids[t]); k = train_by_cat[c].index(t)
        donor_cat = int(derange[c])
        donor = train_by_cat[donor_cat][k % len(train_by_cat[donor_cat])]
        deranged_block[t] = donor
    b3, pr3, cr3, rr3, cb3, rb3 = build_propagation_bridge(N_V1_COMPLEX, a.n_concept_per, a.n_readout_per, seed, a)
    xp3 = b3._cp if hasattr(b3, "_cp") else None
    for ep in range(a.epochs):
        order = np.random.RandomState(seed * 7 + ep).permutation(train)
        for t in order:
            perc_local = np.asarray(vis_sets[t]) - pr3[0]
            conc_local = cb3[deranged_block[t]] - cr3[0]            # WRONG-category concept block
            n_perc_l = pr3.shape[0]; n_conc_l = cr3.shape[0]
            full_perc = np.zeros(n_perc_l, np.float32); full_perc[perc_local] = a.perc_scale
            full_conc = np.zeros(n_conc_l, np.float32); full_conc[conc_local] = a.conc_scale
            b3.cp_external_input_current[:] = 0.0
            b3.cp_external_input_current[pr3] = xp3.asarray(full_perc) if xp3 is not None else full_perc
            b3.cp_external_input_current[cr3] = xp3.asarray(full_conc) if xp3 is not None else full_conc
            for _ in range(a.scene_steps):
                b3._run_one_simulation_step()
    b3.cp_external_input_current[:] = 0.0
    P = evaluate_arm_spikes(b3, xp3, pr3, cr3, rr3, cb3, rb3, vis_sets, cat_ids, held_out, train, a)
    del b3

    chance = 1.0 / N_CAT
    out = {
        "seed": seed, "held_out": held_out, "candidate": a.candidate,
        "gabor_code": {"within": code_within, "between": code_between, "margin": code_margin},
        "active_set": {"within": set_within, "between": set_between, "margin": set_margin,
                       "structure_preserved": structure_preserved},
        "structured": S, "flat": Fl, "permuted": P,
        "moat": {"heldout_familiarity": ho_fam, "novel_familiarity": novel_fam, "moat_ok": moat_ok},
        "sample_meta": meta[:N_PER_CAT + 1],
    }
    print(f"  [seed {seed}] VISION concept-spike cat-acc {S['concept_cat_acc']:.2f} (chance {chance:.2f}) margin "
          f"{S['concept_margin']:+.3f} | FLAT {Fl['concept_cat_acc']:.2f} | PERMUTED {P['concept_cat_acc']:.2f} "
          f"margin {P['concept_margin']:+.3f} | moat {'OK' if moat_ok else 'BREACH'} (ho {ho_fam:.2f} vs novel "
          f"{novel_fam:.2f})", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--candidate", default="nmda", choices=["nmda", "pool", "graded"],
                   help="read-out propagation mechanism (default nmda; passed through to the graded-prop bridge)")
    # the conversion knob
    p.add_argument("--top-k", type=int, default=60, help="top-K most-active V1-complex features = the perception "
                   "drive size per shape (matched to the convergence's validated active-ensemble size ~60).")
    p.add_argument("--min-set-margin", type=float, default=0.05, help="min within-vs-between active-SET overlap "
                   "margin for the structure-preservation assert (else PARTIAL: the conversion lost the structure)")
    # concept / read-out config mirror the graded-prop GO (the documented population-code lift).
    p.add_argument("--n-concept-per", type=int, default=100)
    p.add_argument("--n-readout-per", type=int, default=100)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--scene-steps", type=int, default=16)
    p.add_argument("--read-steps", type=int, default=80)
    p.add_argument("--perc-scale", type=float, default=300.0)
    p.add_argument("--conc-scale", type=float, default=600.0)
    p.add_argument("--read-weight", type=float, default=30.0)
    p.add_argument("--nmda-ratio", type=float, default=2.0)
    p.add_argument("--hebbian-rate", type=float, default=0.05)
    p.add_argument("--hebbian-max", type=float, default=20.0)
    p.add_argument("--out", default="research/findings/raw/_genfrontier_capstone_vision_to_concept.json")
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[genfrontier CAPSTONE stage-1 vision->concept] real shapes -> Gabor/V1 -> top-{a.top_k} perception "
          f"drive -> rate-Hebbian convergence -> NMDA concept SPIKES. Does a HELD-OUT novel shape, perceived "
          f"through real vision, fire its concept neurons in the correct category? candidate={a.candidate} "
          f"seeds={seeds}", flush=True)
    rows = [run_seed(s, a) for s in seeds]
    chance = 1.0 / N_CAT

    def m(arm, k):
        return float(np.mean([r[arm][k] for r in rows]))
    s_cat, s_margin = m("structured", "concept_cat_acc"), m("structured", "concept_margin")
    s_conc_sp = m("structured", "concept_spikes_per_cue")
    s_read_sp = m("structured", "readout_spikes_per_cue")
    s_read_cat = m("structured", "readout_cat_acc")
    f_cat = m("flat", "concept_cat_acc")
    p_cat, p_margin = m("permuted", "concept_cat_acc"), m("permuted", "concept_margin")
    moat_all = all(r["moat"]["moat_ok"] for r in rows)
    set_margin_min = float(np.min([r["active_set"]["margin"] for r in rows]))
    structure_all = all(r["active_set"]["structure_preserved"] for r in rows)
    code_margin_min = float(np.min([r["gabor_code"]["margin"] for r in rows]))

    concept_spikes_present = s_conc_sp > 0.0
    # GO: the structure survives the conversion (active-set margin > 0 every seed), the concept assembly SPIKES and
    # lands category-correctly for a HELD-OUT VISION cue (cat-acc > chance every seed + positive margin), flat
    # ~chance (visual structure load-bearing), the derangement collapses, the no-confab moat survives.
    go = (structure_all
          and concept_spikes_present
          and all(r["structured"]["concept_cat_acc"] > chance + 1e-9 for r in rows)
          and s_margin > 0.005
          and f_cat <= chance + 0.15
          and p_margin <= s_margin - 0.005
          and moat_all)
    partial = (concept_spikes_present and s_cat > chance + 0.10 and s_margin > 0.0 and s_cat > f_cat + 0.10)
    verdict = "GO" if go else ("PARTIAL" if partial else "NEGATIVE")

    print(f"\n{'='*114}\n  MEAN ({len(rows)} seeds) [{a.candidate}]: Gabor code margin (min) {code_margin_min:+.3f} "
          f"-> active-SET margin (min) {set_margin_min:+.3f} [structure {'PRESERVED' if structure_all else 'LOST'}] "
          f"|| concept spikes/cue {s_conc_sp:.0f} -> readout {s_read_sp:.0f} | VISION concept-spike cat-acc "
          f"{s_cat:.2f} (chance {chance:.2f}) margin {s_margin:+.4f} [readout cat-acc {s_read_cat:.2f}] | FLAT "
          f"{f_cat:.2f} | PERMUTED {p_cat:.2f} margin {p_margin:+.4f} | moat {'INTACT' if moat_all else 'BREACH'}  "
          f"==> {verdict}\n{'='*114}", flush=True)
    if verdict == "GO":
        print(f"  GO -- OPTION B -> A CLOSED END-TO-END: a HELD-OUT object, perceived through the REAL Gabor/V1 "
              f"front end, drives its CONCEPT NEURONS to SPIKE ({s_conc_sp:.0f} spikes/cue, real cp_firing_states) "
              f"in the correct semantic CATEGORY ({s_cat:.0%} >> chance {chance:.0%}, margin {s_margin:+.4f}); the "
              f"Gabor similarity structure SURVIVED the top-{a.top_k} conversion (active-set margin {set_margin_min:+.3f}) "
              f"all the way to the concept spikes; the FLAT-distinct (no-visual-structure) baseline is ~chance "
              f"({f_cat:.0%}) => the VISUAL structure is load-bearing; the derangement collapses ({p_margin:+.4f}); "
              f"the no-confab moat survives (a visually-novel no-category shape does not drive confident category "
              f"spikes). Generalization from PIXELS to SPIKING CONCEPTS through real vision. NO sim/ edit.",
              flush=True)
    elif verdict == "PARTIAL":
        print(f"  PARTIAL: vision-derived transfer is above flat ({s_cat:.0%} vs {f_cat:.0%}) but below the GO bar "
              f"-- localize: the conversion (top-K {a.top_k}: structure margin {set_margin_min:+.3f}; "
              f"{'survived' if structure_all else 'LOST -> the Gabor code did not convert to a structured ensemble'}), "
              f"n-concept-per, perc-scale, or epochs.", flush=True)
    else:
        if not structure_all:
            why = (f"the top-{a.top_k} conversion did NOT preserve the Gabor similarity structure (active-set "
                   f"margin min {set_margin_min:+.3f} <= {a.min_set_margin}) -- the vision code's structure is "
                   f"lost at the perception-drive conversion; tune top-K or use a graded proportional drive")
        elif not concept_spikes_present:
            why = "the concept assembly does NOT spike from the vision-derived perception drive (route to graded)"
        else:
            why = (f"concept spikes ({s_conc_sp:.0f}/cue) but the vision-derived transfer is not clean "
                   f"(vision {s_cat:.0%}, flat {f_cat:.0%}, permuted margin {p_margin:+.4f})")
        print(f"  NEGATIVE [{a.candidate}]: {why}. Moat {'INTACT' if moat_all else 'BREACH'}. Honest negative + "
              f"the localized next step.", flush=True)

    os.makedirs(os.path.dirname(os.path.join(_REPO, a.out)), exist_ok=True)
    with open(os.path.join(_REPO, a.out), "w") as fh:
        json.dump({"verdict": verdict, "candidate": a.candidate, "chance": chance, "top_k": a.top_k,
                   "gabor_code_margin_min": code_margin_min, "active_set_margin_min": set_margin_min,
                   "structure_preserved_all": structure_all,
                   "concept_spikes_per_cue": s_conc_sp, "readout_spikes_per_cue": s_read_sp,
                   "vision_concept_cat_acc": s_cat, "vision_concept_margin": s_margin,
                   "vision_readout_cat_acc": s_read_cat, "flat_concept_cat_acc": f_cat,
                   "permuted_concept_cat_acc": p_cat, "permuted_concept_margin": p_margin,
                   "moat_intact": moat_all, "per_seed": rows}, fh, indent=2, default=str)
    print(f"  [saved] {a.out}\n  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    raise SystemExit(0 if verdict == "GO" else (2 if verdict == "PARTIAL" else 1))


if __name__ == "__main__":
    main()
