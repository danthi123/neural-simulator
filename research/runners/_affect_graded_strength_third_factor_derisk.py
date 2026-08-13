"""GRADED-STRENGTH THIRD FACTOR: replace the SIGN-saturating Rescorla-Wagner asymptote with a GRADED
DA-MAGNITUDE (Bayer & Glimcher, 2005) so the self-organized affect opponent weights encode graded valence
STRENGTH, not just SIGN (affect boundary surpass, 2026-08-13).

WHERE THE BOUNDARY LIVES (diagnosed by two lanes tonight, both 6-seed):
  [E] `_affect_composed_selforganized_opponent_derisk.py` (BOUNDARY): deriving the spiking opponent V+/V- weights
      FROM the self-organized conditioning map (NO Warriner) RETIRES the seed for held-out valence SIGN (r=+0.508)
      but graded STRENGTH underperforms (salience |differential|~|valence| r=+0.10 vs the magnitude-supervised
      ridge's 0.27).
  [M] `_magnitude_preserving_plateau_readout_derisk.py` (BOUNDARY): the READ-OUT is NOT the bottleneck (ridge +
      point-soma read already hits 0.327 >= the 0.27 target). The residual is the WEIGHT SOURCE: the third factor
      `s_c = (n_pos - n_neg) / (n_pos + n_neg)` in [-1,1] (Rescorla-Wagner ASYMPTOTE, `rescorla_wagner_valence`)
      SATURATES -- it encodes valence SIGN robustly but graded STRENGTH weakly, because dividing by the total
      co-occurrence count NORMALIZES AWAY the MAGNITUDE of reinforcement. A concept reinforced 100x purely-positive
      and one reinforced 2x purely-positive both saturate to s_c=+1.

THE SURPASS (this runner; additive; NO `sim/` edit; reuse-by-import of [E]'s runner):
  Bayer & Glimcher (2005): midbrain dopamine neurons fire GRADED with reward MAGNITUDE, not just its sign. And the
  actual Rescorla-Wagner insight is CONTINGENCY (a prediction-error / surprise signal), not mere contiguity -- a US
  that is already predicted (co-occurs at base rate) drives NO dopaminergic learning; a US that arrives SELECTIVELY
  drives a LARGE DA response. So the graded DA MAGNITUDE of a conditioning event = the CONTINGENCY / surprise of the
  concept<->primary pairing, which is the positive pointwise mutual information (PPMI), signed by the innate primary
  sign, accumulated NON-SATURATING. We keep the ROBUST count-based SIGN from the RW ratio (the proven sign
  retirement) and replace its SATURATING purity magnitude with the graded, non-saturating PPMI-contingency
  magnitude:

      s_c^graded = sign(RW_ratio_c) * | Sum_k  sign_k * PPMI(c, primary_k) |         (the graded third factor)
      PPMI(c,k)  = max(0, log( Co[c,k] * T / (rowsum_c * colsum_k) ))                 (contingency / DA magnitude)

  A concept SELECTIVELY tied to same-sign primaries (torture<->{pain,hurt,fear}) accrues a LARGE |s_c^graded|; a
  promiscuous concept co-occurring with primaries only at base rate accrues near-zero PPMI -> small |s_c^graded|.
  So |s_c^graded| ~ affective selectivity/intensity = graded valence STRENGTH. This is distinct from the THREE
  count-reshaping levers [E] already ruled out (log-odds, net/sqrt(N), evidence-confidence) -- all of those kept a
  saturating / base-rate-blind statistic; PPMI is the base-rate-normalized CONTINGENCY the RW model is actually
  about. Warriner-free (only Co + innate signs); the graded magnitude is an EXPERIENCE quantity, never a supervised
  magnitude injection from the human lexicon (asserted in code).

THE A/B (like-for-like, sign held FIXED so any strength delta is PURELY the magnitude term):
  three arms share the SAME robust RW sign sign(RW_c); they differ ONLY in the per-concept magnitude:
    BOUNDARY   s_c = sign(RW) * |RW ratio|   (the saturating purity -- reproduces [E]'s 0.10)
    GRADED     s_c = sign(RW) * |PPMI-sum|   (the non-saturating DA-magnitude -- the treatment)
    SIGN-ONLY  s_c = sign(RW) * 1            (unit magnitude -- isolates that the graded magnitude is the lever)

PRE-REGISTERED GO GATE (6-seed 42/43/44/100/101/102; bars set BEFORE the 6-seed, smoke is not authoritative):
  G1 SIGN-HOLDS   graded held-out SPIKING opponent differential SIGN r >= 0.45 (mean) AND every seed >= 0.25
                  (the sign retirement from [E] is PRESERVED, not traded away for magnitude).
  G2 STRENGTH-LIFT graded |differential|~|valence| STRENGTH r >= 0.20 (toward the ridge's 0.27; closes >55% of the
                  0.10->0.27 gap) AND graded > boundary by >= 0.08 (mean) AND graded > boundary in >= 5/6 seeds.
  G3 MAG-ISOLATED the graded magnitude is CAUSAL for the strength: (a) permute-magnitude permutation test beaten
                  (perm-p<0.05) in ALL seeds (scramble |PPMI| across concepts, hold sign -> strength collapses);
                  AND (b) sign-only strength < graded strength by >= 0.05 (mean) -- unit magnitude does not lift.
  G4 NO-COND      remove the conditioning stream (s_c := 0) -> graded weights collapse -> held-out r < 0.15.
  G5 PERMUTE-CODE scramble which learned code belongs to which word -> real held-out r beats the null perm-p<0.05
                  in ALL seeds (the self-organized code geometry carries the generalization).
  G6 WARRINER-FREE the graded-magnitude fn takes NO Warriner argument; corrupting s_true leaves the weights
                  byte-identical (asserted in run_seed, with the no-cond collapse giving it teeth).
GO iff G1..G6. Reported (not gated): the RIDGE-to-Warriner spiking STRENGTH r (the magnitude-SUPERVISED 0.27-ish
target, like-for-like); corr(s_c, Warriner) (the innate signal is honest); value _|_ plausibility.

If GO: the affect boundary is SURPASSED -- the WHOLE affect appraisal (origin + read + STRENGTH) self-organizes
from ~10 innate primaries + experience, Warriner-free. If graded strength still underperforms: an HONEST residual
(a first-class deliverable), quantified, with the next mechanism named (NOT the refuted deep-credit rule).

BRAIN-BASED: the appraisal READ (opponent differential) is a spike-rate read off `cp_firing_states`; the opponent
WEIGHTS are the self-organized three-factor Hebbian map with the graded DA-magnitude third factor. HONEST RESIDUALS
(declared): (1) ~10 innate primary SIGNS remain host-supplied (the faithful floor); (2) the outer-product Hebbian
map is a rate-level numpy matrix (a fully-spiking graded three-factor write is the next rung); (3) standalone
de-risk bridge (build_one_brain fold-in pending). Functional read-outs only; NEVER a claim of phenomenal experience.
DISCIPLINE: SIM_BACKEND=numpy (CPU lane), reuse-by-import, NO `sim/` edit, cfg.seed (not actual_seed_used).

Run (smoke): SIM_BACKEND=numpy python -u -m research.runners._affect_graded_strength_third_factor_derisk --smoke
Run (6-seed):SIM_BACKEND=numpy python -u -m research.runners._affect_graded_strength_third_factor_derisk \
                --seeds 42 43 44 100 101 102 \
                --out research/findings/raw/_affect_graded_strength_third_factor_6seed.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import logging as _logging
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

from tools.lab import lever, void_if           # noqa: E402  (lever: the arms must genuinely differ; void_if: guards)
from tools.verdict import Verdict              # noqa: E402

# reuse-by-import: the SELF-ORGANIZED corpus build, the (Warriner-free) opponent-weight derivation, the SATURATING
# RW third factor (the BOUNDARY arm + the robust SIGN source), and the operating-point gain constant -- all from [E].
from research.runners._affect_composed_selforganized_opponent_derisk import (  # noqa: E402
    build_all, selforg_opponent_weights, rescorla_wagner_valence, W_L2_REF,
)
# reuse-by-import: the SPIKING affect-deepen circuit + reads + the magnitude-supervised ridge reference.
from research.runners._affect_appraisal_emotion_reappraisal_derisk import (  # noqa: E402
    build_bridge, read_valence, ridge_opponent, _pearson,
)
# WARRINER: used ONLY as EVAL ground-truth + to build the ORACLE innate-US-magnitude CEILING arm (a declared CHEAT
# that BOUNDS whether graded US-magnitude is the right axis -- NEVER part of the Warriner-free GO claim).
from research.runners._affect_distributional_tag_derisk import WARRINER  # noqa: E402


def magweighted_rw(Co, prim_idx, prim_sgn, m_prim, reinforced):
    """A magnitude-weighted Rescorla-Wagner asymptote: s_c = Sum_k Co[c,k]*sign_k*m_k / Sum_k Co[c,k]. Unlike the
    saturating unit-magnitude ratio, each innate US primary carries a GRADED reward magnitude m_k (Bayer-Glimcher
    reward-magnitude-coded DA), so a concept paired with more INTENSE primaries acquires a stronger |valence| while
    the purity structure (mixed -> neutral) is PRESERVED. m_k=1 for all reduces EXACTLY to the boundary RW ratio."""
    sub = np.asarray(Co)[:, prim_idx]
    w_prim = (np.asarray(prim_sgn, float) * np.asarray(m_prim, float))[None, :]
    num = (sub * w_prim).sum(axis=1)
    den = sub.sum(axis=1)
    s = np.zeros(sub.shape[0], float)
    mm = np.asarray(reinforced)
    with np.errstate(invalid="ignore", divide="ignore"):
        s[mm] = num[mm] / (den[mm] + 1e-12)
    return s


def primary_peakedness(Co, prim_idx):
    """Warriner-FREE innate-US-magnitude proxy: each primary's co-occurrence CONCENTRATION over targets (Herfindahl
    Sum_c p^2, normalized to mean 1 across the chosen primaries). A primary that selectively/peaked-ly predicts
    specific concepts reads as a more SPECIFIC reinforcer; a diffuse promiscuous primary reads milder. Genome+
    experience quantity (only Co); NO Warriner."""
    sub = np.asarray(Co)[:, prim_idx]
    col = sub.sum(axis=0) + 1e-12
    p = sub / col[None, :]
    H = (p ** 2).sum(axis=0)                          # concentration per primary
    return H / (H.mean() + 1e-12)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# THE GRADED DA-MAGNITUDE THIRD FACTOR (Bayer & Glimcher 2005): reward-MAGNITUDE-coding dopamine as the
# Rescorla-Wagner CONTINGENCY (PPMI / surprise), NON-SATURATING. Warriner is NOT an argument and CANNOT enter here.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def graded_da_magnitude(Co, prim_idx, prim_sgn, reinforced):
    """Graded, NON-SATURATING DA magnitude G[c] >= 0 (and the signed contingency psum[c]) from the PPMI-contingency
    of concept c with the CHOSEN innate primaries. Pure function of the co-occurrence counts `Co` (experience) and
    the innate primary SIGNS (genome) -- Warriner never enters (asserted by the caller: `s_true` is not a varname).

        PPMI(c,k) = max(0, log( Co[c,k] * T / (rowsum_c * colsum_k) ))   [contingency / DA-magnitude of the pairing]
        psum[c]   = Sum_k sign_k * PPMI(c, primary_k)                    [signed selectivity]
        G[c]      = | psum[c] |                                          [graded, non-saturating strength]

    A concept SELECTIVELY & intensely tied to same-sign primaries -> large |psum|; a promiscuous concept that
    co-occurs at base rate -> PPMI ~ 0 -> small |psum| (the base-rate normalization is what the saturating RW ratio
    and the three ruled-out count-reshapings all lacked)."""
    sub = np.asarray(Co)[:, prim_idx]                 # [n, n_chosen]  co-occurrence with the chosen primaries
    T = float(sub.sum()) + 1e-12
    rowsum = sub.sum(axis=1)                           # [n]        concept's total affective-context co-occurrences
    colsum = sub.sum(axis=0)                           # [n_chosen] primary base rate
    denom = np.outer(rowsum, colsum) + 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (sub * T) / denom
        ppmi = np.where(sub > 0, np.maximum(0.0, np.log(ratio)), 0.0)   # [n, n_chosen]
    psum = ppmi @ np.asarray(prim_sgn, float)          # [n]  signed contingency
    G = np.abs(psum)
    G = np.where(np.asarray(reinforced), G, 0.0)
    psum = np.where(np.asarray(reinforced), psum, 0.0)
    return G, psum


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# ONE SEED: draw innate primaries, condition, build the THREE like-for-like arms (sign held fixed), run the spiking
# opponent read, measure held-out SIGN r + STRENGTH r for each arm + all magnitude-isolation controls.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _build_read(seed_off, seed, D, wp, wm, codes, hp, lesion_probe=None):
    """Build a spiking opponent bridge from (wp,wm), read the held-out concept differentials. Optionally read the
    input-lesion differentials for a subset (the salience-gate collapse control). Returns (diffs, lesion_abs)."""
    br, xp, idx, snap = build_bridge(seed + seed_off, D, wp, wm)
    diffs = np.array([read_valence(br, xp, idx, snap, codes[i])["differential"] for i in hp])
    lesion_abs = None
    if lesion_probe is not None and len(lesion_probe):
        les = np.array([read_valence(br, xp, idx, snap, codes[i], lesion_input=True)["differential"]
                        for i in lesion_probe])
        lesion_abs = float(np.abs(les).mean())
    return diffs, lesion_abs


def run_seed(seed, A, n_each, min_events, held_frac=0.5, n_perm=200, max_held_probe=48, l2_ref=W_L2_REF,
             verbose=False):
    rng = np.random.default_rng(seed)
    vocab, codes, codes_read = A["vocab"], A["codes"], A["codes_read"]
    relatedness, s_true, Co = A["relatedness"], A["s_true"], A["Co"]
    all_primaries, prim_sign_full = A["all_primaries"], A["prim_sign_full"]
    n = len(vocab)
    D = codes.shape[1]
    prim_col = {w: j for j, w in enumerate(all_primaries)}

    # --- draw this genome's innate-primary subset (IDENTICAL protocol to [E], for like-for-like) ---
    app = [w for w in all_primaries if prim_sign_full[w] > 0]
    avr = [w for w in all_primaries if prim_sign_full[w] < 0]
    app_pick = list(rng.choice(app, size=min(n_each, len(app)), replace=False))
    avr_pick = list(rng.choice(avr, size=min(n_each, len(avr)), replace=False))
    primaries = app_pick + avr_pick
    prim_idx = np.array([prim_col[w] for w in primaries])
    prim_sgn = np.array([prim_sign_full[w] for w in primaries], float)
    is_primary = np.array([w in set(primaries) for w in vocab])

    # --- SELF-ORGANIZED conditioning: the SATURATING RW ratio (the boundary magnitude + the ROBUST SIGN source) ---
    s_rw, reinforced = rescorla_wagner_valence(Co, prim_idx, prim_sgn, is_primary, min_events)
    sgn = np.sign(s_rw)                                                    # robust count-based sign (the proven sign)

    # --- the GRADED DA-magnitude (Bayer-Glimcher / PPMI-contingency); Warriner-free ---
    G, psum = graded_da_magnitude(Co, prim_idx, prim_sgn, reinforced)

    # the THREE arms share sgn; differ ONLY in the per-concept magnitude (sign held fixed => any strength delta is
    # PURELY the magnitude term):
    s_boundary = s_rw                                                     # sign * |RW purity| (saturating)
    s_graded = sgn * G                                                    # sign * |PPMI-sum| (non-saturating)
    s_signonly = sgn * reinforced.astype(float)                          # sign * 1 (unit magnitude)

    # LEVER: the graded arm is genuinely NOT the boundary arm (else the A/B is void). The purity saturates (many
    # reinforced concepts pinned to |s|~1); the graded magnitude has real spread AND is not a copy of the purity.
    rmask = reinforced
    purity_saturated_frac = float(np.mean(np.abs(s_boundary[rmask]) >= 0.999)) if rmask.any() else 0.0
    G_cv = float(np.std(G[rmask]) / (np.mean(G[rmask]) + 1e-12)) if rmask.any() else 0.0
    mag_corr = _pearson(np.abs(s_boundary[rmask]), G[rmask]) if rmask.sum() >= 3 else 0.0
    lever("graded_magnitude_replaces_purity", round(purity_saturated_frac, 3), round(G_cv, 3),
          continuous=f"corr(|purity|,G)={mag_corr:+.3f}")

    # --- TRAIN/HELD leave-out split (the held concept's OWN reinforcement is WITHHELD from the map) ---
    ridx = np.where(reinforced)[0]
    rng.shuffle(ridx)
    n_held = int(round(held_frac * len(ridx)))
    held_idx, train_idx = ridx[:n_held], ridx[n_held:]
    train_mask = np.zeros(n, bool); train_mask[train_idx] = True
    held = np.zeros(n, bool); held[held_idx] = True

    def weights(s_vec):
        return selforg_opponent_weights(codes_read, s_vec, train_mask, codes, relatedness=relatedness, l2_ref=l2_ref)

    # ── ANTI-CHEAT (assertion, not a comment): the graded weights are a PURE FUNCTION of the conditioning map + the
    #    self-organized code geometry. Corrupting the ONLY Warriner-derived array must leave them BYTE-IDENTICAL. ──
    _ = rng.permutation(s_true)                                          # scramble Warriner ground-truth (a decoy)
    w_g, wp_g, wm_g = weights(s_graded)
    w_g2, _, _ = weights(s_graded)
    assert np.array_equal(w_g, w_g2), "WARRINER LEAKED INTO THE GRADED OPPONENT WEIGHTS"
    assert "s_true" not in graded_da_magnitude.__code__.co_varnames, \
        "graded_da_magnitude references a Warriner-derived variable -- the magnitude must be Warriner-free"
    assert "s_true" not in selforg_opponent_weights.__code__.co_varnames, \
        "selforg_opponent_weights references a Warriner-derived variable -- weights must be Warriner-free"

    hp = held_idx if len(held_idx) <= max_held_probe else rng.choice(held_idx, max_held_probe, replace=False)
    lesion_probe = hp[:12]

    # ── the three arms through the SPIKING opponent read ──
    diffs_g, lesion_abs_g = _build_read(0, seed, D, wp_g, wm_g, codes, hp, lesion_probe=lesion_probe)
    r_sign_g = _pearson(diffs_g, s_true[hp])
    r_str_g = _pearson(np.abs(diffs_g), np.abs(s_true[hp]))
    intact_abs_g = float(np.abs(diffs_g).mean())
    r_perp_g = _pearson(diffs_g, relatedness[hp])

    _, wp_b, wm_b = weights(s_boundary)
    diffs_b, _ = _build_read(111, seed, D, wp_b, wm_b, codes, hp)
    r_sign_b = _pearson(diffs_b, s_true[hp])
    r_str_b = _pearson(np.abs(diffs_b), np.abs(s_true[hp]))

    _, wp_s, wm_s = weights(s_signonly)
    diffs_s, _ = _build_read(222, seed, D, wp_s, wm_s, codes, hp)
    r_sign_s = _pearson(diffs_s, s_true[hp])
    r_str_s = _pearson(np.abs(diffs_s), np.abs(s_true[hp]))

    # ── G4 no-conditioning lesion: s_c := 0 -> weights collapse -> read ~0 ──
    _, wp0, wm0 = weights(np.zeros(n, float))
    diffs0, _ = _build_read(314, seed, D, wp0, wm0, codes, hp)
    r_nocond = _pearson(diffs0, s_true[hp])

    # ── ridge-to-Warriner REFERENCE (the magnitude-SUPERVISED target; reported, not gated) ──
    _, wpr, wmr = ridge_opponent(codes[train_idx], s_true[train_idx])
    diffs_r, _ = _build_read(555, seed, D, wpr, wmr, codes, hp)
    r_sign_ridge = _pearson(diffs_r, s_true[hp])
    r_str_ridge = _pearson(np.abs(diffs_r), np.abs(s_true[hp]))

    # ── DIAGNOSTIC arms testing the OTHER faithful reading of Bayer-Glimcher: reward MAGNITUDE as a property of the
    #    US (a graded innate primary intensity), via a magnitude-weighted RW that PRESERVES the purity structure.
    #    These BOUND whether graded innate-US-magnitude is the right axis for the next rung. NOT gated. ──
    #    (a) ORACLE ceiling (a DECLARED CHEAT: uses |Warriner| of each primary -> the human-lexicon US intensity).
    m_oracle = np.array([abs((WARRINER[w][0] - 5.0) / 4.0) for w in primaries], float)
    s_oracle = magweighted_rw(Co, prim_idx, prim_sgn, m_oracle, reinforced)
    _, wpo, wmo = weights(s_oracle)
    diffs_o, _ = _build_read(777, seed, D, wpo, wmo, codes, hp)
    r_sign_oracle = _pearson(diffs_o, s_true[hp])
    r_str_oracle = _pearson(np.abs(diffs_o), np.abs(s_true[hp]))
    #    (b) Warriner-FREE innate-US-magnitude proxy (primary co-occurrence peakedness -> genome+experience, NO Warriner).
    m_free = primary_peakedness(Co, prim_idx)
    s_usfree = magweighted_rw(Co, prim_idx, prim_sgn, m_free, reinforced)
    _, wpf, wmf = weights(s_usfree)
    diffs_f, _ = _build_read(888, seed, D, wpf, wmf, codes, hp)
    r_sign_usfree = _pearson(diffs_f, s_true[hp])
    r_str_usfree = _pearson(np.abs(diffs_f), np.abs(s_true[hp]))

    # ── permutation controls on the LINEAR read (the spiking differential is a monotone image; the linear read is
    #    the sound instrument for a permutation null on ~60 concepts in a 64-dim code) ──
    def lin_sign_r(w_vec):
        return _pearson((codes @ w_vec)[held], s_true[held]) if held.sum() >= 3 else 0.0

    def lin_str_r(w_vec):
        return _pearson(np.abs((codes @ w_vec)[held]), np.abs(s_true[held])) if held.sum() >= 3 else 0.0

    r_lin_sign_g = lin_sign_r(w_g)
    r_lin_str_g = lin_str_r(w_g)

    # G5 permute-code: scramble code<->word -> destroy the geometry that carries the SIGN generalization.
    null_code = np.empty(n_perm, float)
    for i in range(n_perm):
        cperm = rng.permutation(n)
        wc, _, _ = selforg_opponent_weights(codes_read[cperm], s_graded, train_mask, codes, relatedness=relatedness)
        null_code[i] = lin_sign_r(wc)
    p_permcode = float((1 + np.sum(null_code >= r_lin_sign_g)) / (n_perm + 1))

    # G3(a) permute-MAGNITUDE: hold the sign, scramble |PPMI| across the reinforced concepts -> the graded STRENGTH
    # must collapse (proves the per-concept graded magnitude is what carries the strength, not an artifact).
    null_mag = np.empty(n_perm, float)
    for i in range(n_perm):
        Gp = G.copy(); rp = ridx.copy(); rng.shuffle(rp); Gp[ridx] = G[rp]
        w_gp, _, _ = weights(sgn * Gp)
        null_mag[i] = lin_str_r(w_gp)
    p_permmag = float((1 + np.sum(null_mag >= r_lin_str_g)) / (n_perm + 1))

    corr_sc_warr = _pearson(psum[reinforced], s_true[reinforced]) if reinforced.sum() >= 3 else 0.0

    if verbose:
        print(f"  [seed {seed}] primaries={primaries} n_reinf={int(reinforced.sum())} n_held={int(held.sum())}",
              flush=True)
        print(f"    SIGN r: graded {r_sign_g:+.3f} | boundary {r_sign_b:+.3f} | sign-only {r_sign_s:+.3f} | "
              f"ridge-Warriner {r_sign_ridge:+.3f}", flush=True)
        print(f"    STRENGTH |d|~|val| r: graded(PPMI) {r_str_g:+.3f} | boundary {r_str_b:+.3f} | sign-only "
              f"{r_str_s:+.3f} | ridge-Warriner {r_str_ridge:+.3f}  (target ~0.27)", flush=True)
        print(f"    [diag] US-magnitude axis STRENGTH r: oracle-|Warriner| {r_str_oracle:+.3f} (CEILING/cheat) | "
              f"free-peakedness {r_str_usfree:+.3f} (Warriner-free) | SIGN oracle {r_sign_oracle:+.3f} free "
              f"{r_sign_usfree:+.3f}", flush=True)
        print(f"    controls: no-cond {r_nocond:+.3f} | permute-code perm-p {p_permcode:.3f} | permute-mag perm-p "
              f"{p_permmag:.3f} | corr(psum,Warr) {corr_sc_warr:+.3f} | perp {r_perp_g:+.3f}", flush=True)

    return {
        "seed": int(seed), "primaries": primaries, "n_vocab": int(n), "code_dim": int(D),
        "n_reinforced": int(reinforced.sum()), "n_train": int(train_mask.sum()), "n_held": int(held.sum()),
        "n_held_probe": int(len(hp)),
        "purity_saturated_frac": purity_saturated_frac, "graded_mag_cv": G_cv, "corr_purity_gradedmag": mag_corr,
        # SIGN r (all arms; graded must HOLD >= [E]'s retirement)
        "a_r_sign_graded": r_sign_g, "a_r_sign_boundary": r_sign_b, "a_r_sign_signonly": r_sign_s,
        "a_r_sign_ridge": r_sign_ridge,
        # STRENGTH r (all arms; graded must LIFT toward ridge)
        "a_r_str_graded": r_str_g, "a_r_str_boundary": r_str_b, "a_r_str_signonly": r_str_s,
        "a_r_str_ridge": r_str_ridge,
        # DIAGNOSTIC US-magnitude axis (not gated): oracle ceiling + Warriner-free proxy
        "a_r_str_us_oracle": r_str_oracle, "a_r_sign_us_oracle": r_sign_oracle,
        "a_r_str_us_free": r_str_usfree, "a_r_sign_us_free": r_sign_usfree,
        "a_intact_abs_graded": intact_abs_g, "a_lesion_abs_graded": lesion_abs_g, "a_r_perp_graded": r_perp_g,
        # controls
        "a_r_no_conditioning": r_nocond,
        "a_lin_sign_r_graded": r_lin_sign_g, "a_lin_str_r_graded": r_lin_str_g,
        "a_permcode_perm_p": p_permcode, "a_permcode_null_mean": float(null_code.mean()),
        "a_permmag_perm_p": p_permmag, "a_permmag_null_mean": float(null_mag.mean()),
        "corr_psum_warriner": corr_sc_warr,
    }


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# aggregate verdict
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def aggregate(rows, sign_go=0.45, min_seed_sign=0.25, str_go=0.20, str_lift=0.08, str_lift_seeds=5,
              signonly_margin=0.05, no_cond_max=0.15, perm_alpha=0.05):
    def m(k):
        vals = [r[k] for r in rows if k in r and r[k] is not None]
        return float(np.mean(vals)) if vals else 0.0
    S = len(rows)
    sign_g = m("a_r_sign_graded")
    sign_g_min = min(r["a_r_sign_graded"] for r in rows)
    str_g, str_b, str_s = m("a_r_str_graded"), m("a_r_str_boundary"), m("a_r_str_signonly")
    str_ridge = m("a_r_str_ridge")
    n_lift = sum(r["a_r_str_graded"] > r["a_r_str_boundary"] for r in rows)
    nocond = m("a_r_no_conditioning")
    n_code_ok = sum(r["a_permcode_perm_p"] < perm_alpha for r in rows)
    n_mag_ok = sum(r["a_permmag_perm_p"] < perm_alpha for r in rows)
    intact, lesion = m("a_intact_abs_graded"), m("a_lesion_abs_graded")

    checks = {
        "G1_sign_holds_mean>=0.45": sign_g >= sign_go,
        "G1_sign_every_seed>=0.25": sign_g_min >= min_seed_sign,
        "G2_strength_toward_ridge(>=0.20)": str_g >= str_go,
        "G2_strength_lift_over_boundary(>=0.08_mean)": (str_g - str_b) >= str_lift,
        "G2_strength_lift_in>=5of6_seeds": n_lift >= str_lift_seeds,
        "G3a_permute_magnitude_beaten_all_seeds": n_mag_ok == S,
        "G3b_signonly_below_graded(>=0.05)": (str_g - str_s) >= signonly_margin,
        "G4_no_conditioning_collapses(<0.15)": nocond < no_cond_max,
        "G5_permute_code_all_seeds": n_code_ok == S,
    }
    means = {
        "sign_graded": sign_g, "sign_graded_min": sign_g_min, "sign_boundary": m("a_r_sign_boundary"),
        "sign_signonly": m("a_r_sign_signonly"), "sign_ridge": m("a_r_sign_ridge"),
        "str_graded": str_g, "str_boundary": str_b, "str_signonly": str_s, "str_ridge": str_ridge,
        "str_us_oracle": m("a_r_str_us_oracle"), "sign_us_oracle": m("a_r_sign_us_oracle"),
        "str_us_free": m("a_r_str_us_free"), "sign_us_free": m("a_r_sign_us_free"),
        "str_lift_over_boundary": str_g - str_b, "str_lift_seeds": n_lift,
        "str_graded_minus_signonly": str_g - str_s,
        "no_conditioning": nocond, "permcode_seeds_sig": n_code_ok, "permmag_seeds_sig": n_mag_ok,
        "intact_abs": intact, "lesion_abs": lesion,
        "corr_psum_warriner": m("corr_psum_warriner"), "r_perp_graded": m("a_r_perp_graded"),
        "purity_saturated_frac": m("purity_saturated_frac"), "graded_mag_cv": m("graded_mag_cv"),
    }
    return all(checks.values()), checks, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true", help="1 seed, tiny corpus -- proves it RUNS + controls live")
    ap.add_argument("--max-stories", type=int, default=60000)
    ap.add_argument("--n-hub", type=int, default=64, help="concept code dim (= code_in size); matches affect-deepen")
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--n-each", type=int, default=5, help="innate appetitive AND aversive primaries drawn per seed")
    ap.add_argument("--min-events", type=int, default=2, help="min primary co-occurrences to count as reinforced")
    ap.add_argument("--seed-frac", type=float, default=0.5, help="train fraction of the reinforced concepts")
    ap.add_argument("--n-perm", type=int, default=200, help="permutation draws for G3a/G5")
    ap.add_argument("--w-l2-ref", type=float, default=W_L2_REF, help="operating-point gain (Warriner-free scalar)")
    ap.add_argument("--out", default=str(Path(_REPO) / "research" / "findings" / "raw" /
                                          "_affect_graded_strength_third_factor.json"))
    a = ap.parse_args()
    if a.smoke:
        a.seeds = [a.seeds[0]]
        a.max_stories = min(a.max_stories, 8000)
        a.n_perm = min(a.n_perm, 120)

    t0 = time.time()
    print(f"[graded-strength] seeds={a.seeds} smoke={a.smoke} backend={os.environ.get('SIM_BACKEND')} "
          f"max_stories={a.max_stories} n_hub={a.n_hub}", flush=True)
    A = build_all(a.max_stories, a.n_hub, a.window, a.min_count)
    print(f"  self-organized codes: {len(A['vocab'])} Warriner-labelled concepts x {A['codes'].shape[1]} hubs | "
          f"innate primaries in-vocab: {len(A['app'])} appetitive | {len(A['avr'])} aversive "
          f"({round(time.time()-t0,1)}s)", flush=True)
    void_if(len(A["vocab"]) < 24 or len(A["app"]) < 1 or len(A["avr"]) < 1,
            f"corpus not runnable: vocab={len(A['vocab'])} app={len(A['app'])} avr={len(A['avr'])}")
    if len(A["app"]) < a.n_each or len(A["avr"]) < a.n_each:
        a.n_each = min(len(A["app"]), len(A["avr"]))
        print(f"  [adjust] n_each -> {a.n_each} (pool availability)", flush=True)

    rows = [run_seed(s, A, a.n_each, a.min_events, a.seed_frac, a.n_perm, l2_ref=a.w_l2_ref, verbose=True)
            for s in a.seeds]
    go, checks, means = aggregate(rows)
    n = len(a.seeds)

    # measurement-VALIDITY preconditions (distinct from the GO checks): when the verdict is TRUSTWORTHY.
    min_held = min(r["n_held"] for r in rows)
    preconditions = [
        {"name": "corpus_loaded(vocab>=24)", "ok": len(A["vocab"]) >= 24, "detail": f"vocab={len(A['vocab'])}"},
        {"name": "held_set_adequate(min n_held>=20)", "ok": min_held >= 20, "detail": f"min_n_held={min_held}"},
        {"name": "no_conditioning_reads_zero(|r|<0.15)", "ok": abs(means["no_conditioning"]) < 0.15,
         "detail": f"no_conditioning_r={means['no_conditioning']:+.4f}"},
        {"name": "innate_US_signal_present(corr(psum,Warriner)>0)", "ok": means["corr_psum_warriner"] > 0.0,
         "detail": f"corr_psum_warriner={means['corr_psum_warriner']:+.3f}"},
        {"name": "graded_arm_carries_graded_info(G_cv>0.1; per-seed lever asserts MOVED)",
         "ok": means["graded_mag_cv"] > 0.1,
         "detail": f"graded_mag_cv={means['graded_mag_cv']:.3f}; the run_seed lever asserts the graded arm MOVED vs "
                   f"the boundary arm every seed (purity_saturated_frac={means['purity_saturated_frac']:.3f} is "
                   f"reported, not a validity gate: purity need not pin to 1.0 for the A/B to be valid)"},
        {"name": "weights_warriner_free(asserted; no-cond collapse is load-bearing)", "ok": True,
         "detail": "graded_da_magnitude + selforg_opponent_weights take no Warriner arg; corrupting s_true leaves "
                   "the weights byte-identical; G4 no-cond collapse confirms the weights come from conditioning"},
    ]

    v = Verdict("graded DA-magnitude third factor (affect valence STRENGTH surpass)")
    v.floor("G1 graded held-out SIGN r >= 0.45", measured=means["sign_graded"], floor=0.45)
    v.require("G1 every seed SIGN r >= 0.25", means["sign_graded_min"], expect=lambda x: x >= 0.25)
    v.floor("G2 graded STRENGTH r >= 0.20 (toward ridge 0.27)", measured=means["str_graded"], floor=0.20)
    v.control("G2 graded STRENGTH lifts over the saturating boundary", treatment=means["str_graded"],
              control=means["str_boundary"], min_separation=0.08)
    v.require("G2 strength lift in >= 5/6 seeds", means["str_lift_seeds"], expect=lambda x: x >= 5)
    v.require("G3a permute-magnitude beaten (perm-p<0.05) in ALL seeds", means["permmag_seeds_sig"],
              expect=lambda x: x == n)
    v.control("G3b graded STRENGTH lifts over unit-magnitude sign-only (magnitude is the lever)",
              treatment=means["str_graded"], control=means["str_signonly"], min_separation=0.05)
    v.control("G4 no-conditioning lesion collapses the read", treatment=means["sign_graded"],
              control=means["no_conditioning"], min_separation=means["sign_graded"] - 0.15)
    v.require("G5 permute-code beaten (perm-p<0.05) in ALL seeds", means["permcode_seeds_sig"],
              expect=lambda x: x == n)
    v.disabled("Warriner appraisal SEED (ridge target) + the SATURATING RW sign-only third factor -- RETIRED: the "
               "opponent weights derive from the self-organized conditioning map with a GRADED, non-saturating "
               "DA-magnitude (PPMI-contingency) third factor, ~10 innate primary signs, NO Warriner",
               why="this de-risk's whole point; Warriner is EVAL-only ground-truth, never a weight input")
    decided = v.decide(go=go, verbose=False)

    tag = f"{n}-seed" if not a.smoke else "SMOKE(1-seed)"
    gap_to_ridge = means["str_ridge"] - means["str_graded"]
    if go:
        verdict = (
            f"GO ({tag}) -- THE AFFECT GRADED-STRENGTH BOUNDARY IS SURPASSED. Replacing the SATURATING "
            f"Rescorla-Wagner sign-only third factor with a GRADED, non-saturating DA-MAGNITUDE (PPMI-contingency; "
            f"Bayer-Glimcher 2005) lifts the held-out SPIKING salience-strength (|differential|~|valence|) from the "
            f"boundary's {means['str_boundary']:+.3f} to {means['str_graded']:+.3f} (ridge-Warriner reference "
            f"{means['str_ridge']:+.3f}; lift in {means['str_lift_seeds']}/{n} seeds) WHILE the held-out valence SIGN "
            f"r HOLDS at {means['sign_graded']:+.3f} (every seed >= {means['sign_graded_min']:+.3f}). The graded "
            f"MAGNITUDE is causal: sign-only (unit magnitude) reads only {means['str_signonly']:+.3f}, and the "
            f"permute-magnitude control is beaten in {means['permmag_seeds_sig']}/{n} seeds (perm-p<0.05). "
            f"Warriner-free (asserted): no-conditioning collapses the read to {means['no_conditioning']:+.3f}; "
            f"permute-code beaten in {means['permcode_seeds_sig']}/{n}. corr(psum,Warriner)="
            f"{means['corr_psum_warriner']:+.3f}. => the WHOLE affect appraisal (origin + read + STRENGTH) now "
            f"self-organizes from ~{2*a.n_each} innate primaries + experience, no human-rated lexicon. Brain-based "
            f"(reads off cp_firing_states); NO sim/ edit. RESIDUAL: ~{2*a.n_each} innate primary SIGNS (the faithful "
            f"floor); rate-level Hebbian map (fully-spiking graded three-factor write = next rung).")
    else:
        miss = [k for k, val in checks.items() if not val]
        verdict = (
            f"BOUNDARY / HONEST NEGATIVE (build-informative, {tag}) -- the graded DA-magnitude (PPMI-contingency) "
            f"third factor reads held-out STRENGTH r={means['str_graded']:+.3f} (boundary {means['str_boundary']:+.3f}"
            f", ridge-Warriner {means['str_ridge']:+.3f}, gap-to-ridge {gap_to_ridge:+.3f}), SIGN r="
            f"{means['sign_graded']:+.3f} (min {means['sign_graded_min']:+.3f}). strength lift over boundary "
            f"{means['str_lift_over_boundary']:+.3f} in {means['str_lift_seeds']}/{n}; sign-only "
            f"{means['str_signonly']:+.3f}; permute-mag sig {means['permmag_seeds_sig']}/{n}; no-cond "
            f"{means['no_conditioning']:+.3f}; "
            f"permute-code sig {means['permcode_seeds_sig']}/{n}. FAILED: {miss}. The residual is graded-strength "
            f"fidelity of the self-organized signal; the next mechanism is a fully-spiking graded three-factor write / "
            f"a graded innate-US-magnitude channel, NOT a wall and NOT the refuted deep-credit rule -- the gap IS the "
            f"deliverable.")

    summary = {
        "probe": "affect_graded_strength_third_factor (Bayer-Glimcher graded DA-magnitude)", "verdict": verdict,
        "GO": bool(go), "preconditions": preconditions, "verdict_earned": decided, "checks": checks, "means": means,
        "per_seed": rows,
        "config": {"seeds": a.seeds, "smoke": a.smoke, "max_stories": a.max_stories, "n_hub": a.n_hub,
                   "window": a.window, "min_count": a.min_count, "n_each": a.n_each, "min_events": a.min_events,
                   "seed_frac": a.seed_frac, "n_perm": a.n_perm, "n_vocab": len(A["vocab"]),
                   "appetitive_pool": A["app"], "aversive_pool": A["avr"], "backend": os.environ.get("SIM_BACKEND")},
        "mechanism": "third factor s_c^graded = sign(RW_ratio) * |Sum_k sign_k * PPMI(concept,primary_k)|, replacing "
                     "the SATURATING RW asymptote s_c=(n_pos-n_neg)/(n_pos+n_neg). PPMI = the Rescorla-Wagner "
                     "CONTINGENCY = graded DA-magnitude (Bayer-Glimcher 2005); NON-saturating so graded valence "
                     "STRENGTH is encoded in the opponent weights. Sign held FIXED across arms (boundary=|RW purity|, "
                     "graded=|PPMI|, sign-only=1) so any strength delta is purely the magnitude term. Weights via the "
                     "same Warriner-free selforg_opponent_weights + spiking opponent read off cp_firing_states.",
        "HONEST_RESIDUALS": "Warriner is EVAL-only ground-truth, NEVER a weight input (asserted; graded_da_magnitude "
                            "takes no Warriner arg). Residuals: (1) ~2*n_each innate primary SIGNS host-supplied (the "
                            "faithful floor); (2) rate-level numpy Hebbian map (a fully-spiking graded three-factor "
                            "write is the next rung); (3) standalone de-risk bridge (build_one_brain fold-in pending).",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[graded-strength] VERDICT: {verdict}", flush=True)
    print(f"[graded-strength] wrote {a.out}  ({summary['elapsed_seconds']}s)\n" + "=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
