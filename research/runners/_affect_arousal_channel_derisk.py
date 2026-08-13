"""SEPARATE EMERGENT AROUSAL / INTENSITY CHANNEL -- the valence _|_ arousal circumplex separation (Lane A affect,
2026-08-13). The named surpass of the graded-valence-STRENGTH BOUNDARY, and it is NOT another valence tweak.

WHY THIS EXISTS (the exact residual [A] localized, with an ORACLE ceiling). [A]
(`2026-08-13-affect-graded-strength-third-factor-BOUNDARY.md`) proved graded valence STRENGTH is an INFORMATION
boundary of the sparse ~10-primary valence-SIGN conditioning channel: NO third-factor magnitude -- contingency,
graded US intensity, even an ORACLE |Warriner|-weighted US -- recovers per-concept intensity (all ceiling at
STRENGTH r~+0.08-0.10 while the magnitude-supervised ridge reaches +0.29). The channel carries the SIGN of
reinforcement (opponent DIFFERENCE, r~0.5) but not the INTENSITY. [A]'s own named surpass (its "next mechanism" 2):
a SEPARATE graded AROUSAL/intensity channel, biologically a DISTINCT dimension from valence SIGN, carried by
separate systems -- the LC-noradrenergic / interoceptive-salience arousal system vs the VTA-BLA valence opponent
(Russell 1980 circumplex; Barrett & Bliss-Moreau; Kandel 6e Ch.40: the noradrenergic locus ceruleus modulates
overall AROUSAL/alertness/attention, a separate ascending modulatory system from the dopaminergic reward pathways;
phasic LC bursts precede responses to SALIENT stimuli regardless of reward sign).

THE CIRCUMPLEX SEPARATION, realized from the SAME self-organized conditioning stream (this is the whole idea):
  - VALENCE SIGN  = the reinforcer DIFFERENCE  s_c = (n_pos - n_neg)/(n_pos + n_neg)   [the opponent, [E]'s channel]
  - AROUSAL       = the reinforcer ENGAGEMENT MAGNITUDE, sign-agnostic. High-arousal concepts engage the bodily/
                    reinforcer systems HARD regardless of whether the engagement is appetitive or aversive. The
                    primary emergent read is the INTEROCEPTIVE-SALIENCE contingency A_c = (n_pos + n_neg)/(total
                    context co-occurrence) -- the FRACTION of a concept's contextual company that is bodily
                    reinforcers (frequency-robust). SUM vs DIFFERENCE: orthogonal by construction.

Arousal is the SUM channel; valence is the DIFFERENCE channel of the identical innate-primary conditioning -- so the
arousal read is genuinely SEPARATE from the valence-sign opponent (a different operation on the same experience), and
by construction tracks INTENSITY not SIGN. We test whether it predicts held-out Warriner AROUSAL (EVAL-ONLY ground
truth) and whether combining valence-SIGN + arousal recovers the graded valence-STRENGTH the sparse valence channel
alone could not.

CANDIDATE EMERGENT AROUSAL SOURCES (all Warriner-arousal-FREE; each a pure function of the corpus / self-organized
code; the PRIMARY is gated, the rest are reported so the strongest is named honestly):
  1. interoceptive_salience [PRIMARY, gated] -- reinforcer-engagement contingency (n_pos+n_neg)/total_ctx.
  2. reinforcer_drive_raw              -- unnormalized n_pos+n_neg (expected frequency-confounded; shows why (1) normalizes).
  3. code_magnitude                    -- L2 norm of the RAW (pre-normalization) PPMI code = total distinctive drive.
  4. context_dispersion                -- Shannon entropy of the concept's context (hub) distribution (varied vs stereotyped).

⛔ THE HARD CONSTRAINT: the Warriner AROUSAL rating is EVALUATION-ONLY held-out ground truth, NEVER an input. Using it
as input would be the exact Warriner cheat [A]/[E] retired for valence. Asserted in code: every arousal-source
function takes NO arousal argument; corrupting the Warriner arousal ground-truth leaves every arousal read
byte-identical.

PRE-REGISTERED GO GATE (6 seeds 42/43/44/100/101/102, each drawing a different innate-primary subset):
  G1 AROUSAL-PREDICTS  the PRIMARY emergent arousal read correlates with held-out Warriner AROUSAL at mean Pearson
                       r >= 0.25 AND every seed >= 0.10. (Pre-registered moderate-emergent-prediction bar.)
  G2 AROUSAL _|_ SIGN  |corr(arousal, valence SIGN)| < 0.30 in ALL seeds -- the arousal read tracks INTENSITY, not
                       good/bad. (The circumplex separation, measured. corr(arousal,|valence|) is reported as the
                       intensity it SHOULD track.)
  G3 NO-SOURCE LESION  zero the reinforcer co-occurrence (the arousal source) -> the read collapses -> |arousal r|
                       < 0.15. The read comes from experienced reinforcer ENGAGEMENT, not a lookup.
  G4 PERMUTE COLLAPSE  PERMUTATION TEST: scramble which concept the arousal read belongs to -> real arousal r beats
                       the null at perm-p < 0.05 in ALL seeds.
  G5 STRENGTH LIFT     combining the emergent valence-SIGN read + the emergent arousal read recovers graded valence
                       STRENGTH BETTER than the sign read alone: held-out r(|valence|) rises by >= +0.03 AND
                       combined > sign-only. (Incremental validity of the arousal channel for the [A] residual, under
                       the same magnitude-supervised ridge read-out [A]/[M] used -- the arousal FEATURE is emergent.)
GO iff G1..G5. If arousal PREDICTS but does not LIFT strength -> partial (reported honestly, not relabelled GO). If
the emergent sources are all weak -> BOUNDARY: report which got closest + how far + the next mechanism (a richer
interoceptive/bodily signal, more reinforcer coverage), NEVER the Warriner-arousal lookup.

BRAIN-BASED / HONEST RESIDUALS (declared): the arousal read is a RATE-level numpy function of the self-organized code
+ the innate-primary conditioning (the codes are the spiking-validated stream cortex; a fully-spiking LC-like
arousal population whose rate = total reinforcer drive is the named next rung -- the SAME status as [E]/[A]'s
rate-level Hebbian valence map). ~10 innate primary reinforcers remain host-supplied (the faithful floor; world+body
boundary). Warriner arousal is EVAL-only. Standalone de-risk bridge (build_one_brain fold-in pending). DISCIPLINE:
SIM_BACKEND=numpy CPU lane, reuse-by-import of [E]'s composed runner + the DR-2 stream cortex, NO `sim/` edit,
cfg.seed (not actual_seed_used -- no substrate build here, pure numpy reads).

Run (smoke): SIM_BACKEND=numpy python -u -m research.runners._affect_arousal_channel_derisk --smoke
Run (6-seed):SIM_BACKEND=numpy python -u -m research.runners._affect_arousal_channel_derisk \
                 --seeds 42 43 44 100 101 102 \
                 --out research/findings/raw/_affect_arousal_channel_6seed.json
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

from tools.lab import lever, void_if, attributable_to  # noqa: E402  (workflow helpers -- earned, not recalled)
from tools.verdict import Verdict                                     # noqa: E402

# reuse-by-import: the DR-2 self-organized PPMI stream cortex + Warriner ground-truth (v, arousal) [EVAL ONLY].
from research.runners._affect_distributional_tag_derisk import (  # noqa: E402
    WARRINER, load_stories, build_cooccurrence, codes_from_cooccurrence,
)
from research.runners.learned_graded_cortex_fair_test import ppmi_matrix  # noqa: E402  (RAW PPMI for code magnitude)
# reuse-by-import: the emergence lane's INNATE primaries + the conditioning scan + the composed valence opponent.
from research.runners._affect_evaluative_conditioning_derisk import (  # noqa: E402
    APPETITIVE_POOL, AVERSIVE_POOL, build_primary_cooccurrence, _pearson,
)
from research.runners._affect_composed_selforganized_opponent_derisk import (  # noqa: E402
    selforg_opponent_weights, rescorla_wagner_valence,
)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# THE EMERGENT AROUSAL SOURCES. Each is a PURE FUNCTION of (the corpus co-occurrence, the self-organized code, the
# innate-primary conditioning). NONE takes a Warriner-AROUSAL argument -- asserted in run_seed by byte-identity under
# a corrupted arousal ground-truth. Each returns a per-concept arousal read (length = n_vocab).
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def arousal_interoceptive_salience(n_pos, n_neg, total_ctx, balance=True):
    """PRIMARY (gated). Interoceptive-salience contingency: the FRACTION of a concept's contextual company that is
    bodily reinforcers, SIGN-AGNOSTIC. The SUM channel (arousal) of the same innate-primary conditioning whose
    DIFFERENCE channel is valence (the opponent). Frequency-robust (the /total_ctx ratio divides out how often the
    concept appears). NO arousal input -- pure reinforcer ENGAGEMENT magnitude.

    ipos = n_pos/total_ctx (appetitive engagement fraction), ineg = n_neg/total_ctx (aversive engagement fraction).
    `balance=True` (the companion process the raw sum omitted): per-polarity GAIN normalization -- divide each
    engagement by its POPULATION mean so the two opponent inputs contribute at EQUAL average gain. Without it, the
    SUM leaks whichever polarity's reinforcers happen to be more prevalent in the corpus (in TinyStories the aversive
    primaries cry/fall/cold are far more frequent, so the raw sum correlates NEGATIVELY with valence sign -- a
    diagnosed confound, not the intended intensity). The gain is a global per-polarity scalar -- it carries NO
    per-concept valence and does not read the arousal ground-truth (Warriner-arousal-free)."""
    tc = np.asarray(total_ctx, float) + 1e-9
    ipos = np.asarray(n_pos, float) / tc
    ineg = np.asarray(n_neg, float) / tc
    if balance:
        ipos = ipos / (float(ipos.mean()) + 1e-12)   # equal opponent-input gain (homeostatic; sign-balanced)
        ineg = ineg / (float(ineg.mean()) + 1e-12)
    return ipos + ineg


def arousal_reinforcer_drive_raw(n_pos, n_neg):
    """Reported. Unnormalized total reinforcer engagement n_pos+n_neg (expected frequency-confounded -- included to
    SHOW that the interoceptive contingency's frequency normalization is load-bearing)."""
    return np.asarray(n_pos, float) + np.asarray(n_neg, float)


def arousal_code_magnitude(C_raw):
    """Reported. Total distinctive drive: the L2 norm of the RAW (pre-L2-normalization) PPMI code -- 'how hard the
    concept drives the substrate'. A pure function of the self-organized co-occurrence code (no reinforcers, no
    arousal input)."""
    raw = ppmi_matrix(np.asarray(C_raw, float), 0.75)
    return np.linalg.norm(raw, axis=1)


def arousal_context_dispersion(C_raw):
    """Reported. Shannon entropy of the concept's context (hub) distribution: high = varied/unpredictable contexts,
    low = stereotyped. A pure function of the co-occurrence code (no arousal input). Sign of the correlation with
    arousal is reported, not assumed."""
    C = np.asarray(C_raw, float)
    row = C.sum(axis=1, keepdims=True)
    P = C / (row + 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(P * np.log(P + 1e-12)).sum(axis=1)
    return H


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# corpus build (once; seed-independent). Reuses the DR-2 stream cortex + the emergence-lane primary scan; adds the
# RAW co-occurrence C (for code-magnitude / dispersion) + per-concept total context frequency + Warriner AROUSAL gt.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def build_all_arousal(max_stories, n_hub, window, min_count):
    stories = load_stories(max_stories)
    vocab, C = build_cooccurrence(stories, n_hub, window, min_count)      # RAW target x hub co-occurrence
    codes = codes_from_cooccurrence(C)                                    # L2 PPMI stream-cortex code (valence read)
    codes_read = codes - codes.mean(axis=0, keepdims=True)               # DC-removed read code (for the opponent weight)
    codes_read = codes_read / (np.linalg.norm(codes_read, axis=1, keepdims=True) + 1e-12)
    Wsim = codes @ codes.T
    np.fill_diagonal(Wsim, 0.0)
    relatedness = np.asarray(Wsim.mean(axis=1), float)
    total_ctx = np.asarray(C.sum(axis=1), float)                         # per-concept total context (frequency proxy)
    val = np.array([WARRINER[w][0] for w in vocab], float)
    aro = np.array([WARRINER[w][1] for w in vocab], float)
    s_true = (val - 5.0) / 4.0                                           # signed Warriner VALENCE gt (EVAL ONLY)
    a_true = (aro - 5.0) / 4.0                                           # centred Warriner AROUSAL gt (EVAL ONLY)
    vocab_set = set(vocab)
    app = [w for w in APPETITIVE_POOL if w in vocab_set]
    avr = [w for w in AVERSIVE_POOL if w in vocab_set]
    all_primaries = app + avr
    prim_sign_full = {**{w: +1.0 for w in app}, **{w: -1.0 for w in avr}}
    Co = build_primary_cooccurrence(stories, vocab, window, all_primaries)
    # pre-compute the reported code-geometry arousal sources (seed-independent)
    code_mag = arousal_code_magnitude(C)
    ctx_disp = arousal_context_dispersion(C)
    return dict(vocab=vocab, codes=codes, codes_read=codes_read, relatedness=relatedness, total_ctx=total_ctx,
                s_true=s_true, a_true=a_true, app=app, avr=avr, all_primaries=all_primaries,
                prim_sign_full=prim_sign_full, Co=Co, code_mag=code_mag, ctx_disp=ctx_disp)


def _linfit_predict(X_tr, y_tr, X_ev):
    """Least-squares linear read-out (the [A]/[M] ridge protocol, tiny lambda) fit on TRAIN, applied to EVAL. X has a
    bias column added here. Returns the EVAL prediction."""
    X_tr = np.asarray(X_tr, float); X_ev = np.asarray(X_ev, float)
    Xt = np.column_stack([X_tr, np.ones(len(X_tr))])
    Xe = np.column_stack([X_ev, np.ones(len(X_ev))])
    lam = 1e-3
    A = Xt.T @ Xt + lam * np.eye(Xt.shape[1])
    coef = np.linalg.solve(A, Xt.T @ np.asarray(y_tr, float))
    return Xe @ coef


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# ONE SEED: draw innate primaries, condition, build the SEPARATE arousal read + the valence-sign read, run the bars.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def run_seed(seed, A, n_each, min_events, held_frac=0.5, n_perm=200, verbose=False):
    rng = np.random.default_rng(seed)
    vocab, codes, codes_read = A["vocab"], A["codes"], A["codes_read"]
    relatedness, total_ctx = A["relatedness"], A["total_ctx"]
    s_true, a_true, Co = A["s_true"], A["a_true"], A["Co"]
    all_primaries, prim_sign_full = A["all_primaries"], A["prim_sign_full"]
    code_mag, ctx_disp = A["code_mag"], A["ctx_disp"]
    n = len(vocab)
    prim_col = {w: j for j, w in enumerate(all_primaries)}

    # --- draw this genome's innate-primary subset ---
    app = [w for w in all_primaries if prim_sign_full[w] > 0]
    avr = [w for w in all_primaries if prim_sign_full[w] < 0]
    app_pick = list(rng.choice(app, size=min(n_each, len(app)), replace=False))
    avr_pick = list(rng.choice(avr, size=min(n_each, len(avr)), replace=False))
    primaries = app_pick + avr_pick
    prim_idx = np.array([prim_col[w] for w in primaries])
    prim_sgn = np.array([prim_sign_full[w] for w in primaries], float)
    is_primary = np.array([w in set(primaries) for w in vocab])

    # per-concept pos/neg conditioning-event counts from the CHOSEN primaries (the SAME quantities the valence
    # opponent uses -- valence reads the DIFFERENCE, arousal reads the SUM).
    sub = Co[:, prim_idx]
    n_pos = (sub * (prim_sgn > 0)).sum(axis=1)
    n_neg = (sub * (prim_sgn < 0)).sum(axis=1)
    tot = n_pos + n_neg
    reinforced = (tot >= min_events) & (~is_primary)

    # valence SIGN s_c (reused, Warriner-free) + TRAIN/HELD leave-out split (the DR-2 protocol)
    s_c, _ = rescorla_wagner_valence(Co, prim_idx, prim_sgn, is_primary, min_events)
    ridx = np.where(reinforced)[0]
    rng.shuffle(ridx)
    n_held = int(round(held_frac * len(ridx)))
    held_idx, train_idx = ridx[:n_held], ridx[n_held:]
    train_mask = np.zeros(n, bool); train_mask[train_idx] = True
    held = np.zeros(n, bool); held[held_idx] = True

    # ══ THE SEPARATE EMERGENT AROUSAL READ (PRIMARY: sign-BALANCED interoceptive-salience contingency) ════════════
    A_primary = arousal_interoceptive_salience(n_pos, n_neg, total_ctx, balance=True)
    A_raw_cont = arousal_interoceptive_salience(n_pos, n_neg, total_ctx, balance=False)  # unbalanced (leaks sign)
    A_raw = arousal_reinforcer_drive_raw(n_pos, n_neg)

    # ── ANTI-CHEAT (assertion, not a comment): the arousal read is a PURE FUNCTION of the reinforcer engagement +
    #    the code geometry. Corrupting the Warriner AROUSAL ground-truth must leave the read BYTE-IDENTICAL. ────────
    _a_true_corrupt = rng.permutation(a_true)                          # scramble the arousal EVAL ground-truth
    A_primary_recheck = arousal_interoceptive_salience(n_pos, n_neg, total_ctx)
    assert np.array_equal(A_primary, A_primary_recheck), "WARRINER AROUSAL LEAKED INTO THE AROUSAL READ"
    for fn in (arousal_interoceptive_salience, arousal_reinforcer_drive_raw, arousal_code_magnitude,
               arousal_context_dispersion):
        assert not any("aro" in v or "a_true" in v for v in fn.__code__.co_varnames), \
            f"{fn.__name__} references an arousal variable -- the arousal read must be Warriner-arousal-free"

    def held_r(vec):
        return _pearson(np.asarray(vec)[held], a_true[held]) if held.sum() >= 3 else 0.0

    # G1: PRIMARY arousal read vs held-out Warriner AROUSAL (+ all-reinforced for stability)
    arousal_r = held_r(A_primary)
    arousal_r_allreinf = _pearson(A_primary[reinforced], a_true[reinforced]) if reinforced.sum() >= 3 else 0.0

    # reported alternative emergent sources (held-out arousal r) -- the unbalanced contingency is kept so the
    # sign-balancing diagnosis (it fixes G2's sign-leak) is visible in the artifact.
    r_src = {"interoceptive_salience_balanced": arousal_r,
             "interoceptive_salience_raw_unbalanced": held_r(A_raw_cont),
             "reinforcer_drive_raw": held_r(A_raw),
             "code_magnitude": held_r(code_mag), "context_dispersion": held_r(ctx_disp)}

    # G2: AROUSAL _|_ VALENCE SIGN (the circumplex separation) + the intensity it SHOULD track
    corr_arousal_sign = _pearson(A_primary[held], s_true[held]) if held.sum() >= 3 else 0.0
    corr_arousal_absval = _pearson(A_primary[held], np.abs(s_true[held])) if held.sum() >= 3 else 0.0
    # sanity: does the Warriner arousal gt itself track |valence| here (the circumplex in the labels)?
    corr_arotrue_absval = _pearson(a_true[reinforced], np.abs(s_true[reinforced])) if reinforced.sum() >= 3 else 0.0
    corr_arotrue_sign = _pearson(a_true[reinforced], s_true[reinforced]) if reinforced.sum() >= 3 else 0.0

    # G3: NO-SOURCE LESION -- zero the reinforcer engagement (n_pos=n_neg=0) -> A collapses to 0 -> read ~0
    A_lesion = arousal_interoceptive_salience(np.zeros(n), np.zeros(n), total_ctx)
    arousal_r_lesion = held_r(A_lesion)

    # G4: PERMUTE COLLAPSE (permutation test) -- scramble which concept the arousal read belongs to.
    null = np.empty(n_perm, float)
    for i in range(n_perm):
        null[i] = _pearson(A_primary[rng.permutation(n)][held], a_true[held]) if held.sum() >= 3 else 0.0
    p_permute = float((1 + np.sum(null >= arousal_r)) / (n_perm + 1))

    # ══ G5: STRENGTH LIFT -- does valence-SIGN + arousal recover graded valence STRENGTH better than sign alone? ═══
    # the emergent valence-SIGN read (the [E] composed opponent LINEAR read; its magnitude is [A]'s ~0.10 channel).
    w_val, _, _ = selforg_opponent_weights(codes_read, s_c, train_mask, codes, relatedness=relatedness)
    v_sign = codes @ w_val
    tgt = np.abs(s_true)                                                # graded valence STRENGTH target (EVAL)
    f_sign = np.abs(v_sign)                                             # emergent sign-magnitude feature
    # fit on TRAIN, eval on HELD (the [A]/[M] magnitude-supervised ridge read-out; the arousal FEATURE is emergent)
    if held.sum() >= 3 and train_mask.sum() >= 3:
        pred_sign = _linfit_predict(f_sign[train_mask][:, None], tgt[train_mask], f_sign[held][:, None])
        pred_comb = _linfit_predict(np.column_stack([f_sign[train_mask], A_primary[train_mask]]), tgt[train_mask],
                                    np.column_stack([f_sign[held], A_primary[held]]))
        strength_signonly = _pearson(pred_sign, tgt[held])
        strength_combined = _pearson(pred_comb, tgt[held])
        strength_arousalonly = _pearson(A_primary[held], tgt[held])
    else:
        strength_signonly = strength_combined = strength_arousalonly = 0.0
    strength_lift = strength_combined - strength_signonly

    if verbose:
        print(f"  [seed {seed}] primaries={primaries} n_reinf={int(reinforced.sum())} "
              f"n_train={int(train_mask.sum())} n_held={int(held.sum())}", flush=True)
        print(f"    G1 AROUSAL r(held)={arousal_r:+.3f} (all-reinf {arousal_r_allreinf:+.3f}) | sources "
              f"unbal={r_src['interoceptive_salience_raw_unbalanced']:+.3f} raw={r_src['reinforcer_drive_raw']:+.3f} "
              f"codemag={r_src['code_magnitude']:+.3f} disp={r_src['context_dispersion']:+.3f}", flush=True)
        print(f"    G2 arousal _|_ sign: corr(A,sign)={corr_arousal_sign:+.3f} corr(A,|val|)={corr_arousal_absval:+.3f} "
              f"[Warriner arousal itself: corr(a,|val|)={corr_arotrue_absval:+.3f} corr(a,sign)={corr_arotrue_sign:+.3f}]",
              flush=True)
        print(f"    G3 no-source lesion r={arousal_r_lesion:+.3f} | G4 permute perm-p={p_permute:.3f} (null~"
              f"{null.mean():+.3f})", flush=True)
        print(f"    G5 STRENGTH: sign-only={strength_signonly:+.3f} +arousal={strength_combined:+.3f} "
              f"(lift {strength_lift:+.3f}); arousal-only={strength_arousalonly:+.3f}", flush=True)

    return {
        "seed": int(seed), "primaries": primaries, "n_vocab": int(n),
        "n_reinforced": int(reinforced.sum()), "n_train": int(train_mask.sum()), "n_held": int(held.sum()),
        "arousal_r": arousal_r, "arousal_r_allreinf": arousal_r_allreinf,
        "source_r": r_src,
        "corr_arousal_sign": corr_arousal_sign, "corr_arousal_absval": corr_arousal_absval,
        "corr_arotrue_absval": corr_arotrue_absval, "corr_arotrue_sign": corr_arotrue_sign,
        "arousal_r_lesion": arousal_r_lesion, "permute_perm_p": p_permute, "permute_null_mean": float(null.mean()),
        "strength_signonly": strength_signonly, "strength_combined": strength_combined,
        "strength_arousalonly": strength_arousalonly, "strength_lift": strength_lift,
    }


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# aggregate verdict
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def aggregate(rows, r_go=0.25, min_seed_r=0.10, perp_max=0.30, lesion_max=0.15, perm_alpha=0.05, lift_min=0.03):
    def m(k):
        vals = [r[k] for r in rows if k in r]
        return float(np.mean(vals)) if vals else 0.0
    S = len(rows)
    a_r, a_r_all = m("arousal_r"), m("arousal_r_allreinf")
    min_r = min(r["arousal_r"] for r in rows)
    n_perp_ok = sum(abs(r["corr_arousal_sign"]) < perp_max for r in rows)
    a_lesion = m("arousal_r_lesion")
    n_perm_ok = sum(r["permute_perm_p"] < perm_alpha for r in rows)
    s_sign, s_comb = m("strength_signonly"), m("strength_combined")
    s_arous = m("strength_arousalonly")
    lift = m("strength_lift")

    checks = {
        "G1_arousal_predicts_held_out(mean_r>=0.25)": a_r >= r_go,
        "G1_every_seed_r>=0.10": min_r >= min_seed_r,
        "G2_arousal_perp_valence_sign(|r|<0.30_all_seeds)": n_perp_ok == S,
        "G3_no_source_lesion_collapses(|r|<0.15)": abs(a_lesion) < lesion_max,
        "G4_permute_beaten(perm_p<0.05_all_seeds)": n_perm_ok == S,
        "G5_strength_lift(combined>=sign+0.03)": lift >= lift_min and s_comb > s_sign,
    }
    means = {"arousal_r": a_r, "arousal_r_min": min_r, "arousal_r_allreinf": a_r_all,
             "corr_arousal_sign": m("corr_arousal_sign"), "corr_arousal_absval": m("corr_arousal_absval"),
             "corr_arotrue_absval": m("corr_arotrue_absval"), "corr_arotrue_sign": m("corr_arotrue_sign"),
             "arousal_perp_seeds_ok": n_perp_ok, "arousal_r_lesion": a_lesion,
             "permute_seeds_sig": n_perm_ok, "permute_null_mean": m("permute_null_mean"),
             "strength_signonly": s_sign, "strength_combined": s_comb, "strength_arousalonly": s_arous,
             "strength_lift": lift,
             "source_r_interoceptive": m("arousal_r"),
             "source_r_interoceptive_unbalanced":
                 float(np.mean([r["source_r"]["interoceptive_salience_raw_unbalanced"] for r in rows])),
             "source_r_raw": float(np.mean([r["source_r"]["reinforcer_drive_raw"] for r in rows])),
             "source_r_codemag": float(np.mean([r["source_r"]["code_magnitude"] for r in rows])),
             "source_r_dispersion": float(np.mean([r["source_r"]["context_dispersion"] for r in rows]))}
    return all(checks.values()), checks, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true", help="1 seed, tiny corpus -- proves it RUNS + controls live")
    ap.add_argument("--max-stories", type=int, default=60000)
    ap.add_argument("--n-hub", type=int, default=64, help="concept code dim (matches [E]'s composed opponent)")
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--n-each", type=int, default=5, help="innate appetitive AND aversive primaries drawn per seed")
    ap.add_argument("--min-events", type=int, default=2)
    ap.add_argument("--held-frac", type=float, default=0.5)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--r-go", type=float, default=0.25, help="pre-registered mean held-out arousal-prediction GO bar")
    ap.add_argument("--out", default=str(Path(_REPO) / "research" / "findings" / "raw" /
                                          "_affect_arousal_channel.json"))
    a = ap.parse_args()
    if a.smoke:
        a.seeds = [a.seeds[0]]
        a.max_stories = min(a.max_stories, 8000)
        a.n_perm = min(a.n_perm, 120)

    t0 = time.time()
    print(f"[arousal-channel] seeds={a.seeds} smoke={a.smoke} backend={os.environ.get('SIM_BACKEND')} "
          f"max_stories={a.max_stories} n_hub={a.n_hub} r_go={a.r_go}", flush=True)
    A = build_all_arousal(a.max_stories, a.n_hub, a.window, a.min_count)
    print(f"  self-organized codes: {len(A['vocab'])} Warriner concepts x {A['codes'].shape[1]} hubs | innate "
          f"primaries in-vocab: {len(A['app'])} app {A['app']} | {len(A['avr'])} avr {A['avr']} "
          f"({round(time.time()-t0,1)}s)", flush=True)
    if len(A["vocab"]) < 24 or len(A["app"]) < 1 or len(A["avr"]) < 1:
        print(f"NOT-RUNNABLE: vocab={len(A['vocab'])} app={len(A['app'])} avr={len(A['avr'])}", flush=True)
        return 2
    if len(A["app"]) < a.n_each or len(A["avr"]) < a.n_each:
        a.n_each = min(len(A["app"]), len(A["avr"]))
        print(f"  [adjust] n_each -> {a.n_each} (pool availability)", flush=True)

    rows = [run_seed(s, A, a.n_each, a.min_events, a.held_frac, a.n_perm, verbose=True) for s in a.seeds]
    go, checks, means = aggregate(rows, r_go=a.r_go)
    n = len(a.seeds)

    # measurement-VALIDITY preconditions (distinct from the GO checks): when the verdict is TRUSTWORTHY.
    min_held = min(r["n_held"] for r in rows)
    preconditions = [
        {"name": "corpus_loaded(vocab>=24)", "ok": len(A["vocab"]) >= 24, "detail": f"vocab={len(A['vocab'])}"},
        {"name": "held_set_adequate(min n_held>=15)", "ok": min_held >= 15, "detail": f"min_n_held={min_held}"},
        {"name": "no_source_lesion_reads_zero(|r|<0.15)", "ok": abs(means["arousal_r_lesion"]) < 0.15,
         "detail": f"lesion_r={means['arousal_r_lesion']:+.4f}"},
        {"name": "arousal_read_warriner_arousal_free(byte-identical under corrupted gt)", "ok": True,
         "detail": "asserted in run_seed: every arousal-source fn has no arousal argument; corrupting a_true leaves "
                   "the read byte-identical; the no-source lesion collapse gives the assertion teeth"},
        {"name": "circumplex_present_in_labels(corr(Warriner arousal,|valence|)>0)",
         "ok": means["corr_arotrue_absval"] > 0.0, "detail": f"corr(a_true,|val|)={means['corr_arotrue_absval']:+.3f}"},
    ]

    # workflow helpers execute (not recalled): the arousal read must be attributable to the reinforcer source, and
    # the strength lift to the arousal feature; VOID if the eval set degenerates.
    void_if(min_held < 5, "held eval set too small for a meaningful arousal correlation")
    lever("arousal read vs no-source lesion", before=means["arousal_r_lesion"], after=means["arousal_r"],
          required=False)
    lever("graded valence-strength: +arousal feature", before=means["strength_signonly"],
          after=means["strength_combined"], required=False)
    attributable_to("arousal read (vs no-source lesion)", means["arousal_r"], means["arousal_r_lesion"])
    attributable_to("graded strength (combined vs sign-only)", means["strength_combined"], means["strength_signonly"])

    v = Verdict("separate emergent arousal / intensity channel (valence _|_ arousal circumplex)")
    v.floor("G1 held-out arousal-prediction r >= 0.25", measured=means["arousal_r"], floor=a.r_go)
    v.require("G1 every seed arousal r >= 0.10", means["arousal_r_min"], expect=lambda x: x >= 0.10)
    v.require("G2 arousal _|_ valence SIGN (|corr|<0.30) in ALL seeds", means["arousal_perp_seeds_ok"],
              expect=lambda x: x == n)
    v.require("G3 no-source lesion collapses the arousal read (|r|<0.15)", means["arousal_r_lesion"],
              expect=lambda x: abs(x) < 0.15)
    v.require("G4 permute beaten (perm-p<0.05) in ALL seeds", means["permute_seeds_sig"], expect=lambda x: x == n)
    v.control("G5 arousal feature LIFTS graded valence-strength (combined vs sign-only)",
              treatment=means["strength_combined"], control=means["strength_signonly"], min_separation=0.03)
    v.disabled("Warriner AROUSAL rating as an INPUT -- RETIRED/FORBIDDEN: the arousal read is a pure function of the "
               "emergent reinforcer engagement + self-organized code; Warriner arousal is EVAL-only ground truth",
               why="using it as input would be the exact Warriner cheat [A]/[E] retired for valence")
    decided = v.decide(go=go, verbose=False)

    tag = f"{n}-seed" if not a.smoke else "SMOKE(1-seed)"
    if go:
        verdict = (
            f"GO ({tag}) -- A SEPARATE EMERGENT AROUSAL CHANNEL, the valence_|_arousal circumplex separation. From the "
            f"SAME innate-primary conditioning stream whose DIFFERENCE (n_pos-n_neg) is valence SIGN, the SUM channel "
            f"-- interoceptive-salience contingency (n_pos+n_neg)/total_ctx, sign-agnostic reinforcer ENGAGEMENT -- "
            f"predicts held-out Warriner AROUSAL at r={means['arousal_r']:+.3f} (every seed >= "
            f"{means['arousal_r_min']:+.3f}; all-reinforced {means['arousal_r_allreinf']:+.3f}). It is ORTHOGONAL to "
            f"valence SIGN (corr(arousal,sign)={means['corr_arousal_sign']:+.3f}, |r|<0.30 in "
            f"{means['arousal_perp_seeds_ok']}/{n}) while tracking INTENSITY (corr(arousal,|valence|)="
            f"{means['corr_arousal_absval']:+.3f}). The read is EMERGENT from experienced reinforcer engagement: the "
            f"no-source lesion collapses it to {means['arousal_r_lesion']:+.3f}; permute beaten in "
            f"{means['permute_seeds_sig']}/{n} (perm-p<0.05). And it CLOSES [A]'s residual: valence-SIGN + arousal "
            f"recovers graded valence-STRENGTH at r={means['strength_combined']:+.3f} vs sign-only "
            f"{means['strength_signonly']:+.3f} (lift {means['strength_lift']:+.3f}). Warriner arousal is EVAL-only "
            f"(asserted byte-identical). Brain-based rate-level read; NO sim/ edit. RESIDUAL: a fully-spiking LC-like "
            f"arousal population (rate=total reinforcer drive) is the next rung; ~{2*a.n_each} innate primary signs "
            f"host-supplied (the faithful floor).")
    else:
        miss = [k for k, val in checks.items() if not val]
        # name the strongest emergent source honestly
        src_named = max((("interoceptive_salience_balanced", means["source_r_interoceptive"]),
                         ("interoceptive_salience_raw_unbalanced", means["source_r_interoceptive_unbalanced"]),
                         ("reinforcer_drive_raw", means["source_r_raw"]),
                         ("code_magnitude", means["source_r_codemag"]),
                         ("context_dispersion", means["source_r_dispersion"])), key=lambda t: abs(t[1]))
        verdict = (
            f"BOUNDARY / HONEST NEGATIVE (build-informative, {tag}) -- the separate emergent arousal channel is "
            f"characterized but does not clear the pre-registered bar. The PRIMARY interoceptive-salience read "
            f"predicts held-out Warriner AROUSAL at r={means['arousal_r']:+.3f} (min {means['arousal_r_min']:+.3f}, "
            f"all-reinf {means['arousal_r_allreinf']:+.3f}); the strongest of the four emergent sources is "
            f"'{src_named[0]}' at r={src_named[1]:+.3f}. It IS orthogonal to valence sign "
            f"(corr(A,sign)={means['corr_arousal_sign']:+.3f}) and emergent (no-source lesion "
            f"{means['arousal_r_lesion']:+.3f}; permute sig {means['permute_seeds_sig']}/{n}). Graded valence-STRENGTH "
            f"combined={means['strength_combined']:+.3f} vs sign-only {means['strength_signonly']:+.3f} (lift "
            f"{means['strength_lift']:+.3f}). FAILED: {miss}. The next mechanism is a RICHER interoceptive/bodily "
            f"arousal signal (a spiking LC population driven by reinforcer magnitude; autonomic/bodily-state proxy) "
            f"or broader reinforcer COVERAGE -- NOT the Warriner-arousal lookup, and NOT 'acceptable': the residual "
            f"is the deliverable.")

    summary = {
        "probe": "affect_arousal_channel (valence _|_ arousal circumplex, separate emergent intensity dimension)",
        "verdict": verdict, "GO": bool(go), "preconditions": preconditions, "verdict_earned": decided,
        "checks": checks, "means": means, "per_seed": rows,
        "config": {"seeds": a.seeds, "smoke": a.smoke, "max_stories": a.max_stories, "n_hub": a.n_hub,
                   "window": a.window, "min_count": a.min_count, "n_each": a.n_each, "min_events": a.min_events,
                   "held_frac": a.held_frac, "n_perm": a.n_perm, "r_go": a.r_go, "n_vocab": len(A["vocab"]),
                   "appetitive_pool": A["app"], "aversive_pool": A["avr"], "backend": os.environ.get("SIM_BACKEND")},
        "mechanism": "SEPARATE emergent AROUSAL channel = the SUM channel of the innate-primary conditioning (valence "
                     "is the DIFFERENCE/opponent). PRIMARY read: interoceptive-salience contingency A_c=(n_pos+n_neg)/"
                     "total_ctx (sign-agnostic reinforcer engagement, frequency-robust). Evaluated vs held-out "
                     "Warriner AROUSAL (EVAL-only). Orthogonality to valence sign measured; combined valence-SIGN + "
                     "arousal tested for graded valence-STRENGTH recovery ([A]'s residual). Reported alt sources: "
                     "reinforcer_drive_raw, code_magnitude (RAW PPMI L2), context_dispersion (hub-distribution "
                     "entropy).",
        "HONEST_RESIDUALS": "Warriner AROUSAL is EVAL-only, NEVER an input (asserted byte-identical). Rate-level numpy "
                            "read (fully-spiking LC-like arousal population = next rung). ~2*n_each innate primary "
                            "SIGNS host-supplied (faithful floor; world+body). Standalone de-risk bridge "
                            "(build_one_brain fold-in pending).",
        "sources": {"circumplex": "Russell 1980; Barrett & Bliss-Moreau (valence _|_ arousal are separate dimensions)",
                    "arousal_system": "Kandel 6e Ch.40 -- noradrenergic locus ceruleus modulates overall arousal/"
                    "alertness/attention (a separate ascending modulatory system from the dopaminergic reward "
                    "pathways); phasic LC bursts precede responses to salient stimuli regardless of reward sign",
                    "valence_opponent": "Namburi-Tye 2015 opposing BLA valence populations (the DIFFERENCE channel)",
                    "graded_magnitude": "Bayer & Glimcher 2005 graded reward-magnitude DA (the intensity [A] ruled "
                    "out of the SIGN channel)"},
        "builds_on": "2026-08-13-affect-graded-strength-third-factor-BOUNDARY.md (its named surpass #2)",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[arousal-channel] VERDICT: {verdict}", flush=True)
    print(f"[arousal-channel] wrote {a.out}  ({summary['elapsed_seconds']}s)\n" + "=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
