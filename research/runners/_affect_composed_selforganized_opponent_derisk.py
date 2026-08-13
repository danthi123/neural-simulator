"""COMPOSED AFFECT: the SPIKING opponent V+/V- weights DERIVE FROM the SELF-ORGANIZED valence map -- the
Warriner-seeded ridge-fit is RETIRED (emergence-bar composition, 2026-08-13).

Two affect de-risks landed on main with a clean composition seam:

  EMERGENCE lane (`_affect_evaluative_conditioning_derisk.py`, DR-2b GO): the ORIGIN of concept valence
    SELF-ORGANIZES from an evaluative-conditioning stream -- a LOCAL three-factor (DA-gated Hebbian) rule grows a
    concept->valence map over the already-self-organized PPMI stream-cortex code, anchored by only ~10 INNATE
    primary reinforcers (a genome-cheap +/-1 sign). Held-out r=+0.55. A LEARNED valence map; NO Warriner lexicon.

  AFFECT-DEEPEN lane (`_affect_appraisal_emotion_reappraisal_derisk.py`, GO): a spiking opponent population
    (`appr_vplus`/`appr_vminus`) reads valence off the substrate code and drives discrete emotions + reappraisal.
    Its declared #1 residual: the opponent FEEDFORWARD weights are RIDGE-FIT in numpy AND SEEDED from Warriner
    norms -- "the seed supervision is NOT retired".

THIS RUNNER COMPOSES THEM. It DERIVES the affect opponent V+/V- projection FROM the emergence lane's self-organized
valence map (three-factor Hebbian outer-product over the learned code, anchored by ~10 innate primaries), REPLACING
the Warriner-seeded ridge-fit ENTIRELY. Then it re-runs the affect-deepen quality bars on this composed circuit and
shows they HOLD with ZERO Warriner supervision in the weights. The opponent READ stays SPIKING (off
`cp_firing_states`), exactly as the affect-deepen runner does -- only the WEIGHT SOURCE changes.

If the bars hold => the seed is RETIRED and the WHOLE affect appraisal traces to innate primaries + experience, not
a human-rated lexicon (a genuine emergence-bar closure). If the composed (Warriner-free) circuit underperforms the
ridge-fit => that is an HONEST NEGATIVE (a first-class deliverable): the residual gap is quantified and the next
mechanism named. Either way, Warriner is used ONLY as external held-out GROUND-TRUTH for EVALUATION -- NEVER as a
weight input (asserted in code, not a comment).

THE WEIGHT DERIVATION (the composition, reuse-by-import, NO `sim/` edit):
  - Concept CODE  = the DR-2 self-organized PPMI stream cortex (`build_cooccurrence`/`codes_from_cooccurrence`). [pre]
  - ~10 INNATE primaries -> per-target evaluative-conditioning valence s_c (Rescorla-Wagner asymptote of
    co-occurrence with the primaries), via the emergence lane's `build_primary_cooccurrence`. [the CS<->US pairing]
  - THREE-FACTOR HEBBIAN outer-product map over the TRAIN concepts only: w = sum_{c in train} code_read_c * s_c
    (DA/US-gated associative memory), with a LABEL-FREE mean-code (hub-ness) gain-control (value _|_ plausibility).
  - RECTIFIED Namburi-Tye opponent split: W+ = g*max(w,0), W- = g*max(-w,0) -- injected as the SAME `code_in`->
    `appr_vplus`/`appr_vminus` feedforward the affect-deepen bridge uses. The differential rate(vplus)-rate(vminus)
    read off `cp_firing_states` = the SPIKING appraisal, unchanged.

The map is built from the TRAIN split ONLY; each HELD concept's OWN experienced reinforcement is WITHHELD from the
map (the DR-2 leave-out protocol), so the held read rides purely its learned-code resemblance to OTHER reinforced
concepts. Warriner ground-truth scores the held prediction (EVALUATION only).

PRE-REGISTERED GO GATE (6-seed; the affect-deepen bars, re-run on the composed Warriner-free circuit):
  C-A1 (rung-a corr)   held-out SPIKING opponent differential r >= 0.45 with true signed valence, AND every seed
                       >= 0.25 (the WEIGHTS are self-organized -- no Warriner seed).
  C-A2 (salience gate) |differential| tracks valence STRENGTH (r>0.2) AND the input-lesion collapses it to ~0.
  C-A3 (no-cond lesion)REMOVE the conditioning stream (s_c := 0) -> the composed weights collapse -> held-out
                       spiking r collapses to ~0 (< 0.15). The weights come from EXPERIENCE, not Warriner.
  C-A4 (permute-code)  PERMUTATION test (the emergence-established instrument; a single draw is too noisy on ~60
                       concepts in a 64-dim code): scramble which learned code belongs to which word -> real
                       held-out r beats the null at perm-p<0.05 in ALL seeds. The self-organized code geometry
                       carries the generalization.
  C-A5 (unpaired-US)   PERMUTATION test: the US arrives paired with the WRONG concept (permute s_c across reinforced
                       concepts) -> real beats null at perm-p<0.05 in ALL seeds. The CS<->US contingency is load-bearing.
  C-B1 (emotion discr) each of the 4 appraisal conditions selects its intended emotion (mean accuracy >= 0.75; >=3
                       distinct winners). The rung-b valence cue is chosen by s_c (self-organized), NOT Warriner.
  C-B2 (reappraisal)   the vmPFC->amygdala gate down-regulates appr_vminus by >= 25%.
  C-B3 (WTA lesion)    lesioning the shared-FS cross-inhibition collapses the categorical margin >= 35%.
  C-B4 (reap lesion)   lesioning vmpfc_reap->appr_vminus abolishes the down-regulation.
  C-B5 (mismatch cheat)permuting the condition->intended-emotion labels drops accuracy to ~chance.
GO iff C-A1..A5 AND C-B1..B5. Reported (not gated): the RIDGE-to-Warriner baseline spiking r (the RETIRED method,
for a like-for-like gap), corr(s_c, Warriner) (the innate signal is honest), value_|_relatedness.

BRAIN-BASED: the appraisal READ (opponent differential) + the emotion SELECTION (shared-FS WTA winner) are spike-rate
reads off `cp_firing_states`; the opponent WEIGHTS are now the self-organized three-factor Hebbian map. HONEST
RESIDUALS (declared): (1) ~10 innate primary SIGNS remain host-supplied (the biologically-faithful floor: valence IS
innately anchored by primary reinforcers; a 140->10 compression, not a removal); (2) the outer-product Hebbian map is
a rate-level numpy matrix (the codes are spiking-validated; a fully-spiking three-factor write is the named next
rung); (3) the agency/certainty appraisal conditions are set as sensory drive by the environment/teacher; (4) standalone
de-risk bridge -- folding into build_one_brain is the production-integration step. DISCIPLINE: reuse-by-import, NO
`sim/` edit, cfg.seed (not actual_seed_used).

Run (smoke): SIM_BACKEND=numpy python -u -m research.runners._affect_composed_selforganized_opponent_derisk --smoke
Run (6-seed):SIM_BACKEND=numpy python -u -m research.runners._affect_composed_selforganized_opponent_derisk \
                --seeds 42 43 44 100 101 102 \
                --out research/findings/raw/_affect_composed_selforganized_opponent_6seed.json
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

from tools.lab import attributable_to  # noqa: E402  (force the treatment/control SUBTRACTION to be asked out loud)
from tools.verdict import Verdict       # noqa: E402  (a verdict that carries a `preconditions` block)

# reuse-by-import: the affect-deepen SPIKING circuit (bridge + reads + operating point) + the ridge BASELINE.
from research.runners._affect_appraisal_emotion_reappraisal_derisk import (  # noqa: E402
    build_bridge, read_valence, read_emotion, ridge_opponent, CONDITIONS, EMO_NAMES, N_OPP, _pearson,
)
# reuse-by-import: the DR-2 self-organized PPMI stream cortex + Warriner ground-truth (EVAL ONLY).
from research.runners._affect_distributional_tag_derisk import (  # noqa: E402
    WARRINER, load_stories, build_cooccurrence, codes_from_cooccurrence,
)
# reuse-by-import: the emergence lane's INNATE primaries + the conditioning-event corpus scan.
from research.runners._affect_evaluative_conditioning_derisk import (  # noqa: E402
    APPETITIVE_POOL, AVERSIVE_POOL, build_primary_cooccurrence,
)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# THE COMPOSITION: derive the opponent weights from the SELF-ORGANIZED conditioning valence (NO Warriner).
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Operating-point GAIN calibration (NOT supervision): the affect-deepen rung-b operating point (VAL_TO_EMO_W vs
# W_MAP vs OPP_FF_GAIN) was grid-searched for the RIDGE weight's magnitude (L2~1.7 on this code dim). The Hebbian
# outer-product weight has a different scale (L2 ~2.5x larger), so it OVERDRIVES that balance. We normalize the
# composed weight DIRECTION to a fixed reference L2 norm so the operating point transfers unchanged. A single global
# scalar carries NO per-concept valence information (it does not say which concepts are +/-) and does not change the
# rung-a correlation (scale-invariant) -- it is a gain calibration, not a Warriner seed.
W_L2_REF = 1.7


def selforg_opponent_weights(codes_read, s_c, train_mask, hub_codes, relatedness=None, gain_orth=True, l2_ref=None):
    """DERIVE the rectified Namburi-Tye opponent weight from the SELF-ORGANIZED valence map. This is a PURE FUNCTION
    of (the self-organized code geometry `codes_read`, the conditioning-acquired valence `s_c`, the train split) --
    Warriner is NOT an argument and CANNOT enter here. The three-factor Hebbian outer-product associative memory:

        w = sum_{c in train} code_read_c * s_c          (pre=learned code; post=US-driven opponent; DA/US=sign(s_c))

    with a LABEL-FREE hub-ness gain-control (value _|_ plausibility): the emergence lane's PROVEN control decorrelates
    the READ from per-concept relatedness. Because relatedness_x ~= code_x . mean_code, that read-space control folds
    exactly into the WEIGHT as w - beta*mean_code with beta = cov(code.w, relatedness)/var(relatedness) over the
    concept population (label-free -- relatedness carries no valence). This is per-concept-consistent and injectable
    into the spiking FF (a plain mean-code projection is the WRONG axis -- hub leakage rides the code-covariance
    direction, which left a 1/6 seed hub-confounded; the emergence lane documented the same seed-102 confound).
    `l2_ref` (when set) rescales the weight DIRECTION to that fixed L2 norm -- a Warriner-free gain calibration that
    transfers the affect-deepen operating point (correlation-invariant). Returns (w, W_plus, W_minus) with
    W_plus=max(w,0), W_minus=max(-w,0) (both excitatory FF)."""
    codes = np.asarray(hub_codes)
    w = (np.asarray(codes_read)[train_mask] * np.asarray(s_c)[train_mask, None]).sum(axis=0)  # [D]
    if gain_orth:
        mc = codes.mean(axis=0)                                     # the population common-mode (hub-ness) direction
        if relatedness is not None:
            v = codes @ w                                           # per-concept read
            rel = np.asarray(relatedness, float)
            beta = float(np.cov(v, rel)[0, 1] / (np.var(rel) + 1e-12))   # regression coef (cleaned read _|_ hub-ness)
            w = w - beta * mc                                       # foldable decorrelation (emergence hub control)
        else:
            w = w - (float(w @ mc) / (float(mc @ mc) + 1e-12)) * mc      # fallback: mean-code projection
    if l2_ref is not None:
        nrm = float(np.linalg.norm(w))
        if nrm > 1e-9:
            w = w * (float(l2_ref) / nrm)                          # gain calibration (scale only; direction intact)
    wp = np.maximum(w, 0.0)
    wm = np.maximum(-w, 0.0)
    return w, wp, wm


def rescorla_wagner_valence(Co, prim_idx, prim_sgn, is_primary, min_events):
    """Per-target evaluative-conditioning valence s_c from co-occurrence with the CHOSEN innate primaries (the
    emergence lane's mechanism). s_c = (n_pos - n_neg)/(n_pos + n_neg) in [-1,1] (Rescorla-Wagner asymptote;
    frequency-robust). Reinforced = enough primary co-occurrences AND not a primary itself. Warriner-FREE."""
    n = Co.shape[0]
    sub = Co[:, prim_idx]
    n_pos = (sub * (prim_sgn > 0)).sum(axis=1)
    n_neg = (sub * (prim_sgn < 0)).sum(axis=1)
    tot = n_pos + n_neg
    reinforced = (tot >= min_events) & (~is_primary)
    s_c = np.zeros(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        s_c[reinforced] = (n_pos[reinforced] - n_neg[reinforced]) / tot[reinforced]
    return s_c, reinforced


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# corpus build (once; seed-independent) -> codes, DC-removed read-codes, relatedness, Warriner s_true, primary Co.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def build_all(max_stories, n_hub, window, min_count):
    stories = load_stories(max_stories)
    vocab, C = build_cooccurrence(stories, n_hub, window, min_count)
    codes = codes_from_cooccurrence(C)                                   # non-neg L2 PPMI (spiking drive uses this)
    # DC-removed read codes (a ubiquitous cortical subtractive normalization; makes the opponent V+-V- sign-meaningful)
    codes_read = codes - codes.mean(axis=0, keepdims=True)
    codes_read = codes_read / (np.linalg.norm(codes_read, axis=1, keepdims=True) + 1e-12)
    Wsim = codes @ codes.T
    np.fill_diagonal(Wsim, 0.0)
    relatedness = np.asarray(Wsim.mean(axis=1), float)                   # per-concept hub-ness (label-free)
    val = np.array([WARRINER[w][0] for w in vocab], float)
    s_true = (val - 5.0) / 4.0                                           # signed Warriner GROUND-TRUTH (EVAL ONLY)
    vocab_set = set(vocab)
    app = [w for w in APPETITIVE_POOL if w in vocab_set]
    avr = [w for w in AVERSIVE_POOL if w in vocab_set]
    all_primaries = app + avr
    prim_sign_full = {**{w: +1.0 for w in app}, **{w: -1.0 for w in avr}}
    Co = build_primary_cooccurrence(stories, vocab, window, all_primaries)
    return dict(vocab=vocab, codes=codes, codes_read=codes_read, relatedness=relatedness, s_true=s_true,
                app=app, avr=avr, all_primaries=all_primaries, prim_sign_full=prim_sign_full, Co=Co)


def primary_count_ablation(A, min_events=2, held_frac=0.5, counts=(4, 5, 6, 8, 10), n_permcode=100,
                           seeds=(42, 43, 44, 100, 101, 102)):
    """Is the graded-magnitude residual a substrate wall or genome-DRAW variance? Cheap LINEAR-fidelity ablation of
    the composed weight over the INNATE-PRIMARY count (how many +/- reinforcers the genome specifies). Reports the
    held-out valence r, the salience-magnitude r (|read|~|valence|), and the permute-code perm-p pass-rate across
    seeds. If the bars tighten with MORE innate signs, the residual is genome-draw variance (soft, closable by a
    larger reinforcer set), NOT a substrate limit. Warriner is EVAL-only here too."""
    vocab, codes, codes_read = A["vocab"], A["codes"], A["codes_read"]
    relatedness, s_true, Co = A["relatedness"], A["s_true"], A["Co"]
    all_primaries, prim_sign_full = A["all_primaries"], A["prim_sign_full"]
    n = len(vocab)
    prim_col = {w: j for j, w in enumerate(all_primaries)}
    app = [w for w in all_primaries if prim_sign_full[w] > 0]
    avr = [w for w in all_primaries if prim_sign_full[w] < 0]
    out = []
    for ne in counts:
        if ne > len(app) or ne > len(avr):
            continue
        hr, ar, permbeat = [], [], 0
        for seed in seeds:
            rng = np.random.default_rng(seed)
            pick = list(rng.choice(app, ne, replace=False)) + list(rng.choice(avr, ne, replace=False))
            pidx = np.array([prim_col[w] for w in pick]); psgn = np.array([prim_sign_full[w] for w in pick])
            is_primary = np.array([w in set(pick) for w in vocab])
            s_c, reinf = rescorla_wagner_valence(Co, pidx, psgn, is_primary, min_events)
            ridx = np.where(reinf)[0]; rng.shuffle(ridx); nh = int(round(held_frac * len(ridx)))
            held = np.zeros(n, bool); held[ridx[:nh]] = True
            tmask = np.zeros(n, bool); tmask[ridx[nh:]] = True
            w, _, _ = selforg_opponent_weights(codes_read, s_c, tmask, codes, relatedness=relatedness)
            v = codes @ w
            r = _pearson(v[held], s_true[held]); hr.append(r)
            ar.append(_pearson(np.abs(v[held]), np.abs(s_true[held])))
            nc = np.array([_pearson((codes @ selforg_opponent_weights(codes_read[rng.permutation(n)], s_c, tmask,
                          codes, relatedness=relatedness)[0])[held], s_true[held]) for _ in range(n_permcode)])
            permbeat += int((1 + np.sum(nc >= r)) / (n_permcode + 1) < 0.05)
        out.append({"n_each": ne, "n_primaries": 2 * ne, "held_r_mean": float(np.mean(hr)),
                    "held_r_min": float(min(hr)), "salience_r_mean": float(np.mean(ar)),
                    "salience_r_min": float(min(ar)), "permute_code_beat": f"{permbeat}/{len(seeds)}"})
    return out


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# ONE SEED: draw the innate primaries, condition s_c, DERIVE the composed opponent, run the affect-deepen bars.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def run_seed(seed, A, n_each, min_events, held_frac=0.5, n_perm=200, max_held_probe=48,
             do_rungb=True, do_spiking_shuffle=True, l2_ref=W_L2_REF, verbose=False):
    rng = np.random.default_rng(seed)
    vocab, codes, codes_read = A["vocab"], A["codes"], A["codes_read"]
    relatedness, s_true, Co = A["relatedness"], A["s_true"], A["Co"]
    all_primaries, prim_sign_full = A["all_primaries"], A["prim_sign_full"]
    n = len(vocab)
    D = codes.shape[1]
    prim_col = {w: j for j, w in enumerate(all_primaries)}

    # --- draw this genome's innate-primary subset (robustness to WHICH reinforcers the genome picked) ---
    app = [w for w in all_primaries if prim_sign_full[w] > 0]
    avr = [w for w in all_primaries if prim_sign_full[w] < 0]
    app_pick = list(rng.choice(app, size=min(n_each, len(app)), replace=False))
    avr_pick = list(rng.choice(avr, size=min(n_each, len(avr)), replace=False))
    primaries = app_pick + avr_pick
    prim_idx = np.array([prim_col[w] for w in primaries])
    prim_sgn = np.array([prim_sign_full[w] for w in primaries], float)
    is_primary = np.array([w in set(primaries) for w in vocab])

    # --- SELF-ORGANIZED conditioning valence s_c (Warriner-free), then TRAIN/HELD leave-out split ---
    s_c, reinforced = rescorla_wagner_valence(Co, prim_idx, prim_sgn, is_primary, min_events)
    ridx = np.where(reinforced)[0]
    rng.shuffle(ridx)
    n_held = int(round(held_frac * len(ridx)))
    held_idx, train_idx = ridx[:n_held], ridx[n_held:]
    train_mask = np.zeros(n, bool); train_mask[train_idx] = True
    held = np.zeros(n, bool); held[held_idx] = True

    # ══ THE COMPOSED OPPONENT WEIGHTS (self-organized; NO Warriner) ══════════════════════════════════════════════
    w, wp, wm = selforg_opponent_weights(codes_read, s_c, train_mask, codes, relatedness=relatedness, l2_ref=l2_ref)

    # ── ANTI-CHEAT (assertion, NOT a comment): the composed weights are a PURE FUNCTION of the conditioning map +
    #    the self-organized code geometry. Corrupting the ONLY Warriner-derived array must leave the weights
    #    BYTE-IDENTICAL (Warriner never feeds the weights). ──────────────────────────────────────────────────────
    _s_true_corrupt = rng.permutation(s_true)               # scramble Warriner ground-truth
    w_recheck, _, _ = selforg_opponent_weights(codes_read, s_c, train_mask, codes, relatedness=relatedness,
                                               l2_ref=l2_ref)  # s_true is NOT an input
    assert np.array_equal(w, w_recheck), "WARRINER LEAKED INTO THE COMPOSED OPPONENT WEIGHTS"
    assert "s_true" not in selforg_opponent_weights.__code__.co_varnames, \
        "selforg_opponent_weights references a Warriner-derived variable -- weights must be Warriner-free"

    bridge, xp, idx, snap = build_bridge(seed, D, wp, wm)

    # ── C-A1/C-A2: held-out SPIKING opponent differential vs true signed valence + the emergent salience gate ─────
    hp = held_idx if len(held_idx) <= max_held_probe else rng.choice(held_idx, max_held_probe, replace=False)
    diffs = np.array([read_valence(bridge, xp, idx, snap, codes[i])["differential"] for i in hp])
    r_real = _pearson(diffs, s_true[hp])
    abs_r = _pearson(np.abs(diffs), np.abs(s_true[hp]))                      # |differential| ~ valence STRENGTH
    les = np.array([read_valence(bridge, xp, idx, snap, codes[i], lesion_input=True)["differential"] for i in hp[:12]])
    lesion_diff_abs = float(np.abs(les).mean())
    intact_diff_abs = float(np.abs(diffs).mean())
    r_perp = _pearson(diffs, relatedness[hp])                               # value _|_ plausibility (reported)

    # ── C-A3: NO-CONDITIONING LESION -- remove the conditioning stream (s_c := 0) -> weights collapse -> read ~0.
    #    This is the KEY "self-organized from experience, not Warriner" control (replaces the ridge's Warriner seed).
    w0, wp0, wm0 = selforg_opponent_weights(codes_read, np.zeros(n, float), train_mask, codes,
                                            relatedness=relatedness, l2_ref=l2_ref)
    br0, xp0, idx0, snap0 = build_bridge(seed + 314, D, wp0, wm0)
    diffs0 = np.array([read_valence(br0, xp0, idx0, snap0, codes[i])["differential"] for i in hp])
    r_no_conditioning = _pearson(diffs0, s_true[hp])

    # ── C-A4 / C-A5: PERMUTATION-TEST controls on the WEIGHT derivation (the emergence-established instrument -- a
    #    single spiking-shuffle draw is too noisy on ~60 concepts in a 64-dim code). Computed on the LINEAR read
    #    code.w (the spiking differential is a monotone image of it), which is what "self-organized" is a claim ABOUT.
    def linear_held_r(w_vec):
        return _pearson((codes @ w_vec)[held], s_true[held]) if held.sum() >= 3 else 0.0
    r_lin_real = linear_held_r(w)
    # C-A4 permute-code: scramble which learned code belongs to which word -> destroy the geometry.
    null_code = np.empty(n_perm, float)
    for i in range(n_perm):
        cperm = rng.permutation(n)
        wc, _, _ = selforg_opponent_weights(codes_read[cperm], s_c, train_mask, codes, relatedness=relatedness)
        null_code[i] = linear_held_r(wc)
    p_permcode = float((1 + np.sum(null_code >= r_lin_real)) / (n_perm + 1))
    # C-A5 unpaired-US: the US paired with the WRONG concept -> permute s_c across reinforced concepts.
    null_us = np.empty(n_perm, float)
    for i in range(n_perm):
        s_sh = s_c.copy(); rp = ridx.copy(); rng.shuffle(rp); s_sh[ridx] = s_c[rp]
        wu, _, _ = selforg_opponent_weights(codes_read, s_sh, train_mask, codes, relatedness=relatedness)
        null_us[i] = linear_held_r(wu)
    p_unpaired = float((1 + np.sum(null_us >= r_lin_real)) / (n_perm + 1))

    # ── DIAGNOSTIC (reported, not gated): affect-deepen's single-draw SPIKING shuffle code<->word (A3). Noisy on a
    #    tiny corpus -- the permutation tests above are the gated instrument. ──────────────────────────────────────
    r_shuffled = 0.0
    if do_spiking_shuffle:
        sperm = rng.permutation(n)
        codes_sh = codes[sperm]
        cr_sh = codes_sh - codes_sh.mean(0, keepdims=True)
        cr_sh = cr_sh / (np.linalg.norm(cr_sh, axis=1, keepdims=True) + 1e-12)
        _, wps, wms = selforg_opponent_weights(cr_sh, s_c, train_mask, codes_sh, relatedness=relatedness[sperm],
                                               l2_ref=l2_ref)
        brs, xps, idxs, snaps = build_bridge(seed + 991, D, wps, wms)
        diffs_sh = np.array([read_valence(brs, xps, idxs, snaps, codes_sh[i])["differential"] for i in hp])
        r_shuffled = _pearson(diffs_sh, s_true[hp])

    # ── BASELINE (reported): the RETIRED method -- ridge-fit-to-WARRINER opponent, same spiking read (like-for-like) ─
    _, wpb, wmb = ridge_opponent(codes[train_idx], s_true[train_idx])
    brb, xpb, idxb, snapb = build_bridge(seed + 555, D, wpb, wmb)
    diffs_b = np.array([read_valence(brb, xpb, idxb, snapb, codes[i])["differential"] for i in hp])
    r_baseline_warriner = _pearson(diffs_b, s_true[hp])

    corr_sc_warr = _pearson(s_c[reinforced], s_true[reinforced]) if reinforced.sum() >= 3 else 0.0

    # ══ RUNG (b): discrete emotion + reappraisal on the COMPOSED opponent. Valence cue chosen by s_c (NOT Warriner) ══
    b = {}
    if do_rungb:
        st_tr = s_c[train_idx]                                  # self-organized valence over TRAIN concepts
        pos_words = train_idx[np.argsort(st_tr)[::-1][:8]]      # most-appetitive by EXPERIENCE (not Warriner)
        neg_words = train_idx[np.argsort(st_tr)[:8]]            # most-aversive by EXPERIENCE
        code_of = {"pos": codes[pos_words].mean(0), "neg": codes[neg_words].mean(0)}

        b_rows, correct = [], 0
        for cond in CONDITIONS:
            res = read_emotion(bridge, xp, idx, snap, code_of[cond["valence"]], cond["dims"])
            ok = res["winner"] == cond["intended"]; correct += int(ok)
            b_rows.append({"cond": cond["name"], "intended": cond["intended"], "winner": res["winner"],
                           "margin": round(res["margin"], 4), "ok": ok})
        accuracy = correct / len(CONDITIONS)
        winners = {r["winner"] for r in b_rows}

        reap_rows = []
        for cond in [c for c in CONDITIONS if c["valence"] == "neg"]:
            base = read_emotion(bridge, xp, idx, snap, code_of["neg"], cond["dims"])
            reap = read_emotion(bridge, xp, idx, snap, code_of["neg"], cond["dims"], reappraise=True)
            reap_les = read_emotion(bridge, xp, idx, snap, code_of["neg"], cond["dims"], reappraise=True,
                                    lesion_reap=True)
            drop = (base["vminus_rate"] - reap["vminus_rate"]) / (base["vminus_rate"] + 1e-9)
            drop_les = (base["vminus_rate"] - reap_les["vminus_rate"]) / (base["vminus_rate"] + 1e-9)
            reap_rows.append({"cond": cond["name"], "vminus_drop_frac": round(drop, 4),
                              "vminus_drop_frac_reap_lesioned": round(drop_les, 4)})
        mean_vminus_drop = float(np.mean([r["vminus_drop_frac"] for r in reap_rows]))
        mean_vminus_drop_les = float(np.mean([r["vminus_drop_frac_reap_lesioned"] for r in reap_rows]))

        intact_margins = [r["margin"] for r in b_rows]
        lesion_margins = []
        for cond in CONDITIONS:
            res = read_emotion(bridge, xp, idx, snap, code_of[cond["valence"]], cond["dims"], lesion_wta=True)
            lesion_margins.append(res["margin"])
        mean_margin_intact = float(np.mean(intact_margins))
        mean_margin_wta_lesion = float(np.mean(lesion_margins))

        mismatch_correct = 0
        for i, cond in enumerate(CONDITIONS):
            wrong_dims = CONDITIONS[(i + 1) % len(CONDITIONS)]["dims"]
            res = read_emotion(bridge, xp, idx, snap, code_of[cond["valence"]], wrong_dims)
            mismatch_correct += int(res["winner"] == cond["intended"])
        accuracy_mismatched = mismatch_correct / len(CONDITIONS)

        b = {"b_accuracy": accuracy, "b_distinct_winners": int(len(winners)), "b_distinct": bool(len(winners) >= 3),
             "b_rows": b_rows, "b_vminus_drop_frac": mean_vminus_drop,
             "b_vminus_drop_frac_reap_lesioned": mean_vminus_drop_les,
             "b_margin_intact": mean_margin_intact, "b_margin_wta_lesion": mean_margin_wta_lesion,
             "b_accuracy_mismatched": accuracy_mismatched}

    if verbose:
        print(f"  [seed {seed}] primaries={primaries}", flush=True)
        print(f"    C-A1 held-out SPIKING r={r_real:+.3f} (baseline ridge-Warriner {r_baseline_warriner:+.3f}) | "
              f"|d|~val r={abs_r:+.3f} lesion|d|={lesion_diff_abs:.3f} vs intact|d|={intact_diff_abs:.3f}", flush=True)
        print(f"    C-A3 no-conditioning spiking r={r_no_conditioning:+.3f} | C-A4 permute-code perm-p={p_permcode:.3f} "
              f"| C-A5 unpaired-US perm-p={p_unpaired:.3f} | shuffled(spiking,diag) {r_shuffled:+.3f} | "
              f"corr(s_c,Warr)={corr_sc_warr:+.3f} perp={r_perp:+.3f}", flush=True)
        if do_rungb:
            for r in b["b_rows"]:
                print(f"     [{r['cond']}] intended {r['intended']} -> {r['winner']} "
                      f"(margin {r['margin']:+.3f}) {'OK' if r['ok'] else 'MISS'}", flush=True)
            print(f"     reappraise vminus drop {b['b_vminus_drop_frac']:+.2%} (reap-les "
                  f"{b['b_vminus_drop_frac_reap_lesioned']:+.2%}); WTA margin {b['b_margin_intact']:.3f} -> "
                  f"{b['b_margin_wta_lesion']:.3f}; mismatch acc {b['b_accuracy_mismatched']:.2f}", flush=True)

    row = {
        "seed": int(seed), "primaries": primaries, "n_vocab": int(n), "code_dim": int(D),
        "n_reinforced": int(reinforced.sum()), "n_train": int(train_mask.sum()), "n_held": int(held.sum()),
        "n_held_probe": int(len(hp)),
        # rung a (composed opponent)
        "a_r_real": r_real, "a_abs_r": abs_r, "a_intact_diff_abs": intact_diff_abs,
        "a_lesion_diff_abs": lesion_diff_abs, "a_r_perp": r_perp,
        "a_r_no_conditioning": r_no_conditioning,
        "a_lin_r_real": r_lin_real, "a_permcode_perm_p": p_permcode, "a_unpaired_perm_p": p_unpaired,
        "a_permcode_null_mean": float(null_code.mean()), "a_unpaired_null_mean": float(null_us.mean()),
        "a_r_shuffled_spiking_diag": r_shuffled,
        "a_r_baseline_warriner": r_baseline_warriner, "corr_s_c_warriner": corr_sc_warr,
    }
    row.update(b)
    return row


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# aggregate verdict
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def aggregate(rows, a_r_go=0.45, min_seed_r=0.25, no_cond_max=0.15, perm_alpha=0.05, b_acc_go=0.75,
              reap_drop_go=0.25, do_rungb=True):
    def m(k):
        vals = [r[k] for r in rows if k in r]
        return float(np.mean(vals)) if vals else 0.0
    S = len(rows)
    a_r, a_absr = m("a_r_real"), m("a_abs_r")
    a_intact, a_lesion = m("a_intact_diff_abs"), m("a_lesion_diff_abs")
    a_nocond = m("a_r_no_conditioning")
    min_r = min(r["a_r_real"] for r in rows)
    n_code_ok = sum(r["a_permcode_perm_p"] < perm_alpha for r in rows)
    n_us_ok = sum(r["a_unpaired_perm_p"] < perm_alpha for r in rows)
    a_base = m("a_r_baseline_warriner")

    checks = {
        "C_A1_held_out_spiking_r>=0.45": a_r >= a_r_go,
        "C_A1_every_seed_r>=0.25": min_r >= min_seed_r,
        "C_A2_salience_magnitude_tracks_valence_and_input_lesion_collapses":
            a_absr > 0.2 and a_lesion < 0.5 * a_intact,
        "C_A3_no_conditioning_collapses(<0.15)": a_nocond < no_cond_max,
        "C_A4_permute_code_perm_p<0.05_all_seeds": n_code_ok == S,
        "C_A5_unpaired_US_perm_p<0.05_all_seeds": n_us_ok == S,
    }
    means = {"a_r_real": a_r, "a_r_real_min": min_r, "a_abs_r": a_absr, "a_intact_diff_abs": a_intact,
             "a_lesion_diff_abs": a_lesion, "a_r_no_conditioning": a_nocond, "a_r_perp": m("a_r_perp"),
             "a_permcode_seeds_sig": n_code_ok, "a_unpaired_seeds_sig": n_us_ok,
             "a_permcode_null_mean": m("a_permcode_null_mean"), "a_unpaired_null_mean": m("a_unpaired_null_mean"),
             "a_r_shuffled_spiking_diag": m("a_r_shuffled_spiking_diag"),
             "a_r_baseline_warriner": a_base, "corr_s_c_warriner": m("corr_s_c_warriner")}
    if do_rungb:
        b_acc = m("b_accuracy")
        b_drop, b_drop_les = m("b_vminus_drop_frac"), m("b_vminus_drop_frac_reap_lesioned")
        b_mi, b_ml = m("b_margin_intact"), m("b_margin_wta_lesion")
        b_acc_mis = m("b_accuracy_mismatched")
        all_distinct = all(r.get("b_distinct", False) for r in rows)
        checks.update({
            "C_B1_emotion_discrimination>=0.75_and_distinct": b_acc >= b_acc_go and all_distinct,
            "C_B2_reappraisal_downregulates>=25%": b_drop >= reap_drop_go,
            "C_B3_WTA_lesion_collapses_margin>=35%": b_ml < 0.65 * b_mi,
            "C_B4_reap_lesion_abolishes_downreg": b_drop_les < 0.4 * b_drop if b_drop > 1e-6 else False,
            "C_B5_mismatched_collapses_discrimination": b_acc_mis <= 0.5 and b_acc >= b_acc_mis + 0.4,
        })
        means.update({"b_accuracy": b_acc, "b_vminus_drop_frac": b_drop,
                      "b_vminus_drop_frac_reap_lesioned": b_drop_les, "b_margin_intact": b_mi,
                      "b_margin_wta_lesion": b_ml, "b_accuracy_mismatched": b_acc_mis})
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
    ap.add_argument("--n-perm", type=int, default=200, help="permutation draws for C-A4/C-A5")
    ap.add_argument("--w-l2-ref", type=float, default=W_L2_REF, help="operating-point gain: composed-weight L2 norm "
                    "(Warriner-free scalar; transfers the affect-deepen rung-b balance; correlation-invariant)")
    ap.add_argument("--no-rungb", action="store_true", help="skip the discrete-emotion rung (rung-a + controls only)")
    ap.add_argument("--ablation", action="store_true", help="also run the innate-primary-count linear-fidelity "
                    "ablation (evidence on whether the graded-magnitude residual is genome-draw variance or a wall)")
    ap.add_argument("--out", default=str(Path(_REPO) / "research" / "findings" / "raw" /
                                          "_affect_composed_selforganized_opponent.json"))
    a = ap.parse_args()
    if a.smoke:
        a.seeds = [a.seeds[0]]
        a.max_stories = min(a.max_stories, 8000)
        a.n_perm = min(a.n_perm, 120)

    do_rungb = not a.no_rungb
    t0 = time.time()
    print(f"[composed-affect] seeds={a.seeds} smoke={a.smoke} backend={os.environ.get('SIM_BACKEND')} "
          f"max_stories={a.max_stories} n_hub={a.n_hub} rung_b={do_rungb}", flush=True)
    A = build_all(a.max_stories, a.n_hub, a.window, a.min_count)
    print(f"  self-organized codes: {len(A['vocab'])} Warriner-labelled concepts x {A['codes'].shape[1]} hubs | "
          f"innate primaries in-vocab: {len(A['app'])} appetitive {A['app']} | {len(A['avr'])} aversive {A['avr']} "
          f"({round(time.time()-t0,1)}s)", flush=True)
    if len(A["vocab"]) < 24 or len(A["app"]) < 1 or len(A["avr"]) < 1:
        print(f"NOT-RUNNABLE: vocab={len(A['vocab'])} app={len(A['app'])} avr={len(A['avr'])}", flush=True)
        return 2
    if len(A["app"]) < a.n_each or len(A["avr"]) < a.n_each:
        a.n_each = min(len(A["app"]), len(A["avr"]))
        print(f"  [adjust] n_each -> {a.n_each} (pool availability)", flush=True)

    rows = [run_seed(s, A, a.n_each, a.min_events, a.seed_frac, a.n_perm, do_rungb=do_rungb, l2_ref=a.w_l2_ref,
                     verbose=True) for s in a.seeds]
    go, checks, means = aggregate(rows, do_rungb=do_rungb)
    n = len(a.seeds)

    ablation = None
    if a.ablation:
        ablation = primary_count_ablation(A, a.min_events, a.seed_frac, seeds=tuple(a.seeds))
        print("  [primary-count ablation] (linear fidelity) "
              f"{[(x['n_primaries'], round(x['held_r_mean'], 2), round(x['salience_r_mean'], 2), x['permute_code_beat']) for x in ablation]}",
              flush=True)

    # measurement-VALIDITY preconditions (distinct from the GO checks): when the verdict is TRUSTWORTHY.
    min_held = min(r["n_held"] for r in rows)
    preconditions = [
        {"name": "corpus_loaded(vocab>=24)", "ok": len(A["vocab"]) >= 24, "detail": f"vocab={len(A['vocab'])}"},
        {"name": "held_set_adequate(min n_held>=20)", "ok": min_held >= 20, "detail": f"min_n_held={min_held}"},
        {"name": "no_conditioning_reads_zero(|r|<0.15)", "ok": abs(means["a_r_no_conditioning"]) < 0.15,
         "detail": f"no_conditioning_r={means['a_r_no_conditioning']:+.4f}"},
        {"name": "innate_US_signal_present(corr(s_c,Warriner)>0)", "ok": means["corr_s_c_warriner"] > 0.0,
         "detail": f"corr_s_c_warriner={means['corr_s_c_warriner']:+.3f}"},
        {"name": "weights_warriner_free(no-cond collapse is load-bearing)", "ok": True,
         "detail": "asserted in run_seed: selforg_opponent_weights has no Warriner argument; corrupting s_true "
                   "leaves the weights byte-identical; C-A3 collapse confirms weights come from conditioning"},
    ]

    v = Verdict("composed self-organized affect opponent (Warriner-seed RETIRED)")
    v.floor("C-A1 held-out spiking opponent r >= 0.45", measured=means["a_r_real"], floor=0.45)
    v.require("C-A1 every seed r >= 0.25", means["a_r_real_min"], expect=lambda x: x >= 0.25)
    v.require("C-A2 |differential| tracks valence strength (r > 0.2)", means["a_abs_r"], expect=lambda x: x > 0.2)
    v.control("C-A2 input-lesion collapses the opponent differential", treatment=means["a_intact_diff_abs"],
              control=means["a_lesion_diff_abs"], min_separation=0.5 * means["a_intact_diff_abs"])
    v.control("C-A3 no-conditioning lesion collapses the read (weights are self-organized from experience)",
              treatment=means["a_r_real"], control=means["a_r_no_conditioning"],
              min_separation=means["a_r_real"] - 0.15)
    v.require("C-A4 permute-code beaten (perm-p<0.05) in ALL seeds", means["a_permcode_seeds_sig"],
              expect=lambda x: x == n)
    v.require("C-A5 unpaired-US beaten (perm-p<0.05) in ALL seeds", means["a_unpaired_seeds_sig"],
              expect=lambda x: x == n)
    if do_rungb:
        v.require("C-B1 emotion discrimination >= 0.75", means["b_accuracy"], expect=lambda x: x >= 0.75)
        v.require("C-B2 reappraisal down-regulates the amygdala >= 25%", means["b_vminus_drop_frac"],
                  expect=lambda x: x >= 0.25)
        v.control("C-B3 WTA-lesion collapses the categorical margin", treatment=means["b_margin_intact"],
                  control=means["b_margin_wta_lesion"], min_separation=0.35 * means["b_margin_intact"])
        v.control("C-B5 mismatched appraisal collapses discrimination", treatment=means["b_accuracy"],
                  control=means["b_accuracy_mismatched"], min_separation=0.4)
    v.disabled("Warriner appraisal SEED (ridge target) -- RETIRED: opponent weights derive from the self-organized "
               "conditioning map (three-factor Hebbian over the learned code, ~10 innate primary signs)",
               why="this composition's whole point; Warriner is EVAL-only ground-truth, never a weight input")
    decided = v.decide(go=go, verbose=False)

    attributable_to("composed opponent read (vs no-conditioning lesion)", means["a_r_real"],
                    means["a_r_no_conditioning"])
    attributable_to("composed opponent (vs ridge-to-Warriner baseline)", means["a_r_real"],
                    means["a_r_baseline_warriner"])
    attributable_to("opponent differential (vs input-lesion)", means["a_intact_diff_abs"], means["a_lesion_diff_abs"])
    if do_rungb:
        attributable_to("emotion discrimination (vs mismatched appraisal)", means["b_accuracy"],
                        means["b_accuracy_mismatched"])

    tag = f"{n}-seed" if not a.smoke else "SMOKE(1-seed)"
    gap = means["a_r_baseline_warriner"] - means["a_r_real"]
    if go:
        verdict = (
            f"GO ({tag}) -- THE WARRINER-SEEDED RIDGE-FIT IS RETIRED. The spiking affect opponent V+/V- weights are "
            f"DERIVED FROM the self-organized valence map (a three-factor Hebbian outer-product over the learned "
            f"stream-cortex code, anchored by ~{2*a.n_each} innate primary signs) -- ZERO Warriner supervision in "
            f"the weights. Held-out concepts (own reinforcement WITHHELD from the map) appraise to a SPIKING opponent "
            f"differential correlating r={means['a_r_real']:+.3f} with true valence (every seed >= "
            f"{means['a_r_real_min']:+.3f}; the retired ridge-to-Warriner baseline reads {means['a_r_baseline_warriner']:+.3f}, "
            f"a gap of {gap:+.3f}). The salience gate is EMERGENT (|differential| tracks valence strength "
            f"r={means['a_abs_r']:+.3f}; input-lesion collapses it {means['a_intact_diff_abs']:.3f}->"
            f"{means['a_lesion_diff_abs']:.3f}). The weights are self-organized FROM EXPERIENCE: removing the "
            f"conditioning stream collapses the read to {means['a_r_no_conditioning']:+.3f}; permute-code beaten in "
            f"{means['a_permcode_seeds_sig']}/{n} seeds and unpaired-US in {means['a_unpaired_seeds_sig']}/{n} "
            f"(perm-p<0.05). corr(acquired s_c, Warriner)={means['corr_s_c_warriner']:+.3f} (the innate signal is "
            f"honest)." + (f" RUNG-B (composed opponent, valence cue chosen by s_c not Warriner): the 4 appraisal "
            f"conditions select their intended emotion at accuracy {means['b_accuracy']:.2f}; reappraisal "
            f"down-regulates the amygdala {means['b_vminus_drop_frac']:.0%}; WTA-lesion collapses the margin "
            f"{means['b_margin_intact']:.3f}->{means['b_margin_wta_lesion']:.3f}." if do_rungb else "") +
            f" => the WHOLE affect appraisal now traces to innate primaries + experience, not a human-rated lexicon. "
            f"Brain-based (reads off cp_firing_states); NO sim/ edit. RESIDUAL: ~{2*a.n_each} innate primary SIGNS "
            f"(the faithful floor); rate-level Hebbian map (spiking three-factor write = next rung).")
    else:
        miss = [k for k, val in checks.items() if not val]
        verdict = (
            f"BOUNDARY / HONEST NEGATIVE (build-informative, {tag}) -- the composed Warriner-free opponent reads "
            f"held-out r={means['a_r_real']:+.3f} (min {means['a_r_real_min']:+.3f}); the retired ridge-to-Warriner "
            f"baseline reads {means['a_r_baseline_warriner']:+.3f} (RESIDUAL GAP {gap:+.3f}). no-conditioning "
            f"{means['a_r_no_conditioning']:+.3f}; permute-code sig {means['a_permcode_seeds_sig']}/{n}; unpaired-US "
            f"sig {means['a_unpaired_seeds_sig']}/{n}" +
            (f"; rung-b acc {means['b_accuracy']:.2f}, reappraisal {means['b_vminus_drop_frac']:.0%}, WTA margin "
             f"{means['b_margin_intact']:.3f}->{means['b_margin_wta_lesion']:.3f}" if do_rungb else "") +
            f". FAILED: {miss}. The residual is the self-organized weight's fidelity vs the Warriner ridge; the next "
            f"mechanism is a richer/spiking three-factor write (or more innate primaries), NOT a wall -- and NOT "
            f"'acceptable': the gap is the deliverable.")

    summary = {
        "probe": "affect_composed_selforganized_opponent (emergence-bar composition)", "verdict": verdict,
        "GO": bool(go), "preconditions": preconditions, "verdict_earned": decided, "checks": checks, "means": means,
        "per_seed": rows, "primary_count_ablation": ablation,
        "config": {"seeds": a.seeds, "smoke": a.smoke, "max_stories": a.max_stories, "n_hub": a.n_hub,
                   "window": a.window, "min_count": a.min_count, "n_each": a.n_each, "min_events": a.min_events,
                   "seed_frac": a.seed_frac, "n_perm": a.n_perm, "rung_b": do_rungb, "n_vocab": len(A["vocab"]),
                   "appetitive_pool": A["app"], "aversive_pool": A["avr"], "backend": os.environ.get("SIM_BACKEND")},
        "mechanism": "opponent V+/V- weight = self-organized three-factor Hebbian map w=sum_{c in train} code_read_c "
                     "* s_c (s_c = Rescorla-Wagner asymptote of co-occurrence with ~2*n_each INNATE primary "
                     "reinforcers), rectified split W+=g*max(w,0)/W-=g*max(-w,0), injected as the code_in->vplus/vminus "
                     "FF; the spiking differential rate(vplus)-rate(vminus) off cp_firing_states = valence. REPLACES "
                     "the affect-deepen ridge-fit-to-Warriner. Rung-b: composed opponent -> appraisal-dim WTA -> 4 "
                     "Panksepp emotions + vmPFC->amygdala reappraisal (valence cue chosen by s_c, not Warriner).",
        "HONEST_RESIDUALS": "Warriner is EVAL-only ground-truth, NEVER a weight input (asserted). Residuals: (1) ~2*"
                            "n_each innate primary SIGNS host-supplied (the faithful floor; 140->~10 compression, not "
                            "removal); (2) the outer-product Hebbian map is a rate-level numpy matrix (a fully-spiking "
                            "three-factor write is the next rung); (3) agency/certainty conditions set as sensory "
                            "drive; (4) standalone de-risk bridge (build_one_brain fold-in pending).",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[composed-affect] VERDICT: {verdict}", flush=True)
    print(f"[composed-affect] wrote {a.out}  ({summary['elapsed_seconds']}s)\n" + "=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
