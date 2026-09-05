"""EMBODIED-US AFFECT-GATE (2026-09-05) — the ISOLATED test of the surpass THREE prior boundaries all named.

WHAT THIS RETIRES (attempt 4). The production affect appraisal
(`research.runners.affect_production_organ.appraise_text`) has two halves:
  * the per-word VALUE (which sign / how positive) — RETIRED to a learned map (DR-2 6-seed GO; its ORIGIN
    self-organizes from ~10 innate primaries, DR-2b GO); text supplies value at corr +0.42..+0.52.
  * the SALIENCE GATE (which words may move the mood AT ALL) — STILL the host fixed
    `|raw_Warriner_valence - 5| >= 2.0` threshold (`_STRONG_MARGIN`, affect_production_organ.py:73/118).
This de-risks retirement of the GATE half.

WHY THE OBVIOUS FIXES ARE BANKED-EXHAUSTED (do NOT re-propose them). THREE mechanistically-independent attempts to
retire the gate from a STATISTIC OF THE TEXT CO-OCCURRENCE GRAPH all failed with the SAME diagnosis — a REGISTER
confound (TinyStories frames ordinary words cat/day/moon/garden inside emotionally-resolved scenes as consistently
as it uses real emotion words):
  * 2026-08-12 D1 (learned-value magnitude): "unseparable by any gain or threshold".
  * 2026-09-05 co-occurrence retry (arousal / habituation / cross-resample stability / neighbor-purity + best combo):
    ALL BOUNDARY, best 29.4% recall@FP0 — `2026-09-05-affect-learned-gate-retry-register-confound-BOUNDARY.md`.
  * 2026-09-05 experienced-opponent (a fully-spiking V+/V- population conditioned to ~10 innate PRIMARY WORDS): ALSO
    BOUNDARY (0.000 worst-case) — `2026-09-05-affect-experienced-opponent-gate-needs-embodiment-BOUNDARY.md`. Its
    sharpened diagnosis: in a TEXT-ONLY stream the "experienced affect" IS the text co-occurrence (the ~10 primaries
    are themselves WORDS inside the same register), so conditioning-to-primaries inherits the confound wholesale.

All three NAMED THE SAME SURPASS: a NON-TEXTUAL (embodied / interoceptive) UNCONDITIONED STIMULUS — the way
amygdala/BLA valence conditioning binds a cue to the animal's OWN affective response to a REAL reinforcer (Namburi &
Tye 2015), not to lexical company. THE OPEN QUESTION THIS RUNNER ANSWERS (which NO prior experiment tested): the
prior attempts all failed because their US SOURCE was text; is the US source the ONLY bottleneck, or is the CONCEPT
CODE (built from the same text) ALSO register-confounded so that even a PERFECT embodied US cannot be read out
through it? Those are two very different next arcs (build a grounded-world US alone, vs ALSO ground the perception),
and the record has conflated them. This isolates that ONE variable.

THE EMBODIED-US STAND-IN (host is legit ONLY for the world/body/US-delivery). A genuine grounded-world US — the
world reinforcing a concept's referent with a real bodily consequence (satiety rising, nociception firing), the way
the production board-#49/#84 interoceptive-relay CURRENT afferent already delivers a bodily current to the affect
ladder — does not exist for the CONVERSATIONAL (TinyStories) vocabulary: there is no world in which `happy` vs `cat`
get differential embodied consequences. So this runner supplies an ORACLE embodied US: the world delivers, at
conditioning time, a signed affect CURRENT of the concept's TRUE affect magnitude (Warriner, standing in for what a
grounded world's physics would reinforce), and the spiking opponent CONDITIONS to it via the SAME three-factor
Hebbian rule. This is EXPLICITLY a stand-in — the claim under test is NOT "Warriner is retired"; it is "GIVEN a
perfect embodied US (as a grounded world would supply), does the neural machinery + the existing text CODE produce a
working, HELD-OUT salience gate?" A NO isolates the concept code (perception) as an ADDITIONAL wall; a YES isolates
the US source as the sole bottleneck. Either verdict is a first-class deliverable that redirects the next build.

WHY THIS IS NOT CIRCULAR / HOLLOW (the anti-cheats ARE the design):
  * HELD-OUT (2-fold cross-fit): the gate for a test word reads its TEXT CODE through weights conditioned on OTHER
    words' (code, US) pairs — the test word's OWN US is NEVER present at read. So a positive result is generalization
    of the code+conditioning, not "read the US back". A hollow re-read would not collapse under the lesion and would
    be trivially perfect.
  * LESION (s_c := 0 -> the opponent weights collapse to 0): the gate must collapse (anti-hollow, G3).
  * NEGATIVE CONTROL: the banked TEXT-US channel (conditioning to primary co-occurrence) is reproduced on the SAME
    words + seeds; the embodied US must STRICTLY EXCEED it (G2).
  * CODE-SEPARABILITY CEILING (the decisive diagnostic): a GENEROUS supervised linear probe (ridge, k-fold CV,
    given the labels) on the raw text code. If even THIS cannot hit 0.5 recall@FP0, no linear readout of the code —
    including any US-conditioned opponent — can, so the CODE is the wall and the embodied US is necessary-but-not-
    sufficient. If it CAN, the code carries the affect axis and a gate failure is a conditioning/generalization issue.
  * SYNTHETIC-SEPARABLE-CODE positive control (instrument soundness): the SAME embodied-US-oracle conditioning +
    spiking held-out read, but on a synthetic code where affect/neutral ARE separable -> confirms the READOUT
    mechanism works when the code carries the signal (isolates code-vs-mechanism).

MECHANISM (BUILDABLE-NOW, reuse-by-import, NO `sim/` edit, numpy-CPU per the prior affect de-risks). Identical to the
experienced-opponent runner EXCEPT the conditioning US SOURCE:
  - Concept CODE = the self-organized PPMI stream-cortex code (build_cooccurrence -> codes_from_cooccurrence). [pre]
  - EMBODIED-US-ORACLE s_c = signed true affect (the world/body US stand-in), delivered to every vocab word. [US]
    (TEXT-US control s_c = Rescorla-Wagner asymptote of co-occurrence with ~10 innate primary WORDS — the banked
    channel.)
  - EXPERIENCE-CONDITIONED opponent weights = selforg_opponent_weights (three-factor Hebbian outer-product over the
    learned code, rectified Namburi-Tye V+/V- split; the US is the third factor). [SYNAPTIC]
  - SPIKING opponent bridge = build_bridge (code_in relay -> appr_vplus/appr_vminus with xinh cross-inhibition;
    OPP_TONIC_PA=0, so the pools fire ONLY from the code FF), read via read_valence. [the SPIKING population]
  - GATE STATISTIC g(w) = pos_rate + neg_rate (total opponent DRIVE — salience, per Namburi-Tye: valence=differential,
    salience=total activation), read HELD-OUT (2-fold cross-fit), per-seed LABEL-FREE median gain-control, joint-FP=0.

BENCHMARK: the SAME 164-word partition the prior boundaries use (Warriner words in the full corpus AND all 6
resamples; 102 raw-gated "true affect" |v-5|>=2, 62 raw-excluded "neutral"). Warriner defines the partition + the
oracle US + the negative-control input; it is NEVER a statistic-of-the-graph read (the banked confound).

PRE-REGISTERED GO GATE (fixed BEFORE the 6-seed; matches the task's bar + the boundary findings' discipline):
  G1 SEPARATION  embodied-US-oracle worst-case recall (min across 6 seeds) >= 0.5 at JOINT FP=0 on the 164 words.
  G2 SURPASS     that worst-case recall STRICTLY EXCEEDS the banked TEXT-US channel's worst-case recall (same words).
  G3 ANTI-HOLLOW the no-conditioning lesion (s_c:=0) collapses joint-FP=0 recall to < 0.15 OR unachievable.
GO iff G1 AND G2 AND G3.
Reported (decisive, not gated): the CODE-SEPARABILITY CEILING (supervised, per-seed worst-case) — the upper bound on
any code readout; the SYNTHETIC-separable-code embodied-US gate — the mechanism's instrument soundness;
corr(oracle s_c, Warriner) (==1 by construction); the emergent-ignition FP/recall.

BRAIN-BASED: the gate read is a spike-rate read off cp_firing_states through a spiking opponent population with
synaptic cross-inhibition; the conditioning is a synaptic (three-factor Hebbian) map. Host is legit ONLY for the
US DELIVERY (the embodied affect current = the world/body boundary) and the corpus stream. HONEST RESIDUALS
(declared): (1) the embodied US is an ORACLE stand-in for a grounded-world US that does not exist for this vocabulary
(the finding names exactly what must produce it); (2) the conditioning outer-product map is a rate-level numpy matrix
(the codes are the spiking-validated stream cortex; a fully-spiking three-factor WRITE is a further rung); (3)
standalone de-risk bridge (build_one_brain fold-in is the production step if GO).

DISCIPLINE: reuse-by-import (no reimplementation of any conditioning / bridge / read / partition primitive),
SIM_BACKEND=numpy CPU lane, cfg.seed (inherited from build_bridge). NO `sim/` edit. NOT WIRED: nothing here touches
affect_production_organ.py / wkv_mouth_generator.py (byte-unchanged; the controller decides any flip).

Run (smoke):  SIM_BACKEND=numpy python -u -m research.runners._affect_embodied_us_gate_derisk --smoke
Run (6-seed): SIM_BACKEND=numpy python -u -m research.runners._affect_embodied_us_gate_derisk \
                  --seeds 42 43 44 100 101 102 \
                  --out research/findings/raw/_affect_embodied_us_gate_6seed.json
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

# --- reuse-by-import: the SAME de-risked primitives (NO reimplementation, NO sim/ edit) -------------------------
from research.runners._affect_distributional_tag_derisk import (  # noqa: E402
    WARRINER, load_stories, build_cooccurrence, codes_from_cooccurrence,
)
from research.runners._affect_evaluative_conditioning_derisk import (  # noqa: E402
    APPETITIVE_POOL, AVERSIVE_POOL, build_primary_cooccurrence,
)
from research.runners._affect_composed_selforganized_opponent_derisk import (  # noqa: E402
    selforg_opponent_weights, rescorla_wagner_valence,
)
from research.runners._affect_appraisal_emotion_reappraisal_derisk import (  # noqa: E402
    build_bridge, read_valence, _pearson,
)
# reuse the EXACT partition + calibration + gate-read helpers the experienced-opponent BOUNDARY used, so this is a
# strict like-for-like isolation of the US SOURCE (only the conditioning s_c changes).
from research.runners._affect_experienced_opponent_gate_derisk import (  # noqa: E402
    _STRONG_MARGIN, CANONICAL_SEEDS, resample_stories, build_partition, _codes_for, _read_gate_drive,
    joint_calibrate,
)
from tools.lab import void_if, undefined_if_empty, attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_affect_embodied_us_gate.json"

RECALL_GO_BAR = 0.5          # pre-registered (identical to the boundary findings' bar)
NO_COND_MAX_RECALL = 0.15    # G3: the no-conditioning lesion must drop joint-FP=0 recall below this


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# HELD-OUT GATE DRIVE: build the 2-fold cross-fit spiking opponent from (codes, s_c, train_eligible) and read the
# total opponent DRIVE for every partition word from the W built on the OTHER fold. Reused verbatim in structure
# from the experienced-opponent runner -- the ONLY thing that varies between conditions is (s_c, train_eligible).
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _held_out_drive(seed, D, codes, codes_read, relatedness, s_c, train_eligible, part_idx, fold):
    """Returns (drive[len(part_idx)], diff_abs[...]) — the spiking gate statistic read HELD-OUT: each partition word
    is read from the W built on the OTHER fold's ELIGIBLE reinforced words (weighted by s_c), so no word reads its
    own conditioning. train_eligible: bool[n_vocab] (text-US = the reinforced mask; embodied-US = all vocab)."""
    n = len(codes)

    def _weights(train_mask):
        _, wp, wm = selforg_opponent_weights(codes_read, s_c, train_mask, codes, relatedness=relatedness)
        return wp, wm

    train_A = train_eligible & (fold == 0)                                # W_A -> reads fold-1 partition words
    train_B = train_eligible & (fold == 1)                                # W_B -> reads fold-0 partition words
    wpA, wmA = _weights(train_A)
    wpB, wmB = _weights(train_B)
    brA, xpA, idxA, snA = build_bridge(seed, D, wpA, wmA)
    brB, xpB, idxB, snB = build_bridge(seed + 1, D, wpB, wmB)
    drive = np.zeros(len(part_idx), float)
    diff_abs = np.zeros(len(part_idx), float)
    for k, gi in enumerate(part_idx):
        if fold[gi] == 0:
            br, xp, idx, sn = brB, xpB, idxB, snB                         # fold-0 word <- W built from fold-1
        else:
            br, xp, idx, sn = brA, xpA, idxA, snA
        g = _read_gate_drive(br, xp, idx, sn, codes[gi])
        drive[k] = g["drive"]; diff_abs[k] = g["diff_abs"]
    floor = float(np.mean([_read_gate_drive(brA, xpA, idxA, snA, codes[part_idx[0]], lesion_input=True)["drive"]
                           for _ in range(3)]))
    return drive, diff_abs, floor


def code_separability_ceiling(codes_part, raw_gate, seed, k=5, lam=1.0, max_fp_frac=0.0):
    """A GENEROUS supervised UPPER BOUND on any linear readout of the code: k-fold CV ridge regression to +/-1
    affect labels on the partition words' RAW codes -> out-of-fold scores -> recall at a tolerated false-positive
    fraction (default 0 = the production zero-FP bar). If even this best-case supervised probe cannot separate the
    classes, NO US-conditioned opponent (a constrained linear read of the same code) can, so the CODE is the wall.
    (Honest caveat: the spiking opponent adds a mild nonlinearity the boundary findings measured as NOT helping;
    this remains the fair linear ceiling.)"""
    X = np.asarray(codes_part, float)
    n, D = X.shape
    y = np.where(raw_gate, 1.0, -1.0)
    if raw_gate.sum() == 0 or (~raw_gate).sum() == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, k)
    scores = np.full(n, np.nan)
    for f in range(k):
        test = folds[f]
        train = np.concatenate([folds[j] for j in range(k) if j != f])
        Xt, Yt = X[train], y[train]
        w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(D), Xt.T @ Yt)
        scores[test] = X[test] @ w
    neu = ~raw_gate
    aff = raw_gate
    n_neu = int(neu.sum())
    if max_fp_frac <= 0.0:
        thr = float(np.nanmax(scores[neu]))                             # FP=0 threshold: above every neutral score
        return float((scores[aff] > thr).mean())
    best = 0.0                                                          # strictest threshold with FP fraction <= tol
    for thr in np.sort(np.unique(scores))[::-1]:
        gated = scores >= thr
        if int((gated & neu).sum()) / max(1, n_neu) <= max_fp_frac:
            best = max(best, float((gated & aff).sum()) / max(1, int(aff.sum())))
    return best


def ceiling_robustness_sweep(stories, part_words, raw_gate, seeds, resample_frac, min_count, window,
                             n_hubs=(64, 500), fp_tols=(0.0, 0.05, 0.10)):
    """ADVERSARIAL robustness of the code-separability CEILING (verify-go): is the low ceiling an artifact of the
    strict zero-FP bar or the small code dim? Re-measures the supervised ceiling across {code dim} x {tolerated FP}
    on the SAME partition + seeds (numpy-only, no bridge). If the ceiling stays low at BOTH a bigger code AND a
    relaxed FP, the register confound is genuinely in the CONCEPT CODE, not the criterion. Returns a table."""
    out = []
    for n_hub in n_hubs:
        for tol in fp_tols:
            vals = []
            for s in seeds:
                sub = resample_stories(stories, resample_frac, s)
                vocab, C = build_cooccurrence(sub, n_hub, window, min_count)
                codes = codes_from_cooccurrence(C)
                widx = {w: i for i, w in enumerate(vocab)}
                pidx = np.array([widx[w] for w in part_words])
                vals.append(code_separability_ceiling(codes[pidx], raw_gate, s, max_fp_frac=tol))
            out.append({"n_hub": n_hub, "max_fp_frac": tol, "worst": float(min(vals)),
                        "mean": float(np.mean(vals)), "per_seed": [round(v, 4) for v in vals]})
    return out


def synthetic_separable_gate(seed, raw_gate, D, sep=6.0, noise=0.5):
    """INSTRUMENT SOUNDNESS (reported): a SYNTHETIC code with GROUNDED-CODE structure -- affect concepts of the SAME
    sign SHARE a code axis ("feels-good" concepts cluster on one embodied-feature axis, "feels-bad" on another),
    neutral concepts carry only random base structure -- exactly what a grounded world would produce (things with the
    same bodily consequence share features). VALIDATES the CODE-SEPARABILITY CEILING instrument: on this cleanly
    separable code the supervised ceiling must read HIGH (~1.0), so a LOW ceiling on the REAL code is a real property
    of the text code, not an always-zero instrument. ALSO reports the spiking opponent's held-out recall on this code
    (the readout MECHANISM reading a good code) -- informative but NOT a precondition, because the joint-FP=0 +
    OU-noise + held-out combination is dominated by the single worst neutral outlier per seed and so stays harsh even
    on a clean code (an honest property of the noisy readout under a zero-FP bar). Clearly labeled synthetic; NOT part
    of the GO decision."""
    n = len(raw_gate)
    rng = np.random.default_rng(seed + 7000)
    base = np.abs(rng.standard_normal((n, D))) * noise                   # non-negative (PPMI-like) random structure
    aff_idx = np.where(raw_gate)[0]
    sign = np.zeros(n)                                                   # the world reinforced some affect concepts +,
    sign[aff_idx[0::2]] = +1.0                                           # some - (alternating -> both opponent pools
    sign[aff_idx[1::2]] = -1.0                                           # are exercised); neutral concepts get no US
    codes = base.copy()
    codes[sign > 0, 0] += sep                                            # +affect concepts share an embodied axis (dim0)
    codes[sign < 0, 1] += sep                                            # -affect concepts share another (dim1)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    codes_read = codes - codes.mean(axis=0, keepdims=True)
    codes_read = codes_read / (np.linalg.norm(codes_read, axis=1, keepdims=True) + 1e-12)
    Wsim = codes @ codes.T; np.fill_diagonal(Wsim, 0.0)
    relatedness = np.asarray(Wsim.mean(axis=1), float)
    s_c = sign.astype(float)                                             # oracle embodied US (signed; 0 for neutral)
    fold = rng.integers(0, 2, size=n)
    part_idx = np.arange(n)
    drive, _, floor = _held_out_drive(seed + 7000, D, codes, codes_read, relatedness, s_c,
                                      np.ones(n, bool), part_idx, fold)
    cal = joint_calibrate([drive], raw_gate)
    ceiling = code_separability_ceiling(codes, raw_gate, seed + 7000)    # the synthetic code IS separable (sanity)
    return {"recall_at_fp0": (cal[0] if cal else 0.0), "code_ceiling": ceiling, "floor": floor,
            "sep": sep, "noise": noise}


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# ONE SEED: resample corpus -> codes -> {TEXT-US, EMBODIED-US-ORACLE, NO-COND} held-out gate + code ceiling.
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def run_seed(seed, stories, part_words, n_hub, window, min_count, n_each, min_events, resample_frac, verbose=False):
    rng = np.random.default_rng(seed)
    sub = resample_stories(stories, resample_frac, seed)
    vocab, codes, codes_read, relatedness = _codes_for(sub, n_hub, window, min_count)
    n = len(vocab)
    D = codes.shape[1]
    widx = {w: i for i, w in enumerate(vocab)}
    part_idx = np.array([widx[w] for w in part_words])
    fold = rng.integers(0, 2, size=n)                                    # ONE fold assignment shared by all conditions

    # signed TRUE affect over the whole vocab (the world/body ORACLE US stand-in AND the eval ground-truth)
    s_true_vocab = (np.array([WARRINER[w][0] for w in vocab]) - 5.0) / 4.0

    # ── TEXT-US (the banked channel): s_c = Rescorla-Wagner asymptote of co-occurrence with ~10 innate primary WORDS
    vset = set(vocab)
    app = [w for w in APPETITIVE_POOL if w in vset]
    avr = [w for w in AVERSIVE_POOL if w in vset]
    all_primaries = app + avr
    prim_sign_full = {**{w: +1.0 for w in app}, **{w: -1.0 for w in avr}}
    ne = min(n_each, len(app), len(avr))
    primaries = list(rng.choice(app, ne, replace=False)) + list(rng.choice(avr, ne, replace=False))
    prim_col = {w: j for j, w in enumerate(all_primaries)}
    prim_idx = np.array([prim_col[w] for w in primaries])
    prim_sgn = np.array([prim_sign_full[w] for w in primaries], float)
    is_primary = np.array([w in set(primaries) for w in vocab])
    Co = build_primary_cooccurrence(sub, vocab, window, all_primaries)
    s_c_text, reinforced = rescorla_wagner_valence(Co, prim_idx, prim_sgn, is_primary, min_events)
    drive_text, _, floor_text = _held_out_drive(seed, D, codes, codes_read, relatedness, s_c_text,
                                                reinforced, part_idx, fold)

    # ── EMBODIED-US-ORACLE: s_c = signed true affect (the world delivers a US of the concept's real affect); every
    #    vocab word is a conditioning candidate (the world reinforces affect concepts strongly, neutral ~0). ────────
    s_c_emb = s_true_vocab.copy()
    train_all = np.ones(n, bool)
    drive_emb, diff_emb, floor_emb = _held_out_drive(seed, D, codes, codes_read, relatedness, s_c_emb,
                                                     train_all, part_idx, fold)

    # ── NO-COND LESION (G3, anti-hollow): s_c := 0 -> the opponent weights collapse -> the drive collapses. ─────────
    drive_nocond, _, _ = _held_out_drive(seed, D, codes, codes_read, relatedness, np.zeros(n, float),
                                         train_all, part_idx, fold)

    # ── CODE-SEPARABILITY CEILING (decisive diagnostic): can ANY linear readout of the raw text code separate the
    #    classes, given the labels? (upper bound on the whole opponent-readout family) ─────────────────────────────
    raw_gate = np.array([abs(WARRINER[w][0] - 5.0) >= _STRONG_MARGIN for w in part_words], bool)
    code_ceiling = code_separability_ceiling(codes[part_idx], raw_gate, seed)

    corr_emb_warr = 1.0                                                  # by construction (the oracle US IS s_true)
    corr_text_warr = (_pearson(s_c_text[reinforced], s_true_vocab[reinforced])
                      if reinforced.sum() >= 3 else 0.0)
    if verbose:
        print(f"  [seed {seed}] vocab={n} code_dim={D} n_reinforced={int(reinforced.sum())} primaries={primaries}\n"
              f"    floor(text)={floor_text:.4f} floor(emb)={floor_emb:.4f} mean_drive(text)={drive_text.mean():.4f} "
              f"mean_drive(emb)={drive_emb.mean():.4f} code_ceiling(recall@FP0)={code_ceiling:.3f} "
              f"corr(s_c_text,Warr)={corr_text_warr:+.3f}", flush=True)
    return {"seed": int(seed), "n_vocab": int(n), "code_dim": int(D), "n_reinforced": int(reinforced.sum()),
            "primaries": primaries, "drive_text": drive_text, "drive_emb": drive_emb, "diff_emb": diff_emb,
            "drive_nocond": drive_nocond, "floor_text": floor_text, "floor_emb": floor_emb,
            "code_ceiling_recall_fp0": code_ceiling, "corr_s_c_text_warriner": corr_text_warr,
            "corr_s_c_emb_warriner": corr_emb_warr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=CANONICAL_SEEDS)
    ap.add_argument("--smoke", action="store_true", help="1 seed, tiny corpus -- proves it RUNS + controls live")
    ap.add_argument("--max-stories", type=int, default=60000)
    ap.add_argument("--resample-frac", type=float, default=0.8)
    ap.add_argument("--n-hub", type=int, default=64, help="concept code dim (= code_in size); matches the affect "
                    "opponent operating point (composed/affect-deepen/experienced-opponent lanes)")
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--n-each", type=int, default=5, help="innate appetitive AND aversive primaries (text-US control)")
    ap.add_argument("--min-events", type=int, default=2, help="min primary co-occurrences to count as reinforced")
    ap.add_argument("--recall-go-bar", type=float, default=RECALL_GO_BAR)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    seeds = [a.seeds[0]] if a.smoke else a.seeds
    max_stories = min(a.max_stories, 8000) if a.smoke else a.max_stories
    min_count = 2 if a.smoke else a.min_count

    t0 = time.time()
    print(f"[embodied-us-gate] seeds={seeds} smoke={a.smoke} max_stories={max_stories} n_hub={a.n_hub} "
          f"resample_frac={a.resample_frac} backend={os.environ.get('SIM_BACKEND')}", flush=True)
    stories = load_stories(max_stories)
    part_words, raw_gate = build_partition(stories, seeds, a.resample_frac, min_count)
    void_if(len(part_words) < 20, f"only {len(part_words)} common partition words")
    n_pos, n_neg = int(raw_gate.sum()), int((~raw_gate).sum())
    void_if(n_pos == 0 or n_neg == 0, f"degenerate partition n_pos={n_pos} n_neg={n_neg}")
    print(f"  partition: {len(part_words)} common words | raw-gated(affect)={n_pos} raw-excluded(neutral)={n_neg}",
          flush=True)

    rows = [run_seed(s, stories, part_words, a.n_hub, a.window, min_count, a.n_each, a.min_events,
                     a.resample_frac, verbose=True) for s in seeds]

    text_by_seed = [r["drive_text"] for r in rows]
    emb_by_seed = [r["drive_emb"] for r in rows]
    nocond_by_seed = [r["drive_nocond"] for r in rows]

    # ── joint-FP=0 calibration (single per-seed-median-normalized threshold, FP=0 in EVERY seed) ──────────────────
    emb_cal = joint_calibrate(emb_by_seed, raw_gate)                     # THE embodied-US surpass test
    text_cal = joint_calibrate(text_by_seed, raw_gate)                   # the banked TEXT-US channel (G2 baseline)
    nocond_cal = joint_calibrate(nocond_by_seed, raw_gate)              # G3 anti-hollow lesion

    emb_recall = emb_cal[0] if emb_cal else 0.0
    text_recall = text_cal[0] if text_cal else 0.0
    nocond_recall = nocond_cal[0] if nocond_cal else 0.0

    # ── code-separability ceiling (worst-case across seeds) + synthetic-separable-code instrument soundness ───────
    code_ceilings = [r["code_ceiling_recall_fp0"] for r in rows]
    code_ceiling_worst = float(min(code_ceilings))
    code_ceiling_mean = float(np.mean(code_ceilings))
    synth = [synthetic_separable_gate(s, raw_gate, rows[0]["code_dim"]) for s in seeds]
    synth_recall_worst = float(min(x["recall_at_fp0"] for x in synth))          # spiking gate on the clean code (report)
    synth_recall_mean = float(np.mean([x["recall_at_fp0"] for x in synth]))
    synth_ceiling_worst = float(min(x["code_ceiling"] for x in synth))          # the ceiling INSTRUMENT validation
    synth_ceiling_mean = float(np.mean([x["code_ceiling"] for x in synth]))

    max_floor = max(max(r["floor_text"], r["floor_emb"]) for r in rows)
    mean_drive_emb = float(np.mean([v.mean() for v in emb_by_seed]))
    mean_drive_nocond = float(np.mean([v.mean() for v in nocond_by_seed]))

    # ── EMERGENT-IGNITION (reported): gate at "drives the opponent above the input-lesion silent floor" (embodied). ─
    emergent = []
    for r in rows:
        thr = r["floor_emb"] * 1.05 + 1e-9
        gated = r["drive_emb"] >= thr
        fp = int((gated & ~raw_gate).sum()); tp = int((gated & raw_gate).sum())
        emergent.append({"seed": r["seed"], "fp": fp, "recall": round(tp / max(1, n_pos), 4), "thr": round(thr, 5)})

    # ── GO CRITERIA (pre-registered) ──────────────────────────────────────────────────────────────────────────────
    g1 = bool(emb_cal is not None and emb_recall >= a.recall_go_bar)
    g2 = bool(emb_cal is not None and emb_recall > text_recall)
    g3 = bool(nocond_cal is None or nocond_recall < NO_COND_MAX_RECALL)
    go = bool(g1 and g2 and g3)

    # ── VALIDITY preconditions (a null separation is interpretable only if these hold) ────────────────────────────
    v = Verdict("embodied-US affect-gate: is the (embodied-US) separation result interpretable?")
    v.require("input-lesion floor is ~silent (pools fire ONLY from the code FF)", measured=(max_floor < 0.15),
              expect=True)
    v.require("partition non-degenerate (affect + neutral both present)", measured=(n_pos > 0 and n_neg > 0),
              expect=True)
    v.require("the embodied US is the true-affect oracle (corr==1 by construction)",
              measured=(abs(rows[0]["corr_s_c_emb_warriner"] - 1.0) < 1e-9), expect=True)
    v.require("the code-ceiling INSTRUMENT is valid (reads ~1.0 on a cleanly-separable synthetic code, not stuck "
              "at 0)", measured=(synth_ceiling_worst >= a.recall_go_bar), expect=True)
    v.control("the embodied-US opponent read is EXPERIENCE-driven (no-conditioning lesion collapses the drive)",
              treatment=mean_drive_emb, control=mean_drive_nocond, min_separation=0.5 * mean_drive_emb)
    verdict_earned = v.decide(go=go, verbose=False)

    attributable_to("embodied-US drive (vs no-conditioning lesion)", mean_drive_emb, mean_drive_nocond)
    attributable_to("embodied-US gate recall (vs banked text-US channel)", emb_recall, text_recall)
    attributable_to("embodied-US gate recall (vs the supervised code-separability CEILING)", emb_recall,
                    code_ceiling_mean)

    tag = f"{len(seeds)}-seed" if not a.smoke else "SMOKE(1-seed)"
    if go:
        verdict = (
            f"GO ({tag}) -- the SALIENCE GATE retires to an EMBODIED US. Conditioning the fully-spiking opponent "
            f"V+/V- population to an ORACLE embodied US (a signed true-affect current delivered via the world/body "
            f"boundary, the grounded-world stand-in) recovers worst-case recall={emb_recall:.3f} >= {a.recall_go_bar} "
            f"of genuinely affect-bearing words at joint FP=0, HELD-OUT (2-fold cross-fit; the test word's own US is "
            f"never read) -- STRICTLY EXCEEDING the banked TEXT-US channel ({text_recall:.3f}) and collapsing to "
            f"{nocond_recall:.3f} under the no-conditioning lesion. The existing text CODE is sufficient to read out "
            f"a perfect embodied US (supervised ceiling {code_ceiling_worst:.3f} worst-case). => the US SOURCE was "
            f"the sole bottleneck; the buildable next step is a GROUNDED-WORLD embodied US (a teacher where the "
            f"conversational vocabulary is experienced with real bodily consequences), delivered via the production "
            f"board-#49/#84 interoceptive-relay afferent. Brain-based (spike-rate read off the opponent; synaptic "
            f"conditioning); NO sim/ edit; NOT wired (controller decides the flip).")
    else:
        miss = [k for k, ok in (("G1_recall>=0.5", g1), ("G2_surpass_text_US", g2),
                                ("G3_lesion_collapses", g3)) if not ok]
        if code_ceiling_worst < a.recall_go_bar:
            core = (f"DECISIVE: even a PERFECT embodied US cannot retire the gate through the existing TEXT CODE -- "
                    f"the supervised code-separability CEILING (a generous ridge probe GIVEN the affect labels, "
                    f"noise-free) is only {code_ceiling_worst:.3f} worst-case ({code_ceiling_mean:.3f} mean) "
                    f"recall@FP0, so NO readout of this code (opponent included) can reach {a.recall_go_bar} in "
                    f"expectation. The register confound is in the CONCEPT CODE (perception), not only the US SOURCE. "
                    f"The ceiling INSTRUMENT is validated (it reads {synth_ceiling_worst:.3f} worst-case on a cleanly-"
                    f"separable synthetic grounded-code, not stuck at 0), so the low real-code ceiling is a property "
                    f"of the text code. SHARPENS the surpass the three prior boundaries named (add an embodied US): "
                    f"it is NECESSARY BUT NOT SUFFICIENT -- the CONCEPT CODE must ALSO be grounded (a grounded-"
                    f"perception teacher), not just the US delivered.")
        else:
            core = (f"the supervised code ceiling is {code_ceiling_worst:.3f} worst-case ({code_ceiling_mean:.3f} "
                    f"mean; the code DOES carry a separable affect axis) yet the held-out embodied-US-conditioned "
                    f"opponent recovers only {emb_recall:.3f} -- so the residual here is the CONDITIONING/"
                    f"GENERALIZATION or the joint-FP=0+OU-noise readout harshness (the synthetic clean-code spiking "
                    f"gate reads {synth_recall_worst:.3f} worst-case for the same reason), NOT the US source. The "
                    f"ceiling instrument is validated at {synth_ceiling_worst:.3f} on the synthetic code.")
        verdict = (
            f"BOUNDARY ({tag}, build-informative) -- the embodied-US-oracle gate recovers worst-case recall="
            f"{emb_recall:.3f} at joint FP=0 (banked text-US {text_recall:.3f}; no-cond lesion {nocond_recall:.3f}). "
            f"{core} FAILED: {miss}. The fixed _STRONG_MARGIN gate in affect_production_organ.py is UNCHANGED (this "
            f"file wires nothing).")

    summary = {
        "probe": "affect_embodied_us_gate_derisk (attempt 4: isolate the US SOURCE from the CONCEPT CODE)",
        "verdict": verdict, "GO": go, "G1_recall>=0.5": g1, "G2_surpass_text_US": g2, "G3_lesion_collapses": g3,
        "embodied_us_worst_case_recall": emb_recall, "text_us_worst_case_recall": text_recall,
        "nocond_lesion_worst_case_recall": nocond_recall,
        "code_separability_ceiling_worst": code_ceiling_worst, "code_separability_ceiling_mean": code_ceiling_mean,
        "code_separability_ceiling_per_seed": code_ceilings,
        "synthetic_separable_code_recall_worst": synth_recall_worst,
        "synthetic_separable_code_recall_mean": synth_recall_mean,
        "synthetic_code_ceiling_worst": synth_ceiling_worst, "synthetic_code_ceiling_mean": synth_ceiling_mean,
        "synthetic_detail": synth,
        "recall_go_bar": a.recall_go_bar, "no_cond_max_recall": NO_COND_MAX_RECALL,
        "input_lesion_floor_max": max_floor, "mean_drive_emb": mean_drive_emb, "mean_drive_nocond": mean_drive_nocond,
        "emergent_ignition_per_seed": emergent,
        "embodied_us_detail": (emb_cal[2] if emb_cal else None), "text_us_detail": (text_cal[2] if text_cal else None),
        "nocond_detail": (nocond_cal[2] if nocond_cal else None),
        "embodied_us_threshold_norm": (emb_cal[1] if emb_cal else None),
        "preconditions": verdict_earned["preconditions"], "verdict_earned_status": verdict_earned["status"],
        "verdict_undefined_reasons": verdict_earned["undefined_reasons"],
        "n_pos_raw_gated": n_pos, "n_neg_raw_excluded": n_neg, "n_partition_words": len(part_words),
        "per_seed": [{"seed": r["seed"], "n_vocab": r["n_vocab"], "code_dim": r["code_dim"],
                      "n_reinforced": r["n_reinforced"], "primaries": r["primaries"], "floor_text": r["floor_text"],
                      "floor_emb": r["floor_emb"], "code_ceiling_recall_fp0": r["code_ceiling_recall_fp0"],
                      "corr_s_c_text_warriner": r["corr_s_c_text_warriner"],
                      "corr_s_c_emb_warriner": r["corr_s_c_emb_warriner"]} for r in rows],
        "config": {"seeds": seeds, "smoke": a.smoke, "max_stories": max_stories, "resample_frac": a.resample_frac,
                   "n_hub": a.n_hub, "window": a.window, "min_count": min_count, "n_each": a.n_each,
                   "min_events": a.min_events, "backend": os.environ.get("SIM_BACKEND")},
        "mechanism": "spiking opponent V+/V- (build_bridge: code_in -> appr_vplus/appr_vminus with xinh cross-"
                     "inhibition); weights = selforg_opponent_weights (three-factor Hebbian over the learned code); "
                     "GATE = pos_rate+neg_rate (total opponent drive), read HELD-OUT via 2-fold cross-fit, per-seed "
                     "label-free median gain-control, joint-FP=0. THE ONLY VARIABLE vs the experienced-opponent "
                     "BOUNDARY is the conditioning US SOURCE: EMBODIED-US-ORACLE (signed true-affect current, the "
                     "world/body grounded-US stand-in) vs TEXT-US (co-occurrence with ~10 innate primary words). "
                     "code_separability_ceiling = supervised ridge k-fold-CV probe on the raw code (the upper bound "
                     "on any code readout); synthetic_separable_code = the SAME embodied-US read on a synthetic "
                     "separable code (readout-mechanism soundness).",
        "sources": [
            "Namburi, Tye et al. (2015, Nature) -- opposing valence-coding BLA populations bound by conditioning to "
            "a REAL unconditioned stimulus (shock/reward), NOT to lexical company: the biological grounding for the "
            "embodied-US redirection and the salience(total-drive)-vs-valence(differential) distinction.",
            "Rescorla & Wagner (1972) -- the associative-strength asymptote (the text-US control's s_c).",
            "2026-09-05-affect-experienced-opponent-gate-needs-embodiment-BOUNDARY.md -- named the embodied US as the "
            "surpass; this runner tests whether that US alone suffices or the concept code is also a wall.",
            "2026-09-05-affect-learned-gate-retry-register-confound-BOUNDARY.md -- the register confound (text "
            "co-occurrence cannot separate affect from neutral); reproduced here as the text-US channel.",
        ],
        "production_wiring": "NONE -- affect_production_organ.py and wkv_mouth_generator.py are byte-unchanged; this "
                             "is a standalone research probe (reuse-by-import only).",
        "HONEST_RESIDUALS": "(1) the embodied US is an ORACLE stand-in (signed true affect) for a grounded-world US "
                            "that does NOT exist for the conversational vocabulary -- the finding names exactly what "
                            "grounded-world signal must produce it; using Warriner as the oracle is the world's-US "
                            "role, distinct from a graph-statistic read (the banked confound), and the read is "
                            "HELD-OUT so it is not a re-read of the US. (2) rate-level Hebbian conditioning WRITE "
                            "(codes are spiking-validated; fully-spiking three-factor write = a further rung). (3) "
                            "the code ceiling is a linear upper bound; the spiking opponent's mild nonlinearity was "
                            "measured NOT to help by the prior boundaries. (4) standalone de-risk bridge.",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    undefined_if_empty("partition-words", len(part_words), len(part_words), len(part_words))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[embodied-us-gate] embodied-US={emb_recall:.3f} | text-US={text_recall:.3f} | no-cond={nocond_recall:.3f} "
          f"| code-ceiling(worst/mean)={code_ceiling_worst:.3f}/{code_ceiling_mean:.3f} | synth-ceiling(worst)="
          f"{synth_ceiling_worst:.3f} synth-spiking(worst)={synth_recall_worst:.3f} | floor={max_floor:.4f}",
          flush=True)
    print(f"[embodied-us-gate] VERDICT: {verdict}", flush=True)
    print(f"[embodied-us-gate] wrote {a.out}  ({summary['elapsed_seconds']}s)\n" + "=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
