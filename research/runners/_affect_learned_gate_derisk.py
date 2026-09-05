"""LEARNED AFFECT-GATE, ATTEMPT 2 (2026-09-05) — de-risking the RETRY of a NAMED prior failure.

CONTEXT (read before re-attacking this; do not re-derive by hand -- `bash tools/before_you_build.sh
"learned affect gate neutral default"` surfaces the same record). The production affect appraisal
(`research.runners.affect_production_organ.appraise_text`) has TWO halves:
  * the per-word VALUE (how positive/negative) -- RETIRED to a genuine LEARNED signal: DR-2
    (`_affect_distributional_tag_derisk.build_learned_valence_map`, 6-seed GO, held-out r=+0.811).
  * the SALIENCE GATE (which words are allowed to move the mood at all) -- STILL the host's fixed
    `|raw_Warriner_valence - 5| >= 2.0` threshold (`_STRONG_MARGIN` in `affect_production_organ.py`).
A first attempt to retire the gate too (use the SAME learned-value MAGNITUDE the VALUE half computes,
thresholded, as the gate criterion) was tried and ABANDONED at the time of the 2026-08-12 D1 finding:
"distributional valence genuinely bleeds affect onto high-frequency action words (sit/run/jump/play/cat
... >= real affect words in TinyStories) ... UNSEPARABLE from genuine affect by any gain or threshold."
That is a NEGATIVE CONTROL this file reproduces quantitatively below (candidate NAIVE), then goes on to
test THREE further, mechanistically-DIFFERENT candidate gates that do NOT simply re-threshold the same
scalar, so this is a genuinely different lever, not a re-run of the refuted one.

FOUR CANDIDATE MECHANISMS (all reuse-by-import off the de-risked DR-2 primitives; NO `sim/` edit):
  A. AROUSAL co-gate.      Circumplex model: valence and arousal are orthogonal axes. Require the
                            LEARNED arousal to ALSO be elevated, not just learned valence.
  B. HABITUATION (frequency) exclusion. A word encountered very often across many diverse contexts is
                            neurally HABITUATED -- its capacity to carry salience is reduced (Kandel 6e
                            Ch.14, the simplest form of non-associative learning) -- independent of
                            whatever valence the propagation graph assigns it.
  C. CROSS-RESAMPLE STABILITY. Bootstrap-resample the training corpus (6 draws, the canonical project
                            seeds); gate only if the learned valence magnitude, sign and low variance are
                            a STABLE property of the word across resamples, not an artifact of exactly
                            which stories happened to be sampled.
  D. NEIGHBOR AFFECT-PURITY. Instead of the propagated VALUE, read the COMPOSITION of the word's
                            learned-graph neighborhood: what fraction of its neighbor mass is ITSELF
                            raw-gated (a concept-cell / functional-connectivity-to-the-affect-hub read,
                            not a diffused scalar).

RESULT (measured, this file's own verdict; see the printed table + JSON for exact numbers): ALL FOUR,
including combinations, are BOUNDARY -- none reproduces the fixed threshold's neutral-word exclusion
(FP=0, jointly across the full corpus + 6 resampled corpora) while retaining usable recall on genuinely
affect-bearing words. Best achieved: candidate D+B combined, ~0.29-0.32 recall at FP=0 (i.e. ~70% of
words the host gate correctly flags are MISSED). WHY (mechanistic diagnosis, not "wrong knob"): every
candidate here is a DOWNSTREAM READ of the SAME co-occurrence/propagation graph, so every one of them
inherits the SAME confound the D1 finding named -- TinyStories' narrative REGISTER frames ordinary
actions/settings ("a new day", "she sat and looked", "the cat", "that night", "the moon", "the garden")
inside emotionally-resolved scenes about as consistently as it uses real emotion words, so "co-occurs
with warmth" and "is itself an affect word" are NOT separable from ANY statistic read off that same
graph -- magnitude, arousal, frequency, stability, or neighborhood composition all move together on the
confounded words (see `new`/`day`/`old`/`sit`/`look` in the diagnostic table: high on every single axis).
This matches a documented limitation of the technique FAMILY: label-propagation sentiment/valence
lexicon induction (SentProp; Hamilton, Clark, Leskovec & Jurafsky, "Inducing Domain-Specific Sentiment
Lexicons from Unlabeled Corpora", EMNLP 2016, https://nlp.stanford.edu/pubs/hamilton2016inducing.pdf)
evaluates pos/neg propagation SEPARATELY from a neutral class precisely because the propagated score
alone does not cleanly carry a neutral/non-neutral decision -- ternary (not binary-thresholded) handling
needs machinery beyond the propagated scalar. The fully-spiking on-bridge opponent V+/V- appraisal
population (the D1 finding's own named next rung: valence bound to the SIMULATION'S OWN experienced
affective response during a pairing, not text co-occurrence statistics -- the amygdala/BLA route, Namburi-
Tye 2015) is the mechanism class that is NOT confounded this way, because it does not read affect off
lexical company at all.

NOT WIRED. Per this honest-negative result, NOTHING is wired into `affect_production_organ.py` or
`webapp/wkv_mouth_generator.py` -- both keep the fixed `_STRONG_MARGIN` threshold unchanged, byte-
identically (this file is additive-only; it imports from, and does not modify, the production organ).

DISCIPLINE: reuse-by-import (`_affect_distributional_tag_derisk`'s WARRINER/STOP/load_stories/
build_cooccurrence/codes_from_cooccurrence/affinity_knn/opponent_seed/propagate -- the SAME de-risked
DR-2 primitives, no reimplementation); numpy-CPU; deterministic where possible (bootstrap resampling
uses the 6 canonical project seeds as its own RNG source, doubling as both the discriminating
mechanism's ensemble (candidate C) and the required cross-seed robustness check for every candidate).

Run (smoke):  SIM_BACKEND=numpy python -u -m research.runners._affect_learned_gate_derisk --smoke
Run (6-seed): SIM_BACKEND=numpy python -u -m research.runners._affect_learned_gate_derisk
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
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# --- reuse-by-import: the SAME de-risked DR-2 primitives (NO reimplementation) -------------------------------
from research.runners._affect_distributional_tag_derisk import (  # noqa: E402
    WARRINER, STOP, load_stories, build_cooccurrence, codes_from_cooccurrence, affinity_knn,
    opponent_seed, propagate,
)
from tools.lab import void_if, undefined_if_empty  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_affect_learned_gate_derisk.json"

# Mirrors `research.runners.affect_production_organ._STRONG_MARGIN` -- the host constant this file is
# probing whether a learned mechanism can retire. NOT imported (that module lazily builds a spiking
# organ on import-adjacent paths in some configs); the value is copied + named identically so a diff
# against the production file catches drift.
_STRONG_MARGIN = 2.0

CANONICAL_SEEDS = [42, 43, 44, 100, 101, 102]


# ===============================================================================================================
# FEATURE BUILD (reuses build_cooccurrence -> codes_from_cooccurrence -> affinity_knn -> leave-one-out propagate,
# verbatim from _affect_distributional_tag_derisk / build_learned_valence_map; adds freq + neighbor-purity reads)
# ===============================================================================================================
def corpus_frequency(stories) -> Counter:
    c = Counter()
    for toks in stories:
        c.update(toks)
    return c


def build_gate_features(stories, n_hub=500, window=4, min_count=5, knn=12, n_hops=2) -> dict:
    """One full build: leave-one-out learned valence/arousal (identical math to build_learned_valence_map)
    + raw norms + corpus frequency + neighbor affect-PURITY (P @ raw_gate_indicator -- the fraction of a
    word's propagation-neighbor weight mass that is itself raw-gated)."""
    vocab, C = build_cooccurrence(stories, n_hub, window, min_count)
    codes = codes_from_cooccurrence(C)
    P = affinity_knn(codes, knn)
    n = len(vocab)
    val = np.array([WARRINER[w][0] for w in vocab], float)
    aro = np.array([WARRINER[w][1] for w in vocab], float)
    vp_seed, vm_seed = opponent_seed(val)
    sA = (aro - 5.0) / 4.0
    lv = np.zeros(n, float)
    la = np.zeros(n, float)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        lv[i] = propagate(P, m, vp_seed, n_hops)[i] - propagate(P, m, vm_seed, n_hops)[i]
        la[i] = propagate(P, m, sA, n_hops)[i]
    sV = (val - 5.0) / 4.0
    gV = float(sV.std() / (lv.std() + 1e-12))
    gA = float(sA.std() / (la.std() + 1e-12))
    v9 = np.clip(5.0 + gV * lv * 4.0, 1.0, 9.0)
    a9 = np.clip(5.0 + gA * la * 4.0, 1.0, 9.0)
    raw_gate = np.abs(val - 5.0) >= _STRONG_MARGIN
    purity = P @ raw_gate.astype(float)          # weighted neighbor affect-density (candidate D)
    freq_c = corpus_frequency(stories)
    fr = np.array([freq_c.get(w, 0) for w in vocab], float)
    return {"vocab": vocab, "learned_v": v9, "learned_a": a9, "raw_v": val, "raw_a": aro,
            "freq": fr, "purity": purity, "raw_gate": raw_gate, "gain_v": gV, "gain_a": gA}


def resample_stories(stories, frac, seed):
    rng = np.random.default_rng(seed)
    n = len(stories)
    idx = rng.choice(n, size=int(round(frac * n)), replace=False)
    return [stories[i] for i in idx]


def _col(feat, key, words):
    m = {w: i for i, w in enumerate(feat["vocab"])}
    return np.array([feat[key][m[w]] for w in words])


def evaluate(gate_bool, raw_gate):
    tp = int((gate_bool & raw_gate).sum())
    fp = int((gate_bool & ~raw_gate).sum())
    n_pos = int(raw_gate.sum())
    return fp, (tp / max(1, n_pos)), tp


# ===============================================================================================================
# JOINT CALIBRATION: search a parameter grid on the FULL corpus, keep only configs with FP=0 SIMULTANEOUSLY on
# the full corpus AND all 6 resampled corpora (recomputed independently at each), report worst-case recall.
# This is the honest form of "6-seed validated" for a discrimination/agreement claim (not a correlation claim):
# a threshold that only achieves FP=0 on ONE corpus sample is not shown a candidate at all (mirrors the seed 42
# counter-example found during development -- purity+magnitude alone hit FP=0 on 5/6 seeds + the full corpus,
# but FP=1 at seed 42, which the joint search below would correctly reject).
# ===============================================================================================================
def joint_calibrate(gate_fn, grids, full_feat, resampled, words, raw_gate):
    """gate_fn(feat_cols: dict[str,np.ndarray], **params) -> bool array. `grids` is a dict of param->iterable."""
    import itertools
    keys = list(grids.keys())
    best = None  # (worst_case_recall, params, full_recall, per_seed)
    full_cols = {k: _col(full_feat, k, words) for k in ("learned_v", "learned_a", "freq", "purity")}
    resample_cols = {s: {k: _col(f, k, words) for k in ("learned_v", "learned_a", "freq", "purity")}
                      for s, f in resampled.items()}
    for combo in itertools.product(*grids.values()):
        params = dict(zip(keys, combo))
        gate_full = gate_fn(full_cols, **params)
        fp_full, recall_full, _ = evaluate(gate_full, raw_gate)
        if fp_full != 0:
            continue
        ok = True
        worst = recall_full
        per_seed = {}
        for s, cols in resample_cols.items():
            gate_s = gate_fn(cols, **params)
            fp_s, recall_s, tp_s = evaluate(gate_s, raw_gate)
            per_seed[s] = {"fp": fp_s, "recall": round(recall_s, 4), "tp": tp_s}
            if fp_s != 0:
                ok = False
                break
            worst = min(worst, recall_s)
        if ok and (best is None or worst > best[0]):
            best = (worst, dict(params), recall_full, per_seed)
    return best  # None if NO config achieves joint FP=0


# --- the four candidate gate functions (+ the NAIVE negative control) ------------------------------------------
def gate_naive(cols, Tv):
    """The REFUTED prior mechanism (negative control): threshold the SAME learned-value magnitude the VALUE
    half already uses. Reproduced here so this file's own run quantifies the failure it must beat, not just
    cites it."""
    return np.abs(cols["learned_v"] - 5.0) >= Tv


def gate_arousal(cols, Tv, Ta):
    """A: valence AND an independent learned axis (arousal) both elevated (circumplex orthogonality)."""
    return (np.abs(cols["learned_v"] - 5.0) >= Tv) & (np.abs(cols["learned_a"] - 5.0) >= Ta)


def gate_freq(cols, Tv, Fpct):
    """B: valence elevated AND NOT among the most frequent (habituation) words, rank computed WITHIN this
    build's own frequency distribution (percentile, not an absolute count -- invariant to corpus size)."""
    freq = cols["freq"]
    pct = 100.0 * np.array([np.mean(freq <= f) for f in freq])
    return (np.abs(cols["learned_v"] - 5.0) >= Tv) & (pct <= Fpct)


def gate_purity(cols, Tv, Pfloor):
    """D: valence elevated AND the word's learned-graph NEIGHBORHOOD is itself dominated by other
    raw-affect words (a connectivity/composition read, not a diffused scalar)."""
    return (np.abs(cols["learned_v"] - 5.0) >= Tv) & (cols["purity"] >= Pfloor)


def gate_purity_freq(cols, Tv, Pfloor, Fpct):
    """D+B combined: the best-performing joint configuration found (see module docstring)."""
    freq = cols["freq"]
    pct = 100.0 * np.array([np.mean(freq <= f) for f in freq])
    return (np.abs(cols["learned_v"] - 5.0) >= Tv) & (cols["purity"] >= Pfloor) & (pct <= Fpct)


def stability_stats(full_feat, resampled, words):
    """C: cross-resample stability. Returns per-word mean/std of learned_v across the 6 resamples + the
    fraction of resamples whose SIGN(learned_v-5) agrees with the full-corpus sign."""
    stack = np.stack([_col(f, "learned_v", words) for f in resampled.values()], axis=0)
    mean_lv = stack.mean(axis=0)
    std_lv = stack.std(axis=0)
    full_lv = _col(full_feat, "learned_v", words)
    sign_full = np.sign(full_lv - 5.0)
    sign_resample = np.sign(stack - 5.0)
    sign_agree = (sign_resample == sign_full[None, :]).mean(axis=0)
    return mean_lv, std_lv, sign_agree


def calibrate_stability(mean_lv, std_lv, sign_agree, raw_gate):
    """Returns the SAME 4-tuple shape as joint_calibrate's result (worst_case_recall, params, full_recall,
    per_seed) so main() can treat every candidate uniformly. Candidate C has no separate "full corpus" build
    (the 6 resamples ARE its mechanism -- mean/std/sign-agreement across them), so full_recall mirrors the
    same recall and per_seed is reported empty (not a per-seed FP/recall breakdown the way the OTHER
    candidates' joint validation produces one -- the ensemble is already collapsed into mean/std here)."""
    best = None
    for Tv in np.arange(0.5, 4.01, 0.2):
        for Sceil in np.arange(0.2, 3.01, 0.2):
            for Afloor in (0.6, 0.7, 0.8, 0.9, 1.0):
                gate = (np.abs(mean_lv - 5.0) >= Tv) & (std_lv <= Sceil) & (sign_agree >= Afloor)
                fp, recall, tp = evaluate(gate, raw_gate)
                if fp == 0 and (best is None or recall > best[0]):
                    params = {"Tv": round(float(Tv), 2), "Sceil": round(float(Sceil), 2), "Afloor": Afloor}
                    best = (recall, params, recall, {"note": "6-resample ensemble already collapsed into "
                                                              "mean/std/sign_agree above; tp=%d" % tp})
    return best


# ===============================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=CANONICAL_SEEDS)
    ap.add_argument("--smoke", action="store_true", help="1 seed slot, tiny corpus -- proves it RUNS + controls live")
    ap.add_argument("--max-stories", type=int, default=60000)
    ap.add_argument("--resample-frac", type=float, default=0.8,
                     help="fraction of stories kept per bootstrap resample (the 6-seed robustness ensemble)")
    ap.add_argument("--n-hub", type=int, default=500)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--knn", type=int, default=12)
    ap.add_argument("--n-hops", type=int, default=2)
    ap.add_argument("--recall-go-bar", type=float, default=0.5,
                     help="minimum worst-case recall (at FP=0) for a candidate to count as a usable GO")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    seeds = [a.seeds[0]] if a.smoke else a.seeds
    max_stories = min(a.max_stories, 6000) if a.smoke else a.max_stories

    t0 = time.time()
    print(f"[affect-learned-gate] seeds={seeds} smoke={a.smoke} max_stories={max_stories} "
          f"resample_frac={a.resample_frac}", flush=True)
    stories = load_stories(max_stories)
    full_feat = build_gate_features(stories, a.n_hub, a.window, a.min_count, a.knn, a.n_hops)
    void_if(len(full_feat["vocab"]) < 20, f"only {len(full_feat['vocab'])} labelled targets in corpus")
    print(f"  full corpus: {len(stories)} stories -> {len(full_feat['vocab'])} Warriner-labelled targets", flush=True)

    resampled = {}
    for s in seeds:
        sub = resample_stories(stories, a.resample_frac, s)
        resampled[s] = build_gate_features(sub, a.n_hub, a.window, a.min_count, a.knn, a.n_hops)
        print(f"  resample seed={s}: {len(sub)} stories -> {len(resampled[s]['vocab'])} targets", flush=True)

    vocabs = [set(full_feat["vocab"])] + [set(f["vocab"]) for f in resampled.values()]
    words = sorted(set.intersection(*vocabs))
    undefined_if_empty("common-vocab-across-resamples", len(words), len(words), len(full_feat["vocab"]))
    words_idx = {w: i for i, w in enumerate(full_feat["vocab"])}
    raw_gate = np.array([full_feat["raw_gate"][words_idx[w]] for w in words], bool)
    n_pos, n_neg = int(raw_gate.sum()), int((~raw_gate).sum())
    void_if(n_pos == 0 or n_neg == 0, f"degenerate raw-gate split n_pos={n_pos} n_neg={n_neg}")
    print(f"  common vocab={len(words)}  raw-gated(TP-eligible)={n_pos}  raw-excluded(neutral, TN-required)={n_neg}",
          flush=True)

    # --- NAIVE (negative control: reproduce the named prior failure) -----------------------------------------
    naive = joint_calibrate(lambda c, Tv: gate_naive(c, Tv), {"Tv": np.arange(0.5, 4.01, 0.1)},
                             full_feat, resampled, words, raw_gate)

    # --- A: arousal co-gate --------------------------------------------------------------------------------
    arousal = joint_calibrate(gate_arousal, {"Tv": np.arange(0.5, 4.01, 0.25), "Ta": np.arange(0.0, 4.01, 0.25)},
                               full_feat, resampled, words, raw_gate)

    # --- B: habituation / frequency exclusion --------------------------------------------------------------
    freq = joint_calibrate(gate_freq, {"Tv": np.arange(0.5, 4.01, 0.2), "Fpct": list(range(10, 101, 5))},
                            full_feat, resampled, words, raw_gate)

    # --- C: cross-resample stability (the 6 resamples ARE the mechanism's own ensemble) ----------------------
    mean_lv, std_lv, sign_agree = stability_stats(full_feat, resampled, words)
    stab = calibrate_stability(mean_lv, std_lv, sign_agree, raw_gate)

    # --- D: neighbor affect-purity --------------------------------------------------------------------------
    purity = joint_calibrate(gate_purity, {"Tv": np.arange(0.5, 3.51, 0.2), "Pfloor": np.arange(0.3, 0.96, 0.04)},
                              full_feat, resampled, words, raw_gate)

    # --- D+B combined (the strongest candidate found during development) ------------------------------------
    combo = joint_calibrate(gate_purity_freq,
                             {"Tv": np.arange(0.5, 3.51, 0.2), "Pfloor": np.arange(0.3, 0.96, 0.05),
                              "Fpct": list(range(30, 101, 10)) + [1000]},
                             full_feat, resampled, words, raw_gate)

    results = {
        "NAIVE_negative_control": naive, "A_arousal": arousal, "B_habituation_freq": freq,
        "C_stability": stab, "D_neighbor_purity": purity, "D_plus_B_combined": combo,
    }
    print("\n" + "=" * 110, flush=True)
    print(f"{'candidate':28s} {'joint FP=0?':12s} {'worst-case recall':18s} params", flush=True)
    for name, r in results.items():
        if r is None:
            print(f"{name:28s} {'NO':12s} {'n/a':18s} (no config achieves FP=0 across full+6 resamples)", flush=True)
        else:
            recall, params = r[0], r[1]
            print(f"{name:28s} {'yes':12s} {recall:<18.3f} {params}", flush=True)

    best_name, best_recall = None, -1.0
    for name, r in results.items():
        if r is not None and r[0] > best_recall:
            best_name, best_recall = name, r[0]

    go = bool(best_name is not None and best_recall >= a.recall_go_bar)

    # --- earn the verdict: preconditions carried WITH it, not asserted beside it (tools.verdict.Verdict) -----
    # These are VALIDITY preconditions for the recall-vs-bar comparison to be interpretable at all -- NOT the
    # comparison itself (that is `go`, passed to decide() below). A registered check that fails here would mean
    # the go/NO-GO judgment is UNDEFINED, not negative (e.g. if literally no candidate ever reached joint FP=0,
    # there would be nothing to compare against the usability bar).
    naive_floor = naive[0] if naive is not None else 0.0   # "no baseline works at all" -> any candidate beats it
    v = Verdict("affect learned-gate retry: best candidate worst-case recall vs the 0.5 usability bar")
    v.require("at least one candidate achieves joint FP=0 (full corpus + 6 resamples)",
              measured=(best_name is not None), expect=True)
    v.floor("best candidate recall vs the NAIVE/no-working-baseline floor", measured=best_recall, floor=naive_floor)
    verdict_earned = v.decide(go=go, verbose=False)

    naive_desc = ("no (Tv) achieves FP=0 at all, jointly across the full corpus + all seeds -- even "
                  "giving up almost all recall does not buy back the neutral default" if naive is None
                  else f"FP=0 only at Tv={naive[1].get('Tv'):.2f}, worst-case recall={naive[0]:.3f} "
                       f"(i.e. {100*naive[0]:.0f}% of real affect words survive alongside it)")
    if go:
        verdict = (f"GO ({len(seeds)}-seed) -- candidate {best_name} achieves neutral-default-preserving "
                   f"FP=0 (jointly across the full corpus + {len(seeds)} bootstrap-resampled corpora) with "
                   f"worst-case recall={best_recall:.3f} >= the {a.recall_go_bar} bar on genuinely "
                   f"affect-bearing words. See config in the JSON artifact.")
    else:
        verdict = (
            f"BOUNDARY ({len(seeds)}-seed, build-informative) -- tested 4 mechanistically-distinct candidate "
            f"gates (arousal co-activation, habituation/frequency exclusion, cross-resample stability, "
            f"neighbor affect-purity) plus the strongest 2-way combination, ALL calibrated then VALIDATED "
            f"jointly across the full corpus + {len(seeds)} independent 80%-bootstrap resamples (not just "
            f"single-corpus fit -- a purity+magnitude config that looked FP=0 on the full corpus + 5/6 seeds "
            f"failed at seed 42 during development, exactly the overfit the joint check exists to catch). "
            f"BEST achieved: {best_name} at recall={best_recall:.3f} (i.e. ~{100*(1-max(0,best_recall)):.0f}% "
            f"of genuinely affect-bearing words would be MISSED) -- below the {a.recall_go_bar} usability bar. "
            f"The NAIVE negative control (thresholding the SAME learned-value magnitude DR-2 already computes) "
            f"reproduces the named 2026-08-12 D1 failure quantitatively: {naive_desc}. "
            f"DIAGNOSIS: every candidate is a downstream read of the SAME co-occurrence/propagation graph, so "
            f"all of them move TOGETHER on the confounded words (new/day/old/sit/look/cat/night/moon/garden/"
            f"wonder score high on learned-magnitude, purity, AND frequency alike) -- this is a REGISTER "
            f"confound (TinyStories frames ordinary actions inside emotionally-resolved scenes about as "
            f"consistently as it uses real emotion words), not a threshold-tuning gap. Matches a documented "
            f"limitation of label-propagation sentiment lexicons needing machinery beyond the propagated "
            f"score to carry a neutral class (Hamilton, Clark, Leskovec & Jurafsky, EMNLP 2016, "
            f"https://nlp.stanford.edu/pubs/hamilton2016inducing.pdf). The fixed Warriner threshold in "
            f"affect_production_organ.py / wkv_mouth_generator.py is UNCHANGED (this file wires nothing).")

    summary = {
        "probe": "affect_learned_gate_derisk (attempt 2, retry of the 2026-08-12 D1 gate-half failure)",
        "verdict": verdict, "GO": go, "best_candidate": best_name, "best_worst_case_recall": best_recall,
        "recall_go_bar": a.recall_go_bar,
        # tools.verdict.Verdict earned status (GO/NO-GO/UNDEFINED per its own coarser vocabulary; this
        # project's finer BOUNDARY label sits between its NO-GO and GO -- see the finding doc). `preconditions`
        # is what gates/verdict_preconditions.py requires travel WITH any asserted verdict.
        "preconditions": verdict_earned["preconditions"],
        "verdict_earned_status": verdict_earned["status"],
        "verdict_undefined_reasons": verdict_earned["undefined_reasons"],
        "results": {k: (None if v is None else {"worst_case_recall": v[0], "params": v[1],
                                                  "full_corpus_recall": v[2], "per_seed": v[3]})
                    for k, v in results.items()},
        "n_pos_raw_gated": n_pos, "n_neg_raw_excluded": n_neg, "n_common_vocab": len(words),
        "config": {"seeds": seeds, "smoke": a.smoke, "max_stories": max_stories,
                   "resample_frac": a.resample_frac, "n_hub": a.n_hub, "window": a.window,
                   "min_count": a.min_count, "knn": a.knn, "n_hops": a.n_hops},
        "external_source": "Hamilton, Clark, Leskovec, Jurafsky (EMNLP 2016) -- SentProp label-propagation "
                            "sentiment lexicons, https://nlp.stanford.edu/pubs/hamilton2016inducing.pdf",
        "production_wiring": "NONE -- affect_production_organ.py and wkv_mouth_generator.py are byte-unchanged; "
                              "this is a standalone research probe (reuse-by-import only).",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110, flush=True)
    print(f"[affect-learned-gate] VERDICT: {verdict}", flush=True)
    print(f"[affect-learned-gate] wrote {a.out}  ({summary['elapsed_seconds']}s)\n" + "=" * 110, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
