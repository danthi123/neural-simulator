"""DR-2 (Phase-0, P0.1) — DISTRIBUTIONAL AFFECT TAG: concepts learn how they "should FEEL" from the affective
company they keep. The world-model is affectively COLOURED, not just factual (the owner's world-model point).

THE OWNER REFRAME: the sim already learns relational structure from a corpus stream unsupervised on spikes
(`corr(M,C)+0.686`), but that structure is purely FACTUAL. Affect is *distributionally recoverable* — a concept's
valence emerges from the affective company it keeps (Bestgen-Vincze 2012 k-NN valence inference, r~0.71). So we
attach a learned VALENCE to the sim's ALREADY-learned concept codes and INHERIT/PROPAGATE it over the LEARNED
concept-association graph, seeding only a small core set from affective norms (Warriner VAD).

MECHANISM (BUILDABLE-NOW, NO `sim/` edit):
  1. LEARNED concept-association graph. Build a co-occurrence cortex from a real corpus (TinyStories), the SAME
     rate-Hebbian co-occurrence the project validated as the matched rule (STDP measured-0 because co-occurrence
     has no pre->post order; `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`). Concept CODE =
     PPMI over frequent hubs (reuse `learned_graded_cortex_fair_test.ppmi_matrix`, the stream-cortex read); the
     ASSOCIATION GRAPH = cosine(code_i, code_j), k-NN-sparsified, row-normalized = the learned affinity P.
  2. OPPONENT V+/V- affect population (Namburi-Tye 2015: BLA opposing valence-coding populations, opposite-sign).
     Seed VAD for a small CORE set from Warriner norms: signed s=(val-5)/4 in [-1,1]; V+seed=max(s,0),
     V-seed=max(-s,0) (rectified opponent). Neutral -> both ~0.
  3. INHERIT valence to every OTHER (held-out) concept by SPREADING ACTIVATION over the learned graph — the exact
     EMERGE-30 member->neighbours->shared-tag read, realized here as seed-clamped label propagation (Zhu-Ghahramani
     harmonic function) run separately on the V+ and V- channels. Held-out net valence = V+_pred - V-_pred. A
     concept's valence thus EMERGES from the affective company it keeps (Redondo-Tonegawa 2014: valence is a
     separable, re-writable tag on a fixed identity code -- the license to keep learned codes fixed + bolt on a
     plastic affect tag).

GO GATE (6-seed): predicted valence correlates Pearson r >= 0.55 with Warriner valence on HELD-OUT concepts (never
in the seed set). Arousal is reported as a secondary channel (a known weaker ceiling).

ANTI-CHEATS (all wired + INVOKED):
  (1) PERMUTED-GRAPH  -- scramble which co-occurrence code belongs to which word (EMERGE-30 control verbatim) ->
      neighbourhoods become random -> valence inheritance collapses to chance (proves the LEARNED structure carries
      the affect, not a lookup).
  (2) SEED-ONLY baseline -- predict every held-out with the constant mean seed valence (no graph) -> r ~ 0; the
      real graph must BEAT it.
  (3) SHUFFLED-SEED-LABELS -- permute the valence values across seed words -> meaningless labels spread -> collapse.
  (4) OPPONENT-SIGN -- across held-out, V+_pred and V-_pred are genuinely OPPOSED (corr < 0) AND truly-negative
      concepts read net-negative below truly-positive (aversive drives V- UP and suppresses V+, not merely "low V+").

DISCIPLINE: reuse-by-import, NO `sim/` edit (numpy-CPU cheap-first read of the mechanism; the on-bridge spiking
opponent-population confirm is the GPU follow-on). The Warriner values are an embedded Warriner-APPROXIMATE core
lexicon (the real 13,915-word CSV is not present locally); swap it via --warriner-csv (word,valence[,arousal]).
No cached artifact maps the 320 stream codes to Warriner-labelled words, so the learned co-occurrence graph is
BUILT IN-RUNNER from TinyStories over the Warriner-labelled vocab (stated).

Run (smoke):  SIM_BACKEND=numpy python -u -m research.runners._affect_distributional_tag_derisk --smoke
Run (6-seed): SIM_BACKEND=numpy python -u -m research.runners._affect_distributional_tag_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# --- reuse-by-import: the stream-cortex PPMI read + the project stoplist -----------------------------------------
from research.runners.learned_graded_cortex_fair_test import ppmi_matrix  # noqa: E402  (the stream-cortex code read)
from research.runners.option_c_stageB_fair_test import STOPLIST  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_affect_distributional_tag.json"

# --- embedded Warriner-APPROXIMATE core VAD lexicon (valence, arousal) on the 1..9 Warriner scale ----------------
# Biased toward children's-story vocabulary (so it co-occurs in TinyStories). This is the SEED norm source AND the
# held-out ground-truth. Swap the real Warriner (2013) 13,915-word CSV via --warriner-csv for a production run.
WARRINER = {
    # --- strongly positive ---
    "happy": (8.5, 6.1), "joy": (8.2, 5.9), "love": (8.0, 6.4), "smile": (7.7, 4.9), "laugh": (7.9, 6.0),
    "fun": (8.4, 6.4), "play": (7.9, 5.6), "friend": (7.9, 4.9), "kind": (7.3, 4.3), "nice": (7.4, 4.4),
    "good": (7.5, 5.4), "sweet": (7.5, 5.0), "hug": (7.8, 4.6), "kiss": (7.6, 6.0), "gift": (7.6, 5.5),
    "present": (7.0, 5.3), "cake": (7.3, 4.9), "candy": (7.5, 5.3), "toy": (7.2, 5.1), "sun": (7.4, 5.0),
    "warm": (7.1, 4.6), "cozy": (7.0, 4.0), "home": (7.4, 4.5), "mom": (7.6, 5.0), "dad": (7.2, 4.9),
    "baby": (7.3, 5.6), "puppy": (7.9, 5.5), "kitten": (7.6, 5.2), "flower": (7.4, 4.0), "garden": (6.9, 3.9),
    "rainbow": (7.8, 4.7), "magic": (7.0, 5.5), "dream": (6.9, 4.8), "wonder": (6.9, 5.0), "brave": (7.2, 5.8),
    "proud": (7.6, 5.6), "win": (7.9, 6.4), "share": (7.0, 4.4), "help": (7.3, 4.8), "thank": (7.6, 4.4),
    "please": (6.9, 4.0), "hope": (7.4, 4.9), "peace": (7.7, 3.5), "gentle": (7.0, 3.6), "pretty": (7.3, 5.0),
    "safe": (7.2, 3.9), "smart": (7.2, 5.0), "sing": (7.0, 5.2), "dance": (7.3, 6.0), "cheer": (7.5, 6.2),
    "treat": (7.2, 5.0), "lucky": (7.5, 5.3), "glad": (7.6, 5.0), "sunny": (7.4, 4.6), "cuddle": (7.5, 4.0),
    # --- mild positive / neutral-positive ---
    "dog": (6.9, 5.2), "cat": (6.6, 4.4), "bird": (6.7, 4.4), "fish": (6.0, 4.0), "star": (6.9, 4.5),
    "moon": (6.6, 3.8), "tree": (6.3, 3.6), "book": (6.7, 4.0), "water": (6.5, 3.6), "cup": (5.4, 3.2),
    "jump": (6.6, 5.7), "find": (6.6, 4.6), "new": (6.4, 5.0), "day": (6.0, 4.3), "make": (6.0, 4.4),
    "big": (5.5, 4.5), "hat": (5.5, 3.0), "walk": (5.7, 3.6), "run": (5.7, 5.5), "look": (5.6, 4.0),
    "house": (6.4, 3.8), "road": (5.0, 3.6), "box": (5.2, 3.2), "door": (5.2, 3.5), "small": (5.2, 3.8),
    "clean": (6.6, 4.0), "food": (6.8, 4.6), "grow": (6.5, 4.4), "give": (6.7, 4.6), "wave": (6.0, 4.4),
    # --- neutral / mildly negative ---
    "rock": (5.0, 3.4), "wall": (4.8, 3.2), "clock": (5.0, 3.4), "old": (4.8, 4.0), "work": (5.3, 4.8),
    "sit": (5.0, 3.0), "night": (5.0, 4.4), "rain": (4.4, 4.4), "dragon": (4.8, 6.0), "shy": (4.4, 4.5),
    "wolf": (3.9, 5.6), "cold": (3.8, 4.4), "tired": (3.8, 3.6), "storm": (3.7, 6.0), "mess": (3.6, 4.6),
    "ghost": (3.6, 5.6), "snake": (3.6, 5.8), "dark": (3.6, 4.9), "spider": (3.4, 5.7), "witch": (3.3, 5.3),
    "monster": (3.3, 6.0), "dirty": (3.2, 4.4), "sorry": (3.5, 4.2), "fall": (3.4, 5.0), "trouble": (3.0, 5.4),
    "worried": (3.3, 5.2), "lonely": (2.3, 4.6), "bite": (3.0, 5.6), "nasty": (2.4, 5.4), "scary": (3.1, 6.2),
    # --- strongly negative ---
    "sad": (2.1, 4.6), "cry": (2.4, 5.4), "scared": (2.8, 6.2), "afraid": (3.0, 6.0), "fear": (2.8, 6.1),
    "angry": (2.4, 6.3), "mad": (2.9, 6.0), "mean": (2.7, 5.3), "bad": (2.5, 5.4), "hate": (2.1, 6.3),
    "hurt": (2.2, 5.8), "fight": (2.9, 6.4), "lost": (2.8, 5.0), "broken": (2.9, 5.0), "ugly": (2.5, 5.0),
    "sick": (2.3, 5.0), "lose": (2.5, 5.2), "die": (1.7, 6.0), "kill": (1.9, 6.6), "hungry": (3.2, 5.4),
    "pain": (2.0, 6.0), "cruel": (2.0, 5.9), "evil": (2.2, 6.3), "dead": (1.8, 5.7), "war": (2.1, 6.4),
    # --- adult / strong-emotion conversational vocabulary (2026-08-19 depth fix) ---
    # WHY: the seed above was curated for the TinyStories child corpus, so it MISSED the common
    # adult emotion words a person actually uses when they feel strongly ("thrilled", "devastated").
    # An instrument map found the appraisal read valence 0.0 for "I am furious, devastated" and only
    # fired on moderate words like "sad"/"happy" -> both affect faculties (#13 coloring + #84 tone,
    # same mood) went DORMANT exactly when emotion was strongest. These are Warriner-approximate seed
    # norms (v9,a9), sign-correct + |v-5|>=_STRONG_MARGIN, calibrated to the entries above; all are
    # unambiguously affective so they cannot color a neutral factual query.
    # strongly positive:
    "thrilled": (8.1, 7.0), "delighted": (8.2, 5.6), "excited": (7.9, 6.9), "ecstatic": (8.0, 7.2),
    "overjoyed": (8.2, 6.4), "elated": (7.8, 6.1), "wonderful": (8.2, 5.6), "joyful": (8.1, 5.9),
    "cheerful": (7.9, 5.5), "grateful": (7.9, 4.8), "pleased": (7.6, 5.0), "amazing": (8.0, 6.0),
    "fantastic": (8.1, 6.1), "awesome": (8.0, 6.1), "relieved": (7.2, 4.4), "hopeful": (7.5, 5.0),
    # strongly negative:
    "devastated": (1.7, 5.6), "furious": (2.0, 7.0), "miserable": (1.9, 4.6), "heartbroken": (1.7, 4.8),
    "despair": (1.9, 4.8), "terrified": (2.0, 7.1), "upset": (2.4, 5.0), "anxious": (2.9, 6.0),
    "depressed": (1.8, 4.0), "frustrated": (2.6, 5.8), "disappointed": (2.6, 4.4), "hopeless": (1.9, 4.5),
    "grief": (1.9, 4.8), "awful": (2.0, 5.4), "terrible": (1.9, 5.4), "horrible": (1.9, 5.7),
    "nervous": (3.2, 5.8), "stressed": (2.8, 5.9), "annoyed": (3.0, 5.4), "guilty": (2.6, 5.0),
    "ashamed": (2.4, 4.9), "jealous": (2.7, 5.6), "disgusted": (2.3, 5.4), "anger": (2.3, 6.3),
}

STOP = set(STOPLIST) | {"was", "with", "they", "you", "we", "this", "there", "then", "them", "were", "will",
                        "would", "could", "your", "very", "into", "when", "what", "who", "how", "why", "which"}


# =============================================================================================================
# 1. LEARNED concept-association graph from the corpus (rate-Hebbian co-occurrence -> PPMI code -> cosine kNN)
# =============================================================================================================
def load_stories(max_stories):
    path = os.path.join(_REPO, "data", "corpus", "tinystories.txt")
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    stories = [re.findall(r"[a-z]+", s) for s in text.split("<|endoftext|>")]
    if max_stories and max_stories > 0:
        stories = stories[:max_stories]
    return stories


def build_cooccurrence(stories, n_hub, window, min_count):
    """Learned cortex: co-occurrence of Warriner-vocab TARGETS with frequent context HUBS (the validated
    rate-Hebbian co-occurrence, accumulated online over a WM window). Returns (vocab, C[targets, hubs])."""
    gfreq = Counter()
    for toks in stories:
        gfreq.update(toks)
    # targets = Warriner words that actually appear enough in the corpus (the learnable intersection)
    vocab = [w for w in WARRINER if gfreq.get(w, 0) >= min_count]
    target_set = set(vocab)
    hubs = [w for w, _ in gfreq.most_common() if w not in STOP and w not in target_set][:n_hub]
    hub_idx = {w: i for i, w in enumerate(hubs)}
    tgt_row = {w: i for i, w in enumerate(vocab)}
    C = np.zeros((len(vocab), len(hubs)), dtype=np.float64)
    keep = target_set | set(hubs)
    for toks in stories:
        kept = [t for t in toks if t in keep]
        for c in range(len(kept)):
            w = kept[c]
            if w not in target_set:
                continue
            lo, hi = max(0, c - window), min(len(kept), c + window + 1)
            for u in kept[lo:hi]:
                if u != w and u in hub_idx:
                    C[tgt_row[w], hub_idx[u]] += 1.0
    return vocab, C


def codes_from_cooccurrence(C):
    """Concept CODE = PPMI over hubs (the stream-cortex read), L2-normalised for cosine."""
    code = ppmi_matrix(C, 0.75)
    nrm = np.linalg.norm(code, axis=1, keepdims=True)
    return code / (nrm + 1e-12)


def affinity_knn(codes, k):
    """The LEARNED concept-association graph: cosine affinity, self-zeroed, top-k per node, symmetrised,
    row-normalised -> a row-stochastic propagation operator P."""
    W = codes @ codes.T
    np.fill_diagonal(W, 0.0)
    W = np.maximum(W, 0.0)
    n = W.shape[0]
    if k and k < n - 1:
        Wk = np.zeros_like(W)
        for i in range(n):
            nn = np.argpartition(W[i], -k)[-k:]
            Wk[i, nn] = W[i, nn]
        W = np.maximum(Wk, Wk.T)  # symmetrise
    row = W.sum(1, keepdims=True)
    P = W / (row + 1e-12)
    return P


# =============================================================================================================
# 2 + 3. OPPONENT V+/V- seed + SPREADING-ACTIVATION inheritance (seed-clamped label propagation)
# =============================================================================================================
def opponent_seed(valence_1to9):
    """Warriner valence (1..9) -> rectified opponent (V+, V-). Namburi-Tye: separate populations."""
    s = (np.asarray(valence_1to9, float) - 5.0) / 4.0  # signed in [-1, 1]
    return np.maximum(s, 0.0), np.maximum(-s, 0.0)


def propagate(P, seed_mask, seed_vals, n_hops):
    """Seed-clamped spreading activation (Zhu-Ghahramani harmonic label propagation) of ONE channel over the
    learned graph: held-out nodes iteratively integrate the association-weighted mean of their neighbours; seed
    nodes are re-clamped each hop. n_hops small (1-2) = the EMERGE-30 member->neighbours read."""
    f = np.zeros(P.shape[0], float)
    f[seed_mask] = seed_vals[seed_mask]
    for _ in range(n_hops):
        f = P @ f
        f[seed_mask] = seed_vals[seed_mask]  # clamp the seeds
    return f


def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# =============================================================================================================
# One seed = one held-out split; runs the real arm + all anti-cheat arms
# =============================================================================================================
def run_seed(seed, codes, vocab, seed_frac, k, n_hops):
    rng = np.random.default_rng(seed)
    n = len(vocab)
    val = np.array([WARRINER[w][0] for w in vocab], float)
    aro = np.array([WARRINER[w][1] for w in vocab], float)
    s_true = (val - 5.0) / 4.0

    # seed / held-out split
    perm = rng.permutation(n)
    n_seed = int(round(seed_frac * n))
    seed_idx = perm[:n_seed]
    held_idx = perm[n_seed:]
    seed_mask = np.zeros(n, bool); seed_mask[seed_idx] = True

    vp_seed, vm_seed = opponent_seed(val)      # rectified opponent seeds (valence)

    P = affinity_knn(codes, k)

    def predict(P_, vp, vm):
        fp = propagate(P_, seed_mask, vp, n_hops)
        fm = propagate(P_, seed_mask, vm, n_hops)
        return fp, fm

    # --- REAL arm ---
    fp, fm = predict(P, vp_seed, vm_seed)
    net = fp - fm
    r_real = _pearson(net[held_idx], s_true[held_idx])

    # opponent-sign anti-cheat: V+ / V- genuinely opposed on held-out; net separates neg from pos
    r_opp = _pearson(fp[held_idx], fm[held_idx])
    neg_h = held_idx[s_true[held_idx] < -0.1]
    pos_h = held_idx[s_true[held_idx] > 0.1]
    sep = (float(net[pos_h].mean()) - float(net[neg_h].mean())) if len(neg_h) and len(pos_h) else 0.0

    # arousal (single-channel, reported secondary)
    ar_seed = (aro - 5.0) / 4.0
    fa = propagate(P, seed_mask, ar_seed, n_hops)
    r_arousal = _pearson(fa[held_idx], ar_seed[held_idx])

    # --- (1) PERMUTED-GRAPH: scramble which code belongs to which word -> random neighbourhoods ---
    pperm = rng.permutation(n)
    P_perm = affinity_knn(codes[pperm], k)
    fp2, fm2 = predict(P_perm, vp_seed, vm_seed)
    r_permuted = _pearson((fp2 - fm2)[held_idx], s_true[held_idx])

    # --- (2) SEED-ONLY baseline: constant mean seed valence, no graph ---
    const = np.full(n, s_true[seed_idx].mean())
    r_seedonly = _pearson(const[held_idx], s_true[held_idx])  # ~0 by construction (constant)

    # --- (3) SHUFFLED-SEED-LABELS: permute valence values across seed words ---
    val_sh = val.copy()
    val_sh[seed_idx] = val[rng.permutation(seed_idx)]
    vp_sh, vm_sh = opponent_seed(val_sh)
    fp3, fm3 = predict(P, vp_sh, vm_sh)
    r_shuffled = _pearson((fp3 - fm3)[held_idx], s_true[held_idx])

    return {
        "seed": int(seed), "n_vocab": int(n), "n_seed": int(n_seed), "n_held": int(len(held_idx)),
        "r_real": r_real, "r_permuted": r_permuted, "r_seedonly": r_seedonly, "r_shuffled": r_shuffled,
        "r_arousal": r_arousal, "opp_corr": r_opp, "net_sep_pos_minus_neg": sep,
    }


def _aggregate_verdict(rows, go_r=0.55, collapse=0.20):
    def m(k):
        return float(np.mean([r[k] for r in rows]))
    real, perm, seedonly, shuf = m("r_real"), m("r_permuted"), m("r_seedonly"), m("r_shuffled")
    aro, opp, sep = m("r_arousal"), m("opp_corr"), m("net_sep_pos_minus_neg")
    checks = {
        "held_out_r>=0.55": real >= go_r,
        "beats_seed_only": real >= seedonly + 0.30,
        "permuted_collapses": perm < collapse and real >= perm + 0.30,
        "shuffled_labels_collapse": shuf < collapse and real >= shuf + 0.30,
        "opponent_sign(V+ vs V- corr<0)": opp < 0.0,
        "opponent_separation(pos>neg)": sep > 0.0,
    }
    go = all(checks.values())
    means = {"r_real": real, "r_permuted": perm, "r_seedonly": seedonly, "r_shuffled": shuf,
             "r_arousal": aro, "opp_corr": opp, "net_sep": sep}
    return go, checks, means


# =============================================================================================================
# PRODUCTION MAP: a LEAVE-ONE-OUT learned per-word valence/arousal map (composes the de-risked primitives above,
# NO reimplementation). Every word's value is inferred by seed-clamped propagation over the LEARNED co-occurrence
# graph seeded from ALL OTHER words (the target word held out), so NO word carries its own hand-assigned norm --
# the per-word appraisal VALUE is fully experience-derived. Reused by the production affect organ (appraise_text)
# to source the appraisal VALUE from DR-2 learned propagation instead of the raw norm dict. Deterministic (no RNG).
# HONEST: the SEED norms + the affect-salience vocabulary are still Warriner (this map does NOT retire them -- it is
# SEEDED from them + propagated in numpy); the fully-spiking opponent-population appraisal is the further rung.
# =============================================================================================================
def build_learned_valence_map(max_stories=60000, n_hub=500, window=4, min_count=5, knn=12, n_hops=2):
    """Return ({word: [v9_learned, a9_learned]} on the 1..9 Warriner scale, meta). Leave-one-out label-propagation
    over the learned co-occurrence graph; a single global std-ratio gain restores the norm SPREAD (a scalar readout
    calibration -- per-word sign/rank is 100% learned). Deterministic."""
    stories = load_stories(max_stories)
    vocab, C = build_cooccurrence(stories, n_hub, window, min_count)
    codes = codes_from_cooccurrence(C)
    P = affinity_knn(codes, knn)
    n = len(vocab)
    val = np.array([WARRINER[w][0] for w in vocab], float)
    aro = np.array([WARRINER[w][1] for w in vocab], float)
    sV = (val - 5.0) / 4.0
    sA = (aro - 5.0) / 4.0
    vp_seed, vm_seed = opponent_seed(val)
    lv = np.zeros(n, float)   # learned signed valence, leave-one-out
    la = np.zeros(n, float)   # learned signed arousal,  leave-one-out
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        lv[i] = propagate(P, m, vp_seed, n_hops)[i] - propagate(P, m, vm_seed, n_hops)[i]
        la[i] = propagate(P, m, sA, n_hops)[i]
    gV = float(sV.std() / (lv.std() + 1e-12))   # global std-ratio gains (scalar spread calibration to the norms)
    gA = float(sA.std() / (la.std() + 1e-12))
    v9 = np.clip(5.0 + gV * lv * 4.0, 1.0, 9.0)
    a9 = np.clip(5.0 + gA * la * 4.0, 1.0, 9.0)
    mp = {vocab[i]: [round(float(v9[i]), 4), round(float(a9[i]), 4)] for i in range(n)}
    meta = {"n_words": n, "gain_valence": round(gV, 4), "gain_arousal": round(gA, 4),
            "config": {"max_stories": max_stories, "n_hub": n_hub, "window": window,
                       "min_count": min_count, "knn": knn, "n_hops": n_hops},
            "n_hubs": int(C.shape[1]),
            "note": "leave-one-out DR-2 distributional valence; SEEDED from Warriner norms (NOT a retirement); "
                    "numpy PPMI + label-prop (NOT spiking). Fully-spiking opponent-population appraisal = next rung."}
    return mp, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true", help="1 seed, tiny corpus -- proves it RUNS + controls live")
    ap.add_argument("--max-stories", type=int, default=60000, help="cap stories (smoke uses a tiny slice)")
    ap.add_argument("--n-hub", type=int, default=500)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--seed-frac", type=float, default=0.5, help="fraction of vocab seeded from norms")
    ap.add_argument("--knn", type=int, default=12)
    ap.add_argument("--n-hops", type=int, default=2, help="spreading-activation hops (1-2 = EMERGE-30 read)")
    ap.add_argument("--warriner-csv", default=None, help="optional real Warriner CSV: word,valence[,arousal]")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--emit-map", default=None, help="build+write the LEAVE-ONE-OUT learned valence map (JSON) "
                    "for the production affect organ, then exit (composes the de-risked primitives; deterministic)")
    a = ap.parse_args()

    if a.emit_map:
        t0 = time.time()
        mp, meta = build_learned_valence_map(a.max_stories, a.n_hub, a.window, a.min_count, a.knn, a.n_hops)
        Path(a.emit_map).parent.mkdir(parents=True, exist_ok=True)
        Path(a.emit_map).write_text(json.dumps({"map": mp, "meta": meta}, indent=1, default=str))
        print(f"[emit-map] wrote {len(mp)} learned per-word valence/arousal entries to {a.emit_map} "
              f"(gV={meta['gain_valence']}, gA={meta['gain_arousal']}, {round(time.time()-t0,1)}s)", flush=True)
        return 0

    if a.warriner_csv and os.path.exists(a.warriner_csv):
        WARRINER.clear()
        with open(a.warriner_csv) as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    try:
                        w = parts[0].strip().lower()
                        v = float(parts[1]); ar = float(parts[2]) if len(parts) >= 3 else 5.0
                        WARRINER[w] = (v, ar)
                    except ValueError:
                        pass
        print(f"[warriner] loaded {len(WARRINER)} words from {a.warriner_csv}", flush=True)

    if a.smoke:
        a.seeds = [a.seeds[0]]
        a.max_stories = min(a.max_stories, 8000)

    t0 = time.time()
    print(f"[affect-tag DR-2] seeds={a.seeds} smoke={a.smoke} max_stories={a.max_stories} window={a.window} "
          f"knn={a.knn} hops={a.n_hops} seed_frac={a.seed_frac}", flush=True)
    stories = load_stories(a.max_stories)
    vocab, C = build_cooccurrence(stories, a.n_hub, a.window, a.min_count)
    codes = codes_from_cooccurrence(C)
    print(f"  learned co-occurrence cortex: {len(stories)} stories -> {len(vocab)} Warriner-labelled targets "
          f"x {C.shape[1]} hubs ({int(C.sum())} co-occurrence events)", flush=True)
    if len(vocab) < 20:
        print(f"NOT-RUNNABLE: only {len(vocab)} labelled targets appear in the corpus (need >=20 for a "
              f"meaningful correlation). Lower --min-count or raise --max-stories.", flush=True)
        return 2

    rows = [run_seed(s, codes, vocab, a.seed_frac, a.knn, a.n_hops) for s in a.seeds]
    for r in rows:
        print(f"  [seed {r['seed']}] held-out valence r {r['r_real']:+.3f} (n_held={r['n_held']}) || "
              f"permuted {r['r_permuted']:+.3f} | shuffled-labels {r['r_shuffled']:+.3f} | seed-only "
              f"{r['r_seedonly']:+.3f} || opp-corr {r['opp_corr']:+.3f} sep {r['net_sep_pos_minus_neg']:+.2f} | "
              f"arousal {r['r_arousal']:+.3f}", flush=True)

    go, checks, means = _aggregate_verdict(rows)
    n = len(a.seeds)
    if go:
        verdict = (
            f"GO ({n}-seed) -- DISTRIBUTIONAL AFFECT TAG: a concept's valence EMERGES from the affective company it "
            f"keeps. Held-out concepts (never seeded from norms) inherit valence by spreading activation over the "
            f"LEARNED co-occurrence graph, correlating r={means['r_real']:+.3f} with Warriner valence (>= 0.55). "
            f"PERMUTED-GRAPH collapses ({means['r_permuted']:+.3f} -- random neighbourhoods carry no affect), "
            f"SHUFFLED-SEED-LABELS collapses ({means['r_shuffled']:+.3f}), and the graph BEATS the seed-only "
            f"constant baseline ({means['r_seedonly']:+.3f}). The opponent code is genuine: V+ and V- are OPPOSED "
            f"on held-out (corr {means['opp_corr']:+.3f} < 0) and net valence separates positive from negative "
            f"(sep {means['net_sep']:+.2f}). => the world-model is affectively COLOURED, not just factual; the "
            f"affect tag rides the LEARNED structure, not a lookup. numpy-CPU; NO sim/ edit. (Arousal secondary, "
            f"r={means['r_arousal']:+.3f} -- the known weaker ceiling.)")
    else:
        miss = [k for k, v in checks.items() if not v]
        verdict = (f"BOUNDARY (build-informative, {n}-seed) -- held-out r={means['r_real']:+.3f} "
                   f"(permuted {means['r_permuted']:+.3f} / shuffled {means['r_shuffled']:+.3f} / seed-only "
                   f"{means['r_seedonly']:+.3f}; opp-corr {means['opp_corr']:+.3f}; sep {means['net_sep']:+.2f}). "
                   f"FAILED: {miss}. Tune --knn / --n-hops / --seed-frac / --window / --min-count; distributional "
                   f"affect inheritance is the next tuning, not a wall.")

    summary = {
        "probe": "affect_distributional_tag (DR-2, P0.1)", "verdict": verdict, "GO": bool(go),
        "checks": checks, "means": means, "per_seed": rows,
        "config": {"seeds": a.seeds, "smoke": a.smoke, "max_stories": a.max_stories, "n_hub": a.n_hub,
                   "window": a.window, "min_count": a.min_count, "seed_frac": a.seed_frac, "knn": a.knn,
                   "n_hops": a.n_hops, "n_vocab": len(vocab)},
        "mechanism": "learned co-occurrence graph (rate-Hebbian, matched rule; PPMI stream-cortex code) -> "
                     "opponent V+/V- seed from Warriner norms (Namburi-Tye) -> valence inherited to held-out by "
                     "seed-clamped spreading activation (EMERGE-30 read / Zhu-Ghahramani harmonic) -> net = V+ - V-",
        "HONEST_NOTE": "numpy-CPU read of the mechanism (on-bridge spiking opponent-population confirm = GPU "
                       "follow-on). Warriner values are an embedded APPROXIMATE core lexicon biased to children's "
                       "vocabulary (real 13,915-word CSV swappable via --warriner-csv); the learned co-occurrence "
                       "graph is BUILT IN-RUNNER from TinyStories because no cached artifact maps concept codes to "
                       "Warriner-labelled words. Seed/held-out split varies per RNG seed.",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110, flush=True)
    print(f"[affect-tag] VERDICT: {verdict}", flush=True)
    print(f"[affect-tag] wrote {a.out}  ({summary['elapsed_seconds']}s)\n" + "=" * 110, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
