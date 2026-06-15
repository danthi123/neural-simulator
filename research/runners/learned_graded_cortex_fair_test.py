"""FAIR-TEST de-risk of the LEARNED graded cortex (L1 = PPMI-input + similarity-matching).

Owner directive (2026-06-14): "Better-resourced de-risk first" -- a day-scale FAIR test of the L1
similarity-matching learned-embedding route BEFORE any months-scale commitment. The prior cheap attempt
(`learned_graded_cortex_log_simmatch_derisk.py`) was UNFAIR in two ways that this runner corrects:
  1. its "log + marginal-divide" input was NOT PPMI. PPMI is log OF the marginal-ratio
     (log(P(a,b)/(P(a)P(b)))), i.e. the marginal normalization happens INSIDE the log (a subtraction),
     not as a divide AFTER the log. This runner uses the host's EXACT PPMI (matched byte-for-byte to
     `option_c_paradigmatic_host_precheck.ppmi_svd_sim`, alpha context-smoothing included).
  2. it had NO SVD-grade low-rank denoising -- the very step that lifts the host from cos(PPMI rows) up
     to its +0.442 ceiling. Similarity-matching (Pehlevan-Chklovskii) IS the brain-based online
     equivalent of PPMI+PCA (Levy-Goldberg 2014: word2vec ~ implicit PMI factorization; Pehlevan-
     Chklovskii 2015: similarity-matching ~ online PCA), so its ACHIEVABLE ceiling is the host. This
     runner adds the offline PPMI+PCA reference arm AND runs the online learner well-converged (many
     epochs, convergence trajectory logged).

THE BRAIN-PLAUSIBILITY SCOPING (honest, load-bearing): PPMI-as-input is the brain-plausible half because
PPMI decomposes into operations the project has ALREADY built or identified neural realizations for:
  * log(count)              = Weber-Fechner / dendritic compression (the CONFIRMED input-half fix);
  * /(row_sum * col_sum)    = divisive normalization by the pre/post activity marginals -- EXACTLY the
                              Phase-1 dendritic divisive gain g=sigma/(sigma+EMA) (cp_dendritic_source_activity);
  * max(., 0)               = the spiking rheobase threshold.
So this de-risks the LEARNING half (does the brain-based online rule recover the structure?) holding the
input half at its brain-plausible PPMI form. A GO here is a GREEN LIGHT to BUILD the spiking
similarity-matching cortex (Pehlevan 2019 has a published integrate-and-fire realization) on PPMI-shaped
input -- NOT a claim it is already built.

THE ARMS (all on the SAME concept x context-hub count matrix C; multi-seed; the contrast IS the result):
  HOST           PPMI + truncated-SVD(k) + cosine        -- the +0.442 ceiling (the target; carries-gate).
  PPMI_cos       cosine of PPMI rows, full-rank          -- structure IN PPMI before any denoising.
  PPMI_PCA       offline PCA(k) of PPMI + cosine         -- the OFFLINE simmatch optimum (~= HOST).
  SIMMATCH_PPMI  online similarity-matching on PPMI      -- *** THE brain-based learner under test ***.
  A_SAT_RAW      faithful truly-saturating Hebbian, raw  -- the FAILURE control (uniform blob, must ~0).
  (ablation)     SIMMATCH_RAW / SIMMATCH_LOG            -- the input-encoding lesion (must trail PPMI).

GATES (multi-seed 42/43/44; reuse the project metrics exactly):
  host_carries        : HOST  >= host_bar (the data + ceiling are real).
  A_fails             : A_SAT_RAW <= a_fail_bar (the faithful saturating control reproduces the failure).
  simmatch_reaches    : SIMMATCH_PPMI >= reach_frac * HOST AND >= structure_bar  (reaches the ceiling).
  simmatch_generalizes: held-out (Fodor-Pylyshyn) generalization above chance for SIMMATCH_PPMI.
  permuted_collapses  : permuted-similarity Pearson ~ 0 (structure not an artifact).
  input_lesion        : SIMMATCH_PPMI >> SIMMATCH_RAW (the PPMI input is load-bearing).
  rule_lesion         : SIMMATCH_PPMI >> A_SAT on the SAME input (the learning rule is load-bearing).

OUTCOMES:
  GO       SIMMATCH_PPMI reaches the host ceiling + generalizes + controls clean ==> the brain-based learned
           cortex is VIABLE (PPMI-input + similarity-matching, both brain-plausible); escalate to the
           weeks-scale SPIKING similarity-matching build (Pehlevan 2019). The dendritic D2 rewrite is OFF
           the critical path (the divisive gain lives in the input encoding, already built as Phase 1).
  BOUNDARY SIMMATCH_PPMI clearly beats the saturating + input-lesion controls but FALLS SHORT of the host
           ceiling ==> right input + rule, online-convergence (or input-brain-plausibility) gap; the deeper
           dendritic/spiking build is warranted WITH A SHARP TARGET (the measured shortfall).
  NEGATIVE SIMMATCH_PPMI ~= the saturating control even from PPMI input ==> the gap is deeper than
           input-encoding + learning-rule ==> ship the flat 2,048-concept cortex (Option A) for the
           conversational goal; the learned-from-experience cortex stays the deep artificial-life frontier.

Run (synthetic calibration -- fast, structure known a-priori; confirms the control fails + simmatch works):
  python -u -m research.runners.learned_graded_cortex_fair_test --smoke --seeds 42,43,44
Run (real Option-C TinyStories corpus -- THE decisive test):
  python -u -m research.runners.learned_graded_cortex_fair_test --real-corpus --seeds 42,43,44 \
      --n-hub 2000 --epochs 200 --out research/findings/raw/_l1_fair_real_multiseed.json
NO sim/ edits; CPU/numpy (tiny matrices, minutes not hours). Reuse-by-import for taxonomy + host + metrics.
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

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    build_concept_hub_counts, _cos_sim, _pearson_vs_Strue, heldout_generalization, effective_rank,
)
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


# ===========================================================================
# Input encoding: TRUE PPMI (matched EXACTLY to the host's ppmi_svd_sim PPMI block, alpha smoothing).
# Returns the PPMI ROW matrix (rows = per-concept PPMI profile over the context hubs) to feed as input X.
# ===========================================================================
def ppmi_matrix(C, alpha):
    """Positive Pointwise Mutual Information. log OF the marginal-ratio (the marginalization is INSIDE the
    log), then max(.,0). Identical algebra to option_c_paradigmatic_host_precheck.ppmi_svd_sim. This is the
    brain-plausible input: log (dendritic) + divisive-by-activity-marginal (Phase-1 gain) + threshold (ReLU)."""
    M = np.maximum(C, 0.0).astype(np.float64)
    row_sum = M.sum(1, keepdims=True)
    col_sum = M.sum(0, keepdims=True)
    if alpha != 1.0:
        col_sum = col_sum ** alpha
    total = col_sum.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((M * total) / (row_sum * col_sum + 1e-12) + 1e-12)
    return np.maximum(pmi, 0.0)


def encode_raw(C):
    return np.maximum(C, 0.0).astype(np.float64)


def encode_log(C):
    return np.log1p(np.maximum(C, 0.0).astype(np.float64))


# ===========================================================================
# Offline PPMI + PCA(k): the OFFLINE optimum of similarity-matching (Pehlevan-Chklovskii). ~= the host.
# Centered SVD low-rank -> normalized embedding -> cosine. The achievable brain-based ceiling reference.
# ===========================================================================
def pca_lowrank_sim(X, k):
    Xc = X - X.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    kk = min(k, len(S))
    emb = U[:, :kk] * S[:kk]
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb @ emb.T


# ===========================================================================
# (A) FAITHFUL truly-saturating Hebbian (the failure control): pure Hebbian feedforward accumulation with NO
# output normalization, NO lateral, NO decay -> the weights grow toward a UNIFORM BLOB (every output unit
# learns the same dominant-hub-weighted profile). Reproduces the project's spiking failure (structure ~ 0,
# Pearson(W,counts) ~ +0.06). The only bound is a hard clip = the spiking firing-rate ceiling.
# ===========================================================================
def learn_faithful_saturating(X, k, epochs, lr, seed, w_clip=5.0):
    rng = np.random.RandomState(seed * 7919 + 1)
    Nc, H = X.shape
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)   # bounded input (rate analogue of spiking)
    W_ff = rng.randn(k, H) * 0.01
    order = np.arange(Nc)
    for _ in range(epochs):
        rng.shuffle(order)
        for i in order:
            x = Xn[i]
            y = W_ff @ x                       # NO output normalization -> the saturation is faithful
            W_ff += lr * np.outer(y, x)        # pure Hebbian, no competition -> uniform blob
            np.clip(W_ff, -w_clip, w_clip, out=W_ff)
    codes = (W_ff @ Xn.T).T
    # the saturation signature: Pearson of the learned weight rows' mean-profile vs the raw count column-profile
    w_profile = np.abs(W_ff).mean(0)
    c_profile = Xn.mean(0)
    if w_profile.std() < 1e-12 or c_profile.std() < 1e-12:
        pear_w_counts = 0.0
    else:
        pear_w_counts = float(np.corrcoef(w_profile, c_profile)[0, 1])
    return codes, W_ff, pear_w_counts


# ===========================================================================
# (B) SIMILARITY-MATCHING (Pehlevan-Chklovskii) online learn: settled output (feedforward - LEARNED lateral)
# + Oja feedforward + anti-Hebbian lateral with the FIXED POINT (Delta-M ~ y yT - M, the -M decay gives a
# fixed point so the lateral does NOT fill to a uniform blob). Output similarities match input similarities;
# with k < H it is a low-rank (denoising) similarity-preserving embedding = the brain-based PPMI+PCA.
# Tracks the convergence trajectory (Pearson(S_codes, S_true) per checkpoint).
# ===========================================================================
def learn_simmatch(X, S_true, k, epochs, lr_ff, lr_m, settle_steps, seed, track_every=0):
    rng = np.random.RandomState(seed * 104729 + 3)
    Nc, H = X.shape
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)   # preserve the COSINE structure + stability
    W_ff = rng.randn(k, H) * 0.1
    M = np.zeros((k, k), dtype=np.float64)
    order = np.arange(Nc)
    traj = []

    def read_codes():
        out = np.zeros((Nc, k))
        for i in range(Nc):
            ff = W_ff @ Xn[i]
            y = np.zeros(k)
            for _ in range(settle_steps):
                y = 0.5 * y + 0.5 * (ff - M @ y)
            out[i] = y
        return out

    for ep in range(epochs):
        rng.shuffle(order)
        for i in order:
            x = Xn[i]
            ff = W_ff @ x
            y = np.zeros(k)
            for _ in range(settle_steps):
                y = 0.5 * y + 0.5 * (ff - M @ y)
            W_ff += lr_ff * (np.outer(y, x) - (y ** 2)[:, None] * W_ff)   # Oja feedforward (self-normalizing)
            dM = np.outer(y, y) - M                                       # anti-Hebbian lateral, fixed point
            np.fill_diagonal(dM, 0.0)
            M += lr_m * dM
        if track_every and (ep + 1) % track_every == 0:
            traj.append((ep + 1, float(_pearson_vs_Strue(_cos_sim(read_codes()), S_true))))
    return read_codes(), W_ff, traj


# ===========================================================================
# Corpus builders
# ===========================================================================
def build_real_corpus(seed, n_hub, window=2):
    from research.runners.option_c_stageB_fair_test import build_context_inclusive_cooccurrence
    from research.runners.option_c_real_cooccurrence_derisk import TAXONOMY_8x8, taxonomy_to_vocab_categories
    from research.runners.learned_graded_embedding_diagnose import raw_count_matrix
    vocab, cat_ids, _ = taxonomy_to_vocab_categories(TAXONOMY_8x8)
    corpus = build_context_inclusive_cooccurrence(
        os.path.join(_REPO, "data", "corpus", "tinystories.txt"), vocab, cat_ids,
        window=window, n_context_hubs=n_hub, repeat_cap=40, seed=seed, verbose=False)
    Nt = len(vocab)
    C = raw_count_matrix(corpus["concepts"], corpus["facts"])[:Nt, Nt:]   # targets x context hubs
    return C, np.asarray(cat_ids), corpus["S_true"]


def _arm(name, codes, S_true, labels):
    pear = _pearson_vs_Strue(_cos_sim(codes), S_true)
    gen, chance = heldout_generalization(codes, labels)
    off = float(_cos_sim(codes)[np.triu_indices(codes.shape[0], 1)].mean())
    rank = effective_rank(codes)
    print(f"    [{name:16s}] Pearson(S,S_true)={pear:+.3f}  gen={gen:.3f} (chance {chance:.3f})  "
          f"offdiag-cos={off:+.3f}  eff-rank={rank:.1f}", flush=True)
    return {"pearson": pear, "gen": gen, "chance": chance, "offdiag": off, "eff_rank": rank}


def run_seed(seed, args):
    print(f"\n{'='*96}\n  L1 FAIR-TEST (seed {seed})  --  PPMI-input + similarity-matching vs host PPMI+SVD\n{'='*96}",
          flush=True)
    if args.real_corpus:
        C, labels, S_true = build_real_corpus(seed, args.n_hub, args.window)
    else:
        C, labels, S_true, _ = build_concept_hub_counts(
            args.n_cat, args.per_cat, args.n_common, args.n_sig_per_cat,
            args.lam_common, args.lam_sig, args.lam_bg, seed)
    Nc, H = C.shape
    svd_dim = min(args.host_svd, min(C.shape) - 1)

    # --- reference ceilings (host = PPMI+SVD; PPMI_cos = pre-denoise; PPMI_PCA = offline simmatch optimum) ---
    host_sim = ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=svd_dim, alpha=args.host_alpha)
    host_p, host_margin, host_nn, _ = score(host_sim, labels)
    Xppmi = ppmi_matrix(C, args.host_alpha)
    Xlog = encode_log(C)
    Xraw = encode_raw(C)
    ppmi_cos_p = _pearson_vs_Strue(_cos_sim(Xppmi), S_true)
    ppmi_pca_p = _pearson_vs_Strue(pca_lowrank_sim(Xppmi, args.k), S_true)
    print(f"  {Nc} concepts x {H} hubs;  HOST PPMI+SVD(k={svd_dim},a={args.host_alpha})={host_p:+.3f}  "
          f"(margin {host_margin:+.3f}, nn-same {host_nn:.3f})", flush=True)
    print(f"  reference: cos(PPMI rows) full-rank={ppmi_cos_p:+.3f}  |  offline PPMI+PCA(k={args.k})="
          f"{ppmi_pca_p:+.3f}  (the OFFLINE similarity-matching optimum ~= host)", flush=True)

    # --- the learners ---
    print(f"  [arms: online similarity-matching (k={args.k}, {args.epochs} epochs) + faithful saturating control]",
          flush=True)
    sm_codes, _, traj = learn_simmatch(Xppmi, S_true, args.k, args.epochs, args.lr_ff, args.lr_m,
                                       args.settle_steps, seed, track_every=max(1, args.epochs // 5))
    SM = _arm("SIMMATCH_PPMI", sm_codes, S_true, labels)
    if traj:
        print("    [simmatch convergence] " + "  ".join(f"ep{e}:{p:+.3f}" for e, p in traj), flush=True)
    smraw_codes, _, _ = learn_simmatch(Xraw, S_true, args.k, args.epochs, args.lr_ff, args.lr_m,
                                       args.settle_steps, seed)
    SMR = _arm("SIMMATCH_RAW", smraw_codes, S_true, labels)
    smlog_codes, _, _ = learn_simmatch(Xlog, S_true, args.k, args.epochs, args.lr_ff, args.lr_m,
                                       args.settle_steps, seed)
    SML = _arm("SIMMATCH_LOG", smlog_codes, S_true, labels)
    sat_codes, _, pear_wc = learn_faithful_saturating(Xraw, args.k, args.epochs, args.lr_sat, seed)
    SAT = _arm("A_SAT_RAW", sat_codes, S_true, labels)
    print(f"    [saturation signature] Pearson(|W|-profile, count-profile)={pear_wc:+.3f}  (uniform blob ~ small)",
          flush=True)

    # --- anti-cheat on the TEST arm ---
    rng = np.random.RandomState(seed * 2718281 + 1)
    perm = rng.permutation(labels)
    S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
    sm_perm = _pearson_vs_Strue(_cos_sim(sm_codes), S_perm)
    print(f"  [anti-cheat] SIMMATCH_PPMI permuted-similarity Pearson={sm_perm:+.3f} (~0)", flush=True)

    chance = SM["chance"]
    reach = args.reach_frac * max(host_p, 1e-9)
    gates = {
        "host_carries": bool(host_p >= args.host_bar),
        "A_saturating_fails": bool(abs(SAT["pearson"]) <= args.a_fail_bar),
        "simmatch_reaches_ceiling": bool(SM["pearson"] >= reach and SM["pearson"] >= args.structure_bar),
        "simmatch_generalizes": bool(SM["gen"] > chance + args.gen_margin),
        "permuted_collapses": bool(abs(sm_perm) <= args.a_fail_bar),
        "input_lesion_collapses": bool(SM["pearson"] >= SMR["pearson"] + args.lesion_gap),
        "rule_lesion_collapses": bool(SM["pearson"] >= SAT["pearson"] + args.lesion_gap),
    }
    print(f"  [seed {seed} gates] {gates}", flush=True)
    return {"seed": seed, "n_concepts": Nc, "n_hub": H, "svd_dim": svd_dim,
            "host_ceiling": host_p, "host_margin": host_margin, "host_nn_same": host_nn,
            "ppmi_cos_fullrank": ppmi_cos_p, "ppmi_pca_offline": ppmi_pca_p, "chance": chance,
            "SIMMATCH_PPMI": SM, "SIMMATCH_RAW": SMR, "SIMMATCH_LOG": SML, "A_SAT_RAW": SAT,
            "saturation_pearson_w_counts": pear_wc, "simmatch_permuted_pearson": sm_perm,
            "simmatch_traj": traj, "reach_target": reach, "gates": gates}


def decide_verdict(per_seed, seeds, args):
    def allg(k):
        return all(per_seed[str(s)]["gates"][k] for s in seeds)
    host_ok = allg("host_carries"); a_fails = allg("A_saturating_fails")
    reaches = allg("simmatch_reaches_ceiling"); generalize = allg("simmatch_generalizes")
    controls = allg("permuted_collapses") and allg("input_lesion_collapses") and allg("rule_lesion_collapses")
    hmean = float(np.mean([per_seed[str(s)]["host_ceiling"] for s in seeds]))
    smean = float(np.mean([per_seed[str(s)]["SIMMATCH_PPMI"]["pearson"] for s in seeds]))
    amean = float(np.mean([per_seed[str(s)]["A_SAT_RAW"]["pearson"] for s in seeds]))
    rmean = float(np.mean([per_seed[str(s)]["SIMMATCH_RAW"]["pearson"] for s in seeds]))
    pcamean = float(np.mean([per_seed[str(s)]["ppmi_pca_offline"] for s in seeds]))
    sgen = float(np.mean([per_seed[str(s)]["SIMMATCH_PPMI"]["gen"] for s in seeds]))
    frac = smean / hmean if hmean > 1e-9 else 0.0
    abl = (f"HOST {hmean:+.3f} | PPMI+PCA(offline) {pcamean:+.3f} | SIMMATCH_PPMI {smean:+.3f} "
           f"({frac:.0%} of host, gen {sgen:.3f}) | SIMMATCH_RAW {rmean:+.3f} | A_SAT {amean:+.3f}")
    if not host_ok:
        verdict, why = "NEGATIVE_miscalibrated", f"host ceiling did not carry ({hmean:+.3f}) -> data/encoding issue. {abl}"
    elif not a_fails:
        verdict, why = "NEGATIVE_miscalibrated", (f"the faithful saturating control did NOT fail ({amean:+.3f}) "
                                                  f"-> the toy doesn't reproduce the Option-C failure; re-tune. {abl}")
    elif reaches and generalize and controls:
        verdict = "GO"
        why = (f"online similarity-matching on PPMI input REACHES the host ceiling "
               f"(SIMMATCH_PPMI {smean:+.3f} = {frac:.0%} of host {hmean:+.3f}, generalizes {sgen:.3f}) WHILE the "
               f"faithful saturating control fails ({amean:+.3f}) and the input lesion (raw {rmean:+.3f}) trails; "
               f"controls clean. ==> the brain-based learned cortex is VIABLE (PPMI-input + similarity-matching, "
               f"both brain-plausible); escalate to the weeks-scale SPIKING similarity-matching build "
               f"(Pehlevan 2019). The dendritic D2 rewrite is OFF the critical path. {abl}")
    elif (smean > amean + args.lesion_gap) and (smean > rmean + args.lesion_gap) and controls:
        verdict = "BOUNDARY"
        why = (f"online similarity-matching on PPMI BEATS the saturating + input-lesion controls (SIMMATCH_PPMI "
               f"{smean:+.3f} vs A_SAT {amean:+.3f}, raw {rmean:+.3f}) but FALLS SHORT of the host ceiling "
               f"({frac:.0%} of {hmean:+.3f}; offline PPMI+PCA optimum {pcamean:+.3f}) -> right input + rule, an "
               f"online-convergence (or input-brain-plausibility) gap; the deeper spiking/dendritic build is "
               f"warranted WITH A SHARP TARGET (the {hmean - smean:+.3f} shortfall). {abl}")
    else:
        verdict = "NEGATIVE"
        why = (f"online similarity-matching on PPMI does NOT clear the saturating control on real data "
               f"(SIMMATCH_PPMI {smean:+.3f} vs A_SAT {amean:+.3f}) -> the gap is deeper than input-encoding + "
               f"learning-rule -> ship the flat 2,048-concept cortex (Option A) for the conversational goal; the "
               f"learned-from-experience cortex stays the deep artificial-life frontier. {abl}")
    return verdict, why, {"host_mean": hmean, "simmatch_ppmi_mean": smean, "simmatch_raw_mean": rmean,
                          "a_sat_mean": amean, "ppmi_pca_offline_mean": pcamean, "simmatch_gen_mean": sgen,
                          "reach_fraction_of_host": frac, "host_carries_all": host_ok, "a_fails_all": a_fails,
                          "reaches_all": reaches, "generalize_all": generalize, "controls_all": controls}


def main():
    p = argparse.ArgumentParser(description="L1 fair-test: PPMI-input + similarity-matching vs host PPMI+SVD")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--real-corpus", action="store_true")
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--n-hub", type=int, default=2000, help="(real-corpus) number of context hubs (better-resourced)")
    # synthetic toy
    p.add_argument("--n-cat", type=int, default=8); p.add_argument("--per-cat", type=int, default=8)
    p.add_argument("--n-common", type=int, default=200); p.add_argument("--n-sig-per-cat", type=int, default=12)
    p.add_argument("--lam-common", type=float, default=40.0); p.add_argument("--lam-sig", type=float, default=4.0)
    p.add_argument("--lam-bg", type=float, default=0.3)
    # embedding + learners (better-resourced: more epochs, bigger settle)
    p.add_argument("--k", type=int, default=64, help="embedding dimension")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr-ff", type=float, default=0.01); p.add_argument("--lr-m", type=float, default=0.01)
    p.add_argument("--settle-steps", type=int, default=30)
    p.add_argument("--lr-sat", type=float, default=0.02, help="faithful saturating-Hebbian learn rate")
    p.add_argument("--host-svd", type=int, default=50); p.add_argument("--host-alpha", type=float, default=0.75)
    # gate bars
    p.add_argument("--structure-bar", type=float, default=0.30)
    p.add_argument("--a-fail-bar", type=float, default=0.15)
    p.add_argument("--host-bar", type=float, default=0.30)
    p.add_argument("--gen-margin", type=float, default=0.10)
    p.add_argument("--reach-frac", type=float, default=0.70, help="SIMMATCH must reach this fraction of host")
    p.add_argument("--lesion-gap", type=float, default=0.10, help="min Pearson gap for a lesion to count as collapsed")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.smoke:
        os.environ.setdefault("SIM_BACKEND", "numpy")
        args.epochs = min(args.epochs, 60)

    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    print(f"[L1 fair-test] seeds={seeds} smoke={args.smoke} real_corpus={args.real_corpus} n_hub={args.n_hub} "
          f"k={args.k} epochs={args.epochs} reach_frac={args.reach_frac}", flush=True)
    per_seed = {str(s): run_seed(s, args) for s in seeds}
    verdict, why, detail = decide_verdict(per_seed, seeds, args)
    print(f"\n{'='*96}\n  VERDICT: {verdict}\n  {why}\n{'='*96}", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    out = {"verdict": verdict, "why": why, "detail": detail, "seeds": seeds, "smoke": bool(args.smoke),
           "real_corpus": bool(args.real_corpus), "config": vars(args), "per_seed": per_seed,
           "note": ("Owner-directed 'better-resourced de-risk first': true PPMI input (host-matched) + "
                    "SVD-grade low-rank (offline PPMI+PCA reference) + converged online similarity-matching "
                    "(Pehlevan-Chklovskii) + a faithful truly-saturating Hebbian control, on the real Option-C "
                    "TinyStories corpus. Resolves whether L1 reaches the host ceiling on real data. NO sim/ edits. "
                    "PPMI-as-input is brain-plausible (log=dendritic, /marginal=Phase-1 divisive gain, max=threshold); "
                    "a GO greenlights the spiking similarity-matching build, it does not claim it is built.")}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(raw_dir, f"_l1_fair_{'smoke' if args.smoke else 'real'}_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
