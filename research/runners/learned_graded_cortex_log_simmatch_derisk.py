"""LEARNED graded cortex de-risk -- LOG-compressed input + SIMILARITY-MATCHING learn vs the saturating
Hebbian control (the deep-research §(e) test; resolves whether the dendritic D2 build is on the critical path).

Design: `research/findings/2026-06-14-learned-graded-cortex-deep-research.md` §(e).

THE QUESTION: does (i) LOG-compressing the input (drive ~ log1p(count), Weber-Fechner -- un-silences the
low-count informative signal, the CONFIRMED input-half fix) + (ii) a SIMILARITY-MATCHING (Pehlevan-
Chklovskii, non-saturating, fixed-point) LEARNING rule let the POINT-neuron substrate LEARN the graded
category structure (and generalize) that the project's SATURATING Hebbian rule on raw counts cannot
(Option-C −0.008; the learned W saturates to a uniform blob, Pearson(W,counts)=+0.062)?  POINT-neuron, rate
learner first (the spiking read is the follow-on, mirroring D1->D1.5->D1.7). NO sim/ edits.

THE TWO MECHANISMS (each over the SAME concept x hub count matrix; codes = the learned per-concept embedding):
  SATURATING HEBBIAN (the failure baseline): pure Hebbian feedforward growth with NO normalization / NO
    lateral / NO decay -> the embedding weights grow toward a uniform blob (reproduces the diagnosed
    Pearson(W,counts)~0.06 saturation). `y = W_ff @ x; W_ff += lr * outer(y, x)`.
  SIMILARITY-MATCHING (the fix): the Pehlevan-Chklovskii online rule -- a settled output `y` (feedforward
    drive minus LEARNED lateral inhibition) + Hebbian feedforward (Oja-normalized) + ANTI-Hebbian lateral
    with the FIXED POINT `ΔM ∝ y yᵀ − M` (the `−M` decay is the single most important difference: it gives
    a fixed point at M=⟨y yᵀ⟩ so the lateral does NOT fill to a uniform blob). Output similarities match the
    input similarities -> a similarity-PRESERVING (graded, generalizing) embedding.

THE 4-ARM ABLATION (the fork-resolver; the contrast IS the result):
  A  raw  + saturating  -- the project's current pipeline. MUST reproduce ≈0 (the −0.008 failure).
  B  log  + saturating  -- does log-input rescue the EXISTING rule? (predicted: partial -- still saturates.)
  C  raw  + simmatch     -- does the better rule rescue raw counts? (predicted: partial -- threshold silences.)
  TEST log + simmatch    -- predicted GO (input-half + learning-half together).

GATES (multi-seed; reuse the project's exact metrics): structure Pearson(S_learned,S_true): TEST ≥ +0.30
WHILE A ≈ 0; W-FAITHFULNESS Pearson(W,counts) ≫ +0.06 for simmatch (saturation cured); held-out
GENERALIZATION above chance for TEST, at chance for A; HOST PPMI+SVD ceiling carries (the data is fine);
ANTI-CHEATS: permuted-similarity collapses, lesion (simmatch->saturating) collapses, host-is-a-labelled-
instrument-not-a-deliverable, S_true a-priori.

OUTCOMES: GO ⇒ the dendritic D2 build is OFF the critical path -> escalate to the weeks-scale spiking
similarity-matching build (Pehlevan 2019). BOUNDARY ⇒ dendritic L3 warranted with a sharp target. NEGATIVE
⇒ the gap is deeper -> ship the flat 2,048-concept cortex (Option A).

Run (CPU/numpy; synthetic smoke -- fast, the structure is known a-priori):
  python -u -m research.runners.learned_graded_cortex_log_simmatch_derisk --smoke --seeds 42,43,44
Run (real Option-C TinyStories corpus -- the decisive test):
  python -u -m research.runners.learned_graded_cortex_log_simmatch_derisk --real-corpus --seeds 42,43,44 \
      --out research/findings/raw/_log_simmatch_real_multiseed.json
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
# Input encodings: RAW counts vs LOG-compressed (Weber-Fechner). Rows = per-concept hub-profile inputs x_i.
# ===========================================================================
def encode_inputs(C, log_compress, marginal_norm):
    X = np.log1p(np.maximum(C, 0.0)) if log_compress else np.maximum(C, 0.0).astype(np.float64)
    if marginal_norm:
        # divide each hub column by its slow marginal (the PPMI-marginal / divisive-normalization arm):
        # down-weight high-frequency hubs. (sqrt to mirror PPMI's symmetric P(a)P(b) feel.)
        col = X.mean(0, keepdims=True) + 1e-9
        X = X / np.sqrt(col)
    return X


# ===========================================================================
# (A) SATURATING Hebbian embedding learn (the failure baseline): pure Hebbian feedforward growth, no
# normalization / lateral / decay -> the weights saturate (reproduces Pearson(W,counts)~0.06).
# ===========================================================================
def learn_saturating_hebbian(X, k, epochs, lr, seed, w_clip=2.0):
    """Pure Hebbian feedforward growth with NO normalization / lateral / decay. The weights grow toward the
    clip bound (the bounded 'uniform blob' the project's spiking learn produces -- co-firing overlapping
    patterns with no competition fills every weight to a similar value; the clip is the bound the spiking
    firing-rate ceiling provides, so the failure mode is reproduced WITHOUT numerical overflow). Normalize
    the input per presentation so the Hebbian term is bounded (the rate analogue of bounded spiking)."""
    rng = np.random.RandomState(seed * 7919 + 1)
    Nc, H = X.shape
    W_ff = rng.randn(k, H) * 0.01
    order = np.arange(Nc)
    for ep in range(epochs):
        rng.shuffle(order)
        for i in order:
            x = X[i] / (np.linalg.norm(X[i]) + 1e-9)   # bounded input (rate analogue of spiking ceiling)
            y = W_ff @ x
            y = y / (np.linalg.norm(y) + 1e-9)          # bounded output (no Oja decay -> still saturates)
            W_ff += lr * np.outer(y, x)                 # pure Hebbian -> grows to the clip = uniform blob
            np.clip(W_ff, -w_clip, w_clip, out=W_ff)
    codes = np.array([W_ff @ (X[i] / (np.linalg.norm(X[i]) + 1e-9)) for i in range(Nc)])
    return codes, W_ff


# ===========================================================================
# (B) SIMILARITY-MATCHING (Pehlevan-Chklovskii) online learn: settled output + Hebbian feedforward (Oja) +
# anti-Hebbian lateral with the FIXED POINT (ΔM ∝ y yᵀ − M). Output similarities match input similarities.
# ===========================================================================
def learn_simmatch(X, k, epochs, lr_ff, lr_m, settle_steps, seed, nonneg=False):
    rng = np.random.RandomState(seed * 104729 + 3)
    Nc, H = X.shape
    # unit-normalize each input -> the similarity-matching objective <y_i,y_j> ~ <x_i,x_j> then preserves the
    # COSINE structure (where the category structure lives), and the settle dynamics stay bounded/stable.
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    W_ff = rng.randn(k, H) * 0.1
    M = np.zeros((k, k), dtype=np.float64)     # lateral (learned, anti-Hebbian, fixed point)
    order = np.arange(Nc)
    for ep in range(epochs):
        rng.shuffle(order)
        for i in order:
            x = Xn[i]
            ff = W_ff @ x
            # settle y = ff - M y (lateral inhibition); iterate to the analog fixed point (NOT a host inverse)
            y = np.zeros(k)
            for _ in range(settle_steps):
                y = 0.5 * y + 0.5 * (ff - M @ y)
                if nonneg:
                    y = np.maximum(y, 0.0)
            # feedforward Oja (self-normalizing): ΔW_ff ∝ y xᵀ − (y²) W_ff  (row-wise gain control)
            W_ff += lr_ff * (np.outer(y, x) - (y ** 2)[:, None] * W_ff)
            # anti-Hebbian lateral with the fixed point: ΔM ∝ y yᵀ − M (diag kept 0 -> only cross terms)
            dM = np.outer(y, y) - M
            np.fill_diagonal(dM, 0.0)
            M += lr_m * dM
    # read the settled codes
    codes = np.zeros((Nc, k))
    for i in range(Nc):
        ff = W_ff @ Xn[i]
        y = np.zeros(k)
        for _ in range(settle_steps):
            y = 0.5 * y + 0.5 * (ff - M @ y)
            if nonneg:
                y = np.maximum(y, 0.0)
        codes[i] = y
    return codes, W_ff


def _w_faithfulness(W_ff, X, S_true):
    """A proxy for the diagnosis's Pearson(W, counts): does the learned feedforward map preserve the input
    structure? Pearson( cos(W_ff @ X rows), S_true ) -- but more directly, the saturation signature is the
    code off-diagonal cosine (saturated -> ~uniform high)."""
    codes = (W_ff @ X.T).T
    return _pearson_vs_Strue(_cos_sim(codes), S_true)


def run_arm(name, X, S_true, labels, args, seed, simmatch):
    if simmatch:
        codes, W_ff = learn_simmatch(X, args.k, args.epochs, args.lr_ff, args.lr_m,
                                     args.settle_steps, seed, nonneg=args.nonneg)
    else:
        codes, W_ff = learn_saturating_hebbian(X, args.k, args.epochs, args.lr_sat, seed)
    pear = _pearson_vs_Strue(_cos_sim(codes), S_true)
    gen, chance = heldout_generalization(codes, labels)
    off = float(_cos_sim(codes)[np.triu_indices(codes.shape[0], 1)].mean())
    rank = effective_rank(codes)
    print(f"    [{name:18s}] Pearson(S,S_true)={pear:+.3f}  gen={gen:.3f} (chance {chance:.3f})  "
          f"offdiag-cos={off:+.3f}  eff-rank={rank:.1f}", flush=True)
    return {"pearson": pear, "gen": gen, "chance": chance, "offdiag": off, "eff_rank": rank, "_codes": codes}


def run_seed(seed, args):
    print(f"\n{'='*92}\n  LOG + SIMILARITY-MATCHING de-risk (seed {seed})\n{'='*92}", flush=True)
    if args.real_corpus:
        from research.runners.option_c_stageB_fair_test import build_context_inclusive_cooccurrence
        from research.runners.option_c_real_cooccurrence_derisk import TAXONOMY_8x8, taxonomy_to_vocab_categories
        from research.runners.learned_graded_embedding_diagnose import raw_count_matrix
        vocab, cat_ids, _ = taxonomy_to_vocab_categories(TAXONOMY_8x8)
        corpus = build_context_inclusive_cooccurrence(
            os.path.join(_REPO, "data", "corpus", "tinystories.txt"), vocab, cat_ids,
            window=2, n_context_hubs=args.n_hub, repeat_cap=40, seed=seed, verbose=False)
        concepts = corpus["concepts"]; Nt = len(vocab)
        C = raw_count_matrix(concepts, corpus["facts"])[:Nt, Nt:]   # targets x hubs
        labels = np.asarray(cat_ids); S_true = corpus["S_true"]
    else:
        C, labels, S_true, _ = build_concept_hub_counts(
            args.n_cat, args.per_cat, args.n_common, args.n_sig_per_cat,
            args.lam_common, args.lam_sig, args.lam_bg, seed)
    Nc, H = C.shape
    host_sim = ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(args.host_svd, min(C.shape) - 1), alpha=args.host_alpha)
    host_p, _, _, _ = score(host_sim, labels)
    print(f"  {Nc} concepts x {H} hubs; host PPMI+SVD ceiling={host_p:+.3f}; embedding k={args.k}", flush=True)

    X_raw = encode_inputs(C, log_compress=False, marginal_norm=args.marginal_norm)
    X_log = encode_inputs(C, log_compress=True, marginal_norm=args.marginal_norm)

    print("  [4-arm ablation: {raw,log} x {saturating-Hebbian, similarity-matching}]", flush=True)
    A = run_arm("A raw+saturating", X_raw, S_true, labels, args, seed, simmatch=False)
    B = run_arm("B log+saturating", X_log, S_true, labels, args, seed, simmatch=False)
    Cc = run_arm("C raw+simmatch", X_raw, S_true, labels, args, seed, simmatch=True)
    T = run_arm("TEST log+simmatch", X_log, S_true, labels, args, seed, simmatch=True)

    # anti-cheats on THE TEST arm
    rng = np.random.RandomState(seed * 2718281 + 1)
    perm = rng.permutation(labels)
    S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
    test_perm = _pearson_vs_Strue(_cos_sim(T["_codes"]), S_perm)
    print(f"  [anti-cheat] TEST permuted-similarity Pearson={test_perm:+.3f} (~0)  "
          f"lesion(TEST->A i.e. log+SATURATING)={B['pearson']:+.3f} (the learning-rule lesion)", flush=True)

    chance = A["chance"]
    point_neuron_A_fails = abs(A["pearson"]) <= args.a_fail_bar
    host_carries = host_p >= args.host_bar
    test_structure = (T["pearson"] >= args.structure_bar) and point_neuron_A_fails and host_carries
    test_generalizes = (T["gen"] > chance + args.gen_margin) and (A["gen"] <= chance + 0.20)
    permuted_collapses = abs(test_perm) <= args.a_fail_bar
    # the learning-rule lesion: log+SATURATING (B) must be well below log+SIMMATCH (TEST)
    rule_lesion_collapses = B["pearson"] <= T["pearson"] - 0.10
    gates = {
        "test_structure_vs_A": bool(test_structure),
        "A_saturating_fails": bool(point_neuron_A_fails),
        "host_ceiling_carries": bool(host_carries),
        "test_generalizes": bool(test_generalizes),
        "permuted_collapses": bool(permuted_collapses),
        "learning_rule_lesion_collapses": bool(rule_lesion_collapses),
    }
    print(f"  [seed {seed} gates] {gates}", flush=True)
    return {"seed": seed, "n_concepts": Nc, "n_hub": H, "host_ceiling": host_p, "chance": chance,
            "A_raw_sat": {kk: vv for kk, vv in A.items() if kk != "_codes"},
            "B_log_sat": {kk: vv for kk, vv in B.items() if kk != "_codes"},
            "C_raw_sim": {kk: vv for kk, vv in Cc.items() if kk != "_codes"},
            "TEST_log_sim": {kk: vv for kk, vv in T.items() if kk != "_codes"},
            "test_permuted_pearson": test_perm, "gates": gates}


def decide_verdict(per_seed, seeds, args):
    def allg(k):
        return all(per_seed[str(s)]["gates"][k] for s in seeds)
    structure = allg("test_structure_vs_A"); a_fails = allg("A_saturating_fails")
    host_ok = allg("host_ceiling_carries"); generalize = allg("test_generalizes")
    controls = allg("permuted_collapses") and allg("learning_rule_lesion_collapses")
    tmean = float(np.mean([per_seed[str(s)]["TEST_log_sim"]["pearson"] for s in seeds]))
    amean = float(np.mean([per_seed[str(s)]["A_raw_sat"]["pearson"] for s in seeds]))
    bmean = float(np.mean([per_seed[str(s)]["B_log_sat"]["pearson"] for s in seeds]))
    cmean = float(np.mean([per_seed[str(s)]["C_raw_sim"]["pearson"] for s in seeds]))
    tgen = float(np.mean([per_seed[str(s)]["TEST_log_sim"]["gen"] for s in seeds]))
    if not host_ok:
        verdict, why = "NEGATIVE_miscalibrated", "host ceiling did not carry -> the data/encoding is the issue."
    elif not a_fails:
        verdict, why = "NEGATIVE_miscalibrated", (f"CONTROL A (raw+saturating) did NOT fail (mean {amean:+.3f}) "
                                                   f"-> the toy doesn't reproduce the Option-C failure; re-tune.")
    elif structure and generalize and controls:
        verdict = "GO"
        why = (f"LOG-input + SIMILARITY-MATCHING learn lets the POINT neuron LEARN the graded structure "
               f"(TEST mean Pearson {tmean:+.3f}, gen {tgen:.3f}) AND generalize, WHILE the project's raw+"
               f"saturating-Hebbian control fails (A {amean:+.3f}); controls clean (permuted + learning-rule "
               f"lesion collapse). ABLATION: A {amean:+.3f} (raw+sat) | B {bmean:+.3f} (log+sat) | C "
               f"{cmean:+.3f} (raw+sim) | TEST {tmean:+.3f}. ⇒ the dendritic D2 build is OFF the critical "
               f"path; escalate to the weeks-scale spiking similarity-matching build (Pehlevan 2019).")
    elif tmean > amean + 0.10 and controls:
        verdict = "BOUNDARY"
        why = (f"log+simmatch beats the saturating control (TEST {tmean:+.3f} vs A {amean:+.3f}) but does not "
               f"clear the structure/generalization bar -> the right input+rule but insufficient alone; the "
               f"dendritic L3/D2 marginal-normalization is the warranted next escalation (now with a sharp "
               f"target). ABLATION: A {amean:+.3f} | B {bmean:+.3f} | C {cmean:+.3f} | TEST {tmean:+.3f}.")
    else:
        verdict = "NEGATIVE"
        why = (f"even log+simmatch does not beat the saturating control on the point neuron (TEST {tmean:+.3f} "
               f"vs A {amean:+.3f}) -> the gap is deeper than input-encoding+learning-rule -> ship the flat "
               f"2,048-concept cortex (Option A) / the dendritic build for the artificial-life goal.")
    return verdict, why, {"A_mean": amean, "B_mean": bmean, "C_mean": cmean, "TEST_mean": tmean,
                          "TEST_gen": tgen, "structure_all": structure, "A_fails_all": a_fails,
                          "generalize_all": generalize, "controls_all": controls}


def main():
    p = argparse.ArgumentParser(description="Learned graded cortex: log + similarity-matching de-risk")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--real-corpus", action="store_true")
    p.add_argument("--marginal-norm", action="store_true",
                   help="also divide the input by the per-hub marginal (the PPMI-marginal / divisive arm)")
    # synthetic toy
    p.add_argument("--n-cat", type=int, default=8); p.add_argument("--per-cat", type=int, default=8)
    p.add_argument("--n-common", type=int, default=200); p.add_argument("--n-sig-per-cat", type=int, default=12)
    p.add_argument("--lam-common", type=float, default=40.0); p.add_argument("--lam-sig", type=float, default=4.0)
    p.add_argument("--lam-bg", type=float, default=0.3)
    p.add_argument("--n-hub", type=int, default=500, help="(real-corpus) number of context hubs")
    # embedding + learners
    p.add_argument("--k", type=int, default=64, help="embedding dimension")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr-ff", type=float, default=0.01); p.add_argument("--lr-m", type=float, default=0.01)
    p.add_argument("--settle-steps", type=int, default=20)
    p.add_argument("--lr-sat", type=float, default=0.01, help="saturating-Hebbian learn rate")
    p.add_argument("--nonneg", action="store_true", help="nonnegative similarity-matching (relu output)")
    p.add_argument("--host-svd", type=int, default=50); p.add_argument("--host-alpha", type=float, default=0.75)
    # gate bars
    p.add_argument("--structure-bar", type=float, default=0.30); p.add_argument("--a-fail-bar", type=float, default=0.15)
    p.add_argument("--host-bar", type=float, default=0.30); p.add_argument("--gen-margin", type=float, default=0.10)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.smoke:
        os.environ.setdefault("SIM_BACKEND", "numpy")
        args.epochs = 20

    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    print(f"[log+simmatch de-risk] seeds={seeds} smoke={args.smoke} real_corpus={args.real_corpus} "
          f"marginal_norm={args.marginal_norm} k={args.k}", flush=True)
    per_seed = {str(s): run_seed(s, args) for s in seeds}
    verdict, why, detail = decide_verdict(per_seed, seeds, args)
    print(f"\n{'='*92}\n  VERDICT: {verdict}\n  {why}", flush=True)
    print(f"  ABLATION means: A(raw+sat) {detail['A_mean']:+.3f} | B(log+sat) {detail['B_mean']:+.3f} | "
          f"C(raw+sim) {detail['C_mean']:+.3f} | TEST(log+sim) {detail['TEST_mean']:+.3f} (gen {detail['TEST_gen']:.3f})",
          flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n{'='*92}\n", flush=True)
    out = {"verdict": verdict, "why": why, "detail": detail, "seeds": seeds, "smoke": bool(args.smoke),
           "real_corpus": bool(args.real_corpus), "config": vars(args), "per_seed": per_seed,
           "note": ("Deep-research §(e) de-risk: log-compressed input + similarity-matching (Pehlevan-"
                    "Chklovskii fixed-point) learn vs the project's saturating-Hebbian control, on the "
                    "point neuron. GO ⇒ the dendritic D2 build is off the critical path. NO sim/ edits.")}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(raw_dir, f"_log_simmatch_{'smoke' if args.smoke else 'multiseed'}_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
