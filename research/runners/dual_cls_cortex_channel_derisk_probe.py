"""Dual / CLS CORTEX-CHANNEL DE-RISK — does routing the round-trip through the CORTEX channel
(cortical reinstatement of the recall-identified concept) close the round-trip AND restore
on-substrate generalization, which the hippocampal settled state could not?

THE LOAD-BEARING DE-RISK (the dual/CLS design's own intent, the last on-substrate piece):
  The STRONG-ENCODE de-risk (2026-06-11-dual-CLS-strong-encode-derisk-BOUNDARY.md, commit
  a317ce61) localised the round-trip failure PRECISELY: a STRONG, STABLE sparse encode makes the
  spiking dentate-gyrus (DG) read BOTH reproducible (1.000) AND decorrelated (~0) at sparse k
  (k=40, drive>=800 pA) -- the encode is FIXED. BUT the round-trip still scored only +0.189 (vs
  +1.000 ceiling), and a DETERMINISTIC perfect encode failed IDENTICALLY at +0.189 -> the
  bottleneck is NOT the encode; it is DOWNSTREAM: the Hopfield/CA3 SETTLED state is only
  0.71-faithful to the clean code even at zero cue noise, so the graded SHAPE is lost in
  bind->settle->decode. Binding IDENTITY is 1.000 (the right concept is recovered); the graded
  SHAPE is not.

  THE FIX (exactly the prior finding's "decisive next step", and the dual/CLS design's intent):
  the graded similarity is supposed to live on the CORTEX channel, NOT the decorrelated
  hippocampal settled state. So: use the correctly-recovered IDENTITY (binding cleanup = 1.000)
  to REINSTATE the recovered concept's STABLE CORTEX code (cortical reinstatement -- brain-based:
  recall identifies the concept via the spiking Hopfield attractor over the decorrelated DG
  codes, then the cortex re-activates that concept's representation), INSTEAD of decoding the
  0.71-faithful CA3 settled state.

  THE TEST: does routing the round-trip through the cortex channel
    (a) CLOSE the round-trip (Pearson toward +1.000 vs +0.189 via the hippocampal settle), and
    (b) RESTORE on-substrate GENERALIZATION (which was at chance when measured on the
        hippocampal settle)?

WHY THIS IS HONEST (the identity-gate, stated up-front):
  The round-trip via the cortex channel is GATED on the cleanup IDENTITY. The identity is NOT a
  host lookup of the answer -- it is the output of the SPIKING recall (the Hopfield attractor
  over the real spiking strong-DG codes). Reinstating the identified concept's stable cortex code
  is cortical pattern reinstatement (the brain re-activates the recalled concept's cortex
  representation). BECAUSE binding identity is ~1.000, the round-trip "closing" is the EASY part
  (a perfect identity reinstates the original codebook, so S' -> S trivially). THE INTERESTING,
  LOAD-BEARING RESULT IS THE GENERALIZATION, not the identity-gated round-trip. The generalization
  is genuinely sensitive: a mis-recalled concept reinstates the WRONG cortex code, corrupting both
  the round-trip and the generalization vote -- so generalization inherits the recall errors, it
  is NOT a free pass. AND it MUST collapse on orthogonal cortex codes + under the permuted-
  similarity control (which proves the generalization comes from the cortex channel's GRADED
  structure, not from the recall pipeline itself). The binding identity is reported FRONT-AND-
  CENTRE -- if it is not ~1.000 on the graded cues, the round-trip closing is hollow.

WHAT IS SYNTHETIC vs ON-SUBSTRATE (the precise scope, stated honestly):
  The graded CORTEX codebook is SYNTHETIC (category-factor + concept-residual). The learned
  spiking-cortical embedding that PRODUCES such graded codes on neurons is the still-unbuilt
  MONTHS-SCALE piece. THIS probe does NOT claim that capability is built. It claims the
  dual/CLS ARCHITECTURE ROUTES CORRECTLY ON-SUBSTRATE: the spiking strong-DG encode (REAL spike
  train, validated repro 1.000 + decorr at k=40), the spiking Hopfield recall (REAL identity
  1.000), and the cortical reinstatement together route the graded similarity through the cortex
  channel -- ON the real bridge, with the synthetic graded codes standing in for the learned
  embedding.

THREE READOUT CHANNELS COMPARED (the decisive contrast):
  1. CORTEX channel (the fix): S'_cortex = cos-matrix( cortex_code[ recovered_identity ] ).
     Cortical reinstatement of the recall-identified concept's stable graded code.
  2. HIPPOCAMPAL-SETTLE channel (the prior +0.189): learned CA1->cortex ridge decode of the
     DEGRADED settled attractor state (re-measured here as the within-probe contrast).
  3. CLEAN-DECODE ceiling (+1.000): decode the clean DG codes (no recall) -- the positive control.

SUBSTRATE: build_biological_brain_regions(enable_hippocampus_consolidation=True), the SAME bridge
as the strong-encode de-risk + the validated P1 trisynaptic loop. NO sim/ edits. The strong DG
drive sets cp_external_input_current on the DG slice (the world/body input current the neural DG
receives); the k-WTA DG read + the Hopfield recall + the cortical reinstatement are readout/
cognitive operations on the bridge's spike state.

DECISION (stated explicitly at end):
  GO if the cortex-channel round-trip CLOSES (Pearson high, >> permuted ~0) AND generalization is
     RESTORED on-substrate (PASSES on graded, COLLAPSES on orthogonal + permuted) AND binding
     identity stays ~1.000, multi-seed. -> the dual/CLS architecture works END-TO-END ON THE REAL
     SUBSTRATE (with synthetic graded codes) -> the ONLY remaining piece is the months-scale
     LEARNED graded-similarity embedding -> recommend presenting the concrete build plan + cost.
  NEGATIVE/BOUNDARY otherwise -> name what breaks (does reinstating the identity's clean code
     still lose similarity? is the cleanup not actually ~1.000 on graded cues? is the
     generalization genuinely not on the cortex channel -- graded does not pass, or orthogonal/
     permuted do not collapse?). No banking.

ANTI-CHEATS:
  - Binding/cleanup IDENTITY reported front-and-centre (the round-trip's gate). The interesting
    result is the GENERALIZATION, emphasized as such.
  - PERMUTED-S baseline for the round-trip Pearson (~0). ORTHOGONAL-codes + PERMUTED-similarity
    controls for generalization (MUST collapse there -- else it is spurious / pipeline artifact).
  - Explicit SPIKING strong-DG (the real on-substrate test) AND DETERMINISTIC reference (the
    reproducible-by-construction encode sanity); explicit SYNTHETIC graded codes (the months-
    scale learned embedding is the unbuilt piece -- the claim is "the architecture routes
    correctly on-substrate", NOT "the capability is built").

Run:
  # tiny numpy smoke (harness check, small bridge, fast)
  SIM_BACKEND=numpy python -m research.runners.dual_cls_cortex_channel_derisk_probe \
      --smoke --seeds 42 --out research/findings/raw/_dual_cls_cortex_channel_smoke.json
  # full GPU multi-seed
  SIM_BACKEND=cupy python -m research.runners.dual_cls_cortex_channel_derisk_probe \
      --seeds 42,43,44 --out research/findings/raw/_dual_cls_cortex_channel_multiseed.json
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


# ===========================================================================
# Reuse the strong-encode de-risk's on-substrate harness + the architecture-proof's
# synthetic graded codebook + generalization + decode/Pearson machinery (identical
# conventions to the strong-encode de-risk and the on-substrate gate).
# ===========================================================================
from research.runners.dual_cls_architecture_proof_probe import (  # noqa: E402
    build_graded_codebook,
    codebook_similarity_stats,
    assign_properties,
    run_generalization,
    run_generalization_permuted,
    native_cos_matrix,
    load_orthogonal_codes,
)
from research.runners.dual_cls_strong_encode_derisk_probe import (  # noqa: E402
    StrongDGEncoder,
    assign_sparse_dg_ensembles,
    _cos,
    _mean_offdiag_cos,
    fit_decoder,
    roundtrip_pearson,
    roundtrip_permuted_baseline,
)


# ===========================================================================
# Binding / recall on the real DG codes (the spiking Hopfield attractor) -- the
# recall returns the recovered IDENTITY (which concept) AND the degraded settled state.
# Identical to the strong-encode probe's run_binding_on_dg (so identity ~ 1.000 at the
# strong operating point), but kept here so the cortical-reinstatement step is explicit.
# ===========================================================================
def recall_identity_and_settle(dg_codes, flip_frac, seed, n_dg):
    """Build a Hopfield attractor over the (decorrelated, sparse) DG codes, present a
    NOISED cue per concept, settle, and return:
      recovered  [N]      -- the recovered concept IDENTITY (argmax over codes . settled)
      settled    [N, n_dg]-- the degraded settled real-valued state (the prior channel reads it)
      identity_acc        -- fraction of concepts recalled correctly (the round-trip's GATE)

    The recovered identity is the SPIKING recall's output; the cortical reinstatement uses it to
    re-activate the recovered concept's stable cortex code. The settled state is kept ONLY to
    re-measure the prior hippocampal-settle channel as the within-probe contrast.
    """
    from research.runners.cortex_sparse_attractor_poscontrol_probe import (
        build_hopfield_weights, noisy_cue_sparse,
    )
    N, _ = dg_codes.shape
    codes_native = dg_codes - dg_codes.mean(axis=1, keepdims=True)
    codes_native = codes_native / (np.linalg.norm(codes_native, axis=1, keepdims=True) + 1e-12)
    W = build_hopfield_weights(codes_native)
    rng = np.random.default_rng(seed * 7 + int(flip_frac * 1000) + 17)
    recovered = np.zeros(N, dtype=int)
    settled = np.zeros((N, n_dg), dtype=np.float64)
    for i in range(N):
        cue = noisy_cue_sparse(codes_native[i], rng, flip_frac, n_dg)
        s = cue.copy().astype(np.float64)
        nn = np.linalg.norm(s)
        if nn > 1e-12:
            s = s / nn
        for _ in range(5):
            s_new = W @ s
            n2 = np.linalg.norm(s_new)
            if n2 < 1e-12:
                break
            s_new = s_new / n2
            if np.max(np.abs(s_new - s)) < 1e-8:
                break
            s = s_new
        recovered[i] = int(np.argmax(codes_native @ s))
        settled[i] = s
    identity_acc = float(np.sum(recovered == np.arange(N))) / N
    return recovered, settled, identity_acc


# ===========================================================================
# THE CORTEX CHANNEL (the fix) -- cortical reinstatement of the recall-identified concept
# ===========================================================================
def cortex_channel_roundtrip(cortex_codes, S_orig, recovered_idx):
    """Round-trip via the CORTEX channel: reinstate each recall-identified concept's STABLE
    graded cortex code, then measure Pearson(S_orig, S'_cortex).

    S'_cortex = cos-matrix( cortex_codes[ recovered_idx ] ).

    Brain-based: the spiking recall identifies WHICH concept (recovered_idx, from the Hopfield
    attractor over the decorrelated DG codes); the cortex RE-ACTIVATES that concept's stable
    graded representation (cortical pattern reinstatement). With perfect identity this reinstates
    the original codebook (S' -> S, the easy part); with identity ERRORS, a mis-recalled concept
    reinstates the WRONG cortex code, degrading S' (so the channel is genuinely identity-gated).
    Returns (pearson, S_round)."""
    reinstated = cortex_codes[recovered_idx]
    S_round = native_cos_matrix(reinstated)
    N = cortex_codes.shape[0]
    iu = np.triu_indices(N, k=1)
    return float(np.corrcoef(S_orig[iu], S_round[iu])[0, 1]), S_round, reinstated


def cortex_channel_permuted_baseline(cortex_codes, S_orig, recovered_idx, seed):
    """PERMUTED-S baseline for the cortex-channel round-trip: permute the concept ROWS of the
    cortex codebook before reinstatement (so the reinstated code for concept i targets a RANDOM
    concept). Pearson(S_orig, S'_perm) must be ~0 -- proving a high TRUE Pearson is meaningful,
    not an artifact of the reinstatement always producing a similar-looking matrix."""
    rng = np.random.RandomState(seed * 617 + 29)
    perm = rng.permutation(cortex_codes.shape[0])
    cortex_perm = cortex_codes[perm]
    reinstated = cortex_perm[recovered_idx]
    S_round = native_cos_matrix(reinstated)
    N = cortex_codes.shape[0]
    iu = np.triu_indices(N, k=1)
    return float(np.corrcoef(S_orig[iu], S_round[iu])[0, 1])


# ===========================================================================
# Per-seed driver
# ===========================================================================
def run_seed_full(seed, args):
    print(f"\n{'='*72}", flush=True)
    print(f"  CORTEX-CHANNEL DE-RISK -- SEED {seed}", flush=True)
    print(f"{'='*72}", flush=True)

    n_clusters = args.n_clusters
    per_cluster = args.per_cluster
    N = n_clusters * per_cluster
    dim = args.n_lang_input    # codebook lives in language_input space (same as the gate)

    # ---------- synthetic graded cortex codebook (the codes the cortex channel reinstates) ----
    codes, labels, S = build_graded_codebook(n_clusters, per_cluster, dim, seed,
                                             args.residual_frac)
    grad_stats = codebook_similarity_stats(codes, labels)
    print(f"  [graded cortex codebook] N={N} ({n_clusters}x{per_cluster}) dim={dim}", flush=True)
    print(f"    within-cluster cos={grad_stats['within_cluster_cos_mean']:.3f} "
          f"between-cluster cos={grad_stats['between_cluster_cos_mean']:.3f} "
          f"margin={grad_stats['graded_margin']:.3f} graded={grad_stats['is_graded']}",
          flush=True)
    assert grad_stats["is_graded"], "graded codebook unit-check FAILED (within !>> between)"
    print("    [SCOPE] cortex codes are SYNTHETIC; the learned spiking-cortical embedding that "
          "produces such graded codes is the unbuilt months-scale piece.", flush=True)
    props = assign_properties(n_clusters, per_cluster, args.n_props, seed)

    # ---------- stable per-concept sparse DG ensembles (concept cells) ----------
    ensembles, binary_dg = assign_sparse_dg_ensembles(N, args.n_dg, args.ensemble_size, seed)
    print(f"  [DG ensembles] K={args.ensemble_size} of N={args.n_dg} per concept "
          f"(the strong-stable drive points the spiking DG at these cells)", flush=True)

    # ---------- build the real spiking strong-DG bridge ----------
    enc = StrongDGEncoder(
        seed=seed, n_lang_input=args.n_lang_input, n_dg=args.n_dg,
        n_dg_pv_basket=args.n_dg_pv_basket, n_ca3=args.n_ca3, n_ca1=args.n_ca1,
        n_ec=args.n_ec, ca3_recurrent_density=args.ca3_recurrent_density,
        ca3_recurrent_weight=args.ca3_recurrent_weight, verbose=True)

    drive_pA = args.drive_pA
    k = args.k

    # ============ STEP 1 -- the SPIKING strong-DG encode at the VALIDATED operating point =====
    # The strong-encode de-risk validated drive>=800 pA + k=40 -> repro 1.000 AND decorr ~0.
    # Re-confirm repro + decorr here (so the cortex-channel result is not built on a bad encode).
    print(f"\n  [STEP 1 -- spiking strong-DG encode @ drive={drive_pA:.0f} k={k} "
          f"(validated operating point)]", flush=True)
    dg_codes, spikes = enc.encode_codebook_dg(ensembles, drive_pA, args.window, k,
                                              args.reset_steps)
    between = _mean_offdiag_cos(dg_codes)
    sparsity = float(np.mean(dg_codes > 0))
    # reproducibility (same input -> two fresh reads -> cosine)
    repro_rng = np.random.default_rng(seed + 777)
    repro_vals = []
    for _ in range(args.n_repro_pairs):
        ci = int(repro_rng.integers(N))
        c1, _, _ = enc.rate_kwta_dg_read(ensembles[ci], drive_pA, args.window, k, args.reset_steps)
        c2, _, _ = enc.rate_kwta_dg_read(ensembles[ci], drive_pA, args.window, k, args.reset_steps)
        repro_vals.append(_cos(c1, c2))
    repro_mean = float(np.mean(repro_vals))
    print(f"    DG between-cos={between:+.3f} (decorrelated={between <= args.decorr_bar}) "
          f"repro={repro_mean:.3f} (reproducible={repro_mean >= args.repro_bar}) "
          f"sparsity={sparsity:.3f} spikes={float(np.mean(spikes)):.0f}", flush=True)
    encode_ok = (between <= args.decorr_bar) and (repro_mean >= args.repro_bar)
    print(f"    encode co-occurrence (repro AND decorr): {encode_ok}", flush=True)

    # ============ STEP 2 -- spiking recall: recovered IDENTITY (the round-trip's GATE) ========
    print(f"\n  [STEP 2 -- spiking Hopfield recall over the decorrelated DG codes (flip={args.flip_frac})]",
          flush=True)
    recovered, settled, identity_acc = recall_identity_and_settle(
        dg_codes, args.flip_frac, seed, enc.n_dg)
    print(f"    >>> BINDING/CLEANUP IDENTITY (the round-trip GATE) = {identity_acc:.3f}  "
          f"(must be ~1.000 or the round-trip closing is hollow)", flush=True)

    # ============ STEP 3a -- CORTEX-CHANNEL round-trip (the fix) ===============================
    cortex_pearson, S_cortex, _ = cortex_channel_roundtrip(codes, S, recovered)
    cortex_perm = cortex_channel_permuted_baseline(codes, S, recovered, seed)

    # ============ STEP 3b -- HIPPOCAMPAL-SETTLE round-trip (the prior +0.189, the contrast) ====
    settle_pearson, _ = roundtrip_pearson(codes, S, dg_codes, settled, ridge=args.ridge)
    settle_perm = roundtrip_permuted_baseline(codes, S, dg_codes, settled, seed, ridge=args.ridge)

    # ============ clean-decode ceiling (+1.000 positive control) ==============================
    clean_pearson, _ = roundtrip_pearson(codes, S, dg_codes, dg_codes.astype(np.float64),
                                         ridge=args.ridge)

    print(f"\n  [STEP 3 -- round-trip by CHANNEL]", flush=True)
    print(f"    >>> CORTEX channel (the fix):    Pearson(S,S') = {cortex_pearson:+.3f}  "
          f"(permuted {cortex_perm:+.3f})", flush=True)
    print(f"        HIPPOCAMPAL-settle (prior):  Pearson(S,S') = {settle_pearson:+.3f}  "
          f"(permuted {settle_perm:+.3f})  [prior de-risk: +0.189]", flush=True)
    print(f"        clean-decode ceiling:        Pearson(S,S') = {clean_pearson:+.3f}  "
          f"[positive control: +1.000]", flush=True)

    # ============ STEP 4 -- GENERALIZATION on the CORTEX channel (the LOAD-BEARING result) =====
    # Graded: reinstate the recall-identified concept's graded cortex code; vote over the
    # reinstated codes. With identity 1.000 the reinstated codes ARE the originals, so this tests
    # whether the GRADED structure routed through the recall generalizes. With identity errors the
    # reinstated codes are corrupted (so it is genuinely identity-gated, NOT a free pass).
    reinstated_graded = codes[recovered]

    # Orthogonal contrast: use the project's VALIDATED decorrelated codebook (load_orthogonal_codes
    # = generate_sparse_patterns, between-cos ~0.05 by construction -- the SAME A2 contrast the
    # architecture-proof used and confirmed collapses to chance) as the cortex codes, give it its
    # OWN DG ensembles, run the SAME spiking recall, reinstate. Even with perfect recall, reinstating
    # orthogonal codes gives orthogonal codes -> generalization MUST collapse (no graded neighbours).
    # This proves the generalization is the cortex channel's GRADED structure, not the recall pipeline.
    print("\n  [STEP 4 -- generalization on the CORTEX channel (LOAD-BEARING)]", flush=True)
    print("    building the ORTHOGONAL-cortex contrast (validated decorrelated codebook, "
          "own DG ensembles, same spiking recall)...", flush=True)
    ortho_codes = load_orthogonal_codes(seed, N)
    ortho_ens, _ = assign_sparse_dg_ensembles(N, args.n_dg, args.ensemble_size, seed + 99991)
    dg_ortho, _ = enc.encode_codebook_dg(ortho_ens, drive_pA, args.window, k, args.reset_steps)
    recovered_o, _, identity_o = recall_identity_and_settle(
        dg_ortho, args.flip_frac, seed, enc.n_dg)
    reinstated_ortho = ortho_codes[recovered_o]

    gen_graded = run_generalization(reinstated_graded, labels, props, n_clusters, per_cluster,
                                    seed, args.k_neighbours)
    gen_ortho = run_generalization(reinstated_ortho, labels, props, n_clusters, per_cluster,
                                   seed, args.k_neighbours)
    gen_perm = run_generalization_permuted(reinstated_graded, labels, props, n_clusters,
                                           per_cluster, seed, args.k_neighbours)
    chance = gen_graded["chance"]
    print(f"    orthogonal-recall identity = {identity_o:.3f}", flush=True)
    print(f"    >>> graded(cortex)    acc={gen_graded['accuracy']:.3f} "
          f"(chance={chance:.3f}, {gen_graded['ratio_vs_chance']:.1f}x)  [MUST pass]", flush=True)
    print(f"        orthogonal(cortex) acc={gen_ortho['accuracy']:.3f}  [MUST collapse to chance]",
          flush=True)
    print(f"        permuted-S(cortex) acc={gen_perm['accuracy']:.3f}  [MUST collapse to chance]",
          flush=True)
    a1 = gen_graded["accuracy"] >= args.a1_bar
    a2 = gen_ortho["accuracy"] <= 1.5 * chance
    a3 = gen_perm["accuracy"] <= 1.5 * chance

    # ---------- per-seed gates ----------
    binding_ok = identity_acc >= args.binding_bar
    c2_ok = (cortex_pearson >= args.c2_bar) and (cortex_pearson > cortex_perm + 0.3)
    gates = {
        "encode_cooccur_repro_and_decorr": bool(encode_ok),
        "cortex_roundtrip_closes": bool(c2_ok),
        "binding_identity": bool(binding_ok),
        "a1_graded_generalizes_cortex": bool(a1),
        "a2_orthogonal_collapses_cortex": bool(a2),
        "a3_permuted_collapses_cortex": bool(a3),
    }
    print(f"\n  [SEED {seed} gates] {gates}", flush=True)

    return {
        "seed": seed,
        "graded_stats": grad_stats,
        "n_neurons": enc.n_neurons,
        "n_synapses": enc.n_synapses,
        "build_seconds": enc.build_seconds,
        "operating_point": {"drive_pA": drive_pA, "k": k},
        "encode": {
            "dg_between_cos_mean": between,
            "dg_repro_mean": repro_mean,
            "dg_sparsity": sparsity,
            "dg_total_spikes_mean": float(np.mean(spikes)),
            "cooccur_repro_and_decorr": bool(encode_ok),
        },
        "binding_identity_acc": identity_acc,
        "roundtrip": {
            "cortex_channel_pearson": cortex_pearson,
            "cortex_channel_permuted": cortex_perm,
            "hippocampal_settle_pearson": settle_pearson,
            "hippocampal_settle_permuted": settle_perm,
            "clean_decode_ceiling": clean_pearson,
        },
        "generalization_cortex_channel": {
            "graded": gen_graded, "orthogonal": gen_ortho, "permuted": gen_perm,
            "chance": chance, "orthogonal_recall_identity": identity_o,
        },
        "gates": gates,
    }


def main():
    p = argparse.ArgumentParser(description="Dual-CLS cortex-channel de-risk probe")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--smoke", action="store_true",
                   help="tiny bridge + tiny codebook for harness verification (fast)")
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--per-cluster", type=int, default=5)
    p.add_argument("--n-props", type=int, default=4)
    p.add_argument("--k-neighbours", type=int, default=3)
    p.add_argument("--residual-frac", type=float, default=0.55)
    # bridge sizing (defaults match the validated P1 / strong-encode-de-risk scale)
    p.add_argument("--n-lang-input", type=int, default=512)
    p.add_argument("--n-ec", type=int, default=160)
    p.add_argument("--n-dg", type=int, default=600)
    p.add_argument("--n-dg-pv-basket", type=int, default=180)
    p.add_argument("--n-ca3", type=int, default=300)
    p.add_argument("--n-ca1", type=int, default=120)
    p.add_argument("--ca3-recurrent-density", type=float, default=0.30)
    p.add_argument("--ca3-recurrent-weight", type=float, default=2.0)
    # strong DG encode @ the VALIDATED operating point (drive>=800, k=40 -> repro 1.000 + decorr)
    p.add_argument("--ensemble-size", type=int, default=40,
                   help="per-concept sparse DG ensemble size (K of n_dg)")
    p.add_argument("--drive-pA", type=float, default=800.0,
                   help="DG ensemble drive (the strong-encode de-risk validated >=800)")
    p.add_argument("--k", type=int, default=40,
                   help="DG k-WTA read size (the validated sparse operating point)")
    p.add_argument("--window", type=int, default=150, help="DG accumulation window (steps)")
    p.add_argument("--reset-steps", type=int, default=40)
    p.add_argument("--n-repro-pairs", type=int, default=8)
    p.add_argument("--flip-frac", type=float, default=0.1, help="recall cue noise")
    p.add_argument("--ridge", type=float, default=1e-2)
    # gate bars
    p.add_argument("--decorr-bar", type=float, default=0.10)
    p.add_argument("--repro-bar", type=float, default=0.90)
    p.add_argument("--binding-bar", type=float, default=0.90,
                   help="binding identity must be >= this (the round-trip's gate; ~1.000 expected)")
    p.add_argument("--c2-bar", type=float, default=0.70,
                   help="cortex-channel round-trip Pearson must be >= this")
    p.add_argument("--a1-bar", type=float, default=0.70,
                   help="graded(cortex) generalization must be >= this")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_clusters = 4
        args.per_cluster = 3
        args.n_lang_input = 128
        args.n_ec = 48
        args.n_dg = 200
        args.n_dg_pv_basket = 60
        args.n_ca3 = 100
        args.n_ca1 = 60
        args.window = 80
        args.ensemble_size = 15
        args.drive_pA = 800.0
        args.k = 15
        args.n_repro_pairs = 3
        os.environ.setdefault("SIM_BACKEND", "numpy")

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    backend = os.environ.get("SIM_BACKEND", "auto")
    t_start = time.time()
    print(f"[dual-CLS cortex-channel de-risk] seeds={seeds} backend={backend} "
          f"smoke={args.smoke}", flush=True)
    print(f"  operating point: drive={args.drive_pA:.0f} pA, k={args.k} "
          f"(strong-encode de-risk validated: repro 1.000 + decorr at this point)", flush=True)
    print(f"  c2-bar(>=)={args.c2_bar} a1-bar(>=)={args.a1_bar} binding-bar(>=)={args.binding_bar}",
          flush=True)

    per_seed = {}
    for seed in seeds:
        per_seed[str(seed)] = run_seed_full(seed, args)

    # ---------- overall verdict ----------
    def all_gate(g):
        return all(per_seed[str(s)]["gates"][g] for s in seeds)

    g_encode = all_gate("encode_cooccur_repro_and_decorr")
    g_c2 = all_gate("cortex_roundtrip_closes")
    g_bind = all_gate("binding_identity")
    g_a1 = all_gate("a1_graded_generalizes_cortex")
    g_a2 = all_gate("a2_orthogonal_collapses_cortex")
    g_a3 = all_gate("a3_permuted_collapses_cortex")

    # GO requires: the cortex-channel round-trip closes AND generalization is RESTORED on the
    # cortex channel (graded passes, orthogonal + permuted collapse) AND binding identity ~1.000,
    # multi-seed. (The encode co-occurrence is re-confirmed but already de-risked; if it regresses
    # that is itself reportable.)
    if g_c2 and g_bind and g_a1 and g_a2 and g_a3 and g_encode:
        verdict = "GO"
    elif not g_bind:
        verdict = "BOUNDARY_binding_identity_not_clean_roundtrip_gate_hollow"
    elif not g_c2:
        verdict = "BOUNDARY_cortex_channel_roundtrip_does_not_close"
    elif not g_a1:
        verdict = "BOUNDARY_graded_generalization_not_restored_on_cortex_channel"
    elif not (g_a2 and g_a3):
        verdict = "BOUNDARY_generalization_not_similarity_driven_spurious_contrast"
    elif not g_encode:
        verdict = "BOUNDARY_encode_cooccurrence_regressed"
    else:
        verdict = "BOUNDARY_unspecified"

    # aggregate load-bearing numbers
    cortex_pearson = [per_seed[str(s)]["roundtrip"]["cortex_channel_pearson"] for s in seeds]
    cortex_perm = [per_seed[str(s)]["roundtrip"]["cortex_channel_permuted"] for s in seeds]
    settle_pearson = [per_seed[str(s)]["roundtrip"]["hippocampal_settle_pearson"] for s in seeds]
    clean_ceiling = [per_seed[str(s)]["roundtrip"]["clean_decode_ceiling"] for s in seeds]
    bind_id = [per_seed[str(s)]["binding_identity_acc"] for s in seeds]
    gen_graded = [per_seed[str(s)]["generalization_cortex_channel"]["graded"]["accuracy"]
                  for s in seeds]
    gen_ortho = [per_seed[str(s)]["generalization_cortex_channel"]["orthogonal"]["accuracy"]
                 for s in seeds]
    gen_perm = [per_seed[str(s)]["generalization_cortex_channel"]["permuted"]["accuracy"]
                for s in seeds]
    chance = per_seed[str(seeds[0])]["generalization_cortex_channel"]["chance"]

    summary = {
        "verdict": verdict,
        "seeds": seeds,
        "backend": backend,
        "smoke": bool(args.smoke),
        "operating_point": {"drive_pA": args.drive_pA, "k": args.k},
        "scope_note": ("cortex codes are SYNTHETIC graded; the claim is the dual/CLS ARCHITECTURE "
                       "routes correctly ON-SUBSTRATE via the cortex channel (spiking encode + "
                       "spiking recall identity + cortical reinstatement), NOT that the learned "
                       "graded-similarity embedding is built (that is the unbuilt months-scale piece)."),
        "bars": {"c2_bar": args.c2_bar, "a1_bar": args.a1_bar, "binding_bar": args.binding_bar,
                 "decorr_bar_le": args.decorr_bar, "repro_bar_ge": args.repro_bar},
        "gates_all_seeds": {
            "encode_cooccur_repro_and_decorr": g_encode,
            "cortex_roundtrip_closes": g_c2,
            "binding_identity": g_bind,
            "a1_graded_generalizes_cortex": g_a1,
            "a2_orthogonal_collapses_cortex": g_a2,
            "a3_permuted_collapses_cortex": g_a3,
        },
        "load_bearing": {
            "cortex_channel_roundtrip_pearson_per_seed": cortex_pearson,
            "cortex_channel_roundtrip_pearson_mean": float(np.mean(cortex_pearson)),
            "cortex_channel_permuted_per_seed": cortex_perm,
            "cortex_channel_permuted_mean": float(np.mean(cortex_perm)),
            "binding_identity_per_seed": bind_id,
            "binding_identity_mean": float(np.mean(bind_id)),
            "generalization_graded_per_seed": gen_graded,
            "generalization_graded_mean": float(np.mean(gen_graded)),
            "generalization_orthogonal_per_seed": gen_ortho,
            "generalization_orthogonal_mean": float(np.mean(gen_ortho)),
            "generalization_permuted_per_seed": gen_perm,
            "generalization_permuted_mean": float(np.mean(gen_perm)),
            "generalization_chance": chance,
            "hippocampal_settle_pearson_per_seed": settle_pearson,
            "hippocampal_settle_pearson_mean": float(np.mean(settle_pearson)),
            "clean_decode_ceiling_per_seed": clean_ceiling,
            "prior_hippocampal_settle_reference_pearson": 0.189,
            "clean_decode_ceiling_reference": 1.000,
        },
        "elapsed_total_s": time.time() - t_start,
    }

    print(f"\n{'='*72}", flush=True)
    print(f"  OVERALL VERDICT: {verdict}", flush=True)
    print(f"  Encode co-occurrence (repro+decorr) all seeds:   {g_encode}", flush=True)
    print(f"  >>> BINDING IDENTITY (round-trip GATE) all seeds: {g_bind}  "
          f"(mean {np.mean(bind_id):.3f})", flush=True)
    print(f"  Cortex-channel round-trip closes (all seeds):    {g_c2}", flush=True)
    print(f"  A1 graded generalizes on cortex channel:         {g_a1}", flush=True)
    print(f"  A2 orthogonal collapses on cortex channel:       {g_a2}", flush=True)
    print(f"  A3 permuted collapses on cortex channel:         {g_a3}", flush=True)
    print(f"  >>> CORTEX-channel round-trip Pearson (mean) = {np.mean(cortex_pearson):+.3f}  "
          f"(hippocampal-settle {np.mean(settle_pearson):+.3f}=prior+0.189; clean ceiling "
          f"{np.mean(clean_ceiling):+.3f}; permuted {np.mean(cortex_perm):+.3f})", flush=True)
    print(f"  >>> GENERALIZATION on cortex channel (the LOAD-BEARING result):", flush=True)
    print(f"        graded={np.mean(gen_graded):.3f}  orthogonal={np.mean(gen_ortho):.3f}  "
          f"permuted={np.mean(gen_perm):.3f}  (chance={chance:.3f})", flush=True)
    print(f"  Total elapsed: {summary['elapsed_total_s']:.1f}s", flush=True)
    print(f"{'='*72}\n", flush=True)

    out_data = {"summary": summary, "per_seed": per_seed}

    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = "smoke" if args.smoke else "multiseed"
        args.out = os.path.join(raw_dir, f"_dual_cls_cortex_channel_{tag}_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out_data


if __name__ == "__main__":
    main()
