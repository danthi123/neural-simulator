"""LEARNED GRADED-EMBEDDING CONFIRMATION -- the FULL-gate, multi-seed confirmation of the RECOVERED
recipe, PLUS the test of whether a BRAIN-BASED read-out can replace the host stand-in.

CONTEXT (the GO this confirms):
  The brain-based learned-embedding RECOVERED at the de-saturated recipe
  (2026-06-11-learned-graded-embedding-desaturate-GO.md, commit 1febdd20):
    * cycles=2 (the faithful low-cycle regime -- the un-normalized Hebbian recurrent has potentiated
      just enough to encode the co-occurrence RANK without saturating toward a uniform floor), AND
    * read-out = host-method (PPMI+SVD) on the FULL hub-inclusive learned W (the hub columns carry the
      2nd-order cat~dog signal; the diagnosis's member-submatrix read-out discarded them)
  -> Pearson(sim_W, S_true) +0.843 / +0.879 (ceiling +0.932 / +0.950), generalization 1.000, every
  control collapsing, 2/2 seeds (42/43). Option A (a homeostatically-regulated brain-based Hebbian
  learn recovers the embedding) is ALIVE.

  But the desaturate fix-test did NOT do TWO things, which THIS confirmation closes:
   (a) it did NOT re-run G3 (the cortex-channel round-trip) + G4 (the spiking strong-encode
       compatibility) on the LEARNED graded codes -- the two ARCHITECTURE gates (G3/G4 already
       passed on SYNTHETIC graded codes in the architecture-proof; this confirms them on the
       brain-LEARNED graded codes).
   (b) the recovering read-out was the HOST method (PPMI+SVD) on the brain-learned W -- a labelled
       STAND-IN. The genuinely BRAIN-BASED read-out (spreading-activation / diffusion through the
       hub nodes) the desaturate fix shipped UNDER-PROPAGATED (alpha 0.5, 2 steps, member-column
       restriction -> didn't reach the hubs). THIS confirmation tests whether a sufficiently-
       propagating diffusion closes the stand-in residual.

THE EXPERIMENT (multi-seed 42/43/44; GPU; FOREGROUND):
  1. FULL gate suite at the recovered recipe (the load-bearing confirmation): learn at cycles=2,
     read out on the FULL hub-inclusive W (host-method stand-in), feed the GRADED codes into the
     FULL de-risk gates -- G1 structure recovery, G2 generalization (+ orthogonal A2 / permuted-
     property A3 controls + permuted-co-occurrence G5 + beats-random), G3 cortex-channel round-trip,
     G4 spiking strong-encode (RUN it) -- multi-seed. Does the COMPLETE architecture pass END-TO-END
     with the brain-based de-saturated learn?
  2. BRAIN-BASED read-out (close the stand-in residual): on the SAME learned W, sweep a tuned
     spreading-activation / diffusion that PROPAGATES THROUGH THE HUBS -- diffusion steps in
     {2,3,4,6} x alpha in {0.5,0.7,0.9} x {member-column, FULL-column (hubs included)} variants.
     Does a sufficiently-propagating diffusion (brain-based: activation spreads through the semantic
     graph incl. hubs) recover Pearson(sim, S_true) + generalization toward the host-method-on-W
     (+0.84) and the ceiling (+0.93)? If yes -> the read-out is FULLY brain-based (residual closed);
     if it plateaus below -> the host-method-on-W stays a documented stand-in + the brain-based
     diffusion is a build-time refinement.

ANTI-CHEATS (all mandatory):
  - G2/G3/G4 with their EXISTING controls (orthogonal + permuted-property collapse; G3 permuted-S
    round-trip baseline; G5 permuted-co-occurrence collapse; G4 repro + decorr). The host PPMI+SVD
    on the RAW counts is the labelled CEILING (NOT the deliverable). The read-out (host-on-W OR
    diffusion) runs on the brain-LEARNED W: assert Pearson(W, raw_counts) < 0.999 so a "recovery"
    is NOT silently re-deriving the host ceiling.
  - multi-seed 42/43/44.

DECISION (stated explicitly):
  GO_full   if all gates G1-G4 pass at cycles=2 multi-seed with the host-method-on-W read-out.
            -> the dual/CLS learned-embedding works END-TO-END on the brain-based path -> the
            months-scale build is justified; report whether the BRAIN-BASED diffusion read-out
            ALSO closes (fully brain-based) or stays a stand-in (read-out = a build refinement).
  BOUNDARY  if G3/G4 fail on the learned graded codes (unlikely) -> characterize. No banking.

Run (GPU, FOREGROUND -- each cycles=2 spiking-learn is ~20-60 s inline; the G4 spiking encode adds a
bit; NO background):
  SIM_BACKEND=cupy python -m research.runners.learned_graded_embedding_confirm_probe \
      --seeds 42,43,44 --run-g4 --out research/findings/raw/_lge_confirm_multiseed.json
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

# Reuse the de-risk's corpus + read-out + structure-recovery + host-ceiling + the FULL gate harnesses
# VERBATIM; the desaturate probe's de-saturated low-cycle learn + the host-method-on-full-W read-out;
# and the diagnose's helpers. This runner only adds the BRAIN-BASED diffusion read-out sweep + drives
# the full G1-G4 battery at cycles=2.
from research.runners.learned_graded_embedding_derisk_probe import (  # noqa: E402
    build_toy_cooccurrence,
    permute_corpus,
    graded_readout,
    structure_recovery,
    host_ceiling_codes,
    random_gaussian_codes,
    architecture_generalization,
    cortex_channel_gate,
    strong_encode_g4,
)
from research.runners.learned_graded_embedding_diagnose import (  # noqa: E402
    raw_count_matrix,
    offdiag_pearson,
    _normalize_codes,
)
from research.runners.learned_graded_embedding_desaturate_probe import (  # noqa: E402
    learn_W_desaturate,
    host_method_codes_on_W,
)
from research.runners.dual_cls_architecture_proof_probe import (  # noqa: E402
    codebook_similarity_stats,
    assign_properties,
    run_generalization,
)


# ===========================================================================
# BRAIN-BASED read-out: a tuned spreading-activation / diffusion that PROPAGATES THROUGH THE HUBS.
# The desaturate fix's `graded_readout` diffuses over the FULL symmetrized W but then RESTRICTS the
# code to MEMBER columns -- which, combined with too-few steps, under-propagated the hub-mediated
# second-order signal. We sweep (steps, alpha) AND a FULL-COLUMN variant (read each member's diffused
# row over ALL columns incl. the hubs, where the shared-neighbour signal lives) -- brain-based:
# activation spreads through the semantic graph including the hub nodes.
# ===========================================================================
def diffusion_readout_full_columns(W: np.ndarray, member_rows: np.ndarray,
                                   alpha: float, steps: int) -> np.ndarray:
    """Spreading-activation diffusion over the FULL symmetrized W, then each MEMBER concept's code =
    its diffused association profile over ALL columns (HUBS INCLUDED). The hub columns carry the
    cat~dog second-order signal (cat and dog both diffuse activation onto the shared 'animal' hub),
    so reading member rows over the full column space keeps that signal -- the brain-based analogue
    of the host-method's PPMI+SVD over the full hub-inclusive W."""
    Ws = 0.5 * (W + W.T)
    np.fill_diagonal(Ws, 0.0)
    rs = Ws.sum(axis=1, keepdims=True)
    Wn = Ws / (rs + 1e-12)
    cur = Ws.copy()
    Wd = Ws.copy()
    for _ in range(max(0, steps)):
        cur = (1.0 - alpha) * cur + alpha * (cur @ Wn)
        Wd = cur
    codes = Wd[member_rows, :].astype(np.float64)   # FULL columns (hubs included)
    return _normalize_codes(codes)


def measure_codes(codes, S_true, second_order_pairs, labels, props, nclu, pclu, seed,
                  k_neighbours, chance):
    """Light measurement of a code matrix: Pearson(sim, S_true) + permuted-S baseline + graded? +
    second-order margin + generalization (graded acc)."""
    rec = structure_recovery(codes, S_true, second_order_pairs, seed)
    grad = codebook_similarity_stats(codes, labels)
    gen = float(run_generalization(codes, labels, props, nclu, pclu, seed, k_neighbours)["accuracy"])
    return {
        "pearson_vs_Strue": rec["pearson_learned_vs_Strue"],
        "pearson_permutedS": rec["pearson_permuted_vs_Strue"],
        "is_graded": bool(grad["is_graded"]),
        "second_order_margin": rec["second_order_margin"],
        "second_order_recovered": bool(rec["second_order_recovered"]),
        "generalization": gen,
        "generalization_ratio_vs_chance": gen / chance if chance > 0 else 0.0,
    }


# ===========================================================================
# Per-seed driver
# ===========================================================================
def run_seed(seed: int, args) -> dict:
    print(f"\n{'='*82}", flush=True)
    print(f"  LEARNED GRADED-EMBEDDING CONFIRMATION -- SEED {seed} (cycles={args.cycles})", flush=True)
    print(f"{'='*82}", flush=True)

    nclu, pclu = args.n_clusters, args.per_cluster

    # ----- corpus (reuse the de-risk's; KNOWN S_true + second-order cat~dog pairs) -----
    corpus = build_toy_cooccurrence(nclu, pclu, seed,
                                    hub_facts_per_member=args.hub_facts_per_member,
                                    bridge_facts=args.bridge_facts,
                                    triplet_facts_per_cluster=args.triplet_facts_per_cluster)
    concepts = corpus["concepts"]
    members = corpus["members"]
    labels = corpus["labels"]
    S_true = corpus["S_true"]
    second_order_pairs = corpus["second_order_pairs"]
    member_rows = np.asarray([concepts.index(m) for m in members], dtype=int)
    Nm = len(members)
    props = assign_properties(nclu, pclu, args.n_props, seed)
    chance = 1.0 / args.n_props
    C_full = raw_count_matrix(concepts, corpus["facts"])

    print(f"  [corpus] {len(concepts)} concepts ({nclu} hubs + {Nm} members), "
          f"{corpus['n_facts']} facts; second-order pairs={len(second_order_pairs)}; "
          f"chance={chance:.3f}", flush=True)

    # ----- labelled HOST CEILING (PPMI+SVD on RAW counts) -- the reachable target -----
    host_codes = host_ceiling_codes(concepts, corpus["facts"], member_rows, Nm, seed)
    host_rec = structure_recovery(host_codes, S_true, second_order_pairs, seed)
    host_gen = float(run_generalization(host_codes, labels, props, nclu, pclu, seed,
                                        args.k_neighbours)["accuracy"])
    host_graded = codebook_similarity_stats(host_codes, labels)
    print(f"  [HOST CEILING (PPMI+SVD on RAW counts, labelled target)] "
          f"Pearson(S,S_true)={host_rec['pearson_learned_vs_Strue']:+.3f} gen={host_gen:.3f} "
          f"graded={host_graded['is_graded']}", flush=True)

    # =========================================================================
    # STEP 1 -- LEARN at the recovered recipe (cycles=2, the de-saturated/faithful regime).
    # =========================================================================
    print(f"\n  [LEARN @ cycles={args.cycles} (de-saturated, brain-based spiking-Hebbian)]", flush=True)
    t_learn = time.time()
    W, learn_info = learn_W_desaturate(concepts, corpus["facts"], seed, args.n_pool,
                                       args.pattern_size, args.cycles, gamma=1.0, cap=None)
    learn_s = time.time() - t_learn
    print(f"    learned recurrent: mean={learn_info['recurrent_mean']:.3f} "
          f"max={learn_info['recurrent_max']:.3f} nnz={learn_info['recurrent_nnz']} "
          f"({learn_info['n_neurons']} neurons, {learn_s:.1f}s)", flush=True)

    # faithfulness anti-cheat: the learned W must track the FULL counts AND be distinct from them.
    pearson_W_counts = offdiag_pearson(W, C_full)
    W_distinct = (pearson_W_counts < 0.999)
    print(f"    [anti-cheat] Pearson(W, raw_counts_full)={pearson_W_counts:+.3f} "
          f"(tracks counts >> +0.06; distinct from counts <0.999: {W_distinct})", flush=True)

    # =========================================================================
    # STEP 2 -- PRIMARY read-out = host-method (PPMI+SVD) on the FULL learned W -> FULL gates G1-G4.
    # =========================================================================
    print(f"\n  {'-'*78}", flush=True)
    print(f"  PRIMARY read-out = host-method (PPMI+SVD) on the FULL hub-inclusive learned W "
          f"(the recovered stand-in)", flush=True)
    print(f"  {'-'*78}", flush=True)
    primary_codes = host_method_codes_on_W(W, member_rows, args.svd_dim)

    # G1 structure recovery (+ permuted-S baseline + second-order margin).
    g1_rec = structure_recovery(primary_codes, S_true, second_order_pairs, seed)
    g1_grad = codebook_similarity_stats(primary_codes, labels)
    g1_ok = (g1_rec["pearson_learned_vs_Strue"] >= args.g1_bar and g1_grad["is_graded"]
             and g1_rec["second_order_recovered"])
    print(f"    G1 structure: Pearson(sim,S_true)={g1_rec['pearson_learned_vs_Strue']:+.3f} "
          f"(permS {g1_rec['pearson_permuted_vs_Strue']:+.3f}) graded={g1_grad['is_graded']} "
          f"2nd-order-margin={g1_rec['second_order_margin']:+.3f} "
          f"recovered={g1_rec['second_order_recovered']} -> G1={g1_ok}", flush=True)

    # G2 generalization (A1 + orthogonal A2 + permuted-property A3), reused VERBATIM.
    gen = architecture_generalization(primary_codes, labels, props, nclu, pclu, seed,
                                      args.k_neighbours, args.a1_bar)
    print(f"    G2 generalization: graded={gen['graded']['accuracy']:.3f} "
          f"(chance {gen['chance']:.3f}, {gen['graded']['ratio_vs_chance']:.1f}x) A1={gen['a1']} | "
          f"orthogonal={gen['orthogonal']['accuracy']:.3f} A2={gen['a2']} | "
          f"permuted-prop={gen['permuted']['accuracy']:.3f} A3={gen['a3']}", flush=True)

    # G3 cortex-channel round-trip (numpy spiking-Hopfield recall -> reinstate the LEARNED graded
    # cortex code -> Pearson(S_orig, S') closes >> permuted), reused VERBATIM.
    cc = cortex_channel_gate(primary_codes, seed, nclu, pclu, args.c2_bar, args.flip_frac,
                             n_dg=args.cc_n_dg, ensemble_size=args.cc_ensemble_size)
    print(f"    G3 cortex-channel: identity={cc['binding_identity_acc']:.3f} "
          f"Pearson={cc['cortex_channel_pearson']:+.3f} (permuted {cc['cortex_channel_permuted']:+.3f}) "
          f"closes={cc['cortex_roundtrip_closes']}", flush=True)

    # G4 strong-encode compatibility (spiking StrongDGEncoder; the strong-vs-graded tension), VERBATIM.
    g4 = None
    if args.run_g4:
        print("    [G4 -- spiking StrongDGEncoder strong-encode compatibility (driving the LEARNED "
              "graded cortex codes' DG ensembles)]", flush=True)
        g4 = strong_encode_g4(primary_codes, seed, args)
        print(f"    G4: DG between-cos={g4['dg_between_cos_mean']:+.3f} (decorr={g4['decorrelated']}) "
              f"repro={g4['dg_repro_mean']:.3f} (repro_ok={g4['reproducible']}) -> "
              f"graded-cortex+decorrelated-DG COEXIST={g4['g4_graded_cortex_decorr_dg_coexist']}",
              flush=True)

    # PERMUTED-CO-OCCURRENCE control (HEADLINE): re-learn on a scrambled corpus at cycles, read out
    # identically -> the recovery must collapse.
    perm_facts = permute_corpus(corpus["facts"], concepts, seed)
    Wp, _ = learn_W_desaturate(concepts, perm_facts, seed, args.n_pool, args.pattern_size,
                               args.cycles, gamma=1.0, cap=None)
    perm_codes = host_method_codes_on_W(Wp, member_rows, args.svd_dim)
    perm_rec = structure_recovery(perm_codes, S_true, second_order_pairs, seed)
    perm_grad = codebook_similarity_stats(perm_codes, labels)
    perm_gen = float(run_generalization(perm_codes, labels, props, nclu, pclu, seed,
                                        args.k_neighbours)["accuracy"])
    g5_permco = (abs(perm_rec["pearson_learned_vs_Strue"]) < args.g1_bar * 0.6
                 and not perm_grad["is_graded"]
                 and perm_gen <= 1.5 * chance)

    # beats-random
    rand_codes = random_gaussian_codes(Nm, Nm, seed)
    rand_gen = float(run_generalization(rand_codes, labels, props, nclu, pclu, seed,
                                        args.k_neighbours)["accuracy"])
    beats_random = gen["graded"]["accuracy"] > rand_gen + 1e-9
    print(f"    G5 permuted-CO-OCCURRENCE collapses (HEADLINE): {g5_permco} "
          f"(Pearson {perm_rec['pearson_learned_vs_Strue']:+.3f}, gen {perm_gen:.3f}) | "
          f"beats-random={beats_random} (learned {gen['graded']['accuracy']:.3f} > "
          f"rand {rand_gen:.3f})", flush=True)

    primary_gates = {
        "g1_structure_recovered": bool(g1_ok),
        "g2_a1_generalizes": bool(gen["a1"]),
        "g2_a2_orthogonal_collapses": bool(gen["a2"]),
        "g2_a3_permuted_property_collapses": bool(gen["a3"]),
        "g3_cortex_roundtrip_closes": bool(cc["cortex_roundtrip_closes"]),
        "g3_binding_identity_clean": bool(cc["binding_identity_acc"] >= args.binding_bar),
        "g5_permuted_cooccurrence_collapses": bool(g5_permco),
        "g5_beats_random_baseline": bool(beats_random),
        "anti_cheat_W_distinct_from_counts": bool(W_distinct),
    }
    if g4 is not None:
        primary_gates["g4_strong_encode_compatible"] = bool(g4["g4_graded_cortex_decorr_dg_coexist"])
    print(f"\n  [SEED {seed} PRIMARY gates] {primary_gates}", flush=True)

    # =========================================================================
    # STEP 3 -- BRAIN-BASED read-out sweep (diffusion through the hubs) on the SAME learned W.
    # =========================================================================
    print(f"\n  {'-'*78}", flush=True)
    print(f"  BRAIN-BASED read-out sweep: spreading-activation diffusion through the hubs on the "
          f"LEARNED W", flush=True)
    print(f"  (target: host-method-on-W +{g1_rec['pearson_learned_vs_Strue']:.2f}, "
          f"ceiling +{host_rec['pearson_learned_vs_Strue']:.2f})", flush=True)
    print(f"  {'-'*78}", flush=True)
    diffusion_sweep = {}
    best_diff_key, best_diff_pearson = None, -2.0
    for steps in args.diffusion_steps_sweep:
        for alpha in args.diffusion_alpha_sweep:
            # variant (i): member-column (the desaturate fix's graded_readout) -- diffuse then read
            # member rows over MEMBER columns.
            mc_codes = graded_readout(W, member_rows, alpha, steps)
            mc = measure_codes(mc_codes, S_true, second_order_pairs, labels, props, nclu, pclu,
                               seed, args.k_neighbours, chance)
            # variant (ii): FULL-column -- read member rows over ALL columns (hubs included).
            fc_codes = diffusion_readout_full_columns(W, member_rows, alpha, steps)
            fc = measure_codes(fc_codes, S_true, second_order_pairs, labels, props, nclu, pclu,
                               seed, args.k_neighbours, chance)
            kmc = f"steps{steps}_alpha{alpha}_membercols"
            kfc = f"steps{steps}_alpha{alpha}_fullcols"
            diffusion_sweep[kmc] = mc
            diffusion_sweep[kfc] = fc
            for k, m in ((kmc, mc), (kfc, fc)):
                if m["pearson_vs_Strue"] > best_diff_pearson:
                    best_diff_pearson, best_diff_key = m["pearson_vs_Strue"], k
            print(f"    steps={steps} alpha={alpha} | member-cols: P={mc['pearson_vs_Strue']:+.3f} "
                  f"2nd={mc['second_order_margin']:+.3f} graded={mc['is_graded']} "
                  f"gen={mc['generalization']:.3f} || full-cols: P={fc['pearson_vs_Strue']:+.3f} "
                  f"2nd={fc['second_order_margin']:+.3f} graded={fc['is_graded']} "
                  f"gen={fc['generalization']:.3f}", flush=True)

    best_diff = diffusion_sweep[best_diff_key]
    # Does the best brain-based diffusion read-out pass the same G1/G2 bar as the host-method stand-in?
    diff_g1_ok = (best_diff["pearson_vs_Strue"] >= args.g1_bar and best_diff["is_graded"]
                  and best_diff["second_order_recovered"])
    diff_a1_ok = best_diff["generalization"] >= args.a1_bar
    diff_closes = diff_g1_ok and diff_a1_ok
    # gap to the host-method-on-W stand-in (the apples-to-apples) and to the ceiling.
    gap_to_standin = g1_rec["pearson_learned_vs_Strue"] - best_diff["pearson_vs_Strue"]
    gap_to_ceiling = host_rec["pearson_learned_vs_Strue"] - best_diff["pearson_vs_Strue"]
    print(f"\n    BEST brain-based diffusion read-out: {best_diff_key} -- "
          f"Pearson(sim,S_true)={best_diff['pearson_vs_Strue']:+.3f} "
          f"(gen {best_diff['generalization']:.3f}, 2nd-margin {best_diff['second_order_margin']:+.3f})",
          flush=True)
    print(f"    gap to host-method-on-W stand-in: {gap_to_standin:+.3f}; gap to ceiling: "
          f"{gap_to_ceiling:+.3f}; passes G1+A1 = {diff_closes} "
          f"(G1={diff_g1_ok} A1={diff_a1_ok})", flush=True)
    print(f"    => BRAIN-BASED read-out {'CLOSES the stand-in residual (fully brain-based)' if diff_closes else 'STAYS a stand-in (read-out is a build refinement)'}",
          flush=True)

    return {
        "seed": seed,
        "cycles": args.cycles,
        "corpus": {"n_concepts": len(concepts), "n_members": Nm, "n_facts": corpus["n_facts"],
                   "n_second_order_pairs": len(second_order_pairs)},
        "learn_info": learn_info,
        "learn_seconds": learn_s,
        "anti_cheat_pearson_W_vs_rawcounts": pearson_W_counts,
        "host_ceiling": {"pearson_vs_Strue": host_rec["pearson_learned_vs_Strue"],
                         "generalization": host_gen, "is_graded": host_graded["is_graded"]},
        "random_baseline_generalization": rand_gen,
        "primary_readout_host_method_on_W": {
            "g1_structure_recovery": g1_rec,
            "g1_graded_stats": g1_grad,
            "g2_generalization": gen,
            "g3_cortex_channel": cc,
            "g4_strong_encode": g4,
            "permuted_cooccurrence": {"structure_recovery": perm_rec,
                                      "generalization": perm_gen,
                                      "is_graded": bool(perm_grad["is_graded"])},
            "gates": primary_gates,
        },
        "brain_based_diffusion_sweep": {
            "sweep": diffusion_sweep,
            "best_key": best_diff_key,
            "best_pearson_vs_Strue": best_diff["pearson_vs_Strue"],
            "best_generalization": best_diff["generalization"],
            "best_second_order_margin": best_diff["second_order_margin"],
            "gap_to_host_method_on_W": gap_to_standin,
            "gap_to_ceiling": gap_to_ceiling,
            "closes_residual": bool(diff_closes),
            "best_passes_g1": bool(diff_g1_ok),
            "best_passes_a1": bool(diff_a1_ok),
        },
    }


def _seed_verdict(rseed, args) -> str:
    """Per-seed verdict from the PRIMARY (host-method-on-W) full gate battery."""
    g = rseed["primary_readout_host_method_on_W"]["gates"]
    core = (g["g1_structure_recovered"] and g["g2_a1_generalizes"]
            and g["g2_a2_orthogonal_collapses"] and g["g2_a3_permuted_property_collapses"]
            and g["g3_cortex_roundtrip_closes"] and g["g3_binding_identity_clean"]
            and g["g5_permuted_cooccurrence_collapses"] and g["g5_beats_random_baseline"]
            and g["anti_cheat_W_distinct_from_counts"])
    if args.run_g4:
        core = core and g.get("g4_strong_encode_compatible", False)
    if core:
        return "GO_full"
    # which gate(s) failed -> BOUNDARY characterization
    return "BOUNDARY"


def main():
    p = argparse.ArgumentParser(description="Learned graded-embedding CONFIRMATION (full gates + "
                                            "brain-based read-out)")
    p.add_argument("--seeds", default="42,43,44")
    # the recovered recipe
    p.add_argument("--cycles", type=int, default=2, help="de-saturated low-cycle regime (recovered)")
    # toy corpus (MUST match the de-risk/desaturate defaults)
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--per-cluster", type=int, default=5)
    p.add_argument("--n-props", type=int, default=4)
    p.add_argument("--hub-facts-per-member", type=int, default=6)
    p.add_argument("--bridge-facts", type=int, default=8)
    p.add_argument("--triplet-facts-per-cluster", type=int, default=4)
    # learned-assoc-graph (brain-based learner)
    p.add_argument("--n-pool", type=int, default=2000)
    p.add_argument("--pattern-size", type=int, default=100)
    # read-out
    p.add_argument("--svd-dim", type=int, default=40)
    # BRAIN-BASED diffusion sweep
    p.add_argument("--diffusion-steps-sweep", type=int, nargs="+", default=[2, 3, 4, 6])
    p.add_argument("--diffusion-alpha-sweep", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    # gate bars (match the de-risk)
    p.add_argument("--g1-bar", type=float, default=0.5)
    p.add_argument("--a1-bar", type=float, default=0.7)
    p.add_argument("--c2-bar", type=float, default=0.7)
    p.add_argument("--binding-bar", type=float, default=0.9)
    p.add_argument("--k-neighbours", type=int, default=3)
    p.add_argument("--flip-frac", type=float, default=0.1)
    # cortex-channel DG (numpy)
    p.add_argument("--cc-n-dg", type=int, default=2000)
    p.add_argument("--cc-ensemble-size", type=int, default=100)
    # GATE 4 strong-encode (spiking) -- the de-risk's validated operating point
    p.add_argument("--run-g4", action="store_true")
    p.add_argument("--g4-n-lang-input", type=int, default=256)
    p.add_argument("--g4-n-ec", type=int, default=120)
    p.add_argument("--g4-n-dg", type=int, default=600)
    p.add_argument("--g4-n-dg-pv-basket", type=int, default=180)
    p.add_argument("--g4-n-ca3", type=int, default=300)
    p.add_argument("--g4-n-ca1", type=int, default=120)
    p.add_argument("--g4-ensemble-size", type=int, default=40)
    p.add_argument("--g4-drive-pA", type=float, default=800.0)
    p.add_argument("--g4-k", type=int, default=40)
    p.add_argument("--g4-window", type=int, default=150)
    p.add_argument("--g4-reset-steps", type=int, default=40)
    p.add_argument("--g4-n-repro-pairs", type=int, default=6)
    p.add_argument("--g4-decorr-bar", type=float, default=0.10)
    p.add_argument("--g4-repro-bar", type=float, default=0.90)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    backend = os.environ.get("SIM_BACKEND", "auto")
    chance = 1.0 / args.n_props
    t0 = time.time()
    print(f"[learned-graded-embedding CONFIRMATION] seeds={seeds} backend={backend} "
          f"run_g4={args.run_g4} cycles={args.cycles}", flush=True)
    print(f"  toy: {args.n_clusters}x{args.per_cluster} (+hubs); learner=LearnedAssocGraph "
          f"(n_pool={args.n_pool}, pattern_size={args.pattern_size})", flush=True)
    print(f"  PRIMARY read-out = host-method (PPMI+SVD) on FULL learned W; BRAIN-BASED sweep: "
          f"diffusion steps={args.diffusion_steps_sweep} x alpha={args.diffusion_alpha_sweep}",
          flush=True)
    print(f"  bars: G1(Pearson>={args.g1_bar}) A1(gen>={args.a1_bar}) C2(round-trip>={args.c2_bar})",
          flush=True)

    per_seed = {}
    for s in seeds:
        per_seed[str(s)] = run_seed(s, args)

    # ---------- overall verdict ----------
    verdicts = {str(s): _seed_verdict(per_seed[str(s)], args) for s in seeds}
    vset = set(verdicts.values())
    if vset == {"GO_full"}:
        consensus = "GO_full"
    elif "GO_full" in vset:
        consensus = "MIXED_with_GO_full:" + ",".join(f"{s}={v}" for s, v in verdicts.items())
    else:
        consensus = "BOUNDARY:" + ",".join(f"{s}={v}" for s, v in verdicts.items())

    def agg(path):
        out = []
        for s in seeds:
            d = per_seed[str(s)]
            for k in path:
                d = d[k]
            out.append(d)
        return out

    # PRIMARY (host-method-on-W) load-bearing aggregates
    prim_pearson = agg(["primary_readout_host_method_on_W", "g1_structure_recovery",
                        "pearson_learned_vs_Strue"])
    prim_so_margin = agg(["primary_readout_host_method_on_W", "g1_structure_recovery",
                          "second_order_margin"])
    prim_gen = [per_seed[str(s)]["primary_readout_host_method_on_W"]["g2_generalization"]
                ["graded"]["accuracy"] for s in seeds]
    prim_cortex_pearson = agg(["primary_readout_host_method_on_W", "g3_cortex_channel",
                               "cortex_channel_pearson"])
    prim_bind_id = agg(["primary_readout_host_method_on_W", "g3_cortex_channel",
                        "binding_identity_acc"])
    host_pearson = agg(["host_ceiling", "pearson_vs_Strue"])
    W_counts = agg(["anti_cheat_pearson_W_vs_rawcounts"])

    # BRAIN-BASED diffusion aggregates
    diff_best_pearson = agg(["brain_based_diffusion_sweep", "best_pearson_vs_Strue"])
    diff_best_gen = agg(["brain_based_diffusion_sweep", "best_generalization"])
    diff_gap_standin = agg(["brain_based_diffusion_sweep", "gap_to_host_method_on_W"])
    diff_gap_ceiling = agg(["brain_based_diffusion_sweep", "gap_to_ceiling"])
    diff_closes = agg(["brain_based_diffusion_sweep", "closes_residual"])

    def all_gate(g):
        return all(per_seed[str(s)]["primary_readout_host_method_on_W"]["gates"].get(g, False)
                   for s in seeds)

    gates_all = {
        "g1_structure_recovered": all_gate("g1_structure_recovered"),
        "g2_a1_generalizes": all_gate("g2_a1_generalizes"),
        "g2_a2_orthogonal_collapses": all_gate("g2_a2_orthogonal_collapses"),
        "g2_a3_permuted_property_collapses": all_gate("g2_a3_permuted_property_collapses"),
        "g3_cortex_roundtrip_closes": all_gate("g3_cortex_roundtrip_closes"),
        "g3_binding_identity_clean": all_gate("g3_binding_identity_clean"),
        "g5_permuted_cooccurrence_collapses": all_gate("g5_permuted_cooccurrence_collapses"),
        "g5_beats_random_baseline": all_gate("g5_beats_random_baseline"),
        "anti_cheat_W_distinct_from_counts": all_gate("anti_cheat_W_distinct_from_counts"),
        "g4_strong_encode_compatible": (all_gate("g4_strong_encode_compatible")
                                        if args.run_g4 else None),
    }

    brain_based_closes_all = all(bool(x) for x in diff_closes)

    summary = {
        "consensus_verdict": consensus,
        "per_seed_verdict": verdicts,
        "seeds": seeds,
        "backend": backend,
        "cycles": args.cycles,
        "run_g4": bool(args.run_g4),
        "brain_based_note": ("the learn is the project's spiking-Hebbian recurrent (LearnedAssocGraph) "
                             "at the de-saturated low-cycle (cycles=2) regime. PRIMARY read-out = "
                             "host-method (PPMI+SVD) on the FULL hub-inclusive learned W -- the "
                             "validated STAND-IN. The BRAIN-BASED read-out tested = spreading-"
                             "activation/diffusion through the hubs on the same learned W. The host "
                             "PPMI+SVD on RAW counts is the labelled CEILING ONLY (NOT the deliverable)."),
        "gates_all_seeds_primary": gates_all,
        "primary_load_bearing": {
            "pearson_struct_recovery_per_seed": prim_pearson,
            "pearson_struct_recovery_mean": float(np.mean(prim_pearson)),
            "second_order_margin_mean": float(np.mean(prim_so_margin)),
            "generalization_graded_per_seed": prim_gen,
            "generalization_graded_mean": float(np.mean(prim_gen)),
            "cortex_channel_pearson_per_seed": prim_cortex_pearson,
            "cortex_channel_pearson_mean": float(np.mean(prim_cortex_pearson)),
            "binding_identity_mean": float(np.mean(prim_bind_id)),
            "host_ceiling_pearson_mean": float(np.mean(host_pearson)),
            "anti_cheat_pearson_W_vs_rawcounts_per_seed": W_counts,
            "anti_cheat_pearson_W_vs_rawcounts_mean": float(np.mean(W_counts)),
            "generalization_chance": chance,
        },
        "brain_based_readout": {
            "best_pearson_vs_Strue_per_seed": diff_best_pearson,
            "best_pearson_vs_Strue_mean": float(np.mean(diff_best_pearson)),
            "best_generalization_per_seed": diff_best_gen,
            "best_generalization_mean": float(np.mean(diff_best_gen)),
            "gap_to_host_method_on_W_mean": float(np.mean(diff_gap_standin)),
            "gap_to_ceiling_mean": float(np.mean(diff_gap_ceiling)),
            "closes_residual_per_seed": [bool(x) for x in diff_closes],
            "closes_residual_all_seeds": bool(brain_based_closes_all),
            "best_keys_per_seed": agg(["brain_based_diffusion_sweep", "best_key"]),
        },
        "recovered_recipe_reference": {
            "source": "2026-06-11-learned-graded-embedding-desaturate-GO.md (commit 1febdd20)",
            "cycles": 2,
            "readout": "host-method (PPMI+SVD) on the FULL hub-inclusive learned W",
            "seed42_pearson": 0.843, "seed43_pearson": 0.879,
            "ceiling_seed42": 0.932, "ceiling_seed43": 0.950,
            "generalization": 1.000,
        },
        "elapsed_total_s": time.time() - t0,
    }
    if args.run_g4:
        g4_decorr = agg(["primary_readout_host_method_on_W", "g4_strong_encode", "dg_between_cos_mean"])
        g4_repro = agg(["primary_readout_host_method_on_W", "g4_strong_encode", "dg_repro_mean"])
        summary["primary_load_bearing"]["g4_dg_between_cos_mean_per_seed"] = g4_decorr
        summary["primary_load_bearing"]["g4_dg_repro_mean_per_seed"] = g4_repro

    print(f"\n{'='*82}", flush=True)
    print(f"  CONFIRMATION SUMMARY", flush=True)
    print(f"{'='*82}", flush=True)
    print(f"  CONSENSUS VERDICT: {consensus}", flush=True)
    for s in seeds:
        print(f"  seed {s}: {verdicts[str(s)]}", flush=True)
    print(f"\n  PRIMARY (host-method-on-W) full gate battery, all seeds:", flush=True)
    for k, v in gates_all.items():
        print(f"    {k}: {v}", flush=True)
    print(f"  PRIMARY load-bearing (mean): Pearson(sim,S_true)={np.mean(prim_pearson):+.3f} "
          f"(ceiling {np.mean(host_pearson):+.3f}); gen {np.mean(prim_gen):.3f} (chance {chance:.3f}); "
          f"G3 cortex Pearson {np.mean(prim_cortex_pearson):+.3f} (binding id "
          f"{np.mean(prim_bind_id):.3f})", flush=True)
    if args.run_g4:
        print(f"    G4: DG between-cos {np.mean(g4_decorr):+.3f} repro {np.mean(g4_repro):.3f}",
              flush=True)
    print(f"  anti-cheat Pearson(W, raw_counts) mean {np.mean(W_counts):+.3f} (<0.999 distinct)",
          flush=True)
    print(f"\n  BRAIN-BASED diffusion read-out (best per seed): "
          f"Pearson {np.mean(diff_best_pearson):+.3f} gen {np.mean(diff_best_gen):.3f}; "
          f"gap-to-stand-in {np.mean(diff_gap_standin):+.3f} gap-to-ceiling "
          f"{np.mean(diff_gap_ceiling):+.3f}", flush=True)
    print(f"  BRAIN-BASED read-out closes the stand-in residual (all seeds): {brain_based_closes_all}",
          flush=True)
    print(f"  Total elapsed: {summary['elapsed_total_s']:.1f}s", flush=True)
    print(f"{'='*82}\n", flush=True)

    out_data = {"summary": summary, "per_seed": per_seed}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        args.out = os.path.join(raw_dir, "_lge_confirm_multiseed.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out_data


if __name__ == "__main__":
    main()
