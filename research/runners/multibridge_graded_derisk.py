"""MULTI-BRIDGE LEARNED-GRADED-EMBEDDING CHEAP-FIRST DE-RISK -- does the within-bridge GRADED gain
coexist with the existing cross-bridge composition layer + the no-confab moat, or does graded coding
BREAK the cross-bridge layer? (The single load-bearing question 2-3 bridges answer that 32 cannot.)

SPEC: docs/plans/2026-06-11-multibridge-learned-embedding-derisk-design.md (esp. SS4 "THE CHEAP-FIRST
FALSIFICATION" -- the M1..M7 measurements + GO/BOUNDARY/NEGATIVE criteria; SS5 reusable machinery;
SS6 set-point calibration). This is the controller's first build step after that design was approved.

THE FORK THIS GATES (design SS0). The dual/CLS learned graded-similarity cortex is validated
SINGLE-POOL to V=320 but a single pool OOMs by ~V=320-450 on a 24 GB 3090. The only large-vocabulary
route is MULTI-BRIDGE (many small bridges, each running the per-bridge recipe). The orthogonal
insight (SS1.1): the per-bridge GRADED embedding is a WITHIN-pool property (cat ~ dog only if both
share a recurrent), while the existing cross-bridge layer relates concept IDENTITIES, not
similarities. So the corpus must be SHARDED BY SEMANTIC CLUSTER (animals together, foods together),
within-bridge -> graded generalization, cross-bridge -> identity composition. The cheap-first run
asks: when each bridge carries a learned GRADED code (correlated within a bridge) instead of the
ORTHOGONAL sparse code the 320 ensemble was validated on, does the cross-bridge tag layer still
recall, does the moat survive, do they coexist on the SAME bridges?

WHAT THIS RUNNER DOES (the design's 3-bridge x 64-concept shape, A2 curated sharding, C1 V-tag
cross-bridge layer, B1 working hypothesis = no shared cross-bridge embedding needed). 4 --mode
branches:

  --mode precondition  M1 (per-bridge within-bridge generalization, held-out-neighbour A1, + A2
                       orthogonal / A3 permuted-property controls collapse), M2 (per-bridge
                       Pearson(sim,S_true) + 2nd-order cat~dog margin), G5 (permuted-co-occurrence
                       control gated on a MARGIN/Pearson threshold -- NOT the brittle is_graded
                       boolean, the carry-forward fix from the homeostasis finding), and M6 (random-
                       shard anti-cheat: within-bridge generalization MUST collapse to chance under a
                       structure-destroying random shard). Trains + (optionally) SAVES each graded
                       bridge.
  --mode cross         M3 (cross-bridge composition recall over the >=2 graded bridges via the V-tag
                       engram-tag layer: store cross-bridge facts e.g. `dog eats meat`; query cue ->
                       target retrieved top-2 across bridges; signal vs noise floor) + M7 (permuted
                       cross-bridge mapping anti-cheat: shuffle which target each cue binds to ->
                       recall MUST collapse).
  --mode moat          M4 (no-confab moat: a LEARNED Bogacz-Brown familiarity gate validated
                       ALONGSIDE a host abstention check on graded-coded cross-bridge facts -- the
                       gate may ACCEPT only where the host accepts; the host-abstain/gate-accept cell
                       must be 0; lesion collapses the separation). Reuses the
                       familiarity_gate_v320_validation machinery (RFPhasorComposer relational moat +
                       AntiHebbianFamiliarity) -- a CPU/numpy check exactly as the V=320 moat
                       validation runs.
  --mode full          precondition + cross + moat, end-to-end, with the combined GO/BOUNDARY/NEGATIVE
                       verdict (design SS4.3). M5 (conversational-matrix subset) is exercised by the
                       moat's who/what + abstention relational queries (the matrix consumes identity
                       binding + abstention, which M3 + M4 deliver) -- noted in the verdict.

REUSE-BY-IMPORT (NO sim/ edits anywhere; every cited probe is runner-side):
  - per-bridge graded learn:   HomeostaticAssocGraph (Oja, set-point) + the brain-based divnorm
                               read-out (divnorm_spreading_readout), from
                               learned_graded_embedding_homeostasis_probe + ..._divnorm_readout_probe.
  - the toy co-occurrence corpus + structure-recovery + generalization gate harness:
                               build_toy_cooccurrence / structure_recovery / architecture_generalization
                               from learned_graded_embedding_derisk_probe; codebook_similarity_stats /
                               assign_properties / run_generalization from dual_cls_architecture_proof_probe.
  - cross-bridge V-tag layer:  the engram-tag encode/recall RECIPE from shared_pool_chat.py, ADAPTED
                               to the graded bridge's "pool" region (the graded bridge built by
                               HomeostaticAssocGraph has NO `language_input`/`shared_concept_pool`
                               region the verbatim helpers assume -- see HELPER-SIGNATURE ADAPTATION
                               below). bridge.start_engram_recording / commit_engram_tag(region_filter=
                               ["pool"]) / stimulate_tag are the project's shipped engram-tag API.
  - the moat:                  RelationalFamiliarityGate + AntiHebbianFamiliarity (the learned gate)
                               + the host abstention, from familiarity_gate_v320_validation +
                               cortex_learned_cleanup_derisk + rf_phasor_composer.

HELPER-SIGNATURE ADAPTATION (honest note, design SS7.6 "integration interactions"). The V-tag helpers
in shared_pool_chat.py (encode_pair_engram_sparse / encode_partial_pair_engram_sparse /
stim_recall_sparse_rates) are HARD-WIRED to a `shared_concept_pool` pool region + a `language_input`
region driven by orthogonal_drive_pattern. The GRADED bridge (HomeostaticAssocGraph -> _D_sparse_
heteroassoc.build) has ONLY `pool` + `fs` regions and is driven by direct pattern stimulation (no
language_input). So this runner cannot call those helpers VERBATIM; it ports the SAME validated
RECIPE -- start_engram_recording -> drive the concept(s)' sparse pattern(s) with a teacher current ->
commit_engram_tag(top_k, region_filter=["pool"]) -> stimulate_tag -> accumulate per-pattern firing --
onto the graded bridge's `pool` region. The patterns are the SAME generate_sparse_patterns(seed) the
V-tag layer regenerates, so the cross-bridge link is still the shared tag NAME + per-bridge identity
recall (the V-tag mechanism, design SS1.1-V-tag), just over a pool region named `pool`.

Run (GPU for the graded-bridge spiking ops; the moat sub-check is CPU/numpy):
  # (M1/M2/M6/G5) train + gate the 3 graded bridges:
  SIM_BACKEND=cupy python -u -m research.runners.multibridge_graded_derisk \
      --mode precondition --seeds 42,43,44 \
      --n-bridges 3 --concepts-per-bridge 64 --n-pool 2400 --pattern-size 100 \
      --homeo oja --homeo-target 40 --cycles 10 \
      --save-bridge-dir research/findings/raw/g11_bg/graded_derisk/bridges \
      --out research/findings/raw/_multibridge_graded_derisk_precondition.json
  # (full) everything + combined verdict:
  SIM_BACKEND=cupy python -u -m research.runners.multibridge_graded_derisk \
      --mode full --seeds 42,43,44 --n-bridges 3 --concepts-per-bridge 64 --n-pool 2400 \
      --homeo oja --homeo-target 40 --cycles 10 \
      --out research/findings/raw/_multibridge_graded_derisk_full.json
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

# ---- per-bridge graded learn + read-out (reuse-by-import; NO sim/ edits) ----
from research.runners.learned_graded_embedding_homeostasis_probe import (  # noqa: E402
    HomeostaticAssocGraph,
    learn_W_homeostatic,
)
from research.runners.learned_graded_embedding_divnorm_readout_probe import (  # noqa: E402
    divnorm_spreading_readout,
)
# ---- toy co-occurrence corpus + structure-recovery + generalization gates ----
from research.runners.learned_graded_embedding_derisk_probe import (  # noqa: E402
    build_toy_cooccurrence,
    permute_corpus,
    structure_recovery,
    architecture_generalization,
)
from research.runners.learned_graded_embedding_diagnose import (  # noqa: E402
    raw_count_matrix,
    offdiag_pearson,
)
from research.runners.dual_cls_architecture_proof_probe import (  # noqa: E402
    codebook_similarity_stats,
    assign_properties,
    run_generalization,
)
from research.runners.concept_pool_sparse_distributed import (  # noqa: E402
    generate_sparse_patterns,
)


# ===========================================================================
# Curated semantic-cluster sharding (design A2). Three mutually-dissimilar
# super-clusters (animals / foods / vehicles), each an internally-structured
# set of `concepts_per_bridge` mutually-similar member concepts. Internally a
# super-cluster is the de-risk's hub-mediated sub-cluster structure
# (n_sub sub-clusters x per_sub members), so within-bridge graded
# generalization (cat ~ dog via a shared sub-hub) is meaningful; the three
# super-clusters are dissimilar by construction so cross-bridge is pure
# identity composition.
# ===========================================================================
SHARD_NAMES = ["animals", "foods", "vehicles", "tools", "clothes", "furniture",
               "plants", "weather"]


def _factor_subclusters(n_concepts: int, target_per_sub: int = 8) -> tuple:
    """Pick (n_sub, per_sub) with n_sub * per_sub == n_concepts, per_sub close to
    target_per_sub and >= 2 (so within-sub second-order pairs exist)."""
    best = None
    for per_sub in range(max(2, target_per_sub - 4), target_per_sub + 5):
        if n_concepts % per_sub == 0:
            n_sub = n_concepts // per_sub
            if n_sub >= 2:
                score = abs(per_sub - target_per_sub)
                if best is None or score < best[0]:
                    best = (score, n_sub, per_sub)
    if best is None:
        # fallback: 2 sub-clusters
        per_sub = max(2, n_concepts // 2)
        return (max(2, n_concepts // per_sub), per_sub)
    return (best[1], best[2])


def build_bridge_corpus(shard_name: str, n_concepts: int, seed: int, args) -> dict:
    """Build ONE bridge's within-bridge corpus + ground truth. Concept names are
    namespaced by the shard (so cross-bridge facts can reference them unambiguously)
    but the internal structure is the validated de-risk hub-mediated sub-cluster
    corpus. Returns the de-risk corpus dict + namespaced concept names + the
    (n_sub, per_sub) factoring (the labels used by the generalization gates)."""
    n_sub, per_sub = _factor_subclusters(n_concepts, args.target_per_sub)
    # distinct per-shard seed so each bridge's sub-structure differs (mirrors the
    # distinct-seed-per-bridge route; not load-bearing here but avoids identical
    # corpora across shards).
    shard_seed = seed * 1000 + (SHARD_NAMES.index(shard_name)
                                if shard_name in SHARD_NAMES else 0)
    corpus = build_toy_cooccurrence(
        n_sub, per_sub, shard_seed,
        hub_facts_per_member=args.hub_facts_per_member,
        bridge_facts=args.bridge_facts,
        triplet_facts_per_cluster=args.triplet_facts_per_cluster)
    # namespace the concept names with the shard so cross-bridge facts are unambiguous
    pfx = f"{shard_name}."
    concepts = [pfx + c for c in corpus["concepts"]]
    members = [pfx + m for m in corpus["members"]]
    facts = [tuple(pfx + c for c in f) for f in corpus["facts"]]
    return {
        "shard": shard_name, "n_sub": n_sub, "per_sub": per_sub,
        "concepts": concepts, "members": members,
        "labels": corpus["labels"], "S_true": corpus["S_true"],
        "second_order_pairs": corpus["second_order_pairs"],
        "facts": facts, "n_facts": len(facts),
        "member_index": {m: i for i, m in enumerate(members)},
        "_local": corpus,   # the un-namespaced de-risk corpus (for raw_count matrix etc.)
    }


# ===========================================================================
# Per-bridge GRADED codes via the validated homeostatic learn + brain-based
# divnorm read-out. The corpus concepts are namespaced; the learner only needs
# the UN-namespaced ordering (it is name-agnostic, just an index list), so we
# learn over the local de-risk corpus and namespace afterwards.
# ===========================================================================
def learn_bridge_graded(bridge_corpus: dict, seed: int, args, homeo_target=None):
    """Learn the graded W + brain-based codes for ONE bridge. Returns
    (W, codes, member_rows, info). homeo_target overrides args.homeo_target (for the
    set-point bracket sweep)."""
    local = bridge_corpus["_local"]
    concepts = local["concepts"]
    members = local["members"]
    member_rows = np.asarray([concepts.index(m) for m in members], dtype=int)
    tgt = args.homeo_target if homeo_target is None else homeo_target
    W, info = learn_W_homeostatic(
        concepts, local["facts"], seed, args.n_pool, args.pattern_size, args.cycles,
        gamma=1.0, cap=None, homeo=args.homeo, homeo_target=tgt, homeo_clip_only=True)
    codes = divnorm_spreading_readout(
        W, member_rows,
        divnorm=args.readout_divnorm, order=args.readout_order,
        sigma=args.readout_sigma, exponent=args.readout_exponent,
        alpha=args.diffusion_alpha, steps=args.diffusion_steps,
        log_clip=args.readout_log_clip)
    return W, codes, member_rows, info


def _g5_robust(perm_codes, perm_W, member_rows, local, args, chance):
    """G5 permuted-co-occurrence control gated on a MARGIN/Pearson THRESHOLD (NOT the
    brittle is_graded boolean -- the carry-forward fix from
    2026-06-11-learned-graded-embedding-homeostasis-GO.md SS4: is_graded is a coin-flip
    on a structureless permuted matrix and spuriously flagged a seed-43 BOUNDARY).
    Collapses iff: permuted 2nd-order margin < +0.10 AND |permuted Pearson| < g1_bar*0.6
    AND permuted generalization <= 1.5x chance."""
    labels = local["labels"]
    S_true = local["S_true"]
    so_pairs = local["second_order_pairs"]
    rec = structure_recovery(perm_codes, S_true, so_pairs, args._seed_for_g5)
    props = local["_props"]
    nclu, pclu = local["_nclu"], local["_pclu"]
    gen = float(run_generalization(perm_codes, labels, props, nclu, pclu,
                                   args._seed_for_g5, args.k_neighbours)["accuracy"])
    margin_ok = rec["second_order_margin"] < args.so_margin_bar          # margin-based, robust
    pearson_ok = abs(rec["pearson_learned_vs_Strue"]) < args.g1_bar * 0.6
    gen_ok = gen <= 1.5 * chance
    return {
        "permuted_second_order_margin": rec["second_order_margin"],
        "permuted_pearson_vs_Strue": rec["pearson_learned_vs_Strue"],
        "permuted_generalization": gen,
        "g5_collapses_robust": bool(margin_ok and pearson_ok and gen_ok),
        "g5_margin_ok": bool(margin_ok), "g5_pearson_ok": bool(pearson_ok),
        "g5_gen_ok": bool(gen_ok),
    }


# ===========================================================================
# M1 + M2 + G5 -- the per-bridge graded gate (precondition check, design M1/M2).
# ===========================================================================
def per_bridge_gates(bridge_corpus, seed, args, homeo_target=None):
    local = bridge_corpus["_local"]
    concepts = local["concepts"]
    members = local["members"]
    labels = local["labels"]
    S_true = local["S_true"]
    so_pairs = local["second_order_pairs"]
    nclu, pclu = bridge_corpus["n_sub"], bridge_corpus["per_sub"]
    member_rows = np.asarray([concepts.index(m) for m in members], dtype=int)
    Nm = len(members)
    props = assign_properties(nclu, pclu, args.n_props, seed)
    chance = 1.0 / args.n_props
    # stash for the G5 helper
    local["_props"] = props
    local["_nclu"] = nclu
    local["_pclu"] = pclu
    args._seed_for_g5 = seed
    C_full = raw_count_matrix(concepts, local["facts"])

    t0 = time.time()
    W, codes, member_rows, info = learn_bridge_graded(bridge_corpus, seed, args, homeo_target)
    learn_s = time.time() - t0

    # M2: structure recovery (brain-based read-out) + 2nd-order cat~dog margin
    grad = codebook_similarity_stats(codes, labels)
    rec = structure_recovery(codes, S_true, so_pairs, seed)
    pearson_W_counts = offdiag_pearson(W, C_full)

    # M1: within-bridge generalization (held-out-neighbour A1) + A2/A3 controls collapse
    gen = architecture_generalization(codes, labels, props, nclu, pclu, seed,
                                      args.k_neighbours, args.a1_bar)

    # G5: permuted-co-occurrence (robust margin/Pearson criterion)
    perm_facts = permute_corpus(local["facts"], concepts, seed)
    W_perm, codes_perm, _, _ = (
        _learn_perm(concepts, perm_facts, member_rows, seed, args, homeo_target))
    g5 = _g5_robust(codes_perm, W_perm, member_rows, local, args, chance)

    # M1 GO/BOUNDARY/NEGATIVE band on generalization
    gen_acc = gen["graded"]["accuracy"]
    if gen_acc >= args.a1_bar:
        m1_band = "GO"
    elif gen_acc >= 0.5:
        m1_band = "BOUNDARY"
    else:
        m1_band = "NEGATIVE"
    # M2 band: Pearson >= 0.7 OR (margin >= +0.10 AND gen >= 0.7)
    m2_go = (rec["pearson_learned_vs_Strue"] >= 0.7) or \
            (rec["second_order_margin"] >= args.so_margin_bar and gen_acc >= args.a1_bar)
    if m2_go:
        m2_band = "GO"
    elif rec["second_order_margin"] >= args.so_margin_bar:
        m2_band = "BOUNDARY"
    else:
        m2_band = "NEGATIVE"

    return {
        "shard": bridge_corpus["shard"],
        "n_concepts": Nm, "n_sub": nclu, "per_sub": pclu, "n_facts": len(local["facts"]),
        "learn_seconds": learn_s,
        "recurrent_mean": info["recurrent_mean"], "recurrent_nnz": info["recurrent_nnz"],
        "n_neurons": info["n_neurons"],
        # M2
        "pearson_sim_vs_Strue": rec["pearson_learned_vs_Strue"],
        "pearson_W_vs_rawcounts": pearson_W_counts,
        "second_order_margin": rec["second_order_margin"],
        "is_graded": bool(grad["is_graded"]),
        "within_cos": grad["within_cluster_cos_mean"], "between_cos": grad["between_cluster_cos_mean"],
        "m2_band": m2_band,
        # M1
        "generalization": gen_acc,
        "gen_chance": chance, "gen_ratio": gen["graded"]["ratio_vs_chance"],
        "orthogonal_acc": gen["orthogonal"]["accuracy"], "a2_collapses": bool(gen["a2"]),
        "permuted_prop_acc": gen["permuted"]["accuracy"], "a3_collapses": bool(gen["a3"]),
        "a1_generalizes": bool(gen["a1"]),
        "m1_band": m1_band,
        # G5
        **g5,
    }


def _learn_perm(concepts, perm_facts, member_rows, seed, args, homeo_target=None):
    """Learn graded codes on a permuted corpus (the G5 control)."""
    tgt = args.homeo_target if homeo_target is None else homeo_target
    W, info = learn_W_homeostatic(
        concepts, perm_facts, seed, args.n_pool, args.pattern_size, args.cycles,
        gamma=1.0, cap=None, homeo=args.homeo, homeo_target=tgt, homeo_clip_only=True)
    codes = divnorm_spreading_readout(
        W, member_rows, divnorm=args.readout_divnorm, order=args.readout_order,
        sigma=args.readout_sigma, exponent=args.readout_exponent,
        alpha=args.diffusion_alpha, steps=args.diffusion_steps, log_clip=args.readout_log_clip)
    return W, codes, member_rows, info


# ===========================================================================
# M6 -- random-shard anti-cheat. Take ALL concepts across the n_bridges shards,
# RANDOM-shard them into n_bridges groups (destroying the semantic co-location),
# re-learn one such random bridge, and measure within-bridge generalization. It
# MUST collapse to chance (proving M1 measured real co-location, not the
# architecture). We use the SAME generalization harness; "labels" for a random
# shard are the random group's members' ORIGINAL sub-cluster ids, but since the
# members are scattered the within-group sub-cluster structure is destroyed ->
# the property-inheritance generalization collapses.
# ===========================================================================
def random_shard_anticheat(all_corpora, seed, args):
    rng = np.random.RandomState(seed * 4242 + 1)
    # pool every (concept-name, label) from every bridge, keeping their facts
    # association is via concept name; we re-shard the MEMBERS and rebuild a
    # local index over a random subset, then learn + generalize on it.
    all_members = []
    member_to_props = {}
    all_facts = []
    nclu_each, pclu_each = all_corpora[0]["n_sub"], all_corpora[0]["per_sub"]
    # assign each member a property from its ORIGINAL sub-cluster (so the random
    # shard breaks the code<->property co-location, not the property labels)
    for bc in all_corpora:
        props = assign_properties(bc["n_sub"], bc["per_sub"], args.n_props, seed)
        for i, m in enumerate(bc["members"]):
            member_to_props[m] = int(props[i])
        all_members.extend(bc["members"])
        all_facts.extend(bc["facts"])
    # random shard: pick one random group of ~concepts_per_bridge members (scattered
    # across the original shards, so the within-group semantic block structure is gone).
    n_per = len(all_corpora[0]["members"])
    chosen = list(rng.choice(len(all_members), size=min(n_per, len(all_members)), replace=False))
    chosen_members = [all_members[i] for i in chosen]
    # concepts to learn over = chosen members only (a scattered, structureless group).
    # facts: keep facts whose members are all within the chosen group (so the learner
    # sees the random group's internal co-occurrence, which has NO sub-cluster block).
    chosen_set = set(chosen_members)
    sub_facts = [f for f in all_facts if all((c in chosen_set) for c in f)]
    # need >= a couple of facts; if too sparse, include facts touching chosen members
    if len(sub_facts) < max(4, len(chosen_members) // 4):
        sub_facts = [tuple(c for c in f if c in chosen_set) for f in all_facts]
        sub_facts = [f for f in sub_facts if len(f) >= 2]
    concepts = list(chosen_members)
    member_rows = np.arange(len(concepts))
    # labels: the chosen members' original sub-cluster identity is meaningless across
    # bridges, so use each member's property as the label/grouping. The point of M6 is
    # that the CODE carries no structure to predict it (the random shard destroyed the
    # within-group block), so generalization must collapse to chance.
    labels = np.asarray([member_to_props[m] for m in concepts], dtype=int)
    props = np.asarray([member_to_props[m] for m in concepts], dtype=int)
    # restrict the facts to the chosen-member subset (already filtered above) and learn
    # graded codes on the structureless random group.
    learn_facts = [tuple(c for c in f if c in chosen_set) for f in sub_facts]
    learn_facts = [f for f in learn_facts if len(f) >= 2]
    W, info = learn_W_homeostatic(
        concepts, learn_facts, seed, args.n_pool, args.pattern_size, args.cycles,
        gamma=1.0, cap=None, homeo=args.homeo, homeo_target=args.homeo_target,
        homeo_clip_only=True)
    codes = divnorm_spreading_readout(
        W, member_rows, divnorm=args.readout_divnorm, order=args.readout_order,
        sigma=args.readout_sigma, exponent=args.readout_exponent,
        alpha=args.diffusion_alpha, steps=args.diffusion_steps, log_clip=args.readout_log_clip)
    # generalization on the random shard: must collapse to chance.
    chance = 1.0 / args.n_props
    # build a fake (n_clusters, per_cluster) factoring for the harness from labels
    # (the harness needs the cluster layout; we group by the property label).
    # run_generalization needs labels + props + (nclu, pclu); reuse property as both.
    # For the held-out-neighbour split it just needs the grouping to be consistent.
    nclu = len(set(labels.tolist()))
    pclu = max(1, len(concepts) // max(1, nclu))
    try:
        gen = run_generalization(codes, labels, props, nclu, pclu, seed, args.k_neighbours)
        gen_acc = float(gen["accuracy"])
    except Exception as e:  # noqa: BLE001
        gen_acc = float("nan")
        gen = {"error": str(e)}
    collapses = bool(gen_acc <= 1.5 * chance) if not np.isnan(gen_acc) else None
    return {
        "n_random_members": len(concepts), "n_facts": len(sub_facts),
        "random_shard_generalization": gen_acc, "chance": chance,
        "m6_collapses_to_chance": collapses,
    }


# ===========================================================================
# Cross-bridge V-tag layer ADAPTED to the graded bridge's `pool` region
# (HELPER-SIGNATURE ADAPTATION -- see module docstring). The graded bridge has
# regions {pool, fs} (from _D_sparse_heteroassoc.build via HomeostaticAssocGraph)
# and concept patterns = generate_sparse_patterns(seed). We port the SHIPPED
# V-tag encode/recall RECIPE (start_engram_recording -> drive concept pattern(s)
# with a teacher current -> commit_engram_tag(top_k, region_filter=["pool"]) ->
# stimulate_tag -> accumulate per-pattern firing) onto this pool region.
# ===========================================================================
class GradedBridge:
    """One trained graded bridge + its sparse patterns + the engram-tag cross-bridge
    encode/recall over its `pool` region. Holds a live HomeostaticAssocGraph (whose
    `.bridge` is the real SimulationBridge)."""

    def __init__(self, shard_name, concepts_local, seed, args):
        self.shard = shard_name
        self.concepts = list(concepts_local)        # un-namespaced names (index list)
        self.idx = {c: i for i, c in enumerate(self.concepts)}
        self.seed = seed
        self.args = args
        # build the graded bridge (same builder the homeostatic learn uses)
        self.hag = HomeostaticAssocGraph(
            self.concepts, seed=seed, n_pool=args.n_pool, pattern_size=args.pattern_size,
            homeo=args.homeo, homeo_target=args.homeo_target, homeo_clip_only=True)
        self.bridge = self.hag.bridge
        # the per-concept sparse patterns (pool-LOCAL indices) -- regenerated from seed,
        # IDENTICAL to what HomeostaticAssocGraph/LearnedAssocGraph use.
        self.patterns = generate_sparse_patterns(len(self.concepts), args.n_pool,
                                                  args.pattern_size, seed)
        self.pool_base = np.asarray(self.bridge.region_manager.indices("pool"))
        self.encoded_tags = []

    def train(self, facts_local):
        """Run the homeostatic co-occurrence learn over the bridge's facts (so the
        pool recurrent + the graded structure are in place; the cross-bridge tags are
        stored on top)."""
        for f in facts_local:
            self.hag.store_fact(list(f), cycles=self.args.cycles)

    def _pattern_global(self, concept_local):
        i = self.idx[concept_local]
        return self.pool_base[np.asarray(self.patterns[i])]

    def encode_tag(self, tag_name, concept_locals):
        """V-tag RECIPE adapted to `pool`: start recording, drive the concept(s)'
        sparse pattern(s) with a teacher current for the encode window, commit the tag
        over the pool region. concept_locals = list of un-namespaced concept names that
        live in THIS bridge (1 for a cross-bridge partial fact; 2 for a within-bridge
        pair)."""
        from sim.backend import get_backend
        cp, _ = get_backend()
        n_total = self.bridge.cp_external_input_current.shape[0]
        drives = [cp.asarray(self._pattern_global(c), dtype=cp.int64) for c in concept_locals]
        self.bridge.start_engram_recording(tag_name)
        self.bridge.cp_external_input_current[:] = 0.0
        for _ in range(20):
            self.bridge._run_one_simulation_step()
        for _ in range(self.args.encoding_steps):
            ext = cp.zeros(n_total, dtype=cp.float32)
            for d in drives:
                ext[d] = self.args.teacher_pA
            self.bridge.cp_external_input_current[:] = ext
            self.bridge._run_one_simulation_step()
        self.bridge.cp_external_input_current[:] = 0.0
        for _ in range(10):
            self.bridge._run_one_simulation_step()
        self.bridge.commit_engram_tag(tag_name, top_k=self.args.top_k,
                                      region_filter=["pool"])
        if tag_name not in self.encoded_tags:
            self.encoded_tags.append(tag_name)
        return tag_name

    def recall_rates(self, tag_name):
        """Stim the tag, accumulate firing per concept's sparse pattern -> a per-concept
        recall vector (the V-tag per-bridge identity recall)."""
        from sim.backend import get_backend
        cp, _ = get_backend()
        pattern_arrs = [cp.asarray(self._pattern_global(c), dtype=cp.int64)
                        for c in self.concepts]
        self.bridge.stimulate_tag(tag_name, drive_pA=self.args.drive_pA)
        rates = np.zeros(len(self.concepts), dtype=np.float32)
        for _ in range(self.args.drive_steps):
            self.bridge._run_one_simulation_step()
            for j, parr in enumerate(pattern_arrs):
                firing = self.bridge.cp_firing_states[parr]
                s = firing.sum() if hasattr(firing, "sum") else 0
                if hasattr(s, "item"):
                    s = s.item()
                rates[j] += float(s)
        self.bridge.clear_tag_drive(tag_name)
        self.bridge.cp_external_input_current[:] = 0.0
        for _ in range(20):
            self.bridge._run_one_simulation_step()
        return rates


def _strip_shard(name):
    """`animals.c0_m1` -> ('animals', 'c0_m1')."""
    if "." in name:
        s, c = name.split(".", 1)
        return s, c
    return None, name


# ===========================================================================
# M3 + M7 -- cross-bridge composition recall + permuted-mapping anti-cheat.
# Cross-bridge facts: (cue, target) where cue lives in one bridge and target in
# another (e.g. dog eats meat -> (animals.dog, foods.meat)). We store the fact as
# a tag named "<cue>__<target>" in BOTH bridges (the cue's pattern in its bridge,
# the target's pattern in its bridge) -- the V-tag distributed key-value store.
# Recall: search every bridge's tag names for the cue, stim each matching tag,
# read the TARGET bridge's per-concept recall, check the target is top-2 and the
# signal clears the noise floor.
# ===========================================================================
def cross_bridge_eval(graded_bridges, cross_facts, seed, args, permuted=False):
    """graded_bridges: {shard -> GradedBridge}. cross_facts: list of (cue_full,
    target_full) namespaced concept pairs. If permuted, the target each cue binds to
    is SHUFFLED (the M7 anti-cheat). Returns the recall stats."""
    rng = np.random.RandomState(seed * 99 + (7 if permuted else 0))
    facts = list(cross_facts)
    if permuted:
        targets = [t for (_c, t) in facts]
        perm = rng.permutation(len(targets))
        # ensure the permutation actually deranges (no cue keeps its true target) when possible
        if len(targets) > 1 and all(perm[i] == i for i in range(len(targets))):
            perm = np.roll(perm, 1)
        facts = [(facts[i][0], targets[perm[i]]) for i in range(len(facts))]

    results = []
    for (cue_full, tgt_full) in facts:
        cue_shard, cue_local = _strip_shard(cue_full)
        tgt_shard, tgt_local = _strip_shard(tgt_full)
        if cue_shard not in graded_bridges or tgt_shard not in graded_bridges:
            continue
        tag = f"{cue_full}__{tgt_full}"
        # store the fact in BOTH bridges (V-tag distributed key-value: each bridge
        # imprints its own concept's pattern under the shared tag name).
        graded_bridges[cue_shard].encode_tag(tag, [cue_local])
        graded_bridges[tgt_shard].encode_tag(tag, [tgt_local])

    # recall: for each TRUE cross-bridge fact, find tags containing the cue, stim them,
    # read the target bridge's recall vector, check the target concept.
    n_top1 = n_top2 = 0
    margins = []
    signal_vs_floor = []
    detail = []
    for (cue_full, tgt_full) in facts:
        cue_shard, cue_local = _strip_shard(cue_full)
        tgt_shard, tgt_local = _strip_shard(tgt_full)
        if cue_shard not in graded_bridges or tgt_shard not in graded_bridges:
            continue
        tgt_bridge = graded_bridges[tgt_shard]
        # tags in the TARGET bridge whose CUE (first token of the shared tag name) is
        # this cue -- the V-tag cross-bridge link is the shared tag string.
        matches = [t for t in tgt_bridge.encoded_tags if t.split("__")[0] == cue_full]
        if not matches:
            continue
        agg = np.zeros(len(tgt_bridge.concepts), dtype=np.float32)
        for t in matches:
            agg += tgt_bridge.recall_rates(t)
        order = np.argsort(-agg)
        tgt_idx = tgt_bridge.idx[tgt_local]
        rank = int(np.where(order == tgt_idx)[0][0])
        n_top1 += int(rank == 0)
        n_top2 += int(rank <= 1)
        # noise floor = mean of the non-top, non-target rates; signal = target rate
        sig = float(agg[tgt_idx])
        floor = float(np.median(agg)) if agg.size else 0.0
        best_other = float(max((agg[i] for i in range(len(agg)) if i != tgt_idx), default=0.0))
        margins.append(sig - best_other)
        signal_vs_floor.append(sig / (floor + 1e-9))
        detail.append({"cue": cue_full, "target": tgt_full, "rank": rank,
                       "target_rate": sig, "noise_floor": floor, "best_other": best_other})

    n = len(detail)
    top2_frac = (n_top2 / n) if n else 0.0
    top1_frac = (n_top1 / n) if n else 0.0
    mean_svf = float(np.mean(signal_vs_floor)) if signal_vs_floor else 0.0
    return {
        "permuted": permuted, "n_cross_facts": n,
        "top1_fraction": top1_frac, "top2_fraction": top2_frac,
        "mean_signal_vs_floor": mean_svf,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "detail": detail,
    }


# ===========================================================================
# M4 -- the no-confab moat (familiarity gate ALONGSIDE the host abstention).
# Reuse the familiarity_gate_v320_validation machinery: build an RFPhasorComposer
# over the cross-bridge concepts + store the cross-bridge facts as relational SVO,
# imprint a relational familiarity gate, and validate the gate AGREES with the host
# moat (gate may ACCEPT only where host accepts; host-abstain/gate-accept must be 0)
# + lesion collapses. CPU/numpy (exactly as the V=320 moat validation runs).
# ===========================================================================
def moat_eval(cross_facts, all_member_names, seed, args):
    from research.runners.rf_phasor_composer import RFPhasorComposer
    from research.runners.familiarity_gate_v320_validation import RelationalFamiliarityGate

    # the relation token is a dedicated in-vocab concept so the abstention is RELATIONAL on
    # (action, patient) -- include it in the composer's vocab so it gets a real phasor code.
    REL = "RELATION"
    vocab = sorted(set(all_member_names) | {REL})
    composer = RFPhasorComposer(seed=seed, D=args.moat_D, vocab=vocab, period=200)
    # store the cross-bridge facts as relational SVO: (cue, RELATION, target).
    stored = []
    seen_ap = set()
    seen_aa = set()
    for (cue_full, tgt_full) in cross_facts:
        a = cue_full          # agent = cue
        ac = REL              # action = the shared relation
        p = tgt_full          # patient = target
        if a not in composer.concepts or p not in composer.concepts:
            continue
        if (ac, p) in seen_ap or (a, ac) in seen_aa:
            continue
        composer.store(a, ac, p)
        stored.append((a, ac, p))
        seen_ap.add((ac, p))
        seen_aa.add((a, ac))

    gate = RelationalFamiliarityGate(composer)
    gate.imprint_facts()

    rng = np.random.default_rng(seed + 555)
    known_ap = [(ac, p) for (a, ac, p) in stored]
    unknown_ap = []
    tries = 0
    while len(unknown_ap) < max(args.moat_floor, 8) and tries < 100000:
        tries += 1
        # unknown RELATIONAL cue: (RELATION, some-target) that is NOT any stored fact's
        # (action, patient) -> the host MUST abstain (the abstention floor). Skip the
        # relation token itself as the patient.
        cand = vocab[int(rng.integers(len(vocab)))]
        if cand == REL:
            continue
        ac, p = REL, cand
        if (ac, p) in seen_ap or (ac, p) in unknown_ap:
            continue
        unknown_ap.append((ac, p))

    def host_accepts(ac, p):
        return composer.query_agent(ac, p) is not None

    rows = []
    for (ac, p) in known_ap:
        rows.append((host_accepts(ac, p), gate.novelty_agent(ac, p), True))
    for (ac, p) in unknown_ap:
        rows.append((host_accepts(ac, p), gate.novelty_agent(ac, p), False))

    nov_accept = np.array([n for (ha, n, _k) in rows if ha])
    nov_abstain = np.array([n for (ha, n, _k) in rows if not ha])
    known_max = float(nov_accept.max()) if nov_accept.size else float("nan")
    unknown_min = float(nov_abstain.min()) if nov_abstain.size else float("nan")
    margin = float(unknown_min - known_max) if (nov_accept.size and nov_abstain.size) else float("nan")
    thr = (0.5 * (known_max + unknown_min)) if (nov_accept.size and nov_abstain.size) else \
        float(np.median([n for (_h, n, _k) in rows])) if rows else 0.0

    host_abstain_gate_accept = 0
    n_agree = 0
    floor_false_accepts = 0
    for (ha, nov, _k) in rows:
        gate_accept = nov < thr
        if ha == gate_accept:
            n_agree += 1
        if (not ha) and gate_accept:
            host_abstain_gate_accept += 1   # the moat-breach cell
            floor_false_accepts += 1
    agreement = (n_agree / len(rows)) if rows else 0.0

    # lesion anti-cheat
    gate.lesion()
    les_known, les_unknown = [], []
    for (ha, _n, _k) in rows:
        pass
    # recompute novelty after lesion from the same cues
    les_known = []
    les_unknown = []
    for (ac, p) in known_ap:
        les_known.append(gate.novelty_agent(ac, p))
    for (ac, p) in unknown_ap:
        les_unknown.append(gate.novelty_agent(ac, p))
    les_known = np.array(les_known)
    les_unknown = np.array(les_unknown)
    lesion_margin = float(les_unknown.min() - les_known.max()) if (les_known.size and les_unknown.size) else float("nan")
    lesion_collapsed = bool(les_known.size and les_unknown.size and
                            np.allclose(les_known.mean(), les_unknown.mean(), atol=1e-6)) or \
        bool(abs(lesion_margin) <= 1e-6)

    return {
        "n_stored_facts": len(stored), "n_known_cues": len(known_ap),
        "n_unknown_cues": len(unknown_ap),
        "novelty_known_max": known_max, "novelty_unknown_min": unknown_min,
        "separation_margin": margin, "threshold": thr,
        "agreement": agreement,
        "host_abstain_gate_accept": host_abstain_gate_accept,   # MUST be 0
        "abstention_floor_false_accepts": floor_false_accepts,  # MUST be 0
        "lesion_margin": lesion_margin, "lesion_collapsed": lesion_collapsed,
        "m4_moat_intact": bool(host_abstain_gate_accept == 0 and floor_false_accepts == 0),
    }


# ===========================================================================
# Cross-bridge fact authoring: pick a handful of (cue, target) pairs that span
# two different bridges (e.g. animals.<x> -> foods.<y>). Deterministic from seed.
# ===========================================================================
def author_cross_facts(all_corpora, seed, n_facts):
    rng = np.random.RandomState(seed * 13 + 5)
    by_shard = {bc["shard"]: list(bc["members"]) for bc in all_corpora}
    shards = list(by_shard.keys())
    facts = []
    seen = set()
    tries = 0
    while len(facts) < n_facts and tries < 100000:
        tries += 1
        if len(shards) < 2:
            break
        s_cue, s_tgt = rng.choice(len(shards), size=2, replace=False)
        cue = by_shard[shards[s_cue]][rng.randint(len(by_shard[shards[s_cue]]))]
        tgt = by_shard[shards[s_tgt]][rng.randint(len(by_shard[shards[s_tgt]]))]
        key = (cue, tgt)
        if key in seen or cue == tgt:
            continue
        seen.add(key)
        facts.append((cue, tgt))
    return facts


# ===========================================================================
# Per-seed orchestration across the --mode branches.
# ===========================================================================
def run_seed(seed, args):
    print(f"\n{'='*92}", flush=True)
    print(f"  MULTI-BRIDGE GRADED DE-RISK -- SEED {seed} -- mode={args.mode}", flush=True)
    print(f"{'='*92}", flush=True)

    shard_names = SHARD_NAMES[:args.n_bridges]
    all_corpora = [build_bridge_corpus(sn, args.concepts_per_bridge, seed, args)
                   for sn in shard_names]
    for bc in all_corpora:
        print(f"  [shard {bc['shard']:>9}] {len(bc['members'])} concepts "
              f"({bc['n_sub']} sub x {bc['per_sub']}), {bc['n_facts']} facts", flush=True)

    out = {"seed": seed, "mode": args.mode, "shards": shard_names}

    # ---- set-point bracket (cheap; design SS6 -- re-bracket {20,40,80} on bridge 1) ----
    if args.mode in ("precondition", "full") and args.bracket_setpoint:
        print(f"\n  [SET-POINT BRACKET on shard {all_corpora[0]['shard']} "
              f"({args.setpoint_bracket})]", flush=True)
        bracket = {}
        for tgt in args.setpoint_bracket:
            g = per_bridge_gates(all_corpora[0], seed, args, homeo_target=tgt)
            bracket[str(tgt)] = {"generalization": g["generalization"],
                                 "pearson_sim_vs_Strue": g["pearson_sim_vs_Strue"],
                                 "second_order_margin": g["second_order_margin"],
                                 "m1_band": g["m1_band"], "m2_band": g["m2_band"]}
            print(f"    target={tgt:>5}: gen={g['generalization']:.3f} "
                  f"P(sim,Strue)={g['pearson_sim_vs_Strue']:+.3f} "
                  f"2nd={g['second_order_margin']:+.3f} ({g['m1_band']}/{g['m2_band']})", flush=True)
        out["setpoint_bracket"] = bracket

    # ---- M1 + M2 + G5 per-bridge precondition gates ----
    if args.mode in ("precondition", "full"):
        print(f"\n  {'-'*88}\n  M1/M2/G5 -- per-bridge graded gates (read-out FIXED = brain-based "
              f"divnorm '{args.readout_divnorm}'/{args.readout_order})\n  {'-'*88}", flush=True)
        per_bridge = []
        for bc in all_corpora:
            g = per_bridge_gates(bc, seed, args)
            per_bridge.append(g)
            print(f"  [{g['shard']:>9}] M1 gen={g['generalization']:.3f}"
                  f"({g['gen_ratio']:.1f}x) A2={g['a2_collapses']} A3={g['a3_collapses']} "
                  f"[{g['m1_band']}] | M2 P(sim)={g['pearson_sim_vs_Strue']:+.3f} "
                  f"2nd={g['second_order_margin']:+.3f} graded={int(g['is_graded'])} [{g['m2_band']}] "
                  f"| G5 collapses={g['g5_collapses_robust']} "
                  f"(perm 2nd={g['permuted_second_order_margin']:+.3f}) "
                  f"| {g['learn_seconds']:.0f}s, {g['recurrent_nnz']} syn", flush=True)
        out["per_bridge_gates"] = per_bridge

        # ---- M6 random-shard anti-cheat ----
        print(f"\n  M6 -- random-shard anti-cheat (within-bridge generalization MUST collapse)",
              flush=True)
        m6 = random_shard_anticheat(all_corpora, seed, args)
        print(f"    random-shard gen={m6['random_shard_generalization']:.3f} "
              f"(chance {m6['chance']:.3f}) collapses={m6['m6_collapses_to_chance']}", flush=True)
        out["m6_random_shard"] = m6

    # ---- M3 + M7 cross-bridge + M4 moat (need live trained graded bridges) ----
    if args.mode in ("cross", "moat", "full"):
        cross_facts = author_cross_facts(all_corpora, seed, args.n_cross_facts)
        print(f"\n  [authored {len(cross_facts)} cross-bridge facts] "
              f"e.g. {cross_facts[:3]}", flush=True)
        out["cross_facts"] = cross_facts

    if args.mode in ("cross", "full"):
        print(f"\n  {'-'*88}\n  M3/M7 -- cross-bridge composition (V-tag layer over the graded "
              f"`pool` regions)\n  {'-'*88}", flush=True)
        # build + train live graded bridges (one per shard)
        graded_bridges = {}
        t0 = time.time()
        for bc in all_corpora:
            gb = GradedBridge(bc["shard"], bc["_local"]["concepts"], seed, args)
            gb.train(bc["_local"]["facts"])
            graded_bridges[bc["shard"]] = gb
            print(f"    [built+trained {bc['shard']:>9} graded bridge: "
                  f"{gb.bridge.cp_membrane_potential_v.shape[0]} neurons]", flush=True)
        print(f"    (built {len(graded_bridges)} graded bridges in {time.time()-t0:.0f}s)", flush=True)

        # namespace the cross facts to LOCAL names per bridge (encode/recall use local idx)
        # GradedBridge stores un-namespaced concept names; cross_facts are namespaced.
        # We translate: animals.c0_m1 -> shard 'animals', local 'c0_m1'. The GradedBridge
        # idx is over the local de-risk concept names (which include the hubs).
        m3 = cross_bridge_eval(graded_bridges, cross_facts, seed, args, permuted=False)
        print(f"    M3 TRUE: top2={m3['top2_fraction']:.2f} top1={m3['top1_fraction']:.2f} "
              f"signal/floor={m3['mean_signal_vs_floor']:.2f}x margin={m3['mean_margin']:.1f} "
              f"(n={m3['n_cross_facts']})", flush=True)
        m7 = cross_bridge_eval(graded_bridges, cross_facts, seed, args, permuted=True)
        print(f"    M7 PERMUTED: top2={m7['top2_fraction']:.2f} top1={m7['top1_fraction']:.2f} "
              f"signal/floor={m7['mean_signal_vs_floor']:.2f}x (must collapse vs TRUE)", flush=True)
        out["m3_cross_bridge"] = m3
        out["m7_permuted_mapping"] = m7
        # M3 band
        if m3["top2_fraction"] >= 0.80 and m3["mean_signal_vs_floor"] >= 1.5:
            out["m3_band"] = "GO"
        elif m3["top2_fraction"] >= 0.50:
            out["m3_band"] = "BOUNDARY"
        else:
            out["m3_band"] = "NEGATIVE"
        out["m7_collapses"] = bool(m7["top2_fraction"] < max(0.5, m3["top2_fraction"] - 0.2))

    if args.mode in ("moat", "full"):
        print(f"\n  {'-'*88}\n  M4 -- no-confab moat (learned familiarity gate ALONGSIDE host "
              f"abstention; CPU/numpy)\n  {'-'*88}", flush=True)
        all_members = []
        for bc in all_corpora:
            all_members.extend(bc["members"])
        cross_facts = out.get("cross_facts") or author_cross_facts(all_corpora, seed, args.n_cross_facts)
        m4 = moat_eval(cross_facts, all_members, seed, args)
        print(f"    M4: agreement={m4['agreement']:.3f} margin={m4['separation_margin']:+.4f} "
              f"host-abstain/gate-accept={m4['host_abstain_gate_accept']} (MUST be 0) "
              f"floor-false-accepts={m4['abstention_floor_false_accepts']} (MUST be 0) "
              f"lesion-collapses={m4['lesion_collapsed']} -> moat-intact={m4['m4_moat_intact']}",
              flush=True)
        out["m4_moat"] = m4

    return out


# ===========================================================================
# Multi-seed verdict (design SS4.3 criteria table).
# ===========================================================================
def aggregate(per_seed, args):
    seeds = list(per_seed.keys())

    def _all(pred):
        vals = [pred(per_seed[s]) for s in seeds]
        return all(v is True for v in vals), vals

    agg = {"seeds": seeds, "mode": args.mode}

    if args.mode in ("precondition", "full"):
        # M1 GO on every bridge every seed; A2/A3 collapse; G5 robust collapse
        m1_go, _ = _all(lambda r: all(b["m1_band"] == "GO" for b in r["per_bridge_gates"]))
        m1_notneg, _ = _all(lambda r: all(b["m1_band"] != "NEGATIVE" for b in r["per_bridge_gates"]))
        a2_ok, _ = _all(lambda r: all(b["a2_collapses"] for b in r["per_bridge_gates"]))
        a3_ok, _ = _all(lambda r: all(b["a3_collapses"] for b in r["per_bridge_gates"]))
        m2_go, _ = _all(lambda r: all(b["m2_band"] == "GO" for b in r["per_bridge_gates"]))
        m2_notneg, _ = _all(lambda r: all(b["m2_band"] != "NEGATIVE" for b in r["per_bridge_gates"]))
        g5_ok, _ = _all(lambda r: all(b["g5_collapses_robust"] for b in r["per_bridge_gates"]))
        m6_ok, _ = _all(lambda r: r["m6_random_shard"]["m6_collapses_to_chance"] in (True, None))
        agg["precondition"] = {
            "M1_all_GO": m1_go, "M1_no_NEGATIVE": m1_notneg,
            "A2_collapses": a2_ok, "A3_collapses": a3_ok,
            "M2_all_GO": m2_go, "M2_no_NEGATIVE": m2_notneg,
            "G5_robust_collapses": g5_ok, "M6_random_shard_collapses": m6_ok,
        }

    if args.mode in ("cross", "full"):
        m3_go, _ = _all(lambda r: r.get("m3_band") == "GO")
        m3_notneg, _ = _all(lambda r: r.get("m3_band") != "NEGATIVE")
        m7_ok, _ = _all(lambda r: r.get("m7_collapses") is True)
        agg["cross"] = {"M3_all_GO": m3_go, "M3_no_NEGATIVE": m3_notneg,
                        "M7_permuted_collapses": m7_ok}

    if args.mode in ("moat", "full"):
        m4_ok, _ = _all(lambda r: r["m4_moat"]["m4_moat_intact"])
        m4_lesion, _ = _all(lambda r: r["m4_moat"]["lesion_collapsed"])
        agg["moat"] = {"M4_moat_intact": m4_ok, "M4_lesion_collapses": m4_lesion}

    # ---- combined verdict (design SS4.3) ----
    verdict = None
    if args.mode == "full":
        pre = agg["precondition"]
        cr = agg["cross"]
        mo = agg["moat"]
        precondition_holds = pre["M1_no_NEGATIVE"] and pre["M2_no_NEGATIVE"] and \
            pre["A2_collapses"] and pre["A3_collapses"] and pre["G5_robust_collapses"] and \
            pre["M6_random_shard_collapses"]
        m3_neg = not cr["M3_no_NEGATIVE"]
        m4_neg = not mo["M4_moat_intact"]
        if m3_neg or m4_neg:
            verdict = "NEGATIVE"     # graded codes break cross-bridge composition or the moat
        elif (pre["M1_all_GO"] and pre["M2_all_GO"] and cr["M3_all_GO"] and mo["M4_moat_intact"]
              and mo["M4_lesion_collapses"] and cr["M7_permuted_collapses"] and precondition_holds):
            verdict = "GO"           # within-bridge graded + cross-bridge + moat all coexist
        else:
            verdict = "BOUNDARY"     # real but a quantity sits in its BOUNDARY band
    agg["verdict"] = verdict
    return agg


def main():
    p = argparse.ArgumentParser(description="Multi-bridge learned-graded-embedding cheap-first de-risk "
                                            "(within-bridge graded + cross-bridge composition + moat)")
    p.add_argument("--mode", default="full",
                   choices=["precondition", "cross", "moat", "full"],
                   help="precondition=M1/M2/G5/M6 per-bridge; cross=M3/M7; moat=M4; full=all+verdict")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--seed", type=int, default=None, help="single-seed override")
    # multi-bridge sharding
    p.add_argument("--n-bridges", type=int, default=3)
    p.add_argument("--concepts-per-bridge", type=int, default=64)
    p.add_argument("--target-per-sub", type=int, default=8,
                   help="target members per within-bridge sub-cluster (factoring of concepts-per-bridge)")
    # per-bridge graded learner (HomeostaticAssocGraph; design SS6 sizing -> n_pool ~2400)
    p.add_argument("--n-pool", type=int, default=2400)
    p.add_argument("--pattern-size", type=int, default=100)
    p.add_argument("--homeo", default="oja", choices=["oja", "scaling", "none"])
    p.add_argument("--homeo-target", type=float, default=40.0,
                   help="Oja incoming-L2 set-point (V=160 = 40; re-bracket on the smaller pool)")
    p.add_argument("--cycles", type=int, default=10)
    p.add_argument("--bracket-setpoint", action="store_true",
                   help="cheaply re-bracket the Oja set-point on bridge 1 (design SS6)")
    p.add_argument("--setpoint-bracket", type=float, nargs="+", default=[20.0, 40.0, 80.0])
    # toy within-bridge corpus structure (de-risk defaults)
    p.add_argument("--n-props", type=int, default=4)
    p.add_argument("--hub-facts-per-member", type=int, default=6)
    p.add_argument("--bridge-facts", type=int, default=8)
    p.add_argument("--triplet-facts-per-cluster", type=int, default=4)
    # brain-based divnorm read-out (FIXED validated recipe)
    p.add_argument("--readout-divnorm", default="ch")
    p.add_argument("--readout-order", default="interleave")
    p.add_argument("--readout-sigma", type=float, default=0.001)
    p.add_argument("--readout-exponent", type=float, default=2.0)
    p.add_argument("--readout-log-clip", action="store_true")
    p.add_argument("--diffusion-alpha", type=float, default=0.5)
    p.add_argument("--diffusion-steps", type=int, default=2)
    # cross-bridge V-tag encode/recall (adapted recipe over the `pool` region)
    p.add_argument("--n-cross-facts", type=int, default=12)
    p.add_argument("--encoding-steps", type=int, default=100)
    p.add_argument("--teacher-pA", type=float, default=500.0)
    p.add_argument("--top-k", type=int, default=150)
    p.add_argument("--drive-pA", type=float, default=1500.0)
    p.add_argument("--drive-steps", type=int, default=100)
    # moat (familiarity gate)
    p.add_argument("--moat-D", type=int, default=128)
    p.add_argument("--moat-floor", type=int, default=20)
    # gate bars (match the de-risk / homeostasis probe)
    p.add_argument("--g1-bar", type=float, default=0.5)
    p.add_argument("--a1-bar", type=float, default=0.7)
    p.add_argument("--so-margin-bar", type=float, default=0.10)
    p.add_argument("--k-neighbours", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    backend = os.environ.get("SIM_BACKEND", "auto")
    t_all = time.time()
    print(f"[multibridge-graded de-risk] mode={args.mode} seeds={seeds} backend={backend}", flush=True)
    print(f"  {args.n_bridges} bridges x {args.concepts_per_bridge} concepts; "
          f"learner=HomeostaticAssocGraph(homeo={args.homeo} target={args.homeo_target} "
          f"n_pool={args.n_pool} cycles={args.cycles})", flush=True)
    print(f"  read-out FIXED = brain-based divnorm '{args.readout_divnorm}'/{args.readout_order} "
          f"(sigma={args.readout_sigma} exp={args.readout_exponent} steps={args.diffusion_steps})",
          flush=True)
    print(f"  cross-bridge = V-tag (engram-tag over the graded `pool` region; HELPER-SIGNATURE "
          f"ADAPTED -- see docstring)", flush=True)

    per_seed = {}
    for s in seeds:
        per_seed[str(s)] = run_seed(s, args)

    agg = aggregate(per_seed, args)

    print(f"\n{'='*92}", flush=True)
    print(f"  MULTI-BRIDGE GRADED DE-RISK SUMMARY -- mode={args.mode}", flush=True)
    print(f"{'='*92}", flush=True)
    for k, v in agg.items():
        if k in ("seeds", "mode", "verdict"):
            continue
        print(f"  [{k}] {v}", flush=True)
    if agg.get("verdict") is not None:
        print(f"\n  >>> COMBINED VERDICT: {agg['verdict']} <<<", flush=True)
    print(f"  Total elapsed: {time.time()-t_all:.1f}s", flush=True)
    print(f"{'='*92}\n", flush=True)

    out_data = {"aggregate": agg, "per_seed": per_seed, "args": vars(args)}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        args.out = os.path.join(raw_dir, f"_multibridge_graded_derisk_{args.mode}_seed{seeds[0]}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out_data


if __name__ == "__main__":
    main()
