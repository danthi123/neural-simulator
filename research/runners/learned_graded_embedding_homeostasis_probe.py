"""LEARNED GRADED-EMBEDDING HOMEOSTASIS PROBE -- does a SELF-REGULATING BIOLOGICAL HOMEOSTATIC
recurrent give CYCLE-INDEPENDENT faithfulness, so the dual/CLS learned-embedding is robust WITHOUT
hand-picking cycles=2?

CONTEXT (the last open mechanism question):
  The brain-based learned-embedding is fully de-risked + fully brain-based end-to-end
  (2026-06-11-learned-graded-embedding-confirm-GO_full.md + -divnorm-readout-GO.md; commits
  e6e277e3 + 9fa90d74) at a HAND-PICKED operating point: the un-normalized excitatory Hebbian
  recurrent (LearnedAssocGraph) SATURATES with cycles -- Pearson(W, raw_counts) +0.69 @2cyc ->
  +0.06 @20cyc -- so the recipe uses cycles=2. A multiplicative gamma-decay arm (the desaturate
  probe) was REAL but INFERIOR (gamma=0.95 -> Pearson +0.43/gen 0.77; gamma=0.9 over-decays). Both
  are HAND-PICKED operating points, not a self-regulating mechanism.

THE DE-RISK (this runner):
  Does a proper BIOLOGICAL HOMEOSTATIC mechanism give CYCLE-INDEPENDENT faithfulness (W tracks the
  co-occurrence counts AND the graded structure + generalization hold ACROSS cycle counts {2..40}
  AND under store-volume stress)? The two canonical bounded-Hebbian mechanisms real neurons have
  (they prevent runaway potentiation -> adding them makes the learner MORE realistic):

    (c) OJA'S RULE (Oja 1982): Hebbian + the -y^2 w normalization. The fixed-point of Oja is the
        weight vector normalized to UNIT L2 norm. Equivalently (and the form we apply runner-side,
        post-cycle): renormalize each POSTSYNAPTIC neuron's INCOMING pool<->pool weight vector to a
        fixed target L2 norm. This is the canonical bounded-Hebbian -- it preserves the RELATIVE
        co-occurrence structure (the direction of the weight vector) while bounding the magnitude,
        so it should NOT saturate toward a uniform floor as cycles accumulate.

    (d) SYNAPTIC SCALING (Turrigiano): multiplicatively rescale each neuron's TOTAL incoming
        pool<->pool synaptic input to a fixed target (L1 / sum). Homeostatic: preserves relative
        weights, normalizes the total drive -> de-correlates "how much a neuron has learned" from
        "how active it was", which is exactly the runaway-potentiation the un-normalized recurrent
        suffers.

  Both are applied POST-CYCLE, restricted to the pool<->pool edges, grouped by the POSTSYNAPTIC
  column (cp_connections is (pre->post) layout: cp_connections[i,j] = weight i->j, so neuron j's
  INCOMING weights = the j-th COLUMN). Runner-side (NO sim/ edits): a per-column rescale of
  cp_connections.data restricted to the pool<->pool mask. NEITHER fits any parameter to S_true --
  the target norm / total is a FIXED homeostatic set-point.

THE EXPERIMENT (multi-seed 42/43/44; GPU; FOREGROUND):
  1. VARIANTS: (a) un-normalized (the saturating baseline; reproduce +0.69->+0.06);
     (b) gamma-decay (the inferior reference, gamma=0.95); (c) OJA-style (per-post-neuron incoming
     L2 renorm to target, sweep target); (d) SYNAPTIC SCALING (per-post-neuron incoming L1/sum to
     target, sweep target).
  2. CYCLE-COUNT SWEEP: for each variant, learn at cycles in {2,5,10,20,40}. CORE METRIC: is
     Pearson(W, raw_counts) + Pearson(sim_W, S_true) [via the brain-based divnorm read-out] + G2
     generalization + the G1 2nd-order cosine margin CYCLE-INDEPENDENT (hold at/near the cycles=2
     level across all cycle counts) -- vs the un-normalized which collapses by cycle 20?
  3. STORE-VOLUME / SCALE STRESS: at the best homeostatic variant, increase the store volume
     (more repetitions per fact -- a larger corpus) at a FIXED cycle count -> does faithfulness
     HOLD as the store accumulates (the production concern: many facts shouldn't saturate the
     recurrent)?
  4. GATE RE-CONFIRM at the best cycle-independent operating point: G1 (graded + 2nd-order margin
     >= +0.10), G2 (generalization 1.000), with the FULLY BRAIN-BASED divnorm read-out (the
     validated divnorm recipe). Does the homeostatic learn still pass?

  READ-OUT: the FULLY BRAIN-BASED divisive-normalization read-out (spreading-activation + Carandini-
  Heeger divisive normalization), the validated recipe from 2026-06-11-learned-graded-embedding-
  divnorm-readout-GO.md. We FIX the read-out and test the LEARN, so the variation measured is the
  homeostatic recurrent's, not the read-out's. (We also report the host-method-on-W stand-in Pearson
  as a secondary number, since that is the de-risk's primary; but the gates run on the brain-based
  divnorm read-out.)

DECISION LOGIC (stated explicitly):
  GO (cycle-independent)   if a biological homeostatic mechanism (Oja and/or synaptic scaling) gives
                           CYCLE-INDEPENDENT faithfulness (W<->counts + graded + generalization hold
                           across cycles {2..40} AND under store-volume stress, all near/above the
                           cycles=2 level) AND the gates still pass with the brain-based read-out,
                           multi-seed. -> the last open mechanism unknown is RETIRED; the build is
                           robust without hand-picking cycles -> the build can start with ZERO open
                           mechanism risks. Report the recommended homeostatic recipe.
  BOUNDARY                 if homeostasis helps but degrades at high cycles or under store-volume
                           stress -> characterize precisely (cycles=2/low-cycle stays the operating
                           constraint = a documented build constraint, not a blocker; the gate
                           already passes there). No banking.

ANTI-CHEATS (all mandatory):
  - Pearson(W, raw_counts) < 0.99 (genuine learning, not pass-through); the homeostatic rule is a
    FIXED mechanism (no fitting the target norm/total to S_true); the permuted-S baseline ~0; the
    G2 controls (orthogonal A2 + permuted-property A3) collapse; generalization stays 1.000. NO
    sim/ edits (runner-side weight normalization).

Run (GPU, FOREGROUND -- each spiking-learn is ~30-120 s inline depending on cycles; the read-out is
pure numpy; NO background):
  SIM_BACKEND=cupy python -m research.runners.learned_graded_embedding_homeostasis_probe \
      --seeds 42,43,44 --out research/findings/raw/_lge_homeostasis_multiseed.json
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

# Reuse the de-risk's corpus + structure-recovery + host-ceiling + generalization harnesses VERBATIM;
# the desaturate probe's DesaturatingAssocGraph (the post-cycle weight hook + pool<->pool data-mask)
# and the host-method-on-W read-out (the de-risk's primary, kept as a secondary number); and the
# divnorm probe's FULLY BRAIN-BASED divisive-normalization read-out (the validated recipe -- we FIX
# this read-out and test the LEARN). This runner ADDS the Oja + synaptic-scaling homeostatic arms and
# sweeps (variant x cycles) + a store-volume stress + the gate re-confirm.
from research.runners.learned_graded_embedding_derisk_probe import (  # noqa: E402
    build_toy_cooccurrence,
    permute_corpus,
    structure_recovery,
    host_ceiling_codes,
    random_gaussian_codes,
    architecture_generalization,
)
from research.runners.learned_graded_embedding_diagnose import (  # noqa: E402
    raw_count_matrix,
    offdiag_pearson,
)
from research.runners.learned_graded_embedding_desaturate_probe import (  # noqa: E402
    DesaturatingAssocGraph,
    host_method_codes_on_W,
)
from research.runners.learned_graded_embedding_divnorm_readout_probe import (  # noqa: E402
    divnorm_spreading_readout,
)
from research.runners.dual_cls_architecture_proof_probe import (  # noqa: E402
    codebook_similarity_stats,
    assign_properties,
    run_generalization,
)


# ===========================================================================
# The HOMEOSTATIC learner: DesaturatingAssocGraph + a per-POST-neuron biological homeostatic rule
# applied POST-CYCLE to the pool<->pool recurrent. cp_connections is (pre->post): cp_connections[i,j]
# = weight i->j, so neuron j's INCOMING weights = the j-th COLUMN. Oja + synaptic scaling both
# normalize the INCOMING weight vector per postsynaptic neuron -> we group the pool<->pool data
# entries by their POSTSYNAPTIC column index and rescale each group.
# ===========================================================================
class HomeostaticAssocGraph(DesaturatingAssocGraph):
    """LearnedAssocGraph + (optionally) a BIOLOGICAL HOMEOSTATIC rule on the pool<->pool recurrent,
    applied ONCE per fact (the natural between-experiences synaptic-scaling timescale, mirroring the
    desaturate arm's per-fact gamma) -- BUT here we want the homeostasis applied per-CYCLE inside a
    fact (the homeostatic set-point should hold throughout learning, not only at fact boundaries), so
    we expose `homeo` and apply it after EACH co-fire cycle. Two rules:

      'oja'    : per-postsynaptic-neuron INCOMING pool<->pool weight vector renormalized to a fixed
                 target L2 norm (the Oja-1982 fixed-point form: w <- w * (target / ||w||_2)). Only
                 neurons whose incoming norm EXCEEDS the target are scaled DOWN (the bound is a
                 ceiling; below-target vectors are left to grow -- the homeostatic set-point is an
                 upper bound on total potentiation, matching real synaptic scaling which down-scales
                 over-potentiated inputs). [oja_clip_only=True; if False, scale to EXACTLY target.]
      'scaling': per-postsynaptic-neuron INCOMING pool<->pool total (L1 / sum) rescaled to a fixed
                 target (Turrigiano: w <- w * (target / sum(w))). Same clip-only-above-target option.

    NEITHER fits a parameter to S_true (the target is a fixed homeostatic set-point). gamma<1 + the
    old cap path are inherited from DesaturatingAssocGraph (the inferior reference). NO sim/ edits.
    """

    def __init__(self, concepts, seed=42, n_pool=2000, pattern_size=100,
                 gamma=1.0, cap=None, homeo="none", homeo_target=1.0,
                 homeo_clip_only=True):
        super().__init__(concepts, seed=seed, n_pool=n_pool, pattern_size=pattern_size,
                         gamma=gamma, cap=cap)
        self.homeo = str(homeo)
        self.homeo_target = float(homeo_target)
        self.homeo_clip_only = bool(homeo_clip_only)
        # Precompute, ONCE, the postsynaptic column index of every pool<->pool data entry, so the
        # per-post-neuron group-reduce is a fast segmented op on the backend. We map post columns to
        # DENSE pool-local group ids [0..Npool) so the per-column reduce is a fixed-size scatter.
        self._prep_homeo_indexers()

    def _prep_homeo_indexers(self):
        """Build (on the backend): for each pool<->pool data entry, its DENSE pool-local postsynaptic
        group id. cp_connections is (pre->post): the COLUMN index `indices[k]` of data entry k is the
        postsynaptic neuron. We restrict to the pool<->pool mask and map the post column -> a dense id
        in [0, Npool) so a segment-sum / segment-max over `post_group` gives per-post-neuron stats."""
        cp = self._cp
        M = self.bridge.cp_connections
        # host copies (built once; small relative to the run)
        indices = np.asarray(self._to_host(M.indices))
        mask_h = np.asarray(self._to_host(self._pool_data_mask)).astype(bool)
        pool = np.asarray(self.pool_base)
        pmin = int(pool.min())
        Npool = int(pool.max()) - pmin + 1
        post_cols = indices[mask_h]                       # postsynaptic column per masked entry
        post_group = (post_cols - pmin).astype(np.int64)  # dense pool-local id [0, Npool)
        assert post_group.min() >= 0 and post_group.max() < Npool, "post group id out of pool range"
        self._homeo_Npool = Npool
        self._homeo_post_group = cp.asarray(post_group)   # [n_masked]
        self._homeo_n_masked = int(post_group.size)
        # index of the masked entries within the full data array (so we can scatter back).
        self._homeo_data_idx = cp.asarray(np.nonzero(mask_h)[0])

    def _apply_homeo(self):
        """Apply the chosen homeostatic rule to the pool<->pool recurrent (per postsynaptic neuron)."""
        if self.homeo == "none":
            return
        cp = self._cp
        data = self.bridge.cp_connections.data
        idx = self._homeo_data_idx
        grp = self._homeo_post_group
        Npool = self._homeo_Npool
        w = data[idx]                                     # the pool<->pool weights (masked)
        wpos = cp.maximum(w, cp.float32(0.0))             # excitatory recurrent (non-negative)

        if self.homeo == "oja":
            # per-post-neuron L2 norm of the incoming weight vector
            sq = cp.zeros(Npool, dtype=cp.float64)
            cp.add.at(sq, grp, (wpos.astype(cp.float64)) ** 2)
            norm = cp.sqrt(sq) + 1e-12                     # [Npool]
            target = cp.float64(self.homeo_target)
            factor = target / norm                         # scale each post-neuron group to target L2
        elif self.homeo == "scaling":
            # per-post-neuron total incoming drive (L1 / sum)
            tot = cp.zeros(Npool, dtype=cp.float64)
            cp.add.at(tot, grp, wpos.astype(cp.float64))
            tot = tot + 1e-12
            target = cp.float64(self.homeo_target)
            factor = target / tot                          # scale each post-neuron group's sum to target
        else:
            raise ValueError(f"unknown homeo rule '{self.homeo}'")

        if self.homeo_clip_only:
            # only DOWN-scale groups that EXCEED the target (the set-point is a ceiling; below-target
            # vectors keep growing) -- factor capped at 1.0.
            factor = cp.minimum(factor, cp.float64(1.0))
        fac_per_entry = factor[grp].astype(cp.float32)     # broadcast group factor to each entry
        data[idx] = (w * fac_per_entry)                    # rescale in place (sign preserved)

    def store_fact(self, concept_list, cycles=20):
        """Co-fire the fact's concept patterns (LearnedAssocGraph's Hebbian growth), applying the
        homeostatic rule AFTER EACH co-fire cycle (the set-point holds throughout learning). gamma/cap
        are applied ONCE at the end (inherited inferior reference). homeo='none' + gamma=1 + cap=None
        == the original un-normalized learner."""
        from research.runners._D_sparse_heteroassoc import _drive
        cp = self._cp
        ids = [self.lag.idx[c] for c in concept_list if c in self.lag.idx]
        if len(ids) < 2:
            return
        try:
            self.bridge.set_plasticity_gate("recurrent", 1.0)
        except KeyError:
            pass
        drive = [self.lag.pg[i] for i in ids]
        for _ in range(cycles):
            _drive(self.bridge, drive, 1100.0)
            for _ in range(10):
                self.bridge._run_one_simulation_step()
            self.bridge.cp_external_input_current[:] = 0.0
            for _ in range(5):
                self.bridge._run_one_simulation_step()
            # ---- BIOLOGICAL HOMEOSTASIS (per cycle), pool<->pool incoming-per-post-neuron ----
            self._apply_homeo()
        # ---- inferior reference arms (gamma decay / cap), ONCE per fact ----
        if (self.gamma < 1.0) or (self.cap is not None):
            data = self.bridge.cp_connections.data
            m = self._pool_data_mask
            if self.gamma < 1.0:
                data[m] = data[m] * self.gamma
            if self.cap is not None:
                data[m] = cp.minimum(data[m], cp.float32(self.cap))
        try:
            self.bridge.set_plasticity_gate("recurrent", 0.0)
        except KeyError:
            pass


def learn_W_homeostatic(concepts, facts, seed, n_pool, pattern_size, cycles,
                        gamma=1.0, cap=None, homeo="none", homeo_target=1.0,
                        homeo_clip_only=True):
    """Learn the brain-based co-occurrence recurrent with the chosen biological homeostatic rule, then
    read it as the concept->concept association matrix W [Nc, Nc] (mean a->b recurrent weight between
    each pair's sparse patterns -- the SAME extraction the de-risk/diagnose/desaturate use).
    Returns (W, info)."""
    from sim.backend import to_host
    hag = HomeostaticAssocGraph(concepts, seed=seed, n_pool=n_pool, pattern_size=pattern_size,
                                gamma=gamma, cap=cap, homeo=homeo, homeo_target=homeo_target,
                                homeo_clip_only=homeo_clip_only)
    for f in facts:
        hag.store_fact(list(f), cycles=cycles)
    M = to_host(hag.bridge.cp_connections)
    pb = hag.pool_base
    sub = M[pb][:, pb]
    dense = np.asarray(sub.todense())
    Nc = len(concepts)
    W = np.zeros((Nc, Nc), dtype=np.float64)
    pats = [np.asarray(p) for p in hag.patterns]
    for a in range(Nc):
        for b in range(Nc):
            if a == b:
                continue
            W[a, b] = float(dense[np.ix_(pats[a], pats[b])].mean())
    data = np.asarray(sub.data)
    info = {
        "recurrent_mean": float(np.abs(data).mean()) if data.size else 0.0,
        "recurrent_max": float(np.abs(data).max()) if data.size else 0.0,
        "recurrent_nnz": int(data.size),
        "n_neurons": int(hag.bridge.cp_membrane_potential_v.shape[0]),
        "cycles": int(cycles), "gamma": float(gamma),
        "cap": (float(cap) if cap is not None else None),
        "homeo": str(homeo), "homeo_target": float(homeo_target),
        "homeo_clip_only": bool(homeo_clip_only),
    }
    return W, info


# ===========================================================================
# Store-volume stress: repeat each fact `reps` times in the corpus (a larger store). The raw counts
# scale linearly with reps; the co-occurrence STRUCTURE (S_true) is invariant. A faithful homeostatic
# recurrent should keep Pearson(W, raw_counts) + the graded structure HOLDING as reps grows; the
# un-normalized one should saturate FASTER (more co-fire -> more uniform-floor fill).
# ===========================================================================
def inflate_corpus(facts, reps):
    """Repeat the corpus `reps` times (each fact stored `reps` times). reps=1 == the base corpus."""
    out = []
    for _ in range(int(reps)):
        out.extend(list(facts))
    return out


# ===========================================================================
# Brain-based divnorm read-out at a FIXED validated recipe (the divnorm-GO recipe). We FIX the read-
# out and vary the LEARN, so the measured variation is the homeostatic recurrent's. We pick the
# 'marginal' (PPMI-marginal-division analogue) divnorm + 'post' order at the validated steps; this is
# the canonical brain-based read-out. We pass the chosen recipe through divnorm_spreading_readout.
# ===========================================================================
def brain_based_codes(W, member_rows, args):
    return divnorm_spreading_readout(
        W, member_rows,
        divnorm=args.readout_divnorm, order=args.readout_order,
        sigma=args.readout_sigma, exponent=args.readout_exponent,
        alpha=args.diffusion_alpha, steps=args.diffusion_steps,
        log_clip=args.readout_log_clip)


def measure_point(W, concepts, members, member_rows, S_true, second_order_pairs, labels, props,
                  C_full, nclu, pclu, seed, args, chance):
    """Given a learned W, compute the cycle-independence + gate numbers, using the FIXED brain-based
    divnorm read-out (the validated recipe). Reports:
      - Pearson(W, raw_counts)            : does the learned W track the co-occurrence COUNTS?
      - Pearson(sim, S_true) [brain-based]: does the brain-based read-out recover the graded structure?
      - 2nd-order cat~dog margin, graded?, generalization (the G2 graded acc)
      - host-method-on-W Pearson (the de-risk's primary; secondary here)
    """
    pearson_W_counts = offdiag_pearson(W, C_full)
    W_distinct = bool(pearson_W_counts < 0.99)

    # PRIMARY (this de-risk): the FULLY BRAIN-BASED divnorm read-out.
    bb_codes = brain_based_codes(W, member_rows, args)
    bb_rec = structure_recovery(bb_codes, S_true, second_order_pairs, seed)
    bb_grad = codebook_similarity_stats(bb_codes, labels)
    bb_gen = float(run_generalization(bb_codes, labels, props, nclu, pclu, seed,
                                      args.k_neighbours)["accuracy"])

    # SECONDARY: the host-method-on-W stand-in (the de-risk's primary read-out).
    sw_codes = host_method_codes_on_W(W, member_rows, args.svd_dim)
    sw_rec = structure_recovery(sw_codes, S_true, second_order_pairs, seed)

    return {
        "pearson_W_vs_rawcounts": pearson_W_counts,
        "W_distinct_from_counts": W_distinct,
        # brain-based read-out (PRIMARY)
        "bb_pearson_vs_Strue": bb_rec["pearson_learned_vs_Strue"],
        "bb_pearson_permutedS": bb_rec["pearson_permuted_vs_Strue"],
        "bb_is_graded": bool(bb_grad["is_graded"]),
        "bb_second_order_margin": bb_rec["second_order_margin"],
        "bb_second_order_recovered": bool(bb_rec["second_order_recovered"]),
        "bb_generalization": bb_gen,
        # host-method-on-W (SECONDARY)
        "standin_pearson_vs_Strue": sw_rec["pearson_learned_vs_Strue"],
        "standin_second_order_margin": sw_rec["second_order_margin"],
    }


def _fmt(tag, m):
    return (f"    [{tag:22s}] P(W,counts)={m['pearson_W_vs_rawcounts']:+.3f} || "
            f"BRAIN-BASED: P(sim,Strue)={m['bb_pearson_vs_Strue']:+.3f} "
            f"2nd={m['bb_second_order_margin']:+.3f} graded={int(m['bb_is_graded'])} "
            f"gen={m['bb_generalization']:.3f} | standin P={m['standin_pearson_vs_Strue']:+.3f}")


# ===========================================================================
# Cycle-independence metric: across the cycle sweep, how much does each number VARY? A cycle-
# independent variant holds Pearson(W,counts) + Pearson(sim,Strue) + gen flat (low spread) and ABOVE
# the floor; the un-normalized COLLAPSES (large negative slope w/ cycles).
# ===========================================================================
def cycle_independence_stats(points_by_cycle, cycles):
    """points_by_cycle: {cycle:int -> measure dict}. Returns spread + min + slope summaries for the
    key numbers across the cycle sweep."""
    cyc = sorted(points_by_cycle.keys())
    pcounts = np.array([points_by_cycle[c]["pearson_W_vs_rawcounts"] for c in cyc])
    pstrue = np.array([points_by_cycle[c]["bb_pearson_vs_Strue"] for c in cyc])
    gen = np.array([points_by_cycle[c]["bb_generalization"] for c in cyc])
    margin = np.array([points_by_cycle[c]["bb_second_order_margin"] for c in cyc])
    cyc_arr = np.array(cyc, dtype=float)

    def _slope(y):
        if len(cyc_arr) < 2 or np.std(cyc_arr) < 1e-12:
            return 0.0
        return float(np.polyfit(cyc_arr, y, 1)[0])

    return {
        "cycles": cyc,
        "pearson_W_counts_per_cycle": pcounts.tolist(),
        "pearson_W_counts_min": float(pcounts.min()),
        "pearson_W_counts_max": float(pcounts.max()),
        "pearson_W_counts_spread": float(pcounts.max() - pcounts.min()),
        "pearson_W_counts_slope_per_cycle": _slope(pcounts),
        "bb_pearson_Strue_per_cycle": pstrue.tolist(),
        "bb_pearson_Strue_min": float(pstrue.min()),
        "bb_pearson_Strue_spread": float(pstrue.max() - pstrue.min()),
        "bb_pearson_Strue_slope_per_cycle": _slope(pstrue),
        "bb_generalization_per_cycle": gen.tolist(),
        "bb_generalization_min": float(gen.min()),
        "bb_second_order_margin_per_cycle": margin.tolist(),
        "bb_second_order_margin_min": float(margin.min()),
    }


# ===========================================================================
# Per-seed driver
# ===========================================================================
def run_seed(seed: int, args) -> dict:
    print(f"\n{'='*86}", flush=True)
    print(f"  LEARNED GRADED-EMBEDDING HOMEOSTASIS PROBE -- SEED {seed}", flush=True)
    print(f"{'='*86}", flush=True)

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
    print(f"  [HOST CEILING (PPMI+SVD on RAW counts, labelled target)] "
          f"Pearson(S,S_true)={host_rec['pearson_learned_vs_Strue']:+.3f} "
          f"2nd-margin={host_rec['second_order_margin']:+.3f} gen={host_gen:.3f}", flush=True)

    rand_codes = random_gaussian_codes(Nm, Nm, seed)
    rand_gen = float(run_generalization(rand_codes, labels, props, nclu, pclu, seed,
                                        args.k_neighbours)["accuracy"])

    # =========================================================================
    # VARIANT x CYCLE-COUNT sweep -- the CORE cycle-independence test.
    # =========================================================================
    # variant spec: (key, dict-of-kwargs-to-learn_W_homeostatic)
    variants = []
    variants.append(("unnormalized", dict(gamma=1.0, cap=None, homeo="none")))
    variants.append((f"gamma{args.gamma_ref}", dict(gamma=args.gamma_ref, cap=None, homeo="none")))
    for tgt in args.oja_targets:
        variants.append((f"oja_t{tgt}", dict(gamma=1.0, cap=None, homeo="oja",
                                             homeo_target=tgt, homeo_clip_only=True)))
    for tgt in args.scaling_targets:
        variants.append((f"scaling_t{tgt}", dict(gamma=1.0, cap=None, homeo="scaling",
                                                 homeo_target=tgt, homeo_clip_only=True)))

    print(f"\n  {'-'*82}", flush=True)
    print(f"  VARIANT x CYCLE sweep: variants={[v[0] for v in variants]} x cycles={args.cycles_sweep}",
          flush=True)
    print(f"  (read-out FIXED = brain-based divnorm '{args.readout_divnorm}'/{args.readout_order} "
          f"steps={args.diffusion_steps})", flush=True)
    print(f"  {'-'*82}", flush=True)

    sweep = {}            # {variant_key: {cycle: measure}}
    cycindep = {}         # {variant_key: cycle_independence_stats}
    for vkey, vkw in variants:
        print(f"\n  >>> variant {vkey}", flush=True)
        by_cycle = {}
        for cyc in args.cycles_sweep:
            t0 = time.time()
            W, info = learn_W_homeostatic(concepts, corpus["facts"], seed, args.n_pool,
                                          args.pattern_size, cyc, **vkw)
            m = measure_point(W, concepts, members, member_rows, S_true, second_order_pairs, labels,
                              props, C_full, nclu, pclu, seed, args, chance)
            m["learn_info"] = info
            m["learn_seconds"] = time.time() - t0
            by_cycle[cyc] = m
            print(f"  cycles={cyc:3d} (rec mean={info['recurrent_mean']:.3f}, {m['learn_seconds']:.0f}s)",
                  flush=True)
            print(_fmt(f"{vkey} c{cyc}", m), flush=True)
        sweep[vkey] = {str(c): by_cycle[c] for c in by_cycle}
        ci = cycle_independence_stats(by_cycle, args.cycles_sweep)
        cycindep[vkey] = ci
        print(f"    [cycle-independence {vkey}] P(W,counts) {ci['pearson_W_counts_per_cycle']} "
              f"(spread {ci['pearson_W_counts_spread']:.3f}, slope {ci['pearson_W_counts_slope_per_cycle']:+.4f}/cyc) "
              f"| P(sim,Strue) min {ci['bb_pearson_Strue_min']:+.3f} spread {ci['bb_pearson_Strue_spread']:.3f} "
              f"| gen min {ci['bb_generalization_min']:.3f}", flush=True)

    # =========================================================================
    # Choose the BEST homeostatic variant by CYCLE-INDEPENDENCE (not by the cycles=2 peak):
    #   - W must track the counts at EVERY cycle (min Pearson(W,counts) > counts_bar),
    #   - the brain-based read-out must recover the graded structure at EVERY cycle (min
    #     Pearson(sim,Strue) high, min generalization >= a1_bar),
    #   - among the homeostatic variants (oja / scaling) only.
    # Rank by: (min generalization >= a1_bar) then (min bb Pearson) then (min W-counts) -- the worst
    # case across the cycle sweep is what cycle-independence MEANS.
    # =========================================================================
    homeo_keys = [v[0] for v in variants if v[0].startswith("oja") or v[0].startswith("scaling")]

    def variant_worstcase_score(vkey):
        ci = cycindep[vkey]
        gen_ok = ci["bb_generalization_min"] >= args.a1_bar
        counts_ok = ci["pearson_W_counts_min"] > args.counts_bar
        # worst-case brain-based Pearson + bonuses for holding the gates everywhere
        return ((1000.0 if (gen_ok and counts_ok) else 0.0)
                + ci["bb_pearson_Strue_min"] * 100.0
                + ci["pearson_W_counts_min"] * 10.0)

    best_homeo_key = max(homeo_keys, key=variant_worstcase_score) if homeo_keys else None
    print(f"\n  {'-'*82}", flush=True)
    print(f"  BEST HOMEOSTATIC variant (by worst-case cycle-independence): {best_homeo_key}", flush=True)
    print(f"  {'-'*82}", flush=True)

    # =========================================================================
    # STORE-VOLUME / SCALE STRESS at the best homeostatic variant (FIXED cycle count = args.stress_cycles).
    # =========================================================================
    best_kw = dict([v for v in variants if v[0] == best_homeo_key][0][1]) if best_homeo_key else None
    stress = {}
    if best_homeo_key is not None:
        print(f"\n  STORE-VOLUME STRESS @ {best_homeo_key} (cycles={args.stress_cycles}, "
              f"reps={args.store_reps_sweep}):", flush=True)
        for reps in args.store_reps_sweep:
            inflated = inflate_corpus(corpus["facts"], reps)
            C_full_rep = raw_count_matrix(concepts, inflated)
            t0 = time.time()
            W, info = learn_W_homeostatic(concepts, inflated, seed, args.n_pool, args.pattern_size,
                                          args.stress_cycles, **best_kw)
            # Pearson(W, counts): use the structure-normalized counts -- reps scales counts uniformly,
            # so offdiag_pearson is rep-invariant (Pearson is scale-invariant). Use base C_full so the
            # number is comparable across reps (the STRUCTURE is what must be tracked).
            m = measure_point(W, concepts, members, member_rows, S_true, second_order_pairs, labels,
                              props, C_full, nclu, pclu, seed, args, chance)
            m["learn_info"] = info
            m["learn_seconds"] = time.time() - t0
            m["n_facts_stored"] = len(inflated)
            stress[str(reps)] = m
            print(f"  reps={reps:2d} (={len(inflated)} stored facts, rec mean={info['recurrent_mean']:.3f}, "
                  f"{m['learn_seconds']:.0f}s)", flush=True)
            print(_fmt(f"{best_homeo_key} x{reps}", m), flush=True)
        # also stress the UN-NORMALIZED at the same reps (the contrast -- it should saturate faster)
        print(f"\n  STORE-VOLUME STRESS @ unnormalized (cycles={args.stress_cycles}, contrast):",
              flush=True)
        stress_unnorm = {}
        for reps in args.store_reps_sweep:
            inflated = inflate_corpus(corpus["facts"], reps)
            t0 = time.time()
            W, info = learn_W_homeostatic(concepts, inflated, seed, args.n_pool, args.pattern_size,
                                          args.stress_cycles, gamma=1.0, cap=None, homeo="none")
            m = measure_point(W, concepts, members, member_rows, S_true, second_order_pairs, labels,
                              props, C_full, nclu, pclu, seed, args, chance)
            m["learn_seconds"] = time.time() - t0
            m["n_facts_stored"] = len(inflated)
            stress_unnorm[str(reps)] = m
            print(_fmt(f"unnorm x{reps}", m), flush=True)
    else:
        stress_unnorm = {}

    # store-volume holds? min over reps of the homeostatic variant's numbers.
    stress_holds = None
    if best_homeo_key is not None and stress:
        s_counts = [stress[r]["pearson_W_vs_rawcounts"] for r in stress]
        s_pstrue = [stress[r]["bb_pearson_vs_Strue"] for r in stress]
        s_gen = [stress[r]["bb_generalization"] for r in stress]
        stress_holds = bool(min(s_counts) > args.counts_bar and min(s_pstrue) >= args.stress_pearson_bar
                            and min(s_gen) >= args.a1_bar)
        print(f"\n  STORE-VOLUME HOLDS @ {best_homeo_key}: {stress_holds} "
              f"(min P(W,counts)={min(s_counts):+.3f}, min P(sim,Strue)={min(s_pstrue):+.3f}, "
              f"min gen={min(s_gen):.3f})", flush=True)

    # =========================================================================
    # GATE RE-CONFIRM at the best homeostatic variant, ACROSS the cycle sweep (the cycle-independent
    # gate). For each cycle in a re-confirm subset, run the FULL G1 + G2 (with controls) + the
    # permuted-co-occurrence headline, on the brain-based divnorm read-out.
    # =========================================================================
    gate_reconfirm = {}
    if best_homeo_key is not None:
        print(f"\n  {'-'*82}", flush=True)
        print(f"  GATE RE-CONFIRM @ {best_homeo_key} across cycles {args.gate_cycles} "
              f"(brain-based divnorm read-out)", flush=True)
        print(f"  {'-'*82}", flush=True)
        for cyc in args.gate_cycles:
            W, info = learn_W_homeostatic(concepts, corpus["facts"], seed, args.n_pool,
                                          args.pattern_size, cyc, **best_kw)
            codes = brain_based_codes(W, member_rows, args)
            pearson_W_counts = offdiag_pearson(W, C_full)
            W_distinct = bool(pearson_W_counts < 0.99)
            # G1
            g1_rec = structure_recovery(codes, S_true, second_order_pairs, seed)
            g1_grad = codebook_similarity_stats(codes, labels)
            g1_ok = bool(g1_rec["pearson_learned_vs_Strue"] >= args.g1_bar and g1_grad["is_graded"]
                         and g1_rec["second_order_margin"] >= args.so_margin_bar)
            # G2
            gen = architecture_generalization(codes, labels, props, nclu, pclu, seed,
                                              args.k_neighbours, args.a1_bar)
            # G5 permuted co-occurrence (headline)
            perm_facts = permute_corpus(corpus["facts"], concepts, seed)
            Wp, _ = learn_W_homeostatic(concepts, perm_facts, seed, args.n_pool, args.pattern_size,
                                        cyc, **best_kw)
            perm_codes = brain_based_codes(Wp, member_rows, args)
            perm_rec = structure_recovery(perm_codes, S_true, second_order_pairs, seed)
            perm_grad = codebook_similarity_stats(perm_codes, labels)
            perm_gen = float(run_generalization(perm_codes, labels, props, nclu, pclu, seed,
                                                args.k_neighbours)["accuracy"])
            g5_permco = bool(abs(perm_rec["pearson_learned_vs_Strue"]) < args.g1_bar * 0.6
                             and not perm_grad["is_graded"] and perm_gen <= 1.5 * chance)
            gates_ok = bool(g1_ok and gen["a1"] and gen["a2"] and gen["a3"] and g5_permco and W_distinct)
            gate_reconfirm[str(cyc)] = {
                "pearson_W_vs_rawcounts": pearson_W_counts,
                "W_distinct": W_distinct,
                "g1_pearson_vs_Strue": g1_rec["pearson_learned_vs_Strue"],
                "g1_second_order_margin": g1_rec["second_order_margin"],
                "g1_is_graded": bool(g1_grad["is_graded"]),
                "g1_ok": g1_ok,
                "g2_graded_acc": gen["graded"]["accuracy"],
                "g2_a1": bool(gen["a1"]), "g2_a2": bool(gen["a2"]), "g2_a3": bool(gen["a3"]),
                "g2_orthogonal_acc": gen["orthogonal"]["accuracy"],
                "g2_permuted_prop_acc": gen["permuted"]["accuracy"],
                "g5_permuted_cooccurrence_collapses": g5_permco,
                "permuted_cooccurrence_pearson": perm_rec["pearson_learned_vs_Strue"],
                "all_gates_ok": gates_ok,
            }
            print(f"  cycles={cyc:3d}: G1={g1_ok} (P={g1_rec['pearson_learned_vs_Strue']:+.3f} "
                  f"2nd={g1_rec['second_order_margin']:+.3f} graded={int(g1_grad['is_graded'])}) | "
                  f"G2 A1={gen['a1']}({gen['graded']['accuracy']:.2f}) A2={gen['a2']} A3={gen['a3']} | "
                  f"G5permco={g5_permco} | W-distinct={W_distinct} => ALL={gates_ok}", flush=True)

    return {
        "seed": seed,
        "corpus": {"n_concepts": len(concepts), "n_members": Nm, "n_facts": corpus["n_facts"],
                   "n_second_order_pairs": len(second_order_pairs)},
        "host_ceiling": {"pearson_vs_Strue": host_rec["pearson_learned_vs_Strue"],
                         "generalization": host_gen,
                         "second_order_margin": host_rec["second_order_margin"]},
        "random_baseline_generalization": rand_gen,
        "variant_cycle_sweep": sweep,
        "cycle_independence": cycindep,
        "best_homeostatic_variant": best_homeo_key,
        "best_homeostatic_config": best_kw,
        "store_volume_stress": {"homeostatic": stress, "unnormalized": stress_unnorm,
                                "holds": stress_holds, "cycles": args.stress_cycles,
                                "reps_sweep": args.store_reps_sweep},
        "gate_reconfirm_across_cycles": gate_reconfirm,
    }


def _seed_verdict(rseed, args):
    """Per-seed verdict: GO (cycle-independent) if the best homeostatic variant holds the gates ACROSS
    the gate-cycle subset (cycle-independence) AND the store-volume stress holds. BOUNDARY if it helps
    but degrades at high cycles or under store-volume stress."""
    best = rseed["best_homeostatic_variant"]
    if best is None:
        return "BOUNDARY"
    ci = rseed["cycle_independence"][best]
    gr = rseed["gate_reconfirm_across_cycles"]
    stress_holds = rseed["store_volume_stress"]["holds"]

    # cycle-independence: W tracks counts at every swept cycle, brain-based read-out recovers + gen
    # holds at every swept cycle.
    counts_ok_all = ci["pearson_W_counts_min"] > args.counts_bar
    gen_ok_all = ci["bb_generalization_min"] >= args.a1_bar
    pstrue_ok_all = ci["bb_pearson_Strue_min"] >= args.cycindep_pearson_bar

    # gate re-confirm: the FULL gates pass at EVERY gate-cycle (the cycle-independent gate).
    gates_ok_all = bool(gr) and all(v["all_gates_ok"] for v in gr.values())

    if counts_ok_all and gen_ok_all and pstrue_ok_all and gates_ok_all and stress_holds:
        return "GO"
    # helps but doesn't fully clear cycle-independence / store-volume -> BOUNDARY.
    return "BOUNDARY"


def main():
    p = argparse.ArgumentParser(description="Learned graded-embedding HOMEOSTASIS probe "
                                            "(cycle-independent faithfulness via Oja / synaptic scaling)")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--seed", type=int, default=None, help="single-seed override")
    # toy corpus (MUST match the de-risk/desaturate/confirm/divnorm defaults)
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--per-cluster", type=int, default=5)
    p.add_argument("--n-props", type=int, default=4)
    p.add_argument("--hub-facts-per-member", type=int, default=6)
    p.add_argument("--bridge-facts", type=int, default=8)
    p.add_argument("--triplet-facts-per-cluster", type=int, default=4)
    # learned-assoc-graph (brain-based learner)
    p.add_argument("--n-pool", type=int, default=2000)
    p.add_argument("--pattern-size", type=int, default=100)
    # VARIANT x CYCLE sweep
    p.add_argument("--cycles-sweep", type=int, nargs="+", default=[2, 5, 10, 20, 40])
    p.add_argument("--gamma-ref", type=float, default=0.95, help="the inferior gamma-decay reference")
    p.add_argument("--oja-targets", type=float, nargs="+", default=[1.0, 2.0, 4.0],
                   help="Oja per-post-neuron incoming L2-norm set-points (FIXED; not fit to S_true)")
    p.add_argument("--scaling-targets", type=float, nargs="+", default=[5.0, 10.0, 20.0],
                   help="synaptic-scaling per-post-neuron incoming SUM set-points (FIXED)")
    # store-volume stress
    p.add_argument("--store-reps-sweep", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--stress-cycles", type=int, default=20,
                   help="fixed cycle count for the store-volume stress (a high-cycle regime)")
    p.add_argument("--stress-pearson-bar", type=float, default=0.30,
                   help="min brain-based Pearson(sim,Strue) under store-volume stress to 'hold'")
    # gate re-confirm cycles (subset of the cycle sweep where we run the full gate battery)
    p.add_argument("--gate-cycles", type=int, nargs="+", default=[2, 10, 40])
    # FIXED brain-based divnorm read-out -- the VALIDATED divnorm-GO recipe
    # (2026-06-11-learned-graded-embedding-divnorm-readout-GO.md, commit 9fa90d74):
    # ch (Carandini-Heeger) / interleave / steps2 / sigma0.001 / exp2.0 / logclip off -- closes 3/3.
    p.add_argument("--readout-divnorm", default="ch",
                   help="brain-based divnorm form: ch(Carandini-Heeger, validated) / marginal / none")
    p.add_argument("--readout-order", default="interleave",
                   help="pre / post / interleave(validated) / diffuse_only")
    p.add_argument("--readout-sigma", type=float, default=0.001)
    p.add_argument("--readout-exponent", type=float, default=2.0)
    p.add_argument("--readout-log-clip", action="store_true")
    p.add_argument("--diffusion-alpha", type=float, default=0.5)
    p.add_argument("--diffusion-steps", type=int, default=2)
    # host-method-on-W secondary
    p.add_argument("--svd-dim", type=int, default=40)
    # gate bars (match the de-risk / divnorm)
    p.add_argument("--g1-bar", type=float, default=0.5, help="Pearson(sim, S_true) >= this = recovered")
    p.add_argument("--a1-bar", type=float, default=0.7, help="generalization >= this (1.000-class)")
    p.add_argument("--so-margin-bar", type=float, default=0.10, help="2nd-order cat~dog margin bar")
    p.add_argument("--counts-bar", type=float, default=0.30,
                   help="Pearson(W, raw_counts) >= this = the learned W tracks the counts (>> +0.06)")
    p.add_argument("--cycindep-pearson-bar", type=float, default=0.50,
                   help="min brain-based Pearson(sim,Strue) across the cycle sweep to call it "
                        "cycle-independent (worst-case)")
    p.add_argument("--k-neighbours", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    backend = os.environ.get("SIM_BACKEND", "auto")
    chance = 1.0 / args.n_props
    t_all = time.time()
    print(f"[learned-graded-embedding HOMEOSTASIS probe] seeds={seeds} backend={backend}", flush=True)
    print(f"  toy: {args.n_clusters}x{args.per_cluster} (+hubs); learner=LearnedAssocGraph "
          f"(n_pool={args.n_pool}, pattern_size={args.pattern_size})", flush=True)
    print(f"  VARIANTS: unnormalized + gamma{args.gamma_ref} + oja{args.oja_targets} + "
          f"scaling{args.scaling_targets}; cycles={args.cycles_sweep}", flush=True)
    print(f"  read-out FIXED = brain-based divnorm '{args.readout_divnorm}'/{args.readout_order} "
          f"(sigma={args.readout_sigma} exp={args.readout_exponent} steps={args.diffusion_steps})",
          flush=True)
    print(f"  bars: G1(P>={args.g1_bar}) A1(gen>={args.a1_bar}) 2nd(>={args.so_margin_bar}) "
          f"counts(W~counts>={args.counts_bar}) cycindep(minP>={args.cycindep_pearson_bar})", flush=True)

    per_seed = {}
    for s in seeds:
        per_seed[str(s)] = run_seed(s, args)

    verdicts = {str(s): _seed_verdict(per_seed[str(s)], args) for s in seeds}
    vset = set(verdicts.values())
    if vset == {"GO"}:
        consensus = "GO"
    elif "GO" in vset:
        consensus = "MIXED_with_GO:" + ",".join(f"{s}={v}" for s, v in verdicts.items())
    else:
        consensus = "BOUNDARY"

    # aggregate: the best homeostatic variant per seed + its cycle-independence + the unnormalized
    # collapse for the headline contrast.
    def agg_unnorm_cycle(field):
        out = []
        for s in seeds:
            ci = per_seed[str(s)]["cycle_independence"]["unnormalized"]
            out.append(ci[field])
        return out

    best_variant_per_seed = [per_seed[str(s)]["best_homeostatic_variant"] for s in seeds]

    def agg_best_ci(field):
        out = []
        for s in seeds:
            best = per_seed[str(s)]["best_homeostatic_variant"]
            out.append(per_seed[str(s)]["cycle_independence"][best][field] if best else None)
        return out

    summary = {
        "consensus_verdict": consensus,
        "per_seed_verdict": verdicts,
        "seeds": seeds,
        "backend": backend,
        "chance": chance,
        "brain_based_note": (
            "the learned W is the project's spiking-Hebbian recurrent (LearnedAssocGraph). The "
            "HOMEOSTATIC arms are BIOLOGICAL bounded-Hebbian mechanisms applied per-cycle, "
            "pool<->pool, per-POSTSYNAPTIC-neuron (cp_connections is (pre->post); neuron j's incoming "
            "weights = the j-th COLUMN): (oja) Oja-1982 incoming-L2-norm renorm to a fixed set-point; "
            "(scaling) Turrigiano synaptic-scaling incoming-SUM renorm to a fixed set-point. Both are "
            "FIXED set-points (NOT fit to S_true). Applied runner-side (NO sim/ edits). The read-out "
            "is FIXED to the validated FULLY BRAIN-BASED divnorm recipe (the LEARN is what varies). "
            "The host PPMI+SVD on RAW counts is the labelled CEILING ONLY."),
        "best_homeostatic_variant_per_seed": best_variant_per_seed,
        "headline_cycle_independence": {
            "unnormalized_pearson_W_counts_per_cycle": agg_unnorm_cycle("pearson_W_counts_per_cycle"),
            "unnormalized_pearson_W_counts_slope_mean": float(np.mean(
                agg_unnorm_cycle("pearson_W_counts_slope_per_cycle"))),
            "unnormalized_bb_pearson_Strue_per_cycle": agg_unnorm_cycle("bb_pearson_Strue_per_cycle"),
            "unnormalized_bb_generalization_per_cycle": agg_unnorm_cycle("bb_generalization_per_cycle"),
            "best_homeo_pearson_W_counts_per_cycle": agg_best_ci("pearson_W_counts_per_cycle"),
            "best_homeo_pearson_W_counts_slope_mean": float(np.mean(
                [x for x in agg_best_ci("pearson_W_counts_slope_per_cycle") if x is not None]))
            if any(x is not None for x in agg_best_ci("pearson_W_counts_slope_per_cycle")) else None,
            "best_homeo_bb_pearson_Strue_per_cycle": agg_best_ci("bb_pearson_Strue_per_cycle"),
            "best_homeo_bb_pearson_Strue_min_per_seed": agg_best_ci("bb_pearson_Strue_min"),
            "best_homeo_bb_generalization_per_cycle": agg_best_ci("bb_generalization_per_cycle"),
            "best_homeo_bb_generalization_min_per_seed": agg_best_ci("bb_generalization_min"),
            "cycles": args.cycles_sweep,
        },
        "store_volume_holds_per_seed": [per_seed[str(s)]["store_volume_stress"]["holds"] for s in seeds],
        "gate_reconfirm_all_pass_per_seed": [
            (all(v["all_gates_ok"] for v in per_seed[str(s)]["gate_reconfirm_across_cycles"].values())
             if per_seed[str(s)]["gate_reconfirm_across_cycles"] else False) for s in seeds],
        "host_ceiling_mean": {
            "pearson_vs_Strue": float(np.mean([per_seed[str(s)]["host_ceiling"]["pearson_vs_Strue"]
                                               for s in seeds])),
            "generalization": float(np.mean([per_seed[str(s)]["host_ceiling"]["generalization"]
                                             for s in seeds])),
        },
        "collapsed_baseline_reference": {
            "source": "2026-06-11-learned-graded-embedding-diagnosis.md / confirm-GO_full.md",
            "unnormalized_pearson_W_counts_2cyc": 0.69, "unnormalized_pearson_W_counts_20cyc": 0.06,
            "gamma095_pearson": 0.43, "gamma095_gen": 0.77,
            "recovered_recipe": "cycles=2 (hand-picked low-cycle); host-on-W +0.84; ceiling +0.93; gen 1.000",
        },
        "elapsed_total_s": time.time() - t_all,
    }

    print(f"\n{'='*86}", flush=True)
    print(f"  HOMEOSTASIS PROBE SUMMARY", flush=True)
    print(f"{'='*86}", flush=True)
    print(f"  CONSENSUS VERDICT: {consensus}", flush=True)
    for s in seeds:
        rs = per_seed[str(s)]
        best = rs["best_homeostatic_variant"]
        ci = rs["cycle_independence"][best] if best else None
        ciu = rs["cycle_independence"]["unnormalized"]
        print(f"\n  seed {s}: verdict={verdicts[str(s)]} | best homeostatic = {best}", flush=True)
        print(f"    UNNORM   P(W,counts) per cycle {ciu['cycles']}: {ciu['pearson_W_counts_per_cycle']}",
              flush=True)
        print(f"             (slope {ciu['pearson_W_counts_slope_per_cycle']:+.4f}/cyc) "
              f"P(sim,Strue) {ciu['bb_pearson_Strue_per_cycle']} gen {ciu['bb_generalization_per_cycle']}",
              flush=True)
        if ci:
            print(f"    {best:9s} P(W,counts) per cycle {ci['cycles']}: {ci['pearson_W_counts_per_cycle']}",
                  flush=True)
            print(f"             (slope {ci['pearson_W_counts_slope_per_cycle']:+.4f}/cyc) "
                  f"P(sim,Strue) {ci['bb_pearson_Strue_per_cycle']} gen {ci['bb_generalization_per_cycle']}",
                  flush=True)
            print(f"             worst-case: P(W,counts)min={ci['pearson_W_counts_min']:+.3f} "
                  f"P(sim,Strue)min={ci['bb_pearson_Strue_min']:+.3f} genmin={ci['bb_generalization_min']:.3f}",
                  flush=True)
        print(f"    store-volume holds: {rs['store_volume_stress']['holds']} | "
              f"gate re-confirm all-pass: "
              f"{all(v['all_gates_ok'] for v in rs['gate_reconfirm_across_cycles'].values()) if rs['gate_reconfirm_across_cycles'] else False}",
              flush=True)
    print(f"\n  HOST CEILING (target): Pearson {summary['host_ceiling_mean']['pearson_vs_Strue']:+.3f} "
          f"gen {summary['host_ceiling_mean']['generalization']:.3f}", flush=True)
    print(f"  COLLAPSED BASELINE (un-normalized): Pearson(W,counts) +0.69@2cyc -> +0.06@20cyc", flush=True)
    print(f"  Total elapsed: {summary['elapsed_total_s']:.1f}s", flush=True)
    print(f"{'='*86}\n", flush=True)

    out_data = {"summary": summary, "per_seed": per_seed}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        args.out = os.path.join(raw_dir, f"_lge_homeostasis_seed{seeds[0]}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out_data


if __name__ == "__main__":
    main()
