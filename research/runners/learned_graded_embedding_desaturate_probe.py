"""LEARNED GRADED-EMBEDDING DE-SATURATE FIX-TEST -- does DE-SATURATING the brain-based Hebbian
LEARN recover the graded co-occurrence structure (toward the host PPMI+SVD ceiling), and pass the
architecture gates?

CONTEXT (the diagnosis that points here):
  The brain-based learned-embedding mechanism collapsed
  (2026-06-11-learned-graded-embedding-derisk-NEGATIVE.md). The diagnosis
  (2026-06-11-learned-graded-embedding-diagnosis.md, commit 61100385) localized it to the
  Hebbian LEARN, NOT the read-out: the un-normalized excitatory recurrent (LearnedAssocGraph --
  NO LTD, NO competition, NO synaptic scaling) SATURATES. After 20 store-cycles 2.40M/2.46M
  pool<->pool edges grow to a near-uniform floor (off-diag mean 0.864, CV 0.27; recurrent mean
  0.998) that swamps the graded co-occurrence signal, so Pearson(W, raw_counts) = +0.062.
  THE KEY POSITIVE TELL: at 2 store-cycles a CPU smoke had Pearson(W, raw_counts) = +0.724 -- the
  structure IS captured early, then washed out by saturation. The host ceiling (PPMI+SVD on the raw
  counts) is +0.932 / generalization 1.000 = the reachable target.

THE FIX-TEST (this runner):
  Does de-saturating the learn -- the missing biological homeostasis (fewer cycles, and/or an
  LTD/synaptic-scaling DECAY arm, and/or a weight cap) -- make the brain-LEARNED W track the graded
  counts AND recover the graded second-order structure (cat~dog) AND pass the gates? This is
  brain-based: real synapses HAVE LTD + synaptic scaling + competition; the toy learner omitted
  them. Two levers, swept:

  (1) STORE-CYCLES sweep (the cheapest lever, the +0.724 hint, NO code change to the learner):
      learn at store-cycles in {2,3,5,8,12}. At each: Pearson(W, raw_counts), Pearson(sim_W, S_true),
      G1 graded?, G2 generalization. Find where W tracks counts (>> +0.06) AND the graded structure
      survives (Pearson(sim_W, S_true) rises toward the +0.93 ceiling).

  (2) DE-SATURATION arm (homeostasis -- synaptic scaling / LTD analogue): a per-cycle MULTIPLICATIVE
      DECAY (gamma < 1) applied ONLY to the pool<->pool recurrent weights (the plastic edges) each
      store-cycle, so MORE cycles can accumulate the graded structure WITHOUT the uniform-floor
      collapse. Sweep gamma in {0.8,0.9,0.95} x cycles in {8,20}. Brain-based: Turrigiano synaptic
      scaling + Bienenstock-Cooper-Munro-style decay restore CONTRAST (non-co-firing pairs lose
      weight; the graded co-occurrence signal is preserved). Implemented runner-side by overriding
      LearnedAssocGraph.store_fact in a subclass -- NO sim/ edits, NO edit to learned_assoc_graph.py;
      the decay is a post-cycle rescale of cp_connections.data restricted to the pool<->pool edges
      (the pool is a contiguous index slice, so the data-mask is exact).

  (3) BEST OPERATING POINT -> the FULL de-risk gates: re-run G1 (structure recovery, incl. the
      second-order cat~dog margin, permuted-co-occurrence collapse) + G2 (generalization >= 0.7 with
      the orthogonal + permuted-property + permuted-co-occurrence controls collapsing) on the best
      operating point's learned codes. Does the de-saturated brain-based learn now PASS toward the
      host ceiling?

ANTI-CHEATS (all mandatory):
  - Pearson(W, raw_counts) is the BRAIN-LEARNED W (not the raw counts). At every operating point we
    assert W != raw counts (W_vs_counts < 0.999) so a "recovery" is NOT silently re-deriving the
    host ceiling from the counts. We report Pearson(W, raw_counts) AND Pearson(sim_W, S_true)
    SEPARATELY (tracking the counts is necessary; the graded second-order structure is the goal).
  - the permuted-S baseline (~0) for any recovered Pearson(sim_W, S_true).
  - the generalization gate (chance 0.25) with the orthogonal + permuted-property + permuted-code
    controls collapsing; the permuted-CO-OCCURRENCE control (re-learn on a scrambled corpus) must
    collapse; beats the random-Gaussian baseline.
  - the host PPMI+SVD on the RAW counts is the labelled CEILING ONLY (never the deliverable).

DECISION (stated explicitly):
  GO                if a de-saturated operating point (fewer cycles and/or homeostatic decay) makes
                    the brain-LEARNED W track the counts AND recover the graded structure
                    (Pearson(sim_W, S_true) high, generalization >= 0.7, controls collapse) ->
                    Option A is ALIVE: a homeostatically-regulated Hebbian learn recovers the
                    embedding -> re-run the full de-risk + scope the build.
  BOUNDARY_weak     if de-saturation tracks the counts but the graded SECOND-ORDER (cat~dog) stays
                    weak, OR it plateaus below the gates -> characterize + recommend the next
                    brain-based rule (competitive/predictive Hebbian: Oja / BCM / contrastive-
                    predict-the-context, the spiking analogue of what the host ceiling does).
  NEGATIVE          if no de-saturation operating point tracks the counts at all. No banking.

Run (GPU, FOREGROUND -- each spiking-learn is ~1-2.5 min inline; NO background):
  SIM_BACKEND=cupy python -m research.runners.learned_graded_embedding_desaturate_probe \
      --seed 42 --out research/findings/raw/_lge_desaturate_seed42.json
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

# Reuse the de-risk's corpus + read-out + structure-recovery + host-ceiling machinery, and the
# diagnose's per-stage analysis helpers -- VERBATIM (this runner only sweeps the LEARN config).
from research.runners.learned_graded_embedding_derisk_probe import (  # noqa: E402
    build_toy_cooccurrence,
    permute_corpus,
    graded_readout,
    structure_recovery,
    host_ceiling_codes,
    random_gaussian_codes,
    architecture_generalization,
)
from research.runners.learned_graded_embedding_diagnose import (  # noqa: E402
    raw_count_matrix,
    offdiag_pearson,
    member_submatrix,
    rows_to_codes,
    ppmi_transform,
    svd_lowdim,
    _normalize_codes,
)
from research.runners.dual_cls_architecture_proof_probe import (  # noqa: E402
    codebook_similarity_stats,
    assign_properties,
    run_generalization,
)


# ===========================================================================
# The DE-SATURATING learner: LearnedAssocGraph + a per-cycle homeostatic DECAY on the recurrent.
# ===========================================================================
class DesaturatingAssocGraph:
    """Wrap LearnedAssocGraph and add a brain-based homeostatic DECAY arm to store_fact: after each
    co-fire cycle, multiply the PLASTIC pool<->pool recurrent weights by gamma (< 1) -- a synaptic-
    scaling / LTD analogue (Turrigiano; the decay restores contrast: non-co-firing pairs lose weight
    so they don't fill to a uniform floor). gamma=1.0 + cap=None == the original (no de-saturation).

    The decay is applied ONLY to the pool<->pool edges (the recurrent being learned), identified by a
    data-mask over cp_connections.data (the pool is a contiguous index slice -> the mask is exact).
    NO sim/ edits; NO edit to learned_assoc_graph.py -- this is a runner-side post-cycle rescale.
    """

    def __init__(self, concepts, seed=42, n_pool=2000, pattern_size=100,
                 gamma=1.0, cap=None):
        from research.runners.learned_assoc_graph import LearnedAssocGraph
        from sim.backend import get_backend, to_host
        self._cp, _ = get_backend()
        self._to_host = to_host
        self.lag = LearnedAssocGraph(concepts, seed=seed, n_pool=n_pool, pattern_size=pattern_size)
        self.bridge = self.lag.bridge
        self.pool_base = self.lag.pool_base
        self.patterns = self.lag.patterns
        self.concepts = self.lag.concepts
        self.gamma = float(gamma)
        self.cap = cap  # float or None
        # Build the pool<->pool data-mask ONCE (the plastic recurrent edges in cp_connections.data).
        self._pool_data_mask = self._build_pool_data_mask()

    def _build_pool_data_mask(self):
        """Boolean mask over cp_connections.data selecting the pool->pool entries.
        CSR: row r's data is data[indptr[r]:indptr[r+1]] with columns indices[...]. The pool is a
        contiguous slice [pool_min, pool_max] -> an entry is pool->pool iff row in [min,max] AND
        col in [min,max]."""
        M = self.bridge.cp_connections
        pool = np.asarray(self.pool_base)
        pmin, pmax = int(pool.min()), int(pool.max())
        assert np.array_equal(pool, np.arange(pmin, pmax + 1)), \
            "pool indices are expected to be contiguous for the exact data-mask"
        indptr = np.asarray(self._to_host(M.indptr))
        indices = np.asarray(self._to_host(M.indices))
        nnz = int(M.data.size)
        # row id per data entry
        rows = np.zeros(nnz, dtype=np.int64)
        for r in range(len(indptr) - 1):
            rows[indptr[r]:indptr[r + 1]] = r
        mask = (rows >= pmin) & (rows <= pmax) & (indices >= pmin) & (indices <= pmax)
        # move to backend (cupy) bool array for fast in-place rescale
        return self._cp.asarray(mask)

    def store_fact(self, concept_list, cycles=20):
        """Co-fire the fact's concept patterns (LearnedAssocGraph's Hebbian growth), then apply the
        homeostatic DECAY/cap ONCE on the pool<->pool recurrent (per-FACT = the natural between-
        experiences synaptic-scaling timescale; per-inner-cycle decay was far too aggressive -- it
        rescales ~cycles x more often and zeroes the recurrent). gamma=1.0 + cap=None == original."""
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
        # ---- homeostatic de-saturation arm (ONCE per fact), pool<->pool edges only ----
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


def learn_W_desaturate(concepts, facts, seed, n_pool, pattern_size, cycles,
                       gamma=1.0, cap=None):
    """Learn the brain-based co-occurrence recurrent with the (optional) homeostatic de-saturation
    arm, then read it as the concept->concept association matrix W [Nc, Nc] (mean a->b recurrent
    weight between each pair's sparse patterns -- the SAME extraction the de-risk/diagnose use).
    Returns (W, info)."""
    from sim.backend import to_host
    dag = DesaturatingAssocGraph(concepts, seed=seed, n_pool=n_pool, pattern_size=pattern_size,
                                 gamma=gamma, cap=cap)
    for f in facts:
        dag.store_fact(list(f), cycles=cycles)
    M = to_host(dag.bridge.cp_connections)
    pb = dag.pool_base
    sub = M[pb][:, pb]
    dense = np.asarray(sub.todense())
    Nc = len(concepts)
    W = np.zeros((Nc, Nc), dtype=np.float64)
    pats = [np.asarray(p) for p in dag.patterns]
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
        "n_neurons": int(dag.bridge.cp_membrane_potential_v.shape[0]),
        "cycles": int(cycles), "gamma": float(gamma),
        "cap": (float(cap) if cap is not None else None),
    }
    return W, info


# ===========================================================================
# The HOST-METHOD read-out on the FULL learned W (mirrors host_ceiling_codes EXACTLY, but on the
# brain-LEARNED W instead of the raw counts). This is the load-bearing read-out: it runs PPMI+SVD
# over the FULL concept set (hubs INCLUDED) and extracts member rows -- so the SVD propagates the
# HUB-mediated second-order structure into each member's embedding row. The diagnose's member-
# SUBMATRIX read-outs structurally CANNOT recover second-order structure (they discard the hub
# columns where cat~dog's shared-neighbour signal lives), which is why a W that tracks the counts
# still scored negative on those. If a de-saturated W tracks the FULL counts, THIS read-out is the
# one that can reach the ceiling.
# ===========================================================================
def host_method_codes_on_W(W: np.ndarray, member_rows: np.ndarray, dim: int) -> np.ndarray:
    """PPMI + truncated-SVD over the FULL learned W (all concepts), then member rows of the embedding
    -- the host_ceiling pipeline applied to the BRAIN-LEARNED W (NOT the raw counts)."""
    Ws = 0.5 * (W + W.T)
    np.fill_diagonal(Ws, 0.0)
    ppmi = ppmi_transform(Ws)                       # PPMI over the FULL W (hubs included)
    U, Sv, _ = np.linalg.svd(ppmi, full_matrices=False)
    d = min(dim, U.shape[1])
    emb = (U[:, :d] * Sv[:d])
    codes = emb[member_rows]
    return _normalize_codes(codes.astype(np.float64))


# ===========================================================================
# Per-operating-point measurement (the LEARN-quality numbers + a light generalization read).
# ===========================================================================
def measure_operating_point(W, concepts, members, member_rows, S_true, second_order_pairs,
                            labels, props, C_members, C_full, nclu, pclu, seed, args, chance,
                            light=True):
    """Given a learned W, compute the LEARN-quality numbers:
       - Pearson(W, raw_counts)      : does the learned W track the co-occurrence COUNTS?
       - Pearson(sim_W, S_true)      : does the W-row code cosine recover the graded structure?
       - graded? (within/between cos margin), second-order cat~dog margin
       - generalization on the W-row codes (the brain-learned codes, before any read-out)
       - (light) optional PPMI+SVD-on-W read-out generalization (the faithful-W re-opens the read-out)
    Returns a dict."""
    Nm = len(members)
    W_members = member_submatrix(W, member_rows)

    # (A) faithfulness: does the learned W track the raw counts? Report BOTH the member-submatrix
    # Pearson (the diagnose's check) AND the FULL-matrix Pearson (the load-bearing one -- the host
    # ceiling reads the FULL counts; the hub columns carry the second-order signal).
    pearson_W_counts_members = offdiag_pearson(W_members, C_members)
    pearson_W_counts_full = offdiag_pearson(W, C_full)
    iu = np.triu_indices(Nm, k=1)
    w_off = W_members[iu]
    w_cv = float(np.std(w_off) / (np.abs(np.mean(w_off)) + 1e-12))

    # (B) PRIMARY read-out = HOST-METHOD on the FULL learned W (PPMI+SVD over all concepts incl.
    # hubs -> member embedding rows). This is the apples-to-apples to the host ceiling on the
    # brain-LEARNED W; the ONLY tested read-out that can recover the hub-mediated second-order signal.
    hostW_codes = host_method_codes_on_W(W, member_rows, args.svd_dim)
    rec_hostW = structure_recovery(hostW_codes, S_true, second_order_pairs, seed)
    grad_hostW = codebook_similarity_stats(hostW_codes, labels)
    gen_hostW = float(run_generalization(hostW_codes, labels, props, nclu, pclu, seed,
                                         args.k_neighbours)["accuracy"])

    # (C) STAGE W: member-submatrix W-row codes -> cosine -> Pearson vs S_true (the diagnose's stage).
    sim_W_codes = rows_to_codes(W_members)
    rec = structure_recovery(sim_W_codes, S_true, second_order_pairs, seed)
    grad = codebook_similarity_stats(sim_W_codes, labels)
    gen_W = float(run_generalization(sim_W_codes, labels, props, nclu, pclu, seed,
                                     args.k_neighbours)["accuracy"])

    # (D) diffusion read-out on this W (the de-risk's current read-out) -- one number.
    diff_codes = graded_readout(W, member_rows, args.diffusion_alpha, args.diffusion_steps)
    rec_diff = structure_recovery(diff_codes, S_true, second_order_pairs, seed)
    grad_diff = codebook_similarity_stats(diff_codes, labels)
    gen_diff = float(run_generalization(diff_codes, labels, props, nclu, pclu, seed,
                                        args.k_neighbours)["accuracy"])

    return {
        # faithfulness
        "pearson_W_vs_rawcounts": pearson_W_counts_full,            # PRIMARY (full matrix)
        "pearson_W_vs_rawcounts_members": pearson_W_counts_members,  # member-submatrix (diagnose's)
        "W_offdiag_mean": float(np.mean(w_off)),
        "W_offdiag_std": float(np.std(w_off)),
        "W_offdiag_cv": w_cv,
        # PRIMARY read-out: host-method (PPMI+SVD) on the FULL learned W
        "hostW_pearson_vs_Strue": rec_hostW["pearson_learned_vs_Strue"],
        "hostW_pearson_permutedS": rec_hostW["pearson_permuted_vs_Strue"],
        "hostW_within_cos": grad_hostW["within_cluster_cos_mean"],
        "hostW_between_cos": grad_hostW["between_cluster_cos_mean"],
        "hostW_is_graded": bool(grad_hostW["is_graded"]),
        "hostW_second_order_margin": rec_hostW["second_order_margin"],
        "hostW_second_order_recovered": bool(rec_hostW["second_order_recovered"]),
        "hostW_generalization": gen_hostW,
        # STAGE W (member-submatrix rows -- the diagnose's stage)
        "stageW_pearson_simW_vs_Strue": rec["pearson_learned_vs_Strue"],
        "stageW_pearson_permutedS": rec["pearson_permuted_vs_Strue"],
        "stageW_is_graded": bool(grad["is_graded"]),
        "stageW_second_order_margin": rec["second_order_margin"],
        "stageW_generalization": gen_W,
        # diffusion read-out
        "diff_pearson_vs_Strue": rec_diff["pearson_learned_vs_Strue"],
        "diff_is_graded": bool(grad_diff["is_graded"]),
        "diff_generalization": gen_diff,
        "diff_second_order_margin": rec_diff["second_order_margin"],
    }


def _fmt_point(tag, m):
    return (f"    [{tag:16s}] P(W,counts_full)={m['pearson_W_vs_rawcounts']:+.3f} "
            f"(memb {m['pearson_W_vs_rawcounts_members']:+.3f}) CV={m['W_offdiag_cv']:.3f} "
            f"|| HOST(W): P(sim,Strue)={m['hostW_pearson_vs_Strue']:+.3f} "
            f"graded={m['hostW_is_graded']} 2nd-margin={m['hostW_second_order_margin']:+.3f} "
            f"gen={m['hostW_generalization']:.3f} "
            f"| STAGE-W: P={m['stageW_pearson_simW_vs_Strue']:+.3f} gen={m['stageW_generalization']:.3f} "
            f"| diff: P={m['diff_pearson_vs_Strue']:+.3f} gen={m['diff_generalization']:.3f}")


# ===========================================================================
# per-seed driver
# ===========================================================================
def run_seed(seed: int, args) -> dict:
    print(f"\n{'='*82}", flush=True)
    print(f"  LEARNED GRADED-EMBEDDING DE-SATURATE FIX-TEST -- SEED {seed}", flush=True)
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

    # raw co-occurrence COUNT matrix (FULL incl. hubs + member submatrix) -- the W-vs-counts
    # faithfulness reference. The FULL matrix is the load-bearing one (the host ceiling reads it).
    C_full = raw_count_matrix(concepts, corpus["facts"])
    C_members = member_submatrix(C_full, member_rows)

    print(f"  [corpus] {len(concepts)} concepts ({nclu} hubs + {Nm} members), "
          f"{corpus['n_facts']} facts; second-order pairs={len(second_order_pairs)}; chance={chance:.3f}",
          flush=True)

    # ----- labelled HOST CEILING (PPMI+SVD on RAW counts) -- the reachable target -----
    host_codes = host_ceiling_codes(concepts, corpus["facts"], member_rows, Nm, seed)
    host_rec = structure_recovery(host_codes, S_true, second_order_pairs, seed)
    host_gen = float(run_generalization(host_codes, labels, props, nclu, pclu, seed,
                                        args.k_neighbours)["accuracy"])
    host_graded = codebook_similarity_stats(host_codes, labels)
    print(f"  [HOST CEILING (PPMI+SVD on RAW counts, labelled target)] "
          f"Pearson(S,S_true)={host_rec['pearson_learned_vs_Strue']:+.3f} gen={host_gen:.3f} "
          f"graded={host_graded['is_graded']}", flush=True)

    rand_codes = random_gaussian_codes(Nm, Nm, seed)
    rand_gen = float(run_generalization(rand_codes, labels, props, nclu, pclu, seed,
                                        args.k_neighbours)["accuracy"])

    # =========================================================================
    # SWEEP 1 -- STORE-CYCLES (no code change; the +0.724 hint)
    # =========================================================================
    print(f"\n  {'-'*78}", flush=True)
    print(f"  SWEEP 1: store-cycles {args.cycles_sweep} (de-saturate by RUNNING FEWER cycles; "
          f"the +0.724-at-2-cycles hint)", flush=True)
    print(f"  {'-'*78}", flush=True)
    cycles_points = {}
    for cyc in args.cycles_sweep:
        t0 = time.time()
        W, info = learn_W_desaturate(concepts, corpus["facts"], seed,
                                     args.n_pool, args.pattern_size, cyc, gamma=1.0, cap=None)
        m = measure_operating_point(W, concepts, members, member_rows, S_true, second_order_pairs,
                                    labels, props, C_members, C_full, nclu, pclu, seed, args, chance)
        m["learn_info"] = info
        m["learn_seconds"] = time.time() - t0
        key = f"cycles{cyc}"
        cycles_points[key] = m
        print(f"  cycles={cyc:3d} (recurrent mean={info['recurrent_mean']:.3f} nnz={info['recurrent_nnz']}, "
              f"{m['learn_seconds']:.0f}s)", flush=True)
        print(_fmt_point(key, m), flush=True)

    # =========================================================================
    # SWEEP 2 -- DE-SATURATION DECAY arm (gamma x cycles)
    # =========================================================================
    print(f"\n  {'-'*78}", flush=True)
    print(f"  SWEEP 2: homeostatic DECAY gamma {args.gamma_sweep} x cycles {args.gamma_cycles_sweep} "
          f"(per-cycle pool<->pool synaptic-scaling/LTD analogue)", flush=True)
    print(f"  {'-'*78}", flush=True)
    decay_points = {}
    for gamma in args.gamma_sweep:
        for cyc in args.gamma_cycles_sweep:
            t0 = time.time()
            W, info = learn_W_desaturate(concepts, corpus["facts"], seed,
                                         args.n_pool, args.pattern_size, cyc, gamma=gamma, cap=None)
            m = measure_operating_point(W, concepts, members, member_rows, S_true, second_order_pairs,
                                        labels, props, C_members, C_full, nclu, pclu, seed, args, chance)
            m["learn_info"] = info
            m["learn_seconds"] = time.time() - t0
            key = f"gamma{gamma}_cycles{cyc}"
            decay_points[key] = m
            print(f"  gamma={gamma} cycles={cyc:3d} (recurrent mean={info['recurrent_mean']:.3f}, "
                  f"{m['learn_seconds']:.0f}s)", flush=True)
            print(_fmt_point(key, m), flush=True)

    # =========================================================================
    # BEST OPERATING POINT -> the FULL de-risk gates
    # =========================================================================
    all_points = {**cycles_points, **decay_points}

    # "best" = the brain-LEARNED W that best recovers the graded structure. We rank by the BEST of
    # {host-method-on-full-W, diffusion} Pearson(sim,S_true) -- the goal is graded recovery (incl.
    # the hub-mediated second-order signal), not just count tracking. The host-method-on-full-W
    # read-out is the load-bearing one (mirrors the host ceiling pipeline on the LEARNED W).
    def point_score(m):
        return max(m["hostW_pearson_vs_Strue"], m["diff_pearson_vs_Strue"])

    best_key = max(all_points, key=lambda k: point_score(all_points[k]))
    best_m = all_points[best_key]
    # which read-out won at the best point (host-method on full W vs diffusion)?
    best_readout = ("host_method_full_W" if best_m["hostW_pearson_vs_Strue"] >= best_m["diff_pearson_vs_Strue"]
                    else "diffusion")
    best_cfg = best_m["learn_info"]
    print(f"\n  {'-'*78}", flush=True)
    print(f"  BEST OPERATING POINT: {best_key} (read-out={best_readout}) -- "
          f"P(sim,Strue)={point_score(best_m):+.3f}  "
          f"Pearson(W,counts_full)={best_m['pearson_W_vs_rawcounts']:+.3f}", flush=True)
    print(f"  {'-'*78}", flush=True)

    # Re-learn at the best config and run the FULL gates (G1 + G2 with all controls), reusing the
    # de-risk's gate harnesses VERBATIM on the de-saturated codes.
    print(f"  [re-learn @ best ({best_cfg['cycles']}cyc gamma={best_cfg['gamma']} cap={best_cfg['cap']}) "
          f"-> FULL de-risk gates]", flush=True)
    t0 = time.time()
    Wb, infob = learn_W_desaturate(concepts, corpus["facts"], seed, args.n_pool, args.pattern_size,
                                   best_cfg["cycles"], gamma=best_cfg["gamma"], cap=best_cfg["cap"])
    # the codes the gates run on = the winning read-out at the best point.
    if best_readout == "host_method_full_W":
        best_codes = host_method_codes_on_W(Wb, member_rows, args.svd_dim)
    else:
        best_codes = graded_readout(Wb, member_rows, args.diffusion_alpha, args.diffusion_steps)

    # G1 structure recovery (+ permuted-S baseline + second-order margin).
    g1_rec = structure_recovery(best_codes, S_true, second_order_pairs, seed)
    g1_grad = codebook_similarity_stats(best_codes, labels)
    g1_ok = (g1_rec["pearson_learned_vs_Strue"] >= args.g1_bar and g1_grad["is_graded"]
             and g1_rec["second_order_recovered"])

    # G2 generalization (A1 + orthogonal A2 + permuted-property A3), reused VERBATIM.
    gen = architecture_generalization(best_codes, labels, props, nclu, pclu, seed,
                                      args.k_neighbours, args.a1_bar)

    # PERMUTED-CO-OCCURRENCE control (HEADLINE): re-learn on a scrambled corpus at the best config.
    perm_facts = permute_corpus(corpus["facts"], concepts, seed)
    Wp, _ = learn_W_desaturate(concepts, perm_facts, seed, args.n_pool, args.pattern_size,
                               best_cfg["cycles"], gamma=best_cfg["gamma"], cap=best_cfg["cap"])
    if best_readout == "host_method_full_W":
        perm_codes = host_method_codes_on_W(Wp, member_rows, args.svd_dim)
    else:
        perm_codes = graded_readout(Wp, member_rows, args.diffusion_alpha, args.diffusion_steps)
    perm_rec = structure_recovery(perm_codes, S_true, second_order_pairs, seed)
    perm_grad = codebook_similarity_stats(perm_codes, labels)
    perm_gen = float(run_generalization(perm_codes, labels, props, nclu, pclu, seed,
                                        args.k_neighbours)["accuracy"])
    g5_permco = (abs(perm_rec["pearson_learned_vs_Strue"]) < args.g1_bar * 0.6
                 and not perm_grad["is_graded"]
                 and perm_gen <= 1.5 * chance)

    # ANTI-CHEAT: the best-point W is distinct from the raw counts (not re-deriving the host ceiling).
    # Use the FULL matrix (the read-out runs on the full W; the member submatrix can be ~uniform).
    pearson_Wb_counts = offdiag_pearson(Wb, C_full)
    W_distinct = (pearson_Wb_counts < 0.999)

    beats_random = gen["graded"]["accuracy"] > rand_gen + 1e-9

    print(f"    G1 structure: Pearson(sim,S_true)={g1_rec['pearson_learned_vs_Strue']:+.3f} "
          f"(permS {g1_rec['pearson_permuted_vs_Strue']:+.3f}) graded={g1_grad['is_graded']} "
          f"2nd-order-margin={g1_rec['second_order_margin']:+.3f} "
          f"recovered={g1_rec['second_order_recovered']} -> G1={g1_ok}", flush=True)
    print(f"    G2 generalization: graded={gen['graded']['accuracy']:.3f} "
          f"(chance {gen['chance']:.3f}, {gen['graded']['ratio_vs_chance']:.1f}x) A1={gen['a1']} | "
          f"orthogonal={gen['orthogonal']['accuracy']:.3f} A2={gen['a2']} | "
          f"permuted-prop={gen['permuted']['accuracy']:.3f} A3={gen['a3']}", flush=True)
    print(f"    G5 permuted-CO-OCCURRENCE collapses (HEADLINE): {g5_permco} "
          f"(Pearson {perm_rec['pearson_learned_vs_Strue']:+.3f}, gen {perm_gen:.3f}) | "
          f"beats-random={beats_random} (learned {gen['graded']['accuracy']:.3f} > rand {rand_gen:.3f})",
          flush=True)
    print(f"    [ANTI-CHEAT] Pearson(W_best, raw_counts)={pearson_Wb_counts:+.3f} "
          f"(<0.999 => runs on the LEARNED W, distinct from counts: {W_distinct})", flush=True)
    print(f"    [best-point gates took {time.time()-t0:.0f}s]", flush=True)

    best_gates = {
        "g1_structure_recovered": bool(g1_ok),
        "g2_a1_generalizes": bool(gen["a1"]),
        "g2_a2_orthogonal_collapses": bool(gen["a2"]),
        "g2_a3_permuted_property_collapses": bool(gen["a3"]),
        "g5_permuted_cooccurrence_collapses": bool(g5_permco),
        "g5_beats_random_baseline": bool(beats_random),
        "anti_cheat_W_distinct_from_counts": bool(W_distinct),
    }

    return {
        "seed": seed,
        "corpus": {"n_concepts": len(concepts), "n_members": Nm, "n_facts": corpus["n_facts"],
                   "n_second_order_pairs": len(second_order_pairs)},
        "host_ceiling": {"pearson_vs_Strue": host_rec["pearson_learned_vs_Strue"],
                         "generalization": host_gen, "is_graded": host_graded["is_graded"]},
        "random_baseline_generalization": rand_gen,
        "cycles_sweep": cycles_points,
        "decay_sweep": decay_points,
        "best_operating_point": {
            "key": best_key,
            "readout": best_readout,
            "config": best_cfg,
            "score_pearson_sim_Strue": point_score(best_m),
            "pearson_W_vs_rawcounts": best_m["pearson_W_vs_rawcounts"],
            "g1_structure_recovery": g1_rec,
            "g1_graded_stats": g1_grad,
            "g2_generalization": gen,
            "permuted_cooccurrence": {"structure_recovery": perm_rec,
                                      "generalization": perm_gen,
                                      "is_graded": bool(perm_grad["is_graded"])},
            "anti_cheat_pearson_Wbest_vs_counts": pearson_Wb_counts,
            "gates": best_gates,
        },
    }


def _seed_verdict(rseed, args, chance):
    """Per-seed verdict from the best operating point + sweep."""
    bop = rseed["best_operating_point"]
    g = bop["gates"]
    host_gen = rseed["host_ceiling"]["generalization"]
    gen_graded = bop["g2_generalization"]["graded"]["accuracy"]
    pearson = bop["score_pearson_sim_Strue"]
    pW_counts = bop["pearson_W_vs_rawcounts"]

    tracks_counts = pW_counts > args.counts_bar       # W tracks the counts (>> +0.06)
    recovers_graded = (g["g1_structure_recovered"] and g["anti_cheat_W_distinct_from_counts"])
    generalizes = g["g2_a1_generalizes"]
    controls_collapse = (g["g2_a2_orthogonal_collapses"] and g["g2_a3_permuted_property_collapses"]
                         and g["g5_permuted_cooccurrence_collapses"] and g["g5_beats_random_baseline"])

    if recovers_graded and generalizes and controls_collapse:
        return "GO"
    if tracks_counts and (gen_graded > 1.2 * chance or pearson > 0.2):
        # de-saturation tracks the counts + above-chance graded signal but below the gates.
        return "BOUNDARY_weak"
    return "NEGATIVE"


def main():
    p = argparse.ArgumentParser(description="Learned graded-embedding DE-SATURATE fix-test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", default=None, help="comma list (overrides --seed; usually just 42)")
    # toy corpus (MUST match the de-risk/diagnose defaults so the NEGATIVE is the baseline)
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--per-cluster", type=int, default=5)
    p.add_argument("--n-props", type=int, default=4)
    p.add_argument("--hub-facts-per-member", type=int, default=6)
    p.add_argument("--bridge-facts", type=int, default=8)
    p.add_argument("--triplet-facts-per-cluster", type=int, default=4)
    # learned-assoc-graph (brain-based learner)
    p.add_argument("--n-pool", type=int, default=2000)
    p.add_argument("--pattern-size", type=int, default=100)
    # SWEEP 1: store-cycles
    p.add_argument("--cycles-sweep", type=int, nargs="+", default=[2, 3, 5, 8, 12])
    # SWEEP 2: gamma decay x cycles
    p.add_argument("--gamma-sweep", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    p.add_argument("--gamma-cycles-sweep", type=int, nargs="+", default=[8, 20])
    # read-outs
    p.add_argument("--diffusion-alpha", type=float, default=0.5)
    p.add_argument("--diffusion-steps", type=int, default=2)
    p.add_argument("--svd-dim", type=int, default=40)
    # gate bars
    p.add_argument("--g1-bar", type=float, default=0.5, help="Pearson(sim, S_true) >= this = recovered")
    p.add_argument("--a1-bar", type=float, default=0.7, help="generalization >= this = generalizes")
    p.add_argument("--counts-bar", type=float, default=0.30,
                   help="Pearson(W, raw_counts) >= this = the learned W tracks the counts (>> +0.06)")
    p.add_argument("--k-neighbours", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = [args.seed]
    backend = os.environ.get("SIM_BACKEND", "auto")
    chance = 1.0 / args.n_props
    t_all = time.time()
    print(f"[learned-graded-embedding DE-SATURATE fix-test] seeds={seeds} backend={backend}", flush=True)
    print(f"  toy: {args.n_clusters}x{args.per_cluster} (+hubs); learner=LearnedAssocGraph "
          f"(n_pool={args.n_pool}, pattern_size={args.pattern_size})", flush=True)
    print(f"  SWEEP1 cycles={args.cycles_sweep}; SWEEP2 gamma={args.gamma_sweep} x "
          f"cycles={args.gamma_cycles_sweep}", flush=True)
    print(f"  bars: G1(Pearson>={args.g1_bar}) A1(gen>={args.a1_bar}) "
          f"counts(W~counts>={args.counts_bar})", flush=True)

    per_seed = {}
    for s in seeds:
        per_seed[str(s)] = run_seed(s, args)

    # per-seed verdicts + consensus
    verdicts = {str(s): _seed_verdict(per_seed[str(s)], args, chance) for s in seeds}
    vset = set(verdicts.values())
    if len(vset) == 1:
        consensus = next(iter(vset))
    elif "GO" in vset:
        consensus = "MIXED_with_GO:" + ",".join(f"{s}={v}" for s, v in verdicts.items())
    else:
        consensus = "MIXED:" + ",".join(f"{s}={v}" for s, v in verdicts.items())

    # aggregate the best operating point across seeds
    def best_field(path):
        out = []
        for s in seeds:
            d = per_seed[str(s)]["best_operating_point"]
            for k in path:
                d = d[k]
            out.append(d)
        return out

    summary = {
        "consensus_verdict": consensus,
        "per_seed_verdict": verdicts,
        "seeds": seeds,
        "backend": backend,
        "chance": chance,
        "brain_based_note": ("the learned W is the project's spiking-Hebbian recurrent "
                             "(LearnedAssocGraph). The DE-SATURATION arm is a per-cycle multiplicative "
                             "DECAY (gamma<1) on the pool<->pool recurrent = a synaptic-scaling/LTD "
                             "analogue (Turrigiano), applied runner-side (NO sim/ edits). The host "
                             "PPMI+SVD on RAW counts is the labelled CEILING ONLY."),
        "host_ceiling_mean": {
            "pearson_vs_Strue": float(np.mean([per_seed[str(s)]["host_ceiling"]["pearson_vs_Strue"]
                                               for s in seeds])),
            "generalization": float(np.mean([per_seed[str(s)]["host_ceiling"]["generalization"]
                                             for s in seeds])),
        },
        "best_operating_point_per_seed": {
            "key": best_field(["key"]),
            "readout": best_field(["readout"]),
            "score_pearson_sim_Strue": best_field(["score_pearson_sim_Strue"]),
            "pearson_W_vs_rawcounts": best_field(["pearson_W_vs_rawcounts"]),
            "g2_generalization_graded": [per_seed[str(s)]["best_operating_point"]
                                         ["g2_generalization"]["graded"]["accuracy"] for s in seeds],
            "gates": best_field(["gates"]),
        },
        "collapsed_baseline_reference": {
            "stageW_pearson_simW_vs_Strue": -0.026, "diffusion_pearson": -0.024,
            "pearson_W_vs_rawcounts": 0.062, "generalization": 0.237,
            "note": "from 2026-06-11-learned-graded-embedding-diagnosis.md (20-cycle, no de-saturation)",
        },
        "elapsed_total_s": time.time() - t_all,
    }

    print(f"\n{'='*82}", flush=True)
    print(f"  DE-SATURATE FIX-TEST SUMMARY", flush=True)
    print(f"{'='*82}", flush=True)
    print(f"  CONSENSUS VERDICT: {consensus}", flush=True)
    for s in seeds:
        bop = per_seed[str(s)]["best_operating_point"]
        print(f"  seed {s}: verdict={verdicts[str(s)]} | best={bop['key']} ({bop['readout']}) "
              f"P(sim,Strue)={bop['score_pearson_sim_Strue']:+.3f} "
              f"Pearson(W,counts)={bop['pearson_W_vs_rawcounts']:+.3f} "
              f"gen={bop['g2_generalization']['graded']['accuracy']:.3f}", flush=True)
    print(f"  HOST CEILING (target): Pearson {summary['host_ceiling_mean']['pearson_vs_Strue']:+.3f} "
          f"gen {summary['host_ceiling_mean']['generalization']:.3f}", flush=True)
    print(f"  COLLAPSED BASELINE (20cyc, no de-sat): Pearson(W,counts)=+0.062 "
          f"P(sim,Strue)=-0.026 gen=0.237", flush=True)
    print(f"  Total elapsed: {summary['elapsed_total_s']:.1f}s", flush=True)
    print(f"{'='*82}\n", flush=True)

    out_data = {"summary": summary, "per_seed": per_seed}
    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        args.out = os.path.join(raw_dir, f"_lge_desaturate_seed{seeds[0]}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out_data


if __name__ == "__main__":
    main()
