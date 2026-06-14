"""Phase 1 production build: the 32-bridge / 2,048-concept learned-graded cortex on the
curated Option-B substrate.

The build is a MINIMAL corpus-source swap on the VALIDATED ROUTE-A de-risk flow
(`phase1_composer_ab_derisk.run_seed --composer per-bridge`): every gate -- A (the WITHIN-bridge
conversational matrix PER bridge, on per-bridge composers at `per_bridge_D` = the route-A
vocabulary-independent architecture), B (within-bridge generalization + C1/C4 controls), X
(cross-bridge V-tag IDENTITY composition + Cx anti-cheat), C3 (the no-confab moat) -- and the whole
agent stack are reused VERBATIM. (Route A, not the ensemble's UNION composer, which would need
D~5500 at 32 bridges -- the route-B cost route A was chosen to avoid.) The only
new piece is `build_curated_bridge_corpus`: it takes a REAL semantic cluster (its 64 words from
`g20_vocab_spec_2048`) + the CURATED within-cluster sub-taxonomy (`g20_subtaxonomy_2048`, 8 semantic
sub-groups of 8) and produces a bridge corpus in the EXACT shape `build_bridge_corpus` returns -- so
the bridge's graded code makes `cluster.lion` ~ `cluster.tiger` (both in the curated `big_cats`
sub-group) MEANINGFUL rather than arbitrary. That meaningfulness is what the de-risk's synthetic
(arbitrary-sub-cluster) corpus could not give, and it is validated by the conversational matrix +
generalization gates (NOT the per-bridge G1/G2 gate, which is label-agnostic).

Mechanism, architecture (route A), vocab, and corpus-source (B) are all de-risked; this wires them
at scale. NO `sim/` edits (reuse-by-import + a module-attribute swap of the corpus builder).

Usage (cheap validation, then scale):
  SIM_BACKEND=cupy python -u -m research.runners.production_cortex_build \
      --mode full --seeds 42 --cortex learned --n-bridges 4 \
      --out research/findings/raw/_production_cortex_build_4bridge.json
  # ... GO -> scale to --n-bridges 32 multi-seed.
  # CPU smoke (plumbing): SIM_BACKEND=numpy ... --cortex synthetic --n-bridges 3 --skip-vtag
"""
from __future__ import annotations

import numpy as np

import research.runners.phase1_composer_ab_derisk as routeA
from research.runners.learned_graded_embedding_derisk_probe import build_toy_cooccurrence
from research.runners.multibridge_graded_derisk import _factor_subclusters
from research.runners.g20_vocab_spec_2048 import ALL_CLUSTERS_2048
from research.runners.g20_subtaxonomy_2048 import cluster_sublabels

# The production bridge order = the 32 real semantic clusters (vocab-spec order).
CLUSTER_NAMES = list(ALL_CLUSTERS_2048.keys())


def build_curated_bridge_corpus(cluster_name: str, n_concepts: int, seed: int, args) -> dict:
    """ONE new piece. Signature-compatible with `multibridge_graded_derisk.build_bridge_corpus`
    (so it is a drop-in for the ensemble flow), but the bridge's 64 members are the REAL cluster
    words grouped by the CURATED semantic sub-taxonomy instead of arbitrary synthetic sub-clusters.

    Reuses the VALIDATED `build_toy_cooccurrence` hub-mediated 8x8 fact structure (S_true,
    second_order_pairs, hub facts -- all unchanged), then RELABELS the synthetic members
    `c{sub}_m{i}` -> the i-th real word of curated sub-group `sub`. The hubs stay synthetic
    (the internal shared-context carrier). Result: same dict shape, but `cluster.dog` ~
    `cluster.cat` (curated `pets` sub-group) is graded-similar by construction."""
    n_sub, per_sub = _factor_subclusters(n_concepts, args.target_per_sub)   # 64 -> (8, 8)
    # distinct per-cluster seed (mirrors build_bridge_corpus's shard_seed; cluster index for variety)
    cl_idx = CLUSTER_NAMES.index(cluster_name) if cluster_name in CLUSTER_NAMES else 0
    cl_seed = seed * 1000 + cl_idx
    corpus = build_toy_cooccurrence(
        n_sub, per_sub, cl_seed,
        hub_facts_per_member=args.hub_facts_per_member,
        bridge_facts=args.bridge_facts,
        triplet_facts_per_cluster=args.triplet_facts_per_cluster)

    # real words ordered by curated sub-group: real_by_sub[s] = the per_sub words whose sublabel == s
    words, sublabels = cluster_sublabels(cluster_name)
    sublabels = np.asarray(sublabels, dtype=int)
    real_by_sub = [[w for w, sl in zip(words, sublabels) if sl == s] for s in range(n_sub)]
    for s in range(n_sub):
        assert len(real_by_sub[s]) == per_sub, (
            f"{cluster_name} sub-group {s} has {len(real_by_sub[s])} words, expected {per_sub}")

    # build the rename map: synthetic members are sub-major (c0_m0..c0_m{per_sub-1}, c1_m0, ...),
    # so member index k -> sub = k // per_sub, pos = k % per_sub (consistent with corpus['labels']).
    pfx = f"{cluster_name}."
    # two parallel relabelings of the synthetic structure onto the real words:
    #  - `rename` -> NAMESPACED (cluster.word), the globally-unique cross-bridge / composer vocab;
    #  - `local_rename` -> UN-namespaced (word), used for `_local` so the within-bridge learn AND the
    #    V-tag's GradedBridge idx (`gate_X_vtag` builds it from `_local["concepts"]`) match the local
    #    names the cross-bridge facts strip to (`mammals.hyena` -> `hyena`). A synthetic `_local` here was
    #    the cause of the gate_X_vtag `KeyError: 'hyena'` (idx had `c{N}_m{M}`, the cross-facts had real words).
    rename = {f"hub{c}": f"{pfx}__hub{c}" for c in range(n_sub)}
    local_rename = {f"hub{c}": f"__hub{c}" for c in range(n_sub)}
    for k, m in enumerate(corpus["members"]):
        s, i = k // per_sub, k % per_sub
        assert int(corpus["labels"][k]) == s, "build_toy_cooccurrence members not sub-major"
        rename[m] = pfx + real_by_sub[s][i]
        local_rename[m] = real_by_sub[s][i]

    concepts = [rename[c] for c in corpus["concepts"]]
    members = [rename[m] for m in corpus["members"]]
    facts = [tuple(rename[c] for c in f) for f in corpus["facts"]]
    # `_local`: the SAME structure with REAL un-namespaced names (the learn is name-agnostic/positional, so
    # the graded codes are unchanged; the cortex codebook is keyed off the namespaced top-level `members`).
    local = {
        "concepts": [local_rename[c] for c in corpus["concepts"]],
        "members": [local_rename[m] for m in corpus["members"]],
        "hubs": [local_rename[h] for h in corpus["hubs"]],
        "labels": corpus["labels"], "S_true": corpus["S_true"],
        "second_order_pairs": corpus["second_order_pairs"],
        "facts": [tuple(local_rename[c] for c in f) for f in corpus["facts"]],
        "member_index": {local_rename[m]: i for i, m in enumerate(corpus["members"])},
    }
    return {
        "shard": cluster_name, "n_sub": n_sub, "per_sub": per_sub,
        "concepts": concepts, "members": members,
        "labels": corpus["labels"], "S_true": corpus["S_true"],
        "second_order_pairs": corpus["second_order_pairs"],
        "facts": facts, "n_facts": len(facts),
        "member_index": {m: i for i, m in enumerate(members)},
        "_local": local,
    }


def main():
    # Swap the route-A flow's corpus source -> the curated real-cluster builder, and its bridge
    # roster -> the 32 real semantic clusters. run_seed() builds all_corpora via the (now curated)
    # `build_bridge_corpus` over SHARD_NAMES[:n_bridges]; everything downstream (per-bridge composers,
    # gate_A_routeA_per_bridge, gate_B, gate_X_vtag, the moat) is reused verbatim. Use --composer
    # per-bridge (the validated route-A architecture).
    routeA.build_bridge_corpus = build_curated_bridge_corpus
    routeA.SHARD_NAMES = CLUSTER_NAMES
    print(f"[production_cortex_build] curated Option-B substrate: {len(CLUSTER_NAMES)} real semantic "
          f"clusters available; corpus source = build_curated_bridge_corpus (vocab g20_vocab_spec_2048 "
          f"+ sub-taxonomy g20_subtaxonomy_2048); architecture = ROUTE A (per-bridge composers).", flush=True)
    routeA.main()


if __name__ == "__main__":
    main()
