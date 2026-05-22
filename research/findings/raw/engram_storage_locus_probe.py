"""Storage-locus probe: does engram-tag stimulation reactivate the
bound adjective's POOL?

The difference-readout probe (2026-05-21) relocated the compositional
blocker from readout to storage-and-reactivation: engram-tag
stimulation produces no readable bound-attribute signal at the
language-output population. This probe localizes further -- it
measures the concept POOLS directly during tag stimulation, bypassing
the language-output readout entirely.

For each (noun, adjective) binding encoded as an engram tag:
- Stimulate the tag; accumulate per-pool firing across the readout
  window for ALL concept pools (motor / noun / verb / adjective).
- Report: which pool fires strongest overall; the rank of the bound
  adjective's pool; the rank of the cued noun's pool.

DIAGNOSTIC -- no PASS/FAIL bar. Pre-registered routing rule (fixed):
- If the bound adjective's pool fires strongest among adjective pools
  on a majority of bindings: storage IS capturing the binding; the
  gap is the pool -> language-output pathway. Next work targets that
  pathway.
- If the bound adjective's pool does NOT fire strongest: the engram
  tag did not capture the adjective at encoding. If the noun's pool
  dominates, the tag is noun-dominated -- next work targets the
  encoding mechanism (cue/teacher balance).

Reuse-by-import only: stimulate_tag, region_manager, _encode_facts,
the substrate builder -- all byte-unchanged. No protected/frozen/moat
module modified. No autograd. Controller-only; single seed 42; cached
200-event unified substrate.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.runners.unified_per_regime_monitor_runner import (
    _build_bridge_with_phase1_recipe,
    _phase1_cache_path,
    _phase1_recipe,
    _freeze_phase1_gates,
    _encode_facts,
    _all_words_word_to_idx,
    _all_pool_regions,
)
from research.runners.compose_retrieval_runner import _N_WORDS_ORTHOGONAL

SEED = 42
CACHE_DIR = "research/findings/raw/unified_per_regime/phase1"
STIM_STEPS = 100
FACTS = [("apple", "big"), ("river", "small"), ("dog", "hot"), ("cat", "cold")]

# pool name for a given concept word
_POOL_OF = {
    "apple": "noun_pool_APPLE", "river": "noun_pool_RIVER",
    "dog": "noun_pool_DOG", "cat": "noun_pool_CAT",
    "big": "adjective_pool_BIG", "small": "adjective_pool_SMALL",
    "hot": "adjective_pool_HOT", "cold": "adjective_pool_COLD",
}


def _pool_firing_during_tag_stim(bridge, tag_name, pools, drive_pA=1500.0,
                                   stim_steps=100):
    """Stimulate the engram tag; accumulate mean per-pool firing rate
    across the readout window. Mirrors lang_output_pattern_during_stim
    but measures concept POOLS, not language-output."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    rm = bridge.region_manager

    pool_arrs = {}
    for p in pools:
        idx = list(rm.indices(p))
        pool_arrs[p] = cp.asarray(idx, dtype=cp.int64)

    bridge.cp_external_input_current[:] = 0.0
    bridge.clear_tag_drive()
    for _ in range(30):
        bridge._run_one_simulation_step()

    bridge.stimulate_tag(tag_name, drive_pA=drive_pA, additive=False)
    accum = {p: 0.0 for p in pools}
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        firing = bridge.cp_firing_states
        for p in pools:
            accum[p] += float(cp.sum(firing[pool_arrs[p]].astype(cp.float32)))

    bridge.clear_tag_drive(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()

    # mean per-neuron firing rate per pool
    rates = {}
    for p in pools:
        n = len(pool_arrs[p])
        rates[p] = accum[p] / (stim_steps * max(1, n))
    return rates


def main():
    print("=== Engram storage-locus probe ===")
    print(f"seed={SEED}; cache={CACHE_DIR}; facts={FACTS}")

    bridge = _build_bridge_with_phase1_recipe(SEED, tiny_synth=False)
    cache_path = _phase1_cache_path(CACHE_DIR, SEED)
    print(f"Loading {cache_path}")
    bridge.load_checkpoint(str(cache_path))
    _freeze_phase1_gates(bridge)

    recipe_dims = _phase1_recipe(False)
    all_words, _ = _all_words_word_to_idx()
    dims = {
        "n_lang_input": int(recipe_dims["n_lang_input"]),
        "n_per_pool": int(recipe_dims["n_per_pool"]),
        "n_fs_per_pool": int(recipe_dims["n_fs_per_pool"]),
        "sparsity": 0.05, "dt_ms": 0.5,
        "n_words_for_orthogonal": max(_N_WORDS_ORTHOGONAL, len(all_words)),
    }

    tags = _encode_facts(bridge, FACTS, dims, encoding_steps=200)
    print(f"Encoded tags: {tags}")

    pools = _all_pool_regions(enable_adjective=True)
    adj_pools = [p for p in pools if p.startswith("adjective_pool_")]

    results = []
    n_bound_adj_top = 0
    n_noun_dominates = 0
    for i, (noun, adj) in enumerate(FACTS):
        tag = tags[i]
        rates = _pool_firing_during_tag_stim(
            bridge, tag, pools, drive_pA=1500.0, stim_steps=STIM_STEPS)

        ranked = sorted(rates.items(), key=lambda kv: -kv[1])
        top_pool, top_rate = ranked[0]
        bound_adj_pool = _POOL_OF[adj]
        cued_noun_pool = _POOL_OF[noun]

        # rank of the bound adjective pool among ALL pools and among adjective pools
        all_rank = [p for p, _ in ranked].index(bound_adj_pool) + 1
        adj_ranked = sorted(
            ((p, rates[p]) for p in adj_pools), key=lambda kv: -kv[1])
        adj_rank = [p for p, _ in adj_ranked].index(bound_adj_pool) + 1
        noun_rank = [p for p, _ in ranked].index(cued_noun_pool) + 1

        bound_adj_is_top_adj = (adj_rank == 1)
        noun_dominates = (noun_rank < adj_rank)
        n_bound_adj_top += int(bound_adj_is_top_adj)
        n_noun_dominates += int(noun_dominates)

        results.append({
            "tag": tag, "noun": noun, "adj": adj,
            "top_pool": top_pool, "top_rate": top_rate,
            "bound_adj_pool": bound_adj_pool,
            "bound_adj_rate": rates[bound_adj_pool],
            "bound_adj_rank_all": all_rank,
            "bound_adj_rank_among_adj_pools": adj_rank,
            "cued_noun_pool": cued_noun_pool,
            "cued_noun_rate": rates[cued_noun_pool],
            "cued_noun_rank_all": noun_rank,
            "bound_adj_is_top_adj_pool": bound_adj_is_top_adj,
            "noun_dominates_adj": noun_dominates,
        })
        print(f"  tag {tag} ({noun},{adj}): top_pool={top_pool}"
              f"({top_rate:.3f}) | bound_adj {bound_adj_pool} "
              f"rate={rates[bound_adj_pool]:.3f} rank_all={all_rank} "
              f"rank_among_adj={adj_rank} | cued_noun rate="
              f"{rates[cued_noun_pool]:.3f} rank_all={noun_rank}")

    n = len(FACTS)
    print(f"\n=== STORAGE-LOCUS RESULT (seed {SEED}; {n} bindings) ===")
    print(f"  bound adjective pool fires strongest among adjective pools: "
          f"{n_bound_adj_top}/{n}")
    print(f"  cued noun pool outranks bound adjective pool: "
          f"{n_noun_dominates}/{n}")

    if n_bound_adj_top > n // 2:
        locus = "POOL_LEVEL_OK_PATHWAY_GAP"
        print("  --> Storage CAPTURES the binding: the bound adjective pool "
              "reactivates on tag stim. The gap is the pool -> "
              "language-output readout pathway.")
    elif n_noun_dominates > n // 2:
        locus = "ENCODING_GAP_NOUN_DOMINATED"
        print("  --> The engram tag is NOUN-DOMINATED: the cued noun's pool "
              "outranks the bound adjective's pool. The gap is the encoding "
              "mechanism (cue/teacher balance).")
    else:
        locus = "ENCODING_GAP_DIFFUSE"
        print("  --> The engram tag does NOT selectively reactivate the bound "
              "adjective pool, and the noun does not dominate either -- the "
              "tag reactivation is diffuse. The gap is the encoding mechanism.")

    out = {
        "seed": SEED, "cache_dir": CACHE_DIR, "facts": FACTS, "tags": tags,
        "stim_steps": STIM_STEPS, "per_binding": results,
        "n_bound_adj_top": n_bound_adj_top,
        "n_noun_dominates": n_noun_dominates, "n": n, "locus": locus,
    }
    out_path = "research/findings/raw/engram_storage_locus_probe.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
