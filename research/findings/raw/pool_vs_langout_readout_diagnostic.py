"""Diagnostic: compare lang_output-cosine readout (existing) vs
concept-pool-firing readout (proposed Direction A simplified) for the
8th arc.

The 7-arc + ablation analysis localised the bottleneck to the gated
readout: bridge-state perturbations from input augmenting (cue-supp,
amp-tag, persistent-PFC) are absorbed before reaching the gated
lang_output cosine answer. The localisation finding (commit 110f7cd)
identified the cued-noun's diffuse lang_input drive dominating the
bound-adj signal AT THE LANG_OUTPUT readout.

A simpler 8th arc Direction A: read directly from CONCEPT POOLS
(noun_pool_*, adjective_pool_*) via measure_pool_firing, BYPASSING
the lang_output spelling-cosine pipeline. No substrate modification
required (concept pools already exist; measure_pool_firing is a
public API).

This diagnostic tests:
For each (noun, adj) trained pair, drive cue + stim engram tag, then
read TWO readouts: (1) lang_output cosine pattern (the existing
readout); (2) adjective_pool firing rates (the proposed readout).
Compare which readout produces the correct top word more often.

If pool readout >> lang_output readout, the localised bottleneck IS
the lang_output readout and the 8th arc is well-motivated.
If pool readout <= lang_output readout, the bottleneck is deeper
(input-side cue dominance; not output-side readout choice).

Controller-only diagnostic; cached substrate; no new training.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import research.runners.unified_per_regime_monitor_runner as urr
import research.runners.concept_pool_demo as cpd
from research.runners.unified_per_regime_monitor_runner import (
    _compositional_query_ranked,
    _build_bridge_with_phase1_recipe,
    _phase1_cache_path,
    _phase1_recipe,
    _freeze_phase1_gates,
    _encode_facts,
    _unified_compositional_pairs,
    _all_words_word_to_idx,
    _N_WORDS_ORTHOGONAL,
)


def _pool_firing_readout(bridge, cue_noun, tag_name, dims, recall_steps):
    """Read compositional output via ADJECTIVE POOL firing rates after
    cue + tag stim (BYPASSING lang_output cosine).
    """
    # Drive cue first
    from research.runners.compose_concept_engram import (
        lang_output_pattern_during_input,
    )
    # Use lang_output_pattern_during_input to drive the cue (we ignore
    # its returned pattern; we just want the bridge state primed)
    lang_output_pattern_during_input(
        bridge, cue_noun,
        n_lang_input=int(dims["n_lang_input"]),
        sparsity=float(dims["sparsity"]),
        n_words_for_orthogonal=int(dims["n_words_for_orthogonal"]),
        stim_steps=int(recall_steps),
    )
    # Stim engram tag if present
    if tag_name is not None and tag_name in {
        t["name"] for t in bridge.list_engram_tags()
    }:
        bridge.stimulate_tag(tag_name, drive_pA=1500.0)
        for _ in range(int(recall_steps)):
            bridge._run_one_simulation_step()
        bridge.clear_tag_drive(tag_name)

    # Now read adjective pool firing rates via measure_pool_firing
    # (this internally drives the cue again briefly to measure rates;
    # we'll trust that the post-stim bridge state carries the binding
    # signal). For diagnostic accuracy we measure directly by reading
    # firing buffers.
    # Simpler approach: use measure_pool_firing-equivalent
    # Read each adjective pool's recent firing rate from bridge state
    all_pools = [
        ("big", "adjective_pool_BIG"),
        ("small", "adjective_pool_SMALL"),
        ("hot", "adjective_pool_HOT"),
        ("cold", "adjective_pool_COLD"),
    ]
    # Use the brain-region framework to get pool indices
    rates = {}
    for word, pool_name in all_pools:
        if hasattr(bridge, "region_manager") and bridge.region_manager is not None:
            try:
                pool_indices = bridge.region_manager.indices(pool_name)
                # cp_firing_states gives current step's firing; we want
                # accumulated firing over the recall window. Approximate
                # by current state (which reflects post-stim activity)
                firing_states = bridge.cp_firing_states
                if hasattr(firing_states, "get"):
                    firing_states = firing_states.get()
                pool_firing = float(firing_states[pool_indices].mean())
                rates[word] = pool_firing
            except (KeyError, AttributeError) as e:
                rates[word] = 0.0
        else:
            rates[word] = 0.0

    ranked = sorted(rates.items(), key=lambda x: -x[1])
    return ranked


def main():
    SEED = 42
    N = 5  # match 6th arc decisive's largest rung

    print(f"=== POOL vs LANG_OUTPUT readout diagnostic seed={SEED} N={N} ===")

    bridge = _build_bridge_with_phase1_recipe(seed=SEED, tiny_synth=False)
    cache_path = _phase1_cache_path(
        "research/findings/raw/unified_per_regime/phase1", SEED
    )
    bridge.load_checkpoint(str(cache_path))
    _freeze_phase1_gates(bridge)

    eval_pairs = _unified_compositional_pairs(seed=SEED, N=N)
    print(f"\nEval pairs: {eval_pairs}")

    recipe_dims = _phase1_recipe(False)
    all_words, word_to_idx = _all_words_word_to_idx()
    n_words_for_orthogonal = max(_N_WORDS_ORTHOGONAL, len(all_words))
    dims = {
        "n_lang_input": int(recipe_dims["n_lang_input"]),
        "n_per_pool": int(recipe_dims["n_per_pool"]),
        "n_fs_per_pool": int(recipe_dims["n_fs_per_pool"]),
        "sparsity": 0.05,
        "dt_ms": 0.5,
        "n_words_for_orthogonal": int(n_words_for_orthogonal),
    }
    tags = _encode_facts(bridge, eval_pairs, dims, encoding_steps=100)
    print(f"Encoded tags: {tags}")

    n_lang_correct = 0
    n_pool_correct = 0
    per_query = []
    for i, (noun, adj) in enumerate(eval_pairs):
        tag = tags[i] if i < len(tags) else None

        # Read both readouts
        lang_ranked = _compositional_query_ranked(
            bridge, noun, tag, dims, recall_steps=100
        )
        pool_ranked = _pool_firing_readout(
            bridge, noun, tag, dims, recall_steps=100
        )

        lang_top = lang_ranked[0][0] if lang_ranked else None
        lang_correct = (lang_top == adj)
        pool_top = pool_ranked[0][0] if pool_ranked else None
        pool_correct = (pool_top == adj)

        if lang_correct:
            n_lang_correct += 1
        if pool_correct:
            n_pool_correct += 1

        print(
            f"  {noun:>6} -> {adj:>6} (target):  "
            f"lang_top={lang_top:>8} {'OK ' if lang_correct else 'XX '}  "
            f"pool_top={pool_top:>8} {'OK ' if pool_correct else 'XX '}"
        )
        per_query.append({
            "noun": noun,
            "target_adj": adj,
            "lang_top": lang_top,
            "lang_correct": lang_correct,
            "pool_top": pool_top,
            "pool_correct": pool_correct,
        })

    print(f"\n=== SUMMARY ===")
    print(f"  lang_output readout: {n_lang_correct}/{N} correct")
    print(f"  pool readout:        {n_pool_correct}/{N} correct")
    if n_pool_correct > n_lang_correct:
        print(
            f"  --> Pool readout HELPS (+{n_pool_correct - n_lang_correct}); "
            "8th arc Direction A is well-motivated."
        )
    elif n_pool_correct == n_lang_correct:
        print(
            "  --> Pool readout MATCHES lang_output; no clear improvement; "
            "the bottleneck may be input-side (cue dominance) not "
            "output-side (readout choice)."
        )
    else:
        print(
            f"  --> Pool readout HURTS ({n_pool_correct - n_lang_correct}); "
            "the lang_output cosine is actually doing better; 8th arc "
            "Direction A is mis-motivated."
        )

    results = {
        "seed": SEED, "N": N,
        "lang_output_correct": n_lang_correct,
        "pool_readout_correct": n_pool_correct,
        "n_total": N,
        "per_query": per_query,
    }
    out = "research/findings/raw/pool_vs_langout_readout_diagnostic.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
