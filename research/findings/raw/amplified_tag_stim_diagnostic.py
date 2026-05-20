"""Cheap controller-only diagnostic after the 6-architecture convergent
ceiling: test whether AMPLIFYING the engram tag stim during retrieve
produces a stronger compositional advantage than the 6th arc's
replay + PFC-frame mechanisms.

The 6th arc decisive (commit `cc8b791`) showed:
- N=2 advantage NEGATIVE -0.178 (replay+PFC hurts at low load)
- N=3 advantage POSITIVE +0.137 (replay+PFC helps at medium load)
- N=5 advantage marginal +0.056

The 11th adversarial review observed: replay at tiny_synth modifies
membrane potentials + eligibility trace but NOT cp_connections.data
(synaptic weights). So the 6th arc's positive advantage at N=3 comes
from dynamics-state effects, not synaptic consolidation.

This diagnostic tests the direct hypothesis: if the bound-adj
amplification is the load-bearing mechanism, then simply AMPLIFYING the
engram tag stim during retrieve (without replay + PFC-frame) should
produce a comparable or stronger advantage. If yes, the substrate-
level refinement direction is per-pathway transmission-gain modulation
on the engram-tag pathway. If no, the load-bearing mechanism is
elsewhere (probably dynamics priming).

Protocol:
- Load unified Phase-1 seed-42 checkpoint
- Encode 3 (noun, adj) pairs (matching N=3 from 6th arc, where the
  positive advantage emerged)
- For each pair, query the cue at THREE amplification levels:
  baseline 1500 pA, 2x = 3000 pA, 5x = 7500 pA
- Record top word + top rate + is-correct per query
- Report: which amplification level produces most correct answers?
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
from research.runners.compose_concept_engram import encode_concept_pair


def main() -> int:
    SEED = 42
    N = 3
    cache_dir = "research/findings/raw/unified_per_regime/phase1"

    print(f"=== AMPLIFIED-TAG-STIM DIAGNOSTIC seed={SEED} N={N} ===")

    # Build substrate + load Phase-1 cache + freeze gates
    bridge = urr._build_bridge_with_phase1_recipe(seed=SEED, tiny_synth=False)
    cache_path = urr._phase1_cache_path(cache_dir, SEED)
    bridge.load_checkpoint(str(cache_path))
    urr._freeze_phase1_gates(bridge)

    # Generate 3 compositional pairs (matches 6th arc N=3)
    eval_pairs = urr._unified_compositional_pairs(seed=SEED, N=N)
    print(f"\nEval pairs: {eval_pairs}")

    # Encode each pair via the standard helper
    recipe_dims = urr._phase1_recipe(False)
    all_words, word_to_idx = urr._all_words_word_to_idx()
    n_words_for_orthogonal = max(urr._N_WORDS_ORTHOGONAL, len(all_words))
    dims = {
        "n_lang_input": int(recipe_dims["n_lang_input"]),
        "n_per_pool": int(recipe_dims["n_per_pool"]),
        "n_fs_per_pool": int(recipe_dims["n_fs_per_pool"]),
        "sparsity": 0.05,
        "dt_ms": 0.5,
        "n_words_for_orthogonal": int(n_words_for_orthogonal),
    }
    tags = urr._encode_facts(bridge, eval_pairs, dims, encoding_steps=100)
    print(f"Encoded tags: {tags}")

    # Test three amplification levels
    amp_levels = [
        ("1x baseline", 1.0),
        ("2x amplified", 2.0),
        ("5x amplified", 5.0),
    ]

    results = {"pairs": eval_pairs, "tags": tags, "amplifications": {}}

    for label, amp_factor in amp_levels:
        print(f"\n--- {label} (amp_factor={amp_factor}) ---")
        amp_results = []
        n_correct = 0
        for i, (noun, adj) in enumerate(eval_pairs):
            tag = tags[i] if i < len(tags) else None
            # Drive cue + amplified tag stim, read lang_output
            # The runner's _compositional_query_ranked uses default
            # tag stim. We need to override the drive amplitude.
            # Use _compositional_query_ranked with the tag stim
            # amplitude implicitly multiplied; since the helper doesn't
            # accept an amp kwarg, run a minimal manual loop:
            from research.runners.unified_per_regime_monitor_runner import (
                _compositional_query_ranked,
            )
            # The helper hardcodes tag_drive_pA=1500.0; manually run an
            # equivalent loop with amplified amplitude.

            # Drive the cue noun
            from research.runners.concept_pool_demo import (
                orthogonal_drive_pattern,
            )
            cue_pattern = orthogonal_drive_pattern(
                cue_idx=word_to_idx[noun],
                n_cues=n_words_for_orthogonal,
                n_lang_input=int(dims["n_lang_input"]),
                sparsity=float(dims["sparsity"]),
            )
            # Stim window: drive cue + amplified tag stim simultaneously
            stim_steps = 100
            drive_pA = 200.0
            tag_drive_pA = 1500.0 * amp_factor
            # Reset external input
            import numpy as _np
            try:
                import cupy as cp
                xp = cp
            except ImportError:
                xp = _np

            n_lang_input = int(dims["n_lang_input"])
            for step in range(stim_steps):
                # cue drive
                bridge.cp_external_input_current[:n_lang_input] = (
                    cue_pattern * drive_pA
                )
                if tag:
                    bridge.stimulate_tag(tag, drive_pA=float(tag_drive_pA))
                bridge._run_one_simulation_step()

            # Clear tag drive
            if tag:
                bridge.clear_tag_drive(tag)
            bridge.cp_external_input_current[:n_lang_input] = 0.0

            # Read lang_output via the standard helper (cosine match
            # vs each word pattern; reused from the runner)
            ranked = _compositional_query_ranked(
                bridge, noun, tag, dims, recall_steps=20
            )

            top_word = ranked[0][0] if ranked else None
            top_rate = float(ranked[0][1]) if ranked else 0.0
            is_correct = (top_word == adj)
            if is_correct:
                n_correct += 1

            print(
                f"  {noun:>6} -> {adj:>6} (target): "
                f"top={top_word:>8} rate={top_rate:.4f}  "
                f"{'CORRECT' if is_correct else 'WRONG'}"
            )
            amp_results.append({
                "noun": noun,
                "target_adj": adj,
                "top_word": top_word,
                "top_rate": top_rate,
                "is_correct": is_correct,
            })

        results["amplifications"][label] = {
            "amp_factor": amp_factor,
            "n_correct": n_correct,
            "n_total": len(eval_pairs),
            "per_query": amp_results,
        }
        print(f"  {label} TOTAL: {n_correct}/{len(eval_pairs)} correct")

    print("\n=== SUMMARY ===")
    print(json.dumps({
        label: {
            "amp_factor": data["amp_factor"],
            "n_correct": data["n_correct"],
            "n_total": data["n_total"],
        }
        for label, data in results["amplifications"].items()
    }, indent=2))

    out = "research/findings/raw/amplified_tag_stim_diagnostic.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
