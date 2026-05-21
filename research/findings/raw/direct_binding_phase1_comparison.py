"""Direct-binding diagnostic: compare 200-event vs 800-event Phase-1
checkpoints on direct retrieval of all 16 trained words.

The longer-Phase-1 6th-arc decisive showed direct_retain=0.833 at N=5
(seed 42) but only 0.250-0.333 at N=2/N=3 with much smaller n_direct
(3-4 queries). The variance could be sample size; this diagnostic
runs ALL 16 trained words for a 16-query measurement at each Phase-1
duration.

PROTOCOL:
For each Phase-1 checkpoint (200-event original; 800-event longer):
  For each of 16 trained words:
    Drive lang_input(word) via orthogonal_drive_pattern
    Measure firing rates over all 16 trained pools
    Check: is top pool == expected target pool?
  Report: n_correct / 16
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import research.runners.concept_pool_demo as cpd
from research.runners.unified_per_regime_monitor_runner import (
    _build_bridge_with_phase1_recipe,
    _phase1_cache_path,
    _phase1_recipe,
    _freeze_phase1_gates,
    _all_words_word_to_idx,
    _all_pool_regions,
    _direct_pool_target,
    _N_WORDS_ORTHOGONAL,
)


SEED = 42


def test_one_checkpoint(seed, cache_dir, label):
    print(f"\n=== {label} ===")
    bridge = _build_bridge_with_phase1_recipe(seed=seed, tiny_synth=False)
    cache_path = _phase1_cache_path(cache_dir, seed)
    print(f"Loading {cache_path}")
    bridge.load_checkpoint(str(cache_path))
    _freeze_phase1_gates(bridge)

    recipe_dims = _phase1_recipe(False)
    all_words, word_to_idx = _all_words_word_to_idx()
    n_words_for_orthogonal = max(_N_WORDS_ORTHOGONAL, len(all_words))
    all_pools = _all_pool_regions(enable_adjective=True)

    print(f"Querying {len(all_words)} trained words...")
    n_correct = 0
    per_word = []
    for word in all_words:
        try:
            target_pool = _direct_pool_target(word)
        except KeyError:
            continue
        per_pool = cpd.measure_pool_firing(
            bridge, word, all_pools,
            stim_steps=100,
            reset_steps=50,
            drive_pA=200.0,
            sparsity=0.05,
            n_lang_input=int(recipe_dims["n_lang_input"]),
            orthogonal_codes=True,
            n_words_for_orthogonal=int(n_words_for_orthogonal),
            word_to_idx=word_to_idx,
        )
        # Top pool
        top_pool = max(per_pool.items(), key=lambda x: x[1])[0]
        correct = (top_pool == target_pool)
        if correct:
            n_correct += 1
        per_word.append({
            "word": word,
            "target_pool": target_pool,
            "top_pool": top_pool,
            "top_rate": float(per_pool[top_pool]),
            "target_rate": float(per_pool[target_pool]),
            "correct": correct,
        })
        marker = "OK " if correct else "XX "
        print(
            f"  {marker} {word:>8} -> target {target_pool:>22}; "
            f"top={top_pool:>22} rate={per_pool[top_pool]:.3f} "
            f"(target_rate={per_pool[target_pool]:.3f})"
        )

    accuracy = n_correct / len(all_words)
    print(f"\n  {label}: {n_correct}/{len(all_words)} = {100.0*accuracy:.1f}% direct binding accuracy")
    return {
        "label": label,
        "cache_dir": cache_dir,
        "n_correct": n_correct,
        "n_total": len(all_words),
        "accuracy": accuracy,
        "per_word": per_word,
    }


def main():
    results = {}

    # 200-event baseline (existing cached checkpoint)
    results["200ev_baseline"] = test_one_checkpoint(
        SEED,
        "research/findings/raw/unified_per_regime/phase1",
        "200ev baseline Phase-1",
    )

    # 800-event longer training
    results["800ev_longer"] = test_one_checkpoint(
        SEED,
        "research/findings/raw/unified_per_regime/phase1_800ev",
        "800ev longer Phase-1",
    )

    print("\n=== AGGREGATE COMPARISON ===")
    for key, data in results.items():
        print(f"  {data['label']}: {data['n_correct']}/16 = {100.0*data['accuracy']:.1f}%")

    delta = results["800ev_longer"]["accuracy"] - results["200ev_baseline"]["accuracy"]
    print(f"\n  delta (800ev - 200ev) = {100.0*delta:+.1f}pp")
    if delta > 0.1:
        print(
            f"  --> Longer training HELPS direct binding substantially "
            f"(+{100.0*delta:.1f}pp). Multi-seed investment warranted."
        )
    elif delta > 0:
        print(
            f"  --> Longer training helps direct binding modestly "
            f"(+{100.0*delta:.1f}pp). Likely within noise."
        )
    elif delta == 0:
        print(
            "  --> Longer training NEUTRAL on direct binding. "
            "The single-seed 6th-arc N=5 direct_retain=0.833 was "
            "sample-variance, not a real effect."
        )
    else:
        print(
            f"  --> Longer training HURTS direct binding ({100.0*delta:+.1f}pp). "
            "Confirms over-training degrades multiple capabilities."
        )

    out = "research/findings/raw/direct_binding_phase1_comparison.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
