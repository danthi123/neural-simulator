"""Multi-seed expansion of the pool-vs-lang_output readout diagnostic.

Single-seed result (seed 42 N=5; commit `c8f08e6`):
- lang_output readout: 0/5 correct
- pool readout: 1/5 correct
- Pool readout HELPS marginally (+1)

This multi-seed expansion repeats the same diagnostic at seeds 43 and
44 to resolve whether the +1 signal is real or noise. Decision rule:
- If pool readout aggregate > lang_output aggregate across all 3 seeds
  with a consistent direction (e.g., +1 or more per seed), the signal
  is real and motivates the full 8th arc.
- If the signal flips or shows large variance (e.g., +1 / 0 / -1
  across seeds), it was noise and honest closure of the design line
  is the right terminus.
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
    _unified_compositional_pairs,
    _all_words_word_to_idx,
    _N_WORDS_ORTHOGONAL,
    _compositional_query_ranked,
)
# Reuse the pool readout helper from the single-seed diagnostic.
from importlib import util as _import_util
_diag_path = os.path.join(
    _HERE, "pool_vs_langout_readout_diagnostic.py"
)
_spec = _import_util.spec_from_file_location("_diag", _diag_path)
_diag = _import_util.module_from_spec(_spec)
_spec.loader.exec_module(_diag)
_pool_firing_readout = _diag._pool_firing_readout


def run_one_seed(seed: int, N: int = 5):
    print(f"\n=== seed={seed} ===")
    bridge = _build_bridge_with_phase1_recipe(seed=seed, tiny_synth=False)
    cache_path = _phase1_cache_path(
        "research/findings/raw/unified_per_regime/phase1", seed
    )
    bridge.load_checkpoint(str(cache_path))
    _freeze_phase1_gates(bridge)

    eval_pairs = _unified_compositional_pairs(seed=seed, N=N)
    print(f"pairs: {eval_pairs}")

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

    n_lang_correct = 0
    n_pool_correct = 0
    per_query = []
    for i, (noun, adj) in enumerate(eval_pairs):
        tag = tags[i] if i < len(tags) else None
        lang_ranked = _compositional_query_ranked(
            bridge, noun, tag, dims, recall_steps=100
        )
        pool_ranked = _pool_firing_readout(
            bridge, noun, tag, dims, recall_steps=100
        )
        lang_top = lang_ranked[0][0] if lang_ranked else None
        pool_top = pool_ranked[0][0] if pool_ranked else None
        lc = (lang_top == adj)
        pc = (pool_top == adj)
        if lc: n_lang_correct += 1
        if pc: n_pool_correct += 1
        print(
            f"  {noun:>6} -> {adj:>6}: lang={lang_top:>8} {'OK' if lc else 'XX'}  "
            f"pool={pool_top:>8} {'OK' if pc else 'XX'}"
        )
        per_query.append({
            "noun": noun, "target": adj,
            "lang_top": lang_top, "lang_correct": lc,
            "pool_top": pool_top, "pool_correct": pc,
        })

    print(f"  seed={seed}: lang={n_lang_correct}/{N}  pool={n_pool_correct}/{N}")
    return {
        "seed": seed, "N": N,
        "lang_correct": n_lang_correct,
        "pool_correct": n_pool_correct,
        "per_query": per_query,
    }


def main():
    SEEDS = [42, 43, 44]
    N = 5

    print(f"=== MULTI-SEED POOL-vs-LANG_OUTPUT DIAGNOSTIC seeds={SEEDS} N={N} ===")
    per_seed = []
    for seed in SEEDS:
        result = run_one_seed(seed, N=N)
        per_seed.append(result)

    total_lang = sum(r["lang_correct"] for r in per_seed)
    total_pool = sum(r["pool_correct"] for r in per_seed)
    n_total = len(SEEDS) * N

    print(f"\n=== AGGREGATE ACROSS SEEDS ===")
    print(f"  lang_output: {total_lang}/{n_total} = {100.0*total_lang/n_total:.1f}%")
    print(f"  pool:        {total_pool}/{n_total} = {100.0*total_pool/n_total:.1f}%")
    print(f"  delta: pool - lang = {total_pool - total_lang}/{n_total}")

    deltas = [r["pool_correct"] - r["lang_correct"] for r in per_seed]
    print(f"\n  per-seed deltas: {deltas}")
    consistent_positive = all(d >= 0 for d in deltas)
    any_positive = any(d > 0 for d in deltas)
    if consistent_positive and total_pool > total_lang:
        print(
            "  --> Pool readout CONSISTENTLY >= lang_output across all 3 seeds; "
            "signal is real; 8th arc with pool readout is well-motivated."
        )
    elif any_positive and total_pool > total_lang:
        print(
            "  --> Pool readout AGGREGATE > lang_output but signal varies; "
            "weak positive; multi-seed run could be more informative; "
            "8th arc is plausible but not strongly motivated."
        )
    elif total_pool == total_lang:
        print(
            "  --> Pool readout AGGREGATE = lang_output; no clear signal; "
            "the +1 at seed 42 was likely noise; honest closure is the right terminus."
        )
    else:
        print(
            "  --> Pool readout AGGREGATE < lang_output; the +1 at seed 42 was "
            "noise; lang_output is actually doing more; honest closure is the right terminus."
        )

    results = {
        "seeds": SEEDS, "N": N,
        "per_seed": per_seed,
        "aggregate": {
            "lang_correct": total_lang, "pool_correct": total_pool,
            "n_total": n_total,
            "lang_pct": 100.0*total_lang/n_total,
            "pool_pct": 100.0*total_pool/n_total,
            "delta": total_pool - total_lang,
        },
        "per_seed_deltas": deltas,
    }
    out = "research/findings/raw/pool_vs_langout_multiseed.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
