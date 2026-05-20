"""Localisation diagnostic for the unified decisive run's
per_regime_advantage = 0.000 finding.

The decisive run (commit 3735fec) showed full_acc EXACTLY equals
uniform_ctrl_acc on every 9 (seed, N) cell. The mechanism hypothesis:
the deployment-time compositional readout's top-rate distribution is
BIMODAL -- either below BOTH thresholds (0.198 and 0.284 -> both arms
abstain) or above BOTH (both arms emit the SAME ranked[0]). The
(0.198, 0.284] region where the arms would disagree is statistically
empty.

This diagnostic:
1. Loads the unified substrate at seed 42 + the cached Phase-1
   checkpoint.
2. Encodes N=5 (noun, adj) pairs the same way the decisive eval does
   (sub-seed `seed + _UNIFIED_SUBSEED_OFFSET`; reuse the runner's
   pair-generation helpers via import).
3. Queries each encoded noun; records the top rate of the
   `_compositional_query_ranked` output.
4. Also queries ungroundable nouns (not encoded); records the top rate.
5. Reports the distribution: how many queries fall in each of:
     A: (-inf, 0.198]   (both arms abstain; same outcome)
     B: (0.198, 0.284]  (full emits; uniform_ctrl abstains; arms disagree)
     C: (0.284, inf)    (both arms emit; SAME ranked[0]; same outcome)
6. If A + C >> B (or B = 0), the bimodal hypothesis is confirmed.

This is a controller-only diagnostic; reuses the cached substrate; no
new training; no protected-set modification; no autograd. Output is
diagnostic numbers, not a verdict.
"""
from __future__ import annotations

import json
import os
import sys

# Add repo root to sys.path for `research.runners.*` imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import research.runners.unified_per_regime_monitor_runner as urr
from research.runners.abstention_gate_compositional_unified import (
    COMPOSITIONAL_UNIFIED_THRESHOLD,
)
from research.runners.abstention_gate_direct_unified import (
    DIRECT_UNIFIED_THRESHOLD,
)


def main() -> int:
    SEED = 42
    N = 5  # largest rung; matches the decisive eval's worst-case load

    # Build the unified substrate + load the Phase-1 checkpoint.
    print(
        f"=== UNIFIED LOCALISATION DIAGNOSTIC seed={SEED} N={N} ==="
    )
    bridge = urr._build_bridge_with_phase1_recipe(seed=SEED, tiny_synth=False)
    cache_path = urr._phase1_cache_path(
        "research/findings/raw/unified_per_regime/phase1", SEED
    )
    bridge.load_checkpoint(str(cache_path))
    urr._freeze_phase1_gates(bridge)

    # Match the decisive eval's pair generation (reuse runner helper).
    recipe_dims = urr._phase1_recipe(False)
    eval_pairs = urr._unified_compositional_pairs(seed=SEED, N=N)
    all_words, word_to_idx = urr._all_words_word_to_idx()
    n_words_for_orthogonal = max(
        urr._N_WORDS_ORTHOGONAL, len(all_words)
    )
    dims = {
        "n_lang_input": int(recipe_dims["n_lang_input"]),
        "n_per_pool": int(recipe_dims["n_per_pool"]),
        "n_fs_per_pool": int(recipe_dims["n_fs_per_pool"]),
        "sparsity": 0.05,
        "dt_ms": 0.5,
        "n_words_for_orthogonal": int(n_words_for_orthogonal),
    }
    recall_steps = 100
    encoding_steps = 100

    # Encode the N pairs (reuse runner helper; same protocol as decisive eval).
    print(f"\n--- Encoding {N} pairs via reused _encode_facts ---")
    for i, (noun, adj) in enumerate(eval_pairs):
        print(f"  pair[{i}]: ({noun}, {adj})")
    tags = urr._encode_facts(bridge, eval_pairs, dims, encoding_steps)
    print(f"  tags: {tags}")

    # Query each encoded noun (groundable compositional).
    groundable_rates = []
    print("\n--- GROUNDABLE compositional queries (encoded nouns) ---")
    for i, (noun, adj) in enumerate(eval_pairs):
        tag = tags[i] if i < len(tags) else None
        ranked = urr._compositional_query_ranked(
            bridge, noun, tag, dims, recall_steps
        )
        top_rate = float(ranked[0][1]) if ranked else 0.0
        top_word = ranked[0][0] if ranked else None
        groundable_rates.append(top_rate)
        bucket = _bucket(top_rate)
        print(
            f"  groundable {noun:>7} -> {adj:>7}: top={top_word:>10} "
            f"rate={top_rate:.4f}  [{bucket}]"
        )

    # Query ungroundable nouns (not encoded).
    encoded_nouns = {n for n, _ in eval_pairs}
    ungroundable_words = [
        w for w in urr.cpd.NOUN_VOCAB if w not in encoded_nouns
    ]
    ungroundable_rates = []
    print("\n--- UNGROUNDABLE compositional queries (un-encoded nouns) ---")
    for w in ungroundable_words:
        ranked = urr._compositional_query_ranked(
            bridge, w, None, dims, recall_steps
        )
        top_rate = float(ranked[0][1]) if ranked else 0.0
        top_word = ranked[0][0] if ranked else None
        ungroundable_rates.append(top_rate)
        bucket = _bucket(top_rate)
        print(
            f"  ungroundable {w:>7}: top={top_word:>10} "
            f"rate={top_rate:.4f}  [{bucket}]"
        )

    # Distribution report.
    print(
        f"\n--- DISTRIBUTION ANALYSIS (thresholds: "
        f"compositional_unified={COMPOSITIONAL_UNIFIED_THRESHOLD:.4f}, "
        f"direct_unified={DIRECT_UNIFIED_THRESHOLD:.4f}) ---"
    )
    all_rates = groundable_rates + ungroundable_rates
    a = sum(1 for r in all_rates if r <= COMPOSITIONAL_UNIFIED_THRESHOLD)
    b = sum(
        1 for r in all_rates
        if COMPOSITIONAL_UNIFIED_THRESHOLD < r <= DIRECT_UNIFIED_THRESHOLD
    )
    c = sum(1 for r in all_rates if r > DIRECT_UNIFIED_THRESHOLD)
    print(f"  A (rate <= 0.198, both abstain):                 {a:3d}/{len(all_rates)}")
    print(f"  B (0.198 < rate <= 0.284, ARMS DISAGREE):        {b:3d}/{len(all_rates)}")
    print(f"  C (rate > 0.284, both emit same answer):         {c:3d}/{len(all_rates)}")

    print(
        f"\n  bimodal hypothesis: B small + (A + C) large. "
        f"observed: A+C={a+c} B={b}"
    )
    if b == 0:
        print(
            "  --> B = 0: bimodal CONFIRMED (no rate falls in the "
            "between-thresholds region; arms cannot disagree by construction)."
        )
    elif b < (a + c) * 0.10:
        print(
            f"  --> B << A+C: bimodal partially confirmed (only {b} of "
            f"{a+b+c} = {100.0*b/(a+b+c):.1f}% in the between-thresholds region)."
        )
    else:
        print(
            f"  --> B is non-trivial ({b} of {a+b+c} = {100.0*b/(a+b+c):.1f}%); "
            "bimodal hypothesis FALSIFIED -- the arms could differ but "
            "happen to agree for other reasons; deeper investigation needed."
        )

    results = {
        "seed": SEED,
        "N": N,
        "compositional_unified_threshold": COMPOSITIONAL_UNIFIED_THRESHOLD,
        "direct_unified_threshold": DIRECT_UNIFIED_THRESHOLD,
        "groundable_rates": groundable_rates,
        "ungroundable_rates": ungroundable_rates,
        "n_in_bucket_A_below_compositional": a,
        "n_in_bucket_B_between_thresholds": b,
        "n_in_bucket_C_above_direct": c,
        "n_total": len(all_rates),
    }
    out = "research/findings/raw/unified_LOCALISATION_compositional_distribution.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")
    return 0


def _bucket(rate: float) -> str:
    if rate <= COMPOSITIONAL_UNIFIED_THRESHOLD:
        return "A (below 0.198; both abstain)"
    if rate <= DIRECT_UNIFIED_THRESHOLD:
        return "B (between thresholds; arms disagree)"
    return "C (above 0.284; both emit same)"


if __name__ == "__main__":
    sys.exit(main())
