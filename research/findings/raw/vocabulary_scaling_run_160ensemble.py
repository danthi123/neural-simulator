"""160-concept ensemble vocabulary scaling: per-bridge biologized
grounded-composition pipeline against the frozen 0.80 bar at K=16.

The next vocabulary tier the vocab-scaling design doc names, executed
per-bridge on the project's validated 5-bridge sparse-distributed
concept ensemble. Each of the 5 bridges (A nouns / B verbs / C
adjectives / D spatial / E functional, 32 concepts each, 160 unique
total) is built at the validated G.20 sparse defaults, trained via
the validated G.20 encoding, captured at M_OBS=16 observations per
concept, and tested through the biologized grounded-composition
pipeline at K_VOCAB=16 against the SAME frozen 0.80 bar, the SAME
multi-seed grid, and the SAME compositional loads {2, 3, 5} the
64-concept K=16 PASS arc used.

PRE-REGISTERED reading (fixed; never tuned):
- PASS: multi-seed mean >= 0.80 at every (bridge, load) cell -- 5
  bridges x 3 loads = 15 cells, all must clear. The activity-grounded
  biologized compositional capability extends to the 160-concept
  ensemble per-bridge at K=16.
- NEGATIVE: multi-seed mean below 0.80 at some (bridge, load) cell.
  The honest finding is which bridge or category misses, with the
  per-bridge breakdown reported.

SCOPE: per-bridge compositional capability only. Cross-bridge
composition (a single composite spanning multiple bridges' phasor
representations) is explicitly OUT OF SCOPE for this step -- a
separate larger design.

Reuse-by-import only: the validated sparse-pool bridge builder, the
validated G.20 encoding (`train_substrate`), the activity-capture
helper, the biologized grounded-composition pipeline (`run_pipeline`,
`recognition_accuracy`, `_ground_symbols`), the cache helpers, and
the 5-bridge vocabulary specification -- all byte-unchanged. The
genuinely-new code is this multi-bridge orchestration runner + Task
1's `bridge_vocab_and_patterns` helper. No protected/frozen/moat
module modified. No automatic differentiation. Plain ASCII.

`--smoke` runs a reduced-scale grounding check (2 bridges, tiny
bridge sizes, tiny vocab subset, few training events) -- toy numbers
NOT propagated as a result. Kill-safe per-bridge per-seed cache lets
a re-run resume from the next uncached (bridge, seed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The validated sparse-pool bridge builder (G.20 sparse) and the
# validated G.20 encoding -- byte-unchanged via reuse.
from research.runners.concept_pool_sparse_distributed import (
    build_sparse_pool_bridge,
)
# The validated training stage from the trained-substrate runner --
# byte-unchanged via reuse.
from research.findings.raw.vocabulary_scaling_run_trained import (
    train_substrate, N_TRAIN_EVENTS,
)
# The biologized pipeline + cache helpers, byte-unchanged via reuse.
from research.findings.raw.vocabulary_scaling_run import (
    BAR, LOADS, SEEDS, N_DIM, K_RECOG, N_TRIALS, SPARSITY,
    STIM_STEPS, RESET_STEPS,
    SMOKE_M_OBS, SMOKE_K, SMOKE_TRIALS, SMOKE_STIM_STEPS,
    capture_concept_activity, recognition_accuracy, run_pipeline,
    _save_cache, _load_cache,
)
# Task 1's helper, re-exported here so the Task 0 pin can verify the
# runner's public surface.
from research.findings.raw.vocabulary_scaling_160ensemble_helpers import (
    BRIDGE_NAMES, bridge_vocab_and_patterns,
)

# =====================================================================
# Pre-registered constants for THIS arc. The K_VOCAB=16 recipe (the
# recipe the 64-concept K=16 refined CAPABILITY PASS used) is fixed
# and not tuned per-bridge. Each bridge has 32 concepts; M_OBS = 16
# matches the K_VOCAB = K_RECOG-relevant 16-observation budget.
# =====================================================================
N_CONCEPTS_PER_BRIDGE = 32
K_VOCAB_TARGET = 16          # the K=16 PASS recipe
M_OBS_FULL = 16              # cache 16 observations per concept

# Substrate sizing per bridge (the validated G.20 sparse defaults the
# 64-concept thread used).
DEFAULT_N_LANG_INPUT = 8192
DEFAULT_N_SHARED_POOL = 2000
DEFAULT_N_SHARED_FS = 300
DEFAULT_PATTERN_SIZE = 100

# Smoke: 2 bridges, reduced sizes, tiny vocab subset, few train events.
SMOKE_BRIDGE_NAMES = ["bridgeA_nouns", "bridgeB_verbs"]
SMOKE_VOCAB_PER_BRIDGE = 8
SMOKE_N_LANG_INPUT = 512
SMOKE_N_SHARED_POOL = 512
SMOKE_N_SHARED_FS = 60
SMOKE_PATTERN_SIZE = 24
SMOKE_TRAIN_EVENTS = 10

CACHE_DIR = os.path.join(_HERE, "vocabulary_scaling_160ensemble_cache")


def _cache_path(bridge_name, seed, smoke):
    tag = "smoke" if smoke else "full"
    return os.path.join(CACHE_DIR, f"{tag}_{bridge_name}_seed{seed}.npz")


def _build_one_bridge(bridge_name, seed, smoke, verbose):
    """Build one bridge sized for its 32 concepts (or the smoke
    subset), generate its per-bridge patterns, and return
    (bridge, words, patterns, n_lang_input, n_concepts_trained,
    stim_steps)."""
    if smoke:
        n_lang = SMOKE_N_LANG_INPUT
        n_pool = SMOKE_N_SHARED_POOL
        n_fs = SMOKE_N_SHARED_FS
        k = SMOKE_PATTERN_SIZE
        stim_steps = SMOKE_STIM_STEPS
    else:
        n_lang = DEFAULT_N_LANG_INPUT
        n_pool = DEFAULT_N_SHARED_POOL
        n_fs = DEFAULT_N_SHARED_FS
        k = DEFAULT_PATTERN_SIZE
        stim_steps = STIM_STEPS

    bridge = build_sparse_pool_bridge(
        seed=seed, n_lang_input=n_lang, n_shared_pool=n_pool,
        n_shared_fs=n_fs, n_lang_output=n_lang, verbose=verbose)

    full_vocab, full_patterns = bridge_vocab_and_patterns(
        bridge_name, seed=seed, n_pool=n_pool, k=k)

    # Smoke uses a vocab subset; full uses all 32.
    n_concepts_trained = (SMOKE_VOCAB_PER_BRIDGE if smoke
                          else N_CONCEPTS_PER_BRIDGE)
    words = full_vocab[:n_concepts_trained]
    patterns = [full_patterns[i] for i in range(n_concepts_trained)]
    return bridge, words, patterns, n_lang, n_concepts_trained, stim_steps


def build_and_train_bridge_smoke(bridge_name, seed):
    """Helper for soundness tests: build + train one bridge at the
    smoke configuration and return ``(bridge, words, patterns)``.
    Exposed at module scope so tests/test_vocabulary_scaling_160ensemble.py
    can pin the train -> bridge handoff."""
    bridge, words, patterns, n_lang, n_concepts, _stim_steps = (
        _build_one_bridge(bridge_name, seed, smoke=True, verbose=False))
    n_train_events = SMOKE_TRAIN_EVENTS
    train_substrate(
        bridge, patterns, n_lang_input=n_lang,
        n_concepts=n_concepts, seed=seed,
        n_train_events=n_train_events, sparsity=SPARSITY,
        n_words_for_orthogonal=n_concepts, verbose=False)
    return bridge, words, patterns


def run_one_bridge_seed(bridge_name, seed, smoke=False):
    """Build + train + capture + run the biologized pipeline for one
    bridge at one seed. Returns the per-load result dict. Per-bridge
    per-seed cache: a re-run loads the cache and skips build+train+
    capture for the matching (bridge_name, seed).
    """
    print(f"\n--- bridge {bridge_name} seed {seed} ---", flush=True)
    m_obs = SMOKE_M_OBS if smoke else M_OBS_FULL
    k_recog = SMOKE_K if smoke else K_RECOG
    k_vocab = SMOKE_K if smoke else K_VOCAB_TARGET
    n_trials = SMOKE_TRIALS if smoke else N_TRIALS
    n_train_events = SMOKE_TRAIN_EVENTS if smoke else N_TRAIN_EVENTS

    cache_p = _cache_path(bridge_name, seed, smoke)
    if os.path.exists(cache_p):
        acts, words, _patterns = _load_cache(cache_p)
        print(f"  [{bridge_name}/{seed}] loaded cached activity "
              f"({len(words)} concepts, "
              f"{acts[words[0]].shape[0]} obs/concept)", flush=True)
    else:
        t0 = time.time()
        bridge, words, patterns, n_lang, n_concepts, stim_steps = (
            _build_one_bridge(bridge_name, seed, smoke, verbose=True))
        print(f"  [{bridge_name}/{seed}] training "
              f"({n_concepts} concepts x {n_train_events} events)",
              flush=True)
        train_substrate(
            bridge, patterns, n_lang_input=n_lang,
            n_concepts=n_concepts, seed=seed,
            n_train_events=n_train_events, sparsity=SPARSITY,
            n_words_for_orthogonal=n_concepts, verbose=True)
        print(f"  [{bridge_name}/{seed}] capturing {m_obs} observations "
              f"per concept", flush=True)
        acts = capture_concept_activity(
            bridge, words, patterns, m_obs=m_obs, n_lang_input=n_lang,
            n_words_for_orthogonal=n_concepts, stim_steps=stim_steps,
            verbose=True)
        _save_cache(cache_p, acts, words, patterns)
        captured = float(np.mean([np.mean(acts[w] > 0.0)
                                  for w in words]))
        print(f"  [{bridge_name}/{seed}] trained + captured + cached "
              f"in {time.time() - t0:.1f}s (captured pool density "
              f"{captured:.4f})", flush=True)

    # Recognition (reused unchanged).
    d_act = acts[words[0]].shape[1]
    consolidated = {w: acts[w][:k_vocab].mean(axis=0) for w in words}
    rec_per_obs, rec_avg = recognition_accuracy(
        acts, words, consolidated, k_recog,
        np.random.default_rng(seed + 7))

    per_load = run_pipeline(seed, acts, words, LOADS, n_trials,
                            k_recog, k_vocab)

    for load in LOADS:
        e = per_load[load]
        print(f"    L={load}: int={e['integrated_accuracy']:.4f}  "
              f"comp-only={e['composition_only_accuracy']:.4f} "
              f"(n={e['n_composition_only']})", flush=True)
    print(f"    recognition: per-obs={rec_per_obs:.4f}  "
          f"temporally-avg={rec_avg:.4f}", flush=True)

    return {
        "bridge": bridge_name, "seed": seed, "smoke": bool(smoke),
        "k_vocab": k_vocab, "k_recog": k_recog,
        "n_concepts_tested": len(words),
        "recognition_per_observation": float(rec_per_obs),
        "recognition_temporally_averaged": float(rec_avg),
        "per_load": {str(load): {
            "integrated_accuracy":
                float(per_load[load]["integrated_accuracy"]),
            "composition_only_accuracy":
                float(per_load[load]["composition_only_accuracy"]),
            "n_composition_only":
                int(per_load[load]["n_composition_only"]),
            "effective_load":
                int(per_load[load]["effective_load"]),
        } for load in LOADS},
    }


def main():
    ap = argparse.ArgumentParser(
        description="160-concept ensemble vocab-scaling: per-bridge "
                    "biologized grounded-composition pipeline at K=16 "
                    "against the frozen 0.80 bar.")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced-scale grounding check (2 bridges, "
                         "tiny sizes, few train events) -- toy numbers "
                         "NOT propagated as a result")
    args = ap.parse_args()
    smoke = bool(args.smoke)

    bridges = SMOKE_BRIDGE_NAMES if smoke else BRIDGE_NAMES
    seeds = [42] if smoke else list(SEEDS)

    print("=== 160-concept ensemble vocab-scaling (per-bridge, K=16) ===",
          flush=True)
    if smoke:
        print("  *** SMOKE MODE: 2 bridges, reduced sizes, tiny vocab "
              "subset, few train events -- toy numbers, NOT a result ***",
              flush=True)
    print(f"bridges={bridges}; seeds={seeds}; loads={LOADS}; "
          f"bar={BAR}; K_VOCAB={SMOKE_K if smoke else K_VOCAB_TARGET}; "
          f"K_RECOG={SMOKE_K if smoke else K_RECOG}; "
          f"substrate=TRAINED per-bridge", flush=True)

    cell_results = []
    for bridge_name in bridges:
        for seed in seeds:
            cell_results.append(run_one_bridge_seed(
                bridge_name, seed, smoke=smoke))

    # --- Aggregate per (bridge, load) over seeds ----------------------
    print(f"\n=== PER-BRIDGE MULTI-SEED AGGREGATE ===", flush=True)
    print("              " +
          "  ".join(f"L={load} (int mean)" for load in LOADS), flush=True)
    per_bridge_agg = {}
    all_pass = True
    for bridge_name in bridges:
        per_bridge_agg[bridge_name] = {}
        row = [f"  {bridge_name:>18}:"]
        for load in LOADS:
            ints = [c["per_load"][str(load)]["integrated_accuracy"]
                    for c in cell_results
                    if c["bridge"] == bridge_name]
            m = float(np.mean(ints)) if ints else float("nan")
            per_bridge_agg[bridge_name][load] = {
                "mean_integrated": m,
                "per_seed_integrated": ints,
            }
            row.append(f"{m:.4f} {'>=' if m >= BAR else '<'}{BAR}")
            if m < BAR:
                all_pass = False
        print("  ".join(row), flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if smoke:
        verdict = "SMOKE"
        print("  SMOKE run -- toy numbers, not propagated as a result.",
              flush=True)
    elif all_pass:
        verdict = "ENSEMBLE_160CONCEPT_K16_PASS"
        print("  Every (bridge, load) cell clears the frozen 0.80 bar "
              "multi-seed. The activity-grounded biologized "
              "compositional capability at K=16 extends to the full "
              "160-concept 5-bridge ensemble per-bridge. Subject to a "
              "dedicated adversarial review before the capability "
              "claim.", flush=True)
    else:
        verdict = "ENSEMBLE_160CONCEPT_K16_BELOW_BAR"
        print("  Some (bridge, load) cell is below 0.80. The honest "
              "finding is which bridge / category misses; per-bridge "
              "breakdown above.", flush=True)

    out = {
        "seeds": seeds, "bridges": bridges, "loads": LOADS, "bar": BAR,
        "k_vocab": SMOKE_K if smoke else K_VOCAB_TARGET,
        "k_recog": SMOKE_K if smoke else K_RECOG,
        "n_trials": SMOKE_TRIALS if smoke else N_TRIALS,
        "n_train_events": SMOKE_TRAIN_EVENTS if smoke else N_TRAIN_EVENTS,
        "n_concepts_per_bridge_full": N_CONCEPTS_PER_BRIDGE,
        "n_dim": N_DIM, "substrate": "trained_per_bridge",
        "symbol_grounding": "meancenter_activity",
        "smoke": bool(smoke),
        "cell_results": cell_results,
        "per_bridge_aggregate": {b: {str(l): v for l, v in d.items()}
                                 for b, d in per_bridge_agg.items()},
        "verdict": verdict,
    }
    tag = "smoke" if smoke else "full"
    out_path = os.path.join(
        _HERE, f"vocabulary_scaling_run_160ensemble_{tag}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
