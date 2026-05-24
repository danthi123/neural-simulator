"""Diagnostic comparison: same multitag pair-retrieval probe on the
OPTION 3 (no-hippo, no-dlpfc) substrate.

If multitag PASSes on OPTION 3 but NOT dlpfc-extension: localizes
that the substrate extensions broke multitag specifically.
If multitag NEGATIVE on both: the probe's recipe differs from the
original 2026-05-14 validation recipe in some material way (e.g.,
different encoding parameters, different drive levels, engram-
tagging mechanism not used in original validation).

Reuses every primitive from multitag_pair_dlpfc_extension_probe
byte-unchanged except the substrate builder. ~3 min GPU.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.findings.raw.vocabulary_scaling_run import BAR
from research.findings.raw.generative_replay_decisive import (
    _vocab_words, set_sleep_gates, set_awake_gates, freeze_all_gates,
)
# Re-use the multitag primitives byte-unchanged.
from research.findings.raw.multitag_pair_dlpfc_extension_probe import (
    K_PAIRS, TOP_K_READOUT, N_REPEATS_PER_TAG, STIM_STEPS,
    STIM_DRIVE_PA, READOUT_STEPS, N_REPLAYS_PER_TAG,
    DEFAULT_N_LANG_INPUT, _engram_tag_name_pair,
    _generate_concept_pairs, _encode_pair_engram,
    _capture_lang_output_during_stim, _word_score,
)
# Build the OPTION 3 substrate (NO hippo, NO dlpfc).
from research.findings.raw.mode_unification_on_bio_brain_regions_probe import (
    _build_and_train as _build_and_train_option3,
    CACHE_DIR as OPTION3_CACHE_DIR,
)
from research.runners.consolidation_trainer import (
    run_concept_replay_phase,
)
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(
    _HERE, "multitag_pair_option3_diagnostic.json")


def _load_option3_substrate(seed: int, verbose: bool = True):
    """Load the OPTION 3 cached substrate (no hippo, no dlpfc).
    Uses _build_and_train which is kill-safe via cached checkpoint."""
    bridge, words, _ = _build_and_train_option3(
        seed=seed, smoke=False, verbose=verbose)
    # Re-freeze all known plasticity gates.
    for g in ("language_input_to_motor",
              "language_input_to_noun_pool",
              "language_input_to_verb_pool",
              "language_input_to_adjective_pool",
              "motor_to_language_output",
              "noun_pool_to_language_output",
              "verb_pool_to_language_output",
              "adjective_pool_to_language_output"):
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass
    return bridge


def run_one_seed(seed: int, verbose: bool = True) -> dict:
    print(f"\n--- seed {seed} (OPTION 3 substrate, no hippo, no dlpfc) ---",
          flush=True)
    words = _vocab_words(enable_adjective=True)
    n_lang_output = DEFAULT_N_LANG_INPUT

    bridge = _load_option3_substrate(seed, verbose=verbose)
    if verbose:
        print(f"  [seed {seed}] V={len(words)}; "
              f"n_lang_output={n_lang_output}; K_PAIRS={K_PAIRS}",
              flush=True)

    pairs = _generate_concept_pairs(seed, K_PAIRS, words)
    print(f"  [seed {seed}] pairs:", flush=True)
    for i, p in enumerate(pairs):
        print(f"    pair {i}: ({p[0]}, {p[1]})", flush=True)

    tag_names = []
    for pair_idx, pair in enumerate(pairs):
        tag, stats = _encode_pair_engram(
            bridge, pair, pair_idx, words)
        tag_names.append(tag)
    print(f"  [seed {seed}] {K_PAIRS} pair engrams committed",
          flush=True)

    # Try Phase 1.3 consolidation. The OPTION 3 substrate may not
    # have the consolidation gates; wrap in try/except just to be safe
    # (but the OPTION 3 substrate is built WITHOUT
    # enable_hippocampus_consolidation, so the gates may not exist).
    try:
        set_sleep_gates(bridge)
        run_concept_replay_phase(
            bridge, tag_names=tag_names,
            n_replays_per_tag=N_REPLAYS_PER_TAG,
            burst_duration_ms=100, inter_burst_ms=50,
            drive_pA=100.0, randomize_order=True,
            rng=np.random.default_rng(seed + 2))
        set_awake_gates(bridge)
        freeze_all_gates(bridge)
        consolidation_run = True
    except Exception as e:
        if verbose:
            print(f"  [seed {seed}] consolidation skipped: {e}",
                  flush=True)
        consolidation_run = False

    # Run the multitag readout.
    per_pair_results = []
    n_both = 0
    n_at_least_one = 0
    for pair_idx, (pair, tag_name) in enumerate(zip(pairs, tag_names)):
        avg_counts = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = _capture_lang_output_during_stim(
                bridge, tag_name, n_lang_output,
                stim_steps=STIM_STEPS, readout_steps=READOUT_STEPS,
                drive_pA=STIM_DRIVE_PA)
            avg_counts = avg_counts + counts.astype(np.float64)
        avg_counts = avg_counts / N_REPEATS_PER_TAG

        word_scores = {w: _word_score(avg_counts, w, n_lang_output, words)
                       for w in words}
        topK = sorted(word_scores.items(),
                       key=lambda x: x[1], reverse=True)[:TOP_K_READOUT]
        topK_words = [w for w, s in topK]

        a_in = (pair[0] in topK_words)
        b_in = (pair[1] in topK_words)
        both = (a_in and b_in)
        if both:
            n_both += 1
        if a_in or b_in:
            n_at_least_one += 1

        per_pair_results.append({
            "pair_idx": pair_idx, "tag_name": tag_name,
            "pair_A": pair[0], "pair_B": pair[1],
            "topK_words": topK_words,
            "topK_scores": [round(s, 4) for w, s in topK],
            "A_in": a_in, "B_in": b_in, "both_in": both,
        })
        print(f"    pair {pair_idx} ({pair[0]}, {pair[1]}): "
              f"top{TOP_K_READOUT}={topK_words}; both={both}",
              flush=True)

    both_acc = n_both / K_PAIRS
    at_least_acc = n_at_least_one / K_PAIRS
    print(f"\n  [seed {seed}] both_in_top5: {n_both}/{K_PAIRS}"
          f" = {both_acc:.3f}", flush=True)
    print(f"  [seed {seed}] at_least_one_in_top5: "
          f"{n_at_least_one}/{K_PAIRS} = {at_least_acc:.3f}",
          flush=True)
    return {
        "seed": seed,
        "consolidation_run": consolidation_run,
        "n_both_in_topK": n_both,
        "n_at_least_one_in_topK": n_at_least_one,
        "both_accuracy": both_acc,
        "at_least_one_accuracy": at_least_acc,
        "per_pair": per_pair_results,
    }


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print("=== multitag pair-retrieval diagnostic on OPTION 3 "
          "(no-hippo, no-dlpfc) substrate ===", flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  bar={BAR}; K_PAIRS={K_PAIRS}; TOP_K_READOUT={TOP_K_READOUT}",
          flush=True)
    print("  Diagnostic: if multitag PASSes here but NOT on dlpfc-"
          "extension, the substrate extensions broke it.", flush=True)
    print("  If multitag NEGATIVE both: the probe recipe differs "
          "from original 2026-05-14 multitag validation.", flush=True)

    SEEDS = [42, 43, 44]
    seed_results = []
    t0 = time.time()
    for seed in SEEDS:
        r = run_one_seed(seed, verbose=True)
        seed_results.append(r)
    total_min = (time.time() - t0) / 60

    both_accs = [r["both_accuracy"] for r in seed_results]
    at_least_accs = [r["at_least_one_accuracy"] for r in seed_results]
    both_mean = float(np.mean(both_accs))
    at_least_mean = float(np.mean(at_least_accs))

    print(f"\n=== MULTI-SEED AGGREGATE ===", flush=True)
    print(f"  BOTH in top-{TOP_K_READOUT} mean: {both_mean:.3f} "
          f"per-seed={both_accs}", flush=True)
    print(f"  At-least-one in top-{TOP_K_READOUT} mean: "
          f"{at_least_mean:.3f} per-seed={at_least_accs}",
          flush=True)
    print(f"\n  COMPARISON vs dlpfc-extension:", flush=True)
    print(f"    dlpfc-extension both: 0.083; this OPTION 3 both: "
          f"{both_mean:.3f}", flush=True)
    if both_mean >= BAR:
        diagnosis = "DLPFC_EXTENSIONS_BROKE_MULTITAG"
        print("  Multitag PASSes on OPTION 3 (no hippo/dlpfc) -- the "
              "substrate extensions specifically broke it.",
              flush=True)
    elif both_mean > 0.3:
        diagnosis = "DLPFC_EXTENSIONS_DEGRADED_MULTITAG"
        print("  Multitag PARTIAL on OPTION 3 -- the substrate "
              "extensions further degraded an already-imperfect recipe.",
              flush=True)
    else:
        diagnosis = "RECIPE_DIFFERS_FROM_ORIGINAL_VALIDATION"
        print("  Multitag NEGATIVE on OPTION 3 too -- the probe "
              "recipe differs from the original 2026-05-14 validation "
              "(different encoding, drive, or readout parameters).",
              flush=True)

    out = {
        "backend": backend_name, "gpu": gpu,
        "seeds": SEEDS, "K_PAIRS": K_PAIRS,
        "TOP_K_READOUT": TOP_K_READOUT, "bar": BAR,
        "substrate": "OPTION3_no_hippo_no_dlpfc (pillar n=96)",
        "per_seed": seed_results,
        "aggregate": {
            "both_in_topK_mean": both_mean,
            "at_least_one_mean": at_least_mean,
        },
        "comparison_to_dlpfc_extension_both_mean": 0.083,
        "diagnosis": diagnosis,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nTotal wall-clock: {total_min:.2f} min", flush=True)
    print(f"Wrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
