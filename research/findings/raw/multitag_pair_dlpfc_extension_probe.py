"""Multitag pair-retrieval probe on the dlpfc-extension substrate.

Tests the project's validated MULTITAG mechanism (originally 90%
multi-seed on v14/v16 per the 2026-05-14 finding) on the dlpfc-
extension substrate (pillar n=98). The substrate now includes
hippocampus + dlpfc_wm; this probe verifies the validated multitag
retrieval primitive is intact under those component additions.

CONTEXT (the rationale informed by tonight's findings):
- The (c) generative-replay loop NEGATIVE (n=99) + sequence-multitag
  NEGATIVE characterised that the substrate doesn't store SEQUENTIAL
  slot-position structure
- BUT the validated multitag mechanism (for SIMULTANEOUS concept
  pairs) is what the substrate DOES support
- This probe re-validates multitag on the dlpfc-extension substrate
  (with hippocampus + dlpfc_wm present) -- if PASS, the substrate
  retains the validated conversational retrieval primitive

DESIGN (mirrors the 2026-05-14 multitag recipe):
- Encode K=8 concept-concept pairs as engrams on the dlpfc-extension
  substrate
- For each pair (A, B): drive lang_input(A) + lang_input(B)
  simultaneously; commit engram with top_k=100 region_filter=ca3
- Test: stim each engram; read lang_output cosine to each vocab word;
  check if BOTH A AND B appear in top-5 cosine match

PRE-REGISTERED reading (fixed; never tuned):
- MULTITAG_PASS_ON_DLPFC: multi-seed-mean stim-recall accuracy
  (both A AND B in top-5) >= 0.80 -- matches the validated v14/v16
  multitag bar; confirms substrate retains validated retrieval
  primitive under (c)-build's component additions
- MULTITAG_NEGATIVE_ON_DLPFC: below 0.80 multi-seed; the substrate
  extensions perturbed the multitag mechanism (biology-translatable:
  hippocampus + dlpfc_wm presence breaks the validated retrieval)

Reuses every primitive byte-unchanged; no protected/frozen/moat
module modified; no autograd; no-confab moat must stay 7/7 green.
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

from research.findings.raw.vocabulary_scaling_run import (
    BAR, SEEDS,
)
from research.findings.raw.generative_replay_decisive import (
    _vocab_words, _load_substrate,
    set_sleep_gates, set_awake_gates, freeze_all_gates,
)
from research.runners.consolidation_trainer import (
    run_concept_replay_phase,
)
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(
    _HERE, "multitag_pair_dlpfc_extension_probe.json")

# Pre-registered config matching the validated 2026-05-14 multitag
# recipe.
K_PAIRS = 8
TOP_K_READOUT = 5
N_REPEATS_PER_TAG = 5
STIM_STEPS = 100
STIM_DRIVE_PA = 1500.0
READOUT_STEPS = 50
N_REPLAYS_PER_TAG = 8
DEFAULT_N_LANG_INPUT = 2048


def _engram_tag_name_pair(pair_idx: int) -> str:
    return f"multitag_pair_{pair_idx:03d}"


def _generate_concept_pairs(seed: int, k: int, vocab):
    """Generate K deterministic (A, B) pairs from vocab; both A and
    B distinct."""
    rng = np.random.default_rng(seed + 11)
    n_vocab = len(vocab)
    pairs = []
    seen = set()
    while len(pairs) < k:
        a, b = rng.choice(n_vocab, size=2, replace=False)
        key = tuple(sorted([int(a), int(b)]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((vocab[a], vocab[b]))
    return pairs


def _encode_pair_engram(bridge, pair, pair_idx: int, words):
    """Drive lang_input(A) + lang_input(B) simultaneously; commit
    engram with top_k=100, region_filter=ca3. Returns engram tag
    stats."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import orthogonal_drive_pattern

    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)
    word_to_idx = {w: i for i, w in enumerate(words)}
    n_lang_input = len(lang_input_idx)
    tag_name = _engram_tag_name_pair(pair_idx)

    # Reset.
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    # Drive both words simultaneously via orthogonal drive patterns
    # (add the patterns together so both fire concurrently).
    drive_a = orthogonal_drive_pattern(
        cue_idx=word_to_idx[pair[0]], n_cues=len(words),
        n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=0.05)
    drive_b = orthogonal_drive_pattern(
        cue_idx=word_to_idx[pair[1]], n_cues=len(words),
        n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=0.05)
    drive_combined = cp.asarray(drive_a + drive_b, dtype=cp.float32)
    bridge.cp_external_input_current[lang_input_arr] = drive_combined

    # Begin engram recording + drive for N steps.
    bridge.start_engram_recording(tag_name)
    encoding_steps = 60  # ~2x a single-word encoding to capture both
    for _ in range(encoding_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    stats = bridge.commit_engram_tag(
        tag_name, top_k=100, region_filter=["ca3"])

    # Clear drive.
    bridge.cp_external_input_current[lang_input_arr] = 0.0
    return tag_name, stats


def _capture_lang_output_during_stim(bridge, tag_name: str,
                                       n_lang_output,
                                       stim_steps=STIM_STEPS,
                                       readout_steps=READOUT_STEPS,
                                       drive_pA=STIM_DRIVE_PA):
    """Stim engram + capture lang_output spike counts per neuron."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_output_idx = list(rm.indices("language_output"))
    lang_output_arr = cp.asarray(lang_output_idx, dtype=cp.int64)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    bridge.stimulate_tag(tag_name, drive_pA=drive_pA, additive=False)
    lang_counts = cp.zeros(n_lang_output, dtype=cp.float32)
    for _ in range(stim_steps + readout_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[lang_output_arr]
        lang_counts = lang_counts + fired.astype(cp.float32)
    bridge.clear_tag_drive(tag_name)
    return cp.asnumpy(lang_counts)


def _word_score(lang_counts, word, n_lang_output, words):
    """Cosine to each word's orthogonal drive pattern at lang_output
    resolution."""
    from sim.text_embeddings import orthogonal_drive_pattern
    word_to_idx = {w: i for i, w in enumerate(words)}
    pattern = orthogonal_drive_pattern(
        cue_idx=word_to_idx[word], n_cues=len(words),
        n_neurons=n_lang_output, drive_max_pA=1.0, sparsity=0.05)
    a = lang_counts.astype(np.float64)
    b = pattern.astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_one_seed(seed: int, verbose: bool = True) -> dict:
    print(f"\n--- seed {seed} ---", flush=True)
    enable_adjective = True
    words = _vocab_words(enable_adjective=enable_adjective)
    n_lang_output = DEFAULT_N_LANG_INPUT  # 2048

    bridge = _load_substrate(seed, verbose=verbose)
    if verbose:
        print(f"  [seed {seed}] V={len(words)}; "
              f"n_lang_output={n_lang_output}; K_PAIRS={K_PAIRS}; "
              f"TOP_K_READOUT={TOP_K_READOUT}",
              flush=True)

    # Generate deterministic pairs.
    pairs = _generate_concept_pairs(seed, K_PAIRS, words)
    print(f"  [seed {seed}] pairs:", flush=True)
    for i, p in enumerate(pairs):
        print(f"    pair {i}: ({p[0]}, {p[1]})", flush=True)

    # Encode each pair as an engram (simultaneous lang_input drive).
    tag_names = []
    for pair_idx, pair in enumerate(pairs):
        tag, stats = _encode_pair_engram(
            bridge, pair, pair_idx, words)
        tag_names.append(tag)
    print(f"  [seed {seed}] {K_PAIRS} pair engrams committed",
          flush=True)

    # Phase 1.3 consolidation.
    print(f"  [seed {seed}] consolidating...", flush=True)
    set_sleep_gates(bridge)
    run_concept_replay_phase(
        bridge, tag_names=tag_names,
        n_replays_per_tag=N_REPLAYS_PER_TAG,
        burst_duration_ms=100, inter_burst_ms=50,
        drive_pA=100.0, randomize_order=True,
        rng=np.random.default_rng(seed + 2))
    set_awake_gates(bridge)
    freeze_all_gates(bridge)

    # For each pair: stim + read lang_output; check both A, B in top-K.
    per_pair_results = []
    n_both_in_topK = 0
    n_at_least_one_in_topK = 0

    for pair_idx, (pair, tag_name) in enumerate(
            zip(pairs, tag_names)):
        # Average lang_output across N_REPEATS_PER_TAG repeats.
        avg_counts = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = _capture_lang_output_during_stim(
                bridge, tag_name, n_lang_output,
                stim_steps=STIM_STEPS, readout_steps=READOUT_STEPS,
                drive_pA=STIM_DRIVE_PA)
            avg_counts = avg_counts + counts.astype(np.float64)
        avg_counts = avg_counts / N_REPEATS_PER_TAG

        # Score each vocab word against avg_counts; pick top-K.
        word_scores = {w: _word_score(avg_counts, w, n_lang_output, words)
                       for w in words}
        topK = sorted(word_scores.items(),
                       key=lambda x: x[1], reverse=True)[:TOP_K_READOUT]
        topK_words = [w for w, s in topK]

        a_in = (pair[0] in topK_words)
        b_in = (pair[1] in topK_words)
        both_in = (a_in and b_in)
        at_least_one = (a_in or b_in)
        if both_in:
            n_both_in_topK += 1
        if at_least_one:
            n_at_least_one_in_topK += 1

        per_pair_results.append({
            "pair_idx": pair_idx, "tag_name": tag_name,
            "pair_A": pair[0], "pair_B": pair[1],
            "topK_words": topK_words,
            "topK_scores": [round(s, 4) for w, s in topK],
            "A_in_topK": a_in, "B_in_topK": b_in,
            "both_in_topK": both_in,
        })
        print(f"    pair {pair_idx} ({pair[0]}, {pair[1]}): "
              f"top{TOP_K_READOUT}={topK_words}; A_in={a_in} "
              f"B_in={b_in}; both={both_in}", flush=True)

    both_acc = n_both_in_topK / K_PAIRS
    at_least_acc = n_at_least_one_in_topK / K_PAIRS
    print(f"\n  [seed {seed}] BOTH A AND B in top-{TOP_K_READOUT}: "
          f"{n_both_in_topK}/{K_PAIRS} = {both_acc:.3f}",
          flush=True)
    print(f"  [seed {seed}] at least one in top-{TOP_K_READOUT}: "
          f"{n_at_least_one_in_topK}/{K_PAIRS} = {at_least_acc:.3f}",
          flush=True)
    return {
        "seed": seed, "K_PAIRS": K_PAIRS,
        "TOP_K_READOUT": TOP_K_READOUT,
        "n_both_in_topK": n_both_in_topK,
        "n_at_least_one_in_topK": n_at_least_one_in_topK,
        "both_accuracy": both_acc,
        "at_least_one_accuracy": at_least_acc,
        "per_pair": per_pair_results,
    }


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print("=== multitag pair-retrieval probe on dlpfc-extension "
          "substrate ===", flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Pre-registered bar={BAR}; SEEDS={list(SEEDS)}; "
          f"K_PAIRS={K_PAIRS}; TOP_K_READOUT={TOP_K_READOUT}; "
          f"N_REPEATS_PER_TAG={N_REPEATS_PER_TAG}", flush=True)
    print("  Tests project's validated MULTITAG mechanism on the "
          "dlpfc-extension substrate.", flush=True)

    seed_results = []
    t0 = time.time()
    for seed in SEEDS:
        r = run_one_seed(seed, verbose=True)
        seed_results.append(r)
    total_min = (time.time() - t0) / 60
    print(f"\nTotal wall-clock: {total_min:.2f} min "
          f"(backend={backend_name})", flush=True)

    both_accs = [r["both_accuracy"] for r in seed_results]
    at_least_accs = [r["at_least_one_accuracy"] for r in seed_results]
    both_mean = float(np.mean(both_accs))
    at_least_mean = float(np.mean(at_least_accs))

    print(f"\n=== MULTI-SEED AGGREGATE ===", flush=True)
    print(f"  BOTH A AND B in top-{TOP_K_READOUT} multi-seed mean: "
          f"{both_mean:.3f} per-seed=[{', '.join(f'{a:.3f}' for a in both_accs)}]",
          flush=True)
    print(f"  At least one in top-{TOP_K_READOUT} multi-seed mean: "
          f"{at_least_mean:.3f} per-seed=[{', '.join(f'{a:.3f}' for a in at_least_accs)}]",
          flush=True)

    print(f"\n=== VERDICT (pre-registered) ===", flush=True)
    if both_mean >= BAR:
        verdict = "MULTITAG_PASS_ON_DLPFC"
        print(f"  MULTITAG PASSES on dlpfc-extension substrate "
              f"({both_mean:.3f} >= {BAR}). The validated multitag "
              f"mechanism is INTACT under hippocampus + dlpfc_wm "
              f"additions. Pending fresh adversarial review.",
              flush=True)
    else:
        verdict = "MULTITAG_BELOW_BAR_ON_DLPFC"
        print(f"  Multitag below 0.80 bar on dlpfc-extension "
              f"({both_mean:.3f} < {BAR}). The substrate extensions "
              f"may have degraded the validated retrieval primitive; "
              f"or this configuration differs from the original "
              f"v14/v16 multitag recipe in some material way; honest "
              f"characterisation needed.", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu,
        "seeds": list(SEEDS), "K_PAIRS": K_PAIRS,
        "TOP_K_READOUT": TOP_K_READOUT, "bar": BAR,
        "stim_drive_pA": STIM_DRIVE_PA,
        "n_repeats_per_tag": N_REPEATS_PER_TAG,
        "substrate": ("build_biological_brain_regions_v16_recipe_"
                       "WITH_HIPPO_AND_DLPFC_n=98"),
        "method": ("concept-concept pair engrams via D.14 with "
                    "SIMULTANEOUS lang_input drive; stim_tag "
                    "(validated reactivation); readout via lang_output "
                    "cosine; matches 2026-05-14 validated multitag "
                    "recipe"),
        "per_seed": seed_results,
        "aggregate": {
            "both_in_topK_mean": both_mean,
            "both_per_seed": both_accs,
            "at_least_one_mean": at_least_mean,
            "at_least_per_seed": at_least_accs,
        },
        "verdict": verdict,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
