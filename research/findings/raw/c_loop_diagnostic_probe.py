"""(c) loop diagnostic probe -- localise WHERE the loop fails.

Per the post-NEGATIVE adversarial reviewer's CLEAR verdict + the v2
smoke result (stim_tag + global-mean fixes don't help): the failure
mode is NOT free-running SWR or grounding-mismatch. Something more
fundamental about the SWR-driven cortex activity not carrying
sequence-specific information for the decoder.

DIAGNOSTIC: for ONE engram (specific tag name), measure:
(A) Post-SWR cortical activity similarity to the CORRECT stored
    engram's encoded cortex pattern (does the replay reactivate the
    right pattern?)
(B) Post-SWR cortical activity similarity to OTHER stored engrams'
    encoded cortex patterns (control: should be lower if reactivation
    is specific)
(C) Post-SWR cortical activity similarity to RANDOM word grounded
    symbols (control: should be lower than encoded patterns)

INTERPRETATION:
- HIGH (A) + LOW (B) + LOW (C) -> replay works, decoder/grounding fails
- LOW (A) + LOW (B) + LOW (C) -> replay doesn't reactivate
- HIGH (A) ~ HIGH (B) ~ HIGH (C) -> non-specific reactivation (replay
  drives all stored engrams equally)
- HIGH (A) ~ HIGH (B) > (C) -> non-specific across stored engrams but
  distinguishable from random words

Cheap CPU probe (~15-30 min). Reuses substrate cache from pillar n=98.
Reuses v1/v2 primitives byte-unchanged via import. NO autograd; NO
protected/frozen/moat module modified; no-confab moat 7/7.
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

from research.runners.generative_replay_loop import (
    capture_post_replay_cortical_activity,
)
from research.findings.raw.generative_replay_decisive import (
    _vocab_words, _load_substrate, _build_pool_idx_arr,
    _engram_tag_name, _encode_engram_for_sequence,
    set_sleep_gates, set_awake_gates, freeze_all_gates,
    SLOT_COUNT, SWR_STEPS, CAPTURE_STEPS, N_REPLAYS_PER_TAG,
)
from research.findings.raw.generative_replay_decisive_v2_stim_global import (
    trigger_swr_replay_with_stim,
    _grounded_and_common_from_activity_cache,
    _ground_activity_with_global_mean, DEFAULT_N_LANG_INPUT,
)
from research.findings.raw.generative_replay_sequence_vocab import (
    generate_k_stored_sequences,
)
from research.runners.consolidation_trainer import (
    run_concept_replay_phase,
)
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(_HERE, "c_loop_diagnostic_probe.json")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _capture_baseline_activity(bridge, pool_idx_arr, n_steps=50):
    """Capture cortex activity with NO drive (baseline; what does
    free-running look like?)."""
    return capture_post_replay_cortical_activity(
        bridge, pool_idx_arr, stim_steps=n_steps, zero_drive=True)


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print("=== (c) loop diagnostic probe -- localise failure mode ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print("  Measures post-SWR cortex similarity to: (A) correct "
          "stored engram pattern; (B) other stored engrams; (C) random "
          "vocab grounded symbols.", flush=True)

    seed = 42
    enable_adjective = True
    words = _vocab_words(enable_adjective=enable_adjective)
    K = 4  # small K so the diagnostic is fast
    print(f"\n  seed={seed}; V={len(words)}; K={K} stored sequences",
          flush=True)

    bridge = _load_substrate(seed, verbose=True)
    grounded, common, d_act = _grounded_and_common_from_activity_cache(
        seed, words)
    pool_idx_arr, n_pool_union = _build_pool_idx_arr(
        bridge, enable_adjective)
    print(f"  pool_union={n_pool_union}; d_act={d_act}", flush=True)

    # Encode K stored sequences (deterministic per seed).
    sequences = generate_k_stored_sequences(
        seed=seed, k=K, n_words=len(words),
        slot_count=SLOT_COUNT, vocab=words)
    tag_names = [_engram_tag_name(i) for i in range(K)]
    print(f"  sequences: {[list(s) for s in sequences]}", flush=True)

    # Encode each sequence as an engram + capture the cortex activity
    # DURING encoding (this is the "encoded cortex pattern" we'll
    # compare post-SWR activity to).
    encoded_cortex_patterns = {}
    for i, seq in enumerate(sequences):
        _encode_engram_for_sequence(
            bridge, seq, i, words, DEFAULT_N_LANG_INPUT)
        # After encoding, capture activity to record the "encoded
        # state" of cortex for this engram.
        encoded_cortex_patterns[tag_names[i]] = \
            _capture_baseline_activity(bridge, pool_idx_arr, n_steps=50)
        print(f"  encoded engram {tag_names[i]} for seq {list(seq)}; "
              f"cortex pattern mean={float(np.mean(encoded_cortex_patterns[tag_names[i]])):.4f}",
              flush=True)

    # Run Phase 1.3 consolidation.
    print(f"  running Phase 1.3 consolidation (n_replays={N_REPLAYS_PER_TAG} per tag)...",
          flush=True)
    set_sleep_gates(bridge)
    run_concept_replay_phase(
        bridge, tag_names=tag_names,
        n_replays_per_tag=N_REPLAYS_PER_TAG,
        burst_duration_ms=100, inter_burst_ms=50,
        drive_pA=100.0, randomize_order=True,
        rng=np.random.default_rng(seed + 2))
    set_awake_gates(bridge)
    freeze_all_gates(bridge)
    try:
        bridge.set_plasticity_gate("ca3_swr_burst", 0.0)
    except KeyError:
        pass

    # Baseline: cortex activity with no drive (post-consolidation).
    baseline_activity = _capture_baseline_activity(
        bridge, pool_idx_arr, n_steps=50)
    print(f"  baseline cortex activity (no drive) mean="
          f"{float(np.mean(baseline_activity)):.4f}", flush=True)

    # For each engram, run SWR with stim_tag, capture post-replay
    # cortex, and measure similarities.
    results_per_tag = []
    for chosen_idx, chosen_tag in enumerate(tag_names):
        print(f"\n  Diagnosing engram {chosen_tag} (seq "
              f"{list(sequences[chosen_idx])}):", flush=True)
        # Run SWR with explicit stim of this tag.
        trigger_swr_replay_with_stim(
            bridge, chosen_tag, n_steps=SWR_STEPS, drive_pA=200.0)
        # Capture post-replay cortex.
        post_replay_activity = capture_post_replay_cortical_activity(
            bridge, pool_idx_arr,
            stim_steps=CAPTURE_STEPS, zero_drive=True)
        print(f"    post-replay cortex mean="
              f"{float(np.mean(post_replay_activity)):.4f}",
              flush=True)

        # (A) Similarity to correct stored engram pattern.
        sim_correct = _cosine(
            post_replay_activity,
            encoded_cortex_patterns[chosen_tag])
        # (B) Similarity to OTHER stored engrams.
        sim_other_tags = [
            _cosine(post_replay_activity, encoded_cortex_patterns[t])
            for t in tag_names if t != chosen_tag]
        sim_other_mean = float(np.mean(sim_other_tags))
        # (C) Similarity to random vocab grounded symbols (compare in
        # raw activity space, not phasor space; use a different proxy).
        # Use baseline activity as a "noise floor" reference.
        sim_baseline = _cosine(post_replay_activity, baseline_activity)

        results_per_tag.append({
            "chosen_tag": chosen_tag,
            "chosen_seq": [str(w) for w in sequences[chosen_idx]],
            "post_replay_mean_rate": float(np.mean(post_replay_activity)),
            "sim_to_correct_engram_cortex": sim_correct,
            "sim_to_other_engrams_mean": sim_other_mean,
            "sim_to_other_engrams_individual": sim_other_tags,
            "sim_to_baseline_no_drive": sim_baseline,
        })
        print(f"    sim_to_correct_engram_cortex = {sim_correct:.4f}",
              flush=True)
        print(f"    sim_to_other_engrams_mean   = {sim_other_mean:.4f}",
              flush=True)
        print(f"    sim_to_baseline_no_drive    = {sim_baseline:.4f}",
              flush=True)

    # Aggregate + interpret.
    sim_correct_all = [r["sim_to_correct_engram_cortex"]
                       for r in results_per_tag]
    sim_other_all = [r["sim_to_other_engrams_mean"]
                     for r in results_per_tag]
    sim_baseline_all = [r["sim_to_baseline_no_drive"]
                        for r in results_per_tag]
    mean_correct = float(np.mean(sim_correct_all))
    mean_other = float(np.mean(sim_other_all))
    mean_baseline = float(np.mean(sim_baseline_all))

    print(f"\n=== AGGREGATE (K={K} stored engrams) ===", flush=True)
    print(f"  Mean similarity post-replay -> CORRECT engram cortex: "
          f"{mean_correct:.4f}", flush=True)
    print(f"  Mean similarity post-replay -> OTHER engrams (control): "
          f"{mean_other:.4f}", flush=True)
    print(f"  Mean similarity post-replay -> BASELINE no-drive (control): "
          f"{mean_baseline:.4f}", flush=True)

    print(f"\n=== INTERPRETATION ===", flush=True)
    selectivity = mean_correct - mean_other
    above_baseline = mean_correct - mean_baseline
    print(f"  Selectivity (correct - other) = {selectivity:.4f} "
          f"({'POSITIVE' if selectivity > 0.05 else 'WEAK/NONE'})",
          flush=True)
    print(f"  Above-baseline (correct - baseline) = {above_baseline:.4f} "
          f"({'POSITIVE' if above_baseline > 0.05 else 'WEAK/NONE'})",
          flush=True)
    if mean_correct > 0.5 and mean_other < 0.3:
        diagnosis = "REPLAY_SPECIFIC_DECODER_FAILS"
        print("  -> Replay genuinely reactivates the correct engram; "
              "the decoder/grounding pipeline doesn't read it. Need to "
              "fix the decoder, not the SWR mechanism.", flush=True)
    elif mean_correct > 0.5 and mean_other > 0.5:
        diagnosis = "REPLAY_NONSPECIFIC_REACTIVATES_ALL"
        print("  -> Replay reactivates ALL stored engrams equally "
              "(non-specific). The stim_tag drives the chosen tag but "
              "downstream dynamics propagate to all tagged ensembles. "
              "Need a sharper selection mechanism.", flush=True)
    elif mean_correct < 0.3 and mean_other < 0.3:
        diagnosis = "REPLAY_DOESNT_REACTIVATE"
        print("  -> Replay does NOT reactivate any specific engram "
              "pattern. The stim_tag + SWR window doesn't drive cortex "
              "to the encoded state. Deeper consolidation or different "
              "reactivation mechanism needed.", flush=True)
    else:
        diagnosis = "INTERMEDIATE_OR_UNCLEAR"
        print("  -> Intermediate signal; needs further investigation.",
              flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seed": seed,
        "V": len(words), "K": K, "tag_names": tag_names,
        "sequences": [[str(w) for w in s] for s in sequences],
        "encoded_cortex_pattern_means": {
            t: float(np.mean(p))
            for t, p in encoded_cortex_patterns.items()},
        "baseline_activity_mean": float(np.mean(baseline_activity)),
        "per_engram_results": results_per_tag,
        "aggregate": {
            "mean_sim_correct": mean_correct,
            "mean_sim_other": mean_other,
            "mean_sim_baseline": mean_baseline,
            "selectivity": selectivity,
            "above_baseline": above_baseline,
        },
        "diagnosis": diagnosis,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
