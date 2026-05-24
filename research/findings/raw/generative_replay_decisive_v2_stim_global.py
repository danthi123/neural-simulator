"""(c) generative-replay decisive runner v2 -- DIAGNOSTIC REFINEMENT
incorporating the post-NEGATIVE adversarial reviewer's top-2 leverage
probes (per pillar n=99 reviewer's CLEAR verdict):

FIX 1 (sequence-specific SWR reactivation):
  v1 `trigger_swr_replay` only opened the ca3_swr_burst gate; the
  loop relied on free-running dynamics during the gate-open window
  which had NO per-trial sequence preference. v2 wraps the trigger
  with an explicit `bridge.stimulate_tag(chosen_tag, drive_pA=...)`
  before the gate-open window -- the validated D.14 engram-tagging
  reactivation mechanism. This drives the specific stored ensemble
  during replay.

FIX 2 (global-mean centring at decode time):
  v1's `run_generative_loop` used a local mean (single observation
  mean) when grounding the captured cortical activity for decoder
  input -- acknowledged in the v1 code's own NOTE comment as an
  approximation. The vocab was grounded with GLOBAL mean across all
  16 concepts. The mismatch may add noise. v2 uses the global mean
  consistently at decode time too (the same `common` vector that
  built the vocab).

Reuses v1 loop controller primitives byte-unchanged where possible
(encode_pfc_frame, decode_continuation, update_pfc_frame, the
substrate, the consolidation mechanism); only the trigger + the
ground-then-decode flow is augmented.

PRE-REGISTERED reading (fixed; never tuned -- inherits from (c) plan):
- PASS_V2_REFINED: multi-seed-mean >= 0.80 every K in {4,8,16} on
  the refined loop. The diagnostic fixes turn the NEGATIVE into a
  PASS; the loop's basic architecture WAS sound but needed sequence-
  specific reactivation + grounding-pipeline alignment.
- BOUNDARY_V2_PARTIAL: some K cells PASS some don't. The fixes help
  but aren't sufficient; further refinement needed.
- NEGATIVE_V2_PERSISTS: chance-level despite fixes. The failure mode
  is deeper than reactivation/grounding-alignment; consolidation may
  not capture sequence structure; or the (c) loop architecture
  fundamentally doesn't match this substrate's dynamics.

CHARACTERISATION not new pillar (until adversarial review of result):
the n=99 NEGATIVE pillar stands; this v2 is a focused refinement
probe to localise the failure further. If PASS, a new VALIDATED
pillar pending review. If NEGATIVE, a sharpened characterisation
extension to n=99.

Reuses every primitive byte-unchanged; no protected/frozen/moat
module modified; no autograd; no-confab moat must stay 7/7 green.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse v1 loop controller + decisive runner primitives byte-unchanged.
from research.runners.generative_replay_loop import (
    encode_pfc_frame, capture_post_replay_cortical_activity,
    decode_continuation, update_pfc_frame,
)
from research.findings.raw.generative_replay_decisive import (
    _build_grounded_vocab_phase_matrix, _engram_encode_sequence,
    engram_tag_name, set_sleep_gates, set_awake_gates,
    freeze_all_gates, _load_substrate, _score_completion,
    _trial_cache_path, _save_trial_cache, _load_trial_cache,
    SEEDS, K_LADDER, SLOT_COUNT, N_TRIALS_PER_K, BAR,
    SWR_STEPS, CAPTURE_STEPS, ENCODING_STEPS_PER_SLOT,
    N_REPLAYS_PER_TAG, ENGRAM_TOP_K, ENGRAM_REGION_FILTER,
    DRIVE_PA, VOCAB,
)
from research.findings.raw.generative_replay_sequence_vocab import (
    generate_k_stored_sequences,
)
from research.findings.raw.consolidation_trainer import (
    run_concept_replay_phase,
)
from research.runners.resonate_fire_fhrr import ResonateFireFHRR
from research.findings.raw.biologized_spiking_mode_unification_helpers import (
    gamma_slot_positions,
)
from research.findings.raw.biologized_spiking_mode_unification_parallel_matching_runner import (
    DERIV_SEED,
)
from research.findings.raw.vocabulary_scaling_run import N_DIM
from research.findings.raw.pattern_separation_grounding_probe import (
    make_deriver,
)
from research.runners.spiking_phasor_fhrr import phases_to_spikes
from sim.backend import get_backend, is_gpu_backend, to_host

OUT_DIR = os.path.join(_HERE, "generative_replay_decisive_v2_cache")
os.makedirs(OUT_DIR, exist_ok=True)


def _trial_cache_path_v2(seed: int, smoke: bool) -> str:
    tag = "smoke" if smoke else "full"
    return os.path.join(OUT_DIR, f"trials_v2_{tag}_seed{seed}.json")


def trigger_swr_replay_with_stim(bridge, chosen_tag: str,
                                    n_steps: int = 100,
                                    drive_pA: float = 200.0):
    """FIX 1: drive the specific stored ensemble during the SWR
    replay window via the validated D.14 stimulate_tag mechanism --
    biology-grounded sequence-specific reactivation."""
    # Open the validated Phase 1.3 SWR gate.
    bridge.set_plasticity_gate("ca3_swr_burst", 1.0)
    # Drive the specific tag's ensemble (validated D.14 mechanism).
    bridge.stimulate_tag(chosen_tag, drive_pA=drive_pA, additive=False)
    # Run the replay window.
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    # Clear the tag drive + close the gate.
    bridge.clear_tag_drive(chosen_tag)
    bridge.set_plasticity_gate("ca3_swr_burst", 0.0)
    return {"n_steps": n_steps, "stimulated_tag": chosen_tag,
            "drive_pA": drive_pA}


def _ground_activity_with_global_mean(activity_vector: np.ndarray,
                                       common: np.ndarray,
                                       d_act: int):
    """FIX 2: ground the captured activity using the SAME global mean
    that was used to build the vocab grounded symbols (consistency at
    decode time). Mirrors _ground_symbols in the parallel-matching
    runner exactly except input is one activity_vector instead of a
    dict."""
    deriver = make_deriver(N_DIM, d_act, DERIV_SEED)
    centred = activity_vector - common
    phases = deriver(centred)
    return phases_to_spikes(phases)


def run_one_seed_v2(seed: int, smoke: bool, xp, verbose: bool = True):
    """Refined per-seed runner with both fixes. Mirrors v1's
    _run_one_seed but uses the v2 trigger + global-mean grounding."""
    print(f"\n--- seed {seed} (v2 refined) ---", flush=True)
    cache_p = _trial_cache_path_v2(seed, smoke)
    if os.path.exists(cache_p):
        if verbose:
            print(f"  [seed {seed}] loading cached v2 trials ({cache_p})",
                  flush=True)
        return _load_trial_cache(cache_p)

    bridge, words, word_to_idx, common, grounded, vocab_phase_matrix, \
        max_diff, d_act, pool_idx_arr, positions, net = _load_substrate(
            seed, xp, verbose)
    if verbose:
        print(f"  [seed {seed}] substrate loaded; batched-vs-scalar "
              f"max-diff={max_diff:.2e}; d_act={d_act}; "
              f"V={len(words)}", flush=True)

    qrng = np.random.default_rng(seed + 1)
    n_trials = max(1, N_TRIALS_PER_K // 4) if smoke else N_TRIALS_PER_K
    K_ladder_now = [K_LADDER[0]] if smoke else K_LADDER
    per_K_results = {}
    t_seed = time.time()

    for K in K_ladder_now:
        print(f"  [seed {seed}, K={K}] generating + engramming + "
              f"consolidating + trials (v2: stim_tag + global mean)...",
              flush=True)
        t_K = time.time()
        sequences = generate_k_stored_sequences(
            seed=seed, k=K, n_words=len(words), slot_count=SLOT_COUNT,
            vocab=words)
        tag_names = [engram_tag_name(i) for i in range(K)]
        for i, seq in enumerate(sequences):
            _engram_encode_sequence(
                bridge, seq, words, word_to_idx, tag_names[i],
                encoding_steps_per_slot=ENCODING_STEPS_PER_SLOT,
                top_k=ENGRAM_TOP_K, region_filter=ENGRAM_REGION_FILTER,
                drive_pA=DRIVE_PA)
        set_sleep_gates(bridge)
        run_concept_replay_phase(
            bridge, tag_names=tag_names,
            n_replays_per_tag=N_REPLAYS_PER_TAG,
            burst_duration_ms=100.0, inter_burst_ms=50.0,
            drive_pA=100.0, randomize_order=True)
        set_awake_gates(bridge)
        freeze_all_gates(bridge)

        n_correct = 0
        trials = []
        for trial_idx in range(n_trials):
            seq_idx = int(qrng.integers(0, K))
            full_seq = sequences[seq_idx]
            cue_items = list(full_seq[:SLOT_COUNT - 1])
            cue_positions = positions[:SLOT_COUNT - 1]
            true_continuation = full_seq[SLOT_COUNT - 1]

            initial_C = encode_pfc_frame(
                cue_items, cue_positions, net, grounded)
            chosen_tag = tag_names[seq_idx]

            # FIX 1: stimulate the specific tag during SWR
            trigger_swr_replay_with_stim(
                bridge, chosen_tag,
                n_steps=SWR_STEPS, drive_pA=200.0)
            activity_host = capture_post_replay_cortical_activity(
                bridge, pool_idx_arr,
                stim_steps=CAPTURE_STEPS, zero_drive=True)

            # FIX 2: ground with GLOBAL mean (same `common` as vocab)
            activity_grounded = _ground_activity_with_global_mean(
                activity_host, common, d_act)

            decoded_idx = decode_continuation(
                activity_grounded, vocab_phase_matrix,
                positions[SLOT_COUNT - 1], net, xp)
            decoded_word = words[decoded_idx]

            correct = _score_completion(decoded_word, true_continuation)
            if correct:
                n_correct += 1
            trials.append({
                "trial_idx": trial_idx, "seq_idx": seq_idx,
                "cue": cue_items, "true_continuation": true_continuation,
                "decoded_word": decoded_word, "decoded_idx": decoded_idx,
                "correct": correct, "chosen_tag": chosen_tag,
            })

        per_K_results[str(K)] = {
            "K": K, "n_trials": n_trials, "n_correct": n_correct,
            "completion_accuracy": n_correct / n_trials,
            "trials": trials,
        }
        print(f"    [seed {seed} K={K}] completion_accuracy="
              f"{n_correct/n_trials:.3f} ({n_correct}/{n_trials}); "
              f"K-block wall {time.time()-t_K:.1f}s", flush=True)

    seed_result = {
        "seed": seed, "n_pool_union": pool_idx_arr.shape[0],
        "d_act": d_act, "batched_vs_scalar_max_diff": max_diff,
        "wall_clock_s": time.time() - t_seed,
        "per_k": {k: {kk: v for kk, v in d.items() if kk != "trials"}
                   for k, d in per_K_results.items()},
        "per_k_trials": {k: d["trials"] for k, d in per_K_results.items()},
    }
    _save_trial_cache(cache_p, seed_result)
    print(f"  [seed {seed} done in "
          f"{(time.time()-t_seed)/60:.1f} min, cached]", flush=True)
    return seed_result


def main():
    ap = argparse.ArgumentParser(
        description="(c) generative-replay decisive v2 -- stim_tag + "
                    "global mean refinement")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    smoke = bool(args.smoke)

    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print("=== (c) generative-replay decisive v2 (stim_tag + global "
          "mean) ===", flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print("  FIX 1: trigger_swr_replay_with_stim drives specific tag",
          flush=True)
    print("  FIX 2: decode-time grounding uses GLOBAL mean (same as vocab)",
          flush=True)
    if smoke:
        print(f"  *** SMOKE: K={K_LADDER[0]} + {N_TRIALS_PER_K // 4} "
              f"trials @ seed 42 only; numbers NOT propagated ***",
              flush=True)
    print(f"  bar={BAR} (frozen); SEEDS={list(SEEDS) if not smoke else [42]}; "
          f"K_LADDER={K_LADDER if not smoke else [K_LADDER[0]]}; "
          f"slot_count={SLOT_COUNT}; n_trials/K="
          f"{N_TRIALS_PER_K if not smoke else N_TRIALS_PER_K // 4}",
          flush=True)

    seeds = [42] if smoke else list(SEEDS)
    seed_results = []
    t0 = time.time()
    for seed in seeds:
        r = run_one_seed_v2(seed, smoke, xp, verbose=True)
        seed_results.append(r)
    total_min = (time.time() - t0) / 60
    print(f"\nTotal wall-clock: {total_min:.2f} min "
          f"(backend={backend_name})", flush=True)

    print(f"\n=== MULTI-SEED AGGREGATE (v2) ===", flush=True)
    K_ladder_now = [K_LADDER[0]] if smoke else K_LADDER
    agg = {}
    all_pass = True
    for K in K_ladder_now:
        accs = [r["per_k"][str(K)]["completion_accuracy"]
                for r in seed_results]
        mean = float(np.mean(accs))
        agg[str(K)] = {
            "completion_accuracy_mean": mean,
            "completion_accuracy_per_seed": [round(a, 3) for a in accs],
        }
        print(f"  K={K}: completion_accuracy mean={mean:.3f} "
              f"per-seed={[f'{a:.3f}' for a in accs]} "
              f"({'PASS' if mean >= BAR else '< {:.2f}'.format(BAR)})",
              flush=True)
        if mean < BAR:
            all_pass = False

    print(f"\n=== VERDICT (v2 refined) ===", flush=True)
    if smoke:
        verdict = "SMOKE_V2"
        print("  SMOKE -- numbers NOT propagated.", flush=True)
    elif all_pass:
        verdict = "PASS_V2_REFINED"
        print("  Multi-seed PASS at every K. The diagnostic fixes "
              "(stim_tag + global mean) turn the NEGATIVE into a PASS; "
              "(c) loop's basic architecture WAS sound but needed "
              "sequence-specific reactivation + grounding-pipeline "
              "alignment. NOT yet a capability claim -- pending fresh "
              "adversarial review.", flush=True)
    else:
        verdict = "NEGATIVE_V2_PERSISTS_OR_BOUNDARY"
        print("  At least one K cell remains below the bar. Honest "
              "characterisation: the fixes help (or don't) "
              "specifically per K; deeper investigation needed.",
              flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "smoke": smoke,
        "seeds": seeds, "k_ladder": K_ladder_now,
        "slot_count": SLOT_COUNT,
        "n_trials_per_k": (N_TRIALS_PER_K if not smoke
                            else N_TRIALS_PER_K // 4),
        "bar": BAR, "swr_steps": SWR_STEPS, "capture_steps": CAPTURE_STEPS,
        "encoding_steps_per_slot": ENCODING_STEPS_PER_SLOT,
        "n_replays_per_tag": N_REPLAYS_PER_TAG,
        "engram_top_k": ENGRAM_TOP_K,
        "engram_region_filter": list(ENGRAM_REGION_FILTER),
        "decoder": "parallel_population_matching_batched",
        "substrate": "build_biological_brain_regions_v16_recipe_WITH_HIPPO_AND_DLPFC_n=98",
        "fixes": ["stim_tag_during_swr", "global_mean_centring_at_decode"],
        "per_seed": [{k: v for k, v in r.items() if k != "per_k_trials"}
                      for r in seed_results],
        "aggregate": agg, "verdict": verdict,
        "wall_clock_minutes": total_min,
    }
    tag = "smoke" if smoke else "full"
    out_path = os.path.join(
        _HERE, f"generative_replay_decisive_v2_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
