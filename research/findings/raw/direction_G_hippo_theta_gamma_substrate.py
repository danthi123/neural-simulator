"""Direction G: theta-gamma substrate sequence storage on HIPPO-OPTION3
(v16-style + hippocampus + Phase 1.3 SWR consolidation pathways).

Per pillar n=104 BOUNDARY: both spatial (ec_context, Direction A) and
temporal (theta-gamma, Direction E Task 1) positional codes FAIL on
v16 cortical-only substrate (multi-seed strict top-1 0.25-0.33; below
0.80 bar; engram load-bearing but positional cue not). The diagnosis:
v16's weak concept-pool dynamics + lack of dedicated sequence-binding
region make the engram-based mechanism inadequate.

Direction G is the cheapest-falsifiable next test: same theta-gamma
mechanism from Direction E Task 1, but on HIPPO-OPTION3 substrate
(pillar n=97 builder; adds hippocampal trisynaptic loop + CA3
recurrent autoassociator + SWR consolidation pathways). The catalog's
load-bearing positional binding combines theta-gamma temporal phase
(N.16) WITH hippocampal sequence-completion machinery (D.04 + D.11).
The cortical-only v16 substrate has the former (algebraically) but
not the latter; Direction G tests if adding the latter closes the
bound.

Hypothesis: HIPPO substrate's CA3 recurrent + CA1 sequence cells +
time cells provide the per-slot pattern-completion that bare concept
pools cannot. If Direction G strict top-1 multi-seed >= 0.80,
hippocampus IS the missing ingredient (biology-translatable; catalog
vindicated). If Direction G also fails, the bound is deeper.

Reuses _build_bridge_with_hippo from mode_unification_with_hippo_probe.py
byte-unchanged. Reuses Direction E Task 1's encoding/retrieval
mechanism (phase_to_gamma_slot, encode_gamma_slot, capture_phase_
windowed) byte-equivalent.

Pre-registered FROZEN bar: 0.80 multi-seed STRICT TOP-1 (per pillar
n=104; never tuned).

Expected ~60-70 min per seed train (HIPPO bigger) + ~10 min encoding/
readout = ~80 min/seed x 3 seeds = ~4 hr GPU. Kill-safe per-seed
cache.

Reuse-by-import only; no protected/frozen/moat module modified; no
autograd.
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

from research.runners.concept_pool_demo import (
    apply_concept_topographic_bias, train_word_to_pool,
    DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB, ADJECTIVE_VOCAB,
)
from research.findings.raw.mode_unification_with_hippo_probe import (
    _build_bridge_with_hippo,
)
from research.findings.raw.generative_replay_sequence_vocab import (
    generate_k_stored_sequences,
)
from research.findings.raw.direction_A_ec_context_sequence_full import (
    _build_region_filter,
)
from research.findings.raw.direction_E_substrate_task1_full import (
    encode_gamma_slot, capture_phase_windowed, phase_to_gamma_slot,
    _word_score,
    N_LANG_INPUT, N_PER_POOL, N_FS_PER_POOL, N_TRAIN_EVENTS,
    TOPOGRAPHIC_FACTOR, OFF_TARGET_FACTOR, SPARSITY,
    K_PAIRS, SLOT_COUNT, N_REPEATS_PER_TAG,
    THETA_MS, N_GAMMA, N_THETA_CYCLES_ENCODE,
    N_THETA_CYCLES_RECALL, BAR,
)
from sim.backend import get_backend, is_gpu_backend


CACHE_DIR = os.path.join(
    _HERE, "direction_G_hippo_theta_gamma_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
OUT_JSON = os.path.join(
    _HERE, "direction_G_hippo_theta_gamma_substrate.json")
SEEDS = [42, 43, 44]


def _bridge_save_path(seed):
    return os.path.join(CACHE_DIR, f"bridge_full_seed{seed}.simstate.h5")


def _trials_save_path(seed):
    return os.path.join(CACHE_DIR, f"trials_full_seed{seed}.json")


def _build_and_train(seed, verbose=True):
    """Build HIPPO-OPTION3 substrate (v16+hippocampus); train via v16
    recipe; save cache."""
    bridge_p = _bridge_save_path(seed)
    words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
             list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))
    word_to_idx = {w: i for i, w in enumerate(words)}
    n_words = len(words)

    bridge = _build_bridge_with_hippo(
        seed=seed, enable_adjective=True,
        n_lang_input=N_LANG_INPUT, n_per_pool=N_PER_POOL,
        n_fs_per_pool=N_FS_PER_POOL, verbose=verbose,
    )
    if os.path.exists(bridge_p):
        print(f"  [seed {seed}] loading cached trained HIPPO "
              f"bridge", flush=True)
        bridge.load_checkpoint(bridge_p)
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
        return bridge, words, word_to_idx

    print(f"  [seed {seed}] training v16+HIPPO substrate (16 words"
          f" x {N_TRAIN_EVENTS} events)", flush=True)
    t_train = time.time()
    apply_concept_topographic_bias(
        bridge, n_lang_input=N_LANG_INPUT,
        topographic_factor=TOPOGRAPHIC_FACTOR,
        off_target_factor=OFF_TARGET_FACTOR,
        sparsity=SPARSITY, orthogonal_codes=True,
        n_words_for_orthogonal=n_words,
        word_to_idx=word_to_idx, verbose=verbose)

    target_pool = {}
    for w in DIRECTION_VOCAB:
        if w == "north": target_pool[w] = "motor_N"
        elif w == "east": target_pool[w] = "motor_E"
        elif w == "south": target_pool[w] = "motor_S"
        elif w == "west": target_pool[w] = "motor_W"
    for w in NOUN_VOCAB:
        target_pool[w] = f"noun_pool_{w.upper()}"
    for w in VERB_VOCAB:
        target_pool[w] = f"verb_pool_{w.upper()}"
    for w in ADJECTIVE_VOCAB:
        target_pool[w] = f"adjective_pool_{w.upper()}"

    rng = np.random.default_rng(seed)
    schedule = []
    for w in words:
        for _ in range(N_TRAIN_EVENTS):
            schedule.append(w)
    rng.shuffle(schedule)
    for ei, w in enumerate(schedule):
        train_word_to_pool(
            bridge, word=w, target_pool_region=target_pool[w],
            n_events=1, n_lang_input=N_LANG_INPUT,
            n_lang_output=N_LANG_INPUT,
            sparsity=SPARSITY, orthogonal_codes=True,
            n_words_for_orthogonal=n_words,
            word_to_idx=word_to_idx, verbose=False)
        if verbose and (ei + 1) % max(1, len(schedule) // 10) == 0:
            print(f"    [seed {seed}] {ei+1}/{len(schedule)} events"
                  f" ({(time.time()-t_train)/60:.1f} min)",
                  flush=True)
    bridge.save_checkpoint(bridge_p)
    print(f"  [seed {seed}] trained + saved in "
          f"{(time.time()-t_train)/60:.1f} min", flush=True)
    return bridge, words, word_to_idx


def run_one_seed(seed, verbose=True):
    print(f"\n--- seed {seed} ---", flush=True)
    trials_p = _trials_save_path(seed)
    if os.path.exists(trials_p):
        print(f"  [seed {seed}] loading cached trials", flush=True)
        with open(trials_p, "r", encoding="utf-8") as f:
            return json.load(f)

    cp, _ = get_backend()
    bridge, words, word_to_idx = _build_and_train(seed, verbose=verbose)
    rm = bridge.region_manager
    n_lang_output = N_LANG_INPUT
    region_filter = _build_region_filter(rm)
    cfg = bridge.core_config
    theta_steps = max(2, int(round(THETA_MS / cfg.dt_ms)))
    print(f"  [seed {seed}] theta_steps={theta_steps} from dt="
          f"{cfg.dt_ms} ms", flush=True)

    sequences = generate_k_stored_sequences(
        seed=seed, k=K_PAIRS, n_words=len(words),
        slot_count=SLOT_COUNT, vocab=words)
    print(f"  [seed {seed}] {K_PAIRS} sequences x SLOT_COUNT="
          f"{SLOT_COUNT}", flush=True)

    tag_names = []
    for seq_idx, seq in enumerate(sequences):
        tag, stats = encode_gamma_slot(
            bridge, seq, words, seq_idx, region_filter,
            theta_steps, N_GAMMA)
        tag_names.append(tag)
        print(f"  encoded {tag} ({list(seq)}); n_tagged="
              f"{stats.get('n_tagged', 0)}", flush=True)

    per_seq = []
    n_top1 = 0
    for seq_idx, (seq, tag_name) in enumerate(zip(sequences, tag_names)):
        cue_slot_idx = SLOT_COUNT - 1
        true = seq[cue_slot_idx]
        avg_counts = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = capture_phase_windowed(
                bridge, tag_name, cue_slot_idx, theta_steps,
                N_GAMMA, n_lang_output)
            avg_counts = avg_counts + counts.astype(np.float64)
        avg_counts = avg_counts / N_REPEATS_PER_TAG
        scores = {w: _word_score(avg_counts, w, n_lang_output, words)
                  for w in words}
        topK = sorted(scores.items(), key=lambda x: x[1],
                       reverse=True)
        top1_word = topK[0][0]
        top1_correct = (top1_word == true)
        if top1_correct: n_top1 += 1
        per_seq.append({
            "seq_idx": seq_idx, "sequence": list(seq),
            "true_slot": true, "top1_word": top1_word,
            "top1_correct": top1_correct,
            "topK_words": [w for w, _ in topK[:5]],
        })
        print(f"    seq {seq_idx} true={true}; top1={top1_word}; "
              f"correct={top1_correct}", flush=True)

    acc = n_top1 / K_PAIRS
    result = {
        "seed": seed, "K_PAIRS": K_PAIRS, "SLOT_COUNT": SLOT_COUNT,
        "n_top1": n_top1, "strict_top1_accuracy": acc,
        "per_seq": per_seq,
    }
    with open(trials_p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  [seed {seed}] strict top-1 acc = {n_top1}/{K_PAIRS}"
          f" = {acc:.3f}", flush=True)
    return result


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction G: theta-gamma substrate on HIPPO-OPTION3 ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Hypothesis: hippocampus + theta-gamma is the catalog's"
          f" load-bearing positional binding (D.04 + D.11 + N.16);"
          f" the cortical-only substrate (n=104 BOUNDARY) is missing"
          f" the hippocampal pieces.", flush=True)
    print(f"  THETA_MS={THETA_MS}, N_GAMMA={N_GAMMA}, K_PAIRS="
          f"{K_PAIRS}, SLOT_COUNT={SLOT_COUNT}", flush=True)
    print(f"  Pre-registered FROZEN bar: {BAR} multi-seed STRICT "
          f"TOP-1", flush=True)

    seed_results = []
    t0 = time.time()
    for seed in SEEDS:
        r = run_one_seed(seed, verbose=True)
        seed_results.append(r)
    total_min = (time.time() - t0) / 60

    accs = [r["strict_top1_accuracy"] for r in seed_results]
    mean = float(np.mean(accs))
    print(f"\n=== MULTI-SEED RESULT ===", flush=True)
    print(f"  strict top-1 mean = {mean:.3f} per-seed={accs}",
          flush=True)
    print(f"  Wall: {total_min:.1f} min", flush=True)
    print(f"  Comparison: Direction A v1 0.333, v2 0.292, "
          f"Direction E Task 1 0.250 (all BOUNDARY)", flush=True)

    chance = 1.0 / 16.0
    if mean >= BAR:
        verdict = "DIRECTION_G_HIPPO_THETA_GAMMA_PASS"
        print(f"  PASS at multi-seed >= {BAR} -- hippocampus + "
              f"theta-gamma combination CLEARS the bar; catalog's "
              f"load-bearing positional binding mechanism VALIDATED"
              f" on substrate; pillar n=105 candidate.",
              flush=True)
    elif mean > 0.40:
        verdict = "DIRECTION_G_PARTIAL_HIPPO_HELPS"
        print(f"  Partial signal {mean:.3f} -- hippocampus HELPS "
              f"(above prior 0.25-0.33 cluster) but doesn't fully"
              f" solve. Diagnose what's still missing.",
              flush=True)
    elif mean > 2 * chance:
        verdict = "DIRECTION_G_NO_IMPROVEMENT_OVER_CORTICAL"
        print(f"  {mean:.3f} similar to cortical-only result; "
              f"hippocampus addition didn't help; bound is deeper"
              f" than hippocampus.", flush=True)
    else:
        verdict = "DIRECTION_G_HIPPO_NEGATIVE"
        print(f"  At chance; hippocampus actually HURT; substrate"
              f" dynamics broken.", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seeds": SEEDS,
        "K_PAIRS": K_PAIRS, "SLOT_COUNT": SLOT_COUNT,
        "theta_ms": THETA_MS, "n_gamma": N_GAMMA,
        "n_theta_cycles_encode": N_THETA_CYCLES_ENCODE,
        "bar": BAR, "chance": chance,
        "strict_top1_mean": mean, "per_seed_acc": accs,
        "per_seed": seed_results, "verdict": verdict,
        "wall_clock_minutes": total_min,
        "comparison": {
            "direction_A_v1_top1": 0.333,
            "direction_A_v2_top1": 0.292,
            "direction_E_task1_top1": 0.250,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
