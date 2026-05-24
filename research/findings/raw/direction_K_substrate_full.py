"""Direction K substrate FULL: multi-seed FHRR-based sequence storage
on v16 substrate. Same frozen 0.80 multi-seed STRICT TOP-1 bar.

Per Direction K smoke (commit 0bdcdaa): SMOKE_PERFECT 3/3 slots
retrieved at single-seed. Full-scale verifies whether the mechanism
holds at multi-seed across K sequences.

Recipe:
1. Build v16 substrate (no training; teacher-driven activity is
   the FHRR shortcut-2 RESOLVED grounding)
2. Per seed: capture per-word mean-centered activity vectors over
   concept-pool regions (substrate-grounded vocab phasors)
3. Per seed: deterministic random per-position phasors (sign vectors)
4. For K=8 sequences x SLOT_COUNT=3:
   - Encode: bundle of (concept_phasor_i bind position_phasor_i)
   - Retrieve slot 2: unbind sequence phasor with position_2;
     cosine-match against vocab phasors; argmax top-1
5. Multi-seed 42/43/44/45/46; strict top-1 mean

Pre-registered FROZEN bar: 0.80 multi-seed STRICT TOP-1.

Expected wall: ~5-10 min per seed (just activity capture; no
training, no engram); ~30-50 min total multi-seed.

Reuse-by-import only; no protected/frozen/moat module modified; no
autograd.
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

from research.runners.concept_pool_demo import (
    build_concept_bridge, DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB,
    ADJECTIVE_VOCAB,
)
from research.runners.concept_compose_train import _WORD_TO_POOL
from research.findings.raw.generative_replay_sequence_vocab import (
    generate_k_stored_sequences,
)
from research.findings.raw.direction_K_substrate_smoke import (
    capture_substrate_activity, mean_center,
    fhrr_bind_real_vec, fhrr_unbind_real_vec, fhrr_bundle,
    cosine_real,
    N_LANG_INPUT, N_PER_POOL, N_FS_PER_POOL, SPARSITY,
    ENCODING_STEPS, SLOT_COUNT,
)
from sim.backend import get_backend, is_gpu_backend


CACHE_DIR = os.path.join(_HERE, "direction_K_substrate_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
OUT_JSON = os.path.join(_HERE, "direction_K_substrate_full.json")
SEEDS = [42, 43, 44, 45, 46]
K_PAIRS = 8
BAR = 0.80


def run_one_seed(seed, verbose=True):
    print(f"\n--- seed {seed} ---", flush=True)
    cache_p = os.path.join(CACHE_DIR, f"seed{seed}.json")
    if os.path.exists(cache_p):
        print(f"  [seed {seed}] loading cached trials", flush=True)
        with open(cache_p, "r", encoding="utf-8") as f:
            return json.load(f)

    t0 = time.time()
    cp, _ = get_backend()
    words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
             list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))
    n_words = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}

    bridge = build_concept_bridge(
        seed=seed, n_lang_input=N_LANG_INPUT, n_per_pool=N_PER_POOL,
        n_fs_per_pool=N_FS_PER_POOL, enable_adjective=True,
        weak_dynamics=True, enable_positional_context=False,
        verbose=False,
    )
    # Freeze plasticity
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

    # Capture per-word activity vectors (substrate grounding)
    print(f"  [seed {seed}] capturing per-word activity vectors",
          flush=True)
    t_cap = time.time()
    vocab_activities = {}
    n_pool_total = None
    for w in words:
        spike_counts, n_pool_total = capture_substrate_activity(
            bridge, w, words, word_to_idx, N_LANG_INPUT, SPARSITY,
            ENCODING_STEPS, _WORD_TO_POOL[w])
        vocab_activities[w] = mean_center(spike_counts)
    print(f"  [seed {seed}] captured {len(vocab_activities)} vocab "
          f"activities (dim={n_pool_total}) in "
          f"{(time.time()-t_cap)/60:.1f} min", flush=True)

    # Position phasors (deterministic per-seed)
    rng = np.random.default_rng(seed * 9999 + 7)
    position_phasors = [
        rng.choice([-1.0, 1.0], size=n_pool_total)
        for _ in range(SLOT_COUNT)
    ]

    # Generate K sequences
    sequences = generate_k_stored_sequences(
        seed=seed, k=K_PAIRS, n_words=n_words,
        slot_count=SLOT_COUNT, vocab=words)

    # Per sequence: encode + retrieve slot 2 (last); strict top-1
    n_top1 = 0
    per_seq = []
    for seq_idx, seq in enumerate(sequences):
        # Encode
        bound = []
        for slot_idx, c_word in enumerate(seq):
            bound.append(fhrr_bind_real_vec(
                vocab_activities[c_word], position_phasors[slot_idx]))
        bundle = fhrr_bundle(*bound)
        # Retrieve slot 2 (last)
        query_slot = SLOT_COUNT - 1
        unbound = fhrr_unbind_real_vec(
            bundle, position_phasors[query_slot])
        scores = {w: cosine_real(unbound, vocab_activities[w])
                  for w in words}
        topK = sorted(scores.items(), key=lambda x: x[1],
                       reverse=True)
        top1 = topK[0][0]
        true = seq[query_slot]
        correct = (top1 == true)
        if correct: n_top1 += 1
        per_seq.append({
            "seq_idx": seq_idx, "sequence": list(seq),
            "true_slot": true, "top1": top1,
            "top1_correct": correct,
            "topK_words": [w for w, _ in topK[:5]],
        })
        if verbose:
            print(f"    seq {seq_idx} {list(seq)} true={true} "
                  f"top1={top1} correct={correct}", flush=True)

    acc = n_top1 / K_PAIRS
    print(f"  [seed {seed}] strict top-1 = {n_top1}/{K_PAIRS} = "
          f"{acc:.3f} (wall {(time.time()-t0)/60:.1f} min)",
          flush=True)
    result = {
        "seed": seed, "n_top1": n_top1, "K_PAIRS": K_PAIRS,
        "strict_top1_accuracy": acc, "per_seq": per_seq,
    }
    with open(cache_p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction K substrate FULL ===", flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Multi-seed FHRR sequence storage on v16 substrate"
          f" (reuses validated shortcut-2 RESOLVED grounding +"
          f" Direction E algebra position phasors).", flush=True)
    print(f"  seeds: {SEEDS}; K={K_PAIRS}; SLOT_COUNT={SLOT_COUNT}",
          flush=True)
    print(f"  Pre-registered FROZEN bar: {BAR} multi-seed STRICT "
          f"TOP-1.", flush=True)

    t0 = time.time()
    seed_results = []
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
    print(f"\n  Comparison to engram-tag attempts:",
          flush=True)
    print(f"    Direction A v1 (cortical+ec_context, frozen):  0.333",
          flush=True)
    print(f"    Direction A v2 (cortical+ec_context, learned): 0.292",
          flush=True)
    print(f"    Direction E Task 1 (cortical+theta-gamma):     0.250",
          flush=True)
    print(f"    Direction G (HIPPO+theta-gamma):               0.333",
          flush=True)
    print(f"    Direction K (FHRR + substrate grounding):      "
          f"{mean:.3f}", flush=True)

    chance = 1.0 / 16.0
    if mean >= BAR:
        verdict = "DIRECTION_K_SUBSTRATE_PASS"
        print(f"\n  PASS at multi-seed >= {BAR} -- substrate-grounded"
              f" FHRR sequence storage VALIDATED; pillar n=105 "
              f"candidate.", flush=True)
    elif mean > 0.5:
        verdict = "DIRECTION_K_SUBSTRATE_PARTIAL"
        print(f"\n  partial signal {mean:.3f} > 0.5 -- mechanism "
              f"helps significantly over the 0.25-0.33 cluster; "
              f"diagnose what's preventing full PASS.", flush=True)
    elif mean > 2 * chance:
        verdict = "DIRECTION_K_SUBSTRATE_ABOVE_CHANCE_BELOW_BAR"
        print(f"\n  {mean:.3f} above chance but below bar; FHRR "
              f"mechanism produces real signal but not above the "
              f"engram-tag cluster.", flush=True)
    else:
        verdict = "DIRECTION_K_SUBSTRATE_NEGATIVE"
        print(f"\n  at chance; substrate-grounded FHRR didn't "
              f"transfer.", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seeds": SEEDS,
        "K_PAIRS": K_PAIRS, "SLOT_COUNT": SLOT_COUNT, "bar": BAR,
        "strict_top1_mean": mean, "per_seed_acc": accs,
        "per_seed": seed_results, "verdict": verdict,
        "wall_clock_minutes": total_min,
        "comparison": {
            "direction_A_v1": 0.333,
            "direction_A_v2": 0.292,
            "direction_E_task1": 0.250,
            "direction_G": 0.333,
            "direction_K": mean,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
