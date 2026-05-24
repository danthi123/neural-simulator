"""Direction K substrate FULL NO-TEACHER variant: fair inference-time
test using TRAINED substrate + lang_input ONLY (no teacher current);
verifies whether the FHRR sequence storage mechanism survives
substrate-realistic per-word recognition noise.

Per Direction K substrate FULL (commit b2299a4): PERFECT 1.000 with
TEACHER_PA=500 -- but the teacher forces correct word recognition,
artificially making per-word activity perfect classifier signals.
The FAIR test uses the trained substrate's lang_input -> pool
routing (validated ~0.67-0.88 strict top-1 single-observation per
the recognition-bound probe).

Recipe (matches Direction K substrate FULL except):
- Uses TRAINED substrate (loads from direction_A_ec_context_cache)
- TEACHER_PA = 0 (no teacher current; lang_input drive only)
- Same per-word activity capture; same FHRR algebra
- Multi-seed strict top-1; same 0.80 frozen bar

If PASS: substrate's recognition + FHRR algebra robustness combine
to clear the bar; substrate-grounded FHRR sequence storage VALIDATED
under fair inference conditions.
If BELOW: recognition noise propagates through FHRR; the algebra's
20x noise budget isn't enough; precise characterization of where
substrate recognition fails.

Reuse-by-import only; no protected/frozen/moat module modified; no
autograd.

Reuses Direction A's cached trained bridges (enable_positional_context
=True; ec_context region present but not used in this test). ~10-15
min wall.
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
from research.findings.raw.direction_A_ec_context_sequence_full import (
    _bridge_save_path,  # reuse Direction A's cached trained bridges
    N_LANG_INPUT, N_PER_POOL, N_FS_PER_POOL, N_EC_CONTEXT,
)
from research.findings.raw.direction_K_substrate_smoke import (
    mean_center, fhrr_bind_real_vec, fhrr_unbind_real_vec,
    fhrr_bundle, cosine_real,
)
from sim.text_embeddings import orthogonal_drive_pattern
from sim.backend import get_backend, is_gpu_backend


CACHE_DIR = os.path.join(
    _HERE, "direction_K_substrate_noteacher_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
OUT_JSON = os.path.join(
    _HERE, "direction_K_substrate_full_noteacher.json")
SEEDS = [42, 43, 44]  # Direction A cached
K_PAIRS = 8
SLOT_COUNT = 3
SPARSITY = 0.05
ENCODING_STEPS = 200
TEACHER_PA = 0.0  # no teacher in fair variant
BAR = 0.80


def capture_no_teacher_activity(bridge, word, words, word_to_idx,
                                   n_lang_input, sparsity,
                                   encoding_steps):
    """Drive lang_input(word) ONLY (no teacher); capture spike
    counts across all concept-pool regions. Inference-time scenario."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_in_idx = list(rm.indices("language_input"))
    lang_in_arr = cp.asarray(lang_in_idx, dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    drive = orthogonal_drive_pattern(
        cue_idx=word_to_idx[word], n_cues=len(words),
        n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=sparsity)

    pool_kinds = [("noun_pool", ["APPLE", "RIVER", "DOG", "CAT"]),
                   ("verb_pool", ["GO", "COME", "STOP", "LOOK"]),
                   ("adjective_pool", ["BIG", "SMALL", "HOT", "COLD"])]
    all_pool_idx = []
    for kind, names in pool_kinds:
        for n in names:
            try:
                all_pool_idx.extend(list(rm.indices(f"{kind}_{n}")))
            except Exception:
                pass
    for m in ["motor_N", "motor_E", "motor_S", "motor_W"]:
        try:
            all_pool_idx.extend(list(rm.indices(m)))
        except Exception:
            pass
    all_pool_arr = cp.asarray(all_pool_idx, dtype=cp.int64)
    n_pool_total = len(all_pool_idx)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    spike_counts = cp.zeros(n_pool_total, dtype=cp.float32)
    ext = cp.zeros(n_total, dtype=cp.float32)
    for _ in range(encoding_steps):
        ext.fill(0)
        ext[lang_in_arr] = cp.asarray(drive, dtype=cp.float32)
        # NO TEACHER (this is the fair test)
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[all_pool_arr]
        spike_counts = spike_counts + fired.astype(cp.float32)
    bridge.cp_external_input_current[:] = 0.0
    return cp.asnumpy(spike_counts), n_pool_total


def run_one_seed(seed, verbose=True):
    print(f"\n--- seed {seed} (no-teacher) ---", flush=True)
    cache_p = os.path.join(CACHE_DIR, f"seed{seed}.json")
    if os.path.exists(cache_p):
        print(f"  loading cached", flush=True)
        with open(cache_p, "r", encoding="utf-8") as f:
            return json.load(f)

    bridge_p = _bridge_save_path(seed)
    if not os.path.exists(bridge_p):
        print(f"  [FATAL] cached trained bridge missing: {bridge_p}",
              flush=True)
        return None

    t0 = time.time()
    cp, _ = get_backend()
    words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
             list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))
    n_words = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}

    # Build same architecture as Direction A trained bridges
    bridge = build_concept_bridge(
        seed=seed, n_lang_input=N_LANG_INPUT, n_per_pool=N_PER_POOL,
        n_fs_per_pool=N_FS_PER_POOL, enable_adjective=True,
        weak_dynamics=True, enable_positional_context=True,
        n_ec_context=N_EC_CONTEXT, verbose=False,
    )
    bridge.load_checkpoint(bridge_p)
    print(f"  loaded TRAINED bridge in {(time.time()-t0):.1f}s",
          flush=True)
    # Freeze plasticity
    for g in ("language_input_to_motor",
              "language_input_to_noun_pool",
              "language_input_to_verb_pool",
              "language_input_to_adjective_pool",
              "motor_to_language_output",
              "noun_pool_to_language_output",
              "verb_pool_to_language_output",
              "adjective_pool_to_language_output",
              "ec_context_to_noun_pool",
              "ec_context_to_verb_pool",
              "ec_context_to_adjective_pool",
              "ec_context_to_motor"):
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass

    # Capture vocab activities WITHOUT teacher
    print(f"  capturing per-word activities (NO TEACHER; "
          f"lang_input only)", flush=True)
    t_cap = time.time()
    vocab_activities = {}
    n_pool_total = None
    for w in words:
        spike_counts, n_pool_total = capture_no_teacher_activity(
            bridge, w, words, word_to_idx, N_LANG_INPUT, SPARSITY,
            ENCODING_STEPS)
        vocab_activities[w] = mean_center(spike_counts)
    print(f"  captured {len(vocab_activities)} (dim={n_pool_total}"
          f") in {(time.time()-t_cap)/60:.1f} min", flush=True)

    # Diagnostic: per-word activity norm + pairwise similarity
    norms = np.array([np.linalg.norm(vocab_activities[w])
                      for w in words])
    print(f"  per-word activity norm: mean={norms.mean():.2f}, "
          f"min={norms.min():.2f}, max={norms.max():.2f}",
          flush=True)
    # Pairwise overlap
    overlaps = []
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            if i < j:
                overlaps.append(cosine_real(vocab_activities[w1],
                                              vocab_activities[w2]))
    overlaps = np.array(overlaps)
    print(f"  pairwise cosine overlap: mean={overlaps.mean():.3f}, "
          f"max={overlaps.max():.3f}", flush=True)

    # Position phasors
    rng = np.random.default_rng(seed * 9999 + 7)
    position_phasors = [
        rng.choice([-1.0, 1.0], size=n_pool_total)
        for _ in range(SLOT_COUNT)
    ]

    sequences = generate_k_stored_sequences(
        seed=seed, k=K_PAIRS, n_words=n_words,
        slot_count=SLOT_COUNT, vocab=words)

    n_top1 = 0
    per_seq = []
    for seq_idx, seq in enumerate(sequences):
        bound = []
        for slot_idx, c_word in enumerate(seq):
            bound.append(fhrr_bind_real_vec(
                vocab_activities[c_word], position_phasors[slot_idx]))
        bundle = fhrr_bundle(*bound)
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
    print(f"  [seed {seed}] strict top-1 (no teacher) = {n_top1}"
          f"/{K_PAIRS} = {acc:.3f} (wall {(time.time()-t0)/60:.1f}"
          f" min)", flush=True)
    result = {
        "seed": seed, "n_top1": n_top1, "K_PAIRS": K_PAIRS,
        "strict_top1_accuracy": acc, "per_seq": per_seq,
        "vocab_norm_mean": float(norms.mean()),
        "vocab_overlap_mean": float(overlaps.mean()),
        "vocab_overlap_max": float(overlaps.max()),
    }
    with open(cache_p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction K substrate FULL NO-TEACHER ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Fair test: trained substrate + lang_input only "
          f"(no teacher); inference-time scenario", flush=True)
    print(f"  Pre-registered FROZEN bar: {BAR} multi-seed STRICT "
          f"TOP-1", flush=True)

    t0 = time.time()
    seed_results = []
    for seed in SEEDS:
        r = run_one_seed(seed, verbose=True)
        if r is not None: seed_results.append(r)
    total_min = (time.time() - t0) / 60

    if not seed_results:
        print("[FATAL] no cached trained bridges", flush=True)
        return 1

    accs = [r["strict_top1_accuracy"] for r in seed_results]
    mean = float(np.mean(accs))
    print(f"\n=== MULTI-SEED RESULT (NO TEACHER) ===", flush=True)
    print(f"  strict top-1 mean = {mean:.3f} per-seed={accs}",
          flush=True)
    print(f"  Wall: {total_min:.1f} min", flush=True)
    print(f"\n  Comparison:", flush=True)
    print(f"    Direction K with TEACHER (artificial):  1.000",
          flush=True)
    print(f"    Direction K NO TEACHER (fair):          "
          f"{mean:.3f}", flush=True)
    print(f"    4 engram-tag attempts cluster:          0.25-0.33",
          flush=True)

    chance = 1.0 / 16.0
    if mean >= BAR:
        verdict = "DIRECTION_K_FAIR_PASS"
        print(f"\n  FAIR PASS at multi-seed >= {BAR} -- substrate-"
              f"grounded FHRR sequence storage VALIDATED under "
              f"inference-time conditions; pillar n=105 candidate.",
              flush=True)
    elif mean > 0.5:
        verdict = "DIRECTION_K_FAIR_PARTIAL"
        print(f"\n  Partial signal {mean:.3f} > 0.5; FHRR + "
              f"substrate-grounded mechanism HELPS significantly "
              f"over the 0.25-0.33 engram-tag cluster.",
              flush=True)
    elif mean > 2 * chance:
        verdict = "DIRECTION_K_FAIR_NO_BETTER_THAN_ENGRAM"
        print(f"\n  {mean:.3f} no better than engram-tag attempts; "
              f"recognition bound dominates regardless of "
              f"mechanism choice.", flush=True)
    else:
        verdict = "DIRECTION_K_FAIR_NEGATIVE"
        print(f"\n  At chance under fair test; substrate recognition"
              f" is the binding constraint.", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seeds": SEEDS,
        "K_PAIRS": K_PAIRS, "SLOT_COUNT": SLOT_COUNT, "bar": BAR,
        "strict_top1_mean": mean, "per_seed_acc": accs,
        "per_seed": seed_results, "verdict": verdict,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
