"""Direction E substrate Task 1: FULL theta-gamma substrate sequence
storage runner (analog of Direction A v1, but using temporal phase
code instead of spatial ec_context).

Per docs/plans/2026-05-24-direction-E-theta-gamma-substrate-design.md
Task 1: implement the gamma-slot-gated encoding + phase-cued retrieval
mechanism on the v16 substrate (no enable_positional_context). The
theta clock is the step-index-modulo-theta_steps pattern (substitution
1a; no new theta_pacemaker region).

DESIGN:
1. Build v16 substrate (no ec_context); train via standard recipe.
2. For each sequence: encode K slot-words via gamma-slot-gated
   lang_input drive + teacher current ONLY during slot-i gamma window
   of each theta cycle; repeat across N_THETA_CYCLES_ENCODE cycles.
   Engram captures co-firing across full encoding window.
3. For retrieval: stim engram tag during a full retrieval theta
   cycle; READ lang_output ONLY during slot-i gamma window for
   slot-i word; cosine-match per word at that window.
4. Multi-seed (42, 43, 44); kill-safe per-seed cache.
5. Pre-registered 0.80 multi-seed STRICT TOP-1 bar (frozen; per the
   strict metric introduced by Direction A's reviewer).

Reuses every validated primitive byte-unchanged; no protected/frozen/
moat module modified; no autograd.

Expected ~50 min train per seed + ~10 min encoding/readout = ~60 min
per seed x 3 seeds = ~3 hr GPU. Kill-safe per-seed cache.

QUEUED for after Direction A v1+v2 results; only run if both fail.
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
    build_concept_bridge, apply_concept_topographic_bias,
    train_word_to_pool, DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB,
    ADJECTIVE_VOCAB,
)
from research.runners.concept_compose_train import _WORD_TO_POOL
from research.findings.raw.generative_replay_sequence_vocab import (
    generate_k_stored_sequences,
)
from research.findings.raw.direction_A_ec_context_sequence_full import (
    _build_region_filter,
)
from sim.text_embeddings import orthogonal_drive_pattern
from sim.backend import get_backend, is_gpu_backend


CACHE_DIR = os.path.join(
    _HERE, "direction_E_substrate_task1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Pre-registered constants (mirror Direction A where applicable).
N_LANG_INPUT = 2048
N_PER_POOL = 200
N_FS_PER_POOL = 24
N_TRAIN_EVENTS = 200
TOPOGRAPHIC_FACTOR = 3.0
OFF_TARGET_FACTOR = 0.3
SPARSITY = 0.05

K_PAIRS = 8
SLOT_COUNT = 3
TOP_K_READOUT = 1  # STRICT TOP-1 per reviewer
N_REPEATS_PER_TAG = 3
STIM_DRIVE_PA = 1500.0
TEACHER_PA = 500.0
ENGRAM_TOP_K = 100

# Theta-gamma timing (Lisman-Idiart): 8Hz theta, 7 gamma slots
THETA_MS = 125.0
N_GAMMA = 7  # catalog cap
N_THETA_CYCLES_ENCODE = 3  # repeat sequence to consolidate engram
N_THETA_CYCLES_RECALL = 1  # single retrieval theta cycle

BAR = 0.80
SEEDS = [42, 43, 44]
OUT_JSON = os.path.join(
    _HERE, "direction_E_substrate_task1_full.json")


def _bridge_save_path(seed):
    return os.path.join(CACHE_DIR, f"bridge_full_seed{seed}.simstate.h5")


def _trials_save_path(seed):
    return os.path.join(CACHE_DIR, f"trials_full_seed{seed}.json")


def phase_to_gamma_slot(step_idx, theta_steps, n_gamma):
    """Mirror Task 0 + pirazzini step-index pattern."""
    phase = int(step_idx) % int(theta_steps)
    return min(n_gamma - 1, (phase * n_gamma) // theta_steps)


def _build_and_train(seed, verbose=True):
    """Build v16 substrate (no ec_context; theta-gamma uses temporal
    phase only); train via v16 recipe; save cache."""
    bridge_p = _bridge_save_path(seed)
    words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
             list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))
    word_to_idx = {w: i for i, w in enumerate(words)}
    n_words = len(words)

    bridge = build_concept_bridge(
        seed=seed, n_lang_input=N_LANG_INPUT, n_per_pool=N_PER_POOL,
        n_fs_per_pool=N_FS_PER_POOL, enable_adjective=True,
        weak_dynamics=True, enable_positional_context=False,
        verbose=verbose,
    )
    if os.path.exists(bridge_p):
        print(f"  [seed {seed}] loading cached trained bridge",
              flush=True)
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

    print(f"  [seed {seed}] training v16 substrate", flush=True)
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


def encode_gamma_slot(bridge, seq, words, seq_idx, region_filter,
                        theta_steps, n_gamma):
    """Encode with gamma-slot-gated drives across N_THETA_CYCLES_ENCODE
    theta cycles. lang_input(word_i) + teacher(pool_i) only during
    slot-i gamma window of each cycle."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_in_idx = list(rm.indices("language_input"))
    lang_in_arr = cp.asarray(lang_in_idx, dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]
    word_to_idx = {w: i for i, w in enumerate(words)}
    tag_name = f"task1_seq_{seq_idx:03d}"

    pool_arrs = []
    for slot_word in seq:
        pool_region = _WORD_TO_POOL.get(slot_word, None)
        if pool_region is None:
            pool_arrs.append(None); continue
        try:
            pool_idx = list(rm.indices(pool_region))
            pool_arrs.append(cp.asarray(pool_idx, dtype=cp.int64))
        except Exception:
            pool_arrs.append(None)

    drives_per_word = [orthogonal_drive_pattern(
        cue_idx=word_to_idx[w], n_cues=len(words),
        n_neurons=N_LANG_INPUT, drive_max_pA=200.0,
        sparsity=SPARSITY) for w in seq]

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    bridge.start_engram_recording(tag_name)
    ext = cp.zeros(n_total, dtype=cp.float32)
    base_step = int(bridge.runtime_state.current_time_step)
    encoding_steps = N_THETA_CYCLES_ENCODE * theta_steps
    for step in range(encoding_steps):
        # Use the gamma slot relative to the start of encoding
        slot = phase_to_gamma_slot(step, theta_steps, n_gamma)
        if slot < SLOT_COUNT:  # active slot
            ext.fill(0)
            ext[lang_in_arr] = cp.asarray(
                drives_per_word[slot], dtype=cp.float32)
            if pool_arrs[slot] is not None:
                ext[pool_arrs[slot]] = TEACHER_PA
            bridge.cp_external_input_current[:] = ext
        else:
            ext.fill(0)
            bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    stats = bridge.commit_engram_tag(
        tag_name, top_k=ENGRAM_TOP_K, region_filter=region_filter)
    return tag_name, stats


def capture_phase_windowed(bridge, tag_name, cue_slot_idx,
                              theta_steps, n_gamma, n_lang_output):
    """Stim engram tag for one full retrieval theta cycle; READ
    lang_output ONLY during the cue_slot_idx gamma window."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_out_idx = list(rm.indices("language_output"))
    lang_out_arr = cp.asarray(lang_out_idx, dtype=cp.int64)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    bridge.stimulate_tag(tag_name, drive_pA=STIM_DRIVE_PA, additive=False)

    # Read full theta cycle but accumulate only during cue slot window
    lang_counts = cp.zeros(n_lang_output, dtype=cp.float32)
    recall_steps = N_THETA_CYCLES_RECALL * theta_steps
    for step in range(recall_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        current_slot = phase_to_gamma_slot(step, theta_steps, n_gamma)
        if current_slot == cue_slot_idx:
            fired = bridge.cp_firing_states[lang_out_arr]
            lang_counts = lang_counts + fired.astype(cp.float32)

    bridge.clear_tag_drive(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    return cp.asnumpy(lang_counts)


def _word_score(lang_counts, word, n_lang_output, words):
    word_to_idx = {w: i for i, w in enumerate(words)}
    pattern = orthogonal_drive_pattern(
        cue_idx=word_to_idx[word], n_cues=len(words),
        n_neurons=n_lang_output, drive_max_pA=1.0, sparsity=SPARSITY)
    a = lang_counts.astype(np.float64); b = pattern.astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return 0.0
    return float(np.dot(a, b) / (na * nb))


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

    cfg = bridge.simulation_config
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
    print(f"=== Direction E substrate Task 1 FULL ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Theta-gamma temporal phase code for sequence storage",
          flush=True)
    print(f"  THETA_MS={THETA_MS}, N_GAMMA={N_GAMMA}, "
          f"N_THETA_CYCLES_ENCODE={N_THETA_CYCLES_ENCODE}",
          flush=True)
    print(f"  K_PAIRS={K_PAIRS}, SLOT_COUNT={SLOT_COUNT}",
          flush=True)
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
    print(f"  strict top-1 mean = {mean:.3f} per-seed="
          f"{accs}", flush=True)
    print(f"  Wall: {total_min:.1f} min", flush=True)

    chance = 1.0 / 16.0
    if mean >= BAR:
        verdict = "DIRECTION_E_SUBSTRATE_TASK1_PASS"
        print(f"  PASS at multi-seed >= {BAR} -- theta-gamma "
              f"substrate sequence storage works; pillar candidate.",
              flush=True)
    elif mean > 2 * chance:
        verdict = "DIRECTION_E_SUBSTRATE_TASK1_BOUNDARY"
        print(f"  partial signal {mean:.3f} > 2*chance "
              f"{2*chance:.3f} but < {BAR}.", flush=True)
    else:
        verdict = "DIRECTION_E_SUBSTRATE_TASK1_NEGATIVE"
        print(f"  at chance ({chance:.3f}); theta-gamma substrate"
              f" mechanism didn't transfer from algebra.",
              flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seeds": SEEDS,
        "K_PAIRS": K_PAIRS, "SLOT_COUNT": SLOT_COUNT,
        "theta_ms": THETA_MS, "n_gamma": N_GAMMA,
        "n_theta_cycles_encode": N_THETA_CYCLES_ENCODE,
        "bar": BAR, "chance": chance,
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
