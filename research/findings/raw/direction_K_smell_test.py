"""Direction K SMELL TEST: 3 anti-cheat controls scrutinize whether
the PASS is genuine substrate-grounded sequence storage or hidden
artifact.

Per discipline (scrutinize PASS harder than FAIL): the Direction K
substrate FULL NO-TEACHER variant cleared the 0.80 bar at 1.000
multi-seed (3 seeds [42,43,44] x 8 sequences). Possible artifacts:
(A) Permutation control: permute slot-to-concept mapping. If retrieval
    still passes with scrambled ground truth, decoder isn't using
    slot-position info; the result is multitag-style set-membership.
(B) Random-phasor-vocab control: replace substrate-grounded concept
    phasors with random vectors. If retrieval still passes, the
    substrate grounding wasn't load-bearing (it'd just be FHRR
    algebra with random codes).
(C) No-position-distinction control: use SAME position phasor for
    all slots. If retrieval still passes, the position phasors
    aren't load-bearing.

For a TRUE PASS: (A) and (C) drop to ~1/SLOT_COUNT chance = 0.333
(random alignment under perm/no-distinction); (B) FAILS because
sustain-phasor norms differ from substrate-norm-scale.

Reuses Direction K substrate NO-TEACHER cached vocab activities +
position phasors from the cache JSONs. NUMPY-only post-processing
mostly; some FHRR re-eval. ~2-5 min wall.
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
    _bridge_save_path, N_LANG_INPUT, N_PER_POOL, N_FS_PER_POOL,
    N_EC_CONTEXT,
)
from research.findings.raw.direction_K_substrate_smoke import (
    mean_center, fhrr_bind_real_vec, fhrr_unbind_real_vec,
    fhrr_bundle, cosine_real,
)
from research.findings.raw.direction_K_substrate_full_noteacher import (
    capture_no_teacher_activity, SPARSITY, ENCODING_STEPS,
    SLOT_COUNT, K_PAIRS, BAR,
)
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(_HERE, "direction_K_smell_test.json")
SEEDS = [42, 43, 44]


def run_controls_for_seed(seed, verbose=True):
    print(f"\n--- seed {seed} controls ---", flush=True)
    bridge_p = _bridge_save_path(seed)
    if not os.path.exists(bridge_p):
        print(f"  bridge missing", flush=True)
        return None

    cp, _ = get_backend()
    words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
             list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))
    n_words = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}

    bridge = build_concept_bridge(
        seed=seed, n_lang_input=N_LANG_INPUT, n_per_pool=N_PER_POOL,
        n_fs_per_pool=N_FS_PER_POOL, enable_adjective=True,
        weak_dynamics=True, enable_positional_context=True,
        n_ec_context=N_EC_CONTEXT, verbose=False,
    )
    bridge.load_checkpoint(bridge_p)
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

    # Re-capture per-word activities (no teacher)
    print(f"  re-capturing vocab activities", flush=True)
    vocab_activities = {}
    n_pool_total = None
    for w in words:
        spike_counts, n_pool_total = capture_no_teacher_activity(
            bridge, w, words, word_to_idx, N_LANG_INPUT, SPARSITY,
            ENCODING_STEPS)
        vocab_activities[w] = mean_center(spike_counts)

    rng = np.random.default_rng(seed * 9999 + 7)
    position_phasors = [
        rng.choice([-1.0, 1.0], size=n_pool_total)
        for _ in range(SLOT_COUNT)
    ]

    sequences = generate_k_stored_sequences(
        seed=seed, k=K_PAIRS, n_words=n_words,
        slot_count=SLOT_COUNT, vocab=words)

    # Control A: permutation of slot-to-concept ground truth
    n_top1_perm = 0
    rng_perm = np.random.default_rng(seed * 31337)
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
        # Permuted ground truth: score against a permuted slot order
        perm = rng_perm.permutation(SLOT_COUNT)
        perm_true = seq[perm[query_slot]]
        if top1 == perm_true:
            n_top1_perm += 1
    acc_perm = n_top1_perm / K_PAIRS

    # Control B: random vocab phasors (replace substrate grounding)
    rng_rand = np.random.default_rng(seed * 12345)
    random_phasors = {w: rng_rand.standard_normal(n_pool_total)
                      for w in words}
    n_top1_rand = 0
    for seq_idx, seq in enumerate(sequences):
        bound = []
        for slot_idx, c_word in enumerate(seq):
            bound.append(fhrr_bind_real_vec(
                random_phasors[c_word], position_phasors[slot_idx]))
        bundle = fhrr_bundle(*bound)
        query_slot = SLOT_COUNT - 1
        unbound = fhrr_unbind_real_vec(
            bundle, position_phasors[query_slot])
        scores = {w: cosine_real(unbound, random_phasors[w])
                  for w in words}
        topK = sorted(scores.items(), key=lambda x: x[1],
                       reverse=True)
        top1 = topK[0][0]
        true = seq[query_slot]
        if top1 == true: n_top1_rand += 1
    acc_rand = n_top1_rand / K_PAIRS

    # Control C: SAME position phasor for all slots
    same_phasor = rng.choice([-1.0, 1.0], size=n_pool_total)
    n_top1_same = 0
    for seq_idx, seq in enumerate(sequences):
        bound = []
        for slot_idx, c_word in enumerate(seq):
            bound.append(fhrr_bind_real_vec(
                vocab_activities[c_word], same_phasor))
        bundle = fhrr_bundle(*bound)
        # Unbind with the same phasor (doesn't distinguish slots)
        unbound = fhrr_unbind_real_vec(bundle, same_phasor)
        scores = {w: cosine_real(unbound, vocab_activities[w])
                  for w in words}
        topK = sorted(scores.items(), key=lambda x: x[1],
                       reverse=True)
        top1 = topK[0][0]
        true = seq[SLOT_COUNT - 1]
        if top1 == true: n_top1_same += 1
    acc_same = n_top1_same / K_PAIRS

    print(f"  (A) perm:   {acc_perm:.3f}", flush=True)
    print(f"  (B) random: {acc_rand:.3f}", flush=True)
    print(f"  (C) same:   {acc_same:.3f}", flush=True)
    return {
        "seed": seed,
        "ctrl_permutation_top1": acc_perm,
        "ctrl_random_phasor_top1": acc_rand,
        "ctrl_same_position_top1": acc_same,
    }


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction K SMELL TEST (no-teacher PASS scrutiny) ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Pre-registered bar: {BAR}", flush=True)

    main_p = os.path.join(
        _HERE, "direction_K_substrate_full_noteacher.json")
    if os.path.exists(main_p):
        with open(main_p, "r", encoding="utf-8") as f:
            main_result = json.load(f)
        main_mean = main_result.get("strict_top1_mean", float("nan"))
        print(f"  Main fair strict top-1 multi-seed: {main_mean:.3f}",
              flush=True)
    else:
        main_mean = float("nan")
        main_result = None

    t0 = time.time()
    seed_ctrls = []
    for seed in SEEDS:
        ctrl = run_controls_for_seed(seed)
        if ctrl is not None:
            seed_ctrls.append(ctrl)
    total_min = (time.time() - t0) / 60

    if not seed_ctrls:
        print("[FATAL]", flush=True); return 1

    perm_mean = float(np.mean(
        [c["ctrl_permutation_top1"] for c in seed_ctrls]))
    rand_mean = float(np.mean(
        [c["ctrl_random_phasor_top1"] for c in seed_ctrls]))
    same_mean = float(np.mean(
        [c["ctrl_same_position_top1"] for c in seed_ctrls]))

    print(f"\n=== CONTROL RESULTS (multi-seed mean) ===",
          flush=True)
    print(f"  (A) permutation:      {perm_mean:.3f}", flush=True)
    print(f"  (B) random phasors:   {rand_mean:.3f}", flush=True)
    print(f"  (C) same position:    {same_mean:.3f}", flush=True)
    print(f"  Wall: {total_min:.1f} min", flush=True)

    chance_top1 = 1.0 / 16.0
    chance_slot = 1.0 / SLOT_COUNT
    # For permutation: expected = P(perm == true) = 1/SLOT_COUNT if
    # decoder picks slot-i correctly always
    # For same-position: expected = ~1/SLOT_COUNT if decoder gives
    # equal weight to all in-sequence words at top-1
    print(f"\n  Expected chance baselines:", flush=True)
    print(f"    (A) perm:   {chance_slot:.3f} (1/SLOT_COUNT)",
          flush=True)
    print(f"    (B) random: {chance_top1:.3f} (1/N_VOCAB)",
          flush=True)
    print(f"    (C) same:   {chance_slot:.3f} (1/SLOT_COUNT; "
          f"argmax over sequence members)", flush=True)

    if main_result is not None:
        margin_perm = main_mean - perm_mean
        margin_rand = main_mean - rand_mean
        margin_same = main_mean - same_mean
        print(f"\n  Margins (main - ctrl):", flush=True)
        print(f"    perm:   {margin_perm:+.3f}", flush=True)
        print(f"    random: {margin_rand:+.3f}", flush=True)
        print(f"    same:   {margin_same:+.3f}", flush=True)

        if main_mean >= BAR:
            # Random control: if it ALSO PASSes at ~0.8, substrate
            # grounding isn't load-bearing.
            # Note: for FHRR with random N_DIM=3200, random phasors
            # might also pass because dim is large enough.
            random_ok = rand_mean < 0.5  # if pure random ~chance
            # Permutation: should drop to ~1/SLOT_COUNT
            perm_at_chance = abs(perm_mean - chance_slot) < 0.15
            same_at_chance = abs(same_mean - chance_slot) < 0.15

            if perm_at_chance and same_at_chance:
                if random_ok:
                    verdict = "PASS_CONTROLS_DECISIVE_SUBSTRATE_LOAD_BEARING"
                    print(f"\n  ALL CONTROLS DECISIVE: perm and same"
                          f" -> ~1/SLOT_COUNT chance; random "
                          f"phasors fail -> substrate grounding "
                          f"load-bearing. PILLAR N=105 CANDIDATE.",
                          flush=True)
                else:
                    verdict = "PASS_CONTROLS_OK_BUT_RANDOM_ALSO_PASSES"
                    print(f"\n  PERM + SAME at chance OK but random"
                          f" phasors ALSO pass; substrate grounding"
                          f" not load-bearing (the FHRR algebra "
                          f"works with random codes at this dim).",
                          flush=True)
            elif not perm_at_chance:
                verdict = "PERMUTATION_LEAK"
                print(f"\n  Permutation control LEAKS; decoder "
                      f"doesn't use slot position; test design issue.",
                      flush=True)
            else:
                verdict = "SAME_POSITION_HOLDS"
                print(f"\n  Same-position control near main; "
                      f"position phasors not load-bearing.",
                      flush=True)
        else:
            verdict = "MAIN_BELOW_BAR"
    else:
        verdict = "CONTROLS_ONLY_NO_MAIN"

    out = {
        "backend": backend_name, "gpu": gpu, "seeds": SEEDS,
        "main_strict_top1_mean": main_mean,
        "ctrl_permutation_mean": perm_mean,
        "ctrl_random_phasor_mean": rand_mean,
        "ctrl_same_position_mean": same_mean,
        "per_seed_ctrl": seed_ctrls,
        "verdict": verdict, "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
