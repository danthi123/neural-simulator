"""Direction G SMELL TEST: anti-cheat controls for HIPPO+theta-gamma
substrate sequence storage.

Per discipline: scrutinize PASS harder than FAIL. Same 3 controls
mirror Direction E Task 1 smell test pattern: wrong-slot window /
no-stim / no-window; load-bearing strict top-1 metric.

For Direction G specifically: extra question -- did the hippocampal
SWR consolidation pathway help, or hurt, or pass-through unchanged?
Smell test answers this via the no-window control: if no-window
~= main, the slot windowing isn't doing positional work; if no-stim
~= main, the engram isn't load-bearing (and the cue alone via
hippocampus is sufficient -- biology-significant).

~10-15 min wall after Direction G completes; reuses cached bridges +
tags via Direction E Task 1 smell test infrastructure.

Reuse-by-import only; no protected/frozen/moat module modified.
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
    encode_gamma_slot, phase_to_gamma_slot, _word_score,
    K_PAIRS, SLOT_COUNT, STIM_DRIVE_PA, N_REPEATS_PER_TAG,
    THETA_MS, N_GAMMA, N_THETA_CYCLES_RECALL,
    N_LANG_INPUT, N_PER_POOL, N_FS_PER_POOL,
)
from research.findings.raw.direction_E_substrate_task1_smell_test import (
    capture_no_window, capture_wrong_slot, capture_no_stim,
)
from research.findings.raw.direction_G_hippo_theta_gamma_substrate import (
    _bridge_save_path,
)
from sim.backend import get_backend, is_gpu_backend


SEEDS = [42, 43, 44]
OUT_JSON = os.path.join(_HERE, "direction_G_smell_test.json")
BAR = 0.80


def run_controls_for_seed(seed, verbose=True):
    print(f"\n--- seed {seed} controls ---", flush=True)
    bridge_p = _bridge_save_path(seed)
    if not os.path.exists(bridge_p):
        print(f"  [seed {seed}] HIPPO bridge cache missing; skip",
              flush=True)
        return None

    cp, _ = get_backend()
    words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
             list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))
    n_words = len(words)
    bridge = _build_bridge_with_hippo(
        seed=seed, enable_adjective=True,
        n_lang_input=N_LANG_INPUT, n_per_pool=N_PER_POOL,
        n_fs_per_pool=N_FS_PER_POOL, verbose=False,
    )
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

    rm = bridge.region_manager
    n_lang_output = N_LANG_INPUT
    region_filter = _build_region_filter(rm)
    cfg = bridge.core_config
    theta_steps = max(2, int(round(THETA_MS / cfg.dt_ms)))

    sequences = generate_k_stored_sequences(
        seed=seed, k=K_PAIRS, n_words=n_words,
        slot_count=SLOT_COUNT, vocab=words)

    tag_names = []
    for seq_idx, seq in enumerate(sequences):
        tag, stats = encode_gamma_slot(
            bridge, seq, words, seq_idx, region_filter,
            theta_steps, N_GAMMA)
        tag_names.append(tag)

    n_top1_wrong = 0; n_top1_nostim = 0; n_top1_nowindow = 0
    per_seq_ctrl = []
    for seq_idx, (seq, tag_name) in enumerate(zip(sequences, tag_names)):
        true = seq[SLOT_COUNT - 1]

        # (A) Wrong slot
        avg_wrong = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = capture_wrong_slot(
                bridge, tag_name, 0, theta_steps, N_GAMMA,
                n_lang_output)
            avg_wrong += counts.astype(np.float64)
        avg_wrong /= N_REPEATS_PER_TAG
        scores = {w: _word_score(avg_wrong, w, n_lang_output, words)
                  for w in words}
        topK = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if topK[0][0] == true: n_top1_wrong += 1

        # (B) No stim
        avg_nostim = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = capture_no_stim(
                bridge, SLOT_COUNT - 1, theta_steps, N_GAMMA,
                n_lang_output)
            avg_nostim += counts.astype(np.float64)
        avg_nostim /= N_REPEATS_PER_TAG
        scores = {w: _word_score(avg_nostim, w, n_lang_output, words)
                  for w in words}
        topK = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if topK[0][0] == true: n_top1_nostim += 1

        # (C) No window
        avg_nowindow = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = capture_no_window(
                bridge, tag_name, theta_steps, N_GAMMA,
                n_lang_output)
            avg_nowindow += counts.astype(np.float64)
        avg_nowindow /= N_REPEATS_PER_TAG
        scores = {w: _word_score(avg_nowindow, w, n_lang_output, words)
                  for w in words}
        topK = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if topK[0][0] == true: n_top1_nowindow += 1

    return {
        "seed": seed,
        "ctrl_wrong_slot_top1": n_top1_wrong / K_PAIRS,
        "ctrl_no_stim_top1": n_top1_nostim / K_PAIRS,
        "ctrl_no_window_top1": n_top1_nowindow / K_PAIRS,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction G SMELL TEST (HIPPO+theta-gamma) ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)

    main_p = os.path.join(
        _HERE, "direction_G_hippo_theta_gamma_substrate.json")
    if os.path.exists(main_p):
        with open(main_p, "r", encoding="utf-8") as f:
            main_result = json.load(f)
        main_mean = main_result.get("strict_top1_mean", float("nan"))
        print(f"  Main strict top-1 multi-seed: {main_mean:.3f}",
              flush=True)
    else:
        print(f"  [WARN] Direction G main result not found",
              flush=True)
        main_mean = float("nan")
        main_result = None

    seed_ctrls = []
    t0 = time.time()
    for seed in args.seeds:
        ctrl = run_controls_for_seed(seed)
        if ctrl is not None:
            seed_ctrls.append(ctrl)
    total_min = (time.time() - t0) / 60

    if not seed_ctrls:
        print("[FATAL] no HIPPO bridges cached", flush=True)
        return 1

    acc_wrong = float(np.mean(
        [c["ctrl_wrong_slot_top1"] for c in seed_ctrls]))
    acc_nostim = float(np.mean(
        [c["ctrl_no_stim_top1"] for c in seed_ctrls]))
    acc_nowindow = float(np.mean(
        [c["ctrl_no_window_top1"] for c in seed_ctrls]))

    print(f"\n=== CONTROLS multi-seed mean ===", flush=True)
    print(f"  (A) wrong-slot:  {acc_wrong:.3f}", flush=True)
    print(f"  (B) no-stim:     {acc_nostim:.3f}", flush=True)
    print(f"  (C) no-window:   {acc_nowindow:.3f}", flush=True)
    print(f"  Wall: {total_min:.1f} min", flush=True)

    if main_result is not None and not np.isnan(main_mean):
        margin_wrong = main_mean - acc_wrong
        margin_nostim = main_mean - acc_nostim
        margin_nowindow = main_mean - acc_nowindow
        print(f"\n  Margins (main - ctrl):", flush=True)
        print(f"    wrong_slot: {margin_wrong:+.3f}", flush=True)
        print(f"    no_stim:    {margin_nostim:+.3f}", flush=True)
        print(f"    no_window:  {margin_nowindow:+.3f}", flush=True)

        if main_mean >= BAR:
            if (margin_wrong > 0.2 and margin_nostim > 0.2
                    and margin_nowindow > 0.2):
                verdict = "PASS_CONTROLS_DECISIVE_HIPPO_THETA_GAMMA"
                print(f"\n  ALL CONTROLS DECISIVE; HIPPO+theta-gamma"
                      f" genuinely load-bearing; pillar n=105 "
                      f"candidate.", flush=True)
            elif margin_nowindow <= 0.1:
                verdict = "PASS_COLLAPSES_TO_MULTITAG"
                print(f"\n  No-window collapses; slot windowing "
                      f"NOT load-bearing.", flush=True)
            elif margin_wrong <= 0.1:
                verdict = "PASS_COLLAPSES_WRONG_SLOT_INSENSITIVE"
                print(f"\n  Wrong-slot collapses; slot selection "
                      f"not load-bearing.", flush=True)
            else:
                verdict = "PASS_WEAK_CONTROLS"
        else:
            verdict = "MAIN_BELOW_BAR_CONTROLS_RECORDED"
    else:
        verdict = "CONTROLS_ONLY_NO_MAIN"

    out = {
        "backend": backend_name, "gpu": gpu, "seeds": args.seeds,
        "main_strict_top1_mean": main_mean,
        "ctrl_wrong_slot_top1_mean": acc_wrong,
        "ctrl_no_stim_top1_mean": acc_nostim,
        "ctrl_no_window_top1_mean": acc_nowindow,
        "per_seed_ctrl": seed_ctrls,
        "verdict": verdict, "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
