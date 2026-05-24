"""Direction E substrate Task 1 SMELL TEST: 3 anti-cheat controls.

Per discipline (mirrors Direction A smell test pattern): scrutinize
any PASS harder than a FAIL via per-bridge controls reusing the
cached bridges + tags.

For theta-gamma temporal phase code, the natural controls are:
  (A) WRONG-SLOT WINDOW: stim engram + read lang_output during
      WRONG gamma slot window (slot 0 instead of slot 2).
      If accuracy stays high -> slot windowing is not load-bearing.
  (B) NO-STIM: read lang_output during slot-2 window but skip
      engram stim. If accuracy stays high -> engram not load-bearing.
  (C) NO-WINDOW: stim engram + read lang_output across the FULL
      retrieval theta cycle (no slot windowing). If accuracy stays
      high -> the entire mechanism collapses to multitag.

Per Task 1's pre-registered bar: 0.80 multi-seed STRICT TOP-1.
The smell test verdict scrutinizes whether the main PASS (if any) is
genuinely load-bearing on the slot-windowing mechanism.

Reuses cached trained bridges from Task 1's per-seed cache. ~10-15
min wall (3 controls x 8 seqs x 3 seeds x 3 repeats stim+settle).
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
    build_concept_bridge, DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB,
    ADJECTIVE_VOCAB,
)
from research.findings.raw.generative_replay_sequence_vocab import (
    generate_k_stored_sequences,
)
from research.findings.raw.direction_A_ec_context_sequence_full import (
    _build_region_filter,
)
from research.findings.raw.direction_E_substrate_task1_full import (
    _bridge_save_path, encode_gamma_slot, _word_score,
    phase_to_gamma_slot, N_LANG_INPUT, N_PER_POOL, N_FS_PER_POOL,
    K_PAIRS, SLOT_COUNT, STIM_DRIVE_PA, N_REPEATS_PER_TAG,
    THETA_MS, N_GAMMA, N_THETA_CYCLES_RECALL,
)
from sim.backend import get_backend, is_gpu_backend


SEEDS = [42, 43, 44]
OUT_JSON = os.path.join(
    _HERE, "direction_E_substrate_task1_smell_test.json")
BAR = 0.80


def capture_no_window(bridge, tag_name, theta_steps, n_gamma,
                        n_lang_output):
    """Control C: stim engram + read FULL theta cycle (no gamma slot
    windowing)."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_out_idx = list(rm.indices("language_output"))
    lang_out_arr = cp.asarray(lang_out_idx, dtype=cp.int64)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    bridge.stimulate_tag(tag_name, drive_pA=STIM_DRIVE_PA,
                          additive=False)

    lang_counts = cp.zeros(n_lang_output, dtype=cp.float32)
    recall_steps = N_THETA_CYCLES_RECALL * theta_steps
    for step in range(recall_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[lang_out_arr]
        lang_counts = lang_counts + fired.astype(cp.float32)
    bridge.clear_tag_drive(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    return cp.asnumpy(lang_counts)


def capture_wrong_slot(bridge, tag_name, wrong_slot_idx, theta_steps,
                          n_gamma, n_lang_output):
    """Control A: stim engram + read WRONG gamma slot window."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_out_idx = list(rm.indices("language_output"))
    lang_out_arr = cp.asarray(lang_out_idx, dtype=cp.int64)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    bridge.stimulate_tag(tag_name, drive_pA=STIM_DRIVE_PA,
                          additive=False)

    lang_counts = cp.zeros(n_lang_output, dtype=cp.float32)
    recall_steps = N_THETA_CYCLES_RECALL * theta_steps
    for step in range(recall_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        current_slot = phase_to_gamma_slot(step, theta_steps, n_gamma)
        if current_slot == wrong_slot_idx:
            fired = bridge.cp_firing_states[lang_out_arr]
            lang_counts = lang_counts + fired.astype(cp.float32)
    bridge.clear_tag_drive(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    return cp.asnumpy(lang_counts)


def capture_no_stim(bridge, cue_slot_idx, theta_steps, n_gamma,
                      n_lang_output):
    """Control B: no engram stim; read lang_output during slot-i
    window across a full theta cycle."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_out_idx = list(rm.indices("language_output"))
    lang_out_arr = cp.asarray(lang_out_idx, dtype=cp.int64)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    lang_counts = cp.zeros(n_lang_output, dtype=cp.float32)
    recall_steps = N_THETA_CYCLES_RECALL * theta_steps
    for step in range(recall_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        current_slot = phase_to_gamma_slot(step, theta_steps, n_gamma)
        if current_slot == cue_slot_idx:
            fired = bridge.cp_firing_states[lang_out_arr]
            lang_counts = lang_counts + fired.astype(cp.float32)
    bridge.cp_external_input_current[:] = 0.0
    return cp.asnumpy(lang_counts)


def run_controls_for_seed(seed, verbose=True):
    print(f"\n--- seed {seed} controls ---", flush=True)
    bridge_p = _bridge_save_path(seed)
    if not os.path.exists(bridge_p):
        print(f"  [seed {seed}] bridge cache missing; skip",
              flush=True)
        return None

    cp, _ = get_backend()
    words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
             list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))
    n_words = len(words)
    bridge = build_concept_bridge(
        seed=seed, n_lang_input=N_LANG_INPUT, n_per_pool=N_PER_POOL,
        n_fs_per_pool=N_FS_PER_POOL, enable_adjective=True,
        weak_dynamics=True, enable_positional_context=False,
        verbose=False,
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

    # Re-encode tags (deterministic; safe)
    tag_names = []
    for seq_idx, seq in enumerate(sequences):
        tag, stats = encode_gamma_slot(
            bridge, seq, words, seq_idx, region_filter,
            theta_steps, N_GAMMA)
        tag_names.append(tag)
        if verbose:
            print(f"  [seed {seed}] re-encoded {tag} n_tagged="
                  f"{stats.get('n_tagged', 0)}", flush=True)

    n_top1_wrong = 0
    n_top1_nostim = 0
    n_top1_nowindow = 0
    per_seq_ctrl = []
    for seq_idx, (seq, tag_name) in enumerate(zip(sequences, tag_names)):
        true = seq[SLOT_COUNT - 1]

        # (A) Wrong slot window: cue slot 0 instead of slot SLOT_COUNT-1
        avg_wrong = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = capture_wrong_slot(
                bridge, tag_name, 0, theta_steps, N_GAMMA,
                n_lang_output)
            avg_wrong = avg_wrong + counts.astype(np.float64)
        avg_wrong = avg_wrong / N_REPEATS_PER_TAG
        scores = {w: _word_score(avg_wrong, w, n_lang_output, words)
                  for w in words}
        topK = sorted(scores.items(), key=lambda x: x[1],
                       reverse=True)
        top1_wrong = (topK[0][0] == true)
        if top1_wrong: n_top1_wrong += 1

        # (B) No stim
        avg_nostim = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = capture_no_stim(
                bridge, SLOT_COUNT - 1, theta_steps, N_GAMMA,
                n_lang_output)
            avg_nostim = avg_nostim + counts.astype(np.float64)
        avg_nostim = avg_nostim / N_REPEATS_PER_TAG
        scores = {w: _word_score(avg_nostim, w, n_lang_output, words)
                  for w in words}
        topK = sorted(scores.items(), key=lambda x: x[1],
                       reverse=True)
        top1_nostim = (topK[0][0] == true)
        if top1_nostim: n_top1_nostim += 1

        # (C) No window (full theta cycle stim)
        avg_nowindow = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = capture_no_window(
                bridge, tag_name, theta_steps, N_GAMMA,
                n_lang_output)
            avg_nowindow = avg_nowindow + counts.astype(np.float64)
        avg_nowindow = avg_nowindow / N_REPEATS_PER_TAG
        scores = {w: _word_score(avg_nowindow, w, n_lang_output, words)
                  for w in words}
        topK = sorted(scores.items(), key=lambda x: x[1],
                       reverse=True)
        top1_nowindow = (topK[0][0] == true)
        if top1_nowindow: n_top1_nowindow += 1

        per_seq_ctrl.append({
            "seq_idx": seq_idx, "sequence": list(seq),
            "true_slot": true,
            "wrong_slot_top1": top1_wrong,
            "no_stim_top1": top1_nostim,
            "no_window_top1": top1_nowindow,
        })

    acc_wrong = n_top1_wrong / K_PAIRS
    acc_nostim = n_top1_nostim / K_PAIRS
    acc_nowindow = n_top1_nowindow / K_PAIRS

    print(f"  [seed {seed}] wrong_slot top-1={acc_wrong:.3f}; "
          f"no_stim top-1={acc_nostim:.3f}; no_window top-1="
          f"{acc_nowindow:.3f}", flush=True)
    return {
        "seed": seed,
        "ctrl_wrong_slot_top1": acc_wrong,
        "ctrl_no_stim_top1": acc_nostim,
        "ctrl_no_window_top1": acc_nowindow,
        "per_seq_ctrl": per_seq_ctrl,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction E substrate Task 1 SMELL TEST ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Controls: (A) wrong slot window, (B) no-stim, "
          f"(C) no-window. For TRUE positional-binding PASS, main "
          f"strict top-1 must exceed all three controls "
          f"significantly.", flush=True)

    main_p = os.path.join(
        _HERE, "direction_E_substrate_task1_full.json")
    main_result = None
    if os.path.exists(main_p):
        with open(main_p, "r", encoding="utf-8") as f:
            main_result = json.load(f)
        main_mean = main_result.get("strict_top1_mean", float("nan"))
        print(f"  Main strict top-1 multi-seed: {main_mean:.3f}",
              flush=True)
    else:
        print(f"  [WARN] main result not found", flush=True)
        main_mean = float("nan")

    seed_ctrls = []
    t0 = time.time()
    for seed in args.seeds:
        ctrl = run_controls_for_seed(seed)
        if ctrl is not None:
            seed_ctrls.append(ctrl)
    total_min = (time.time() - t0) / 60

    if not seed_ctrls:
        print("[FATAL] no cached bridges; cannot run smell test",
              flush=True)
        return 1

    acc_wrong_mean = float(np.mean(
        [c["ctrl_wrong_slot_top1"] for c in seed_ctrls]))
    acc_nostim_mean = float(np.mean(
        [c["ctrl_no_stim_top1"] for c in seed_ctrls]))
    acc_nowindow_mean = float(np.mean(
        [c["ctrl_no_window_top1"] for c in seed_ctrls]))

    print(f"\n=== CONTROL RESULTS (multi-seed mean) ===", flush=True)
    print(f"  (A) wrong slot window: {acc_wrong_mean:.3f}",
          flush=True)
    print(f"  (B) no-stim:           {acc_nostim_mean:.3f}",
          flush=True)
    print(f"  (C) no-window:         {acc_nowindow_mean:.3f}",
          flush=True)
    print(f"  Wall: {total_min:.1f} min", flush=True)

    if main_result is not None and not np.isnan(main_mean):
        margin_wrong = main_mean - acc_wrong_mean
        margin_nostim = main_mean - acc_nostim_mean
        margin_nowindow = main_mean - acc_nowindow_mean
        print(f"\n  Margins (main - control):", flush=True)
        print(f"    wrong_slot: {margin_wrong:+.3f}", flush=True)
        print(f"    no_stim:    {margin_nostim:+.3f}", flush=True)
        print(f"    no_window:  {margin_nowindow:+.3f}", flush=True)

        if main_mean >= BAR:
            if (margin_wrong > 0.2 and margin_nostim > 0.2
                    and margin_nowindow > 0.2):
                verdict = "PASS_CONTROLS_DECISIVE"
                print(f"\n  ALL CONTROLS PASS: positional binding "
                      f"via theta-gamma genuinely load-bearing.",
                      flush=True)
            elif margin_nowindow <= 0.1:
                verdict = "PASS_COLLAPSES_TO_MULTITAG"
                print(f"\n  No-window control near main; slot "
                      f"windowing NOT load-bearing; mechanism "
                      f"collapses to multitag.", flush=True)
            elif margin_wrong <= 0.1:
                verdict = "PASS_COLLAPSES_TO_WRONG_SLOT_INSENSITIVE"
                print(f"\n  Wrong-slot control near main; slot "
                      f"selection NOT load-bearing.", flush=True)
            else:
                verdict = "PASS_WITH_WEAK_CONTROLS"
                print(f"\n  Some controls margins < 0.2; partial.",
                      flush=True)
        else:
            verdict = "MAIN_BELOW_BAR_CONTROLS_RECORDED"
            print(f"\n  Main {main_mean:.3f} below 0.80 bar; "
                  f"controls recorded for diagnosis.", flush=True)
    else:
        verdict = "CONTROLS_ONLY_NO_MAIN"

    out = {
        "backend": backend_name, "gpu": gpu, "seeds": args.seeds,
        "main_strict_top1_mean": main_mean,
        "ctrl_wrong_slot_top1_mean": acc_wrong_mean,
        "ctrl_no_stim_top1_mean": acc_nostim_mean,
        "ctrl_no_window_top1_mean": acc_nowindow_mean,
        "per_seed_ctrl": seed_ctrls,
        "verdict": verdict, "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
