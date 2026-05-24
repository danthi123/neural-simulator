"""Direction A POST-RUN smell test: scrutinize any PASS harder than a
FAIL via three anti-cheat controls reusing the cached bridges + tags.

Controls:
  (A) WRONG-POSITION CUE: stim each engram tag with positional cue
      for slot 0 (not slot 2). If slot-2 accuracy stays high, the
      ec_context positional cue is NOT load-bearing -- the engram
      stim alone is doing the work (which would mean Direction A
      collapses to plain multitag, not sequence storage).
  (B) NO-STIM CONTROL: provide positional cue but do NOT stim the
      tag. If accuracy stays high, engram is not load-bearing.
  (C) NO-CUE CONTROL: stim the tag but do NOT drive ec_context.
      If accuracy stays high, ec_context positional binding is not
      load-bearing.

For a TRUE positional-binding PASS, the main result must significantly
exceed (A) and (B) and (C). For a TRUE multitag-only result (degenerate
to the validated multitag mechanism), (C) ~= main result.

Per discipline: recompute from the recorded trials/bridges, no re-run,
no bar changes, no overclaim. Honest propagation of every outcome.

~10-15 min wall clock (loads cached bridges + 3 controls x 8 seqs
each x 3 seeds = 72 stim trials).
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
    _build_region_filter, _encode_sequence_with_ec_context,
    _capture_lang_output_with_pos_cue, _word_score, _bridge_save_path,
    K_PAIRS, SLOT_COUNT, TOP_K_READOUT, EC_DRIVE_PA, STIM_DRIVE_PA,
    STIM_STEPS, READOUT_STEPS, N_REPEATS_PER_TAG, N_EC_CONTEXT,
    N_LANG_INPUT, N_PER_POOL, N_FS_PER_POOL,
)
from sim.text_embeddings import (
    orthogonal_drive_pattern, positional_drive_pattern,
)
from sim.backend import get_backend, is_gpu_backend


SEEDS = [42, 43, 44]
OUT_JSON = os.path.join(_HERE, "direction_A_smell_test.json")


def _capture_wrong_position(bridge, tag_name, wrong_cue_slot_idx,
                              n_ec, ec_arr, n_lang_output):
    """Control A: stim tag with WRONG-POSITION ec_context cue.
    Reuses _capture_lang_output_with_pos_cue (already uses arbitrary
    cue_slot_idx)."""
    return _capture_lang_output_with_pos_cue(
        bridge, tag_name, wrong_cue_slot_idx, n_ec, ec_arr, n_lang_output)


def _capture_no_stim(bridge, cue_slot_idx, n_ec, ec_arr, n_lang_output):
    """Control B: positional cue only, NO tag stim."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_out_idx = list(rm.indices("language_output"))
    lang_out_arr = cp.asarray(lang_out_idx, dtype=cp.int64)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    cue_pos = positional_drive_pattern(
        cue_slot_idx, n_neurons=n_ec, n_max_positions=SLOT_COUNT)
    cue_pos_pa = (cue_pos > 0).astype(np.float32) * EC_DRIVE_PA
    bridge.cp_external_input_current[ec_arr] = cp.asarray(
        cue_pos_pa, dtype=cp.float32)
    # NO stimulate_tag call.

    lang_counts = cp.zeros(n_lang_output, dtype=cp.float32)
    for _ in range(STIM_STEPS + READOUT_STEPS):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[lang_out_arr]
        lang_counts = lang_counts + fired.astype(cp.float32)
    bridge.cp_external_input_current[:] = 0.0
    return cp.asnumpy(lang_counts)


def _capture_no_cue(bridge, tag_name, n_lang_output):
    """Control C: stim tag with NO positional cue."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_out_idx = list(rm.indices("language_output"))
    lang_out_arr = cp.asarray(lang_out_idx, dtype=cp.int64)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    # NO ec_context cue.
    bridge.stimulate_tag(tag_name, drive_pA=STIM_DRIVE_PA,
                          additive=False)

    lang_counts = cp.zeros(n_lang_output, dtype=cp.float32)
    for _ in range(STIM_STEPS + READOUT_STEPS):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[lang_out_arr]
        lang_counts = lang_counts + fired.astype(cp.float32)
    bridge.clear_tag_drive(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    return cp.asnumpy(lang_counts)


def run_controls_for_seed(seed, verbose=True):
    """Loads cached bridge for seed; re-runs encoding to get tags
    (engram tags don't persist through load/restore byte-equivalent);
    then runs three controls per sequence."""
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
        weak_dynamics=True, enable_positional_context=True,
        n_ec_context=N_EC_CONTEXT, verbose=False,
    )
    bridge.load_checkpoint(bridge_p)
    rm = bridge.region_manager
    ec_idx = list(rm.indices("ec_context"))
    ec_arr = cp.asarray(ec_idx, dtype=cp.int64)
    n_ec = len(ec_idx)
    n_lang_output = N_LANG_INPUT
    region_filter = _build_region_filter(rm)

    # Freeze plasticity gates (mirror full runner).
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

    sequences = generate_k_stored_sequences(
        seed=seed, k=K_PAIRS, n_words=n_words,
        slot_count=SLOT_COUNT, vocab=words)

    # Re-encode tags (engram tags persistence not verified across
    # save/load; safe to re-encode same seqs deterministically).
    tag_names = []
    for seq_idx, seq in enumerate(sequences):
        tag, stats = _encode_sequence_with_ec_context(
            bridge, seq, words, seq_idx, region_filter)
        tag_names.append(tag)
        if verbose:
            print(f"  [seed {seed}] re-encoded {tag} n_tagged="
                  f"{stats.get('n_tagged', 0)}", flush=True)

    # Run three controls per sequence.
    n_correct_wrong = 0
    n_correct_nostim = 0
    n_correct_nocue = 0
    per_seq_ctrl = []
    for seq_idx, (seq, tag_name) in enumerate(zip(sequences, tag_names)):
        true_slot3 = seq[SLOT_COUNT - 1]

        # (A) Wrong-position cue: cue slot 0 instead of slot 2.
        avg_wrong = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = _capture_wrong_position(
                bridge, tag_name, 0, n_ec, ec_arr, n_lang_output)
            avg_wrong = avg_wrong + counts.astype(np.float64)
        avg_wrong = avg_wrong / N_REPEATS_PER_TAG
        scores_wrong = {w: _word_score(avg_wrong, w, n_lang_output, words)
                         for w in words}
        topK_wrong = sorted(scores_wrong.items(), key=lambda x: x[1],
                             reverse=True)[:TOP_K_READOUT]
        correct_wrong = true_slot3 in [w for w, s in topK_wrong]
        if correct_wrong: n_correct_wrong += 1

        # (B) No stim (positional cue only).
        avg_nostim = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = _capture_no_stim(
                bridge, SLOT_COUNT - 1, n_ec, ec_arr, n_lang_output)
            avg_nostim = avg_nostim + counts.astype(np.float64)
        avg_nostim = avg_nostim / N_REPEATS_PER_TAG
        scores_nostim = {w: _word_score(avg_nostim, w, n_lang_output, words)
                          for w in words}
        topK_nostim = sorted(scores_nostim.items(), key=lambda x: x[1],
                              reverse=True)[:TOP_K_READOUT]
        correct_nostim = true_slot3 in [w for w, s in topK_nostim]
        if correct_nostim: n_correct_nostim += 1

        # (C) No cue (tag stim only).
        avg_nocue = np.zeros(n_lang_output, dtype=np.float64)
        for _ in range(N_REPEATS_PER_TAG):
            counts = _capture_no_cue(bridge, tag_name, n_lang_output)
            avg_nocue = avg_nocue + counts.astype(np.float64)
        avg_nocue = avg_nocue / N_REPEATS_PER_TAG
        scores_nocue = {w: _word_score(avg_nocue, w, n_lang_output, words)
                         for w in words}
        topK_nocue = sorted(scores_nocue.items(), key=lambda x: x[1],
                             reverse=True)[:TOP_K_READOUT]
        correct_nocue = true_slot3 in [w for w, s in topK_nocue]
        if correct_nocue: n_correct_nocue += 1

        per_seq_ctrl.append({
            "seq_idx": seq_idx, "sequence": list(seq),
            "true_slot3": true_slot3,
            "wrong_pos_topK": [w for w, s in topK_wrong],
            "wrong_pos_correct": correct_wrong,
            "no_stim_topK": [w for w, s in topK_nostim],
            "no_stim_correct": correct_nostim,
            "no_cue_topK": [w for w, s in topK_nocue],
            "no_cue_correct": correct_nocue,
        })
        if verbose:
            print(f"    seq {seq_idx} true={true_slot3}: "
                  f"wrong_pos_corr={correct_wrong}; "
                  f"no_stim_corr={correct_nostim}; "
                  f"no_cue_corr={correct_nocue}", flush=True)

    acc_wrong = n_correct_wrong / K_PAIRS
    acc_nostim = n_correct_nostim / K_PAIRS
    acc_nocue = n_correct_nocue / K_PAIRS

    return {
        "seed": seed,
        "ctrl_wrong_position_acc": acc_wrong,
        "ctrl_no_stim_acc": acc_nostim,
        "ctrl_no_cue_acc": acc_nocue,
        "per_seq_ctrl": per_seq_ctrl,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction A SMELL TEST ===", flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Controls: (A) wrong-position cue, (B) no-stim, "
          f"(C) no-cue. For TRUE positional-binding PASS, main "
          f"slot3-accuracy must exceed all three controls "
          f"significantly.", flush=True)

    # Load main result for comparison.
    main_p = os.path.join(_HERE, "direction_A_ec_context_sequence_full.json")
    main_result = None
    if os.path.exists(main_p):
        with open(main_p, "r", encoding="utf-8") as f:
            main_result = json.load(f)
        main_mean = main_result.get("slot3_accuracy_mean", float("nan"))
        print(f"  Main result mean slot-3-accuracy: {main_mean:.3f}",
              flush=True)
    else:
        print(f"  [WARN] main result not found; smell test will still "
              f"run and report controls", flush=True)

    seed_ctrls = []
    t0 = time.time()
    for seed in args.seeds:
        ctrl = run_controls_for_seed(seed)
        if ctrl is not None:
            seed_ctrls.append(ctrl)
    total_min = (time.time() - t0) / 60

    if not seed_ctrls:
        print("[FATAL] no seeds had cached bridges; smell test "
              "cannot run", flush=True)
        return 1

    acc_wrong_mean = float(np.mean(
        [c["ctrl_wrong_position_acc"] for c in seed_ctrls]))
    acc_nostim_mean = float(np.mean(
        [c["ctrl_no_stim_acc"] for c in seed_ctrls]))
    acc_nocue_mean = float(np.mean(
        [c["ctrl_no_cue_acc"] for c in seed_ctrls]))

    print(f"\n=== CONTROL RESULTS (multi-seed mean) ===", flush=True)
    print(f"  (A) wrong-position cue: {acc_wrong_mean:.3f}",
          flush=True)
    print(f"  (B) no-stim (cue only): {acc_nostim_mean:.3f}",
          flush=True)
    print(f"  (C) no-cue (stim only): {acc_nocue_mean:.3f}",
          flush=True)

    chance = TOP_K_READOUT / 16.0
    print(f"  chance: {chance:.3f}", flush=True)

    if main_result is not None:
        main_mean = main_result.get("slot3_accuracy_mean", float("nan"))
        margin_wrong = main_mean - acc_wrong_mean
        margin_nostim = main_mean - acc_nostim_mean
        margin_nocue = main_mean - acc_nocue_mean
        print(f"\n  Main - wrong_pos margin: {margin_wrong:+.3f}",
              flush=True)
        print(f"  Main - no_stim   margin: {margin_nostim:+.3f}",
              flush=True)
        print(f"  Main - no_cue    margin: {margin_nocue:+.3f}",
              flush=True)

        if main_mean >= 0.80:
            # PASS: scrutinize harder
            if (margin_wrong > 0.2 and margin_nostim > 0.2
                    and margin_nocue > 0.2):
                verdict = "PASS_CONTROLS_DECISIVE"
                print(f"\n  ALL CONTROLS PASS: main {main_mean:.3f}"
                      f" decisively exceeds every control by > 0.2;"
                      f" the positional binding is genuinely "
                      f"load-bearing.", flush=True)
            elif margin_nocue <= 0.1:
                verdict = "PASS_COLLAPSES_TO_MULTITAG"
                print(f"\n  no_cue control near main result -- the "
                      f"ec_context cue is NOT load-bearing; "
                      f"Direction A collapses to plain multitag, "
                      f"not sequence-storage.", flush=True)
            elif margin_nostim <= 0.1:
                verdict = "PASS_COLLAPSES_TO_CUE_ALONE"
                print(f"\n  no_stim control near main result -- the "
                      f"engram tag is NOT load-bearing; cue alone "
                      f"drives concept pools via ec_context "
                      f"pathways.", flush=True)
            else:
                verdict = "PASS_WITH_WEAK_CONTROLS"
                print(f"\n  Some control margins < 0.2 -- positional"
                      f" binding contributes but not exclusively.",
                      flush=True)
        else:
            verdict = "MAIN_BELOW_BAR_CONTROLS_RECORDED"
            print(f"\n  Main result below 0.80 bar; controls "
                  f"recorded for diagnosis.", flush=True)
    else:
        verdict = "CONTROLS_ONLY_NO_MAIN"

    out = {
        "backend": backend_name, "gpu": gpu, "seeds": args.seeds,
        "main_result_summary": (
            {"slot3_accuracy_mean": main_result["slot3_accuracy_mean"],
             "slot3_accuracy_per_seed": main_result["slot3_accuracy_per_seed"]}
            if main_result is not None else None),
        "ctrl_wrong_position_mean": acc_wrong_mean,
        "ctrl_no_stim_mean": acc_nostim_mean,
        "ctrl_no_cue_mean": acc_nocue_mean,
        "per_seed_ctrl": seed_ctrls,
        "chance_baseline": chance,
        "verdict": verdict,
        "wall_clock_minutes": total_min,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    print(f"Wall: {total_min:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
