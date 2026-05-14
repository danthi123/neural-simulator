"""Sequential composition test: drive verb_word THEN motor_word (not
simultaneously) and test if engram-tagging still works.

The current compose_engram_demo uses simultaneous (verb+motor) drive
during encoding. Real conversation involves SEQUENTIAL words — user
says "go" then "north" with some delay. This test:

1. Encode each (verb, motor) pair via SEQUENTIAL drive:
   - Drive lang_input(verb) for verb_steps
   - Drive lang_input(motor) for motor_steps
   - Engram-recording captures spikes during BOTH windows
2. Test recall:
   - Stimulate engram → motor fires?
3. Cosine retrieval:
   - Drive lang_input(verb) then lang_input(motor) sequentially
   - Cosine-match the resulting firing pattern to stored engrams
"""
from __future__ import annotations
import argparse
import json
import numpy as np

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from research.runners.compose_engram_demo import recall_compose_tag
from research.runners.compose_engram_retrieval import cosine_sim
from sim.text_embeddings import orthogonal_drive_pattern


def encode_sequential(bridge, verb_word: str, motor_word: str, tag_name: str,
                       verb_steps: int = 100, motor_steps: int = 100,
                       inter_steps: int = 0,
                       drive_pA: float = 200.0,
                       sparsity: float = 0.05,
                       n_lang_input: int = 2048,
                       motor_teacher_pA: float = 0.0,
                       region_filter=None, top_k: int = 100,
                       verbose: bool = True):
    """Encode an engram via SEQUENTIAL drive: verb first, then motor.

    Returns (stats, pattern) where pattern is the per-neuron spike count
    accumulated across BOTH drive windows.
    """
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    rm = bridge.region_manager
    n_total = bridge.cp_external_input_current.shape[0]
    lang_input_idx = list(rm.indices("language_input"))
    lang_arr_gpu = cp.asarray(lang_input_idx, dtype=cp.int64)

    verb_drive = orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[verb_word], n_cues=16,
        n_neurons=n_lang_input, drive_max_pA=drive_pA, sparsity=sparsity,
    )
    motor_drive = orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[motor_word], n_cues=16,
        n_neurons=n_lang_input, drive_max_pA=drive_pA, sparsity=sparsity,
    )
    verb_gpu = cp.asarray(verb_drive, dtype=cp.float32)
    motor_gpu = cp.asarray(motor_drive, dtype=cp.float32)

    use_teacher = motor_teacher_pA > 0.0
    if use_teacher:
        motor_target_idx = list(rm.indices(_WORD_TO_POOL[motor_word]))
        motor_target_arr_gpu = cp.asarray(motor_target_idx, dtype=cp.int64)

    rf_mask = np.zeros(n_total, dtype=bool)
    if region_filter:
        for rname in region_filter:
            try:
                rf_mask[list(rm.indices(rname))] = True
            except Exception:
                pass

    # Brief reset
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()

    bridge.start_engram_recording(tag_name)
    pattern_accum = cp.zeros(n_total, dtype=cp.float32)

    # Phase A: drive verb_word only
    for _ in range(verb_steps):
        ext = cp.zeros(n_total, dtype=cp.float32)
        ext[lang_arr_gpu] = verb_gpu
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        if hasattr(bridge, "cp_firing_states"):
            pattern_accum += bridge.cp_firing_states.astype(cp.float32)

    # Inter-window gap (no drive)
    for _ in range(inter_steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
        if hasattr(bridge, "cp_firing_states"):
            pattern_accum += bridge.cp_firing_states.astype(cp.float32)

    # Phase B: drive motor_word (+ optional motor teacher)
    for _ in range(motor_steps):
        ext = cp.zeros(n_total, dtype=cp.float32)
        ext[lang_arr_gpu] = motor_gpu
        if use_teacher:
            ext[motor_target_arr_gpu] = motor_teacher_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        if hasattr(bridge, "cp_firing_states"):
            pattern_accum += bridge.cp_firing_states.astype(cp.float32)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()

    stats = bridge.commit_engram_tag(tag_name, top_k=top_k,
                                       region_filter=region_filter)
    if verbose:
        print(f"  [{tag_name}] tagged {stats['n_tagged']} neurons "
              f"({verb_steps + motor_steps + inter_steps} total steps)")

    pattern_host = to_host(pattern_accum)
    pattern_host[~rf_mask] = 0.0
    return stats, pattern_host


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--compose-pairs", type=str,
                    default="go:north,come:south,stop:west,look:east")
    p.add_argument("--verb-steps", type=int, default=100)
    p.add_argument("--motor-steps", type=int, default=100)
    p.add_argument("--inter-steps", type=int, default=0,
                    help="Gap between verb and motor windows (steps)")
    p.add_argument("--motor-teacher-pA", type=float, default=1500.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--recall-stim-pA", type=float, default=1500.0)
    p.add_argument("--recall-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    pairs = [tuple(s.strip().split(":")) for s in args.compose_pairs.split(",")]

    bridge = cpd.build_concept_bridge(
        seed=args.seed,
        n_lang_input=args.n_lang_input,
        n_per_pool=args.n_per_pool,
        n_fs_per_pool=args.n_fs_per_pool,
        enable_adjective=True,
        weak_dynamics=True,
        enable_direct_verb_to_motor=True,
        verbose=False,
    )
    bridge.load_checkpoint(args.load_bridge)

    region_filter = (
        [f"verb_pool_{v}" for v in ["GO", "COME", "STOP", "LOOK"]]
        + [f"noun_pool_{n}" for n in ["APPLE", "RIVER", "DOG", "CAT"]]
        + [f"adjective_pool_{a}" for a in ["BIG", "SMALL", "HOT", "COLD"]]
        + [f"motor_{a}" for a in ["N", "E", "S", "W"]]
    )

    print(f"=== compose_sequential_engram (seed={args.seed}) ===")
    print(f"  Pairs: {pairs}")
    print(f"  Verb steps: {args.verb_steps}, Inter: {args.inter_steps}, "
          f"Motor steps: {args.motor_steps}")
    print()

    print("[ENCODE] Sequential drive: verb_word THEN motor_word")
    encoded = {}
    for verb, motor in pairs:
        tag = f"{verb}_{motor}"
        _, pattern = encode_sequential(
            bridge, verb, motor, tag,
            verb_steps=args.verb_steps,
            motor_steps=args.motor_steps,
            inter_steps=args.inter_steps,
            motor_teacher_pA=args.motor_teacher_pA,
            sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
            region_filter=region_filter,
            top_k=args.top_k,
            verbose=True,
        )
        encoded[tag] = {"verb": verb, "motor": motor, "pattern": pattern}

    # Test RETRIEVAL: drive sequential queries, cosine-match to encoded patterns
    print()
    print("[RETRIEVAL] Sequential query drives, cosine match against stored patterns")
    n_match = 0
    n_total = bridge.cp_external_input_current.shape[0]
    rf_mask = np.zeros(n_total, dtype=bool)
    for rname in region_filter:
        try:
            rf_mask[list(bridge.region_manager.indices(rname))] = True
        except Exception:
            pass

    for verb, motor in pairs:
        true_tag = f"{verb}_{motor}"
        # Drive sequentially (verb then motor) and accumulate firing
        from sim.backend import get_backend, to_host
        cp, _ = get_backend()
        rm = bridge.region_manager
        lang_input_idx = list(rm.indices("language_input"))
        lang_arr_gpu = cp.asarray(lang_input_idx, dtype=cp.int64)
        verb_drive_np = orthogonal_drive_pattern(
            cue_idx=_WORD_TO_IDX[verb], n_cues=16,
            n_neurons=args.n_lang_input, drive_max_pA=200.0,
            sparsity=args.sparsity,
        )
        motor_drive_np = orthogonal_drive_pattern(
            cue_idx=_WORD_TO_IDX[motor], n_cues=16,
            n_neurons=args.n_lang_input, drive_max_pA=200.0,
            sparsity=args.sparsity,
        )
        verb_gpu = cp.asarray(verb_drive_np, dtype=cp.float32)
        motor_gpu = cp.asarray(motor_drive_np, dtype=cp.float32)

        bridge.cp_external_input_current[:] = 0.0
        bridge.clear_tag_drive()
        for _ in range(30):
            bridge._run_one_simulation_step()

        pattern_accum = cp.zeros(n_total, dtype=cp.float32)
        # Drive verb
        for _ in range(args.verb_steps):
            ext = cp.zeros(n_total, dtype=cp.float32)
            ext[lang_arr_gpu] = verb_gpu
            bridge.cp_external_input_current[:] = ext
            bridge._run_one_simulation_step()
            if hasattr(bridge, "cp_firing_states"):
                pattern_accum += bridge.cp_firing_states.astype(cp.float32)
        # Inter
        for _ in range(args.inter_steps):
            bridge.cp_external_input_current[:] = 0.0
            bridge._run_one_simulation_step()
            if hasattr(bridge, "cp_firing_states"):
                pattern_accum += bridge.cp_firing_states.astype(cp.float32)
        # Drive motor
        for _ in range(args.motor_steps):
            ext = cp.zeros(n_total, dtype=cp.float32)
            ext[lang_arr_gpu] = motor_gpu
            bridge.cp_external_input_current[:] = ext
            bridge._run_one_simulation_step()
            if hasattr(bridge, "cp_firing_states"):
                pattern_accum += bridge.cp_firing_states.astype(cp.float32)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(20):
            bridge._run_one_simulation_step()

        query_pattern = to_host(pattern_accum)
        query_pattern[~rf_mask] = 0.0
        scores = {t: cosine_sim(query_pattern, d["pattern"])
                   for t, d in encoded.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best = ranked[0][0]
        score = ranked[0][1]
        is_match = (best == true_tag)
        if is_match:
            n_match += 1
        marker = "MATCH" if is_match else "MISS"
        print(f"  {true_tag:18s} matched={best:18s} score={score:.3f} [{marker}]")
    print(f"[RETRIEVAL VERDICT] {n_match}/{len(pairs)} sequential queries retrieve TRUE engram")

    # Test recall via stimulation
    print()
    print("[RECALL] Stimulating each engram tag")
    n_pass = 0
    for tag, d in encoded.items():
        rates = recall_compose_tag(
            bridge, tag,
            drive_pA=args.recall_stim_pA,
            recall_steps=args.recall_steps,
        )
        target_pool = _WORD_TO_POOL[d["motor"]]
        target_rate = rates[target_pool]
        off = max(v for k, v in rates.items() if k != target_pool)
        passed = target_rate > off
        if passed:
            n_pass += 1
        marker = "PASS" if passed else "FAIL"
        print(f"  {tag:18s} target={target_rate:.3f} off={off:.3f} "
              f"ratio={target_rate/max(off, 0.001):.2f}x [{marker}]")
    print()
    print(f"[VERDICT] Sequential encoding: {n_pass}/{len(pairs)} compose pairs recall correctly")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed,
                "verb_steps": args.verb_steps,
                "motor_steps": args.motor_steps,
                "inter_steps": args.inter_steps,
                "motor_teacher_pA": args.motor_teacher_pA,
                "n_pass": n_pass,
                "n_total": len(pairs),
            }, f, indent=2)


if __name__ == "__main__":
    main()
