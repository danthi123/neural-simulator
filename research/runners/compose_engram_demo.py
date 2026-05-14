"""Engram-based composition demo (catalog D.14 — Tonegawa engram-tagging).

Bypasses STDP pathway growth entirely. Instead of growing verb_pool -> motor
weights via compose-training, this approach:

1. ENCODING: For each (verb, motor) compose pair, drive lang_input(verb) +
   lang_input(motor) simultaneously during a recording window. The
   start_engram_recording / commit_engram_tag API records which neurons
   fired together and tags them as "go_north", "come_south", etc.

2. INFERENCE: Stimulate the engram tag with strong external current.
   The tagged neurons (which span verb_pool + motor) fire together,
   reactivating the original compositional ensemble.

3. ANTI-CHEAT: 24 permutations test — does each tag stimulation produce
   preferential firing of its TRUE-trained motor pool?

Catalog grounding: D.14 (Tonegawa 2012-2017 engram cells), T1.C
behavioral check (Liu 2012 inception-of-fear paradigm).

Usage:
    python -m research.runners.compose_engram_demo \\
        --load-bridge research/findings/raw/g11_bg/concept_pool_demo/seed42_v16.simstate.h5 \\
        --seed 42 \\
        --encoding-steps 200 \\
        --recall-stim-pA 1500 \\
        --recall-steps 100 \\
        --out research/findings/raw/g11_bg/concept_pool_demo/seed42_engram.json
"""
from __future__ import annotations
import argparse
import itertools
import json
import time

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from sim.text_embeddings import orthogonal_drive_pattern


def encode_compose_pair(bridge, verb_word: str, motor_word: str,
                         tag_name: str,
                         encoding_steps: int = 200,
                         drive_pA: float = 200.0,
                         sparsity: float = 0.05,
                         n_lang_input: int = 2048,
                         n_words_for_orthogonal: int = 16,
                         region_filter=None,
                         top_k: int = 100,
                         motor_teacher_pA: float = 0.0,
                         verbose: bool = True):
    """Encode one (verb, motor) pair as an engram tag.

    Drives lang_input(verb) + lang_input(motor) together. Records spike
    counts across all neurons in region_filter. Commits the tag using
    top-K selection (sparse Marr-like engram).
    """
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager

    # Drive patterns
    verb_drive = orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[verb_word],
        n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=drive_pA, sparsity=sparsity,
    )
    motor_drive = orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[motor_word],
        n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=drive_pA, sparsity=sparsity,
    )
    both_gpu = cp.asarray(verb_drive + motor_drive, dtype=cp.float32)
    lang_input_idx = list(rm.indices("language_input"))
    lang_arr_gpu = cp.asarray(lang_input_idx, dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    # Optional motor teacher: force the target motor pool to fire strongly
    # during encoding, ensuring the engram tag includes motor neurons.
    use_motor_teacher = motor_teacher_pA > 0.0
    motor_target_pool = _WORD_TO_POOL[motor_word]
    if use_motor_teacher:
        motor_target_idx = list(rm.indices(motor_target_pool))
        motor_target_arr_gpu = cp.asarray(motor_target_idx, dtype=cp.int64)

    # Start recording
    bridge.start_engram_recording(tag_name)

    # Brief reset before encoding
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()

    # Drive both words, record co-firing
    for _ in range(encoding_steps):
        ext = cp.zeros(n_total, dtype=cp.float32)
        ext[lang_arr_gpu] = both_gpu
        if use_motor_teacher:
            ext[motor_target_arr_gpu] = motor_teacher_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()

    # Reset
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()

    # Commit tag — sparse top-K across verb_pool + motor regions
    stats = bridge.commit_engram_tag(
        tag_name, top_k=top_k, region_filter=region_filter,
    )
    if verbose:
        print(f"  [{tag_name}] tagged {stats['n_tagged']} neurons "
              f"in {stats['window_ms']:.0f}ms window (mean rate "
              f"{stats['mean_spike_count']:.1f} sp/n)")
    return stats


def recall_compose_tag(bridge, tag_name: str,
                        drive_pA: float = 1500.0,
                        recall_steps: int = 100):
    """Stimulate the engram tag and measure motor pool firing.
    Returns spike rate per motor pool.
    """
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager

    motor_pools = ["motor_N", "motor_E", "motor_S", "motor_W"]
    motor_arrs = {p: cp.asarray(list(rm.indices(p)), dtype=cp.int64)
                   for p in motor_pools}
    spike_counts = {p: 0 for p in motor_pools}

    # Reset bridge state before recall
    bridge.cp_external_input_current[:] = 0.0
    bridge.clear_tag_drive()
    for _ in range(30):
        bridge._run_one_simulation_step()

    # Stimulate the tag for recall_steps
    bridge.stimulate_tag(tag_name, drive_pA=drive_pA, additive=False)
    for _ in range(recall_steps):
        bridge._run_one_simulation_step()
        if hasattr(bridge, "cp_firing_states"):
            firing = bridge.cp_firing_states
            for p, arr in motor_arrs.items():
                spike_counts[p] += int(firing[arr].sum())

    # Clear tag drive
    bridge.clear_tag_drive(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()

    rates = {p: spike_counts[p] / (recall_steps * len(motor_arrs[p]))
              for p in motor_pools}
    return rates


def main():
    p = argparse.ArgumentParser(description="Engram-based composition (D.14)")
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--compose-pairs", type=str,
                    default="go:north,come:south,stop:west,look:east")
    p.add_argument("--encoding-steps", type=int, default=200,
                    help="Steps to record co-firing (100ms default)")
    p.add_argument("--drive-pA", type=float, default=200.0,
                    help="Drive on lang_input during encoding")
    p.add_argument("--top-k", type=int, default=100,
                    help="Engram tag size (top-K firing neurons)")
    p.add_argument("--recall-stim-pA", type=float, default=1500.0)
    p.add_argument("--recall-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--save-bridge", type=str, default=None)
    p.add_argument("--motor-teacher-pA", type=float, default=0.0,
                    help="Optional teacher current on motor pool during "
                    "encoding (analogous to Phase 1 teacher_pA). Ensures "
                    "engram tag includes motor neurons. Default 0=off.")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    pairs = [tuple(s.strip().split(":")) for s in args.compose_pairs.split(",")]

    print(f"=== compose_engram_demo (seed={args.seed}) ===")
    print(f"  Bridge: {args.load_bridge}")
    print(f"  Pairs: {pairs}")
    print(f"  Encoding steps: {args.encoding_steps}, top_k: {args.top_k}")
    print()

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

    # Region filter: only tag neurons in verb_pool + motor regions
    # so the engram is the compositional ensemble.
    region_filter_full = (
        [f"verb_pool_{v}" for v in ["GO", "COME", "STOP", "LOOK"]]
        + [f"motor_{a}" for a in ["N", "E", "S", "W"]]
    )

    # ENCODING phase: tag each compose pair
    print("[ENCODING] Tagging compose pairs as engrams...")
    t0 = time.time()
    encoding_stats = []
    tag_names = []
    for verb, motor in pairs:
        tag_name = f"{verb}_{motor}"
        tag_names.append(tag_name)
        stats = encode_compose_pair(
            bridge, verb, motor, tag_name,
            encoding_steps=args.encoding_steps,
            drive_pA=args.drive_pA,
            sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
            region_filter=region_filter_full,
            top_k=args.top_k,
            motor_teacher_pA=args.motor_teacher_pA,
            verbose=True,
        )
        encoding_stats.append(stats)
    print(f"[ENCODING] complete ({time.time() - t0:.0f}s)")

    # Save bridge (with tags) if requested
    if args.save_bridge:
        print(f"\n[SAVE] {args.save_bridge}")
        bridge.save_checkpoint(args.save_bridge)

    # RECALL phase: stimulate each tag, measure motor firing
    print()
    print("[RECALL] Stimulating each engram tag, measuring motor firing")
    print(f"  {'tag':18s} {'motor_N':10s} {'motor_E':10s} {'motor_S':10s} {'motor_W':10s}")
    firing = {}
    for tag_name, (verb, motor) in zip(tag_names, pairs):
        rates = recall_compose_tag(
            bridge, tag_name,
            drive_pA=args.recall_stim_pA,
            recall_steps=args.recall_steps,
        )
        firing[tag_name] = {"verb": verb, "motor": motor, "rates": rates}
        print(f"  {tag_name:18s} {rates['motor_N']:.3f}     "
              f"{rates['motor_E']:.3f}     {rates['motor_S']:.3f}     "
              f"{rates['motor_W']:.3f}")

    # PASS test: TRUE-mapped motor fires more than off-target motors
    print()
    n_pass = 0
    for tag_name in tag_names:
        d = firing[tag_name]
        target_motor_pool = _WORD_TO_POOL[d["motor"]]
        target_rate = d["rates"][target_motor_pool]
        off = max(v for k, v in d["rates"].items() if k != target_motor_pool)
        passed = target_rate > off
        if passed:
            n_pass += 1
        marker = "PASS" if passed else "FAIL"
        print(f"  {tag_name:18s} target={target_rate:.3f} off={off:.3f} "
              f"ratio={target_rate/max(off, 0.001):.2f}x [{marker}]")
    print()
    print(f"[VERDICT] {n_pass}/{len(pairs)} engram tags drive TRUE motor pool")

    # ANTI-CHEAT: 24 permutations
    print()
    print("[ANTI-CHEAT] 24 permutations")
    verb_words = [verb for verb, _ in pairs]
    motor_words = [motor for _, motor in pairs]
    perm_results = []
    for motor_perm in itertools.permutations(motor_words):
        mapping = list(zip(verb_words, motor_perm))
        passes = 0
        for verb, motor in mapping:
            tag = f"{verb}_{dict(pairs)[verb]}"  # the trained tag for this verb
            d = firing[tag]
            tgt = _WORD_TO_POOL[motor]
            target_rate = d["rates"][tgt]
            off = max(v for k, v in d["rates"].items() if k != tgt)
            if target_rate > off:
                passes += 1
        true_dict = dict(pairs)
        is_true = all(motor == true_dict[verb] for verb, motor in mapping)
        perm_results.append({"mapping": mapping, "n_pass": passes,
                              "is_true": is_true})
    perm_results.sort(key=lambda r: -r["n_pass"])
    true_rank = next(i for i, r in enumerate(perm_results, start=1)
                      if r["is_true"])
    true_pass = next(r["n_pass"] for r in perm_results if r["is_true"])

    print(f"  Top 5 permutations:")
    for rank, r in enumerate(perm_results[:5], start=1):
        m = ", ".join(f"{v}->{m}" for v, m in r["mapping"])
        tag = "** TRUE **" if r["is_true"] else ""
        print(f"    {rank} {m:50s} {r['n_pass']}/4 {tag}")
    print()
    print(f"  TRUE rank: {true_rank}/24 (chance=12.5/24)")
    print(f"  TRUE pass: {true_pass}/4")
    print(f"  Best perm pass: {perm_results[0]['n_pass']}/4")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "bridge": args.load_bridge,
                "seed": args.seed,
                "encoding_steps": args.encoding_steps,
                "top_k": args.top_k,
                "recall_stim_pA": args.recall_stim_pA,
                "encoding_stats": encoding_stats,
                "firing": firing,
                "n_pass": n_pass,
                "true_rank": true_rank,
                "true_pass": true_pass,
                "best_perm_pass": perm_results[0]["n_pass"],
            }, f, indent=2, default=str)
        print(f"\n[OUT] wrote {args.out}")


if __name__ == "__main__":
    main()
