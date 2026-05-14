"""Compose-training runner for v16 architecture.

Loads a v16-trained bridge (Phase 1 done, direct verb_pool->motor
pathways at weight 0). Runs compose-training events where each
(verb_word, motor_word) pair is driven with a brief temporal offset
so verb_pool fires BEFORE motor pool (LTP-favorable timing). After
training, the v16 direct pathways have non-zero learned weights.

Tests whether composition emerges:
- Drive verb_word alone -> does the associated motor pool also fire?
- Compare to v14/v16 baseline (verb alone produces no motor bias).

Usage:
    python -m research.runners.concept_compose_train \\
        --load-bridge research/findings/raw/g11_bg/concept_pool_demo/seed42_v16.simstate.h5 \\
        --seed 42 \\
        --compose-pairs "go:north,come:south" \\
        --n-events-per-pair 100 \\
        --save-bridge research/findings/raw/g11_bg/concept_pool_demo/seed42_v16_composed.simstate.h5 \\
        --out research/findings/raw/g11_bg/concept_pool_demo/seed42_v16_compose.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import research.runners.concept_pool_demo as cpd
from sim.text_embeddings import orthogonal_drive_pattern, vocab_to_drive_pattern


# v14 training-order word index (must match concept_pool_demo)
_ALL_WORDS = [
    "north", "east", "south", "west",
    "apple", "river", "dog", "cat",
    "go", "come", "stop", "look",
    "big", "small", "hot", "cold",
]
_WORD_TO_IDX = {w: i for i, w in enumerate(_ALL_WORDS)}
_WORD_TO_POOL = {
    "north": "motor_N", "east": "motor_E", "south": "motor_S", "west": "motor_W",
    "apple": "noun_pool_APPLE", "river": "noun_pool_RIVER",
    "dog": "noun_pool_DOG", "cat": "noun_pool_CAT",
    "go": "verb_pool_GO", "come": "verb_pool_COME",
    "stop": "verb_pool_STOP", "look": "verb_pool_LOOK",
    "big": "adjective_pool_BIG", "small": "adjective_pool_SMALL",
    "hot": "adjective_pool_HOT", "cold": "adjective_pool_COLD",
}


def freeze_phase1_gates(bridge, verbose: bool = True) -> List[str]:
    """Freeze all Phase 1 plasticity gates so compose training doesn't
    disturb v14's reciprocal binding. Returns list of gates frozen.
    """
    phase1_gates = [
        "language_input_to_motor",
        "language_input_to_verb_pool",
        "language_input_to_noun_pool",
        "language_input_to_adjective_pool",
        "motor_to_language_output",
        "verb_pool_to_language_output",
        "noun_pool_to_language_output",
        "adjective_pool_to_language_output",
        "motor_FS_to_motor",
        "verb_pool_FS_to_verb_pool",
        "noun_pool_FS_to_noun_pool",
        "adjective_pool_FS_to_adjective_pool",
    ]
    frozen = []
    for g in phase1_gates:
        try:
            bridge.set_plasticity_gate(g, 0.0)
            frozen.append(g)
        except Exception:
            pass
    if verbose:
        print(f"[FREEZE] Phase 1 gates frozen: {len(frozen)}/{len(phase1_gates)}")
    return frozen


def train_compose_pair(bridge, verb_word: str, motor_word: str,
                        n_events: int = 100,
                        verb_only_steps: int = 20,   # 10ms
                        cofire_steps: int = 80,      # 40ms
                        reset_steps: int = 50,       # 25ms
                        drive_pA: float = 200.0,
                        sparsity: float = 0.05,
                        n_lang_input: int = 2048,
                        n_words_for_orthogonal: int = 16,
                        orthogonal_codes: bool = True,
                        motor_teacher_pA: float = 0.0,
                        verbose: bool = False):
    """Train one (verb, motor) compose pair.

    Temporal protocol per event:
      1. verb_only_steps: drive lang_input(verb) only.
         Lets verb_pool fire first (pre-before-post for LTP at
         verb_pool -> motor when motor fires later).
      2. cofire_steps: drive lang_input(verb) + lang_input(motor).
         Both pools active. STDP at verb_pool -> motor sees
         pre-firing already in progress, post-firing starts now.
         LTP grows the v16 direct pathway weight.
      3. reset_steps: free run, decay NMDA + eligibility.
    """
    from sim.backend import get_backend
    cp, _ = get_backend()

    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))

    # Generate drive patterns
    if orthogonal_codes:
        verb_drive = orthogonal_drive_pattern(
            cue_idx=_WORD_TO_IDX[verb_word],
            n_cues=n_words_for_orthogonal,
            n_neurons=n_lang_input,
            drive_max_pA=drive_pA, sparsity=sparsity,
        )
        motor_drive = orthogonal_drive_pattern(
            cue_idx=_WORD_TO_IDX[motor_word],
            n_cues=n_words_for_orthogonal,
            n_neurons=n_lang_input,
            drive_max_pA=drive_pA, sparsity=sparsity,
        )
    else:
        verb_drive = vocab_to_drive_pattern(
            verb_word, n_neurons=n_lang_input,
            drive_max_pA=drive_pA, sparsity=sparsity,
        )
        motor_drive = vocab_to_drive_pattern(
            motor_word, n_neurons=n_lang_input,
            drive_max_pA=drive_pA, sparsity=sparsity,
        )
    verb_gpu = cp.asarray(verb_drive, dtype=cp.float32)
    motor_gpu = cp.asarray(motor_drive, dtype=cp.float32)
    both_gpu = verb_gpu + motor_gpu  # additive co-fire drive
    zero_gpu = cp.zeros_like(verb_gpu)

    lang_arr_gpu = cp.asarray(lang_input_idx, dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    # Optional: motor teacher current during cofire phase.
    # Analogous to Phase 1's teacher_pA on target pool. When non-zero,
    # forces motor pool to fire strongly during co-fire, giving cleaner
    # STDP signal at verb_pool -> motor (post-firing is teacher-driven,
    # not just lang_input-driven via v14 weights).
    use_teacher = motor_teacher_pA > 0.0
    motor_pool_name = _WORD_TO_POOL[motor_word]  # e.g., "motor_N"
    if use_teacher:
        motor_pool_idx = list(rm.indices(motor_pool_name))
        motor_pool_arr_gpu = cp.asarray(motor_pool_idx, dtype=cp.int64)

    t0 = time.time()
    for evt in range(n_events):
        # Phase 1: verb only (10ms) — verb_pool fires first
        for _ in range(verb_only_steps):
            ext = cp.zeros(n_total, dtype=cp.float32)
            ext[lang_arr_gpu] = verb_gpu
            bridge.cp_external_input_current[:] = ext
            bridge._run_one_simulation_step()
        # Phase 2: co-fire (40ms) — motor fires while verb still active.
        # With motor_teacher_pA, ALSO inject teacher current on motor pool
        # for cleaner post-firing during STDP eligibility window.
        for _ in range(cofire_steps):
            ext = cp.zeros(n_total, dtype=cp.float32)
            ext[lang_arr_gpu] = both_gpu
            if use_teacher:
                ext[motor_pool_arr_gpu] = motor_teacher_pA
            bridge.cp_external_input_current[:] = ext
            bridge._run_one_simulation_step()
        # Phase 3: reset (25ms) — decay NMDA + eligibility
        for _ in range(reset_steps):
            bridge.cp_external_input_current[:] = 0.0
            bridge._run_one_simulation_step()

        if verbose and (evt + 1) % 25 == 0:
            print(f"  [{verb_word}+{motor_word}] event {evt+1}/{n_events} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    return {"verb_word": verb_word, "motor_word": motor_word,
            "n_events": n_events, "wall_s": time.time() - t0}


def measure_compose_inference(bridge, verb_word: str, expected_motor_word: str,
                                n_lang_input: int = 2048,
                                stim_steps: int = 100,
                                sparsity: float = 0.05,
                                n_words_for_orthogonal: int = 16,
                                orthogonal_codes: bool = True) -> Dict:
    """Drive verb_word alone, measure firing rates across all motor pools.

    If composition trained correctly, the expected motor pool should
    fire more than the other motor pools (v16 direct pathway provides
    the bias).
    """
    motor_pools = ["motor_N", "motor_E", "motor_S", "motor_W"]
    rates = cpd.measure_pool_firing(
        bridge, verb_word, motor_pools,
        n_lang_input=n_lang_input,
        n_words_for_orthogonal=n_words_for_orthogonal,
        sparsity=sparsity,
        orthogonal_codes=orthogonal_codes,
        word_to_idx=_WORD_TO_IDX,
    )
    expected_pool = _WORD_TO_POOL[expected_motor_word]
    target_rate = rates[expected_pool]
    off_rates = {p: r for p, r in rates.items() if p != expected_pool}
    max_off = max(off_rates.values())
    max_off_pool = max(off_rates, key=off_rates.get)
    passed = target_rate > max_off
    return {
        "verb_word": verb_word, "expected_motor": expected_motor_word,
        "expected_pool": expected_pool,
        "target_rate": float(target_rate),
        "all_rates": {k: float(v) for k, v in rates.items()},
        "max_off": float(max_off),
        "max_off_pool": max_off_pool,
        "passed": bool(passed),
        "ratio": float(target_rate / max(max_off, 0.001)),
    }


def main():
    p = argparse.ArgumentParser(description="v16 compose-training runner")
    p.add_argument("--load-bridge", required=True,
                    help="Path to v16 bridge (after Phase 1 training)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--compose-pairs", type=str,
                    default="go:north,come:south,stop:west,look:east",
                    help="Comma-separated 'verb:motor' pairs to train.")
    p.add_argument("--n-events-per-pair", type=int, default=100)
    p.add_argument("--orthogonal-codes", action="store_true", default=True)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--motor-teacher-pA", type=float, default=0.0,
                    help="Optional teacher current on motor pool during "
                    "co-fire phase (analogous to Phase 1 teacher_pA). "
                    "0.0=off (current default behavior). 1500.0 matches "
                    "Phase 1 teacher_pA. Strengthens STDP signal at "
                    "verb_pool -> motor by forcing strong motor post-firing.")
    p.add_argument("--save-bridge", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    # Parse compose pairs
    compose_pairs: List[Tuple[str, str]] = []
    for pair_str in args.compose_pairs.split(","):
        verb, motor = pair_str.strip().split(":")
        if verb not in _WORD_TO_IDX or motor not in _WORD_TO_IDX:
            print(f"ERROR: unknown word in pair {verb}:{motor}", file=sys.stderr)
            sys.exit(1)
        compose_pairs.append((verb, motor))

    print(f"=== concept_compose_train (seed={args.seed}) ===", flush=True)
    print(f"  Load bridge: {args.load_bridge}", flush=True)
    print(f"  Compose pairs: {compose_pairs}", flush=True)
    print(f"  Events per pair: {args.n_events_per_pair}", flush=True)

    # Build v16 bridge (must match the loaded bridge's architecture)
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
    print(f"\n[LOAD] {args.load_bridge}", flush=True)
    bridge.load_checkpoint(args.load_bridge)

    # Freeze Phase 1 gates
    frozen = freeze_phase1_gates(bridge, verbose=True)

    # Ensure compose gate is open
    try:
        bridge.set_plasticity_gate("verb_to_motor_direct", 1.0)
        print(f"[GATE] verb_to_motor_direct = 1.0 (open for compose STDP)",
              flush=True)
    except Exception as e:
        print(f"ERROR: cannot open verb_to_motor_direct gate: {e}",
              file=sys.stderr)
        print("       (Bridge was not built with --enable-direct-verb-to-motor?)",
              file=sys.stderr)
        sys.exit(1)

    # Train compose pairs
    print(f"\n[COMPOSE-TRAIN] {len(compose_pairs)} pairs x "
          f"{args.n_events_per_pair} events", flush=True)
    t_train = time.time()
    train_results = []
    for verb_word, motor_word in compose_pairs:
        t_pair = time.time()
        r = train_compose_pair(
            bridge, verb_word, motor_word,
            n_events=args.n_events_per_pair,
            n_lang_input=args.n_lang_input,
            sparsity=args.sparsity,
            orthogonal_codes=args.orthogonal_codes,
            motor_teacher_pA=args.motor_teacher_pA,
            verbose=False,
        )
        train_results.append(r)
        print(f"  trained {verb_word}+{motor_word} "
              f"({time.time() - t_pair:.0f}s)", flush=True)
    print(f"[COMPOSE-TRAIN] total {time.time() - t_train:.0f}s",
          flush=True)

    # Save trained bridge
    if args.save_bridge:
        print(f"\n[SAVE] {args.save_bridge}", flush=True)
        bridge.save_checkpoint(args.save_bridge)

    # Test: drive verb alone, check if expected motor pool fires
    print(f"\n[TEST] Composition inference (drive verb alone)", flush=True)
    inference_results = []
    n_pass = 0
    for verb_word, expected_motor_word in compose_pairs:
        r = measure_compose_inference(
            bridge, verb_word, expected_motor_word,
            n_lang_input=args.n_lang_input,
            sparsity=args.sparsity,
            orthogonal_codes=args.orthogonal_codes,
        )
        inference_results.append(r)
        marker = "PASS" if r["passed"] else "FAIL"
        if r["passed"]:
            n_pass += 1
        print(f"  '{verb_word}' alone -> {r['expected_pool']:10s} "
              f"target={r['target_rate']:.2f} off={r['max_off']:.2f}/"
              f"{r['max_off_pool']:10s} ratio={r['ratio']:.2f}x [{marker}]",
              flush=True)

    print(f"\n[VERDICT] {n_pass}/{len(compose_pairs)} compose pairs "
          f"produce target motor pool firing", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed, "load_bridge": args.load_bridge,
                "compose_pairs": [list(p) for p in compose_pairs],
                "n_events_per_pair": args.n_events_per_pair,
                "train_results": train_results,
                "inference_results": inference_results,
                "n_pass": n_pass,
                "n_total": len(compose_pairs),
                "frozen_gates": frozen,
            }, f, indent=2)
        print(f"[OUT] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
