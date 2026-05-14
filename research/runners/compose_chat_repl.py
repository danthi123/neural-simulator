"""Interactive chat REPL for engram-based composition.

Usage:
    # Interactive (requires stdin):
    python -m research.runners.compose_chat_repl \\
        --load-bridge research/findings/raw/g11_bg/concept_pool_demo/seed42_v16.simstate.h5 \\
        --seed 42

    # Scripted (good for CI / demo):
    python -m research.runners.compose_chat_repl \\
        --load-bridge ...  --seed 42 \\
        --scripted "go north,come south,stop west,look east,go south"

User types (verb, motor) pairs separated by space. System:
1. Drives lang_input(verb + motor)
2. Cosine-matches resulting firing pattern to stored engrams
3. Stimulates the best-matched engram
4. Reports motor pool firing → user-facing direction
"""
from __future__ import annotations
import argparse
import sys
import time
import numpy as np

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from research.runners.compose_engram_retrieval import (
    encode_with_pattern, measure_firing_pattern_during_drive, cosine_sim,
)
from research.runners.compose_engram_demo import recall_compose_tag


REGION_FILTER = (
    [f"verb_pool_{v}" for v in ["GO", "COME", "STOP", "LOOK"]]
    + [f"noun_pool_{n}" for n in ["APPLE", "RIVER", "DOG", "CAT"]]
    + [f"adjective_pool_{a}" for a in ["BIG", "SMALL", "HOT", "COLD"]]
    + [f"motor_{a}" for a in ["N", "E", "S", "W"]]
)

# Motor pool name → direction word
_MOTOR_POOL_TO_WORD = {
    "motor_N": "north", "motor_E": "east",
    "motor_S": "south", "motor_W": "west",
}


def banner():
    print("=" * 60)
    print("COMPOSE-ENGRAM CHAT REPL")
    print("Type (verb motor) pairs, e.g.: 'go north', 'come south'")
    print("Single words also accepted: 'go' or 'north'")
    print("Commands: 'tags' (list engrams), 'quit'")
    print("=" * 60, flush=True)


def parse_input(line: str):
    """Parse user input into (verb, motor) tuple or single word."""
    tokens = [t.strip().lower() for t in line.split() if t.strip()]
    if not tokens:
        return None, None
    if len(tokens) == 1:
        # Single word — pick a compatible second word
        w = tokens[0]
        if w in _WORD_TO_IDX:
            return w, None
        return None, None
    elif len(tokens) >= 2:
        v, m = tokens[0], tokens[1]
        if v in _WORD_TO_IDX and m in _WORD_TO_IDX:
            return v, m
        return None, None
    return None, None


def handle_query(bridge, rf_mask, encoded, verb, motor, args):
    """Drive lang_input(verb + motor), match engram, stimulate, output motor."""
    t0 = time.time()

    # For single-word queries, fall back to a neutral second word
    if motor is None:
        # Use the verb itself as motor cue too (or skip and just drive verb)
        # For now: drive only the verb part — measurement still works
        query_motor = "north"  # placeholder, will only drive verb
        query_pattern = measure_firing_pattern_during_drive(
            bridge, verb, query_motor, rf_mask,
            drive_steps=args.retrieval_steps,
            sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
        )
    else:
        query_pattern = measure_firing_pattern_during_drive(
            bridge, verb, motor, rf_mask,
            drive_steps=args.retrieval_steps,
            sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
        )

    # Match to stored engrams
    scores = {tag: cosine_sim(query_pattern, d["pattern"])
               for tag, d in encoded.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked:
        return {"error": "no engrams stored"}

    matched_tag = ranked[0][0]
    match_score = ranked[0][1]
    second_tag = ranked[1][0] if len(ranked) > 1 else None
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # Stimulate matched tag, measure motor
    motor_rates = recall_compose_tag(
        bridge, matched_tag,
        drive_pA=args.recall_stim_pA,
        recall_steps=args.recall_steps,
    )
    recalled_motor_pool = max(motor_rates, key=motor_rates.get)
    recalled_motor_word = _MOTOR_POOL_TO_WORD.get(recalled_motor_pool, "?")

    elapsed = time.time() - t0
    return {
        "matched_tag": matched_tag, "match_score": match_score,
        "second_tag": second_tag, "second_score": second_score,
        "motor_rates": {p: float(r) for p, r in motor_rates.items()},
        "recalled_motor_pool": recalled_motor_pool,
        "recalled_motor_word": recalled_motor_word,
        "elapsed_s": elapsed,
    }


def main():
    p = argparse.ArgumentParser(description="Compose-engram chat REPL")
    p.add_argument("--load-bridge", required=True,
                    help="v16 bridge checkpoint (post Phase 1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--compose-pairs", type=str,
                    default="go:north,come:south,stop:west,look:east",
                    help="Pre-encoded compose pairs (V:M,V:M,...)")
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--retrieval-steps", type=int, default=200)
    p.add_argument("--recall-stim-pA", type=float, default=1500.0)
    p.add_argument("--recall-steps", type=int, default=100)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--motor-teacher-pA", type=float, default=1500.0,
                    help="Motor teacher current during encoding (default 1500). "
                    "Ensures engram includes enough motor neurons for clean "
                    "recall. Set to 0 to disable.")
    p.add_argument("--scripted", type=str, default=None,
                    help="Comma-separated test inputs (skip interactive)")
    args = p.parse_args()

    pairs = [tuple(s.strip().split(":")) for s in args.compose_pairs.split(",")]

    print(f"Loading bridge: {args.load_bridge}", flush=True)
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

    # Freeze plasticity gates so chat doesn't drift
    all_gates = [
        "language_input_to_motor", "language_input_to_verb_pool",
        "language_input_to_noun_pool", "language_input_to_adjective_pool",
        "motor_to_language_output", "verb_pool_to_language_output",
        "noun_pool_to_language_output", "adjective_pool_to_language_output",
        "motor_FS_to_motor", "verb_pool_FS_to_verb_pool",
        "noun_pool_FS_to_noun_pool", "adjective_pool_FS_to_adjective_pool",
        "verb_to_motor_direct",
    ]
    for g in all_gates:
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass

    rm = bridge.region_manager
    n_total = bridge.cp_external_input_current.shape[0]
    rf_mask = np.zeros(n_total, dtype=bool)
    for rname in REGION_FILTER:
        try:
            rf_mask[list(rm.indices(rname))] = True
        except Exception:
            pass

    # PRE-ENCODE engrams
    print(f"\nPre-encoding {len(pairs)} compose engrams...", flush=True)
    encoded = {}
    for verb, motor in pairs:
        tag_name = f"{verb}_{motor}"
        _, pattern = encode_with_pattern(
            bridge, verb, motor, tag_name,
            encoding_steps=args.encoding_steps,
            sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
            region_filter=REGION_FILTER,
            top_k=args.top_k,
            motor_teacher_pA=args.motor_teacher_pA,
            verbose=False,
        )
        encoded[tag_name] = {"verb": verb, "motor": motor, "pattern": pattern}
        print(f"  [{tag_name}] encoded", flush=True)

    banner()

    # SCRIPTED or interactive
    inputs = []
    if args.scripted:
        inputs = [s.strip() for s in args.scripted.split(",") if s.strip()]
        print(f"Scripted mode: {len(inputs)} inputs", flush=True)
        for inp in inputs:
            print(f"\n> {inp}", flush=True)
            verb, motor = parse_input(inp)
            if verb is None:
                print(f"  [unparseable: '{inp}']", flush=True)
                continue
            r = handle_query(bridge, rf_mask, encoded, verb, motor, args)
            if "error" in r:
                print(f"  [error: {r['error']}]", flush=True)
                continue
            cue = verb if motor is None else f"{verb} {motor}"
            print(f"  cue: '{cue}'", flush=True)
            print(f"  matched engram: '{r['matched_tag']}' "
                  f"(score {r['match_score']:.3f}; 2nd: '{r['second_tag']}' "
                  f"@ {r['second_score']:.3f})", flush=True)
            print(f"  motor rates: " + " ".join(
                f"{k.replace('motor_', '')}={v:.2f}"
                for k, v in r['motor_rates'].items()), flush=True)
            print(f"  -> action: {r['recalled_motor_word'].upper()} "
                  f"({r['recalled_motor_pool']}) [{r['elapsed_s']:.1f}s]",
                  flush=True)
    else:
        # Interactive
        print("Ready. Type a (verb motor) pair, or 'quit'.", flush=True)
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line in ("quit", "exit", ""):
                break
            if line == "tags":
                for tag in encoded:
                    print(f"  {tag}", flush=True)
                continue
            verb, motor = parse_input(line)
            if verb is None:
                print(f"  [unparseable]", flush=True)
                continue
            r = handle_query(bridge, rf_mask, encoded, verb, motor, args)
            if "error" in r:
                print(f"  [error: {r['error']}]", flush=True)
                continue
            print(f"  matched: {r['matched_tag']} ({r['match_score']:.3f})",
                  flush=True)
            print(f"  motor rates: " + " ".join(
                f"{k.replace('motor_', '')}={v:.2f}"
                for k, v in r['motor_rates'].items()), flush=True)
            print(f"  -> action: {r['recalled_motor_word'].upper()}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
