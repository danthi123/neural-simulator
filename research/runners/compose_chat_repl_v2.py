"""Chat REPL for v17 extended-vocab (28-word) bridges.

Same interactive loop as compose_chat_repl but with 28-word vocab
support and proper n_words_for_orthogonal=28.
"""
from __future__ import annotations
# Patch v17 vocab BEFORE anything else imports
import research.runners.compose_engram_demo_v2  # noqa: F401

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
    [f"verb_pool_{v}" for v in
     ["GO", "COME", "STOP", "LOOK", "WALK", "RUN", "EAT", "SLEEP"]]
    + [f"noun_pool_{n}" for n in
        ["APPLE", "RIVER", "DOG", "CAT", "TREE", "BIRD", "SUN", "MOON"]]
    + [f"adjective_pool_{a}" for a in
        ["BIG", "SMALL", "HOT", "COLD", "RED", "BLUE", "FAST", "SLOW"]]
    + [f"motor_{a}" for a in ["N", "E", "S", "W"]]
)

_MOTOR_POOL_TO_WORD = {
    "motor_N": "north", "motor_E": "east",
    "motor_S": "south", "motor_W": "west",
}


def banner():
    print("=" * 60)
    print("COMPOSE-ENGRAM CHAT REPL v2 (28-word vocab)")
    print("Verbs: go come stop look walk run eat sleep")
    print("Nouns: apple river dog cat tree bird sun moon")
    print("Adj:   big small hot cold red blue fast slow")
    print("Motor: north east south west")
    print("Type 'tags' to list trained engrams, 'quit' to exit")
    print("=" * 60, flush=True)


def parse_input(line: str):
    tokens = [t.strip().lower() for t in line.split() if t.strip()]
    if not tokens:
        return None, None
    if len(tokens) == 1 and tokens[0] in _WORD_TO_IDX:
        return tokens[0], None
    if len(tokens) >= 2 and tokens[0] in _WORD_TO_IDX and tokens[1] in _WORD_TO_IDX:
        return tokens[0], tokens[1]
    return None, None


def handle_query(bridge, rf_mask, encoded, verb, motor, args):
    t0 = time.time()
    if motor is None:
        motor = "north"  # placeholder; only drives verb
    query_pattern = measure_firing_pattern_during_drive(
        bridge, verb, motor, rf_mask,
        drive_steps=args.retrieval_steps,
        sparsity=args.sparsity, n_lang_input=args.n_lang_input,
        n_words_for_orthogonal=args.n_words_for_orthogonal,
    )
    scores = {tag: cosine_sim(query_pattern, d["pattern"])
               for tag, d in encoded.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked:
        return {"error": "no engrams stored"}
    matched_tag = ranked[0][0]
    match_score = ranked[0][1]
    motor_rates = recall_compose_tag(
        bridge, matched_tag,
        drive_pA=args.recall_stim_pA, recall_steps=args.recall_steps,
    )
    recalled_motor_pool = max(motor_rates, key=motor_rates.get)
    return {
        "matched_tag": matched_tag, "match_score": match_score,
        "motor_rates": {p: float(r) for p, r in motor_rates.items()},
        "recalled_motor_word": _MOTOR_POOL_TO_WORD.get(recalled_motor_pool, "?"),
        "elapsed_s": time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser(description="Compose-engram chat REPL v2 (28-word)")
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=4096)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--n-words-for-orthogonal", type=int, default=28)
    p.add_argument("--compose-pairs", type=str,
                    default="walk:north,run:east,eat:south,sleep:west,"
                            "tree:north,bird:east,sun:south,moon:west,"
                            "red:north,blue:east,fast:south,slow:west",
                    help="Pre-encoded compose pairs (12 v17 pairs)")
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--retrieval-steps", type=int, default=200)
    p.add_argument("--recall-stim-pA", type=float, default=1500.0)
    p.add_argument("--recall-steps", type=int, default=100)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.03)
    p.add_argument("--motor-teacher-pA", type=float, default=1500.0)
    p.add_argument("--scripted", type=str, default=None)
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

    # Freeze plasticity gates
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

    print(f"\nPre-encoding {len(pairs)} compose engrams (v17 vocab)...",
          flush=True)
    encoded = {}
    for verb, motor in pairs:
        tag_name = f"{verb}_{motor}"
        _, pattern = encode_with_pattern(
            bridge, verb, motor, tag_name,
            encoding_steps=args.encoding_steps,
            sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            region_filter=REGION_FILTER,
            top_k=args.top_k,
            motor_teacher_pA=args.motor_teacher_pA,
            verbose=False,
        )
        encoded[tag_name] = {"verb": verb, "motor": motor, "pattern": pattern}
        print(f"  [{tag_name}] encoded", flush=True)

    banner()

    inputs = []
    if args.scripted:
        inputs = [s.strip() for s in args.scripted.split(",") if s.strip()]
        for inp in inputs:
            print(f"\n> {inp}", flush=True)
            verb, motor = parse_input(inp)
            if verb is None:
                print(f"  [unparseable]", flush=True)
                continue
            r = handle_query(bridge, rf_mask, encoded, verb, motor, args)
            if "error" in r:
                print(f"  [error: {r['error']}]", flush=True)
                continue
            cue = verb if motor is None else f"{verb} {motor}"
            print(f"  cue: '{cue}'", flush=True)
            print(f"  matched: '{r['matched_tag']}' (score {r['match_score']:.3f})",
                  flush=True)
            print(f"  motor rates: " + " ".join(
                f"{k.replace('motor_', '')}={v:.2f}"
                for k, v in r['motor_rates'].items()), flush=True)
            print(f"  -> action: {r['recalled_motor_word'].upper()} "
                  f"[{r['elapsed_s']:.1f}s]", flush=True)
    else:
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
                print(f"  [unparseable; valid words: motor/noun/verb/adj]",
                      flush=True)
                continue
            r = handle_query(bridge, rf_mask, encoded, verb, motor, args)
            if "error" in r:
                print(f"  [error: {r['error']}]")
                continue
            print(f"  matched: {r['matched_tag']} ({r['match_score']:.3f})",
                  flush=True)
            print(f"  -> action: {r['recalled_motor_word'].upper()}",
                  flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
