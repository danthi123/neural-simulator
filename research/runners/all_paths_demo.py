"""All-paths integration demo: paths 1 + 2 + 3 + existing multi-bridge.

Showcases:
- Path 1 (G.20 distributed encoding): not yet wired into the chat REPL
  (separate single-pool prototype; runs via concept_pool_demo_shared.py)
- Path 2 (Bozic 2010 morpheme tokenizer): --tokenize flag
- Path 3 (Patterson 2007 hub-and-spoke hierarchy): is_a/descendants
- Multi-bridge 60-word + 11 conversational features (tonight's work)

The result: a chat REPL that handles dogs/dog/dogs ate apples + knows
that dogs are mammals + composes naturally.

Usage:
  python research/runners/all_paths_demo.py --seed 42
  python research/runners/all_paths_demo.py --seed 42 --friendly
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "research/findings/raw/g11_bg/concept_pool_demo"


DEMO_SCRIPT = ",".join([
    # --- Inventory ---
    "vocab",
    # --- Path 2: morphological decomposition ---
    "remember the dogs ate apples",       # tokenize -> dog eat apple
    "remember the cats are sleeping",     # cat sleep (PLURAL+ing stripped)
    "remember the trees are bigger than houses",  # tree big_er house (er stripped)
    # --- Path 3: hierarchy queries ---
    "is a dog an animal?",
    "is a cat an animal?",
    "is a tree an animal?",                # NO (tree -> plant -> living_thing)
    "is an apple a food?",
    "what kind of thing is dog?",
    "what mammals do you know?",
    "what colors do you know?",
    # --- Combined: hierarchy + memory ---
    "remember the apple is red",
    "remember the river is cold",
    "is apple red?",
    "what is apple",                      # multitag retrieval
    # --- Sentence-level role queries (still work) ---
    "who ate apples?",                    # tokenized -> who eat apple
    "what did dogs eat?",
    # --- Tags inventory ---
    "tags",
])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bridges-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--friendly", action="store_true",
                    help="Natural-language output mode")
    args = p.parse_args()

    bridges_dir = Path(args.bridges_dir)
    bridge_paths = [
        bridges_dir / f"seed{args.seed}_v16.simstate.h5",
        bridges_dir / f"seed{args.seed}_set2.simstate.h5",
        bridges_dir / f"seed{args.seed}_set3.simstate.h5",
        bridges_dir / f"seed{args.seed}_set4.simstate.h5",
        bridges_dir / f"seed{args.seed}_set5.simstate.h5",
    ]
    set_names = ["set1", "set2", "set3", "set4", "set5"]

    missing = [(s, p) for s, p in zip(set_names, bridge_paths) if not p.exists()]
    if missing:
        print("ERROR: missing bridges:", flush=True)
        for s, p in missing:
            print(f"  {s}: {p}", flush=True)
        sys.exit(1)

    print(f"=== All-paths demo (paths 1+2+3 + 60-word multi-bridge) ===",
          flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Demo: {len(DEMO_SCRIPT.split(','))} commands\n", flush=True)

    cmd = [
        sys.executable, "-m", "research.runners.multibridge_chat",
        "--bridges", *[str(p) for p in bridge_paths],
        "--names", *set_names,
        "--vocab-sets", *set_names,
        "--seed", str(args.seed),
        "--n-lang-input", "2048",
        "--n-per-pool", "200",
        "--n-fs-per-pool", "24",
        "--n-words-for-orthogonal", "16",
        "--encoding-steps", "500",
        "--sparsity", "0.05",
        "--balanced-teacher-pA", "500.0",
        "--top-k", "100",
        "--drive-steps", "100",
        "--scripted", DEMO_SCRIPT,
        "--tokenize",   # PATH 2
    ]
    if args.friendly:
        cmd.append("--friendly")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
