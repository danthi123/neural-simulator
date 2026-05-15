"""End-to-end demo of 60-word multi-bridge chat REPL.

Loads all 5 bridges (set1-5) and runs a scripted demo exercising:
- Intra-set pair encoding ('remember apple is big' -> set1)
- Cross-set pair encoding ('remember sun is hot' -> set1+set2)
- Intra-set sentence encoding (3-word, 4-word)
- Cross-set sentence encoding (e.g. 'apple eats sun' across set1+set2)
- Single-word multitag retrieval
- 'who X Y?' subject queries
- 'what did X Y?' object queries
- 'vocab' command listing all 60 words

This is the validation milestone for the 60-word multi-bridge target.
Runs only if all 5 bridges exist; otherwise reports which are missing.

Usage:
    python -m research.runners.multibridge_60word_demo --seed 42
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "research/findings/raw/g11_bg/concept_pool_demo"


SCRIPTED_DEMO = ",".join([
    # 1. Inventory: 60-word vocab and starting tags
    "vocab",
    "tags",

    # 2. Intra-set pair encoding (one per set to verify routing)
    "remember apple is big",        # set 1
    "remember tree is fast",        # set 2
    "remember house is tall",       # set 3
    "remember ball is happy",       # set 4
    "remember food is new",         # set 5

    # 3. CROSS-SET pair encoding (no single bridge has both)
    "remember sun is hot",          # sun in set2, hot in set1
    "remember dog is fast",         # dog in set1, fast in set2

    # 4. NEGATION
    "remember apple is not small",  # tag 'NOT_apple_small'

    # 5. CONJUNCTION
    "remember person is happy and ball is full",  # 2 tags created

    # 6. POSSESSIVE: stored as 'color_of_apple_red'
    "remember apple's color is red",
    "remember dog's color is small",  # using vocab; 'small' as color stand-in

    # 7. PRONOUNS (last subject = dog from above)
    "remember it is hot",           # 'it' -> dog, tag 'dog_hot'

    # 8. 3-word sentences (intra-set + cross-set)
    "remember dog ate apple",       # all set 1
    "remember bird ate tree",       # all set 2
    "remember person ate water",    # cross: set4 + set2 + set3

    # 9. List everything we've stored
    "tags",

    # 10. Single-word multitag retrieval
    "apple",
    "sun",
    "food",

    # 11. YES/NO questions
    "is apple big?",
    "is apple small?",              # should be NO (we have NOT_apple_small)
    "is tree fast?",

    # 12. ROLE queries (subject / object)
    "who ate apple?",               # template *_ate_apple
    "what did dog ate?",            # template dog_ate_*
    "what did person ate?",         # cross-set role query

    # 13. RELATIONAL queries
    "what is the color of apple?",  # template color_of_apple_*
    "what color is dog?",           # compact form

    # 14. ABOUT queries
    "about apple",
    "tell me about dog",

    # 15. FORGET
    "forget apple is big",
    "tags",                         # verify 'apple_big' removed
])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bridges-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--script", type=str, default=None,
                    help="Override scripted commands (comma-separated). "
                    "Default: built-in 60-word demo.")
    args = p.parse_args()

    bridges_dir = Path(args.bridges_dir)
    # Set 1 uses the v16 bridge name; sets 2-5 use seed42_set{n}.simstate.h5
    bridge_paths = [
        bridges_dir / f"seed{args.seed}_v16.simstate.h5",      # set 1
        bridges_dir / f"seed{args.seed}_set2.simstate.h5",
        bridges_dir / f"seed{args.seed}_set3.simstate.h5",
        bridges_dir / f"seed{args.seed}_set4.simstate.h5",
        bridges_dir / f"seed{args.seed}_set5.simstate.h5",
    ]
    set_names = ["set1", "set2", "set3", "set4", "set5"]

    # Verify all bridges exist
    missing = [(s, p) for s, p in zip(set_names, bridge_paths) if not p.exists()]
    if missing:
        print("ERROR: missing bridges:", flush=True)
        for s, p in missing:
            print(f"  {s}: {p}", flush=True)
        print("\nTrain missing sets first (chain_set45_runtime.ps1 auto-trains "
              "sets 3, 4, 5).")
        sys.exit(1)

    # Run multibridge_chat with all 5 bridges + the demo script
    scripted = args.script or SCRIPTED_DEMO

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
        "--scripted", scripted,
    ]

    print("=== 60-word multi-bridge chat REPL demo ===", flush=True)
    print(f"Bridges: {[s for s in set_names]}", flush=True)
    print(f"Total vocab: 60 unique concept words across 5 bridges", flush=True)
    print(f"Demo script: {len(scripted.split(','))} commands\n", flush=True)

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
