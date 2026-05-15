"""End-to-end 160-concept G.20 multi-bridge demo.

Loads 5 G.20 shared-pool bridges (160 unique concepts total) and runs
a scripted conversation showcasing:
- Intra-bridge encoding (within a single vocab category)
- Cross-bridge encoding (apple is big — apple in nouns, big in adjectives)
- Single-word multitag retrieval
- Yes/no queries
- Path 3 hierarchy queries (is a dog an animal?)
- Path 2 morpheme tokenization (dogs ate apples)

This is the production demo for the path-1 G.20 BREAKTHROUGH +
path-2 + path-3 integrated stack.

Requires all 5 bridges trained:
  research/findings/raw/g11_bg/g20_bridges/bridge{A,B,C,D,E}*.simstate.h5
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_DIR = REPO_ROOT / "research/findings/raw/g11_bg/g20_bridges"
VOCAB_DIR = REPO_ROOT / "research/findings/raw/g11_bg"


DEMO_SCRIPT = ",".join([
    # --- Inventory ---
    "vocab",
    "tags",
    # --- Path 2 tokenization (dogs -> dog, ate -> eat) ---
    "remember the dogs ate apples",   # tokenized
    "remember the cats are happy",    # PLURAL+ing stripped
    # --- Intra-bridge encoding ---
    "remember apple is red",          # both in their bridges
    "remember dog is small",
    # --- Cross-bridge encoding (the key capability) ---
    "remember apple is big",          # apple in bridgeA, big in bridgeC
    "remember dog is fast",           # dog in bridgeA, fast in bridgeC
    "remember run is fast",           # run in bridgeB, fast in bridgeC
    "remember north is here",         # north in bridgeD, here also bridgeD
    # --- List what's stored ---
    "tags",
    # --- Single-word multitag retrieval ---
    "what is apple",
    "what is dog",
    "what is fast",
    # --- Yes/no questions ---
    "is apple red?",
    "is apple big?",
    "is dog small?",
    "is dog hot?",                    # untrained
    # --- Path 3 hierarchy ---
    "is a dog an animal?",
    "is a cat an animal?",
    "is an apple a food?",
    "what mammals do you know?",
    "what colors do you know?",
    # --- Path 2 morpheme + multitag combined ---
    "what is dogs",                   # tokenized to 'dog'
])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bridge-dir", type=str, default=str(BRIDGE_DIR))
    p.add_argument("--friendly", action="store_true",
                    help="Natural-language output mode")
    p.add_argument("--script", type=str, default=None,
                    help="Override demo script")
    args = p.parse_args()

    bridge_dir = Path(args.bridge_dir)
    bridge_names = [
        "bridgeA_nouns", "bridgeB_verbs", "bridgeC_adj",
        "bridgeD_spatial", "bridgeE_functional",
    ]
    bridge_paths = [bridge_dir / f"{n}.simstate.h5" for n in bridge_names]
    vocab_paths = [VOCAB_DIR / f"g20_{n}_vocab.txt" for n in bridge_names]

    # Verify all bridges + vocab files exist
    missing = []
    for n, bp, vp in zip(bridge_names, bridge_paths, vocab_paths):
        if not bp.exists():
            missing.append((n, "bridge", str(bp)))
        if not vp.exists():
            missing.append((n, "vocab", str(vp)))
    if missing:
        print("ERROR: missing files:", flush=True)
        for n, kind, path in missing:
            print(f"  {n} {kind}: {path}", flush=True)
        print("\nRun the 5-bridge training chain first "
              "(research/runners/g20_train_5bridges_chain.ps1).",
              flush=True)
        sys.exit(1)

    scripted = args.script or DEMO_SCRIPT

    print(f"=== G.20 160-concept multi-bridge demo ===", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Bridges: {bridge_names}", flush=True)
    print(f"Total vocab: 160 unique concepts across 5 G.20 bridges",
          flush=True)
    print(f"Demo script: {len(scripted.split(','))} commands\n",
          flush=True)

    cmd = [
        sys.executable, "-m", "research.runners.g20_multibridge",
        "--bridges", *[str(p) for p in bridge_paths],
        "--vocab-files", *[str(p) for p in vocab_paths],
        "--names", *bridge_names,
        "--seed", str(args.seed),
        "--scripted", scripted,
    ]
    if args.friendly:
        cmd.append("--friendly")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
