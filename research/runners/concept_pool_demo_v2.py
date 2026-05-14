"""Extended vocabulary concept-pool demo: 28 words across 4 kinds.

Adds 12 new words to v1:
- Nouns: tree, bird, sun, moon (4 -> 8)
- Verbs: walk, run, eat, sleep (4 -> 8)
- Adjectives: red, blue, fast, slow (4 -> 8)

Motor stays at 4 (north/east/south/west).

Total: 28 distinct concept words.

For orthogonal codes at 28 cues:
- n_lang_input=2048, stride=73, max sparsity 0.035
- Use sparsity=0.03 (61 neurons per word) for safety

Architecture cost:
- 8 noun pools + 8 verb pools + 8 adj pools + 4 motor = 28 concept pools
- 28 * 224 neurons (200 + 24 FS) = 6272 + lang_input 2048 + lang_output 2048
- Total ~10500 neurons (vs 7680 for v1)
- Bridge GPU memory should still fit (estimated ~3GB)
- Training: 28 words x 200 events = 5600 events x ~0.3s = ~28 min/seed
"""
from __future__ import annotations
import argparse
import sys
import os

# Reuse the v1 build/train machinery but with extended vocab
import research.runners.concept_pool_demo as cpd_v1

# Override the vocabularies BEFORE any usage. Module-level monkey-patch.
DIRECTION_VOCAB = {
    "north": "N", "east": "E", "south": "S", "west": "W",
}
NOUN_VOCAB = {
    "apple": "APPLE", "river": "RIVER", "dog": "DOG", "cat": "CAT",
    "tree": "TREE", "bird": "BIRD", "sun": "SUN", "moon": "MOON",
}
VERB_VOCAB = {
    "go": "GO", "come": "COME", "stop": "STOP", "look": "LOOK",
    "walk": "WALK", "run": "RUN", "eat": "EAT", "sleep": "SLEEP",
}
ADJECTIVE_VOCAB = {
    "big": "BIG", "small": "SMALL", "hot": "HOT", "cold": "COLD",
    "red": "RED", "blue": "BLUE", "fast": "FAST", "slow": "SLOW",
}

# Patch the v1 module so all its uses (build_concept_bridge, training,
# etc.) pick up the extended vocab.
cpd_v1.DIRECTION_VOCAB = DIRECTION_VOCAB
cpd_v1.NOUN_VOCAB = NOUN_VOCAB
cpd_v1.VERB_VOCAB = VERB_VOCAB
cpd_v1.ADJECTIVE_VOCAB = ADJECTIVE_VOCAB
cpd_v1.NOUN_NAMES = list(NOUN_VOCAB.values())
cpd_v1.VERB_NAMES = list(VERB_VOCAB.values())
cpd_v1.MOTOR_NAMES = ["N", "E", "S", "W"]
cpd_v1.ADJECTIVE_NAMES = list(ADJECTIVE_VOCAB.values())

# Total cues for orthogonal patterns: 4 + 8 + 8 + 8 = 28
N_TOTAL_WORDS = 28


def main():
    # Print expanded vocab info
    print(f"=== concept_pool_demo_v2 - extended vocabulary ===", flush=True)
    print(f"  Motor: {list(DIRECTION_VOCAB)}", flush=True)
    print(f"  Nouns: {list(NOUN_VOCAB)} (8 pools)", flush=True)
    print(f"  Verbs: {list(VERB_VOCAB)} (8 pools)", flush=True)
    print(f"  Adjectives: {list(ADJECTIVE_VOCAB)} (8 pools)", flush=True)
    print(f"  Total: {N_TOTAL_WORDS} words / 28 distinct output pools", flush=True)
    print()

    # Delegate to v1's main() with patched vocabs. The CLI is identical.
    cpd_v1.main()


if __name__ == "__main__":
    main()
