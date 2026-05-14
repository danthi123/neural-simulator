"""multitag_eval for v17 (28-word extended-vocab) bridges.

Monkey-patches concept_pool_demo's vocab tables BEFORE delegating to
multitag_eval's main(). The bridge skeleton then includes all 28 pools,
matching v17 checkpoints.

Usage:
  python -m research.runners.multitag_eval_v17 --load-bridge .../seed42_v17.simstate.h5 ...

Maintains parallelism with compose_engram_demo_v2.py (v17 wrapper for
compose_engram_demo).
"""
from __future__ import annotations
import research.runners.concept_pool_demo as cpd_v1

# Extended vocab (matches compose_engram_demo_v2.py)
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

# Patch the v1 module
cpd_v1.NOUN_VOCAB = NOUN_VOCAB
cpd_v1.VERB_VOCAB = VERB_VOCAB
cpd_v1.ADJECTIVE_VOCAB = ADJECTIVE_VOCAB
cpd_v1.NOUN_NAMES = list(NOUN_VOCAB.values())
cpd_v1.VERB_NAMES = list(VERB_VOCAB.values())
cpd_v1.ADJECTIVE_NAMES = list(ADJECTIVE_VOCAB.values())

# Now delegate to multitag_eval.main() which will build a 28-pool bridge
import research.runners.multitag_eval as mev

if __name__ == "__main__":
    mev.main()
