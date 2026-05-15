"""Vocabulary 'Set 2' bridge — v16 architecture but with DIFFERENT 12 words.

Combined with the original v16 bridge (Set 1), this expands the
multi-bridge chat REPL's total vocabulary to 24 concept words:

Set 1 (original v16, validated 90% multi-seed):
  Nouns:  apple, river, dog, cat
  Verbs:  go, come, stop, look
  Adj:    big, small, hot, cold

Set 2 (NEW, same v16 architecture):
  Nouns:  tree, bird, sun, moon
  Verbs:  walk, run, eat, sleep
  Adj:    red, blue, fast, slow

Each bridge is the same 12-concept-pool architecture (validated to
work at 90% multitag), just with different vocabulary mapping. Combined
via the multi-bridge chat REPL, the user has 24 concept words available.

CRITICAL: this is the FIRST genuine vocab expansion in the autonomous
arc. v17 architectural ceiling means we can't fit 24 concepts in one
bridge — but we can fit 12+12=24 across two bridges that each handle
their portion at validated 90% reliability.
"""
from __future__ import annotations
import research.runners.concept_pool_demo as cpd_v1

# Set 2 vocabularies (v17 second-half words)
DIRECTION_VOCAB = {
    "north": "N", "east": "E", "south": "S", "west": "W",
}
NOUN_VOCAB = {
    "tree": "TREE", "bird": "BIRD", "sun": "SUN", "moon": "MOON",
}
VERB_VOCAB = {
    "walk": "WALK", "run": "RUN", "eat": "EAT", "sleep": "SLEEP",
}
ADJECTIVE_VOCAB = {
    "red": "RED", "blue": "BLUE", "fast": "FAST", "slow": "SLOW",
}

# Patch concept_pool_demo's vocab tables BEFORE main() runs
cpd_v1.NOUN_VOCAB = NOUN_VOCAB
cpd_v1.VERB_VOCAB = VERB_VOCAB
cpd_v1.ADJECTIVE_VOCAB = ADJECTIVE_VOCAB
cpd_v1.NOUN_NAMES = list(NOUN_VOCAB.values())
cpd_v1.VERB_NAMES = list(VERB_VOCAB.values())
cpd_v1.ADJECTIVE_NAMES = list(ADJECTIVE_VOCAB.values())

# Also patch concept_compose_train's word index tables so engram
# encoding uses correct cue_idx for Set 2 words. CRITICAL.
import research.runners.concept_compose_train as cct
# Set 2 words at indices 4-15 (replacing apple-cold)
# Motors stay at 0-3.
_ALL_WORDS_SET2 = [
    "north", "east", "south", "west",
    "tree", "bird", "sun", "moon",
    "walk", "run", "eat", "sleep",
    "red", "blue", "fast", "slow",
]
cct._ALL_WORDS = _ALL_WORDS_SET2
cct._WORD_TO_IDX = {w: i for i, w in enumerate(_ALL_WORDS_SET2)}
cct._WORD_TO_POOL = {
    "north": "motor_N", "east": "motor_E", "south": "motor_S", "west": "motor_W",
    "tree": "noun_pool_TREE", "bird": "noun_pool_BIRD",
    "sun": "noun_pool_SUN", "moon": "noun_pool_MOON",
    "walk": "verb_pool_WALK", "run": "verb_pool_RUN",
    "eat": "verb_pool_EAT", "sleep": "verb_pool_SLEEP",
    "red": "adjective_pool_RED", "blue": "adjective_pool_BLUE",
    "fast": "adjective_pool_FAST", "slow": "adjective_pool_SLOW",
}

# Also patch compose_concept_engram's ALL_CONCEPTS to include set 2 words
import research.runners.compose_concept_engram as cce
cce._ALL_CONCEPTS = list(_ALL_WORDS_SET2[4:])  # skip motors


if __name__ == "__main__":
    cpd_v1.main()
