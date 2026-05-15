"""Vocabulary 'Set 3' bridge — v16 architecture with 12 more NEW words.

Set3 adds 12 everyday-concept words. Combined with Set 1 and Set 2,
the multi-bridge chat REPL supports 36 unique concept words.

Set 3 (NEW):
  Nouns:  house, road, fire, water    (physical objects/places)
  Verbs:  give, take, find, lose       (transaction actions)
  Adj:    tall, short, wet, dry        (physical properties)

Combined vocab summary across 3 bridges:
  Set 1: apple, river, dog, cat, go, come, stop, look, big, small, hot, cold
  Set 2: tree, bird, sun, moon, walk, run, eat, sleep, red, blue, fast, slow
  Set 3: house, road, fire, water, give, take, find, lose, tall, short, wet, dry
  Total: 36 concept words

Each bridge uses validated v16 architecture (12 concept pools + 4 motors).
"""
from __future__ import annotations
import research.runners.concept_pool_demo as cpd_v1

DIRECTION_VOCAB = {
    "north": "N", "east": "E", "south": "S", "west": "W",
}
NOUN_VOCAB = {
    "house": "HOUSE", "road": "ROAD", "fire": "FIRE", "water": "WATER",
}
VERB_VOCAB = {
    "give": "GIVE", "take": "TAKE", "find": "FIND", "lose": "LOSE",
}
ADJECTIVE_VOCAB = {
    "tall": "TALL", "short": "SHORT", "wet": "WET", "dry": "DRY",
}

cpd_v1.NOUN_VOCAB = NOUN_VOCAB
cpd_v1.VERB_VOCAB = VERB_VOCAB
cpd_v1.ADJECTIVE_VOCAB = ADJECTIVE_VOCAB
cpd_v1.NOUN_NAMES = list(NOUN_VOCAB.values())
cpd_v1.VERB_NAMES = list(VERB_VOCAB.values())
cpd_v1.ADJECTIVE_NAMES = list(ADJECTIVE_VOCAB.values())

import research.runners.concept_compose_train as cct
_ALL_WORDS_SET3 = [
    "north", "east", "south", "west",
    "house", "road", "fire", "water",
    "give", "take", "find", "lose",
    "tall", "short", "wet", "dry",
]
cct._ALL_WORDS = _ALL_WORDS_SET3
cct._WORD_TO_IDX = {w: i for i, w in enumerate(_ALL_WORDS_SET3)}
cct._WORD_TO_POOL = {
    "north": "motor_N", "east": "motor_E", "south": "motor_S", "west": "motor_W",
    "house": "noun_pool_HOUSE", "road": "noun_pool_ROAD",
    "fire": "noun_pool_FIRE", "water": "noun_pool_WATER",
    "give": "verb_pool_GIVE", "take": "verb_pool_TAKE",
    "find": "verb_pool_FIND", "lose": "verb_pool_LOSE",
    "tall": "adjective_pool_TALL", "short": "adjective_pool_SHORT",
    "wet": "adjective_pool_WET", "dry": "adjective_pool_DRY",
}

import research.runners.compose_concept_engram as cce
cce._ALL_CONCEPTS = list(_ALL_WORDS_SET3[4:])


if __name__ == "__main__":
    cpd_v1.main()
