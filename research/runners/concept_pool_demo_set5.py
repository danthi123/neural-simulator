"""Vocabulary 'Set 5' bridge — 12 more new words for 60-word total.

Set 5:
  Nouns:  food, drink, hand, foot       (body / sustenance)
  Verbs:  speak, listen, read, write    (communication)
  Adj:    new, old, clean, hard         (states / properties)

Combined vocab across 5 sets = 60 unique concept words.
"""
from __future__ import annotations
import research.runners.concept_pool_demo as cpd_v1

DIRECTION_VOCAB = {
    "north": "N", "east": "E", "south": "S", "west": "W",
}
NOUN_VOCAB = {
    "food": "FOOD", "drink": "DRINK", "hand": "HAND", "foot": "FOOT",
}
VERB_VOCAB = {
    "speak": "SPEAK", "listen": "LISTEN", "read": "READ", "write": "WRITE",
}
ADJECTIVE_VOCAB = {
    "new": "NEW", "old": "OLD", "clean": "CLEAN", "hard": "HARD",
}

cpd_v1.NOUN_VOCAB = NOUN_VOCAB
cpd_v1.VERB_VOCAB = VERB_VOCAB
cpd_v1.ADJECTIVE_VOCAB = ADJECTIVE_VOCAB
cpd_v1.NOUN_NAMES = list(NOUN_VOCAB.values())
cpd_v1.VERB_NAMES = list(VERB_VOCAB.values())
cpd_v1.ADJECTIVE_NAMES = list(ADJECTIVE_VOCAB.values())

import research.runners.concept_compose_train as cct
_ALL_WORDS_SET5 = [
    "north", "east", "south", "west",
    "food", "drink", "hand", "foot",
    "speak", "listen", "read", "write",
    "new", "old", "clean", "hard",
]
cct._ALL_WORDS = _ALL_WORDS_SET5
cct._WORD_TO_IDX = {w: i for i, w in enumerate(_ALL_WORDS_SET5)}
cct._WORD_TO_POOL = {
    "north": "motor_N", "east": "motor_E", "south": "motor_S", "west": "motor_W",
    "food": "noun_pool_FOOD", "drink": "noun_pool_DRINK",
    "hand": "noun_pool_HAND", "foot": "noun_pool_FOOT",
    "speak": "verb_pool_SPEAK", "listen": "verb_pool_LISTEN",
    "read": "verb_pool_READ", "write": "verb_pool_WRITE",
    "new": "adjective_pool_NEW", "old": "adjective_pool_OLD",
    "clean": "adjective_pool_CLEAN", "hard": "adjective_pool_HARD",
}

import research.runners.compose_concept_engram as cce
cce._ALL_CONCEPTS = list(_ALL_WORDS_SET5[4:])


if __name__ == "__main__":
    cpd_v1.main()
