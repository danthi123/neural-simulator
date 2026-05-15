"""Vocabulary 'Set 4' bridge — 12 more new words for 48-word total.

Set 4:
  Nouns:  person, baby, ball, key      (people/objects)
  Verbs:  open, close, push, pull      (manipulation actions)
  Adj:    happy, sad, full, empty      (states/feelings)

Combined vocab across 4 sets = 48 unique concept words.
"""
from __future__ import annotations
import research.runners.concept_pool_demo as cpd_v1

DIRECTION_VOCAB = {
    "north": "N", "east": "E", "south": "S", "west": "W",
}
NOUN_VOCAB = {
    "person": "PERSON", "baby": "BABY", "ball": "BALL", "key": "KEY",
}
VERB_VOCAB = {
    "open": "OPEN", "close": "CLOSE", "push": "PUSH", "pull": "PULL",
}
ADJECTIVE_VOCAB = {
    "happy": "HAPPY", "sad": "SAD", "full": "FULL", "empty": "EMPTY",
}

cpd_v1.NOUN_VOCAB = NOUN_VOCAB
cpd_v1.VERB_VOCAB = VERB_VOCAB
cpd_v1.ADJECTIVE_VOCAB = ADJECTIVE_VOCAB
cpd_v1.NOUN_NAMES = list(NOUN_VOCAB.values())
cpd_v1.VERB_NAMES = list(VERB_VOCAB.values())
cpd_v1.ADJECTIVE_NAMES = list(ADJECTIVE_VOCAB.values())

import research.runners.concept_compose_train as cct
_ALL_WORDS_SET4 = [
    "north", "east", "south", "west",
    "person", "baby", "ball", "key",
    "open", "close", "push", "pull",
    "happy", "sad", "full", "empty",
]
cct._ALL_WORDS = _ALL_WORDS_SET4
cct._WORD_TO_IDX = {w: i for i, w in enumerate(_ALL_WORDS_SET4)}
cct._WORD_TO_POOL = {
    "north": "motor_N", "east": "motor_E", "south": "motor_S", "west": "motor_W",
    "person": "noun_pool_PERSON", "baby": "noun_pool_BABY",
    "ball": "noun_pool_BALL", "key": "noun_pool_KEY",
    "open": "verb_pool_OPEN", "close": "verb_pool_CLOSE",
    "push": "verb_pool_PUSH", "pull": "verb_pool_PULL",
    "happy": "adjective_pool_HAPPY", "sad": "adjective_pool_SAD",
    "full": "adjective_pool_FULL", "empty": "adjective_pool_EMPTY",
}

import research.runners.compose_concept_engram as cce
cce._ALL_CONCEPTS = list(_ALL_WORDS_SET4[4:])


if __name__ == "__main__":
    cpd_v1.main()
