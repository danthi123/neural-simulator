"""Direction 4 vocab spec - 5 categories x V=16 = 80 cross-bridge concepts.

Pre-registered FROZEN word lists per design doc
docs/plans/2026-05-25-direction-4-cross-bridge-bio_brain_regions-design.md
Approach A (cheapest first; doesn't depend on Direction 3 outcome).

Each list maps to ONE separate bio_brain_regions bridge:
- BridgeA (nouns, 16 words): apple, river, dog, cat (v14 baseline) +
                                tree, bird, sun, moon, book, chair, house,
                                wheel, ball, cup, lamp, road (extension)
- BridgeB (verbs, 16 words): go, come, stop, look (v14 baseline) +
                                walk, run, eat, sleep, sit, stand, jump,
                                climb, throw, catch, lift, pull (extension)
- BridgeC (adjectives, 16 words): big, small, hot, cold (v14 baseline) +
                                fast, slow, bright, dark, loud, quiet,
                                sweet, sour, heavy, light, sharp, soft
- BridgeD (spatial, 16 words): north, east, south, west, up, down, left,
                                right, in, out, near, far, top, bottom,
                                center, side
- BridgeE (functional, 16 words): i, you, he, she, the, a, and, or, with,
                                for, this, that, these, those, what, when

TOTAL: 80 unique cross-bridge concepts.

Architectural notes (per implementation plan):
- BridgeA / B / C load their vocab into the matching pool kind
  (noun_pool_names / verb_pool_names / adjective_pool_names).
- BridgeD (spatial) and BridgeE (functional) load their vocab into the
  noun_pool_names slot — the protected builder has no dedicated spatial
  or functional pool kind, but the concept-pool architecture is category-
  agnostic at the pool level (each pool is a 200-neuron concept attractor
  with FS interneurons + lang_input/lang_output pathways). This preserves
  build_biological_brain_regions byte-unchanged.

DISCIPLINE: this module is data only (no imports beyond typing). The
words are FROZEN as module-level constants; no runtime override path.
Any PR that silently changes a word triggers the grounding-pin test
test_direction_4_vocab_spec_has_5_categories_v16_each in
tests/test_direction_4_grounding.py.
"""
from __future__ import annotations
from typing import Dict, List


# -----------------------------------------------------------------------
# Pre-registered 5-category V=16 vocab lists (frozen at module load).
# -----------------------------------------------------------------------

# Bridge A - nouns (v14 baseline 4 + extension 12 = 16).
DIRECTION_4_NOUN_VOCAB: Dict[str, str] = {
    # v14 baseline (preserves v14/v16 pool index ordering for any
    # downstream decoder consuming the frozen word order)
    "apple": "APPLE",
    "river": "RIVER",
    "dog": "DOG",
    "cat": "CAT",
    # Bridge A extension (12 new nouns)
    "tree": "TREE",
    "bird": "BIRD",
    "sun": "SUN",
    "moon": "MOON",
    "book": "BOOK",
    "chair": "CHAIR",
    "house": "HOUSE",
    "wheel": "WHEEL",
    "ball": "BALL",
    "cup": "CUP",
    "lamp": "LAMP",
    "road": "ROAD",
}

# Bridge B - verbs (v14 baseline 4 + extension 12 = 16).
DIRECTION_4_VERB_VOCAB: Dict[str, str] = {
    # v14 baseline
    "go": "GO",
    "come": "COME",
    "stop": "STOP",
    "look": "LOOK",
    # Bridge B extension (12 new verbs)
    "walk": "WALK",
    "run": "RUN",
    "eat": "EAT",
    "sleep": "SLEEP",
    "sit": "SIT",
    "stand": "STAND",
    "jump": "JUMP",
    "climb": "CLIMB",
    "throw": "THROW",
    "catch": "CATCH",
    "lift": "LIFT",
    "pull": "PULL",
}

# Bridge C - adjectives (v14 baseline 4 + extension 12 = 16).
DIRECTION_4_ADJECTIVE_VOCAB: Dict[str, str] = {
    # v14 baseline (orthogonal property dimensions)
    "big": "BIG",
    "small": "SMALL",
    "hot": "HOT",
    "cold": "COLD",
    # Bridge C extension (12 new adjectives)
    "fast": "FAST",
    "slow": "SLOW",
    "bright": "BRIGHT",
    "dark": "DARK",
    "loud": "LOUD",
    "quiet": "QUIET",
    "sweet": "SWEET",
    "sour": "SOUR",
    "heavy": "HEAVY",
    "light": "LIGHT",
    "sharp": "SHARP",
    "soft": "SOFT",
}

# Bridge D - spatial words (16). Mapped via noun_pool_names slot in the
# protected builder (no dedicated spatial pool kind; concept-pool
# architecture is category-agnostic at the pool level).
DIRECTION_4_SPATIAL_VOCAB: Dict[str, str] = {
    # Cardinal directions
    "north": "NORTH",
    "east": "EAST",
    "south": "SOUTH",
    "west": "WEST",
    # Vertical
    "up": "UP",
    "down": "DOWN",
    # Horizontal (egocentric)
    "left": "LEFT",
    "right": "RIGHT",
    # Topological
    "in": "IN",
    "out": "OUT",
    "near": "NEAR",
    "far": "FAR",
    # Body-relative
    "top": "TOP",
    "bottom": "BOTTOM",
    "center": "CENTER",
    "side": "SIDE",
}

# Bridge E - functional words (16). Same noun_pool_names slot mapping
# rationale as Bridge D.
DIRECTION_4_FUNCTIONAL_VOCAB: Dict[str, str] = {
    # Pronouns
    "i": "I",
    "you": "YOU",
    "he": "HE",
    "she": "SHE",
    # Determiners / articles
    "the": "THE",
    "a": "A",
    # Conjunctions
    "and": "AND",
    "or": "OR",
    # Prepositions (function-word forms; non-overlapping with spatial
    # bridge's prepositions like 'in' / 'out')
    "with": "WITH",
    "for": "FOR",
    # Demonstratives
    "this": "THIS",
    "that": "THAT",
    "these": "THESE",
    "those": "THOSE",
    # Wh-words
    "what": "WHAT",
    "when": "WHEN",
}


# -----------------------------------------------------------------------
# Derived helpers (also frozen; no runtime override path).
# -----------------------------------------------------------------------

# Pool-name lists in the order expected by build_biological_brain_regions
# (noun_pool_names / verb_pool_names / adjective_pool_names parameters).
DIRECTION_4_NOUN_NAMES: List[str] = list(DIRECTION_4_NOUN_VOCAB.values())
DIRECTION_4_VERB_NAMES: List[str] = list(DIRECTION_4_VERB_VOCAB.values())
DIRECTION_4_ADJECTIVE_NAMES: List[str] = list(DIRECTION_4_ADJECTIVE_VOCAB.values())
DIRECTION_4_SPATIAL_NAMES: List[str] = list(DIRECTION_4_SPATIAL_VOCAB.values())
DIRECTION_4_FUNCTIONAL_NAMES: List[str] = list(DIRECTION_4_FUNCTIONAL_VOCAB.values())

# Per-bridge word lists (the word-as-key order; used by training schedules
# + cross-bridge probe decoder that consumes the FROZEN word order).
DIRECTION_4_BRIDGE_A_WORDS: List[str] = list(DIRECTION_4_NOUN_VOCAB.keys())
DIRECTION_4_BRIDGE_B_WORDS: List[str] = list(DIRECTION_4_VERB_VOCAB.keys())
DIRECTION_4_BRIDGE_C_WORDS: List[str] = list(DIRECTION_4_ADJECTIVE_VOCAB.keys())
DIRECTION_4_BRIDGE_D_WORDS: List[str] = list(DIRECTION_4_SPATIAL_VOCAB.keys())
DIRECTION_4_BRIDGE_E_WORDS: List[str] = list(DIRECTION_4_FUNCTIONAL_VOCAB.keys())

# Frozen union word order (cross-bridge probe at Task 4 consumes this).
# Iteration order: bridgeA (16 nouns) -> bridgeB (16 verbs) ->
# bridgeC (16 adj) -> bridgeD (16 spatial) -> bridgeE (16 functional)
DIRECTION_4_ALL_WORDS: List[str] = (
    DIRECTION_4_BRIDGE_A_WORDS
    + DIRECTION_4_BRIDGE_B_WORDS
    + DIRECTION_4_BRIDGE_C_WORDS
    + DIRECTION_4_BRIDGE_D_WORDS
    + DIRECTION_4_BRIDGE_E_WORDS
)

# Pre-registered total cross-bridge concept count
DIRECTION_4_TOTAL: int = 80

# Bridge -> {ordered words, ordered pool-names, builder-slot} map. Used by
# the bridge builder (Task 2) and the multi-seed runner (Task 5,
# controller-only).
DIRECTION_4_BRIDGE_CATALOG: Dict[str, Dict[str, object]] = {
    "A_nouns": {
        "words": DIRECTION_4_BRIDGE_A_WORDS,
        "pool_names": DIRECTION_4_NOUN_NAMES,
        "builder_slot": "noun_pool_names",
    },
    "B_verbs": {
        "words": DIRECTION_4_BRIDGE_B_WORDS,
        "pool_names": DIRECTION_4_VERB_NAMES,
        "builder_slot": "verb_pool_names",
    },
    "C_adj": {
        "words": DIRECTION_4_BRIDGE_C_WORDS,
        "pool_names": DIRECTION_4_ADJECTIVE_NAMES,
        "builder_slot": "adjective_pool_names",
    },
    "D_spatial": {
        "words": DIRECTION_4_BRIDGE_D_WORDS,
        "pool_names": DIRECTION_4_SPATIAL_NAMES,
        # Maps via noun_pool_names slot (no dedicated spatial pool kind
        # in the protected builder; concept-pool architecture is
        # category-agnostic at the pool level).
        "builder_slot": "noun_pool_names",
    },
    "E_functional": {
        "words": DIRECTION_4_BRIDGE_E_WORDS,
        "pool_names": DIRECTION_4_FUNCTIONAL_NAMES,
        # Same noun_pool_names slot mapping rationale as Bridge D.
        "builder_slot": "noun_pool_names",
    },
}
