"""Direction 5 vocab spec - 5 categories x V=16 = 80 cross-bridge concepts.

Pre-registered FROZEN word lists per design doc
docs/plans/2026-05-25-direction-5-hybrid-sparse-distributed-bio_brain_regions-design.md
and implementation plan
docs/plans/2026-05-25-direction-5-hybrid-sparse-distributed-bio_brain_regions-implementation.md.

DELIBERATELY IDENTICAL to direction_4_vocab_spec.py's 5x16=80 concept set
(mirrors Direction 4 vocab exactly so that the Direction 5 HYBRID test is
DIRECTLY COMPARABLE to the Direction 4 NEGATIVE result on the SAME concept
set; the only architectural difference is the substrate (dedicated-only
vs hybrid dedicated + sparse), not the vocab).

Each list maps to ONE separate HYBRID bio_brain_regions + shared-sparse-pool
bridge:
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
- BridgeA / B / C load their vocab into the matching DEDICATED pool kind
  (noun_pool_names / verb_pool_names / adjective_pool_names) of
  build_biological_brain_regions.
- BridgeD (spatial) and BridgeE (functional) load their vocab into the
  noun_pool_names slot - the protected builder has no dedicated spatial
  or functional pool kind, but the concept-pool architecture is category-
  agnostic at the pool level. This preserves the protected builder
  byte-unchanged.
- ALL 5 bridges ALSO get a NEW shared_concept_pool region (2000 neurons,
  per-concept K=100 sparse pattern). The cross-bridge probe reads OUT of
  the shared_concept_pool ONLY.

DISCIPLINE: this module is data only (no imports beyond typing). The
words are FROZEN as module-level constants; no runtime override path.
Any PR that silently changes a word triggers the grounding-pin test
test_direction_5_vocab_spec_has_5_categories_v16_each in
tests/test_direction_5_grounding.py.
"""
from __future__ import annotations
from typing import Dict, List


# -----------------------------------------------------------------------
# Pre-registered 5-category V=16 vocab lists (frozen at module load).
# Identical to direction_4_vocab_spec by design (see module docstring).
# -----------------------------------------------------------------------

# Bridge A - nouns (v14 baseline 4 + extension 12 = 16).
DIRECTION_5_NOUN_VOCAB: Dict[str, str] = {
    # v14 baseline
    "apple": "APPLE",
    "river": "RIVER",
    "dog": "DOG",
    "cat": "CAT",
    # Extension to V=16
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
DIRECTION_5_VERB_VOCAB: Dict[str, str] = {
    # v14 baseline
    "go": "GO",
    "come": "COME",
    "stop": "STOP",
    "look": "LOOK",
    # Extension to V=16
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
DIRECTION_5_ADJECTIVE_VOCAB: Dict[str, str] = {
    # v14 baseline
    "big": "BIG",
    "small": "SMALL",
    "hot": "HOT",
    "cold": "COLD",
    # Extension to V=16
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

# Bridge D - spatial (V=16). Loaded via the noun_pool_names slot of the
# protected builder (no dedicated spatial pool kind; concept-pool
# architecture is category-agnostic at the pool level).
DIRECTION_5_SPATIAL_VOCAB: Dict[str, str] = {
    # Cardinal directions (Tier 1 motor canon)
    "north": "NORTH",
    "east": "EAST",
    "south": "SOUTH",
    "west": "WEST",
    # Vertical / orthogonal axis
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
    # Containment / topology
    "in": "IN",
    "out": "OUT",
    # Distance
    "near": "NEAR",
    "far": "FAR",
    # Local extrema
    "top": "TOP",
    "bottom": "BOTTOM",
    "center": "CENTER",
    "side": "SIDE",
}

# Bridge E - functional (V=16). Same noun_pool_names slot mapping rationale
# as Bridge D.
DIRECTION_5_FUNCTIONAL_VOCAB: Dict[str, str] = {
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
DIRECTION_5_NOUN_NAMES: List[str] = list(DIRECTION_5_NOUN_VOCAB.values())
DIRECTION_5_VERB_NAMES: List[str] = list(DIRECTION_5_VERB_VOCAB.values())
DIRECTION_5_ADJECTIVE_NAMES: List[str] = list(DIRECTION_5_ADJECTIVE_VOCAB.values())
DIRECTION_5_SPATIAL_NAMES: List[str] = list(DIRECTION_5_SPATIAL_VOCAB.values())
DIRECTION_5_FUNCTIONAL_NAMES: List[str] = list(DIRECTION_5_FUNCTIONAL_VOCAB.values())

# Per-bridge word lists (the word-as-key order; used by training schedules
# + cross-bridge probe decoder that consumes the FROZEN word order).
DIRECTION_5_BRIDGE_A_WORDS: List[str] = list(DIRECTION_5_NOUN_VOCAB.keys())
DIRECTION_5_BRIDGE_B_WORDS: List[str] = list(DIRECTION_5_VERB_VOCAB.keys())
DIRECTION_5_BRIDGE_C_WORDS: List[str] = list(DIRECTION_5_ADJECTIVE_VOCAB.keys())
DIRECTION_5_BRIDGE_D_WORDS: List[str] = list(DIRECTION_5_SPATIAL_VOCAB.keys())
DIRECTION_5_BRIDGE_E_WORDS: List[str] = list(DIRECTION_5_FUNCTIONAL_VOCAB.keys())

# Frozen union word order (cross-bridge probe at Task 4 consumes this).
# Iteration order: bridgeA (16 nouns) -> bridgeB (16 verbs) ->
# bridgeC (16 adj) -> bridgeD (16 spatial) -> bridgeE (16 functional)
DIRECTION_5_ALL_WORDS: List[str] = (
    DIRECTION_5_BRIDGE_A_WORDS
    + DIRECTION_5_BRIDGE_B_WORDS
    + DIRECTION_5_BRIDGE_C_WORDS
    + DIRECTION_5_BRIDGE_D_WORDS
    + DIRECTION_5_BRIDGE_E_WORDS
)

# Pre-registered total cross-bridge concept count
DIRECTION_5_TOTAL: int = 80

# Bridge -> {ordered words, ordered pool-names, builder-slot} map. Used by
# the bridge builder (Task 2) and the multi-seed runner (Task 5,
# controller-only).
DIRECTION_5_BRIDGE_CATALOG: Dict[str, Dict[str, object]] = {
    "A_nouns": {
        "words": DIRECTION_5_BRIDGE_A_WORDS,
        "pool_names": DIRECTION_5_NOUN_NAMES,
        "builder_slot": "noun_pool_names",
    },
    "B_verbs": {
        "words": DIRECTION_5_BRIDGE_B_WORDS,
        "pool_names": DIRECTION_5_VERB_NAMES,
        "builder_slot": "verb_pool_names",
    },
    "C_adj": {
        "words": DIRECTION_5_BRIDGE_C_WORDS,
        "pool_names": DIRECTION_5_ADJECTIVE_NAMES,
        "builder_slot": "adjective_pool_names",
    },
    "D_spatial": {
        "words": DIRECTION_5_BRIDGE_D_WORDS,
        "pool_names": DIRECTION_5_SPATIAL_NAMES,
        # Maps via noun_pool_names slot (no dedicated spatial pool kind
        # in the protected builder; concept-pool architecture is
        # category-agnostic at the pool level).
        "builder_slot": "noun_pool_names",
    },
    "E_functional": {
        "words": DIRECTION_5_BRIDGE_E_WORDS,
        "pool_names": DIRECTION_5_FUNCTIONAL_NAMES,
        # Same noun_pool_names slot mapping rationale as Bridge D.
        "builder_slot": "noun_pool_names",
    },
}
