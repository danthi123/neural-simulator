"""Direction 6 vocab spec - 5 categories x V=32 = 160 cross-bridge concepts.

Pre-registered FROZEN word lists. This is the 2x-vocab extension of
Direction 4 (5 categories x V=16 = 80 cross-bridge concepts) per the
pillar n=108 D4 dedicated-pool result, scaling to a per-bridge vocab
comparable to pillar n=95 G.20 sparse (V=160).

Each list maps to ONE separate bio_brain_regions bridge:
- BridgeA (nouns, 32 words): D4's 16 nouns + 16 more extension
- BridgeB (verbs, 32 words): D4's 16 verbs + 16 more extension
- BridgeC (adjectives, 32 words): D4's 16 adjectives + 16 more extension
- BridgeD (spatial, 32 words): D4's 16 spatial + 16 more extension
- BridgeE (functional, 32 words): D4's 16 functional + 16 more extension

TOTAL: 160 unique cross-bridge concepts.

Architectural notes (per implementation plan; same as D4):
- BridgeA / B / C load their vocab into the matching pool kind
  (noun_pool_names / verb_pool_names / adjective_pool_names).
- BridgeD (spatial) and BridgeE (functional) load their vocab into the
  noun_pool_names slot - the protected builder has no dedicated spatial
  or functional pool kind, but the concept-pool architecture is category-
  agnostic at the pool level (each pool is a 200-neuron concept attractor
  with FS interneurons + lang_input/lang_output pathways). This preserves
  build_biological_brain_regions byte-unchanged.

Predicted via FHRR algebra capacity ratio (capacity proportional to
N_dim/V): D4 V=80 hit boundary at L=6/L=7; D6 V=160 (2x vocab) predicted
boundary at L=3/L=4. Still PASSing for moderate loads {2, 3, 5}.

DISCIPLINE: this module is data only (no imports beyond typing). The
words are FROZEN as module-level constants; no runtime override path.
Any PR that silently changes a word triggers the grounding-pin test
test_direction_6_vocab_spec_has_5_categories_v32_each in
tests/test_direction_6_grounding.py.
"""
from __future__ import annotations
from typing import Dict, List


# -----------------------------------------------------------------------
# Pre-registered 5-category V=32 vocab lists (frozen at module load).
# Each list extends the validated D4 V=16 lists with 16 more unique words
# per category. NO duplicates across the 5 categories.
# -----------------------------------------------------------------------

# Bridge A - nouns (D4's 16 + extension 16 = 32).
DIRECTION_6_NOUN_VOCAB: Dict[str, str] = {
    # D4 baseline (preserves D4 pool index ordering for any downstream
    # decoder consuming the frozen word order)
    "apple": "APPLE",
    "river": "RIVER",
    "dog": "DOG",
    "cat": "CAT",
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
    # D6 extension (16 new nouns; concrete + short; no overlap with
    # spatial/functional categories below)
    "wolf": "WOLF",
    "fish": "FISH",
    "rock": "ROCK",
    "leaf": "LEAF",
    "boat": "BOAT",
    "key": "KEY",
    "bone": "BONE",
    "rope": "ROPE",
    "cake": "CAKE",
    "bread": "BREAD",
    "milk": "MILK",
    "salt": "SALT",
    "stone": "STONE",
    "wave": "WAVE",
    "cloud": "CLOUD",
    "rain": "RAIN",
}

# Bridge B - verbs (D4's 16 + extension 16 = 32).
DIRECTION_6_VERB_VOCAB: Dict[str, str] = {
    # D4 baseline
    "go": "GO",
    "come": "COME",
    "stop": "STOP",
    "look": "LOOK",
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
    # D6 extension (16 new verbs)
    "push": "PUSH",
    "drop": "DROP",
    "find": "FIND",
    "hide": "HIDE",
    "save": "SAVE",
    "lose": "LOSE",
    "win": "WIN",
    "fall": "FALL",
    "swim": "SWIM",
    "fly": "FLY",
    "read": "READ",
    "write": "WRITE",
    "sing": "SING",
    "dance": "DANCE",
    "build": "BUILD",
    "break": "BREAK",
}

# Bridge C - adjectives (D4's 16 + extension 16 = 32).
DIRECTION_6_ADJECTIVE_VOCAB: Dict[str, str] = {
    # D4 baseline
    "big": "BIG",
    "small": "SMALL",
    "hot": "HOT",
    "cold": "COLD",
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
    # D6 extension (16 new adjectives; orthogonal property dimensions)
    "rough": "ROUGH",
    "smooth": "SMOOTH",
    "full": "FULL",
    "empty": "EMPTY",
    "wet": "WET",
    "dry": "DRY",
    "clean": "CLEAN",
    "dirty": "DIRTY",
    "warm": "WARM",
    "cool": "COOL",
    "thick": "THICK",
    "thin": "THIN",
    "wide": "WIDE",
    "narrow": "NARROW",
    "deep": "DEEP",
    "shallow": "SHALLOW",
}

# Bridge D - spatial words (32). Mapped via noun_pool_names slot in the
# protected builder (no dedicated spatial pool kind; concept-pool
# architecture is category-agnostic at the pool level).
DIRECTION_6_SPATIAL_VOCAB: Dict[str, str] = {
    # D4 baseline
    "north": "NORTH",
    "east": "EAST",
    "south": "SOUTH",
    "west": "WEST",
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
    "in": "IN",
    "out": "OUT",
    "near": "NEAR",
    "far": "FAR",
    "top": "TOP",
    "bottom": "BOTTOM",
    "center": "CENTER",
    "side": "SIDE",
    # D6 extension (16 new spatial words; front/back/here/there from
    # the prompt + 12 more body-relative + topological terms)
    "front": "FRONT",
    "back": "BACK",
    "here": "HERE",
    "there": "THERE",
    "above": "ABOVE",
    "below": "BELOW",
    "inside": "INSIDE",
    "outside": "OUTSIDE",
    "behind": "BEHIND",
    "beside": "BESIDE",
    "between": "BETWEEN",
    "across": "ACROSS",
    "along": "ALONG",
    "around": "AROUND",
    "beyond": "BEYOND",
    "within": "WITHIN",
}

# Bridge E - functional words (32). Same noun_pool_names slot mapping
# rationale as Bridge D.
DIRECTION_6_FUNCTIONAL_VOCAB: Dict[str, str] = {
    # D4 baseline
    "i": "I",
    "you": "YOU",
    "he": "HE",
    "she": "SHE",
    "the": "THE",
    "a": "A",
    "and": "AND",
    "or": "OR",
    "with": "WITH",
    "for": "FOR",
    "this": "THIS",
    "that": "THAT",
    "these": "THESE",
    "those": "THOSE",
    "what": "WHAT",
    "when": "WHEN",
    # D6 extension (16 new functional words; pronouns + wh-words +
    # auxiliaries + conjunctions)
    "we": "WE",
    "they": "THEY",
    "it": "IT",
    "me": "ME",
    "us": "US",
    "them": "THEM",
    "who": "WHO",
    "where": "WHERE",
    "why": "WHY",
    "how": "HOW",
    "but": "BUT",
    "if": "IF",
    "then": "THEN",
    "now": "NOW",
    "is": "IS",
    "was": "WAS",
}


# -----------------------------------------------------------------------
# Derived helpers (also frozen; no runtime override path).
# -----------------------------------------------------------------------

# Pool-name lists in the order expected by build_biological_brain_regions
# (noun_pool_names / verb_pool_names / adjective_pool_names parameters).
DIRECTION_6_NOUN_NAMES: List[str] = list(DIRECTION_6_NOUN_VOCAB.values())
DIRECTION_6_VERB_NAMES: List[str] = list(DIRECTION_6_VERB_VOCAB.values())
DIRECTION_6_ADJECTIVE_NAMES: List[str] = list(DIRECTION_6_ADJECTIVE_VOCAB.values())
DIRECTION_6_SPATIAL_NAMES: List[str] = list(DIRECTION_6_SPATIAL_VOCAB.values())
DIRECTION_6_FUNCTIONAL_NAMES: List[str] = list(DIRECTION_6_FUNCTIONAL_VOCAB.values())

# Per-bridge word lists (the word-as-key order; used by training schedules
# + cross-bridge probe decoder that consumes the FROZEN word order).
DIRECTION_6_BRIDGE_A_WORDS: List[str] = list(DIRECTION_6_NOUN_VOCAB.keys())
DIRECTION_6_BRIDGE_B_WORDS: List[str] = list(DIRECTION_6_VERB_VOCAB.keys())
DIRECTION_6_BRIDGE_C_WORDS: List[str] = list(DIRECTION_6_ADJECTIVE_VOCAB.keys())
DIRECTION_6_BRIDGE_D_WORDS: List[str] = list(DIRECTION_6_SPATIAL_VOCAB.keys())
DIRECTION_6_BRIDGE_E_WORDS: List[str] = list(DIRECTION_6_FUNCTIONAL_VOCAB.keys())

# Frozen union word order (cross-bridge probe at Task 4 consumes this).
# Iteration order: bridgeA (32 nouns) -> bridgeB (32 verbs) ->
# bridgeC (32 adj) -> bridgeD (32 spatial) -> bridgeE (32 functional)
DIRECTION_6_ALL_WORDS: List[str] = (
    DIRECTION_6_BRIDGE_A_WORDS
    + DIRECTION_6_BRIDGE_B_WORDS
    + DIRECTION_6_BRIDGE_C_WORDS
    + DIRECTION_6_BRIDGE_D_WORDS
    + DIRECTION_6_BRIDGE_E_WORDS
)

# Pre-registered total cross-bridge concept count
DIRECTION_6_TOTAL: int = 160

# Bridge -> {ordered words, ordered pool-names, builder-slot} map. Used by
# the bridge builder (Task 2) and the multi-seed runner (Task 5,
# controller-only).
DIRECTION_6_BRIDGE_CATALOG: Dict[str, Dict[str, object]] = {
    "A_nouns": {
        "words": DIRECTION_6_BRIDGE_A_WORDS,
        "pool_names": DIRECTION_6_NOUN_NAMES,
        "builder_slot": "noun_pool_names",
    },
    "B_verbs": {
        "words": DIRECTION_6_BRIDGE_B_WORDS,
        "pool_names": DIRECTION_6_VERB_NAMES,
        "builder_slot": "verb_pool_names",
    },
    "C_adj": {
        "words": DIRECTION_6_BRIDGE_C_WORDS,
        "pool_names": DIRECTION_6_ADJECTIVE_NAMES,
        "builder_slot": "adjective_pool_names",
    },
    "D_spatial": {
        "words": DIRECTION_6_BRIDGE_D_WORDS,
        "pool_names": DIRECTION_6_SPATIAL_NAMES,
        # Maps via noun_pool_names slot (no dedicated spatial pool kind
        # in the protected builder; concept-pool architecture is
        # category-agnostic at the pool level).
        "builder_slot": "noun_pool_names",
    },
    "E_functional": {
        "words": DIRECTION_6_BRIDGE_E_WORDS,
        "pool_names": DIRECTION_6_FUNCTIONAL_NAMES,
        # Same noun_pool_names slot mapping rationale as Bridge D.
        "builder_slot": "noun_pool_names",
    },
}
