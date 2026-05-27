"""Direction 7 vocab spec - 5 categories x V=64 = 320 cross-bridge concepts.

Pre-registered FROZEN word lists. This is the 2x-vocab extension of
Direction 6 (5 categories x V=32 = 160 cross-bridge concepts) per the
pillar n=109 D6 dedicated-pool result, scaling to V=64 per bridge x
5 bridges = 320 cross-bridge concepts.

The 320 vocabulary is taken BYTE-FOR-BYTE from the Direction M G.20 sparse
production deliverable (research/findings/raw/g11_bg/g20_bridge{A..E}_*_vocab64.txt).
This makes D7 the biology-faithful counterpart of the user-facing G.20
sparse 320-concept chat capability -- a clean per-concept-per-bridge
comparison of bio_brain_regions vs the sparse-distributed substrate at
identical vocabulary.

Each list maps to ONE separate bio_brain_regions bridge:
- BridgeA (nouns, 64 words):       G.20 g20_bridgeA_nouns_vocab64.txt
- BridgeB (verbs, 64 words):       G.20 g20_bridgeB_verbs_vocab64.txt
- BridgeC (adjectives, 64 words):  G.20 g20_bridgeC_adj_vocab64.txt
- BridgeD (spatial, 64 words):     G.20 g20_bridgeD_spatial_vocab64.txt
- BridgeE (functional, 64 words):  G.20 g20_bridgeE_functional_vocab64.txt

TOTAL: 320 unique cross-bridge concepts.

Architectural notes (per implementation plan; same as D4 / D6):
- BridgeA / B / C load their vocab into the matching pool kind
  (noun_pool_names / verb_pool_names / adjective_pool_names).
- BridgeD (spatial) and BridgeE (functional) load their vocab into the
  noun_pool_names slot - the protected builder has no dedicated spatial
  or functional pool kind, but the concept-pool architecture is category-
  agnostic at the pool level (each pool is a 200-neuron concept attractor
  with FS interneurons + lang_input/lang_output pathways). This preserves
  build_biological_brain_regions byte-unchanged.

Predicted via FHRR algebra capacity ratio (capacity proportional to
N_dim/V): D6 V=160 SHATTERED the predicted L=3/L=4 boundary by passing
L=5 at OI 0.987 (production). The dedicated-pool architecture's near-
orthogonal grounded-symbol geometry has substantially more capacity per
dimension than the algebra prediction assumes. D7 V=320 is the next-tier
test: if D7 also passes the moderate loads {2, 3, 5}, biology-faithful
dedicated-pool architecture matches the Direction M G.20 sparse
production deliverable at V=320 -- the user-facing chat capability tier.

DISCIPLINE: this module is data only (no imports beyond typing). The
words are FROZEN as module-level constants; no runtime override path.
Any PR that silently changes a word triggers the grounding-pin test
test_direction_7_vocab_spec_has_5_categories_v64_each in
tests/test_direction_7_grounding.py.
"""
from __future__ import annotations
from typing import Dict, List


# -----------------------------------------------------------------------
# Pre-registered 5-category V=64 vocab lists (frozen at module load).
# Each list mirrors the G.20 320-concept production deliverable vocab64
# byte-for-byte. NO duplicates across the 5 categories.
# -----------------------------------------------------------------------

# Bridge A - nouns (64). Mirrors g20_bridgeA_nouns_vocab64.txt.
DIRECTION_7_NOUN_VOCAB: Dict[str, str] = {
    "apple": "APPLE",
    "river": "RIVER",
    "dog": "DOG",
    "cat": "CAT",
    "bird": "BIRD",
    "fish": "FISH",
    "mouse": "MOUSE",
    "frog": "FROG",
    "tree": "TREE",
    "flower": "FLOWER",
    "leaf": "LEAF",
    "fruit": "FRUIT",
    "ball": "BALL",
    "key": "KEY",
    "book": "BOOK",
    "cup": "CUP",
    "hand": "HAND",
    "foot": "FOOT",
    "head": "HEAD",
    "eye": "EYE",
    "person": "PERSON",
    "baby": "BABY",
    "child": "CHILD",
    "friend": "FRIEND",
    "house": "HOUSE",
    "road": "ROAD",
    "garden": "GARDEN",
    "park": "PARK",
    "water": "WATER",
    "fire": "FIRE",
    "sun": "SUN",
    "moon": "MOON",
    "horse": "HORSE",
    "cow": "COW",
    "pig": "PIG",
    "sheep": "SHEEP",
    "duck": "DUCK",
    "bee": "BEE",
    "ant": "ANT",
    "snake": "SNAKE",
    "bear": "BEAR",
    "wolf": "WOLF",
    "grass": "GRASS",
    "seed": "SEED",
    "root": "ROOT",
    "branch": "BRANCH",
    "box": "BOX",
    "bag": "BAG",
    "spoon": "SPOON",
    "plate": "PLATE",
    "chair": "CHAIR",
    "table": "TABLE",
    "bed": "BED",
    "door": "DOOR",
    "window": "WINDOW",
    "arm": "ARM",
    "leg": "LEG",
    "ear": "EAR",
    "nose": "NOSE",
    "mouth": "MOUTH",
    "mother": "MOTHER",
    "father": "FATHER",
    "school": "SCHOOL",
    "shop": "SHOP",
}

# Bridge B - verbs (64). Mirrors g20_bridgeB_verbs_vocab64.txt.
DIRECTION_7_VERB_VOCAB: Dict[str, str] = {
    "go": "GO",
    "come": "COME",
    "run": "RUN",
    "walk": "WALK",
    "jump": "JUMP",
    "fall": "FALL",
    "fly": "FLY",
    "swim": "SWIM",
    "look": "LOOK",
    "see": "SEE",
    "hear": "HEAR",
    "listen": "LISTEN",
    "watch": "WATCH",
    "find": "FIND",
    "push": "PUSH",
    "pull": "PULL",
    "open": "OPEN",
    "close": "CLOSE",
    "give": "GIVE",
    "take": "TAKE",
    "hold": "HOLD",
    "drop": "DROP",
    "speak": "SPEAK",
    "read": "READ",
    "write": "WRITE",
    "say": "SAY",
    "eat": "EAT",
    "drink": "DRINK",
    "sleep": "SLEEP",
    "wake": "WAKE",
    "stop": "STOP",
    "wait": "WAIT",
    "sit": "SIT",
    "stand": "STAND",
    "turn": "TURN",
    "climb": "CLIMB",
    "crawl": "CRAWL",
    "ride": "RIDE",
    "throw": "THROW",
    "catch": "CATCH",
    "kick": "KICK",
    "hit": "HIT",
    "touch": "TOUCH",
    "smell": "SMELL",
    "taste": "TASTE",
    "feel": "FEEL",
    "cut": "CUT",
    "break": "BREAK",
    "build": "BUILD",
    "make": "MAKE",
    "fix": "FIX",
    "carry": "CARRY",
    "bring": "BRING",
    "send": "SEND",
    "ask": "ASK",
    "tell": "TELL",
    "call": "CALL",
    "answer": "ANSWER",
    "cook": "COOK",
    "wash": "WASH",
    "laugh": "LAUGH",
    "cry": "CRY",
    "play": "PLAY",
    "work": "WORK",
}

# Bridge C - adjectives (64). Mirrors g20_bridgeC_adj_vocab64.txt.
DIRECTION_7_ADJECTIVE_VOCAB: Dict[str, str] = {
    "big": "BIG",
    "small": "SMALL",
    "tall": "TALL",
    "short": "SHORT",
    "long": "LONG",
    "wide": "WIDE",
    "hot": "HOT",
    "cold": "COLD",
    "warm": "WARM",
    "cool": "COOL",
    "fast": "FAST",
    "slow": "SLOW",
    "red": "RED",
    "blue": "BLUE",
    "green": "GREEN",
    "yellow": "YELLOW",
    "white": "WHITE",
    "black": "BLACK",
    "happy": "HAPPY",
    "sad": "SAD",
    "angry": "ANGRY",
    "scared": "SCARED",
    "new": "NEW",
    "old": "OLD",
    "clean": "CLEAN",
    "dirty": "DIRTY",
    "wet": "WET",
    "dry": "DRY",
    "hard": "HARD",
    "soft": "SOFT",
    "smooth": "SMOOTH",
    "full": "FULL",
    "huge": "HUGE",
    "tiny": "TINY",
    "thin": "THIN",
    "thick": "THICK",
    "deep": "DEEP",
    "narrow": "NARROW",
    "bright": "BRIGHT",
    "dark": "DARK",
    "loud": "LOUD",
    "quiet": "QUIET",
    "sweet": "SWEET",
    "sour": "SOUR",
    "heavy": "HEAVY",
    "light": "LIGHT",
    "strong": "STRONG",
    "weak": "WEAK",
    "rich": "RICH",
    "poor": "POOR",
    "kind": "KIND",
    "mean": "MEAN",
    "nice": "NICE",
    "good": "GOOD",
    "bad": "BAD",
    "empty": "EMPTY",
    "round": "ROUND",
    "flat": "FLAT",
    "sharp": "SHARP",
    "young": "YOUNG",
    "sick": "SICK",
    "well": "WELL",
    "true": "TRUE",
    "false": "FALSE",
}

# Bridge D - spatial words (64). Mirrors g20_bridgeD_spatial_vocab64.txt.
# Mapped via noun_pool_names slot in the protected builder (no dedicated
# spatial pool kind; concept-pool architecture is category-agnostic at
# the pool level).
DIRECTION_7_SPATIAL_VOCAB: Dict[str, str] = {
    "north": "NORTH",
    "south": "SOUTH",
    "east": "EAST",
    "west": "WEST",
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
    "here": "HERE",
    "there": "THERE",
    "near": "NEAR",
    "far": "FAR",
    "in": "IN",
    "out": "OUT",
    "on": "ON",
    "under": "UNDER",
    "above": "ABOVE",
    "below": "BELOW",
    "front": "FRONT",
    "back": "BACK",
    "top": "TOP",
    "bottom": "BOTTOM",
    "side": "SIDE",
    "middle": "MIDDLE",
    "now": "NOW",
    "then": "THEN",
    "before": "BEFORE",
    "after": "AFTER",
    "first": "FIRST",
    "last": "LAST",
    "next": "NEXT",
    "today": "TODAY",
    "inside": "INSIDE",
    "outside": "OUTSIDE",
    "between": "BETWEEN",
    "around": "AROUND",
    "through": "THROUGH",
    "across": "ACROSS",
    "along": "ALONG",
    "toward": "TOWARD",
    "away": "AWAY",
    "beside": "BESIDE",
    "behind": "BEHIND",
    "beyond": "BEYOND",
    "center": "CENTER",
    "edge": "EDGE",
    "corner": "CORNER",
    "forward": "FORWARD",
    "yesterday": "YESTERDAY",
    "tomorrow": "TOMORROW",
    "soon": "SOON",
    "late": "LATE",
    "early": "EARLY",
    "always": "ALWAYS",
    "never": "NEVER",
    "often": "OFTEN",
    "sometimes": "SOMETIMES",
    "once": "ONCE",
    "again": "AGAIN",
    "later": "LATER",
    "until": "UNTIL",
    "since": "SINCE",
    "during": "DURING",
    "whenever": "WHENEVER",
}

# Bridge E - functional words (64). Mirrors g20_bridgeE_functional_vocab64.txt.
# Same noun_pool_names slot mapping rationale as Bridge D.
DIRECTION_7_FUNCTIONAL_VOCAB: Dict[str, str] = {
    "one": "ONE",
    "two": "TWO",
    "three": "THREE",
    "four": "FOUR",
    "five": "FIVE",
    "many": "MANY",
    "few": "FEW",
    "some": "SOME",
    "all": "ALL",
    "none": "NONE",
    "every": "EVERY",
    "any": "ANY",
    "what": "WHAT",
    "where": "WHERE",
    "when": "WHEN",
    "who": "WHO",
    "why": "WHY",
    "how": "HOW",
    "this": "THIS",
    "that": "THAT",
    "yes": "YES",
    "no": "NO",
    "please": "PLEASE",
    "thanks": "THANKS",
    "hello": "HELLO",
    "goodbye": "GOODBYE",
    "sorry": "SORRY",
    "ok": "OK",
    "is": "IS",
    "have": "HAVE",
    "want": "WANT",
    "need": "NEED",
    "six": "SIX",
    "seven": "SEVEN",
    "eight": "EIGHT",
    "nine": "NINE",
    "ten": "TEN",
    "zero": "ZERO",
    "half": "HALF",
    "both": "BOTH",
    "each": "EACH",
    "other": "OTHER",
    "another": "ANOTHER",
    "same": "SAME",
    "which": "WHICH",
    "whose": "WHOSE",
    "these": "THESE",
    "those": "THOSE",
    "it": "IT",
    "maybe": "MAYBE",
    "very": "VERY",
    "too": "TOO",
    "also": "ALSO",
    "only": "ONLY",
    "not": "NOT",
    "can": "CAN",
    "will": "WILL",
    "do": "DO",
    "did": "DID",
    "and": "AND",
    "or": "OR",
    "but": "BUT",
    "if": "IF",
    "because": "BECAUSE",
}


# -----------------------------------------------------------------------
# Derived helpers (also frozen; no runtime override path).
# -----------------------------------------------------------------------

# Pool-name lists in the order expected by build_biological_brain_regions
# (noun_pool_names / verb_pool_names / adjective_pool_names parameters).
DIRECTION_7_NOUN_NAMES: List[str] = list(DIRECTION_7_NOUN_VOCAB.values())
DIRECTION_7_VERB_NAMES: List[str] = list(DIRECTION_7_VERB_VOCAB.values())
DIRECTION_7_ADJECTIVE_NAMES: List[str] = list(DIRECTION_7_ADJECTIVE_VOCAB.values())
DIRECTION_7_SPATIAL_NAMES: List[str] = list(DIRECTION_7_SPATIAL_VOCAB.values())
DIRECTION_7_FUNCTIONAL_NAMES: List[str] = list(DIRECTION_7_FUNCTIONAL_VOCAB.values())

# Per-bridge word lists (the word-as-key order; used by training schedules
# + cross-bridge probe decoder that consumes the FROZEN word order).
DIRECTION_7_BRIDGE_A_WORDS: List[str] = list(DIRECTION_7_NOUN_VOCAB.keys())
DIRECTION_7_BRIDGE_B_WORDS: List[str] = list(DIRECTION_7_VERB_VOCAB.keys())
DIRECTION_7_BRIDGE_C_WORDS: List[str] = list(DIRECTION_7_ADJECTIVE_VOCAB.keys())
DIRECTION_7_BRIDGE_D_WORDS: List[str] = list(DIRECTION_7_SPATIAL_VOCAB.keys())
DIRECTION_7_BRIDGE_E_WORDS: List[str] = list(DIRECTION_7_FUNCTIONAL_VOCAB.keys())

# Frozen union word order (cross-bridge probe at Task 4 consumes this).
# Iteration order: bridgeA (64 nouns) -> bridgeB (64 verbs) ->
# bridgeC (64 adj) -> bridgeD (64 spatial) -> bridgeE (64 functional)
DIRECTION_7_ALL_WORDS: List[str] = (
    DIRECTION_7_BRIDGE_A_WORDS
    + DIRECTION_7_BRIDGE_B_WORDS
    + DIRECTION_7_BRIDGE_C_WORDS
    + DIRECTION_7_BRIDGE_D_WORDS
    + DIRECTION_7_BRIDGE_E_WORDS
)

# Pre-registered total cross-bridge concept count
DIRECTION_7_TOTAL: int = 320

# Bridge -> {ordered words, ordered pool-names, builder-slot} map. Used by
# the bridge builder (Task 2) and the multi-seed runner (Task 5,
# controller-only).
DIRECTION_7_BRIDGE_CATALOG: Dict[str, Dict[str, object]] = {
    "A_nouns": {
        "words": DIRECTION_7_BRIDGE_A_WORDS,
        "pool_names": DIRECTION_7_NOUN_NAMES,
        "builder_slot": "noun_pool_names",
    },
    "B_verbs": {
        "words": DIRECTION_7_BRIDGE_B_WORDS,
        "pool_names": DIRECTION_7_VERB_NAMES,
        "builder_slot": "verb_pool_names",
    },
    "C_adj": {
        "words": DIRECTION_7_BRIDGE_C_WORDS,
        "pool_names": DIRECTION_7_ADJECTIVE_NAMES,
        "builder_slot": "adjective_pool_names",
    },
    "D_spatial": {
        "words": DIRECTION_7_BRIDGE_D_WORDS,
        "pool_names": DIRECTION_7_SPATIAL_NAMES,
        # Maps via noun_pool_names slot (no dedicated spatial pool kind
        # in the protected builder; concept-pool architecture is
        # category-agnostic at the pool level).
        "builder_slot": "noun_pool_names",
    },
    "E_functional": {
        "words": DIRECTION_7_BRIDGE_E_WORDS,
        "pool_names": DIRECTION_7_FUNCTIONAL_NAMES,
        # Same noun_pool_names slot mapping rationale as Bridge D.
        "builder_slot": "noun_pool_names",
    },
}
