"""Direction 3 V=32 vocab spec - pre-registered extended vocab lists.

The frozen V=32 layout per docs/plans/2026-05-25-direction-3-vocab-scaling-
bio_brain_regions-design.md Option A (more pools; cheapest first probe):

  4 motor     + 12 nouns + 12 verbs + 4 adjectives = 32 distinct concepts

This 2x scales the validated v14/v16 V=16 architecture (4 motor + 4 nouns
+ 4 verbs + 4 adjectives = 16) by tripling the noun and verb categories
while keeping the motor and adjective baselines intact. The choice:

- KEEP 4 motor (north/east/south/west): Tier 1 cardinal directions are
  the load-bearing motor canon. Re-using preserves the architectural
  canon under test (build_biological_brain_regions has a 4-motor hard
  assumption; extending it would touch the protected builder).

- 12 nouns (3x the v14 4): apple, river, dog, cat (v14 baseline) plus
  tree, bird, sun, moon, book, chair, house, wheel (extension). All
  short concrete nouns to keep tokenization simple; chosen to span
  natural / artificial / animate categories the way v14's baseline
  spans them. Each gets its own dedicated 200-neuron pool with paired
  teacher current + FS cross-inhibition.

- 12 verbs (3x the v14 4): go, come, stop, look (v14 baseline) plus
  walk, run, eat, sleep, sit, stand, jump, climb (extension). Same
  topology rationale.

- KEEP 4 adjectives (big/small/hot/cold): v14 baseline. The v14 4 are
  fundamental property dimensions; extending would risk semantic overlap
  (e.g. "warm" vs "hot") that adds noise to the OB/OI capacity test.

The reason this is Option A (more pools) and not Option B (sparse coding
within existing pools) is documented in the design doc: more pools is
the cheapest first probe because the existing build_biological_brain_-
regions API already accepts noun_pool_names / verb_pool_names parameters
of arbitrary length. Sparse coding within shared pools (G.20 SDM style)
would require substantial bridge-builder modifications.

DISCIPLINE: this module is data only (no imports beyond typing). The
words are FROZEN as module-level constants; no runtime override path.
Any future PR that silently changes a word triggers the grounding-pin
test test_direction_3_vocab_spec_has_v32_lists in tests/test_direction_3_grounding.py.
"""
from __future__ import annotations
from typing import Dict, List


# -----------------------------------------------------------------------
# Pre-registered V=32 vocab lists (frozen at module load).
# -----------------------------------------------------------------------

# 4 motor pools (Tier 1 baseline; cardinal directions).
# Mapped to existing motor_N / motor_E / motor_S / motor_W regions.
DIRECTION_3_MOTOR_VOCAB: Dict[str, str] = {
    "north": "N",
    "east": "E",
    "south": "S",
    "west": "W",
}

# 12 noun pools (v14 baseline 4 + extension 8 = 12).
# Each maps to noun_pool_<UPPER>. The 4 v14 baseline appear first to
# preserve the v14/v16 pool index ordering for any downstream readout
# decoder that consumes the FROZEN word order.
DIRECTION_3_NOUN_VOCAB: Dict[str, str] = {
    # v14 baseline
    "apple": "APPLE",
    "river": "RIVER",
    "dog": "DOG",
    "cat": "CAT",
    # v32 extension (8 nouns)
    "tree": "TREE",
    "bird": "BIRD",
    "sun": "SUN",
    "moon": "MOON",
    "book": "BOOK",
    "chair": "CHAIR",
    "house": "HOUSE",
    "wheel": "WHEEL",
}

# 12 verb pools (v14 baseline 4 + extension 8 = 12).
# Each maps to verb_pool_<UPPER>.
DIRECTION_3_VERB_VOCAB: Dict[str, str] = {
    # v14 baseline
    "go": "GO",
    "come": "COME",
    "stop": "STOP",
    "look": "LOOK",
    # v32 extension (8 verbs)
    "walk": "WALK",
    "run": "RUN",
    "eat": "EAT",
    "sleep": "SLEEP",
    "sit": "SIT",
    "stand": "STAND",
    "jump": "JUMP",
    "climb": "CLIMB",
}

# 4 adjective pools (v14 baseline; orthogonal property dimensions).
# Each maps to adjective_pool_<UPPER>.
DIRECTION_3_ADJECTIVE_VOCAB: Dict[str, str] = {
    "big": "BIG",
    "small": "SMALL",
    "hot": "HOT",
    "cold": "COLD",
}


# -----------------------------------------------------------------------
# Derived helpers (also frozen; no runtime override path).
# -----------------------------------------------------------------------

# Pool-name lists in the order expected by build_biological_brain_regions
# (noun_pool_names / verb_pool_names / adjective_pool_names parameters).
DIRECTION_3_NOUN_NAMES: List[str] = list(DIRECTION_3_NOUN_VOCAB.values())
DIRECTION_3_VERB_NAMES: List[str] = list(DIRECTION_3_VERB_VOCAB.values())
DIRECTION_3_ADJECTIVE_NAMES: List[str] = list(DIRECTION_3_ADJECTIVE_VOCAB.values())
DIRECTION_3_MOTOR_NAMES: List[str] = list(DIRECTION_3_MOTOR_VOCAB.values())

# Frozen union word order (matches the per-kind concatenation pattern of
# the OPTION 3 V=16 probe + v14/v16 recipe). Iteration order:
#   1. motor (4) -> 2. noun (12) -> 3. verb (12) -> 4. adjective (4) = V=32
DIRECTION_3_V32_WORDS: List[str] = (
    list(DIRECTION_3_MOTOR_VOCAB.keys())
    + list(DIRECTION_3_NOUN_VOCAB.keys())
    + list(DIRECTION_3_VERB_VOCAB.keys())
    + list(DIRECTION_3_ADJECTIVE_VOCAB.keys())
)

# Pre-registered total V
DIRECTION_3_V32_TOTAL: int = 32

# Per-word -> target pool-region map (used by training schedules). The
# pool-region naming convention matches the bridge builder's region names.
DIRECTION_3_V32_TARGET_POOL: Dict[str, str] = {}
for _w, _a in DIRECTION_3_MOTOR_VOCAB.items():
    DIRECTION_3_V32_TARGET_POOL[_w] = "motor_" + _a
for _w, _n in DIRECTION_3_NOUN_VOCAB.items():
    DIRECTION_3_V32_TARGET_POOL[_w] = "noun_pool_" + _n
for _w, _n in DIRECTION_3_VERB_VOCAB.items():
    DIRECTION_3_V32_TARGET_POOL[_w] = "verb_pool_" + _n
for _w, _n in DIRECTION_3_ADJECTIVE_VOCAB.items():
    DIRECTION_3_V32_TARGET_POOL[_w] = "adjective_pool_" + _n
# Tidy up loop variables to keep module namespace clean
del _w, _a, _n
