"""G.20 5-bridge 320-concept vocab spec (production scaling tier).

Extends the VALIDATED 160-concept spec (g20_vocab_spec.py, frozen as the
source of truth for the 5 trained 32-concept sparse bridges + the
end-to-end demo) by +32 distinct words per category -> 5 bridges x 64
concepts = 320 unique concepts.

Why 320: the 256-concept-training-bound conclusion identifies multi-bridge
as the production scaling route (linear in bridge count) and names
5 x 64 = 320 ("~3200 surface forms, age-5 vocabulary") as the production
target. Sparse-distributed is multi-seed validated at 64 concepts/bridge
@ 100% (288/288; 2026-05-15-sparse-distributed-capacity-curve.md), so the
per-bridge science is settled -- 320 is a scaling/integration tier, not a
new experiment.

This module is ADDITIVE: it does not mutate g20_vocab_spec (the 160
artifact stays frozen). The base 160 lists are imported and reused (DRY);
the global-uniqueness assert below is the safety net that makes the
hand-curated +160 extension correct-by-construction (a collision fails
import + the test, never silently).
"""
from __future__ import annotations

from research.runners.g20_vocab_spec import (
    VOCAB_BRIDGE_A_NOUNS as _A32,
    VOCAB_BRIDGE_B_VERBS as _B32,
    VOCAB_BRIDGE_C_ADJECTIVES as _C32,
    VOCAB_BRIDGE_D_SPATIAL as _D32,
    VOCAB_BRIDGE_E_FUNCTIONAL as _E32,
)

# --- +32 per category (2026-05-15, 64-concept extension) ---

EXT_A_NOUNS = [
    # Animals
    "horse", "cow", "pig", "sheep", "duck", "bee", "ant", "snake",
    "bear", "wolf",
    # Plants
    "grass", "seed", "root", "branch",
    # Objects
    "box", "bag", "spoon", "plate", "chair", "table", "bed", "door",
    "window",
    # Body
    "arm", "leg", "ear", "nose", "mouth",
    # People
    "mother", "father",
    # Places
    "school", "shop",
]

EXT_B_VERBS = [
    # Motion / action
    "sit", "stand", "turn", "climb", "crawl", "ride", "throw", "catch",
    "kick", "hit",
    # Perception
    "touch", "smell", "taste", "feel",
    # Manipulation
    "cut", "break", "build", "make", "fix", "carry", "bring", "send",
    # Communication
    "ask", "tell", "call", "answer",
    # Domestic
    "cook", "wash",
    # Activity / state
    "laugh", "cry", "play", "work",
]

EXT_C_ADJECTIVES = [
    # Size
    "huge", "tiny", "thin", "thick", "deep", "narrow",
    # Light
    "bright", "dark",
    # Sound
    "loud", "quiet",
    # Taste
    "sweet", "sour",
    # Weight
    "heavy", "light",
    # Strength
    "strong", "weak",
    # Wealth
    "rich", "poor",
    # Character
    "kind", "mean", "nice", "good", "bad",
    # Condition
    "empty",
    # Shape
    "round", "flat", "sharp",
    # Age
    "young",
    # Health
    "sick", "well",
    # Truth
    "true", "false",
]

EXT_D_SPATIAL = [
    # Spatial relations
    "inside", "outside", "between", "around", "through", "across",
    "along", "toward", "away", "beside", "behind", "beyond", "center",
    "edge", "corner", "forward",
    # Temporal
    "yesterday", "tomorrow", "soon", "late", "early", "always",
    "never", "often", "sometimes", "once", "again", "later", "until",
    "since", "during", "whenever",
]

EXT_E_FUNCTIONAL = [
    # Numbers
    "six", "seven", "eight", "nine", "ten", "zero", "half", "both",
    # Quantifiers / determiners
    "each", "other", "another", "same",
    # Question words
    "which", "whose",
    # Deictics
    "these", "those", "it",
    # Modifiers
    "maybe", "very", "too", "also", "only", "not",
    # Auxiliaries
    "can", "will", "do", "did",
    # Conjunctions
    "and", "or", "but", "if", "because",
]

VOCAB_BRIDGE_A_NOUNS_64 = list(_A32) + EXT_A_NOUNS
VOCAB_BRIDGE_B_VERBS_64 = list(_B32) + EXT_B_VERBS
VOCAB_BRIDGE_C_ADJECTIVES_64 = list(_C32) + EXT_C_ADJECTIVES
VOCAB_BRIDGE_D_SPATIAL_64 = list(_D32) + EXT_D_SPATIAL
VOCAB_BRIDGE_E_FUNCTIONAL_64 = list(_E32) + EXT_E_FUNCTIONAL

ALL_BRIDGES_64 = {
    "bridgeA_nouns": VOCAB_BRIDGE_A_NOUNS_64,
    "bridgeB_verbs": VOCAB_BRIDGE_B_VERBS_64,
    "bridgeC_adj": VOCAB_BRIDGE_C_ADJECTIVES_64,
    "bridgeD_spatial": VOCAB_BRIDGE_D_SPATIAL_64,
    "bridgeE_functional": VOCAB_BRIDGE_E_FUNCTIONAL_64,
}

for _name, _v in ALL_BRIDGES_64.items():
    assert len(_v) == 64, f"{_name} has {len(_v)} concepts, expected 64"
    assert len(_v) == len(set(_v)), \
        f"{_name} has internal duplicates: " \
        f"{sorted(w for w in _v if _v.count(w) > 1)}"

ALL_WORDS_64 = []
for _v in ALL_BRIDGES_64.values():
    ALL_WORDS_64.extend(_v)

# Safety net: any cross-bridge collision in the hand-curated extension
# fails HERE at import (and in the test) -- never silently.
assert len(ALL_WORDS_64) == len(set(ALL_WORDS_64)), (
    "Duplicate words across bridges: "
    f"{sorted(w for w in ALL_WORDS_64 if ALL_WORDS_64.count(w) > 1)}"
)

TOTAL_VOCAB_64 = 320
assert len(ALL_WORDS_64) == TOTAL_VOCAB_64, \
    f"Total vocab size {len(ALL_WORDS_64)}, expected {TOTAL_VOCAB_64}"


def write_vocab_files_64(out_dir: str = "research/findings/raw/g11_bg"):
    """Write a 64-concept vocab file per bridge under a distinct
    *_vocab64.txt name so the validated 160-concept *_vocab.txt files
    (and the 5 trained 32-concept bridges) are NOT clobbered."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, vocab in ALL_BRIDGES_64.items():
        path = out / f"g20_{name}_vocab64.txt"
        path.write_text("\n".join(vocab))
        print(f"  wrote {path}: {len(vocab)} concepts")
        written.append(str(path))
    return written


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true",
                    help="Write 64-concept vocab files (vocab64.txt)")
    p.add_argument("--out-dir", type=str,
                    default="research/findings/raw/g11_bg")
    args = p.parse_args()
    print(f"5-bridge G.20 320-concept vocab spec "
          f"({TOTAL_VOCAB_64} unique concepts, 64/bridge):")
    for name, vocab in ALL_BRIDGES_64.items():
        print(f"  {name} ({len(vocab)}): "
              f"{vocab[:4]} ... {vocab[-3:]}")
    if args.write:
        print(f"\nWriting 64-concept vocab files to {args.out_dir}:")
        write_vocab_files_64(args.out_dir)
