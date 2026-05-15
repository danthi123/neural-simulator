"""Vocab specification for the 5-bridge G.20 ensemble (160 concepts total).

5 bridges × 32 concepts each = 160 unique concept words spanning:
  - Bridge A (nouns): animals + plants + objects + body + people + places
  - Bridge B (verbs): motion + perception + manipulation + communication + state
  - Bridge C (adjectives): size + temp + speed + color + emotion + age + quality
  - Bridge D (spatial): directions + spatial relations + temporal positions
  - Bridge E (functional): numbers + quantifiers + question words + discourse

This gives toddler-vocabulary breadth in concept space. Combined with
path-2 morpheme tokenization (~6x combinatorial reach), the effective
surface-form vocabulary reaches ~960 words.

Each bridge trains at 32 concepts on the validated G.20 architecture
(81.2% top-1, 96.9% top-5 seed 42 baseline).
"""

VOCAB_BRIDGE_A_NOUNS = [
    # Animals
    "apple", "river", "dog", "cat", "bird", "fish", "mouse", "frog",
    # Plants (proxy)
    "tree", "flower", "leaf", "fruit",
    # Objects
    "ball", "key", "book", "cup",
    # Body
    "hand", "foot", "head", "eye",
    # People
    "person", "baby", "child", "friend",
    # Places
    "house", "road", "garden", "park",
    # Substances
    "water", "fire", "sun", "moon",
]
assert len(VOCAB_BRIDGE_A_NOUNS) == 32, \
    f"BridgeA has {len(VOCAB_BRIDGE_A_NOUNS)} concepts, expected 32"

VOCAB_BRIDGE_B_VERBS = [
    # Motion
    "go", "come", "run", "walk", "jump", "fall", "fly", "swim",
    # Perception
    "look", "see", "hear", "listen", "watch", "find",
    # Manipulation
    "push", "pull", "open", "close", "give", "take", "hold", "drop",
    # Communication
    "speak", "read", "write", "say",
    # Consumption
    "eat", "drink",
    # State
    "sleep", "wake", "stop", "wait",
]
assert len(VOCAB_BRIDGE_B_VERBS) == 32, \
    f"BridgeB has {len(VOCAB_BRIDGE_B_VERBS)} concepts, expected 32"

VOCAB_BRIDGE_C_ADJECTIVES = [
    # Size
    "big", "small", "tall", "short", "long", "wide",
    # Temperature
    "hot", "cold", "warm", "cool",
    # Speed
    "fast", "slow",
    # Color
    "red", "blue", "green", "yellow", "white", "black",
    # Emotion
    "happy", "sad", "angry", "scared",
    # Age / freshness
    "new", "old",
    # Quality / condition
    "clean", "dirty", "wet", "dry",
    # Texture
    "hard", "soft", "smooth",
    # Quantity
    "full",
]
assert len(VOCAB_BRIDGE_C_ADJECTIVES) == 32, \
    f"BridgeC has {len(VOCAB_BRIDGE_C_ADJECTIVES)} concepts, expected 32"

VOCAB_BRIDGE_D_SPATIAL = [
    # Cardinal directions
    "north", "south", "east", "west",
    # Spatial directions
    "up", "down", "left", "right",
    # Spatial relations
    "here", "there", "near", "far",
    "in", "out", "on", "under",
    "above", "below", "front", "back",
    "top", "bottom", "side", "middle",
    # Temporal positions
    "now", "then", "before", "after",
    "first", "last", "next", "today",
]
assert len(VOCAB_BRIDGE_D_SPATIAL) == 32, \
    f"BridgeD has {len(VOCAB_BRIDGE_D_SPATIAL)} concepts, expected 32"

VOCAB_BRIDGE_E_FUNCTIONAL = [
    # Numbers (small)
    "one", "two", "three", "four", "five", "many", "few", "some",
    # Quantifiers
    "all", "none", "every", "any",
    # Question words
    "what", "where", "when", "who", "why", "how",
    # Demonstratives / deictics
    "this", "that",
    # Yes/no
    "yes", "no",
    # Discourse / politeness
    "please", "thanks", "hello", "goodbye", "sorry", "ok",
    # Existence
    "is", "have", "want", "need",
]
assert len(VOCAB_BRIDGE_E_FUNCTIONAL) == 32, \
    f"BridgeE has {len(VOCAB_BRIDGE_E_FUNCTIONAL)} concepts, expected 32"

ALL_BRIDGES = {
    "bridgeA_nouns": VOCAB_BRIDGE_A_NOUNS,
    "bridgeB_verbs": VOCAB_BRIDGE_B_VERBS,
    "bridgeC_adj": VOCAB_BRIDGE_C_ADJECTIVES,
    "bridgeD_spatial": VOCAB_BRIDGE_D_SPATIAL,
    "bridgeE_functional": VOCAB_BRIDGE_E_FUNCTIONAL,
}

# Verify no duplicates across bridges
ALL_WORDS = []
for words in ALL_BRIDGES.values():
    ALL_WORDS.extend(words)
assert len(ALL_WORDS) == len(set(ALL_WORDS)), \
    f"Duplicate words across bridges: " \
    f"{[w for w in ALL_WORDS if ALL_WORDS.count(w) > 1]}"

TOTAL_VOCAB = 160
assert len(ALL_WORDS) == TOTAL_VOCAB, \
    f"Total vocab size {len(ALL_WORDS)}, expected {TOTAL_VOCAB}"


def write_vocab_files(out_dir: str = "research/findings/raw/g11_bg"):
    """Write a vocab file per bridge for use with g20_multibridge.py."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, vocab in ALL_BRIDGES.items():
        path = out / f"g20_{name}_vocab.txt"
        path.write_text("\n".join(vocab))
        print(f"  wrote {path}: {len(vocab)} concepts")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true",
                    help="Write vocab files for g20_multibridge")
    p.add_argument("--out-dir", type=str,
                    default="research/findings/raw/g11_bg")
    args = p.parse_args()

    print(f"5-bridge G.20 vocab specification ({TOTAL_VOCAB} unique concepts):")
    for name, vocab in ALL_BRIDGES.items():
        print(f"  {name} ({len(vocab)}): "
              f"{vocab[:6]} ... {vocab[-2:]}")
    print(f"\nTotal: {len(ALL_WORDS)} unique words, "
          f"no duplicates across bridges.")

    if args.write:
        print(f"\nWriting vocab files to {args.out_dir}:")
        write_vocab_files(args.out_dir)
