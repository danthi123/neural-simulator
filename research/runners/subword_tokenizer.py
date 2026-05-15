"""Subword/morpheme tokenizer (path 2) — combinatorial vocab via BPE-like
morpheme decomposition.

Catalog refs:
- Bozic 2010 / Marslen-Wilson 2007: left inferior frontal gyrus performs
  morphological decomposition of complex words (un-happy, run-ning, etc.)
- Project: G.21 Hagoort MUC framework (memory/unification/control) — this
  is the 'unification' component for compositional word forms.

Architecture:
- A small dictionary of MORPHEMES (~50-200 atomic units)
- A rules-based tokenizer that splits surface words into morphemes
- Each morpheme is a CONCEPT in the multi-bridge system (existing v16
  / shared-pool storage)
- Composition: surface word = sequence of morpheme engram-tag activations

Combinatorial reach:
- 100 morphemes -> ~1000-5000 distinct surface words
- 200 morphemes -> ~10K-50K surface words

This is the same trick BPE/SentencePiece uses for LLMs, but with our
biology-grounded morphological-decomposition story (left IFG).
"""
from __future__ import annotations
import re
from typing import List, Tuple

# Small starter morpheme dictionary. Real BPE would learn this from corpus
# statistics; we hand-curate to start.
#
# Each entry: (morpheme, surface_pattern). Surface pattern is a regex
# anchor that the morpheme matches (e.g. "ing" matches at word end).
MORPHEMES_PREFIX = ["un", "re", "pre", "dis", "mis", "over", "under", "anti"]
MORPHEMES_SUFFIX = ["ing", "ed", "er", "est", "ly", "tion", "able", "ful",
                     "less", "ness", "s", "es", "ies"]

# Common irregular forms: map surface -> (root, suffix)
IRREGULAR_PAST = {
    "ate": ("eat", "PAST"),
    "drank": ("drink", "PAST"),
    "ran": ("run", "PAST"),
    "spoke": ("speak", "PAST"),
    "wrote": ("write", "PAST"),
    "took": ("take", "PAST"),
    "gave": ("give", "PAST"),
    "found": ("find", "PAST"),
    "lost": ("lose", "PAST"),
    "went": ("go", "PAST"),
    "came": ("come", "PAST"),
    "saw": ("see", "PAST"),
    "heard": ("hear", "PAST"),
    "slept": ("sleep", "PAST"),
}

# Pluralization
IRREGULAR_PLURAL = {
    "feet": ("foot", "PLURAL"),
    "hands": ("hand", "PLURAL"),
    "people": ("person", "PLURAL"),
    "babies": ("baby", "PLURAL"),
    "balls": ("ball", "PLURAL"),
    "keys": ("key", "PLURAL"),
    "dogs": ("dog", "PLURAL"),
    "cats": ("cat", "PLURAL"),
    "trees": ("tree", "PLURAL"),
    "birds": ("bird", "PLURAL"),
    "apples": ("apple", "PLURAL"),
    "rivers": ("river", "PLURAL"),
    "houses": ("house", "PLURAL"),
    "roads": ("road", "PLURAL"),
    "fires": ("fire", "PLURAL"),
}


def tokenize_word(surface: str, known_roots: set) -> List[str]:
    """Decompose a surface word into morphemes.

    Algorithm:
    1. Check IRREGULAR tables first (highest priority).
    2. Try splitting off prefixes.
    3. Try splitting off suffixes (longest first).
    4. If residue is in known_roots, accept the split.
    5. Else, return surface as-is.

    Returns list of morpheme tokens. Empty list = unknown word.
    """
    surface = surface.lower().strip()
    if not surface:
        return []

    # 1. Irregulars
    if surface in IRREGULAR_PAST:
        root, marker = IRREGULAR_PAST[surface]
        if root in known_roots:
            return [marker, root]
    if surface in IRREGULAR_PLURAL:
        root, marker = IRREGULAR_PLURAL[surface]
        if root in known_roots:
            return [marker, root]

    # 2. Surface is itself a known root
    if surface in known_roots:
        return [surface]

    # 3. Try suffix decomposition FIRST (longest first). Most surface
    # variation in English is suffix-based (running, dogs, bigger).
    # Doing suffix-first means "reading" -> [read, ing] before we
    # consider "re" as a prefix.
    for suffix in sorted(MORPHEMES_SUFFIX, key=len, reverse=True):
        if surface.endswith(suffix):
            root = surface[:-len(suffix)]
            # Common spelling repairs:
            #   running -> run+ning -> run+ing (drop doubled consonant)
            if len(root) >= 3 and root[-1] == root[-2]:
                root_alt = root[:-1]
                if root_alt in known_roots:
                    return [root_alt, suffix]
            #   bigger -> big+ger -> big+er
            if suffix == "er" and len(root) >= 3 and root[-1] == root[-2]:
                root_alt = root[:-1]
                if root_alt in known_roots:
                    return [root_alt, suffix]
            #   tries -> try+ies -> try (drop ies, add y)
            if suffix in ("ies", "es") and root and not root.endswith("y"):
                root_alt = root + "y"
                if root_alt in known_roots:
                    return [root_alt, suffix]
            if root in known_roots:
                return [root, suffix]

    # 4. Try prefix decomposition (only after suffix failed)
    for prefix in sorted(MORPHEMES_PREFIX, key=len, reverse=True):
        if surface.startswith(prefix):
            residue = surface[len(prefix):]
            if residue in known_roots:
                return [prefix, residue]
            # Recursive: residue might also have suffix (e.g. "unhappier"
            # -> "un" + "happier" -> "un" + "happy" + "er"). Only accept
            # if recursive decomp produced KNOWN tokens, not raw residue.
            sub = tokenize_word(residue, known_roots)
            if sub and all(t in known_roots
                            or t in MORPHEMES_PREFIX
                            or t in MORPHEMES_SUFFIX
                            or t in ("PAST", "PLURAL")
                            for t in sub):
                return [prefix] + sub

    # 5. Unknown
    return [surface]


def tokenize_sentence(sentence: str, known_roots: set) -> List[str]:
    """Tokenize a sentence into a flat sequence of morphemes.

    Splits on whitespace, then decomposes each word. Stopwords pass
    through unchanged.
    """
    tokens = []
    for word in sentence.split():
        # Strip punctuation
        word_clean = re.sub(r"[^\w']", "", word).lower()
        if not word_clean:
            continue
        morphemes = tokenize_word(word_clean, known_roots)
        tokens.extend(morphemes)
    return tokens


# Built-in set: 60-word multi-bridge vocab roots + grammar morphemes
DEFAULT_ROOTS_60 = {
    "apple", "river", "dog", "cat", "go", "come", "stop", "look",
    "big", "small", "hot", "cold",
    "tree", "bird", "sun", "moon", "walk", "run", "eat", "sleep",
    "red", "blue", "fast", "slow",
    "house", "road", "fire", "water", "give", "take", "find", "lose",
    "tall", "short", "wet", "dry",
    "person", "baby", "ball", "key", "open", "close", "push", "pull",
    "happy", "sad", "full", "empty",
    "food", "drink", "hand", "foot", "speak", "listen", "read", "write",
    "new", "old", "clean", "hard",
    "north", "south", "east", "west",
}


def get_combinatorial_vocab_estimate(roots: set) -> dict:
    """Estimate how many surface words our morpheme rules can derive
    from a root set."""
    n_roots = len(roots)
    estimate = {
        "n_roots": n_roots,
        "bare_roots": n_roots,
        "with_plural": n_roots * 2,  # +s
        "with_past_tense": n_roots * 2,  # +ed / irregular
        "with_continuous": n_roots * 2,  # +ing
        "with_comparative": n_roots * 2,  # +er
        "with_negation": n_roots * 2,  # un-X
        "combined_max": n_roots * 6,  # approx; many invalid
    }
    return estimate


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sentence", type=str, default=None)
    p.add_argument("--show-vocab-estimate", action="store_true")
    args = p.parse_args()

    if args.show_vocab_estimate:
        est = get_combinatorial_vocab_estimate(DEFAULT_ROOTS_60)
        print("Combinatorial vocab estimate from 64 roots:")
        for k, v in est.items():
            print(f"  {k}: {v}")
        print()

    if args.sentence:
        tokens = tokenize_sentence(args.sentence, DEFAULT_ROOTS_60)
        print(f"Input:  '{args.sentence}'")
        print(f"Tokens: {tokens}")
    else:
        # Built-in demo
        sentences = [
            "the dog ran fast",
            "the dogs are running",
            "she ate the apple",
            "I see two apples",
            "the trees are bigger than the houses",
            "the unhappy person is reading",
            "the babies are sleeping",
            "the keys are smaller than the balls",
        ]
        for s in sentences:
            tokens = tokenize_sentence(s, DEFAULT_ROOTS_60)
            print(f"  '{s}'")
            print(f"    -> {tokens}")
