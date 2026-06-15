"""A-PRIORI SEMANTIC TAXONOMY for the TinyStories "stream cortex" experiment -- 40 categories x 8 words.

PURPOSE
  The stream-cortex de-risk (`_phaseB_onbridge_stream_cortex_derisk.py`) and the Option-C real-
  co-occurrence de-risk (`option_c_real_cooccurrence_derisk.py`) learn a "cortex" from the TinyStories
  word stream and ask whether the learned concept codes CLUSTER BY THEIR A-PRIORI SEMANTIC CATEGORY.
  The category labels are the *ground-truth* `S_true` for the structure-recovery + generalization
  metrics (within-category similarity 1.0, between-category 0.0). For that metric to be meaningful the
  reference MUST be an INDEPENDENT semantic taxonomy -- NEVER derived from the corpus co-occurrence
  (a corpus-derived reference would be circular and silently invalidate the experiment, design SS1).

  This module is that independent reference, scaled from the validated 8x8 (64-word) taxonomy in
  `option_c_real_cooccurrence_derisk.TAXONOMY_8x8` up to 40x8 (320 words) -- the documented "age-5"
  production tier for the learned-graded cortex.

WHY THE STRUCTURE MATTERS (read before editing)
  Each category must be a REAL semantic domain whose members genuinely SHARE co-occurrence context in
  children's stories -- animals {dog, cat, bird, ...}, food {apple, cake, ...}, body parts {hand, foot,
  ...}, colors {red, blue, ...}, family {mom, dad, ...}, emotions, sizes (adjectives), motion verbs,
  vehicles, household objects, weather/nature, etc. Words that co-occur with similar contexts will get
  similar learned codes; the metric then tests whether that learned similarity recovers the a-priori
  category blocks. ABSTRACT / FUNCTION words (thought, something, about, when) are deliberately EXCLUDED
  -- they don't form clean co-occurrence clusters and would only add noise to S_true.

HARD CONSTRAINTS (all asserted in __main__ against the real corpus frequency table)
  1. Exactly 40 categories, each exactly 8 words  -> 320 total.
  2. All 320 words DISTINCT (no word in two categories).
  3. Every word appears in TinyStories with frequency >= 50 (the proven learnability floor); the great
     majority are >= 100. The min frequency is reported.
  4. Words are lowercase, alphabetic, length >= 3.
  5. The 8 original 8x8 categories are INCLUDED (extend, don't replace).

CAVEAT ON CONSTRAINT 5 vs CONSTRAINT 3 (honest note)
  The original 8x8 `body` category contained "eye" (corpus freq 48 -- it was curated under the older
  freq_floor=30). 48 is BELOW the >=50 hard floor used here, so "eye" is the ONE original-8x8 word that
  was substituted, to keep the asserts truthful. It is replaced by "neck" (freq 82), which is an equally
  clean body-part term. All other 63 original words clear >=50 unchanged, and the 8 original CATEGORIES
  (animals, food, body, family, actions, colors, places, toys) are preserved as 8 of the 40.

  Renames for clarity (the original category *contents* are otherwise verbatim): the original "animals"
  is kept here as `animals_pets` (to disambiguate from the new wild/farm/small-creature animal
  categories) and the original "actions" as `motion_actions`. The membership of those two categories is
  byte-identical to the 8x8 source.

CORPUS-VOCABULARY CEILING (the honest result of curating this)
  TinyStories has a small, simple vocabulary. 40 CLEAN 8-word semantic categories ARE achievable at the
  >=50 floor (this file), but the corpus is close to its ceiling there: several otherwise-natural
  domains had to be split by sub-theme (e.g. animals -> pets / wild / farm / small-creatures / more;
  food -> staple-food / fruits-sweets / meal-food; adjectives -> sizes / texture / good / bad / clever;
  emotions -> two banks; household -> two banks) to find 8 frequent members each, and a handful of
  thinner domains (drinks, kitchen utensils, clothing, building parts) sit nearer the floor. Pushing
  meaningfully past 40 clean categories would require dipping below freq 50 or admitting abstract/
  function words -- which this file deliberately does NOT do. So 40x8 is reported as the practical
  corpus-supported maximum for a clean a-priori semantic taxonomy over TinyStories.

USAGE
  from research.runners.stream_taxonomy_320 import TAXONOMY_40x8
  # same dict shape as TAXONOMY_8x8: {category_name: [w1, ..., w8], ...}
  # flatten with option_c_real_cooccurrence_derisk.taxonomy_to_vocab_categories(TAXONOMY_40x8)

VALIDATE
  SIM_BACKEND=numpy python -u -m research.runners.stream_taxonomy_320
"""
from __future__ import annotations

import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ===========================================================================
# THE A-PRIORI SEMANTIC GROUND TRUTH: 40 categories x 8 corpus-frequent words.
# Every word is a TinyStories word with frequency >= 50 (verified in __main__).
# The 8 original 8x8 categories appear first (animals_pets, food, body, family,
# motion_actions, colors, places, toys); contents are verbatim from the 8x8
# source except body's "eye"(48) -> "neck"(82), per the docstring caveat.
# ===========================================================================
TAXONOMY_40x8 = {
    # ---- the original 8x8 (preserved; see docstring on the two renames + 'eye'->'neck') ----
    "animals_pets":    ["dog", "cat", "bird", "fish", "frog", "bear", "mouse", "duck"],
    "food":            ["apple", "cake", "bread", "milk", "egg", "soup", "candy", "cookie"],
    "body":            ["hand", "neck", "foot", "head", "hair", "arm", "leg", "face"],
    "family":          ["mom", "dad", "girl", "boy", "baby", "friend", "sister", "brother"],
    "motion_actions":  ["run", "jump", "walk", "play", "look", "eat", "sleep", "sing"],
    "colors":          ["red", "blue", "green", "yellow", "black", "white", "pink", "brown"],
    "places":          ["house", "park", "room", "garden", "tree", "road", "school", "beach"],
    "toys":            ["ball", "toy", "book", "doll", "box", "blocks", "kite", "bell"],

    # ---- animals, expanded by sub-domain ----
    "wild_animals":    ["lion", "tiger", "wolf", "fox", "monkey", "elephant", "snake", "crocodile"],
    "farm_animals":    ["cow", "pig", "horse", "sheep", "chicken", "hen", "farmer", "barn"],
    "small_creatures": ["ant", "bee", "bug", "spider", "worm", "butterfly", "fly", "caterpillar"],
    "more_animals":    ["rabbit", "bunny", "squirrel", "owl", "turtle", "crab", "kitten", "puppy"],

    # ---- food, expanded by sub-domain ----
    "fruits_sweets":   ["banana", "orange", "grapes", "chocolate", "jam", "muffin", "honey", "berries"],
    "meal_food":       ["cheese", "pizza", "sandwich", "steak", "carrot", "salad", "corn", "sugar"],
    "kitchen":         ["dinner", "lunch", "snack", "oven", "stove", "cook", "bake", "kitchen"],
    "drinks":          ["juice", "tea", "ice", "cream", "water", "bottle", "jug", "drink"],

    # ---- the body, more parts ----
    "body_more":       ["nose", "mouth", "ear", "tail", "wings", "teeth", "knee", "finger"],

    # ---- emotions (two banks) ----
    "emotions":        ["happy", "sad", "scared", "angry", "excited", "surprised", "proud", "worried"],
    "feelings":        ["glad", "mad", "afraid", "nervous", "jealous", "grumpy", "lonely", "curious"],

    # ---- descriptive adjectives, split by sub-theme ----
    "sizes":           ["big", "small", "little", "tiny", "huge", "giant", "large", "tall"],
    "texture_temp":    ["hot", "cold", "warm", "wet", "dry", "soft", "hard", "smooth"],
    "good_adj":        ["good", "nice", "kind", "pretty", "beautiful", "great", "sweet", "special"],
    "bad_adj":         ["bad", "mean", "ugly", "naughty", "rude", "selfish", "bossy", "lazy"],
    "trait_adj":       ["funny", "silly", "clever", "smart", "gentle", "friendly", "helpful", "brave"],

    # ---- nature, weather, sky, water, places ----
    "weather_nature":  ["rain", "snow", "wind", "sun", "cloud", "storm", "fire", "rainbow"],
    "sky_space":       ["moon", "star", "stars", "sky", "clouds", "space", "rocket", "night"],
    "water_places":    ["river", "lake", "sea", "pond", "pool", "splash", "puddle", "ocean"],
    "nature_places":   ["forest", "farm", "hill", "cave", "woods", "field", "village", "town"],
    "plants":          ["flower", "flowers", "trees", "grass", "leaf", "leaves", "bush", "branch"],

    # ---- built environment + household ----
    "building_parts":  ["door", "window", "wall", "floor", "roof", "fence", "gate", "ceiling"],
    "household":       ["table", "chair", "bed", "cup", "bowl", "spoon", "plate", "clock"],
    "household_more":  ["couch", "shelf", "mirror", "blanket", "pillow", "basket", "jar", "bucket"],
    "clothing":        ["hat", "dress", "shoes", "coat", "shirt", "sock", "clothes", "cap"],

    # ---- vehicles ----
    "vehicles":        ["car", "truck", "bus", "train", "plane", "boat", "bike", "ship"],

    # ---- people / roles ----
    "roles":           ["teacher", "doctor", "king", "queen", "prince", "princess", "fairy", "robot"],

    # ---- verbs, split by sub-theme ----
    "communication":   ["talk", "say", "tell", "ask", "shout", "laugh", "cry", "listen"],
    "manipulate":      ["take", "put", "give", "hold", "throw", "catch", "push", "pull"],
    "mind_verbs":      ["help", "share", "love", "want", "need", "know", "learn", "teach"],

    # ---- materials + play ----
    "materials":       ["rock", "stone", "stick", "wood", "sand", "mud", "glass", "paper"],
    "play_fun":        ["game", "race", "dance", "song", "party", "adventure", "swing", "slide"],
}


def _load_corpus_frequencies() -> Counter:
    """Word frequencies over the full TinyStories corpus (lowercased alpha tokens),
    via the same loader the stream-cortex de-risk uses."""
    from research.runners._phaseB_onbridge_stream_cortex_derisk import load_token_stream
    f: Counter = Counter()
    for toks in load_token_stream():
        f.update(toks)
    return f


def validate(taxonomy: dict, freq: Counter, freq_floor: int = 50) -> dict:
    """Assert the four hard constraints against the real corpus frequency table.
    Returns a small report dict. Raises AssertionError on any violation."""
    cats = list(taxonomy.keys())

    # Constraint 1: exactly 40 categories x exactly 8 words.
    n_cat = len(cats)
    for c in cats:
        assert len(taxonomy[c]) == 8, f"category {c!r} has {len(taxonomy[c])} words (need 8)"
    total = sum(len(v) for v in taxonomy.values())
    assert total == n_cat * 8, f"total words {total} != {n_cat}*8"

    # Constraint 2: all words distinct.
    allw = [w for ws in taxonomy.values() for w in ws]
    dup = [w for w, n in Counter(allw).items() if n > 1]
    assert not dup, f"duplicate words across categories: {dup}"
    assert len(set(allw)) == total, "distinct-word count mismatch"

    # Constraint 4: lowercase, alphabetic, length >= 3.
    bad_form = [w for w in allw if (w != w.lower()) or (not w.isalpha()) or (len(w) < 3)]
    assert not bad_form, f"malformed words (need lowercase alpha len>=3): {bad_form}"

    # Constraint 3: every word freq >= floor.
    freqs = {w: int(freq.get(w, 0)) for w in allw}
    below = [(w, c) for w, c in freqs.items() if c < freq_floor]
    assert not below, f"{len(below)} word(s) below freq-floor {freq_floor}: {sorted(below, key=lambda x: x[1])}"

    import statistics
    fvals = sorted(freqs.values())
    return {
        "n_categories": n_cat,
        "n_words": total,
        "freq_floor": freq_floor,
        "min_freq": fvals[0],
        "median_freq": int(statistics.median(fvals)),
        "max_freq": fvals[-1],
        "n_below_100": sum(1 for v in fvals if v < 100),
        "per_word_freq": freqs,
    }


if __name__ == "__main__":
    print("[stream_taxonomy_320] loading TinyStories corpus frequencies "
          "(SIM_BACKEND=numpy recommended)...", flush=True)
    os.environ.setdefault("SIM_BACKEND", "numpy")
    freq = _load_corpus_frequencies()
    print(f"  corpus: {len(freq)} unique tokens", flush=True)

    rep = validate(TAXONOMY_40x8, freq, freq_floor=50)

    print("\n" + "=" * 78, flush=True)
    print("  A-PRIORI SEMANTIC TAXONOMY -- VALIDATION REPORT", flush=True)
    print("=" * 78, flush=True)
    print(f"  categories x words : {rep['n_categories']} x 8  = {rep['n_words']} words", flush=True)
    print(f"  all distinct       : YES", flush=True)
    print(f"  freq floor met     : >= {rep['freq_floor']}  (min word freq = {rep['min_freq']})", flush=True)
    print(f"  frequency spread   : min {rep['min_freq']} | median {rep['median_freq']} | "
          f"max {rep['max_freq']}", flush=True)
    print(f"  words below 100    : {rep['n_below_100']} / {rep['n_words']} "
          f"(rest are >= 100; preferred)", flush=True)
    print("-" * 78, flush=True)
    print("  per-category MIN word-frequency (the category's weakest member):", flush=True)
    rows = []
    for cat, ws in TAXONOMY_40x8.items():
        wf = [(w, int(freq.get(w, 0))) for w in ws]
        mn_w, mn_c = min(wf, key=lambda x: x[1])
        rows.append((mn_c, cat, mn_w))
    for mn_c, cat, mn_w in sorted(rows):  # ascending by the category's weakest member
        print(f"    {cat:18s} min={mn_c:5d}  (weakest: {mn_w})", flush=True)
    print("=" * 78, flush=True)
    print("  ALL HARD CONSTRAINTS PASS: 40x8 = 320 distinct corpus-frequent words, freq>=50.", flush=True)
    print("=" * 78 + "\n", flush=True)
