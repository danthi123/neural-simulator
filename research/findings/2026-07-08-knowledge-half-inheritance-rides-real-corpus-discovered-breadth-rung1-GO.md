# The KNOWLEDGE half of breadth, rung 1 (GO, 6-seed): property inheritance rides categories DISCOVERED from a real broad corpus — a held-out member of a TinyStories-discovered category inherits a property taught only to OTHER members, and label-derangement collapses it to chance. "Discover a broad vocab from a real corpus → reason about it." NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_inheritance_rung1_derisk.py` (reuse-by-import: the emergent breadth discovery — `discover_vocab` + `learn_stream_codes` from `_emergent_vocab_breadth_scale_derisk.py`; the EMERGE-30/42 inheritance mechanism — teach a class property to some members, a held-out member inherits via the shared discovered category structure). numpy-only, offline. NO `sim/` edit.
**Verdict:** rung-1 GO (rate-level). The vocab-STRUCTURE half of breadth was de-risked to 1024-word real-corpus scale (`2026-07-08-open-domain-breadth-is-a-data-scale-lever-...`); this shows the KNOWLEDGE half (reasoning over the broad vocab) rides that same discovered structure.

## Why this ran (the mission-central next step)
Discovering a broad vocab's category STRUCTURE from a real corpus (the breadth thread) is only useful for a talkable brain if the brain can then REASON over it — inherit properties, answer questions never explicitly told. EMERGE-30/42 proved inheritance rides categories discovered from SYNTHETIC co-occurrence streams; the open question was whether it rides categories discovered from a REAL broad corpus. That is the KNOWLEDGE half: "discover a broad vocab from a real corpus → reason about it → (speak it — EMERGE-59..73)."

## The mechanism (rate-first, the cheapest rung; the spiking wire-in reuses EMERGE-42)
- **codes** = the learned co-occurrence codes for the top-K TinyStories vocab (the emergent stream mechanism — the exact codes the breadth thread validated).
- A discovered **category** C = the a-priori-category probe words present in the vocab (distinct concepts, e.g. animals = {dog, cat, bird, fish, frog, bear, mouse}). Split into TAUGHT (half) + HELD-OUT (half) — held-out members are NEVER taught.
- **TEACH** each category its OWN distinct property vector via a Hebbian associative memory `M = Σ_taught outer(unit(code_m), property_{cat(m)})` — one memory over all categories' taught members, held-out members excluded.
- **INHERIT** for a held-out query q: `p̂ = unit(code_q) · M`; the recalled property argmaxes to the category whose taught members q's code is most similar to. Inheritance is CORRECT iff a held-out member's recalled property argmaxes to its OWN category (nearest-taught-category by code similarity — genuine distributed inheritance via shared category structure).

## The result — 6-seed (42/43/44/100/101/102), TinyStories (3.9M tokens)
| scale | held-out inherit-acc | **label-deranged (primary control)** | scrambled (secondary) | mem-ceiling | chance |
|---|---|---|---|---|---|
| **K=256** (4 categories, 13 held-out) | **0.756 ± 0.066** (~3.0× chance) | **0.243** (≈ chance, all seeds) | 0.653 | 1.000 | 0.250 |
| **K=1024** (8 categories, 31 held-out) | **0.656 ± 0.052** (~5.2× chance) | **0.128** (≈ chance, all seeds) | 0.387 | 1.000 | 0.125 |

**GO at BOTH breadth scales** (`beats_chance` and `beats_deranged` by ≥0.15 margin, every seed). A held-out member of a real-corpus-discovered category inherits its category's property ~3–5× above chance, and the **label-derangement control collapses to chance every seed** — the GENUINE discovered category carries the inheritance, not any random grouping. The broader K=1024 test (8 categories, 31 held-out, lower chance) is even stronger as a multiple of chance.

## The anti-cheat story (a control corrected by reading the substance)
- **PRIMARY control = LABEL-DERANGEMENT** (reassign the same words to RANDOM categories of the same sizes, teach+test on the REAL codes): collapses to chance (0.243 / 0.128) every seed → the inheritance rides the ACTUAL discovered category assignment, not an artifact of a few dominant codes. This is the EMERGE-30/32 category-derangement anti-cheat.
- **SECONDARY control = scrambled-corpus** (within-story token shuffle): does NOT collapse (0.653 / 0.387). Diagnosed honestly: within-STORY shuffling destroys word ORDER but PRESERVES story-level bag-of-words co-occurrence, and TinyStories category structure (animals co-occur in animal stories) is bag-of-words-driven — so the scramble barely dents category structure. Scramble is the correct anti-cheat for ORDER-dependent (syntactic) structure, the WRONG one for bag-of-words CATEGORY structure — which is exactly why label-derangement is the load-bearing control here.
- **No leakage:** held-out members never enter the associative memory (construction-verified). **Genuine generalization:** the probe words are DISTINCT concepts (dog/cat/bird/fish/frog/bear/mouse), not morphological variants — teaching {dog,cat,bird} and a held-out {fish,frog,bear,mouse} inheriting is real category generalization, not string similarity.

## Honest scope (it's rung 1)
Rate-level associative-memory read (the cheapest rung of the rate→spike ladder); the SPIKING wire-in (the EMERGE-42 competitive pooler + spiking inference over these real-corpus codes) is the next rung, reuse-by-import. The probe is the fixed `TAXONOMY_8x8` yardstick (4 usable categories at K=256, 8 at K=1024); the taught property is a synthetic target vector (the mechanism, not real facts yet). What is genuinely established: the category structure the emergent cortex discovers from a REAL broad corpus is rich enough that a held-out member inherits a class property via it — the prerequisite for reasoning + answering over a broad, real-corpus-discovered vocab.

## What this establishes + the next rung
The KNOWLEDGE half of breadth rides the real-corpus-discovered structure: discover a broad vocab from a real corpus (breadth thread, de-risked) → a held-out member inherits its category's property (this, rung 1). Combined with the vocab-structure result, the emergent cortex can DISCOVER a broad vocab from real experience AND REASON (inherit) over it. Next rungs: (2) the SPIKING inheritance (EMERGE-42 pooler over the real-corpus codes); (3) real property facts (mine class properties from the corpus, not a synthetic target); (4) wire into a conversational turn (EMERGE-59..73 speaks the inherited answer on spikes).

## Files
`research/runners/_realcorpus_inheritance_rung1_derisk.py`; 6-seed `research/findings/raw/_rc_inherit_s{42,43,44,100,101,102}.json` (K=256) + `_rc_inherit_k1024_s*.json` (K=1024). Prior: `2026-07-08-open-domain-breadth-is-a-data-scale-lever-emergent-vocab-scales-to-1024.md` (the vocab-structure half); `2026-07-02-emerge30-*` / EMERGE-42 (inheritance over synthetic-stream-discovered categories).
