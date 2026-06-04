# (v) Multi-modal grounding works: decorrelation unifies vision + language into one composition-ready codebook — 2026-06-04

**One line:** Grounding the VISUAL concepts (nouns) via the real V1 Gabor bank AND the abstract concepts
(verbs, adjectives) via the project's word encoder, in ONE codebook, then decorrelating (ventral-hierarchy
stand-in), gives the full unified-agent benchmark at **100% (78/78, 2 seeds) — constructed parity** — while the
raw mixed-modality codebook degrades to 66.7% exactly where composition falls within a single high-coherence
modality. So the path to grounding the *whole* vocabulary is real: vision grounds visual concepts, language grounds
abstract ones, and the decorrelating hierarchy maps both into a unified low-coherence concept space where
composition operates modality-agnostically.

## Setup

`research/runners/unified_agent_multimodal_grounded.py` (reuse-by-import). The 320 benchmark concepts are grounded
by modality: the 200 **nouns** → real V1 Gabor responses (`sim/visual_cortex.py`, distinct synthetic visual
stimuli); the 60 **verbs** + 60 **adjectives** → the word encoder (`sim.text_embeddings.vocab_to_drive_pattern`,
the SHA-256-hashed sparse word code — the grounded word-cue level already validated). The two modalities occupy
disjoint blocks of a combined (V1 8192 + word 2048 = 10240)-dim feature matrix, optionally ZCA-decorrelated,
projected to phases, and fed as the agent's `external_codes`. Run raw vs decorrelated (mirroring #4).

## Result (2 seeds, D=2048)

| codes | flat | 1-attr | 2-attr | clause-d1 | clause-d2 | who | abstain | overall |
|---|---|---|---|---|---|---|---|---|
| RAW mixed-modality | 100% | **0%** | **0%** | **60%** | 100% | 100% | 100% | 66.7% |
| **DECORRELATED mixed-modality** | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** | **78/78 = 100%** |

(Single-modality V1-grounded + decorrelate was 92.3% in #4 — measured before the clause-depth2 fix; with that fix
both are at full parity.)

## What it shows

1. **A mixed-modality codebook composes at constructed parity once decorrelated.** The agent is modality-agnostic
   at the concept level: it doesn't care that "dog" came from a V1 receptive-field response and "chase" came from a
   word code — only that the codes are distinct and low-coherence. Decorrelation (the ventral stream's efficient
   coding) delivers exactly that for *both* modalities at once.
2. **Raw mixed-modality degrades precisely where composition is within one high-coherence modality.** Attribute
   composition (1-/2-attribute) collapses to 0% raw because the adjectives all live in the word block and the
   hashed word encoder's inter-code coherence drowns the resonator; clause-depth1 drops to 60% from the V1 nouns'
   coherence in the recursive decode. Retrieval (flat/who/abstain) and clause-depth2 survive (single-cleanup or
   already-flat-biased). This is the same coherence-blocks-composition mechanism as #4's single-modality result.

## The honest reading

Decorrelation orthonormalizes the combined codebook, so it largely *erases* the per-modality similarity structure
— the unified concept codes are low-coherence regardless of where they came from. So "multi-modal grounding works"
means precisely: **the decorrelating hierarchy maps any grounding modality's features to a unified, composition-
ready concept code.** The grounding is genuine (each code is a fixed function of its modality's sensory/lexical
features); decorrelation is what makes the mixed set composable and unified. This closes the multi-modal target #4
flagged (abstract words have no canonical image → ground them by language instead, visual concepts by vision, and
unify by decorrelation) — the path to grounding the full vocabulary, not just the visual subset.

## Files

- `research/runners/unified_agent_multimodal_grounded.py` — nouns→V1, verbs+adjs→word encoder, block-padded,
  raw vs decorrelated benchmark.
- `research/findings/raw/multimodal_grounded.json` — raw per-category-per-seed.
