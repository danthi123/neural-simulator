# Step-2a: spiking visual word recognition off V1_simple — honest characterization (2026-06-02)

**Context.** The input-side-fidelity insight (owner side-chat, validated 4 ways by cheap probes)
says language should be *transduced* as pixels through the existing visual pathway (earned,
shared-structure, data-efficient) instead of *tokenized* into given orthogonal codes. The
production build (`research/runners/text_visual_grounding.py`) realizes this on the GPU: a
region-framework bridge `retina -> V1_simple -> V1_complex -> V2 -> IT` with scaled Gabor V1
weights, reading rendered words as pixels.

This doc records step-2a: reading earned word recognition off the working V1_simple layer via a
plastic STDP pathway (`V1_simple -> word_pool`, teacher-supervised), and the honest limits found.

## What works (verified on GPU, RTX 3090)

- **Construction + transduction.** retina 64: 49,472 neurons, 13.7M synapses, builds ~80s.
- **Per-layer firing diagnostic** (retina 32, drive 2500 pA): retina 0.23, **V1_simple 0.03**.
  The retina -> V1_simple transduction *faithfully responds to rendered words*. This is the
  tokenizer-replacement: words enter as pixels through earned visual transduction, not given
  orthogonal codes. **The owner's input-side-fidelity fix is live on the GPU.**

## Diagnosed cascade gap (V1_complex)

The hierarchy does not propagate past V1_simple for text: V1_complex 0.005, V2/IT ~0. Root cause
is **V1_complex starvation**: text is *sparse* (thin letter strokes -> V1_simple fires 0.03)
whereas the g11 gridworld this pathway was tuned for shows *dense blocks* (many coincident V1s
spikes). The g11 random-density phase-pooling (weight 2.0) rarely gets coincident V1s spikes from
sparse text, so V1_complex stays silent and V2/IT are dead downstream. Strengthening the pooling
(weight 20, 4x density) lifted V1_complex to 0.022 for the strongest word but the full cascade to
IT still did not propagate -> this is multi-knob engineering (structured phase-pooling + V2/IT
inhibition + scale), not a one-line fix. Per the debugging iron law (reassess after 3 attempts),
stopped tuning and read recognition off the working V1_simple layer instead.

## Recognition off V1_simple — the ceiling

Teacher-supervised STDP (`train_word_to_pool` pattern reused from `concept_pool_demo`): drive
retina(word) -> V1_simple word-form fires; drive the target word-pool with teacher current;
STDP on the open-gated `V1_simple -> target-pool` pathway binds the word-form to the pool.
Interleaved events, one gate open at a time (isolated per-word training). Test: drive retina(word),
no teacher, the highest-firing pool is the recognition.

| Readout | Vocab | retina | result | chance |
|---|---|---|---|---|
| whole-word pools | dog,cat,run,sun | 32 | **1/4 = 0.25** | 0.25 |
| single-letter pools | a,e,o,t,x | 32 | **2/5 = 0.40** | 0.20 |

The single-letter 0.40 sits right at the **V1-simple-readout ceiling** the cheap scaled-Gabor probe
independently found (retina 64 = 0.37). Mechanism: V1 *simple* cells are position-specific and do
not build invariant object/word representations; a whole-glyph pool-argmax over sparse spike-counts
loses most of the structure. The cheap probe's **0.91** came specifically from *per-position letter*
readout (compositional — read each letter band, compose the word) on continuous Gabor features with
a trained per-position classifier — a fundamentally different, compositional readout.

## Conclusion + two clear paths

Reading recognition off the spiking V1_simple layer with simple pools is noise/ceiling-limited
(~0.40 single-letter, chance whole-word). This is consistent + honest: the faithful invariant
recognition is not in V1_simple. Two well-specified paths to a faithful spiking word recognizer:

1. **Full V1 -> V2 -> IT hierarchy** (the biologically faithful object-recognition route): fix the
   V1_complex propagation with *structured* phase-pooling complex cells (Hubel-Wiesel quadrature
   pairs, not random density) + bigger retina/bolder text (more V1s activity; owner: "no reason to
   limit retina to 32x32") + V2/IT inhibition tuning, so IT builds invariant word-form
   representations and the recognition reads off IT. Grounded in Riesenhuber-Poggio HMAX /
   DiCarlo IT object recognition.

2. **Per-position letter-composition pools** (the validated 0.91 architecture, in spiking): pools
   per (position, letter); read each letter band of V1_simple; compose into a word. Open-vocabulary
   + data-efficient (learn ~L letters -> read L^n words). Needs bigger retina (each letter band
   well-resolved) + temporal integration (denoise sparse spikes).

**Decisive cheap experiment (in flight):** does bigger retina (64) + long temporal-integration
window (200 steps) + reduced pool inhibition lift single-letter recognition above the 0.40 ceiling?
If yes -> path 2 (letter-composition) is viable without the full hierarchy. If still ~0.40 -> the
ceiling is fundamental to V1-simple-readout and path 1 (full hierarchy) is genuinely required.
[VERDICT APPENDED BELOW WHEN THE RUN LANDS.]

## Honest scope

The input-side-fidelity *science* is validated 4 ways (cheap probes) and the *transduction* is live
on the GPU. A faithful spiking *recognizer* is the well-specified next sub-arc (path 1 or 2 above),
not a one-session tune. No shortcuts; biology-faithful; both remotes.
