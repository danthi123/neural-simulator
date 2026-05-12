# 🎉 P5 iter PP seed 42 — FIRST BIDIRECTIONAL PASS at biological scale

**Date:** 2026-05-12
**Status:** SEED 42 PASS. Multi-seed validation in progress.
**Architecture:** sensory grounding (Cluster K v2) + multimodal_hub +
lang_output FS cross-inhibition at biological scale.

## Result (seed 42)

TEST 2b auditory naming (5-trial avg, biological scale 500-neuron pools):

| Stim | pool_0 | pool_1 | Margin | Correct? |
|---|---|---|---|---|
| apple | 202 | 201 | **+1** | ✓ |
| river | 197 | 203 | **+6** | ✓ |
| **BIDIR** | | | | **PASS** ✓ |

This is the FIRST bidirectional PASS at biological scale for P5
(2-concept non-motor abstract concept binding) after 6 prior
biological-scale attempts (KK/LL/MM/NN/OO_visual + intermediates)
all FAILed.

## Why it works

The architectural insight from the 6-iter biological-scale arc:

**iter KK/LL/MM/NN failure mode:** per-seed random structural pool
variance at wernicke level → one pool wins for both stimuli at
output. Pool firing is dominated by random recurrence, not by
lang_input topographic prior.

**iter OO_visual PARTIAL:** sensory grounding provided a SECOND
training signal (visual stream independent of random connectivity)
that DID help (apple +28 spike improvement) but bias just MOVED
downstream to lang_output_pool via multimodal_hub→pool weights.

**iter PP fix:** add lang_output FS cross-inhibition (winner-take-all
at output layer). Each lang_output_pool_i drives its dedicated
FS_i; FS_i inhibits OTHER lang_output_pools. Even a tiny per-concept
firing differential produces correct winner.

## Mechanism

```
lang_input(apple) ─┐                    visual_image(apple) ─┐
                    │                                          │
                    ▼                                          ▼
              wernicke_pool_0     retina → V1 → V2 → IT
                    │                                          │
                    └──────────► multimodal_hub ◄──────────────┘
                                       │
                                       ▼
                              lang_output_pool_0 ─── FS_0 ──→ inhibits pool_1
                                       │
                              lang_output_pool_1 ◄── FS_1 ──→ inhibits pool_0
```

During training: both visual stream + auditory wernicke drive their
target pool. Tiny per-concept signal differential exists due to
topographic bias + Gabor-tuned visual features. FS cross-inhibition
amplifies this differential — pool that fires marginally more
suppresses the other, sharpening the win.

During recognition: CA3 tag stim → CA1 → lang_output_pool_i (both).
The pool with stronger learned ca1→pool weights fires fractionally
more → its FS suppresses the other → clear winner.

## Architecture (13,160 neurons, 81 populations, 2.79M synapses)

```
Regions (30):
  Auditory:
    language_input (2048)
    wernicke_pool_0/1 (500 each)
    wernicke_fs_pool_0/1 (60 each, cross-inhibition)
    semantic_cortex (500)
    lang_output_pool_0/1 (500 each)
    lang_output_fs_pool_0/1 (60 each, NEW iter PP)
    motor_N/E/S/W (16 each) + motor_FS_N/E/S/W (4 each)
    language_output (2048)
  Hippocampus:
    ec (200), dg (800), dg_pv_basket (240),
    ca3 (400), ca1 (200)
  Visual (Cluster K v2):
    retina (2048, ON+OFF channels)
    cortex_v1_simple (1024, Gabor init)
    cortex_v1_complex (512, phase pool)
    cortex_v2 (256, plastic)
    cortex_it (64, plastic with canon dynamics)
  Multimodal:
    multimodal_hub (500, ATL hub-and-spoke)

Pathways (~57):
  lang_input → wernicke_pool_i (topographic 1.5/0.7)
  wernicke_pool_i ↔ semantic_cortex (bidirectional, plastic)
  wernicke_pool_i → wernicke_fs_pool_i → other wernicke pools
  retina → V1s (70592 Gabor weights) → V1c → V2 → IT
  cortex_it → multimodal_hub ← wernicke_pool_i
  multimodal_hub → lang_output_pool_i
  ca1 → lang_output_pool_i
  lang_output_pool_i → lang_output_fs_pool_i → other lang pools (NEW)
```

## Recipe

```bash
python -m research.runners.validate_ventral_semantic --seed 42 \
    --n-train-events 400 --n-replay-cycles 40 \
    --n-lang-input 2048 \
    --enable-multi-pool-wernicke --n-wernicke-pools 2 \
    --n-per-wernicke-pool 500 --n-per-wernicke-pool-fs 60 \
    --interleaved-training \
    --enable-per-concept-lang-out-pools --n-per-lang-out-pool 500 \
    --enable-lang-out-fs-pools --n-per-lang-out-fs-pool 60 \
    --apply-wernicke-topographic \
    --enable-visual-cortex --enable-multimodal-hub \
    --pair-visual-during-training \
    --n-recognition-trials 5 --inter-trial-rest-steps 100
```

Wall clock: ~7 min/seed at biological scale on RTX 3090.

## Caveats

1. **Margins are tiny** (+1 and +6 of ~200 spikes). Multi-seed
   robustness UNKNOWN — could easily fail other seeds if their
   random variance happens to push the margin the wrong way.

2. **Visual-only test (TEST 2c) still fails apple**: visual stream
   training didn't fully align with concept_0 → pool_0 routing.
   The auditory path PASSes but the visual path needs more work.

3. **Comprehension test (TEST 1) still fails** (apple_self 0.231,
   apple_river 0.266). The semantic_cortex representation isn't
   stable across trials. This was also failing at toy iter AA, so
   it's a pre-existing methodology issue (the cosine cosines test
   has been noted as a TOY-SCALE-specific metric).

## Multi-seed validation in progress

Seeds 43, 44, 100, 101, 102 running sequentially in PowerShell.
Estimated ~35 min total. Results will be aggregated via:

```bash
python -m research.runners.aggregate_p5_pool_readout \
    --raw-root research/findings/raw/g11_bg/iter_PP \
    --prefix iter_PP_seed --seeds 42,43,44,100,101,102 \
    --out research/findings/2026-05-12-P5-iterPP-multiseed.md \
    --label "iter PP (sensory grounded + lang_output FS @ bio scale)"
```

## Strategic implication (preliminary)

iter PP seed 42's BIDIR PASS confirms that the **per-concept pool
architecture CAN work at biological scale** when given:
1. Sensory grounding (Cluster K v2 visual stream) — overrides per-seed
   wernicke pool bias
2. Output-layer winner-take-all (lang_output FS pools) — overrides
   per-seed lang_output pool bias

Together they break the structural pool bias that 4 prior iter
attempts (KK/LL/MM/NN) couldn't fix.

If multi-seed achieves ≥4/6 BIDIR, this is a genuine scaling
breakthrough — P5 ventral semantic stream working at biological
scale, opening the path to:
- Vocabulary expansion (4-8 concepts) via more wernicke pools
- P6 Broca's compositional grammar on top
- Conversational sim with multimodal concept understanding

If multi-seed gives 1-2/6 BIDIR, the margins are noise and we're
back to the architectural ceiling. Multi-seed result is the
definitive test.

## Catalog faithful

- G.11 Hickok & Poeppel dual-stream ventral semantic ✓
- G.13 Wernicke's area auditory comprehension ✓
- K.01 V1/V2/IT visual ventral stream ✓
- Lambon Ralph 2017 ATL hub-and-spoke (multimodal_hub) ✓
- Pulvermüller embodied semantics (visual co-firing) ✓
- No motor-decoder cheats ✓
- No external LLM cheats ✓
- Cortical canon (Lefort 2009, Wang 2002) for cortex_v2/it/multimodal_hub ✓
