# P5 iter OO_visual PARTIAL — sensory grounding HELPS apple but bias moves downstream

**Date:** 2026-05-12
**Status:** PARTIAL. Sensory grounding via Cluster K v2 + multimodal_hub
caused apple to FLIP from wrong (-5 margin) to CORRECT (+23 margin)
relative to iter LL/MM. But river also flipped — both stimuli now drive
pool_0. The structural pool bias moved DOWNSTREAM from wernicke_pool
to lang_output_pool via multimodal_hub→pool weights.

## Results

| Test | iter LL (no visual) | **iter OO_visual** (sensory grounded) |
|---|---|---|
| apple p0 | 218 | **233 (+15)** |
| apple p1 | 223 | **210 (-13)** |
| apple margin | **-5 (WRONG)** | **+23 (CORRECT, big flip)** |
| river p0 | 208 | 237 (+29) |
| river p1 | 216 | 213 (-3) |
| river margin | **+8 (correct)** | **-24 (WRONG, flipped)** |
| BIDIR | NO | NO |

Visual-only test (TEST 2c, new): drive ONLY retina, no lang_input
- visual apple p0=224 p1=213 (margin +11, pool_0 wins ✓)
- visual river p0=236 p1=215 (margin +21, pool_0 wins ✗ — wrong direction)

**Pool_0 wins for BOTH apple and river** under both audio and visual
stim. Discrimination signal (~12 spikes) is smaller than structural
bias toward pool_0 (~24 spikes).

## Architecture (28 regions, 13K neurons, 2.75M synapses, growing to 6.3M)

```
Auditory path (existing):
  lang_input → wernicke_pool_i → semantic_cortex (back & forth)
  lang_input → wernicke_pool_i → lang_output_pool_i

Visual path (NEW, Cluster K v2):
  retina(2048) → V1_simple(1024, Gabor init) → V1_complex(512)
  → cortex_v2(256, plastic) → cortex_it(64, plastic)

Multimodal hub (NEW, ATL hub-and-spoke):
  cortex_it → multimodal_hub(500) ← wernicke_pool_i
  multimodal_hub → lang_output_pool_i

Recognition pathways (existing for naming):
  ca1 → lang_output_pool_i (per-concept)
```

Training: lang_input(word) + retina(concept_image) co-fired per event.

## What worked (significantly)

**Apple discrimination FLIPPED from negative to positive by 28 spikes.**

| Iter | apple p0 | apple p1 | margin |
|---|---|---|---|
| AA (toy) | 92 | 85 | +7 ✓ |
| KK (bio canon) | 236 | 254 | -18 ✗ |
| LL (bio weak) | 218 | 223 | -5 ✗ |
| MM (bio strong topo) | 211 | 217 | -6 ✗ |
| NN (bio orthogonal) | 217 | 212 | +5 ✓ (small) |
| **OO_visual (bio + sensory)** | **233** | **210** | **+23 ✓ (big!)** |

The visual stream provided a SECOND signal that successfully tipped
apple toward pool_0. This is the principle Tier 1 motor binding uses
(motor teacher current) applied to abstract concepts (visual teacher).

## What didn't work

**River discrimination ALSO flipped — wrong direction.**

The structural bias moved from wernicke_pool → multimodal_hub →
lang_output_pool. Pool_0 now wins for BOTH stimuli because:

1. During training, pool_0 has random structural advantage at output
   layer (random init has pool_0 firing slightly more for any drive)
2. STDP at multimodal_hub → lang_output_pool_0 strengthens for ALL
   training events (apple AND river)
3. STDP at multimodal_hub → lang_output_pool_1 stays weaker
4. At test, ANY drive (audio OR visual, apple OR river) routes
   preferentially to pool_0

The visual signal added a strong DIFFERENTIAL between apple and river
at upstream layers (V1 → V2 → IT), but the downstream output bias
overwhelms this.

## Diagnosis: structural bias at LANG_OUTPUT_POOL is the true bottleneck

After 5 biological-scale iterations (KK/LL/MM/NN/OO_visual):
- Per-seed random structural variance creates pool dominance
- Bias can be at wernicke level (iter LL/MM), output level (iter NN),
  or output level even with sensory grounding (iter OO_visual)
- Discrimination signal is FIXED by topographic prior; bias scales
  with N (pool size) faster than signal does

**The architectural fix needs to address output-layer bias specifically.**

## Path forward (3 options, ranked)

### Option A: lang_output FS pools at biological scale (iter PP)
Add cross-inhibition at lang_output (winner-take-all enforced at
output). Mirror of Tier 1 motor_FS_X cross-inhibition that produces
6/6 motor binding. iter CC tried this at toy scale (traded errors)
but never tested at biological scale + multimodal training.

Effort: 30 min (CLI flag already exists, just enable).

### Option B: visual stream pretraining (iter PP_pretrain)
Train visual stream alone (V1→V2→IT) on concept images BEFORE
multimodal binding. Then unfreeze multimodal_hub binding. Mirrors
real developmental critical periods (Hubel & Wiesel).

Effort: ~3 hr (training phase split, ~2x compute per seed).

### Option C: per-concept multimodal_hub regions
Split multimodal_hub into multimodal_hub_apple + multimodal_hub_river
(pre-allocated, like wernicke_pool_i). Forces concept-specific routing.

Effort: 1 hr. But this is just MORE pre-allocation cheating — pushes
the architectural cheat one layer further.

## Recommendation

**Option A (lang_output FS at biological scale)** is the cheapest
test of the "output bias is the bottleneck" hypothesis. iter CC's
toy-scale regression may not predict biological-scale behavior since
the FS pool size (24 neurons) and cross-weight (4.0) may need
scaling.

If iter PP (Option A) fails too, the architectural ceiling is
fundamental and the recommended pivot is shipping iter AA's 4/6
toy-scale result and focusing future work on Tier 2.x synonym
expansion + P6 Broca's compositional grammar on direction vocab.

## Wall clock

iter OO_visual seed 42: ~6 min (much faster than iter LL/MM/NN
~12 min) because the multimodal training takes less compute per
event due to early visual saturation. Multi-seed: ~6 × 6 = ~36 min
estimate.

## Comprehension still failing

apple_self=0.310, apple_river=0.299 — same-concept stability not
above different-concept similarity. Comprehension TEST 1 fails across
all biological-scale iterations. The semantic_cortex representation
isn't stable across trials at biological scale.
