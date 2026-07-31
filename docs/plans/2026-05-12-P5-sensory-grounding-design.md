---
type: plan
status: live
date: 2026-05-12
---

# P5 sensory grounding via Cluster K v2 — design

**Date:** 2026-05-12
**Status:** Proposed (recommended pivot from iter AA/KK/LL/MM/NN-pending arc)
**Catalog entries:** G.11 (Hickok & Poeppel) + G.13 (Wernicke's area) +
                     K.01 (V1/V2/IT visual ventral stream) + Pulvermüller embodied semantics
**Effort estimate:** 1-2 weeks

## Why this design

After ~40 P5 iterations including 4 biological-scale tests on
2026-05-12, the per-concept pool architecture has been thoroughly
characterized as having an architectural ceiling at 4/6 BIDIR.

Root cause: discrimination depends on TOPOGRAPHIC BIAS PRIOR (not
learned weights). At biological scale, random per-seed structural
pool variance can amplify and overcome the bias.

Solution: add a SECOND strong signal during training that's
INDEPENDENT of random connectivity. Visual features for "apple"
(red, round) co-fired with the auditory word "apple" gives the
architecture an embodied semantic anchor.

This mirrors Tier 1's 6/6 PASS: motor teacher current during
training overrides random structure. Visual teacher would do the
same for abstract concepts.

## Architecture

Build on existing infrastructure:
- `sim/visual_cortex.py` (Cluster K v2 visual ventral stream)
- `research/runners/g11_bg_runner.py` (already integrates visual
  cortex with `--enable-visual-cortex`)
- `research/runners/text_minimal_isolation.py` `build_biological_brain_regions`

### Add regions (mirror g11_bg_runner's K v2)

```
retina           (2048: 2*32*32 ON/OFF)
cortex_v1_simple (8192: 8 orient * 4 freq * 16*16)
cortex_v1_complex (2048: 8 * 16*16, phase pooling)
cortex_v2        (200, plastic, internal_density=0.05, exc/inh 2.0/4.0)
cortex_it        (300, plastic, internal_density=0.10, exc/inh 2.0/4.0)
```

Concept binding region (NEW for P5):
```
multimodal_hub   (500, plastic, internal_density=0.05, exc/inh 0.3/0.8)
```

### Add pathways

Visual ventral stream (from K v2):
```
retina -> cortex_v1_simple (Gabor init, sparse 0.05, plastic)
cortex_v1_simple -> cortex_v1_complex (phase pooling, fixed)
cortex_v1_complex -> cortex_v2 (plastic, density 0.10)
cortex_v2 -> cortex_it (plastic, density 0.10)
```

Concept binding (NEW):
```
cortex_it -> multimodal_hub (plastic, density 0.30, weight 2.0)
wernicke_pool_0 -> multimodal_hub (plastic, density 0.30, weight 2.0)
wernicke_pool_1 -> multimodal_hub (plastic, density 0.30, weight 2.0)
multimodal_hub -> lang_output_pool_0 (plastic, density 0.30, weight 2.0)
multimodal_hub -> lang_output_pool_1 (plastic, density 0.30, weight 2.0)
```

The multimodal_hub becomes the convergence point for semantic content:
- Auditory: lang_input → wernicke_pool → multimodal_hub
- Visual: retina → V1 → V2 → IT → multimodal_hub

## Training schedule

For each concept (apple, river):
1. Generate a visual prototype:
   - Apple: red-on-green retina pattern, round shape
   - River: blue-on-tan retina pattern, elongated/wavy shape
2. For each training event:
   - Drive retina with concept's visual prototype (~100 steps)
   - SIMULTANEOUSLY drive lang_input with concept word
   - Both signals propagate to multimodal_hub
   - Hebbian co-firing binds the concept
3. Interleave apple/river per event (per iter AA's interleaved fix)

### Visual prototype generation (deterministic)

```python
def make_concept_image(concept: str, retina_size: int = 32):
    """Generate a deterministic image prototype for a concept.
    Uses concept-specific shape + color templates.
    """
    img_on = np.zeros((retina_size, retina_size), dtype=np.float32)
    img_off = np.zeros((retina_size, retina_size), dtype=np.float32)
    if concept == "apple":
        # Round shape, center, red (high ON, low OFF in red wavelength)
        cx, cy = retina_size // 2, retina_size // 2
        for x in range(retina_size):
            for y in range(retina_size):
                d = np.sqrt((x - cx)**2 + (y - cy)**2)
                if d < retina_size * 0.3:
                    img_on[y, x] = 1.0  # red region
                elif d < retina_size * 0.35:
                    img_off[y, x] = 0.5  # edge
    elif concept == "river":
        # Elongated horizontal wave, blue
        for y in range(retina_size):
            mid = retina_size // 2 + int(3 * np.sin(y * 0.5))
            for x in range(max(0, mid - 5), min(retina_size, mid + 5)):
                img_on[y, x] = 1.0  # blue region
    # Pack to retina ON/OFF format
    flat = np.concatenate([img_on.flatten(), img_off.flatten()])
    return flat  # shape (2*retina_size^2,)
```

## Recognition test (TEST 2c, new)

For each concept:
1. Stim ONLY the visual prototype (no lang_input)
2. Measure firing in wernicke_pool_0 vs wernicke_pool_1
3. Recognized concept = pool with more firing

Bidirectional binding test:
1. Type "apple" → lang_input drives wernicke_pool_0 → multimodal_hub fires
   apple-pattern → cortex_it (read out: visual apple-prototype emerges)
2. Show apple-image to retina → ventral stream → multimodal_hub →
   wernicke_pool_0 fires → lang_output_pool_0 (read out: "apple" word)

If both directions PASS, the sim has TRUE multimodal concept binding.

## Implementation steps

1. **Add visual cortex regions to `build_biological_brain_regions`**
   (~2 hr) — mirror g11_bg_runner's K v2 region creation, add CLI
   flags `--enable-visual-cortex-for-p5`
2. **Add multimodal_hub region + pathways** (~1 hr)
3. **Add `make_concept_image` function** (~1 hr)
4. **Add training schedule with visual + audio co-firing** (~2 hr)
5. **Add recognition tests (TEST 2c, 3c)** (~2 hr)
6. **Smoke test seed 42** (~30 min compute)
7. **Multi-seed validation** (~3 hr compute)
8. **Findings + docs** (~1 hr)

Total: ~10-12 hr engineering + ~4 hr compute = 1-2 days focused work.

If the smoke test passes 6/6 BIDIR, this is the path forward for P5
+ scales to vocabulary (just add more concept-image pairs).

## Why this is the right pivot

1. **Biology-faithful:** Real concepts ARE multimodal. The catalog
   G.11 + K.01 explicitly model both ventral semantic stream AND
   visual ventral stream as separate-but-converging pathways.
   Pulvermüller embodied semantics says concepts emerge from
   multimodal co-firing during development.

2. **No cheats:** Visual features for apple/river are deterministic
   geometric primitives (round-red, wavy-blue) — not external
   pre-trained features. The visual cortex learns to recognize
   them via Gabor RFs → V2 → IT processing.

3. **Solves the root cause:** Adding a strong, independent training
   signal (visual stream) makes concept binding NOT depend on
   per-seed random connectivity. This is the same principle that
   made Tier 1 motor binding work at 6/6.

4. **Scales:** Vocabulary expansion is just adding concept-image
   pairs. Architecture stays the same as concept count grows.

5. **Catalog-grounded:** G.11 + K.01 + Pulvermüller embodied
   semantics. Catalog R-pass corrections preserved.

## Risks

- Visual cortex training (retina → V1 → V2 → IT) is itself a
  multi-stage learning problem. May need pretraining the visual
  stream before binding. Compute cost.
- 6-seed validation is ~4-6 hr compute at biological scale.
- The IT→multimodal_hub pathway needs to learn concept-level
  features (not just visual). May require longer training.
