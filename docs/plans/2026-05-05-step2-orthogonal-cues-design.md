---
type: plan
status: live
date: 2026-05-05
---

# Orthogonal cues experiment — step 2 fallback if high-LR sweep fails

**Date:** 2026-05-05
**Status:** DESIGN ONLY (conditional on high-LR sweep showing 0-1/6
aligned, in which case rule isn't LR-limited and this is the next
test).

---

## Hypothesis

The current W→A encoding hashes tokens to RANDOM sparse codes via
SHA-256, then takes the top 10% by magnitude. Two side effects:

1. **Codes have arbitrary overlap.** "north" and "east" might both
   activate neuron 88 if their hash values happen to land there.
   Overlap means topographic bias (which boosts weights from active
   neurons of word w to motor_target(w)) becomes ambiguous: neuron
   88's boost goes to motor_N AND motor_E.

2. **Code geometry is hash-dependent.** No semantic/spatial
   relationship between codes. Tokens that mean "similar things"
   (e.g., "north" and "up") could map to wildly different neurons.

For supervised gradient (3/3 aligned): per-region error signals
disentangle the overlap automatically.
For 3-factor (1/6 aligned): scalar global feedback can't disentangle.
Overlap may be a contributor to the failure mode.

## The test

Replace random hash codes with **orthogonal banded codes**: each
cue gets a unique non-overlapping band of `n_active = 25` neurons,
spaced so the gaps prevent recurrent-excitation crosstalk.

```
"north" (idx 0): neurons 0..24       active
"east"  (idx 1): neurons 64..88      active
"south" (idx 2): neurons 128..152    active
"west"  (idx 3): neurons 192..216    active
```

(With n_neurons=256, gap=39 between codes.)

Same architecture (biological canon, motor_FS, topographic bias).
Same rule (3-factor). Only the input encoding changes.

## Outcomes

| Aligned/n with orthogonal cues | Interpretation |
|---|---|
| 4-6/6 | **W→A failure was input ambiguity.** Random hash codes confuse 3-factor; orthogonal codes are tractable. Dendritic learning may not be needed. Cheap fix. |
| 2-3/6 | **Partial improvement.** Orthogonal codes help but don't fully fix. Likely a real signal but architecture limits. |
| 0-1/6 | **Encoding isn't the bottleneck.** Rule is genuinely inadequate. Dendritic learning still warranted. |

## Implementation

Add to `sim/text_embeddings.py`:

```python
def orthogonal_drive_pattern(
    cue_idx: int,
    n_cues: int = 4,
    n_neurons: int = 256,
    drive_max_pA: float = 200.0,
    n_active_per_cue: int = 25,
) -> np.ndarray:
    """Non-overlapping banded drive pattern for cue indexing.
    
    Each cue gets a unique band: cue_idx 0 → first n_active neurons,
    cue_idx 1 → next stride neurons, etc. With stride = n_neurons / n_cues
    we get gap = stride - n_active_per_cue between bands.
    """
    drive = np.zeros(n_neurons, dtype=np.float32)
    stride = n_neurons // n_cues
    start = cue_idx * stride
    end = start + n_active_per_cue
    drive[start:end] = drive_max_pA
    return drive
```

Add to `bio_three_factor.py`:
- `orthogonal_cues: bool = False` parameter
- Replace `vocab_to_drive_pattern(token, ...)` with conditional:
  ```python
  if orthogonal_cues:
      vocab = ["north", "east", "south", "west"]
      cue_idx = vocab.index(token)
      drive = orthogonal_drive_pattern(cue_idx, n_cues=len(vocab), ...)
  else:
      drive = vocab_to_drive_pattern(token, ...)
  ```
- Same change at apply_topographic_bias call site

## Cost

- Implementation: ~30 min (helper function + flag + 2 call sites)
- Validation: 6 seeds × 1 condition (orthogonal_cues=True with topo+FS)
  ≈ 50 min at parallel=6
- Total: ~80 min

## Why this is the cheapest step 2 test

Compared to other step 2 candidates:
- **Sequence learning task**: requires new task design + training pipeline. ~1 week.
- **Conditional cue-action**: requires PFC working memory binding. ~3-5 days.
- **Visuomotor cue with arbitrary mapping**: requires synthesizing visual
  inputs that aren't position-correlated. ~2-3 days.

Orthogonal cues isolates ONE variable (input ambiguity) on the
existing infrastructure. Highest information per hour.
