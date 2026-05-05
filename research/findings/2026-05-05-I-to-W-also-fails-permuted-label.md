# I→W (image → word) ALSO fails permuted-label control

**Date:** 2026-05-05 ~19:18 EDT
**Status:** Text I/O failure is bidirectional. Both W→A AND I→W fail
permuted-label across all historical configurations.

---

## The check

After today's W→A verdict (3-factor fails at biological scale, gradient
passes), the user asked about the I→W direction. Did anyone validate
image → word readout with permuted-label control?

**Answer: never run before.** The earlier "152/600 = 25.3% I→W" claim
from May 2 was raw accuracy without permuted-label validation.

```python
# Permute the 24 (image_direction → predicted_word) mappings; check if
# TRUE NESW mapping is the best of 24 perms across N runs.
```

## Result: 2/56 aligned (chance level)

Across 56 historical I→W eval JSONs (covering R3R6, swr500, h1_balanced,
h4_isolation, dpop, BigLang, BigMotor, NoLTD, StrongDrive, multidec,
xcouple, embodied, gabor, contrastive, curriculum, smoke, et al):

```
Aligned: 2/56
Mean true accuracy: 25.5%
Mean best permutation: 34.6%
Mean excess: +9.1pp
```

Chance for purely random alignment to NESW: 1/24 per file × 56 ≈ 2.3
expected. **Observed 2 — exactly chance.**

Best permutations are randomly distributed across files (NSWE, SWEN,
WSNE, etc.) — same architectural noise pattern as W→A.

## Two configs hit ~32-33% true accuracy AND aligned to NESW

Two of the 56 files show "true=best=NESW" with 0-1pp excess:
- `text_eval_R3R6_100ep_HebOff_v2_StrongDrive_seed42.json` (33%, NESW, 0.0pp)
- `text_eval_h1_balanced_seed101.json` (32%, NESW, 0.0pp)

These are the seed-101-architecture-noise alignments we've seen
elsewhere. Without 6-seed validation showing >=4 aligned, single-
seed best-of-24 alignments are not statistically meaningful.

## What this means

The text I/O failure is **bidirectional**:
- W→A (word → action): 1/6 aligned at biological canon (extensively
  studied today)
- I→W (image → word): 2/56 aligned across all historical configs
  (validated today)

Both directions show the same pattern:
- True accuracy ~25% (chance)
- Best perm ~33-35% (architecture noise floor)
- Excess ~7-9pp (consistent structural bias unrelated to task)

The bottleneck is the same: global scalar feedback × eligibility cannot
establish per-region credit assignment for arbitrary cue-action
mapping. Direction (W→A vs I→W) doesn't matter; the rule is the issue.

## Implications

This strengthens the argument for:

**Option 1 (dendritic learning)**: would address BOTH directions
  simultaneously. The Bono-Clopath rule's per-region apical signal works
  in either direction by design.

**Option 2 (biologically-realistic alternatives)**: also a valid path —
  the bidirectional failure suggests the FLASHCARD task structure is
  itself the problem, not just one direction. Real animal language is
  spatially/temporally grounded; flashcards have neither.

**Option 3 (scaling-direction)**: doesn't depend on text I/O at all.
  32×32 navigation hits 2.57 ± 0.11 because visuomotor cross-region
  credit assignment exploits spatial structure.

## Files

- This finding
- Permuted-label tool: `research/runners/permuted_label_check.py`
  (W→A version; I→W variant inlined for this analysis)
- 56 I→W JSONs in `research/findings/raw/g11_bg/text_eval_*.json`
- W→A verdict: `research/findings/2026-05-05-W-to-A-VERDICT-global-scalar-feedback-fails.md`
- Step 2 verdict: `research/findings/2026-05-05-step2-COMPLETE-rule-not-rescuable.md`
