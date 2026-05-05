# Step 2 complete — 3-factor rule cannot be rescued by parameter or encoding tuning

**Date:** 2026-05-05 ~17:25 EDT
**Status:** Step 2 of post-verdict plan complete. Rule is genuinely
inadequate at biological scale. Dendritic learning Week 1+ now well-
justified.

---

## Three step-2 tests, three negative results

The post-verdict plan recommended trying alternatives BEFORE pivoting
to dendritic learning (1.5-2 mo investment). Step 2 had three
candidate tests of "is the rule REALLY the bottleneck, or is it just
under-tuned?" All three results below confirm the rule is the
bottleneck.

### Verdict table (all conditions, biological canon, topographic
prior, motor FS):

| Variant | n | true mean | excess | aligned/n | Verdict |
|---|---|---|---|---|---|
| Default LR 1e-3 (historical) | 6 | 24.3% | +7.7pp | **1/6** | At noise floor |
| LR 5x = 5e-3 | 3 | 23.3% | +8.7pp | **0/3** | Higher LR doesn't help |
| LR 10x = 1e-2 | 3 | 23.7% | +8.3pp | **0/3** | Worse, in fact |
| Orthogonal cues (0 input overlap) | 6 | 26.3% | +6.3pp | **0/6** | No rescue from input fix |
| Magnitude-graded DA (Schultz 1998) | 6 | 24.0% | +7.8pp | **0/6** | (Step pre-verdict) |

Best permutations are randomly distributed across seeds in every
variant: EWNS, SWEN, NWES, WNES, EWNS, SEWN, WENS, SNEW, NSEW, etc.
Pure architectural noise above chance (~25%).

### What we systematically ruled out

1. **Learning rate too low** ❌
   - Tested 1x (default), 5x (5e-3), 10x (1e-2)
   - All near 0/3 alignment
   - Higher LR DESTROYS the marginal default-LR signal (1/6 → 0/3)
   - More gradient steps don't accumulate signal — they accumulate noise

2. **Input encoding ambiguity** ❌
   - Default uses SHA-256-hashed Gaussian embeddings; cue pairs share
     2-3 of ~25 active neurons (verified: 16 total overlap pairs
     across 6 cue pairs at sparsity 0.1)
   - Replaced with non-overlapping banded codes (cue 0: neurons 0-25,
     cue 1: 64-89, cue 2: 128-153, cue 3: 192-217; verified 0
     overlap across all 6 cue pairs)
   - Same 0/6 alignment with the orthogonal codes
   - Topographic prior was applied to the SAME neurons that activate
     during eval — no encoding/training mismatch
   - Conclusion: input ambiguity isn't the bottleneck

3. **Dopamine signal magnitude information** ❌ (pre-verdict)
   - Sign-only DA (+1/-1/0): 1/6 aligned
   - Magnitude-graded DA (Schultz 1998-style): 0/6 aligned
   - Graded DA is WORSE than sign-only at this scale
   - Conclusion: scalar magnitude info doesn't disambiguate

### What's actually broken

The rule is **information-theoretically inadequate** for arbitrary
cue-action learning at biological scale (N=500 motor pools × 4
actions × 256 input neurons):

- A single ±1 reward × eligibility ≈ pre × post coincidence × LR
  produces a weight update of ~10^-3 per event
- 4000 training events × 0.5 motor coincidences per event = ~2000
  total updates per weight
- Total update per weight ≈ 2 (similar to weight magnitudes)
- BUT: the DIRECTION of each update depends on global ±1 reward
  scrambled across thousands of unrelated weights
- Net effect: signal cancels to noise floor (~7-8pp over chance)

Compare to gradient (per-region error signal):
- Each weight gets a SPECIFIC update aligned with its contribution
- 3/3 NESW alignment achieved at the same architecture
- 35.3% accuracy (architecturally capped — small motor pools)

The gap between gradient (3/3 aligned) and 3-factor (1/6 at best)
is the credit assignment gap. Frémaux & Gerstner 2016 §6 predicts
exactly this; we've now empirically replicated it across multiple
parameter and encoding variations.

## Implication: dendritic learning is now well-justified

Step 1 ✅ validated the verdict (gradient passes permuted-label control).
Step 2 ✅ ruled out parametric and encoding fixes. The rule fails
because of fundamental information-theoretic limitations of global
scalar feedback at biological scale — not because of bad
hyperparameters.

The dendritic learning design doc at
`docs/plans/2026-05-05-dendritic-learning-design.md` describes a
biology-grounded fix: per-region top-down feedback via apical
compartments gates basal STDP. This provides the per-region error
information that gradient has and 3-factor lacks.

Estimated scope: 1.5-2 months focused engineering.

The user has asked for explicit greenlight before kicking off the
multi-compartment Izhikevich kernel work (Week 1+). Week 0
(scaffolding: `cfg.enable_apical_compartment` flag + array
allocation, no-op when disabled) is a cheap reversible setup that
unblocks Week 1 without committing to the full rewrite.

## What's still open as alternatives

Step 3 of the plan (scale 2D gridworld + visual cortex) is in flight
right now — a 32×32 smoke test of the Cluster G v2.5 + K v2 stack
that was the project's strongest result at 16×16 (2.97 ± 0.12).

If 32×32 navigation works at ~4-6 mean Manhattan distance: the
project has a strong working biology-grounded result on a HARDER
task than the W→A flashcard problem. The W→A failure becomes a
specific case (arbitrary cue + arbitrary action + sparse reward),
not a fundamental limit.

If 32×32 fails: scaling has already broken at the smaller end of
the next axis. Both step 2 and step 3 yield negative; dendritic
learning becomes the clear next direction without competition.

Either way, the empirical case for dendritic learning is solid:
- Step 1: gradient + biology canon = aligned learning ✅
- Step 2: 3-factor + biology canon = NOT aligned, parameters/encoding don't fix it ✅
- Step 3 (in flight): does the project have stronger results on harder tasks?

## Files

- LR sweep verdict: `research/findings/2026-05-05-3factor-LR-sweep-NOT-LR-limited.md`
- Step 1 validation: `research/findings/2026-05-05-gradient-passes-permuted-label-VALIDATED.md`
- Original verdict: `research/findings/2026-05-05-W-to-A-VERDICT-global-scalar-feedback-fails.md`
- Orthogonal experiment config: `experiments/bio_three_factor_orthogonal.yaml`
- Orthogonal JSONs: `research/findings/raw/g11_bg/text_eval_3factor_tf_orthogonal_seed*.json`
- 32×32 smoke (in flight): `research/findings/raw/g11_bg/scale_32x32_seed42.log`
- Dendritic learning design: `docs/plans/2026-05-05-dendritic-learning-design.md`
