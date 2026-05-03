# I->W eval is also at chance — not just W->A

**Date:** 2026-05-03 ~08:05 EDT (autonomous overnight)
**Context:** Investigating why W->A regression happens with SWR. While
checking, looked at the related I->W eval across conditions.

---

## I->W (image to word) accuracy across conditions

| Condition | seeds | mean +/- std |
|---|---|---|
| v2 baseline (no Phase 1) | 6 | 25.3% +/- 4.5% |
| v2 + SWR (no Phase 1) | 6 | 27.0% +/- 7.2% |
| H4 isolation (no Phase 1+2) | 4 | 22.2% +/- 1.3% |

Chance is 25%. **All conditions are within +/- 5pp of chance**, with
the high-variance v2+SWR drifting up by 2pp and the H4 isolation
drifting down by 3pp.

## Implications

I->W is the eval task: drive `retina` with a gridworld image showing
agent position + goal position; observe which `language_output` neurons
fire, decode via nearest-token. The decoder uses `cortex_it →
language_output` (image-to-word readout pathway).

If I->W is at chance even after v2 baseline training, this means:
1. The visual cortex pathway (retina -> v1_simple -> v1_complex -> v2 -> IT)
   isn't producing direction-discriminative IT firing patterns
2. Or: the IT -> language_output pathway isn't trained enough to map
   IT firing to the correct word
3. Or: language_output is too noisy/overlapping for the readout to work

The fact that H4 isolation (no Phase 1, no IT-related training, no
Phase 2 text-IO) gives 22% — basically the same as v2 baseline 25% —
suggests the visual cortex weights aren't differentiating goal
directions.

## What this means for the W->A debugging

Both eval directions (I->W and W->A) hover near chance. The text I/O
infrastructure is producing chance-level performance across both
modalities. **The 28.5% W->A baseline isn't real word-action learning
— it's chance + ~3pp residual cascade bias.**

Combined with the H4 result (paired-stim alone can't even achieve
chance on W->A), this suggests:

The current architecture has insufficient:
- Visual cortex training (cortex_it doesn't differentiate goals)
- Language readout training (IT->language and cortex->language
  pathways are weakly differentiated)
- Word-motor weight specialization (language->motor doesn't map words
  to specific motor pools cleanly)

**The whole text I/O system is essentially at chance.** This was
masked by the W->A 28.5% number which sounded above-chance but is
actually within noise of chance.

## Next steps (priority shift)

1. **Verify the chance hypothesis** — run a permuted-label control:
   shuffle the (token, action) labels in the eval and see if accuracy
   stays at ~28.5%. If it does, the original number is purely
   architectural noise and we have NO learning at all.
2. **Find what makes text I/O actually learn** — the architecture is
   currently failing across both eval directions. We need a much
   bigger architectural change than the arch sweep tests.

The arch sweep (auto-launches after H1) tests motor pool size,
language region size, sparsity. None of these are likely to bridge
the chance-to-real-learning gap. They might give 30-35% W->A but not
break it.

The real fix probably requires:
- Multi-day training (10x+ Phase 2 episodes)
- Hebbian re-enable with fixed decay
- Stronger initial wiring (text→motor weight 5+ instead of 3)
- Visual cortex pre-training (Phase 1 episodes >> 0)

These are tomorrow-and-beyond projects, not overnight.
