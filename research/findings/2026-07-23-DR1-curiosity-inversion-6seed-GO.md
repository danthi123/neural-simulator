# DR-1 (new-direction Phase-0): the no-confab moat INVERTED into honest CURIOSITY — 6-seed GO (2026-07-23)

The owner's headline point ("don't refuse when unsure — crave knowledge + growth, seek to learn") realized:
the SAME uncertainty signal that drives the no-confab abstention becomes a CURIOSITY drive that ASKS + LEARNS,
kept honest by a learning-progress reward. Reuse-by-import (the Bogacz-Brown `RealAntiHebbianFamiliarity` gate +
the RPE/value machinery), NO `sim/` edit (verified). Runner `_curiosity_seek_learn_cheap_first_probe.py`.

## Mechanism
The familiarity gate supplies the epistemic gap g=novelty(x) (~1 novel / ~0 learned) → a curiosity modulator; when
NOVEL the policy ASKS a teacher, INGESTS the answer (imprint → raises familiarity → lowers future novelty), and the
intrinsic REWARD = LEARNING PROGRESS (g_before − g_after), TD-tracked as a per-concept expected-LP value. On-bridge
this fills the reserved `from_novelty` neuromodulator stub (follow-on); the CPU probe proxies it at rate level.

## 6-seed result — GO 6/6 (seeds 42/43/44/100/101/102)
- **Curiosity drives asking:** corr(gap, modulator) **+0.99**; ask-rate unknown ≫ known; post-answer confidence rises
  **+0.57** above the 0.03-0.04 abstain floor. Reward IS learning-progress: LP(learn) ≈ +0.21 vs LP(noise) ≈ +0.003.
- **Noisy-concept-STOPS-asking (mandatory anti-confabulation guard, INVOKED):** for un-learnable "noisy-TV" concepts,
  asking decays 0.07 → 0.00 **because the value vetoed them** (noisy expected-LP ≈ 0.03 ≤ 0.05 floor) **while their gap
  stays HIGH (g ≈ 0.97)** — i.e. they were NEVER spuriously learned. Curious AND honest.
- **Controls collapse (6/6):** lesion (modulator=0 → 0 asks); yoked-random reward (masters 3-7/8 vs real 8/8, robust
  over ask-budgets 26-40); permuted-gap (corr → 0, masters 5-6/8); moat-by-construction (confident set ⊆ ingested set,
  every seed).

## Rigor
Two real modeling bugs found + fixed in the smoke: (1) at small D the gate's span fills so noise becomes spuriously
learnable → D=1024 keeps noise genuinely novel; (2) `OBS_NOISE·√D` jitter swamps the unit code at large D → replaced
with a dimension-independent unit-direction jitter.

## Net
The moat's uncertainty signal becomes a curiosity drive that seeks a teacher + learns — and the learning-progress
reward keeps it HONEST (it stops chasing noise + never confabulates). This is the biological resolution of the
owner's "don't refuse, grow instead." Phase-0 P0.2. Follow-on: on-bridge (`from_novelty` + spiking-SNc RPE + A→W
question) + wire into the develop-loop teacher hook (P2.1). NO `sim/` edit.
