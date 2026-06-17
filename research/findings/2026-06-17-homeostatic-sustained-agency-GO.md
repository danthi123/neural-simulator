# Sustained homeostatic agency — GO (6 seeds): the self-generated drive keeps the agent ALIVE over time

**Date:** 2026-06-17 (autonomous loop tick)
**Status:** **GO, 6 seeds.** The de-risked motivational core yields a genuinely self-maintaining agent: with a
self-generated homeostatic drive (and no external goal), the agent keeps its energy stable and never crashes over
a long episode, by repeated self-directed food-seeking; without the drive it crashes repeatedly. This demonstrates
the "alive over time" property — sustained self-regulation, the essence of artificial life — at the algorithm
level, building directly on the de-risked motivational core.

## What this adds

The cheapest-first GO (`2026-06-17-homeostatic-drive-rl-cheap-first-GO.md`) showed the agent *learns* a policy
from the intrinsic drive-reduction reward. This probe tests the complementary, longer-horizon property: over a
3000-step survival episode with energy continuously depleting, does the drive produce **sustained homeostatic
regulation** (the agent keeps itself alive) rather than just a one-off learned response?

## The episode (`_homeostatic_sustained_agency_derisk.py`, rate-proxy)

A corridor (food at position 0). Energy `E` depletes each step; the 2-pool drive tracks the deficit; the agent
online-Q-learns to navigate from `r = drive-reduction` (eating a real deficit → reward). The action→direction map
is **remapped per seed** (the agent must learn which way is food). The dynamics are chosen so the *learned* policy
reliably survives while *random* wandering reliably crashes: a refill (0.3) above the learned round-trip cost
(~0.09) but below the random-walk cost (~0.54). The only thing that produces the learned policy is the drive's
intrinsic reward.

## Result (6 seeds: 42/43/44/100/101/102)

| agent | min-energy (2nd half) | crash-rate | mean-energy |
|---|---|---|---|
| **DRIVE** | **0.71 – 0.95** (mean 0.89) | **0.0%** | **1.00** |
| LESION (drive frozen → r=0) | 0.00 (mean 0.01) | 0.3 – 16% | (crashes) |

**GO, 6/6.** The drive agent **never crashes** — energy pinned near 1.0 with min-energy well above the floor — it
genuinely *regulates* itself, keeping alive by self-directed food-seeking with no external goal. The lesioned
agent (no drive → no intrinsic reward → no learned policy) **crashes repeatedly** (starves), surviving only by
chance recovery. The discriminator is crash-avoidance (regulation), not mere band-occupancy.

## Honest scope

This is a **rate-proxy scaffold** (a 2-pool rate drive + tabular Q), not the brain-based spiking realization. It
demonstrates the homeostatic-regulation BEHAVIOUR that the de-risked motivational core produces — the agent stays
alive. The brain-based spiking realization reuses the validated navigation learning loop with the neural
drive-reduction reward (`2026-06-17-homeostatic-reward-plasticity-link-BY-COMPOSITION.md`), the deferred
integration build. Together with the spiking drive-mechanism GO and the cheapest-first learning GO, the
motivational core is de-risked across all three faces: it learns, it works on spikes, and it sustains life.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._homeostatic_sustained_agency_derisk --seeds 42 43 44 100 101 102
```

No `sim/` edit. Reuses the cheapest-first probe's `TwoPoolDrive`.
