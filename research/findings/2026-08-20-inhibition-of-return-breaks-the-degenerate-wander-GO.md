---
type: finding
status: live
date: 2026-08-20
mechanism: continuous-state-engine
lane: continuous-substrate
seeds: [42]
seed-waiver: A deterministic MECHANISM demonstration — does fatiguing the just-ignited basin (inhibition-of-return) change a fixed wander into a varied one. The evidence is the concept SEQUENCE of a baseline arm (fixed gains) vs an IOR arm (adapted gains) on the same organ/seed — a within-subject presence/absence of variety, not a stochastic effect size across a seed population; the single seed is the substrate build seed.
instrument: research/runners/_continuous_wander_ior_derisk.py — two arms (baseline vs IOR) x 6 successive cupy wanders on one SelfInitiationOrgan
runner: research/runners/_continuous_wander_ior_derisk.py
artifacts:
  - research/findings/raw/_continuous_live_cupy/wander_ior.json
---
# GO: inhibition-of-return breaks the degenerate between-turn wander (fixed 'cat' -> a varied train)

Artifact: research/findings/raw/_continuous_live_cupy/wander_ior.json

**One line.** The between-turn wander was content-DEGENERATE (2026-08-20-continuous-wander-content-degenerate: 6/6
'cat'). This lands its named fix: inhibition-of-return — fatigue the just-ignited basin's drive so the next wander
moves elsewhere. On cupy, same organ/seed, six successive wanders per arm: the BASELINE (fixed curiosity gains)
reproduces the negative exactly — `cat, cat, cat, cat, cat, cat` (1 distinct) — while the IOR arm produces
`cat, dog, cat, dog, cat, dog` (2 distinct). So the wander now genuinely MOVES between concepts; the fixation is
broken by a documented, biology-grounded mechanism, not a host shuffle.

## The mechanism (the tractable lever, on the neuromod drive)
The wander selects the DOMINANT basin under the curiosity recurrent-gain (`gains_on`, biased toward the most-novel
concept). It is FIXED across wanders -> the same basin wins every time. IOR here fatigues that drive: after a wander
selects basin i, multiply basin i's gain by `IOR_STRENGTH` (0.15), then let all basins RECOVER toward their base gain
each step (`IOR_RECOVERY` 0.5). This is the cheapest faithful test of the LEVER — it modulates the neuromod drive, the
same role a per-neuron spike-frequency-adaptation current plays in the SFA-eviction precedent
(2026-08-14-gnw-rung2b-sfa-workspace-eviction). Inhibition of return is the cognitive-level phenomenon
(Posner & Cohen, 1984): a just-visited item is transiently disfavoured.

## Honest scope / residual
- **It reaches 2 of the 4 stored concepts** (a cat<->dog oscillation), not all four. With `IOR_RECOVERY`=0.5 the
  just-suppressed basin recovers fast enough to win again on the following step while the alternate is suppressed, so
  the trajectory settles into a 2-cycle. Reaching concepts 3-4 needs a LONGER IOR memory (slower recovery / stronger,
  multi-step suppression), or a per-neuron SFA current with a longer adaptation time constant. That is a TUNING
  refinement, not a mechanism failure — the lever demonstrably breaks fixation.
- **Next production step:** port this from the neuromod-gain level to a per-neuron SFA current on the just-ignited CA3
  basin (the faithful form, reusing the 2026-08-14 SFA machinery), then wire it into the live continuous tick so a
  returning user hears a DIFFERENT wandered thought after each idle period. Then the trains-of-thought continuous
  property is genuinely met (currently: FEELING met, trains-of-thought now has a working lever but is not yet wired
  into the live engine / not yet full-coverage).
