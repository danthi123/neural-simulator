---
type: finding
status: live
date: 2026-07-23
mechanism: value-critic
---

# Value-critic closure — RANK-1 GO (6-seed): the brain's OWN learned spiking value drives a value-driven choice (2026-07-23)

## Result — GO, 6/6 seeds, all anti-cheats
`research/runners/_navcloseout_R5b_learned_value_choice.py --seeds 42,43,44,100,101,102 --value-train-trials 40`
(numpy CPU; the tiny spiking value-WTA decision organ + the merged one-brain `striosome_value` critic):
```
VERDICT: GO   (>= 5/6 seeds per gate; actually 6/6 on every gate)
gates (of 6):  headline 6 | lesion 6 | untrained 6 | discrim 6 | permute 6
acc_intact_mean   1.00  (per-seed [1,1,1,1,1,1])   <- the learned V drives the choice PERFECTLY
acc_lesion_mean   0.49  (~chance)                  <- pin V to the mean -> choice collapses
acc_untrained_mean 0.175 (per-seed [.167,.35,.233,.05,0,.25], all < chance 0.5)  <- THE load-bearing control
equal_agreement   1.00  (equal-V discriminator stays NEUTRAL -> value-specific, not a fixed bias)
acc_permuted      0.50  (permute which cue gets which V -> chance)
value_salience_corr_absmax 0.173 (low -> the choice rides LEARNED VALUE, not a relabeled salience)
V_trained_hz: near cue 32-35 Hz >> far cue 14-18 Hz  (the learned value gradient, a real cp_firing_states read)
```

## What this closes (the "value critic 🟨 partial")
The scope (`2026-07-23-value-critic-closure-scoping.md`) corrected the framing: value is ALREADY computed on the
spiking substrate — the striatal `striosome_value` region LEARNS V(cue) by dopamine-gated STDP (neurons + synapses),
its GABA_B/GIRK conductance subtracts V at the SNc membrane, and the SNc's FIRING is the reward-prediction error.
The residual was a DEMONSTRATION gap: the prior R5 value-driven-choice GO (`2026-06-27-navcloseout-R5-...`) drove its
spiking value-WTA with a HOST value stand-in (`build_concept_value`); the on-substrate form (read the real learned
spiking V) was scaffolded but never run.

RANK-1 closes it: the value-WTA's drift is now set by the **real learned spiking V** — `striosome_value` is trained by
DA-gated STDP, its per-cue FIRING RATE (a `cp_firing_states` read) IS the value, and the neural pool's firing IS the
choice. Spiking decision + spiking value, on the merged one-brain bridge. The NEW load-bearing anti-cheat the prior GO
lacked — **G_UNTRAINED** (read V from the UNTRAINED critic -> flat/anti-goal V -> the trained advantage vanishes,
per-seed all < chance) — proves the SUBSTRATE'S LEARNING is load-bearing, not a wired-in prior.

## Method / discipline
- Reuse-by-import, NO `sim/` edit. CPU/numpy, ~2.5 min/seed.
- The calib (1 seed) mislabelled its verdict `HONEST_NEGATIVE` — a n_seeds=1 artifact of the `headline_n < 5/6` bar
  (`decide_verdict:338`), NOT a real negative; caught by reading the runner's own verdict logic (the verify-the-
  instrument discipline). The full 6-seed reaches GO.
- Moat by construction: the WTA decision organ has no RF/conversational slices (array-disjoint from any composer).

## Honest scope
The close stays **value-CUED** (the cue's learned V drives the choice), not spatially-hidden instrumental credit
(that is the accepted-deep 3x-NEGATIVE dendritic wall, deliberately sidestepped — a separate frontier). This closes
the value-CRITIC (the brain learns + uses value on spikes); it does not claim spatial credit assignment. ⇒ the
"value critic 🟨 partial" is now DONE on the project's terms: fully-spiking, one-brain, learned, all anti-cheats.
