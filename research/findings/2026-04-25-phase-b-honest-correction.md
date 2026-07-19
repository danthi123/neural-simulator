# Phase B Acid Test — Honest Correction

**Date:** 2026-04-25
**Status:** Correction to overstated Phase B.T6 finding
**Branch:** `pfc-working-memory`

## What I claimed

> Phase B.T6 ACID TEST: 3 seeds run. Phase 1 finalQ 5.46 avg (vs G9 baseline 6.74).
> Silent-motor trap DISSOLVED.

## What's actually true

The per-trial **motor counts are 0 in 1799/1800 trials** — the BG cascade
only produced motor activity at trial 0 and nothing thereafter. With all
motor counts at 0, the runner's argmax falls back to RANDOM action
selection (line in code: `int(np.random.default_rng(seed * 10000 + step).integers(0, N_ACTIONS))`).

So the result was random walking on the grid. Random walk on 8×8 grid
to goal (1,6) has expected mean Manhattan distance ≈ 5.5, which is
exactly what we measured (5.46 avg phase 1 finalQ).

## What this means

1. The "silent-motor trap dissolved" claim IS technically true — all 4
   actions used uniformly. But it's because **the actions are completely
   random, not because the BG circuit is producing them**.

2. The 5.46 phase 1 finalQ is NOT better than V1's 6.40 in any meaningful
   sense — it's just random walk vs V1's "agent stuck pursuing wrong
   direction." Different failure modes.

3. The static probe (commit 8415868) DID show the cascade working —
   motor_W = 7 Hz when cortex_W driven. So the architecture IS sound.
   The bug is in the trial-based dynamics.

## Why the cascade dies after trial 0

Investigation shows:
- Step 0: motor counts (e.g.) = (10, 10, 1, 2) — cascade fires
- Step 1+: all zeros — cascade silent

Hypotheses (untested):
1. Synaptic conductance buildup during reward-hold steps with current_reward_signal
   set to ±1 might destabilize internal state
2. BG output nuclei (GPe/GPi) need to be in a specific firing regime that
   the trial timing disrupts
3. Striatum needs its baseline -80 mV down-state preserved between trials,
   but residual depolarization from previous trial keeps it from re-entering
   down-state cleanly
4. The trial mode runs 110 ms stim (transient response only); the static
   probe ran 500 ms continuous (steady state)

Honest current state: I don't know exactly why the cascade dies. More
debugging needed.

## What should be reported

- Phase A preset audit + bug fixes: REAL and verifiable. 30 working presets.
- Static action selection probe: REAL — cascade produces selective output
  when given continuous input (no trial structure).
- Phase B.T6 acid test on moving-goal: NOT a clean win. Motor counts are
  zero in 1799/1800 trials, agent random-walks. The "trap dissolved"
  conclusion was overstated.

## Next steps

1. Debug why cascade dies after trial 0 (likely state-reset issue
   between trial stim windows + reward-hold sim steps)
2. Once cascade produces sustained motor output across trials, re-test
   the moving-goal scenario for genuine learning + readaptation

## What still stands

The architectural framework, presets, and static cascade are all sound.
Phase A and Phase B.T1-T5 are real wins. Only Phase B.T6 (the acid test
on moving-goal) needs the cascade-stability bug fixed before it can give
a meaningful answer.

This is a normal R&D arc: a bunch of architecture works, the integration
test reveals a stability issue, more work needed. I should have looked
at the per-trial motor counts before celebrating.
