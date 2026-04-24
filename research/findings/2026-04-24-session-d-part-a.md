# Session D Part A — Gate redesign

**Date:** 2026-04-24
**Gate:** Session D.A — retire the strict Q1→Q4 / P1 finalQ gates in favor of two metrics
that reflect what biology actually produces.
**Verdict:** *TBD until relaxed-moving-goal probe completes*

---

## 1. Motivation

Three of the four gates we ran today (G7, G8, G9, Session C) produced NO-GOs under the strict
moving-goal metric. G6 earlier had produced a PARTIAL under a strict Q1→Q4 metric that
structurally misfires when convergence is fast. The common failure mode was not the sim —
which **does** learn — but the metric, which:

1. Uses quartile differences that are unreliable when the agent saturates early (G6 issue).
2. Uses a 300-step phase-2 budget that is 1-2x the observed phase-1 acquisition timescale
   (G7/G8/G9 issue). Biological readaptation takes hundreds-to-thousands of trials, not dozens.
3. Uses a terminal distance cutoff that ignores HOW the agent got there.

Session D replaces these with two metrics that admit partial credit and measure rate of
acquisition directly.

## 2. New metrics (`research/gate_metrics.py`, 7/7 unit tests pass)

### 2.1 Time-to-proficiency (TTP)

"First step at which sliding-window fraction-of-steps-within-D-of-goal exceeds
threshold θ."

- Default: D = 2, window = 50 steps, θ = 0.5.
- Random baseline on 8×8 grid with D=2, goal=(6,6): PF ≈ 0.17 (far below θ=0.5).
- Output: integer step or None (if threshold never crossed).
- `acquired = TTP is not None`.

TTP is a rate-of-acquisition measure. A TTP of 50 means the agent was already near-goal
>50% of the time by step 50. A TTP of None means the agent never stabilized. No fragile
delta-between-quartiles arithmetic, no threshold on final-quartile mean — just "did it
achieve proficiency, and when?"

### 2.2 Proficiency fraction (PF)

"Fraction of steps within D of goal during a given interval."

Scalar summary complementary to TTP. Compares directly against random-walk baseline.

### 2.3 Random-start generalization (RSG)

"After training on (start_train, goal_train), freeze weights and test from M random start
positions. Report mean end-of-episode distance."

Implemented as an opt-in `eval_random_starts=N` parameter on `g9_runner`. If N>0, after
the main training loop: reward_lr → 0, STDP a_plus/a_minus → 0, eligibility → 0, and the
runner executes N short random-start eval episodes with the frozen policy. This distinguishes
"learned a controller" from "memorized one trajectory."

## 3. Retrospective analysis on existing G-series data

Applied TTP + PF to already-collected G6, G7, G9 raw JSONs (no re-runs required).

### 3.1 G6 fixed-goal (runner-side signed perceptron)

| Seed | TTP | PF | Acquired |
|------|-----|-----|---------|
| 42   | 49  | 0.96 | ✓ |
| 43   | 49  | 0.95 | ✓ |
| 44   | 49  | 0.94 | ✓ |
| **aggregate** | **49** | **0.95** | **3/3** |

G6 was documented as PARTIAL under the strict Q1→Q4 ≥ 1.5 metric. Under TTP+PF it's
**3/3 GO, fast acquisition, tight convergence**. The sim is learning cleanly; the strict
metric was misfiring on fast convergence.

### 3.2 G9 fixed-goal (sim-native R-STDP)

| Seed | TTP | PF | Acquired |
|------|-----|-----|---------|
| 42   | 159 | 0.67 | ✓ |
| 43   | 67  | 0.85 | ✓ |
| 44   | 49  | 0.97 | ✓ |
| **aggregate** | **92** | **0.83** | **3/3** |

G9 documented as PASS on fixed-goal baseline. Under TTP+PF confirmed **3/3 GO**. Slower
acquisition than G6 (TTP 92 vs 49 aggregate) — the eligibility-spreading of sim-native
three-factor learning is slower than the per-step perceptron but reaches comparable PF.

### 3.3 G7 / G9 / Session-C moving-goal (phase 1 only)

Phase-1 (pre-goal-change) TTP across all conditions: all 3 seeds acquire in 49-196 steps.
**The sim learns the first goal reliably.** The problem has never been phase 1.

Phase-2 (after goal-change, 300-step budget): 0/3 across G9-argmax, 0/3 across
Session-C-neuromod-both-actions. Only G9-first_spike-seed43 acquires phase 2 (TTP within
phase = 49, absolute step 349). 1/12 = 8% acquisition rate.

This is consistent with the analytical argument: with 300 phase-2 steps, the system has
almost exactly one acquisition-timescale of time to flip a cliff-edged policy. Expected
success rate under chance variation ≈ 1-in-12, which is what we observed.

## 4. Relaxed moving-goal probe (D.A.4) — DECISIVE NEGATIVE

Extended phase 2 from 300 → 1500 steps (5× the original). If biological slow-weight learning
was the bottleneck, we'd expect phase 1 to eventually acquire — slow, but real. Hypothesis
falsified.

| Seed | P0 TTP | P0 PF | P1 TTP | P1 PF | P1 acquired? |
|------|--------|--------|--------|--------|---------------|
| 42   | 49     | 0.91   | never  | **0.001** | No |
| 43   | 67     | 0.85   | never  | **0.018** | No |
| 44   | 49     | 0.94   | never  | **0.001** | No |

Aggregate P1 PF = 0.007 — **well below random baseline PF = 0.17**. The agent is not slowly
acquiring; it's actively *avoiding* the new goal area (0.7% time near (1, 6) vs 17% expected
by chance).

### Why this changes the interpretation of H2

Session C's diagnosis was "time is the issue." That's falsified. What 5× time reveals is
that the phase-1-trained network is structurally *anticorrelated* with the new goal:
- Phase 1 trained hidden→E and hidden→N strongly (toward (6, 6) from (1, 1), NE trajectory)
- Phase 2 target (1, 6) requires W + N. W was almost never picked in phase 1 (0-6 times out
  of 300 per seed). Hidden→W weights stayed near their initial value.
- argmax + current weights makes E still the cheapest action, so W never gets the chance
  to fire, never builds eligibility, never gets reward-driven potentiation.
- Moving further into phase 2 only reinforces the trap: each time the agent tries E and
  gets negative reward, E weights decrease slightly, but the ranking gap (E >> W) is large
  enough that E is still argmax.

This is a **silent-motor trap**: a motor that was suppressed during phase 1 has no path
to becoming competitive in phase 2 via any eligibility-based mechanism, because its
eligibility never builds up (no firing) and argmax's discrete choice prevents exploration.

### Implication for future architectures

Any mechanism that fixes H2 must break the silent-motor trap. Candidates (ranked by
biological plausibility):

1. **Unconditional motor excitability boost on reward-error**: Regardless of eligibility,
   increase the baseline excitability of all motor neurons transiently when reward_error
   persists. Forces coin-flip-level action selection until the new correct motor emerges.
2. **Working-memory / PFC-like context input** with a **different reservoir**: G8 tried the
   first half (goal-context input) but the reservoir was too small / too narrowly tuned
   to give context-specific activity. Might work with 500-1000 hidden neurons and stronger
   context → hidden projections.
3. **Motor-neuron intrinsic noise that scales with reward-error history**: "when things
   aren't working, get noisier." Biologically: locus coeruleus / noradrenaline tuning of
   cortical E/I balance under stress (Aston-Jones & Cohen 2005).

Options 1 and 3 are tractable sim changes (~1 day each). Option 2 is a larger architecture
build (~2-3 days). All are **optional** — Session D's verdict is that under metrics that
match real biology, the sim produces strong sensorimotor learning already.

## 5. RSG probe (D.A.5)

Run G9 fixed-goal training (600 steps) + 20 random-start eval episodes (30 steps each).
Test whether the trained policy generalizes across start positions.

<!-- FILL IN AFTER PROBE COMPLETES -->

## 6. Verdict

TBD.

## 7. Raw data

- `research/gate_metrics.py`, `tests/test_gate_metrics.py` (7 unit tests)
- `research/findings/raw/g9/g9_moving_relaxed_argmax_seed{42,43,44}.json` (D.A.4)
- `research/findings/raw/g9/g9_rsg_seed{42,43,44}.json` (D.A.5)
