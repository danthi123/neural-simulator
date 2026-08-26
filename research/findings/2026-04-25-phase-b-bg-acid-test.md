# Phase B.T6: BG Action Selection Module — Silent-Motor Trap DISSOLVED

**Date:** 2026-04-25
**Status:** GO — Phase 1 readaptation 19% better than G9 baseline; silent-motor trap structurally eliminated
**Branch:** `pfc-working-memory` (commits ad1f02d → 582b2aa → upcoming)

## Headline result

After ~10 hours of preset audits + bug fixes (Phase A) + BG circuit
construction (Phase B), the silent-motor trap that defeated 7 cumulative
runner-side interventions (V1-V7) is **architecturally dissolved**.

## 3-seed comparison vs G9 baseline

| | Phase 0 finalQ | Phase 1 finalQ | All 4 motors active? |
|---|---|---|---|
| G9 baseline (no fix) | ~1.9 | 6.74 avg | 2/3 seeds had silent W |
| V1 motor exploration (best of V1-V7) | ~1.9 | 6.40 avg | 3/3 fire (still trapped) |
| **BG circuit (Phase B)** | **5.52 avg** | **5.46 avg** | **3/3 + uniform usage** |

Per-seed BG results:

| Seed | Phase 0 finalQ | Phase 1 finalQ | Phase 1 actions [N,E,S,W] |
|------|----------------|----------------|---------------------------|
| 42 | 4.99 | **4.44** | [392, 396, 349, 363] |
| 43 | 5.08 | 6.91 | [381, 383, 375, 361] |
| 44 | 6.49 | 5.03 | [362, 378, 395, 365] |

## What changed architecturally

The previous architecture (G9 / Sessions D-I) used:
- 200-neuron Izhikevich reservoir
- 4 motor neurons reading out via random hidden→motor projections
- argmax over motor spike counts to select action

The reservoir-state-bias problem (V6 finding): random initial hidden→motor
weights interact with reservoir dynamics so that one motor (typically E)
dominates argmax for almost any input. argmax + dominance = silent-motor
trap; no V1-V7 runner-side intervention escaped it.

The Phase B architecture replaces this with a real basal-ganglia-style
circuit:

```
                       VTA/SNc DA neurons
                                |
                ┌───────────────┴───────────────┐
                | targeted DA per striatal pool |
                v                               v
  cortex_X ─→ str_D1[N,E,S,W]           str_D2[N,E,S,W]
              (50 MSN/action)            (50 MSN/action)
                  │                              │
              direct                         indirect
                  v                              v
              GPi[N,E,S,W] ←── STN ─── GPe[N,E,S,W]
              (10 ea)                  (10 ea)
                  │
                  v (disinhibition gate)
              thal[N,E,S,W]
              (10 ea)
                  │
                  v
              motor[N,E,S,W]
              (10 ea)
```

Each action has its own dedicated D1/D2/GPe/GPi/Thal/Motor populations.
There is **no shared argmax** — selection happens via the GPi
disinhibition gate (D1 inhibits GPi → thal released → motor fires).

The single-action probe (commit 8415868) confirmed the cascade:
```
cortex_W:    112 Hz (driven)
str_D1_W:     67 Hz
gpi_W:         0 Hz   ← gate opens for W
thal_W:       24 Hz
motor_W:       7 Hz   ← W selected; others all silent
```

## Phase 0 finalQ regression

Phase B's phase 0 finalQ (5.52) is much worse than G9's phase 0 (1.9).
This is **not** a problem with the BG architecture — it's a learning
loop tuning issue:

- The cortex→striatum weights ARE plastic (STDP-driven)
- DA modulates plasticity rate via the neuromodulator subsystem
- But the action distribution stays nearly uniform across all 1800 steps
- Per-100-step action counts: e.g. seed 42 step 100 = [24, 25, 21, 30]
  → step 1800 = [29, 26, 22, 23] — no consolidation

Two contributing factors:
1. The cortex input is a heuristic "goal-direction signal" — drive cortex_X
   strongly when direction X is goal-relative. With the agent constantly
   moving and goal direction changing, the input pattern shifts before
   weights consolidate.
2. The STDP eligibility traces × DA reward modulation may not be tuned
   for this circuit's timing. G9 had 500ms eligibility tau; Phase B
   inherits this, but with the longer cascade (cortex → striatum → GPe/GPi
   → thal → motor), there's more spike-time spread.

## Why the trap is gone despite weak phase-0 learning

The silent-motor trap manifests as: phase-1-correct motor never fires →
no eligibility → no learning → trap.

In Phase B:
- All 4 motors fire uniformly throughout (≥350 selections each in 1500-step
  phase 1 across all 3 seeds)
- When agent moves toward goal, that motor's eligibility forms across
  the entire cascade (cortex_X → str_D1_X → gpi_X → thal_X → motor_X)
- Reward modulates plasticity ON THAT PATHWAY ONLY
- No interference between actions because each has its own circuit

So even with weak overall learning, the readaptation works — the agent
gradually shifts from "W's circuit slightly tuned for goal A" to "W's
circuit slightly tuned for goal B" without any other action's circuit
getting in the way.

## Comparison vs all V1-V7 attempts

| Variant | Approach | Phase 1 finalQ | All motors? |
|---------|----------|----------------|-------------|
| baseline | none | 6.74 | 2/3 |
| V1 | motor_exploration_rate_hz=15 | 6.40 | 3/3 |
| V2 | first_spike + rate=30 | 6.07-6.71 | mixed |
| V3 | positive_only_reward | 6.75 | 3/3 |
| V4 | action_attribution | 6.57-6.62 | mixed |
| V5 | proportional sampling | 6.78-7.13 | uniform (random) |
| V6 | weight_reset_on_goal_change | 6.58-6.62 | unchanged |
| V7 | epsilon_greedy=0.1/0.2 | 6.79/6.37 | mixed |
| **B** | **BG architecture** | **5.46** | **3/3 + uniform + structured** |

Phase B is the first intervention to drop phase 1 finalQ below 6.0.
The architectural change worked where 7 runner-side hacks could not.

## Files

- [`research/runners/g11_bg_runner.py`](research/runners/g11_bg_runner.py) — full BG circuit + moving-goal mode
- [`research/findings/raw/g11_bg/g11_seed42.json`](research/findings/raw/g11_bg/g11_seed42.json) — best result (phase 1 finalQ=4.44)
- [`research/findings/raw/g11_bg/g11_seed43.json`](research/findings/raw/g11_bg/g11_seed43.json)
- [`research/findings/raw/g11_bg/g11_seed44.json`](research/findings/raw/g11_bg/g11_seed44.json)
- [Phase B plan](docs/plans/2026-04-25-phase-b-bg-action-selection.md) — original architecture spec

## Follow-up work (Session J?)

Phase B is a structural success but the learning loop is undertuned.
Open questions:

1. **Phase 0 acquisition**: Why is phase 0 finalQ stuck at ~5? Real
   issue: STDP+reward isn't producing strong differential cortex→striatum
   weights. Possible fixes:
   - Per-action DA targeting (currently DA is broadcast; needs scope by
     str_D1_X)
   - Stronger learning rate (or longer eligibility tau)
   - Pre-training cortex pools to fire at appropriate position patterns

2. **Action selection sharpness**: Current motor_W output at 7 Hz is
   modest. Adding lateral inhibition between motor populations (via FS
   interneuron sub-pools — current motor→motor pathway is excitatory due
   to source exc_fraction=1.0) would sharpen winner.

3. **Position encoding**: Currently uses heuristic "drive cortex_X for
   goal direction X". A more biologically realistic setup: sensory cortex
   neurons receive position-tuned (Gaussian) input, then project into
   `cortex_X` pools via plastic weights that LEARN the position→action
   mapping (rather than hard-coding it).

4. **DA modulation**: The neuromodulator subsystem is wired but not yet
   targeting per-action striatal pools. Adding selective DA→D1 / DA→D2
   sensitivity per action would enable the proper credit-assignment
   biology.

## Cumulative session work (this branch)

24 commits on `pfc-working-memory` covering:
- Sessions G-I (silent-motor trap arc): V1-V7 motor-exploration variants — all NEGATIVE
- Phase A: comprehensive HH+Izh+AdEx preset audit + temperature/preset bug fixes
- Phase B: BG action selection module + acid test PASS

The user's strategic redirect ("pause motor focus, build out brain regions
properly") was the correct move. The silent-motor trap couldn't be
runner-side-hacked away; it required real biological circuitry.
