# Session G: Motor Exploration Noise — Plan

> **Status:** Implementation complete (smoke tested), validation probe running.
> Plan kept for post-hoc reference and to outline contingency Session H.

## Goal

Break the silent-motor trap that defeated:
- Route A (parallel subprocess seed-replicas, Session D.C)
- Route C (5,068-neuron reservoirs, Session E.2.5 follow-up)
- NE sensitivity sweep (Session E.3 limitation; tonic excitability axis 0..150)
- PFC bistability tuning (Session F; homeostatic floor blocks persistent activity)

Hypothesis: motor neurons that are silent in phase 1 cannot acquire eligibility
traces under STDP, so reward-mediated weight updates never reach hidden→silent
synapses. Adding stochastic motor input (Poisson spike train) ensures every
motor fires occasionally regardless of upstream drive; STDP can then form
positive eligibility on hidden→silent-motor synapses; reward converts those
into weight changes.

This is structurally identical to ε-greedy exploration in tabular RL, just at
the spike-event level instead of the action-distribution level.

## Architecture

```
                 ┌──────────────────────────┐
sensor input ───►│  StimulusChannel "sensor" │──┐
                 └──────────────────────────┘  │
                                                ├─► current to all neurons
                 ┌──────────────────────────┐  │
motor explore ──►│  StimulusChannel        │──┘
(Poisson 15Hz)   │  "motor_explore"         │
                 │  → motor neurons only    │
                 └──────────────────────────┘
```

Both channels active during the 0–150 ms stimulus window per trial.
Neither active during the reward-hold window (channel duration_ms expires).

## Tasks

### Task 1: Add `motor_exploration_rate_hz` kwarg to G9 runner ✓

**Files:** `research/runners/g9_runner.py`

- Add `motor_exploration_rate_hz=0.0` (default = backward compat)
- Add `motor_exploration_current_pA=1000.0`
- Add `motor_exploration_spike_ms=2.0`
- In per-step stimulus setup, append a second `StimulusChannel` of type
  `POISSON_SPIKE_TRAIN` targeting `layout["motor_idx"]` when rate > 0
- Save `motor_exploration_rate_hz` to results JSON

### Task 2: Smoke test ✓

**Files:** `tests/test_g9_runner_smoke.py::test_g9_smoke_motor_exploration`

- 30-step episode, rate=15 Hz
- Assert every motor fired at least once (sum motor_counts > 0 per motor)
- Assert reservoir still frozen (drift_max == 0)
- All 6 G9 smoke tests must pass (no regression)

### Task 3: Validation probe ✓ (running)

**Files:** `research/run_g9_motor_exploration.py`

- Same scenario as Session D.A.4: relaxed moving-goal `(6,6)→(1,6)`,
  `n_steps=1800`, 3 seeds {42, 43, 44}
- Conditions: rate ∈ {0, 15} (baseline + treatment)
- Output JSONs + summary.csv

### Task 4: Analysis ✓

**Files:** `research/analyze_motor_exploration.py`

- Per-run silent-motor detection
- Per-condition aggregate phase-0/phase-1 metrics
- Pass-criteria evaluation

### Task 5: Decide on rate-sweep follow-up [pending]

**File:** `research/run_g9_motor_exploration_ratesweep.py`

If 15 Hz cleared the silent-motor trap, characterize rate-sensitivity at
{5, 30, 60} Hz. Skip if 15 Hz failed (no point characterizing a non-working
intervention).

### Task 6: Document, commit, push [pending]

- Fill in findings doc with actual numbers
- Commit on `pfc-working-memory` branch (which already has the failed PFC
  probe + this pivot — coherent narrative)
- Update CLAUDE.md ✓ (already done)
- Push to remote

## Pass Criteria

1. Every motor active in Phase 1 for all 3 treatment seeds (silent-motor invariant)
2. Phase 1 finalQ < 4 for ≥2/3 treatment seeds (vs. 6.16/6.85/7.55 baseline)
3. Baseline reproduces prior D.A.4 result (sanity check)

## Contingency: Session H if G fails

If motor exploration injects spikes but the agent still doesn't readapt
(silent motors fire BUT phase-1 weights stay too entrenched), escalate to
one of:

### H.A: Per-phase eligibility reset (cheap)
On `goal_change`, clear `cp_eligibility_trace` and (optionally) decay
hidden→motor weights toward their mean. Biologically loose but addresses
the entrenched-weight problem directly.

### H.B: First-spike action selection (cheap)
The current argmax keeps picking phase-1 winners because they accumulated
the most spike count. Switching to first-spike WTA may surface phase-2
candidates earlier — stronger eligibility traces from short-latency hits.

### H.C: Reduce STDP time constants
Faster decay (`stdp_tau` smaller) means eligibility traces from old
phase-1 winners decay faster, freeing room for new pairings.

### H.D: Stronger exploration rate
If 15 Hz isn't enough, try 30-60 Hz. Trade-off: more noise dominates action
selection, but silent-motor invariant is stronger.

H.B + H.D combo is probably the cheapest test if H.A doesn't work alone.

## Open questions

- Does motor exploration interact with the neuromodulator subsystem? E.g.,
  could DA modulate the exploration rate (tonic vs. phasic)? Future work.
- Is there a clean way to make exploration adaptive (high early, low late)?
  Equivalent to epsilon-decay schedules. Future work.
