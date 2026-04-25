# Session G: Motor Exploration Noise — Direct Attack on the Silent-Motor Trap

**Date:** 2026-04-25
**Status:** [TBD — probe running; will fill in once results land]
**Branches:** main (runner change + smoke test)

## Context

After Sessions D.A.4 / E (Route A parallel), Route C scale, NE sensitivity sweep,
and the PFC bistability tuning attempt all failed to fix the silent-motor trap,
this session attacks the failure mechanism directly: **exploration noise at the
motor layer**.

## The Silent-Motor Trap, Restated

In the moving-goal task `(6,6) → (1,6)`, an agent that learns Phase 1 by mostly
firing motors {N, E} cannot readapt in Phase 2 (which requires W). Why:

1. STDP only forms eligibility traces between pre-post pairs that *both fire*
2. Phase-1-silent motors (e.g., W) never fire → no eligibility on hidden→W synapses
3. No eligibility means no reward-mediated weight update can reach those synapses
4. So even when reward sign flips and W *would* be the right action, the W
   synapses stay frozen at their initial random values
5. The agent stays glued to N/E even though it's now incorrect

Direct evidence (`research/findings/raw/g9/g9_moving_relaxed_argmax_seed*.json`):

| Seed | Phase 0 actions [N,E,S,W] | Phase 1 actions [N,E,S,W] | Phase 1 finalQ dist |
|------|---------------------------|---------------------------|---------------------|
| 42   | [76, 199, 25, **0**]      | [368, 994, 138, **0**]    | 7.55                |
| 43   | [61, 164, 25, 50]         | [404, 677, 102, 317]      | 6.16                |
| 44   | [196, 90, 11, **3**]      | [906, 485, 98, **11**]    | 6.85                |

Seed 42's W-motor was completely silent across both phases. Seed 44's W fired
0.7% of the time. Phase 1 finalQ ≈ 6-7.5 means the agent is ~6-7 cells away
from the goal — essentially stuck on the opposite side of the grid.

Why prior interventions didn't help:
- **Route A (parallel subprocesses)**: same RNG seeds → same silent motors
- **Route C (5,068 neurons)**: more hidden capacity, but motor layer unchanged → same trap
- **NE sensitivity sweep**: tonic excitability adds bias to ALL motors equally,
  doesn't break the asymmetry; without independent noise sources, all motors
  shift together
- **PFC bistability**: would give richer goal-context but doesn't change which
  motors fire; the silent-motor trap is a motor-layer eligibility problem, not
  a context-tracking problem

## Intervention: Stochastic Motor Spike Injection

During the stimulus integration window (0–150 ms per trial), inject a Poisson
spike train at rate `motor_exploration_rate_hz` into each motor neuron
independently. The spikes are real spike-driving currents
(`spike_current_pA=1000`, `duration=2 ms`), so each Poisson event reliably
produces a motor spike.

At 15 Hz, each motor expects ~2.25 spurious spikes per 150 ms stimulus window
(0.75 in the 50–150 ms readout window). This is enough that:
- All 4 motors fire at least occasionally regardless of upstream input
- STDP can form eligibility traces on hidden→silent-motor synapses whenever
  the random spurious motor spike happens to follow recent hidden activity
- When reward arrives, the eligibility converts to weight changes
- The "right" motor for the new goal can have its hidden→W synapses
  potentiated even if it never won the action competition during phase 1

The spike rate is low enough that it doesn't dominate action selection
(typical motor in winning condition fires ~20+ spikes per readout window).
Action selection still tracks the reward-driven weights; exploration just
stops the eligibility from being structurally zero.

### Biological grounding

- Tonic dopamine drives spontaneous striatal/cortical motor activity
  baseline (Schultz 2007). Phasic vs. tonic DA is a known control axis
  for exploration vs. exploitation.
- Cholinergic interneurons add stochastic excitation in BG (Apicella 2007).
- Cortical motor neurons have spontaneous baseline firing (~5-15 Hz)
  even in the absence of task-related drive (Sherrington tradition).
- Action exploration in real animals is well-documented (Tervo 2014;
  Marshall 2016): mice and rats produce stochastic motor variability
  during learning that tracks DA dynamics.

### CS grounding

This is structurally identical to:
- ε-greedy in tabular RL (random action with probability ε)
- Entropy regularization in policy gradient (encourages action distribution
  away from determinism)
- Noisy-net DQN (parameter-space noise on Q-network for exploration)
- Boltzmann exploration (softmax temperature controls action stochasticity)

All of these solve the same exploration/exploitation dilemma. Motor exploration
noise is the spiking-network equivalent.

## Implementation

`research/runners/g9_runner.py`:
- New kwarg `motor_exploration_rate_hz` (default 0.0 — backward compatible)
- New kwargs `motor_exploration_current_pA` (default 1000.0) and
  `motor_exploration_spike_ms` (default 2.0)
- When `rate > 0`, a second `StimulusChannel` of type `POISSON_SPIKE_TRAIN`
  is added alongside the sensor channel during each trial's stimulus window
- Poisson generation handled by existing `StimulusManager` infrastructure
  (`experiment/stimulus.py`); no GPU code change needed

Smoke test `tests/test_g9_runner_smoke.py::test_g9_smoke_motor_exploration`:
- 30-step episode with `motor_exploration_rate_hz=15.0`
- Verifies: every motor fired at least once, reservoir frozen, results saved

## Probe Design

`research/run_g9_motor_exploration.py`:
- Scenario: relaxed moving-goal `(6,6)→(1,6)` — same as Session D.A.4
- `n_steps = 1800` (Phase 0: 0-300, Phase 1: 300-1800)
- 3 seeds × 2 conditions = 6 runs
- Conditions:
  - **baseline**: `motor_exploration_rate_hz = 0` (replicates D.A.4)
  - **treatment**: `motor_exploration_rate_hz = 15`

Pass criteria:
1. **Every motor active in Phase 1** for all 3 treatment seeds
2. **Phase 1 finalQ < 4** for at least 2/3 treatment seeds (vs. 6-7.5 baseline)
3. Baseline must reproduce the prior negative result (sanity check)

## Results — V1 (rate=15, argmax)

| Condition | seed | Phase 1 finalQ | atGoal | actions [N, E, S, **W**] |
|-----------|------|----------------|--------|--------------------------|
| baseline  | 42   | 7.25           | 0      | [350, 998, 152, **0**]   |
| baseline  | 43   | 6.04           | 0      | [356, 707, 123, **314**] |
| baseline  | 44   | 6.92           | 0      | [942, 447, 98, **13**]   |
| baseline avg | —  | **6.74**       | 0.0    | all-4-motors: 2/3        |
| treatment | 42   | 6.90           | 0      | [370, 916, 213, **1**]   |
| treatment | 43   | 5.47           | 13     | [388, 581, 181, **350**] |
| treatment | 44   | 6.84           | 0      | [807, 530, 142, **21**]  |
| treatment avg | —  | **6.40**       | 4.3    | all-4-motors: **3/3**    |

**Pass criteria evaluation:**
1. ✓ Every motor active in Phase 1 for all 3 treatment seeds (3/3 vs 2/3 baseline)
2. ✗ Phase 1 finalQ < 4 — best treatment seed reached 5.47, not 4
3. ✓ Baseline reproduced prior D.A.4 result

**Diagnosis (key observation):** Direct inspection of motor spike counts on
seed 42 treatment (`research/findings/raw/g9_motor_exploration/g9_explore15_seed42.json`):

| Motor | Phase 1 mean spikes/step | Phase 1 max spikes/step |
|-------|--------------------------|-------------------------|
| N     | 3.76                     | 8                       |
| E     | 4.78                     | 8                       |
| S     | 4.13                     | 8                       |
| W     | **1.99**                 | **5**                   |

Exploration noise reliably gives W ~2 spikes/step (≈1.5 from 15 Hz Poisson +
~0.5 from sparse hidden→W input). But E gets ~5 spikes/step from its trained
weights + the same noise floor. **argmax always picks E.** W wins as the
selected action exactly **once in 1500 phase-1 steps** for seed 42.

The trap has two layers we hadn't separated:
- **Layer A (eligibility):** silent motors can't acquire eligibility traces.
  Exploration noise fixes this — W has eligibility now.
- **Layer B (action selection):** even with eligibility, argmax + entrenched
  phase-1 weights mean W never wins → reward never differentiates W vs E
  → W's eligibility doesn't translate to weight changes that affect action.

V1 fixes layer A, leaves layer B intact. To break layer B we need either
(i) a more noise-sensitive action selector or (ii) higher noise rate so W
occasionally crosses the argmax threshold.

## Results — V2 (escalation: first_spike + rate=30)

| Condition           | seed | Phase 1 finalQ | atGoal | actions [N, E, S, **W**] |
|---------------------|------|----------------|--------|--------------------------|
| first_spike + 15Hz  | 42   | **8.98**       | 0      | [383, 546, 425, **146**] |
| first_spike + 15Hz  | 43   | 3.21           | 36     | [353, 396, 330, **421**] |
| first_spike + 15Hz  | 44   | 6.03           | 0      | [505, 449, 332, **214**] |
| first_spike avg     | —    | **6.07**       | 12.0   | —                        |
| argmax + 30Hz       | 42   | 7.10           | 0      | [448, 810, 242, **0**]   |
| argmax + 30Hz       | 43   | 6.23           | 0      | [414, 572, 197, **317**] |
| argmax + 30Hz       | 44   | 6.81           | 0      | [706, 614, 161, **19**]  |
| argmax + 30Hz avg   | —    | **6.71**       | 0.0    | —                        |

**first_spike + rate=15 is unreliable**:
- Seed 42 finalQ=8.98 (worse than baseline). Phase 0 actions become spread —
  first_spike is too noise-sensitive, can't establish a clean policy.
- Seed 43 finalQ=3.21 looks good, but actions [353, 396, 330, 421] are
  near-uniform — the agent is random-walking, not goal-directed. Mean dist
  3.21 is close to uniform-walk expected value (~5.5 for goal (1,6) on 8×8).
- Seed 44 finalQ=6.03 modest improvement.

**argmax + rate=30 doesn't help**: 6.71 average is essentially baseline (6.74).
Doubling noise rate adds proportional bias to ALL motors — argmax still
picks E (which has strongest trained weights). Seed 42's W=0 even worse than
V1 rate=15's W=1.

**Combined V1+V2 conclusion: motor exploration noise alone is necessary but
fundamentally insufficient.**

## Diagnosis: Why exploration alone fails

The trap has three layers, not two:

**Layer A — eligibility:** silent motors can't acquire eligibility traces.
*Fixed by exploration noise* (V1 confirmed).

**Layer B — action selection:** even with eligibility, argmax is dominated
by trained-winner motors. *Not addressed by V1 or V2.*

**Layer C — action-blind reward:** When the trained-winner motor (e.g. E)
wins and goes the wrong direction, *negative* reward depresses the
eligibility trace globally. The noise-driven W eligibility — which actually
correlates with goal-directed hidden activity — gets punished alongside
E's wrong-direction eligibility. *Discovered through V1+V2 analysis.*

Layer C means even higher noise (rate=30, rate=60) can't help: the more W
fires, the more its eligibility is exposed to the global negative reward
when E (still winning) goes wrong way.

The fix that addresses Layer C: **positive-only reward**. When the agent
moves toward goal, +1; when it moves away, *0* (not -1). Now:
- E wins, wrong way → no eligibility update → W's eligibility persists
- W wins (rare via noise), correct way → potentiate W's eligibility
- W weights grow over time even though W rarely wins as the action
- Eventually W crosses argmax threshold and starts winning more often

V3 tests this hypothesis.

## Results — V3 (positive-only reward + exploration)

| Condition          | seed | Phase 1 finalQ | atGoal | actions [N, E, S, **W**] |
|--------------------|------|----------------|--------|--------------------------|
| posrew + rate=0    | 42   | 6.97           | 0      | [334, 1001, 165, **0**]  |
| posrew + rate=0    | 43   | 6.45           | 0      | [425, 681, 118, **276**] |
| posrew + rate=0    | 44   | 6.87           | 0      | [901, 472, 117, **10**]  |
| **posrew + rate=0 avg** | — | **6.76**    | 0.0    | (≈ baseline 6.74)        |
| posrew + rate=15   | 42   | **7.72**       | 0      | [406, 883, 210, **1**]   |
| posrew + rate=15   | 43   | 5.68           | 2      | [414, 619, 152, **315**] |
| posrew + rate=15   | 44   | 6.86           | 0      | [779, 543, 161, **17**]  |
| **posrew + rate=15 avg** | — | **6.75**   | 0.7    | (worse than V1 avg 6.40) |

**V3 is a NEGATIVE result.** Positive-only reward + exploration is
*slightly worse* than V1 (rate=15 with bipolar reward). The seed-42 case
is particularly stark: 7.72 vs V1's 6.90 — removing punishment made it
worse, not better.

## Diagnosis update

The Layer C theory was wrong. Negative reward turns out to be *useful*: it
depresses entrenched-winner weights when they go wrong, allowing room for
new weights to grow. Removing the negative half of the reward signal
removes that depression — entrenched-winners stay entrenched longer.

The real problem isn't the *sign* of reward, it's that reward is
**globally applied** to all eligible synapses. When E wins and goes wrong,
*both* hidden→E and hidden→W eligibility get depressed. We don't want
hidden→W to be depressed when W wasn't even the chosen action.

The principled fix: **action attribution.** When action `a` is chosen,
zero eligibility on hidden→motor[m] for all m≠a *before* the reward
applies. Then the reward signal selectively updates only the chosen
motor's synapses. V4 tests this.

## Results — V4 (action attribution + exploration)

[TBD — `research/run_g9_motor_exploration_v4.py` running. Two conditions:
- `attr_only`: action_attribution + rate=15 (bipolar reward kept)
- `attr_posonly`: action_attribution + rate=15 + positive-only (full stack)

Expected wall: ~80 min.]

## Discussion

[TBD until V4 lands.]

## Files

- [`research/runners/g9_runner.py`](research/runners/g9_runner.py) — added kwargs
- [`research/run_g9_motor_exploration.py`](research/run_g9_motor_exploration.py) — probe driver
- [`tests/test_g9_runner_smoke.py`](tests/test_g9_runner_smoke.py) — `test_g9_smoke_motor_exploration`
- [Raw data](research/findings/raw/g9_motor_exploration/) — JSON per seed×condition
