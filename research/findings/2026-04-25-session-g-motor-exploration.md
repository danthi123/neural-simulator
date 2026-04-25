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

| Condition       | seed | Phase 1 finalQ | atGoal | actions [N, E, S, **W**] |
|-----------------|------|----------------|--------|--------------------------|
| attr_only       | 42   | 7.47           | 0      | [409, 890, 200, **1**]   |
| attr_only       | 43   | 5.40           | 6      | [392, 586, 178, **344**] |
| attr_only       | 44   | 6.83           | 0      | [788, 549, 143, **20**]  |
| **attr_only avg** | — | **6.57**       | 2.0    | (worse than V1 6.40)     |
| attr_posonly    | 42   | 7.61           | 0      | [440, 835, 225, **0**]   |
| attr_posonly    | 43   | 5.42           | 4      | [389, 664, 161, **286**] |
| attr_posonly    | 44   | 6.84           | 0      | [801, 537, 139, **23**]  |
| **attr_posonly avg** | — | **6.62**  | 1.3    | —                        |

V4 is also a NEGATIVE result. attr_only is slightly *worse* than V1
(6.57 vs 6.40). Combining attr + positive-only doesn't help either.

## Diagnosis update (V4)

Action attribution as implemented (zero non-chosen-motor eligibility before
reward) actively *prevents* hidden→silent-motor weights from changing.
Every step where W isn't selected (≈99%), W's nascent eligibility is wiped.
Hidden→W weights stay at random initial values forever — they neither
grow (no positive reward) nor shrink (no negative reward applied to them).

In V1 (no attribution), at least the negative reward when E went wrong
*depressed* hidden→E enough to give W a chance over time. With attribution,
that depression is selective to E, but W's lack of any updates means W
never catches up.

The fundamental block: **W must be SELECTED occasionally** for any of these
mechanisms to matter. Without W winning, no path exists for hidden→W weights
to grow.

## Results — V5 (proportional action selection)

| Condition       | seed | Phase 0 finalQ | Phase 1 finalQ | actions [N, E, S, **W**] |
|-----------------|------|----------------|----------------|--------------------------|
| prop + rate=0   | 42   | 3.76           | **8.67**       | [349, 481, 432, **238**] |
| prop + rate=0   | 43   | 7.60           | 6.19           | [352, 420, 359, **369**] |
| prop + rate=0   | 44   | 3.17           | 5.47           | [421, 436, 343, **300**] |
| **prop+rate=0 avg** | — | **4.84**     | **6.78**       | (~ baseline 6.74)        |
| prop + rate=15  | 42   | 3.73           | 9.27           | [358, 479, 425, **238**] |
| prop + rate=15  | 43   | 5.33           | 5.18           | [376, 401, 352, **371**] |
| prop + rate=15  | 44   | 3.80           | 6.94           | [414, 421, 363, **302**] |
| **prop+rate=15 avg** | — | **4.29**    | **7.13**       | (worse than baseline)    |

**V5 is a NEGATIVE result.** Proportional sampling is too aggressive an
exploration policy: phase 0 finalQ degrades from 1.99 (baseline) to 4-5
across all V5 seeds. The agent can't establish a coherent policy in either
phase — actions are near-uniform (`[~375, ~430, ~370, ~350]` ≈ 25% each).
Phase 1 finalQ ≈ 5.5 (the random-walk expected value on 8×8 grid for
goal (1,6)) confirms the agent is essentially random-walking.

Counterintuitively, V5 **does break the silent-motor invariant** (W=238-371
across seeds, vs baseline 0-13). But it does so by destroying *all*
learning, not by selectively unblocking the silent motor.

## Final cumulative results

| Variant | avg Phase 1 finalQ | Silent-motor invariant | Verdict |
|---------|---------------------|------------------------|---------|
| baseline (rate=0, argmax, bipolar) | 6.74 | 2/3 motors silent | reference |
| **V1: rate=15 + argmax + bipolar** | **6.40** | **3/3 fire** ✓ | **partial GO** |
| V2: rate=30 + argmax | 6.71 | 3/3 (mixed) | NO-GO |
| V2: rate=15 + first_spike | 6.07 | 3/3 (random) | NO-GO (misleading) |
| V3: rate=15 + posrew | 6.75 | 3/3 | NO-GO |
| V3: rate=0 + posrew | 6.76 | 2/3 | NO-GO |
| V4: rate=15 + attr_only | 6.57 | 3/3 (mixed) | NO-GO |
| V4: rate=15 + attr + posrew | 6.62 | mixed | NO-GO |
| V5: rate=0 + proportional | 6.78 | 4/4 (random) | NO-GO |
| V5: rate=15 + proportional | 7.13 | 4/4 (random) | NO-GO |

## Discussion

**V1 (motor exploration noise) is the only real contribution.** With
`motor_exploration_rate_hz=15` + standard argmax + bipolar reward:
- Every motor fires occasionally in phase 1 (silent-motor invariant broken)
- Phase 0 learning preserved (finalQ 1.92 ≈ baseline 1.99)
- Phase 1 finalQ slightly improves (6.40 vs 6.74 baseline)

The improvement is modest because V1 fixes Layer A (silent-motor *firing*)
but not the deeper trap layers. Further direct attacks on Layer B
(action-selection lock-in) and Layer C (action-blind reward) all fail or
make things worse:
- Removing punishment (V3 positive-only) loses information about wrong
  actions; bipolar reward turns out to be useful
- Action attribution (V4) starves silent motors of weight updates entirely
- Proportional sampling (V5) destroys all learning by adding too much noise

**Architectural conclusion:** The silent-motor trap on this 200-neuron
reservoir + 4-motor R-STDP architecture is robust against shallow
interventions on motor activity, reward sign, or action selection. The
trap is a structural property of:
1. Argmax + trained-winner-dominance creating a fixed point that exploration
   noise alone can't escape
2. Global reward eligibility coupling that propagates phase-1-correct
   weight changes back into hidden→silent-motor synapses inappropriately
3. STDP eligibility tau (~500ms) being too short to bridge the readaptation
   gap when the silent motor only fires from sparse noise events

Possible deeper interventions for a hypothetical Session H (NOT in scope):
- **Per-action local circuits**: separate inhibitory pools for each motor
  preventing action-blind weight updates at the architectural level
- **Goal-change detection + targeted weight reset**: when reward variance
  exceeds threshold, reset hidden→motor weights toward initial values
- **Adaptive learning rate**: boost LR after goal change to overcome
  entrenched weights faster
- **Curriculum learning**: start with similar phase-0/phase-1 goals and
  gradually increase distance — gives system a chance to learn the
  general "goal-direction" representation before the hard test
- **More biological architectures**: thalamic relay layer with selective
  attention; basal ganglia disinhibition for selection; hippocampal
  prediction-error-driven replay during goal change

## V1 recommendation

Despite the partial nature of the win, V1's `motor_exploration_rate_hz=15`
should become the **default G9 recommendation** because:
1. It guarantees the silent-motor invariant (every motor fires occasionally)
2. It marginally improves phase-1 readaptation (6.40 vs 6.74)
3. It's biologically grounded (tonic dopamine / cortical baseline activity)
4. It's cheap (no GPU code change, just adds a `StimulusChannel`)

The other variants (V2-V5) are documented as alternatives that explored
the design space but did not yield improvements over V1.

## Files

**Code:**
- [`research/runners/g9_runner.py`](research/runners/g9_runner.py) — added
  `motor_exploration_rate_hz`, `positive_only_reward`,
  `action_attribution_eligibility`, `action_selection="proportional"`
- [`tests/test_g9_runner_smoke.py`](tests/test_g9_runner_smoke.py) — 4 new
  smoke tests, all 9 G9 smokes pass

**Probes:**
- [V1](research/run_g9_motor_exploration.py) — baseline vs rate=15
- [V2](research/run_g9_motor_exploration_v2.py) — first_spike + argmax+rate=30
- [V3](research/run_g9_motor_exploration_v3.py) — positive-only reward
- [V4](research/run_g9_motor_exploration_v4.py) — action attribution
- [V5](research/run_g9_motor_exploration_v5.py) — proportional sampling
- [analyzer](research/analyze_motor_exploration.py)

**Raw data:** [`research/findings/raw/g9_motor_exploration/`](research/findings/raw/g9_motor_exploration/)

**CLAUDE.md:** "Motor Exploration Noise (Session G)" section added.

## Files

- [`research/runners/g9_runner.py`](research/runners/g9_runner.py) — added kwargs
- [`research/run_g9_motor_exploration.py`](research/run_g9_motor_exploration.py) — probe driver
- [`tests/test_g9_runner_smoke.py`](tests/test_g9_runner_smoke.py) — `test_g9_smoke_motor_exploration`
- [Raw data](research/findings/raw/g9_motor_exploration/) — JSON per seed×condition
