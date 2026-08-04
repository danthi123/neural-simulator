---
type: preregistration
status: active
date: 2026-08-04
mechanism: neural-vocal-action-credit-v11-action-boundary
runner: research/runners/_vocal_action_credit_gate_v11_boundary.py
---

# Gate B v11: neural action boundary engagement

**Filed before the V11 runner exists.** V11 asks whether neural motor output can
create a bounded, self-terminating action-active state that prevents a second
vocal action while preserving local policy eligibility. It does not deliver
reward, change a policy weight, or establish conversation.

Construction seed `991` may select one operating point from the finite ladder
below. Seed `1` is the only executable reserved engagement seed. Formal,
development, and held-out phases remain sealed and unassigned.

## Fixed substrate

Start from the complete V10 selector-policy topology and configuration:

- Gate A v2 selector, including symmetric practice arousal and proposal OU
  noise;
- the four plastic `proposal_X -> str_d1_X/str_d2_X` policy routes at density
  `1.0`, weight `400`, jitter `0.05`, bounds `[0, 600]`;
- coactivity eligibility with `300 ms` eligibility decay, `80 ms`
  presynaptic-trace decay, threshold `0.001`, and scale `20.0`; <!--derived-->
- reward learning rate, reward signal, and reward baseline exactly `0`;
- STDP, Hebbian learning, homeostasis, structural plasticity, and the
  neuromodulator subsystem disabled.

Add exactly these regions:

| Region | Count | Type | External current |
|---|---:|---|---:|
| `action_corollary` | 24 | cortical RS excitatory | `0` always |
| `proposal_stop_fs_0/1` | 16 each | cortical FS inhibitory | `0` always |
| `commit_stop_fs_0/1` | 16 each | cortical FS inhibitory | `0` always |

Add exactly these symmetric routes:

| Route | Density | Weight | Receptor / gate |
|---|---:|---:|---|
| `motor_0/1 -> action_corollary` | `1.0` | `30` | AMPA, `boundary_motor_copy` |
| `action_corollary -> action_corollary` | `0.35` | ladder below | AMPA, `boundary_recurrence` |
| `action_corollary -> each *_stop_fs_*` | `1.0` | `30` | AMPA, branch gate |
| `proposal_stop_fs_X -> proposal_X` | `0.70` | `16` | GABA-A |
| `commit_stop_fs_X -> commit_X` | `0.70` | `16` | GABA-A |
| `commit_stop_fs_X -> motor_X` | `0.70` | `16` | GABA-A |

All new jitter is `0`. There is one route declaration per ordered region pair.
The runner must assert that every NMDA source is excitatory, every GABA source
is inhibitory, the two channels have identical local coordinates and weights,
and no route targets D1 or D2 directly from a stop population.

Enable ordinary global NMDA with the unchanged ratio `0.4`, decay `100 ms`,
rise `3 ms`, and magnesium concentration `1.0 mM`. Mark only
`action_corollary` as NMDA-enabled. A default-preserving selector-builder
option must mark `commit_0/1` NMDA-disabled for V11; every older caller retains
the current default. Do not use `nmda_slow`, GABA-B, coincidence plateaus,
graded plateaus, `couple_gate_to_pool()`, a host threshold callback, or a
winner-specific gate.

The existing `selector_reset` population remains structurally present for
matched V10 ancestry, but its external current is exactly zero at every V11
step. No simulator state array may be cleared between trials or conditions.

## Locked construction ladder

Construction uses seed `991` and runs on NumPy and CuPy. All values above stay
fixed. Test recurrent weights in this order and no other order:

`0.25`, `0.50`, `1.00`, `2.00`.

For each value, construct a fresh bridge on each backend and run:

1. `80` neutral warmup steps;
2. `600` weak-arousal catch steps with practice current `250 pA`;
3. `3000` neutral recovery steps;
4. `600` full-action steps with practice current `1000 pA`;
5. `3000` neutral recovery steps;
6. a second `600` full-action step window; and
7. a final `3000` neutral recovery steps.

The first ascending weight qualifies only if, on both backends:

- the catch has no motor threshold crossing and no boundary-population spike;
- both action windows have a unique first motor crossing, loser ratio at most
  `0.25` at crossing, and no later loser crossing;
- first corollary spike is strictly after first motor spike, and first local
  stop-FS spike is strictly after first corollary spike;
- for decisions by action step `450`, the intact boundary has at least one
  spike at or after action step `550`;
- after each drive withdrawal, all boundary populations have zero spikes in
  the final `1000` neutral steps and corollary NMDA conductance is at most
  `1e-6`; and
- the same uninterrupted bridge initiates the second action without reset.

Archive every attempted row and select the first qualifying weight. Do not
average configurations or choose the largest margin. If no weight qualifies,
return `CONSTRUCTION_NO_GO` and do not consume seed `1`. No other current,
weight, density, time constant, duration, threshold, or neuron count may be
tuned from this preregistration.

## Reserved-seed protocol

After committing the runner and its source manifest, execute seed `1` once on
CuPy. Construct these six seed-identical conditions separately:

1. `intact`;
2. `motor_copy_lesion`: `boundary_motor_copy=0`;
3. `proposal_stop_lesion`: both corollary-to-proposal-stop gates `0`;
4. `commit_stop_lesion`: both corollary-to-commit-stop gates `0`;
5. `recurrence_lesion`: `boundary_recurrence=0`; and
6. `coactivity_lesion`: only eligibility coactivity disabled.

Every lesion gate is fixed before the first simulation step and never changes
in response to activity. All other topology, weights, currents, timing, and RNG
initialization are identical. Policy transmission remains `1`; its plasticity
and coactivity-input gains are `1` only during each action window and `0`
during neutral phases. This fixed symmetric measurement gate may not inspect a
winner. Every complete weight array must remain byte-identical.

Warm up for `80` neutral steps, then run `12` trials in the same uninterrupted
brain. Each trial has:

1. `600` full-action steps under unchanged symmetric `1000 pA` practice drive;
2. `100` neutral delayed-consequence steps; and
3. `3000` neutral recovery steps.

There is no reset phase. Phases always run their complete fixed duration. The
first unique motor channel to accumulate `12` spikes is the action. A row is
clean only when the loser has at most `25%` of the winner's spikes at that
crossing and never reaches `12` spikes later in the same action window.

## Required telemetry and validity

Record per-step firing counts for every region, complete per-step firing
hashes, first spike and first threshold times, boundary onset and final spike,
motor counts at crossing and action end, all lesion gates, complete initial and
final weight hashes, raw and net policy eligibility by route, complete
coactivity-trace bounds, corollary NMDA state, recovery latency, neuron and
synapse counts, CuPy device and memory, and wall time.

All validity checks precede the engagement verdict:

1. seed `1` is the only reserved seed and every other phase is sealed;
2. the fixed topology, symmetry, polarity, NMDA scope, and host boundary pass;
3. initial weights and complete initial firing state match across conditions;
4. `motor_copy_lesion` matches intact firing and RNG history through the first
   motor spike inclusive on every trial;
5. proposal, commit, and recurrence lesions match intact through the first
   corollary spike inclusive; the coactivity lesion matches intact firing for
   the complete run;
6. first corollary activity follows motor activity, and local FS activity
   follows corollary activity, in every scored intact trial;
7. both actions occur cleanly at least three times in intact, otherwise return
   `UNDEFINED_ACTION_COVERAGE` without rerunning;
8. pretrial eligibility and coactivity traces meet V10's exact first-trial and
   bounded later washout rules;
9. no weight changes in any condition, including the diagnostic clip-path
   control inherited from V10;
10. `selector_reset` external current is exactly zero at every step; and
11. saving intact state after an action and loading it into a fresh bridge
    restores every allocated slow-conductance array byte-exactly. The focused
    legacy/default checkpoint regressions must also pass.

## Locked causal criteria

`ENGAGEMENT_GO` requires all validity checks and all criteria below.

### One action and recovery

- Intact has at least `11/12` clean trials and both actions have at least three.
- In each clean intact trial decided by step `450`, boundary activity reaches
  at least action step `550`.
- In every intact recovery, boundary populations have zero spikes in the final
  `1000` steps and corollary NMDA conductance is at most `1e-6`.
- At least `11/12` later trials initiate an action after the preceding recovery,
  proving same-brain reuse without host reset or state clearing.

### Causal branches

- `motor_copy_lesion` has at least `9/12` rows in which the initially losing
  channel crosses later, or its clean rate is at least `0.75` below intact.
- After the matched intact boundary-onset step, the proposal-stop lesion's
  summed proposal+D1+D2 spikes have median at least `2x` intact and at least
  `100` additional spikes across the 12 trials.
- The commit-stop lesion has at least `9/12` later-loser crossings or median
  post-decision motor spikes at least `2x` intact.
- The recurrence lesion preserves the first corollary volley but reduces
  median boundary duration by at least `50%` and reduces clean rate by at
  least `0.50`. If the core boundary passes but this criterion alone fails,
  return `QUALIFIED_REMOVE_RECURRENCE`; do not open policy learning until a
  separately locked recurrence-free confirmation removes that branch.

### Eligibility remains local

Using only clean intact trials, separately for D1 and D2, each action, and the
decision and pre-outcome snapshots:

- every net selected eligibility is positive;
- median selected/loser ratio is at least `4.0`;
- mean loser eligibility is at most `25%` of mean selected eligibility;
- selected exceeds loser in at least `80%` of rows; and
- mean selected-minus-loser margin is positive.

The coactivity lesion's maximum policy eligibility must be at most `1%` of
intact mean selected eligibility for D1 and D2. Unclean rows remain archived
but are excluded from winner-labelled eligibility aggregates.

## Performance ceiling

On the same CuPy device, separately build the unchanged V10 topology and the
locked V11 intact topology at seed `1`. After `200` unscored warmup steps, time
`5000` fixed-drive steps three times per topology. Report all repetitions,
median wall time per step, persistent array bytes, and topology counts.

V11 must be no more than `1.50x` the V10 median wall time per step and no more
than `1.50x` its persistent array bytes. Exceeding either ceiling returns
`PERFORMANCE_NO_GO` even if the behavioral criteria pass.

## Stop rule

Do not rerun seed `1`, relax a criterion, shorten the `600`-step window, add a
host reset, or open reward learning after inspecting the artifact. A validity
failure is `UNDEFINED_*`; a valid causal failure is `ENGAGEMENT_FAIL`; neither
authorizes tuning. File the result and return to a new biological evidence gate.

Only `ENGAGEMENT_GO` permits a separate policy-learning preregistration. That
later gate still requires contingent versus reward-count-matched yoked
learning, acquisition and expression lesions, exact restoration, fresh seeds,
same-brain reversal, and a changed future neural action probability.

## Evidence read before filing

- `2026-08-03-neural-vocal-credit-gateB-v11-action-boundary-RESEARCH-GATE.md`
- `2026-08-03-neural-vocal-credit-gateB-v10-policy-eligibility-UNDEFINED.md`
- Kandel, *Principles of Neural Science*, 6e, chapter 38, pp. 941-946.
- Jin and Costa (2010), [Nature/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3477867/).
- Schmidt et al. (2013), [Nature Neuroscience/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3733500/).
- Nelson et al. (2013), [Journal of Neuroscience/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3761045/).
- Schneider et al. (2014), [Nature/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4248668/).
