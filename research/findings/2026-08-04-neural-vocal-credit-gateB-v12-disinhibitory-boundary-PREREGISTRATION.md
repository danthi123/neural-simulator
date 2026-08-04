---
type: preregistration
status: active
date: 2026-08-04
mechanism: neural-vocal-action-credit-v12-disinhibitory-boundary
runner: research/runners/_vocal_action_credit_gate_v12_disinhibitory.py
---

# Gate B v12: motor-disinhibited action boundary

**Filed before the V12 runner exists.** V12 asks whether a normally inhibited
corollary carrier can remain quiet without an action, open only after neural
motor output, recruit symmetric local stopping circuits, and close itself when
motor output ends. It does not deliver reward, dopamine, or policy updates.

Construction seed `997` is the only construction seed. Seed `2` is the only
reserved engagement seed and remains sealed unless construction passes on both
NumPy and CuPy. V11 seed `991` is consumed and V11 seed `1` remains sealed for
that retired topology; neither may be used by V12. Development, formal, and
held-out learning seeds are unassigned.

## Fixed substrate

Start from the complete V10 selector-policy topology and configuration:

- Gate A v2 selector with symmetric practice arousal and proposal OU noise;
- four plastic `proposal_X -> str_d1_X/str_d2_X` policy routes at density
  `1.0`, weight `400`, jitter `0.05`, and bounds `[0, 600]`;
- coactivity eligibility with `300 ms` eligibility decay, `80 ms`
  presynaptic-trace decay, threshold `0.001`, and scale `20.0`; <!--derived-->
- reward learning rate, reward signal, and reward baseline exactly `0`; and
- STDP, Hebbian learning, homeostasis, structural plasticity, NMDA, GABA-B,
  dendritic plateaus, and the neuromodulator subsystem disabled.

Use the V11 default-preserving selector-builder option that removes dormant
NMDA tags from `commit_0/1`. No region in V12 is NMDA-enabled.

Add exactly these regions:

| Region | Count | Type | External current |
|---|---:|---|---:|
| `action_corollary` | 24 | cortical RS excitatory | `0` always |
| `boundary_guard_som` | 24 | cortical FS inhibitory | `0` always |
| `boundary_vip` | 16 | cortical FS inhibitory | `0` always |
| `proposal_stop_fs_0/1` | 16 each | cortical FS inhibitory | `0` always |
| `commit_stop_fs_0/1` | 16 each | cortical FS inhibitory | `0` always |

The guard and disinhibitor names describe circuit roles only. V12 does not
claim that the generic FS point neurons reproduce SOM or VIP subtype dynamics.

Add exactly these routes, all with jitter `0`:

| Route | Density | Weight | Receptor / gate |
|---|---:|---:|---|
| `practice_arousal -> boundary_guard_som` | `1.0` | `8` | AMPA, `boundary_background` |
| `motor_0/1 -> boundary_vip` | `1.0` | `30` | AMPA, `boundary_disinhibitor_drive` |
| `boundary_vip -> boundary_guard_som` | `1.0` | `8` | GABA-A, `boundary_disinhibition` |
| `boundary_guard_som -> action_corollary` | `1.0` | `8` | GABA-A, `boundary_guard` |
| `motor_0/1 -> action_corollary` | `1.0` | `30` | AMPA, `boundary_motor_copy` |
| `action_corollary -> each *_stop_fs_*` | `1.0` | `30` | AMPA, branch gate |
| `proposal_stop_fs_X -> proposal_X` | `1.0` | `16` | GABA-A |
| `commit_stop_fs_X -> commit_X` | `1.0` | `16` | GABA-A |
| `commit_stop_fs_X -> motor_X` | `1.0` | `16` | GABA-A |

There is no corollary recurrence. There is one route declaration per ordered
region pair. All coordinates and weights are exactly mirrored between channels.
The runner must assert excitatory source identity for every AMPA route,
inhibitory source identity for every GABA route, and no new route to D1, D2,
GPi, thalamus, or `selector_reset`.

The inhibitory values `8` and `16` stay within the project's previously tested
net-inhibitory range. The runner must report maximum inhibitory conductance and
reject any row in which target firing increases under an isolated inhibitory
source-on probe relative to its source-off twin. This catches the known
oversized-conductance rebound artifact without authorizing weight changes.

The existing `selector_reset` remains structurally present and receives exactly
zero current at every step. No state array may be cleared after construction.
No helper may observe a winner, close a route, inject a pulse, or change a gate
in response to activity.

## Locked construction protocol

Build one fresh bridge on NumPy and one on CuPy at seed `997`. There is no
parameter ladder and no backend-specific configuration.

Each backend runs one uninterrupted protocol:

1. `1000` baseline warmup steps at symmetric practice current `250 pA`;
2. `600` no-action catch steps at the same `250 pA`;
3. `3000` baseline recovery steps at `250 pA`;
4. `600` full-action steps at symmetric practice current `1000 pA`;
5. `3000` baseline recovery steps at `250 pA`;
6. a second `600` full-action window at `1000 pA`; and
7. a final `3000` baseline recovery at `250 pA`.

The action threshold and cleanliness definition remain V10's: the first unique
motor population to accumulate `12` spikes is the action; at crossing, the
loser has at most `25%` of the winner's spikes and may never reach `12` later
in the same fixed action window.

Construction passes only if both backends satisfy every criterion:

1. warmup and catch have zero spikes in `boundary_vip`, `action_corollary`, and
   every stopping population, and neither motor reaches threshold;
2. `boundary_guard_som` emits at least one spike in every consecutive
   `100`-step bin of warmup and catch;
3. both action windows contain one clean action;
4. first motor spike strictly precedes first `boundary_vip` and
   `action_corollary` spikes, which each strictly precede the first stopping-
   population spike;
5. guard firing in the `100` steps after first `boundary_vip` activity is at
   most `50%` of its rate in the preceding `100` steps;
6. corollary and all four stopping populations emit at least one spike after
   motor onset in each action window;
7. every recovery has zero disinhibitor, corollary, and stopping spikes in its
   final `1000` steps, while the guard again emits at least one spike in each
   `100`-step bin;
8. the same brain initiates the second action without reset or state clearing;
9. complete initial and final weight hashes are byte-identical; and
10. topology, polarity, symmetry, conductance, reset-current, and host-boundary
    audits pass.

Archive complete telemetry even on the first failure. Any baseline boundary
spike, silent guard bin, action preceding failure, later loser crossing,
inhibitory rebound, or backend disagreement returns `CONSTRUCTION_NO_GO`. Do
not consume seed `2`, alter a duration, add recurrence, or tune a weight.

If the motor-triggered causal order and baseline quiet pass but either action
window later admits the other motor channel, report
`CONSTRUCTION_QUALIFIED_BOUNDARY_TOO_SHORT`. This is still a stop. It authorizes
a new evidence review of slow local refractory mechanisms, not an in-run NMDA
or GABA-B addition.

## Reserved engagement protocol

Only a cross-backend construction pass opens one CuPy execution of seed `2`.
Construct these eight seed-identical conditions separately:

1. `intact`;
2. `guard_lesion`: `boundary_guard=0`;
3. `motor_disinhibitor_lesion`: `boundary_disinhibitor_drive=0`;
4. `disinhibition_lesion`: `boundary_disinhibition=0`;
5. `motor_copy_lesion`: `boundary_motor_copy=0`;
6. `proposal_stop_lesion`: both corollary-to-proposal-stop gates `0`;
7. `commit_stop_lesion`: both corollary-to-commit-stop gates `0`; and
8. `coactivity_lesion`: only eligibility coactivity disabled.

Every gate is fixed before construction and never changes in response to
activity. All other topology, weights, currents, phases, and RNG initialization
are identical. Complete weights remain immutable.

Run `1000` baseline warmup steps and then `12` uninterrupted trials. Each trial
contains `600` full-action steps at `1000 pA` followed by `3000` baseline steps
at `250 pA`. There is no reset phase and every phase runs to its fixed end.
Policy transmission remains `1`; policy plasticity and coactivity-input gains
are `1` only during the fixed action windows and `0` during baseline. These
measurement gates are symmetric and cannot inspect a winner.

## Validity and causal criteria

Record complete per-step regional counts and firing hashes; first motor,
disinhibitor, guard-suppression, corollary, and stopping times; action crossings;
guard rates; boundary duration; proposal/MSN/commit/motor activity; all gates;
raw and net policy eligibility; coactivity traces; complete weight hashes;
topology counts; inhibitory conductance; device identity; and wall time.

Validity requires exact topology and host-boundary audits, seed identity,
weight immutability, quiet intact baseline, active intact guard, strict intact
motor-first ordering, both actions appearing cleanly at least three times,
bounded trace washout, zero reset current, and complete firing identity for the
coactivity lesion. Downstream lesions must match intact through their first
possible causal divergence on trial one. Later uninterrupted states may differ
after a lesion acts.

`ENGAGEMENT_GO` additionally requires:

- at least `11/12` clean intact trials and at least `11/12` reusable post-
  recovery actions;
- zero intact boundary spikes in the final `1000` baseline steps after every
  trial and continuously active guard bins there;
- `guard_lesion` produces baseline boundary activity in at least `9/12` trials
  or reduces clean rate by at least `0.75`;
- both disinhibitory-path lesions reduce intact corollary spikes by at least
  `90%` and prevent stopping recruitment in at least `9/12` trials;
- `motor_copy_lesion` reduces corollary spikes by at least `90%` while
  preserving a measurable `boundary_vip` response;
- proposal-stop lesion yields at least `2x` intact post-boundary
  proposal+D1+D2 spikes and at least `100` additional spikes across trials;
- commit-stop lesion yields later loser crossings in at least `9/12` trials or
  at least `2x` intact post-decision motor spikes; and
- on clean intact trials, selected-route policy eligibility meets V10's locked
  D1 and D2 criteria: positive in every row, median selected/loser ratio at
  least `4`, loser mean at most `25%` of selected mean, selected larger in at
  least `80%` of rows, and positive mean margin at decision and pre-outcome.

The coactivity lesion's maximum eligibility must be at most `1%` of intact mean
selected eligibility for D1 and D2. Unclean rows are archived and excluded from
winner-labelled aggregates.

## Performance ceiling

On the RTX 3090, compare unchanged V10 and V12 intact at seed `2` after `200`
unscored steps. Time `5000` fixed-drive steps three times per topology. V12
must use no more than `1.50x` V10 median wall time per step and no more than
`1.50x` persistent array bytes. Report every repetition and topology count.

## Stop rule

Do not rerun construction seed `997` or engagement seed `2`, relax quiet or
cleanliness, change baseline arousal, add a direct current, add recurrence or a
slow receptor, shorten the action window, or clear neural state after viewing
an artifact. A construction failure returns to biology review. An engagement
validity failure is `UNDEFINED_*`; a valid causal failure is
`ENGAGEMENT_FAIL`.

Only `ENGAGEMENT_GO` permits a separate policy-learning preregistration. That
later protocol must still require contingent versus reward-count-matched yoked
learning, acquisition and expression lesions, fresh seeds, exact restoration,
same-brain reversal, and a changed future neural action probability.

## Evidence read before filing

- `2026-08-04-neural-vocal-credit-gateB-v12-disinhibitory-boundary-RESEARCH-GATE.md`
- `2026-08-04-neural-vocal-credit-gateB-v11-action-boundary-CONSTRUCTION-NO-GO.md`
- `2026-06-04-cheat2-genuine-bg-disinhibition-RESOLVED.md`
- Kandel, *Principles of Neural Science*, 6e, Figure 58-5 and pp. 1455-1456.
- Pi et al. (2013), [Nature/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4017628/).
- Schneider et al. (2014), [Nature/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4248668/).
- Zhang et al. (2021), [Cell Reports/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8640223/).
