---
type: research-gate
status: complete
date: 2026-08-03
mechanism: neural-vocal-action-credit-v10-corticostriatal-policy
---

# Gate B v10 research gate: make reward change the selector's policy

## Decision

Gate B should stop strengthening the parallel cue-to-actor-to-GPi route. The
next candidate must make the selector's existing `proposal -> str_d1` and
`proposal -> str_d2` synapses reward-plastic, disable the parallel actor output
during causal tests, and show that changing those synapses changes later neural
action selection.

Proceed in two locked steps. First measure whether normal selector activity
leaves action-local eligibility on those routes without changing weights.
Only if that engagement gate passes may a separately preregistered policy
smoke apply the neural reward-evoked dopamine teaching signal. A policy result must include
contingent, reward-count-matched yoked, acquisition-lesion, and learned-route
expression-lesion conditions. Formal seeds remain sealed.

This corrects an earlier diagnostic overstatement: the v9 actor was not
literally disconnected. It inherited a fixed actor-to-GPi projection and could
in principle bias output. The defect is that reward changed a separate direct-
path bypass while the selector's canonical cortical policy stayed fixed. In
the v9 output probe, the learned bypass did not make the rewarded action recur.

## Functional role in the whole brain

A consequence-learning system must alter what the same brain is likely to do
the next time it encounters a similar state. Remembering that an action was
rewarded is insufficient if that memory does not change the competition that
selects behavior.

For grounded conversation, this policy mechanism eventually has to choose
among vocal acts based on perception, conversational state, affect, memory,
and expected social consequences. The small two-action gate is useful only if
it establishes the same causal loop that can later receive those richer state
representations:

```text
shared state and self-generated proposal
    -> action-specific corticostriatal synapses
    -> D1/D2 basal-ganglia competition
    -> thalamic commitment and vocal motor action
    -> sensory/social consequence
    -> neural dopamine prediction error
    -> change at recently eligible policy synapses
    -> changed probability of a future neural action
```

No host process may choose the action, label the eligible channel, calculate
the prediction error, or force the rewarded action to appear during a probe.

V10 is a one-state action-value/habit prerequisite, not closure of this full
role. Shared arousal carries no information that can support choosing action 0
in one context and action 1 in another. A later crossed two-context test is
required before claiming state-dependent policy learning, and V10 cannot
overturn the project's prior negative context-to-action navigation results.

## What the project already established

1. Gate A v2 provides balanced, target-independent, fully neural two-channel
   selection. Across four development seeds it produced 98-100% clean commits,
   and both shared arousal and the D1-to-GPi path were load-bearing.
2. Gate B v1 made a shared cue-to-D1-actor route plastic. Executed motor
   collateral generated local eligibility, and learned actor activity reached
   GPi through a fixed direct path. Contingent reward biased later neural
   choices, but a shifted yoked schedule could create the same arbitrary bias.
3. V3-v9 retained that actor bypass while developing a neural action-
   conditioned expectation and dopamine prediction-error path. V9 established
   learned, dendritically expressed expectation engagement, but its output
   protocol was undefined because rewarded action 0 never occurred in any
   evaluation block.
4. The real selector routes remained fixed throughout: both
   `proposal -> str_d1` and `proposal -> str_d2` are declared `plastic=False` in
   `research/runners/_vocal_action_selector_gate.py`.
5. Older G11 navigation work did implement plastic cortex-to-D1/D2 routes, but
   did not establish a load-bearing learned policy. Early behavior used a
   host-computed direction input plus host argmax/random fallback; later
   anti-cheat tests found that reward learning did not beat its lesion.
   The closest trial-structured point-neuron actor-critic test also failed to
   form a context-dependent place-to-action policy
   (`research/findings/2026-06-19-fsg-watermaze-trial-structured-derisk.md`).
6. The isolated D1/D2 biology probe established update signs only by assigning
   every eligibility trace to `1.0` from Python. It did not establish natural
   action-local eligibility or learned neural choice.

Therefore no existing result demonstrates all of: natural selected-action
eligibility, dopamine-dependent change in the actual selector policy, changed
neural behavior, and causal collapse after lesioning only that learned policy.

## Biological basis

Kandel chapter 38 frames reinforcement as an intrinsic property of basal-
ganglia selection. Recent activity in the selected channel leaves a decaying
eligibility trace; a later broadly distributed dopamine signal selectively
changes that active channel's corticostriatal transmission. The resulting
change biases which option is selected later. The same chapter describes
positive prediction error as strengthening inputs to direct-pathway neurons
and weakening inputs to indirect-pathway neurons, with the opposite tendency
for a negative error.

The local primary-source record and online papers support each part:

- Kandel, *Principles of Neural Science*, 6e, chapter 38, especially pp.
  943-946, local text
  `/home/dant123/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt`.
- Schultz (1998), conjoint input/output eligibility multiplied by a dopamine
  reinforcement signal, local text
  `/home/dant123/Projects/sim-catalog/references/textbooks/schultz-dopamine/Schultz-1998-JNeurophysiol-PredictiveReward.txt`.
- Reynolds, Hyland and Wickens (2001), nigral stimulation potentiated
  corticostriatal synapses in a dopamine-receptor-dependent manner and the
  potentiation covaried with learned behavior:
  [Nature](https://www.nature.com/articles/35092560).
- Shen et al. (2008), dopamine controls glutamatergic plasticity differently in
  D1- and D2-receptor striatal projection neurons:
  [Science/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2833421/).
- Yagishita et al. (2014), local corticostriatal spine plasticity depends on
  dopamine arriving within a short window after glutamatergic and postsynaptic
  activity: [Science/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4225776/).
- Shindou et al. (2019), corticostriatal synapses retain a silent eligibility
  trace that dopamine can convert into potentiation about two seconds later:
  [OIST repository copy](https://oist.repo.nii.ac.jp/record/1004/files/Shindou-2019-A%20silent%20eligibility%20trace%20enable.pdf).

The simulator's signed D1/D2 multiplier is a deliberately simplified systems
hypothesis, not a direct reproduction or validation of the cited cellular
biology. Bidirectional striatal plasticity depends on receptor, cell, timing,
and network state in ways a single sign cannot reproduce. Before this gate, it
recognized only uppercase `str_D2_*` names. Commit `f14150369` changed tagging
to use the D2 MSN neuron identity and added a lowercase vocal-selector
regression, preventing a silent same-sign D1/D2 error.

## Preserved telemetry

Gate A v2's four development-seed artifacts provide evidence without reopening
any seed. Winning-channel proposal populations averaged 11.4-12.0 spikes per
trial versus 8.7-9.3 for losers. Winning D1 populations averaged 36.4-40.3
spikes versus 5.1-8.0 for losers; winning D2 populations averaged 36.0-41.9
versus 5.2-9.0 for losers.

This suggests that pre/post coactivity may naturally concentrate eligibility
on the selected channel, but it does not prove the actual synaptic trace at the
fixed reward delay. Loser D1 activity was nonzero in roughly half the trials.
The first gate must therefore measure route-level eligibility directly and may
not infer it from spikes or erase losing traces in Python.

## Ranked options

### 1. Plastic proposal-to-D1/D2 policy routes

Replace only the two same-channel proposal-to-MSN pathway declarations with
reward-plastic, policy-gated versions. Use the existing coactivity eligibility,
neural SNc dopamine deviation, and cell-identity D1/D2 sign. Disable or remove
all pathways touching the parallel actor population in the causal policy
conditions.

This is selected because reward then changes the input sensitivity of the
same D1/D2 loop that selects and commits the action. It reuses existing sparse
event-driven machinery and adds no host calculation or new population.

### 2. Connect the learned actor back into proposal cortex

An actor-to-proposal relay could make the v9 tag influence the canonical loop,
but it adds a second learned representation and a recurrent relay rather than
placing plasticity at the established corticostriatal policy site. It also
leaves the old bypass available. Do not lead with this.

### 3. Increase cue-to-actor or actor-to-GPi strength

This may make action 0 reappear, but it tunes the parallel bypass against one
undefined artifact. It does not close the policy-learning gap and is retired.

### 4. Host sampling or forced-choice probes

Extending trials, forcing action 0, masking losing eligibility, or using a host
argmax would make the expected-action rows available while concealing the
missing neural policy. These are forbidden.

## Mechanical gates

### V10 engagement

- Use only shared cue/arousal timing and native proposal noise.
- Keep the selector's existing approximately `400` proposal-to-MSN operating
  point. Do not inherit Gate B's `80` plastic-weight ceiling. Set the candidate
  reward-plasticity bounds to `[0, 600]`, assert every jittered initial policy
  weight lies strictly inside those bounds, and record the initial range.
- Keep policy-route plasticity gains enabled so coactivity can form
  eligibility, but set reward learning rate to exactly zero. Assert nonzero
  route gain and byte-identical weights before and after engagement.
- Run a matched zero-eligibility, zero-teaching-signal clip control and require
  byte-identical policy weights. Any movement in engagement or that control is
  an implementation failure, not learning. Rescaling the initial policy
  weights would require a new Gate A physiology result before proceeding.
- Measure D1 and D2 route eligibility at the fixed pre-outcome time.
- Assert D2 route signs are negative and D1 route signs are positive.
- Preregister winner-minus-loser eligibility, winner/loser ratio, and
  cross-channel leakage ceilings separately for D1 and D2. Require each in
  both selected-action directions; if either action has no observations,
  return `UNDEFINED`.
- An action-collateral host mask, winner-index trace edit, forced action,
  argmax, fallback, or winner-dependent timing invalidates the run.

### V10 policy

- File thresholds, seeds, schedules, and precedence before implementation.
- Counterbalance rewarded channels; a fixed action-0-only success is
  insufficient.
- Compare contingent feedback with a reward-count-matched shifted/yoked
  schedule, policy-acquisition lesion, and reward-US-to-SNc/dopamine-path
  lesion while leaving the external reward schedule unchanged.
- Define the policy-acquisition lesion narrowly: set only the policy-route
  plasticity gain to `0`, leave policy transmission at `1`, start from
  byte-identical weights, and leave reward, SNc, expectation learning, and all
  other dynamics unchanged. A global learning-rate lesion is not equivalent.
- Freeze weights for evaluation. Show a baseline-to-evaluation preference
  change only in the contingent condition.
- Restore the learned policy routes exactly to their pretraining weights for
  the primary expression test. Closing their transmission is not an acceptable
  substitute because it can disable the selector itself.
- Changed synapses must be exactly the declared policy and expectation routes.
- Construction must prove that no actor population or actor-to-GPi pathway is
  present, no actor route is eligible or plastic, and the plastic and eligible
  synapse sets equal exactly the policy routes plus any separately declared
  expectation routes. Post-run changed-weight telemetry alone is insufficient
  because a fixed bypass could still influence action selection.
- Report signed route changes separately. Positive reward must potentiate the
  rewarded channel's proposal-to-D1 route and depress its proposal-to-D2
  route; preregister ceilings for both losing-channel changes and require zero
  changes outside declared routes. Restore learned D1 and D2 routes separately
  as diagnostic evaluation arms so their individual behavioral contributions
  are characterized; the combined exact pretraining restore remains the
  primary expression lesion.
- Policy learning must use a registered neural dopamine modulator. Record the
  effective third-factor source on every condition, assert
  `current_reward_signal == 0` throughout, and assert zero host-fallback
  updates. A dopamine-path lesion must prevent acquisition without changing
  the host's sensory reward schedule.
- Retain Gate A physiology: at least 95% clean commits, single-action
  execution, loser suppression, bounded latency, and both channels still
  usable. Score preference over all trials, so silencing the losing action
  cannot masquerade as learning.
- Same-brain contingency reversal is required before Gate B promotion, even if
  the initial smoke passes.
- Expectation reward-suppression or omission output remains a separate result.
  Every required action subset must be observed or that result is
  `UNDEFINED`.
- Until predicted-reward suppression and omission dip are separately
  demonstrated, call the policy third factor a neural reward-evoked dopamine
  teaching signal, not a proven prediction error.

## Performance boundary

V10 should remove the 48-neuron actor bypass from the causal candidate and add
no new neural population. The only persistent new state is plasticity and
eligibility already allocated by the bridge. Report neurons, synapses, GPU
memory, milliseconds per trial, and concurrent throughput. A dense relay or
per-action simulator copy is out of scope.

## Result of the gate

`PROCEED TO V10 ENGAGEMENT PREREGISTRATION`.

The candidate targets the whole-brain functional gap exposed by V9: learned
consequences must change future action selection. A failed engagement gate
retires natural coactivity on the present selector operating point; it does not
authorize host attribution or stronger actor-bypass tuning.
