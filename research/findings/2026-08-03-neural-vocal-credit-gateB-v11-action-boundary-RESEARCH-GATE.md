---
type: research-gate
status: complete
date: 2026-08-03
mechanism: neural-vocal-action-credit-v11-action-boundary
---

# Gate B v11 research gate: make action completion a neural state

## Decision

The next candidate should add a central motor-command corollary and distributed
feed-forward inhibitory action-active circuit to the V10 selector. A command
from either motor channel must excite the same corollary carrier. That
excitatory carrier must fan out symmetrically to separate local inhibitory
populations in proposal cortex and downstream commit/motor cortex. No single
inhibitory population may project across those anatomical territories.

A weak slow recurrent component may keep the corollary carrier active after
the initiating motor volley, but intrinsic adaptation and withdrawal of drive
must let it terminate without a host reset. A circuit that suppresses activity
for exactly the scoring window but cannot autonomously recover is a failure.

This is an engagement candidate, not permission to train the policy. First show
that the circuit converts one neural commitment into one completed action under
continued symmetric arousal, preserves action-local eligibility on the actual
proposal-to-D1/D2 policy synapses, and is causally necessary. Reward, dopamine
teaching signals, and weight updates remain closed.

## Functional role in the whole brain

A continuous brain needs to know that an action it initiated is now in
progress. Otherwise the same drive can repeatedly launch that action or let a
competitor begin before the first action has acquired a consequence. This is
not just an experimental timing problem. Grounded speech eventually needs a
neural boundary around each vocal act so that the brain can:

1. suppress incompatible motor proposals while speaking;
2. preserve which policy was responsible for the utterance;
3. predict and attenuate its self-generated sound;
4. tag the sound as self-generated for source monitoring; and
5. associate the ensuing social consequence with one completed action.

The same motor-corollary population should therefore be reusable later by the
project's existing authorship/source-monitor and auditory-prediction circuits.
V11 tests only its immediate selector and credit role. It must not be called a
general action model or a speech controller.

## What V10 exposed

Gate A v2 appeared to produce clean single actions because its Python runner
stopped the action phase immediately after the first motor threshold crossing.
The only whole-selector reset was then driven by host current. V10 deliberately
continued the same symmetric arousal for a fixed 600-step window. Both channels
crossed first across the 12 trials (`7/5`), but the other channel crossed later
in every trial. Zero rows met the preregistered clean-action definition.

V10 nevertheless showed that local proposal/MSN coactivity generated nonzero
eligibility and that disabling only coactivity reduced all D1/D2 policy
eligibility to exactly zero. At the first crossing, the initially selected
route had larger decision-time eligibility in `11/12` rows for both D1 and D2.
The untested question is whether a neural action boundary can preserve that
initially local tag until a delayed consequence.

## Biological evidence and limits

The candidate combines established motifs, but no cited experiment directly
demonstrates this exact mammalian vocal-selector circuit.

- Kandel chapter 38 and the local catalog's A.03 entry describe the hyperdirect
  cortex-to-STN-to-GPi/SNr route as a rapid broad brake that raises inhibition
  over competing motor channels. This supports global braking, not a learned or
  self-triggered action-completion latch. Local source:
  `/home/dant123/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt`,
  especially pp. 941-946.
- Schmidt et al. (2013) recorded fast stop-cue responses in STN and successful-
  stop responses in SNr, supporting a timing race between broad STN excitation
  and movement-related striatal inhibition. Their task concerns externally
  cued cancellation before a point of no return, not self-termination after a
  completed action: [Nature Neuroscience/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3733500/).
- Nelson et al. (2013) showed that excitatory motor-cortical projections can
  suppress a recipient cortical population by recruiting local PV-positive
  feed-forward inhibition. Some projecting M2 neurons also collateralize to
  brainstem motor structures. This supplies the central-command collateral and
  recipient-local inhibitory motifs:
  [Journal of Neuroscience/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3761045/).
- Schneider et al. (2014) found motor-related auditory-cortical suppression
  before and during movement, with local PV interneurons causally involved.
  This supports a corollary signal that lasts through an action, but its target
  is sensory cortex rather than the action selector:
  [Nature/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4248668/).
- Jin and Costa (2010) found that striatal and nigral activity develops
  action-sequence start/stop boundary signals during learning. The result shows
  that basal-ganglia activity can represent action boundaries; it does not show
  that those recorded neurons implement the inhibitory reset proposed here:
  [Nature/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3477867/).
- The project's six-seed authorship probe already carries self-production over
  a spiking `production -> corollary discharge` pathway, but that circuit is
  fixed, isolated, and host-presented. It is reusable engineering evidence, not
  proof of natural action boundaries:
  `research/findings/2026-08-01-agency-authorship-tag-corollary-discharge-source-monitor-6seed-GO.md`.

## Selected candidate

Interpret each existing `motor_X` population as a corticofugal vocal
motor-command population upstream of the body, not as a spinal motoneuron pool.
Add one shared excitatory corollary carrier plus separate local inhibitory
populations to the unchanged V10 selector:

```text
motor_0 or motor_1
    -> action_corollary (excitatory motor copy; no external current)
       -> proposal_stop_fs_0 -> proposal_0
       -> proposal_stop_fs_1 -> proposal_1
       -> commit_stop_fs_0   -> commit_0 and motor_0
       -> commit_stop_fs_1   -> commit_1 and motor_1

action_corollary -> weak slow self-excitation
action_corollary intrinsic adaptation + loss of motor drive -> recovery
```

Each inhibitory population is local to one cortical territory and channel; the
shared excitatory corollary projection, not a long-range inhibitory axon,
coordinates them. Proposal inhibition stops new corticostriatal coactivity.
Commit/motor inhibition terminates downstream recurrent output. D1/D2 activity
must decay after proposal input stops; adding an anatomically ungrounded
cross-region inhibitory shortcut to force this result is prohibited.

Neither corollary nor local inhibitory population may receive winner-labelled
current, read a host threshold, change a route-specific gate, or connect
asymmetrically to the channels. Policy plasticity and coactivity-trace input
gates remain on their fixed phase schedule; neural inhibition, not Python, must
stop new policy coactivity after commitment. The existing host-driven
`selector_reset` current must remain zero throughout V11. Trial separation uses
a fixed neutral interval with no action drive, during which the neural state
must decay on its own.

Weak slow excitation is included because a one-step feed-forward volley would
disappear when inhibition silences its own motor source. It must be below the
project's already demonstrated stable-attractor regime: the goal is a bounded
action-active state, not working-memory storage. Its duration must arise from
slow synaptic dynamics plus the cell's intrinsic adaptation, and it must end
after action drive is removed. The exact topology and any construction-only
operating-point ladder must be committed in a preregistration before the runner
exists.

Use the bridge's ordinary region-scoped NMDA path: mark only
`action_corollary` as NMDA-enabled, enable global NMDA, and route an explicit
AMPA `action_corollary -> action_corollary` pathway through a transmission gate
for the recurrence lesion. The selector builder must gain a default-preserving
option to remove the currently dormant NMDA tags from `commit_0/1` in V11. Do
not use the less-tested `nmda_slow` transpose path, direct recurrent NMDA on an
inhibitory population, GABA-B from an excitatory source, or the host-side
`couple_gate_to_pool()` helper.

The runner must assert one declaration per `(from_region, to_region)` pair,
excitatory source identity for every NMDA route, inhibitory source identity for
every GABA route, and exact symmetric synapse coordinate sets. Slow-conductance
checkpoint state is currently absent from bridge save/load; that engineering
debt must be repaired and tested before V11 is treated as production-capable
continuous brain state.

This is a simplified systems hypothesis. A generic recurrent cortical pool is
not equivalent to a characterized biological action-boundary cell type, and
the simulator's point neurons do not reproduce the full dendritic, receptor,
or axonal-delay biology in the cited work.

## Ranked alternatives

### 1. Motor-command corollary to distributed local inhibition

Selected because it closes the exact continuous-action defect, can suppress
both cortical policy activity and downstream output, respects the local reach
of cortical inhibitory interneurons, and creates a reusable self-action signal
for sensory prediction and source monitoring. It is a synthesis of supported
motifs, not a directly observed integrated circuit.

### 2. Motor or frontal cortex to shared STN hyperdirect brake

This is the strongest direct basal-ganglia stopping motif. In the current
selector it would raise GPi output and suppress thalamus, but continued external
arousal would still drive proposal and striatal coactivity. It does not by
itself preserve action-local policy eligibility, so it is a useful future
component or control rather than the first complete candidate.

### 3. Renshaw-like motor recurrent inhibition

Motor axon collateral to inhibitory interneurons is a canonical gain-control
motif, but Renshaw inhibition primarily regulates the same and synergist motor
neurons. It would not suppress the competing cortical/BG policy or supply a
stable whole-action state. Do not lead with it.

### 4. Host stop, winner-triggered gate closure, or shorter scoring window

These would reproduce Gate A's hidden shortcut. They can make V10 rows look
clean without adding the function the continuously operating brain lacks and
are prohibited.

## Required falsification

The preregistered smoke must keep the fixed 600-step symmetric action window
and compare separately constructed, seed-matched conditions. At minimum it
must include intact, motor-to-corollary lesion,
corollary-to-proposal-inhibition lesion, corollary-to-commit-inhibition lesion,
and slow-recurrence lesion arms. It must establish all of the following before
eligibility selectivity is interpreted:

1. both actions still initiate without channel-specific input or fallback;
2. the intact circuit yields one clean action for nearly every trial while a
   boundary lesion restores later competing actions;
3. activity and RNG history are identical through the first motor-command
   spike across matched arms, so the shared circuit does not choose the winner;
4. corollary and local inhibitory activity begins only after neural motor
   output and persists long enough to suppress late proposal/MSN and motor
   activity;
5. the proposal-inhibition lesion restores late proposal/MSN coactivity, while
   the commit-inhibition lesion restores downstream motor persistence or
   re-entry; a branch with no causal contribution is removed rather than kept;
6. removing slow recurrence preserves the first corollary volley but shortens
   the action-active state, establishing whether persistence is load-bearing;
7. neutral and subthreshold-action periods do not engage the boundary;
8. V10's washout, trace hygiene, weight immutability, coactivity lesion, and
   exact structural checks still pass; and
9. clean intact trials retain selected-over-other policy eligibility at the
   neural decision and fixed delayed-consequence snapshots; and
10. after fixed drive withdrawal, the same uninterrupted brain returns to a
    quiet boundary state and can initiate either action in a later epoch with
    no host reset current or state-array clearing.

The artifact must report first motor spike, first threshold crossing, boundary
onset and duration, per-step population telemetry, complete firing/weight
hashes, all pathway gates, neuron/synapse counts, GPU identity, elapsed time,
and matched-lesion timing. It must separately report action-state decay and
same-brain recovery latency. Performance overhead must be bounded in the
preregistration and measured against the V10 topology on the same backend.

## Stop rule and next decision

Use a new reserved smoke seed; do not rerun V10 seed `0`. Do not tune after
viewing that reserved result, relax the clean-action definition, or open policy
learning. If the construction-only ladder cannot produce a quiet pre-action
state, one action before inhibition, a bounded action-active state, and
autonomous recovery, retire the candidate before consuming the reserved seed.
If the locked smoke is
undefined or fails, record it and return to a new evidence gate rather than
substituting a host boundary.

Only a locked engagement GO permits a separate policy-learning preregistration.
That later test must still require contingent versus reward-count-matched yoked
learning, acquisition and expression lesions, exact restoration, fresh seeds,
and changed future neural action probability. A V11 engagement GO would close
one prerequisite, not establish grounded speech or general action learning.
