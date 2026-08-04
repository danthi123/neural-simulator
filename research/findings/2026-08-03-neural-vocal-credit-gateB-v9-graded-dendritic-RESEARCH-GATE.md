---
type: research-gate
status: complete
date: 2026-08-03
mechanism: neural-vocal-action-credit-v9-graded-dendritic-readout
---

# Gate B v9 research gate: let the learned route enter through a dendrite

## Decision

The highest-value Gate B successor is to route the existing learned action-
context synapses through the bridge's graded dendritic plateau before the
MSN-D1 soma. This is a different mechanism from v6-v8: it changes how the
postsynaptic neuron integrates a learned input, not the input's weight,
population size, or addition of a fixed drive.

File a bounded reserved-seed preregistration before code. Keep formal seeds
sealed. Do not interpolate v8's fixed-input ladder, inject expectation current,
read dendritic state in Python to calculate dopamine, or reopen the retired v4
protocol.

## What the project already established

The project record gives a consistent diagnosis:

1. V5's neural commit-plus-arousal circuit creates an action-specific cortical
   trace without a Python winner latch or winner-timed transmission window.
   Its separate trace-to-expectation route learns locally and changes no
   undeclared synapse.
2. V6 increased sparse-route weight and v7 increased the learned population to
   200 cells. Both left the MSN-D1 expectation soma silent before reward.
3. V8 added a distinct fixed convergent input. Weight `2` stayed subthreshold
   but remained silent after learning; weights `4+` predicted without learning.
   The tested point-neuron circuit had no valid operating point.
4. The June critic work independently measured the same point-neuron boundary:
   ordinary distributed input left the MSN soma sub-rheobase, while a steep
   all-or-none plateau over-drove it. A smooth on-bridge dendritic plateau
   carried a learned graded value that the point-neuron readout could not.
5. V4 already applied a graded plateau to vocal action value, but its evidence
   is retired because Python timed the motor route after observing the winner,
   its original checks could accept bilateral activity, and late motor-to-FS
   traffic produced different CPU/GPU inhibition. V5 subsequently removed
   those three confounds with a fixed neural action epoch, neural selectivity
   scoring, and outcome-linked matched excitation and inhibition.

This makes a v5/v7 neural trace plus a dendritic learned-route readout a valid
successor. It is not a rerun of v4: no host observes or times the chosen
channel, and engagement is measured before any outcome pulse.

## Biological basis

Striatal spiny projection neurons rest in a hyperpolarized down-state and
normally require coordinated excitatory input to enter an up-state. Distal,
spatially clustered glutamatergic inputs can instead trigger an NMDA-dependent
dendritic plateau that propagates to the soma. The plateau broadens the window
for later excitation and promotes spiking. Dendritically targeted inhibition
can tune its duration and spike output through NMDA magnesium-block dynamics.

Primary sources:

- Plotkin, Day and Surmeier (2011), distal SPN input produces regenerative
  plateaus and somatic up-states, with dopamine-sensitive duration:
  [PNAS/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3235762/).
- Du et al. (2017), SPN plateaus broaden spatiotemporal integration and are
  controlled by branch- and cell-type-specific inhibition:
  [PNAS/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5594658/).

The simulator's gentle logistic transfer is a reduced model, not a claim that
animal plateaus are smoothly analog at every scale. A positive v9 smoke would
validate the existing reduced integration mechanism for this role; it would
not remove the longer-term need for branch-specific inhibitory regulation.

## Ranked options

### 1. Learned-route graded dendritic integration

Mark only the plastic trace-to-expectation synapses as dendritic plateau input.
The route's changing synaptic weights then alter its weighted dendritic drive.
The plateau current acts inside the bridge and must cause expectation neurons
to spike before reward. Downstream dopamine is not calculated from a host read
of plateau conductance.

This ranks first because the production kernel already exists, the project has
validated it on learned value, and it directly addresses the v6-v8 cellular
boundary with no new simulator feature.

### 2. Branch-specific inhibitory control

Add an SST/NPY-like dendritic inhibitory population to regulate plateau
duration and prevent saturation. This is better biological control if option 1
over-drives the soma, but it introduces another learned or timed pathway before
the basic dendritic-readout hypothesis is tested. Reserve it for a separately
preregistered successor or companion test.

### 3. Intrinsic-excitability or cholinergic regulation

Slow intrinsic plasticity can stabilize excitability, and coincident dopamine,
SPN depolarization, and cholinergic pauses are relevant to corticostriatal
plasticity. Neither directly solves the present readout defect: v5-v8 already
learned the route, while the soma failed to express the learned state. Do not
lead with these mechanisms.

## Mechanical guards

- Preserve v5's fixed neural action epoch and reject any Python channel latch.
- Preserve v7's 200-cell trace, route density, local reward plasticity, reset,
  and all plastic-leakage checks.
- Lesion only the learned route's dendritic mask while leaving its synapses,
  weights, ordinary transmission, and the upstream action-trace plateau intact.
- Require zero expectation spikes before learning.
- Require the learning lesion and dendritic-route lesion each to remove at
  least 80% of intact pre-outcome expectation activity.
- Keep expectation output to dopamine closed during engagement, so a pass
  cannot alter its own training signal. Open and test GABA-B/GIRK only under a
  filed amendment after engagement passes.
- Report the expectation population's spikes as the capability signal. The
  analog plateau conductance is telemetry and causal state, not a host output.

## Result of the gate

`PROCEED TO PREREGISTRATION`. The experiment is small, uses an existing guarded
GPU-capable substrate, directly tests a repeatedly localized point-neuron
boundary, and preserves the project's no-host-shortcut requirement. A negative
retires the learned-route graded-plateau candidate; it does not justify tuning
v6-v8 again.
