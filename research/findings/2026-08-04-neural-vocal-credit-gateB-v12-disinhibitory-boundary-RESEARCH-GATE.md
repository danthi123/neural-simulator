---
type: research-gate
status: complete
date: 2026-08-04
mechanism: neural-vocal-action-credit-v12-disinhibitory-boundary
---

# Gate B v12 research gate: open the action boundary only from motor output

## Decision

The next candidate should replace V11's freely excitable corollary population
with a normally inhibited, motor-disinhibited pulse circuit. A symmetric
background pathway must keep the corollary carrier below firing threshold when
no action occurs. Either motor output must simultaneously excite the corollary
carrier and an inhibitory interneuron that suppresses the corollary's local
inhibitory guard. This creates a brief opening only when neural motor output is
present.

Once open, the corollary carrier may recruit V11's distributed local stopping
populations. Their suppression of motor output must remove drive from the
disinhibitory interneuron. The guard must then recover and terminate the
corollary state. The intended causal loop is therefore action-triggered and
self-closing:

```text
background arousal -> boundary_guard_som -| action_corollary

motor_0 or motor_1 -> boundary_vip -| boundary_guard_som
motor_0 or motor_1 -> action_corollary

action_corollary -> local proposal and commit/motor stop populations
local stop populations -| policy proposals, commitments, and motor outputs

motor output ends -> boundary_vip ends -> guard recovers
                  -> action_corollary ends -> stopping state clears
```

The names `boundary_vip` and `boundary_guard_som` state the hypothesized circuit
roles. The simulator does not have characterized VIP and SOM cell models, so
both will initially use inhibitory point-neuron types. That approximation must
remain explicit and cannot support claims about subtype physiology.

## Why this is the next whole-brain blocker

V10 showed that the selector's real policy synapses can carry local coactivity
eligibility, but continued arousal let both actions occur in every trial. V11
added a symmetric neural action boundary, yet the boundary fired during warmup
and a weak no-action catch while both motor populations stayed silent. The
circuit recovered later, so persistence was bounded; the missing function was
conditional recruitment, not merely decay.

A normally closed, motor-opened boundary directly targets that defect. It also
preserves the long-term whole-brain role: a self-action pulse can later drive
auditory prediction, source attribution, and delayed consequence learning.
Reward learning remains closed until the action signal is demonstrably caused
by motor output and can delimit one completed action.

## Project memory check

The local RAG index was searched across findings, plans, the reference catalog,
Kandel, and specialty papers before selecting this candidate.

- V11 already combined motor collateral, corollary discharge, and local
  feed-forward inhibition. Repeating that topology with another recurrent
  weight is excluded by its construction no-go.
- The project has already validated a normally closed disinhibitory cascade:
  D1 inhibits tonic GPi, releasing thalamus. That work also found that GABA
  weights around `2-20` inhibit correctly on this substrate while a weight near
  `300` can pin the membrane near reversal and cause rebound-like firing. This
  is engineering precedent, not evidence that the proposed cortical circuit
  works: `research/findings/2026-06-04-cheat2-genuine-bg-disinhibition-RESOLVED.md`.
- The reference catalog contains striatal disinhibitory interneuron motifs, but
  none directly implements a cortical action-corollary pulse. It therefore
  does not justify relabeling an existing basal-ganglia cascade as V12.

No prior project result tests the selected motor-to-local-disinhibition-to-
corollary loop.

## Biological evidence and limits

The candidate is a synthesis of established cortical motifs. No cited study
demonstrates this exact mammalian action-termination circuit.

- Kandel describes recurrent excitatory collaterals recruiting feedback
  inhibitory neurons and extrinsic excitatory inputs recruiting feed-forward
  inhibition. The general result is that local GABAergic circuitry confines
  otherwise spreading excitation. Local source:
  `/home/dant123/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt`,
  around Figure 58-5 and pp. 1455-1456.
- Pi et al. (2013) directly showed that cortical VIP interneurons inhibit SOM
  interneurons and, less commonly, PV interneurons, producing local
  disinhibition of pyramidal cells. This supports the inhibitory-on-inhibitory
  opening motif, not its proposed motor timing or action-boundary role:
  [Nature/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4017628/).
- Schneider et al. (2014) showed that motor-cortical input reaches auditory
  cortex before and during movement and recruits local PV-mediated suppression.
  This supports a motor collateral engaging recipient-local inhibition:
  [Nature/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4248668/).
- Zhang et al. (2021) recorded movement-preceding activity in mouse secondary
  motor cortex and found that principal-cell recurrence interacts with PV and
  SOM interneurons to shape preparatory ramps. Their model predicts that
  suppressing SOM can release PV strongly enough to silence the principal
  population. This supports local inhibitory control of recurrent motor-cortex
  activity, but it is not a demonstrated VIP-triggered action pulse:
  [Cell Reports/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8640223/).
- Primary-motor-cortex tracing confirms that VIP interneurons receive diverse
  long-range inputs and summarizes the canonical VIP-to-SST-to-pyramidal
  disinhibitory motif. The tracing is anatomical and does not establish the
  proposed temporal computation:
  [Frontiers/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10237295/).

The strongest unsupported step is assigning motor output itself to open this
specific local disinhibitory gate and relying on loss of that output to close
it. V12 must test that systems hypothesis causally rather than describe it as
known biology.

## Selected candidate

Keep the V10 selector and V11's separate proposal and commit/motor stopping
populations. Replace V11's direct baseline-excitable corollary construction
with three central populations:

1. `action_corollary`: excitatory carrier receiving identical collaterals from
   `motor_0` and `motor_1`.
2. `boundary_guard_som`: inhibitory guard receiving symmetric background drive
   from the existing `practice_arousal` population and inhibiting
   `action_corollary`.
3. `boundary_vip`: inhibitory disinhibitor receiving identical collaterals from
   both motor populations and inhibiting `boundary_guard_som`.

The guard must be driven synaptically. No direct current may target the
corollary, guard, disinhibitor, or stopping populations. The existing
`practice_arousal` current remains a declared experimental scaffold, but the
new boundary may read only its spikes, not its host current value. The same
arousal schedule must run in all matched arms, including warmup and no-action
catch.

Start without recurrent corollary excitation. V11 showed that recurrence is
not the first defect and can amplify startup activity. Add a slow recurrent
route only in a separately preregistered second construction rung if the
motor-triggered corollary volley is causal and quiet at baseline but too short
to recruit the stopping branches. This is a mechanism-ordered ladder, not a
weight sweep.

## Construction gate before formal testing

Commit a preregistration before implementing the runner. It must lock population
sizes, neuron types, all synaptic densities and weights, baseline arousal,
fixed phase lengths, construction seeds, backend order, and stop rules. It must
include a physiological inhibitory-weight audit based on the project's prior
rebound finding.

The first construction stage must run fresh NumPy and CuPy brains and require:

1. zero corollary, disinhibitor, and stopping-population spikes during a fixed
   warmup and weak no-action catch;
2. nonzero guard activity during both periods, caused by the declared neural
   background route;
3. zero motor spikes in the catch;
4. motor output strictly preceding disinhibitor and corollary output;
5. a single action recruiting the corollary and both local stopping
   territories, followed by autonomous return to baseline;
6. a second action on the same uninterrupted brain after recovery;
7. byte-identical complete weights, zero selector-reset current, exact channel
   symmetry, and one pathway declaration per ordered region pair; and
8. one shared construction point passing on both backends before formal seed
   assignment opens.

Any baseline boundary spike retires that construction point. Do not lengthen
warmup after observing activity, delete startup telemetry, lower the definition
of quiet, or interpret guard failure as evidence for increasing corollary
recurrence.

## Required causal arms

If construction passes, the later engagement smoke must preserve V10's fixed
action windows and include matched structural lesions:

- guard-to-corollary lesion: should restore inappropriate baseline or catch
  activity;
- motor-to-disinhibitor lesion: should prevent or shorten boundary opening
  without changing the action prefix;
- disinhibitor-to-guard lesion: should prevent the motor-triggered release;
- motor-to-corollary lesion: should remove the corollary volley while leaving
  the disinhibitory event measurable;
- proposal-stop and commit/motor-stop branch lesions: should separately restore
  late policy coactivity and late motor activity; and
- coactivity lesion: must still collapse policy eligibility exactly.

Matched intact and lesion arms must be byte-identical through the first motor
spike wherever the lesion is downstream of motor output. The runner must report
first spikes and per-step activity for motor, disinhibitor, guard, corollary,
all stop populations, proposals, MSNs, commitments, and motors. No lesion may
be implemented by host current, array clearing, threshold observation, or
route-specific winner logic.

## Stop rule and next decision

V11 construction seed `991` and formal seed `1` are consumed or sealed exactly
as recorded and must not be reused. V12 gets fresh construction seeds only
after its preregistration commits. A construction no-go retires the selected
topology before eligibility or reward learning. A construction go opens one
reserved engagement seed, not formal reward learning.

Only a cross-backend, action-contingent, self-closing boundary with causal
lesion support can reopen V10's policy-eligibility question. Reward,
dopamine-output tuning, policy updates, and same-brain reversal remain closed.
