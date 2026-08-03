---
type: finding
status: negative
date: 2026-08-03
mechanism: neural-vocal-action-credit-v5-learning
runner: research/runners/_vocal_action_credit_gate_v5_learning.py
artifacts:
  - research/findings/raw/vocal_action_credit_gate_v5_learning/dynamics_smoke_seed0_numpy.json
  - research/findings/raw/vocal_action_credit_gate_v5_learning/dynamics_smoke_seed0_cupy.json
---

# Local reward expectation learns, but its prediction-error output is too weak

<!--derived-->
**Verdict: NO-GO at the reserved-seed dynamics smoke, before formal
calibration.** The corrected v5 circuit learns an action-local reward
expectation on both NumPy and CuPy, and the expectation-learning lesion blocks
that change. The current expectation output does not suppress a repeated reward
response by the preregistered 20%, and expected omission never recruits the
LHb-like or RMTg-like populations. All assigned scientific seeds remain sealed.

## Question

Can the fixed neural action trace established by the v5 construction smoke
train a separate reward-expectation population, and can that learned population
produce the positive and negative dopamine prediction errors needed to prevent
action-independent reward from reinforcing an arbitrary action?

This reserved-seed test is intentionally smaller than the formal contingent
versus yoked experiment. It asks whether the implementation has active,
bounded learning and a causal prediction-error signal before any calibration,
development, or held-out seed is consumed.

Artifacts:
`research/findings/raw/vocal_action_credit_gate_v5_learning/dynamics_smoke_seed0_numpy.json`
and
`research/findings/raw/vocal_action_credit_gate_v5_learning/dynamics_smoke_seed0_cupy.json`.

## Circuit and protocol

The action trace is an excitatory spiking population loaded by fixed
commit-plus-arousal coincidence routes. A separate inhibitory striatal-like
expectation population receives a plastic trace route and symmetric outcome
input. Its output uses GABA-A inhibition onto the shared SNc-like population
and GABA-B inhibition of the tonic omission gate. Only cue-to-actor and
trace-to-expectation synapses may change.

Reserved seed `0` ran for 12 fixed contingent trials in three separately
initialized conditions: intact, expectation-learning lesion, and
expectation-output lesion. The host scheduled fixed cue, action, delay, and
outcome windows and delivered reward only when the neurally executed action was
action `0`. It did not store the winner across the delay, choose a neural route,
set eligibility, calculate prediction error, or write a weight. Learning was
then frozen for one omitted-reward probe.

## Cross-backend result

| measure | NumPy mini-PC CPU | CuPy RTX 3090 |
|---|---:|---:|
| intact clean action epochs | `11 / 12` | `10 / 12` |
| intact rewarded trials | `8` | `7` |
| rewarded expectation route, before to after | `0.1000 -> 1.5731` | `0.1000 -> 1.3047` |
| unrewarded expectation route, before to after | `0.1000 -> 0.1356` | `0.1000 -> 0.0479` | <!--derived-->
| expectation-learning-lesion route, before to after | `0.1000 -> 0.1000` | `0.1000 -> 0.1000` |
| early reward dopamine burst | `0.06083` | `0.07129` |
| late reward dopamine burst | `0.05745` | `0.06497` |
| early-to-late suppression | **`5.56%`** | **`8.86%`** |
| preregistered minimum | `20%` | `20%` |
| omission LHb-like / RMTg-like spikes | `0 / 0` | `0 / 0` |
| changed synapses outside declared routes | `0` | `0` |

The local learning result is real within this smoke. The rewarded expectation
route grows by more than twelve times its initial mean on both backends and
separates from the other route. Closing only its plasticity gate leaves both
route means exactly at `0.1`. Every observed weight change stays inside the two
declared route families.

The output result is insufficient. Repeated reward reduces the late dopamine
burst, and disabling expectation output raises the late response, but the
intact reduction is less than half of the fixed minimum on both backends. The
lesion conditions also follow different learned action trajectories, so their
cross-condition restoration number is diagnostic rather than an isolated
same-brain effect. It cannot rescue the failed intact criterion.

Omission lowers measured dopamine on both backends, but no LHb-like or
RMTg-like neuron fires. That dip is therefore not attributable to the declared
negative prediction-error path. The NumPy suite meets the 90% action-validity
precondition in every arm; CuPy does not because its output-lesion arm has only
`9/12` clean epochs. This is an additional implementation failure, not a reason
to change the threshold.

## Development observations

Reserved-seed operating-point work before the clean run found the boundary
rather than a hidden passing setting. These observations are implementation
diagnostics, not formal evidence:

- The inherited postsynaptic GABA-B expectation-to-SNc path was nearly inert.
  Raising its propagation strength produced only small suppression and became
  nonmonotonic at larger values, including rebound-like increases in firing.
- Fast GABA-A output produced a stronger and monotonic direct effect. Weights
  `80` and `160` remained learnable but stayed below the 20% criterion; weight
  `320` disrupted action and expectation separation.
- The original trace population used an inhibitory D1 neuron preset, making
  trace-to-expectation excitation impossible. Replacing the trace with an
  excitatory cortical-like spiking population fixed that construction defect.
- The cue-period winner can differ from the action executed under arousal.
  Reward is now contingent on the executed action epoch only, preventing a host
  scoring error from crediting the wrong neural trace.

Do not repeat a larger direct-output weight sweep. The stable region has been
measured, and stronger direct inhibition degrades the circuit before it reaches
the required prediction-error behavior.

## Biological grounding

The local corpus search surfaced Schultz's primary reward-prediction record.
Predicted reward produces little or no dopamine activation, while omission
depresses activity at the expected time even without an immediately preceding
stimulus; this implies an internal expectation process rather than a simple
sensory response
(`Schultz-1998-JNeurophysiol-PredictiveReward.txt:375-445`).

The local basal-ganglia review supports the switch from postsynaptic GABA-B to
GABA-A: evoked inhibition of nigral dopamine neurons in vivo is predominantly
or exclusively GABA-A, while the clearest GABA-B role is presynaptic modulation
(`TepperAbercrombieBolam-2007-GABAandTheBasalGanglia-PBR160.txt:17823-17849`
and `:18190-18235`). The same review warns that direct striatal inhibition and
indirect disinhibition through more GABA-sensitive nigral output neurons
coexist. That local balance is absent from the present one-projection output.

External primary-source checking also confirms that the negative pathway is
not just an LHb-to-dopamine inhibitory edge. In primates, RMTg neurons receive
excitatory LHb input, encode negative reward-prediction errors, project near
dopamine somata, and inhibit dopamine neurons ([Hong et al.,
2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3315151/)). Lesion evidence in
rats shows that RMTg damage reduces LHb-induced dopamine inhibition
([Brown et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5214632/)).

## Provenance

The CuPy artifact ran from clean detached revision `11868b252` on the local RTX
3090. The NumPy artifact ran on `pool40` from immutable Git archive revision
`4504143cd`; its source manifest verified both before and after execution. The
revision difference is only the archive-runtime seed-validator fix and does not
change the runner. Both sidecars record the requested and resolved backend, a
fresh corpus check, and `git_dirty=false`.

## Decision

Do not open calibration seeds `76405` and `71409`, development seeds `79696`,
`72650`, `77948`, and `75688`, or held-out seeds `71272` and `79796`.

Keep the fixed action trace, separate expectation populations, local
trace-to-expectation plasticity, executed-action scoring correction, and strict
plasticity ownership. Retire the current direct expectation-output operating
point. The successor must model the missing local inhibitory/disinhibitory
balance around dopamine output and establish an active LHb-to-RMTg negative
path before another dynamics smoke. It must retain the same 20% reward
suppression, omission, lesion, action-validity, and zero-leakage checks. Formal
contingent/yoked testing remains downstream of that smoke, not a substitute for
it.
