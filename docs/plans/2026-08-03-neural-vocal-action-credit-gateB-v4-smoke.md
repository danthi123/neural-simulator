---
type: plan
status: retired
date: 2026-08-03
---

# Gate B v4: bounded dendritic expectation smoke

## Purpose and boundary

Gate B v3 learned the contingent action, but its normalized action-value cells
were nearly silent by outcome time. The LHb-RMTg-SNc omission chain worked only
when normalization was removed and the value cells became pathologically
active. V4 tests the smallest successor: can the executed action leave a local,
decaying dendritic state that a later shared outcome event reads without
continuous value-cell firing?

This is a **seed-zero physiology smoke**, not a learning result. Formal
calibration seeds `70001/70003`, development seeds
`70009/70019/70039/70051`, and held-out seeds `70061/70067` are fresh and
sealed. `OPEN_PHASES = ()`; no scientific seed may construct a bridge.

## Circuit change

V4 reconstructs the v3 bridge unchanged except for two additions:

1. The existing plastic `motor_X -> action_value_X` route is marked as an
   input to the existing graded dendritic-plateau substrate. Its ordinary
   synaptic transmission and v3 plasticity gate remain present. Its
   transmission window opens only during action selection and closes after
   the committed motor volley is consumed, preventing pre-trial or
   post-commit activity from retagging the trace.
2. The generic outcome population projects identically and non-plastically to
   both action-value populations. It contains no action label. Residual local
   plateau state can make the executed channel more responsive at outcome.

The v3 local FS loop, action-value-to-SNc GABA-B route, reward veto,
omission-gate/LHb/RMTg chain, collateral and reward lesions, and actor/critic
plasticity scope remain. There is no host expected-value array, and the plateau
conductance is simulator state. Python does, however, observe the committed
motor channel and close the expectation route after one additional step. That
winner-dependent timing is a load-bearing scaffold and was incorrectly
described as absent in the original plan.

## Source-to-design map

The local sources were consulted before implementation:

- `$SIM_CATALOG/references/feature-catalog.md`, especially B.06 (striatal PV
  interneurons), B.07 (patch/striosome), C.28-C.30 (dopamine error and
  actor-critic), and O.22 (action-specific striatal value).
- `$SIM_CATALOG/references/textbooks/kandel-pns-6e/full-book.pdf`, chapter 38.
- `$SIM_CATALOG/references/textbooks/basal-ganglia-reviews/` copies of Bolam
  et al. 2000, Tepper and Koos 2017, and Tepper et al. 2018.
- `$SIM_CATALOG/references/textbooks/schultz-dopamine/` copies of Schultz 1998,
  Hollerman and Schultz 1998, and Schultz 2016.

| Design claim | Direct support | What it does not establish |
|---|---|---|
| Separate action-tagged value populations | Samejima et al. found striatal neurons representing action-specific reward values ([Science 2005](https://doi.org/10.1126/science.1115270)); local catalog O.22. | It does not identify those cells as striosomal, prove this two-pool topology, or provide the delayed dendritic mechanism. “Striosomal/action-value” is therefore a functional approximation. |
| Persistent SPN dendritic state | Plotkin, Day and Surmeier showed distal SPN inputs can produce NMDA/Ca-dependent regenerative plateaus that propagate to the soma and whose duration is dopamine-sensitive ([Nature Neuroscience 2011](https://doi.org/10.1038/nn.2848)). Du et al. showed SPN plateaus broaden the integration window, promote spiking, and are controlled by inhibition ([PNAS 2017](https://doi.org/10.1073/pnas.1704893114)). | These studies do not show a learned action-value plateau waiting for an outcome. Biological plateaus are regenerative and often threshold-like; the simulator's smooth logistic transfer is an engineering approximation. |
| Shared outcome read | SPNs receive convergent cortical and thalamic excitation (Du et al. 2017). Bradfield et al. showed the parafascicular-thalamostriatal/cholinergic pathway is required when action-outcome contingencies change ([Neuron 2013](https://doi.org/10.1016/j.neuron.2013.04.039)). | The evidence does not establish a direct, identical `outcome -> action-value SPN` pulse. This is a declared event-timing scaffold and the main anatomical uncertainty in v4. |
| Omission signal through LHb and RMTg | LHb activity is inverse to reward and can inhibit dopamine neurons ([Matsumoto and Hikosaka, Nature 2007](https://doi.org/10.1038/nature05860)). Primate tracing and electrophysiology show excitatory LHb input to GABAergic RMTg and RMTg inhibition of dopamine neurons ([Hong et al., J Neurosci 2011](https://doi.org/10.1523/JNEUROSCI.1384-11.2011)). | V4's omission-gate population and exact weights are simplified. The cited work supports the sign and relay, not this complete comparator wiring. |
| Bounded local inhibition | Local Kandel/catalog material and Tepper/Koos document strong PV-FSI feed-forward inhibition and control of SPN excitability/spike timing. Paired recordings characterize powerful FSI-to-SPN transmission ([Planert et al., J Neurosci 2010](https://pubmed.ncbi.nlm.nih.gov/20203210/)). | “Divisive normalization” and the `1-20 Hz/cell` range are engineering safety criteria, not measured universal biological constants. The smoke must report FS engagement and reject silent or saturated intact value populations. |

The component biology is grounded. The composite action-tagged,
outcome-read dendritic expectation mechanism remains a falsifiable model
hypothesis and must not be described as an established animal circuit.

## Locked smoke protocol

Only seed `0` runs. The exact dataclass configuration is compared before bridge
construction. The smoke uses shared cue/arousal input and the spiking selector;
it never stimulates a desired motor channel.

The operating point was mapped only on seed `0`: plateau center `400`, slope
`0.025`, strength `900`, decay `500 ms`, and symmetric outcome-route weight
`18`. The all-or-none plateau contribution is fixed at zero. Gap probes are
fixed at `60`, `100`, and `160` steps. These numerical values are simulator
operating parameters, not measurements claimed from animal physiology.

Required checks:

- intact outcome-time value firing is `1-20 Hz/cell` and delay firing is at
  most `0.5 Hz/cell`;
- the dendritic state decreases across all three gaps;
- a plateau lesion clears only the expectation-route plateau mask while
  preserving ordinary weights and transmission;
- a shared-outcome-read lesion removes outcome firing;
- a frozen expectation route changes no expectation synapse while the actor
  route remains plastic;
- a fixed action-channel permutation moves the plateau and its paired FS
  normalizer to the opposite value population;
- all changed weights stay inside inherited actor/critic routes.

The formal battery is preregistered but cannot execute: contingent learning,
reward-count-matched shifted yoked reward, frozen expectation route, plateau
lesion, shared-outcome-read lesion, channel permutation, and all v3 collateral,
reward-to-SNc, critic-output, omission-path, and normalization lesions. Opening
calibration requires a separate review and commit after the seed-zero smoke.

## Seed-zero result

The locked NumPy smoke passed. At the middle `100`-step gap, the executed
channel retained plateau state `83.329`, fired `0 Hz/cell` during the delay and
`9.375 Hz/cell` during the shared outcome. The plateau and outcome-read lesions
both reduced outcome firing to zero; permutation moved the state to the other
value/FS microchannel. The plateau declined `90.269 -> 83.329 -> 73.906` across
the three locked gaps.

The exact CuPy run on the RTX 3090 did not pass the bounded-firing criterion:
its action tag and both lesions remained causal, and delay firing remained
zero, but intact outcome firing was `45.833 Hz/cell`. No backend-specific
configuration was introduced. Cross-backend excitability is therefore an open
physiology/implementation issue, and this smoke is not a universal substrate
pass.

## Adversarial audit and retirement

V4 is retired before formal execution. The original action-channel assertions
used a Python-derived expected-channel label and did not require the neural
state itself to be selective. Equal bilateral plateau and outcome vectors could
pass every old check. Explicit selectivity checks and a regression test now
prevent that false pass, but they do not rescue the candidate's evidential
status.

Instrumentation also localized the CPU/GPU firing difference to the always-open
motor-to-FS route: late motor activity after commitment drove the selected FS
pool on NumPy but predominantly the other FS pool on CuPy. The host had closed
only the expectation route, so the selected value pool reached outcome with
substantially different inhibition.

All named formal seeds remain unused and sealed. The successor must derive its
action tag from a brief neural commit event without host winner timing, score
selectivity from neural state rather than label arithmetic, and couple the
generic outcome read to matched feed-forward inhibition. See the
[v4 NO-GO finding](../../research/findings/2026-08-03-neural-vocal-credit-gateB-v4-smoke-NO-GO.md).

## Interpretation limit

A smoke pass establishes wiring, local state, lesion specificity, and a
non-saturated seed-zero operating point. It does not establish learning,
yoked-reward neutrality, an omission dopamine dip, vocal credit, or biological
identity with striosomal circuitry.
