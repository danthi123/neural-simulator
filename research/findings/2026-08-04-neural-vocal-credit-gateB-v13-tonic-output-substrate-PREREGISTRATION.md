---
type: preregistration
status: active
date: 2026-08-04
mechanism: neural-vocal-action-credit-v13-tonic-output-substrate
runner: research/runners/_vocal_action_credit_gate_v13_tonic_output.py
---

# Gate B v13 Stage 0: autonomous tonic-output substrate

**Filed before the substrate field, runner, or tests exist.** This stage asks
whether the shared Izhikevich substrate can represent an autonomously firing
GPi/SNr-like output population without writing tonic current through the
runtime external-stimulus array. It does not test action selection or learning.

The machine-readable copy of this protocol is
`research/specs/v13_tonic_output_substrate.json`. The Markdown document remains
the human-readable authority; the runner must reject disagreement between its
locked constants and that JSON file.

## Why this stage comes first

The inherited Gate A selector applies `1000 pA` directly to both GPi pools on
every trial and calls the resulting cells tonic output neurons. At zero external
current, unchanged Gate A GPi, GPe, and STN populations produced zero spikes for
`1000 ms` on NumPy and CuPy. The earlier `HH_GPI_OUTPUT` channel retune is not
used by this mixed-region Izhikevich selector.

The implementation audit also found that Gate A leaves global parameter
heterogeneity enabled. Heterogeneity is applied after per-region cell-type
presets and its default distributions are centered on the global cortical
configuration. Four GPi parameters can therefore be overwritten by
cortical-centered samples. Stage 0 disables global heterogeneity and uses the
existing per-region mask with distributions explicitly centered on the
`IZH2007_GPI_OUTPUT` preset. This avoids silently calibrating a mislabeled
cortical/GPi hybrid. It does not repair general cell-type-relative
heterogeneity; that remains separate substrate debt before the full selector is
integrated.

## Locked substrate change

Add exactly one field to `BrainRegion`:

```python
intrinsic_current_pA: float = 0.0
```

Semantics:

- it is a construction-time, region-scoped effective intrinsic current in pA;
- it represents unresolved cell-autonomous depolarizing conductances in a
  reduced Izhikevich model, not sensory input and not an implemented ion
  channel;
- any finite signed value is valid, but Stage 0 uses only nonnegative values;
- nonzero use is initially valid only when the bridge neuron model is
  `IZHIKEVICH`; HH or AdEx use must raise `ValueError` rather than silently
  imply support;
- the bridge creates `cp_intrinsic_current_pA`, a length-`n` float32 vector,
  only when at least one region requests a nonzero value;
- when all regions use the default, the attribute remains `None` and no device
  array is allocated;
- the vector is populated once during initialization and has no public runtime
  setter; and
- every active runner must hash it before and after execution to detect
  mutation.

Apply this current exactly once immediately before neuron-model dynamics. It is
not part of `cp_external_input_current`, must survive calls that clear external
stimuli, and must not be fed through input normalization or input-mean
adaptation. For the two read-only step megakernels, an active bridge may pass a
temporary `external + intrinsic` vector through the existing current argument;
the kernels need no signature change. A default-off bridge must pass the
original external-current object and perform no vector addition.

The optional vector must be cleared with other bridge arrays, included in
memory accounting, saved in new checkpoints, restored when present, and remain
`None` when loading an old checkpoint that lacks it. Recording support may
capture it as static initial state, but playback must never interpret it as an
external stimulus.

This implementation is deliberately narrower than explicit NaP, NALCN-like,
Ih, calcium, and SK conductances. Its scaffold-ledger entry is part of this
preregistration and must remain active until an ion-current replacement passes
its own comparison gate.

## Locked isolated population

Every physiology arm uses:

- one `gpi_snr` region with `40` inhibitory neurons;
- `IZH2007_GPI_OUTPUT` parameters;
- `dt = 1 ms`;
- no functional internal or external synapses in the tonic-only arm;
- OU noise, conductance noise, homeostasis, STDP, Hebbian learning, reward
  modulation, structural plasticity, short-term plasticity, NMDA, GABA-B, and
  neuromodulators disabled; and
- `cp_external_input_current` exactly zero for every GPi/SNr neuron on every
  scored step.

Global parameter heterogeneity is disabled. Only `gpi_snr` sets
`enable_heterogeneity=True`, using these fixed GPi-centered distributions:

| parameter | distribution |
|---|---|
| `izh_a_val` | log-normal, `mean_log = log(0.05)`, `sigma_log = 0.15` |
| `izh_b_val` | Gaussian, mean `2.0`, standard deviation `0.3` |
| `izh_d_val` | Gaussian, mean `25.0`, standard deviation `3.75` |
| `izh_C_val` | Gaussian, mean `60.0`, standard deviation `9.0` |

The runner must assert after initialization that `k`, `vr`, `vt`, and `vpeak`
remain exactly the GPi preset values and that the four heterogeneous parameter
means are closer to the GPi centers above than to the global cortical defaults.

## Seed partitions and backend order

- Audit-only seed `314159` informed the ladder and is never a verdict seed.
- Compatibility seed `271828` has pre-change hashes and is never a physiology
  verdict seed.
- Calibration seed `1013` runs the complete ladder, NumPy first and CuPy
  second.
- Replication seed `1019` runs only the selected point, NumPy and CuPy.
- Held-out seed `1021` runs only after the selected source and runner are
  committed and sealed, CuPy first and NumPy second.
- Seed `1031` is reserved for the later continuous-selector construction and
  remains sealed throughout Stage 0.

No other seed may be substituted after observing a result. A backend crash may
be rerun only from the exact sealed source when the artifact proves no scored
step completed.

## Calibration ladder

On seed `1013`, build a fresh brain for each intrinsic-current point:

```text
75, 100, 125, 150, 175 pA
```

Each brain runs `1000` uninterrupted steps. There is no warmup and scoring
starts in the first `100 ms` bin. Archive all five points on both backends.

A point passes a backend only when all conditions hold:

1. population firing is between `40` and `80 Hz` in every consecutive `100 ms`
   bin, including the first;
2. all `40` neurons fire at least once;
3. the maximum fraction firing in one step is at most `0.25`;
4. first-spike times occupy at least `8` distinct steps and span at least
   `8 ms`;
5. external current remains exactly zero and the intrinsic vector remains
   byte-identical;
6. there are no NaNs, infinities, state clears, weight changes, or host-observed
   phase transitions; and
7. the backend reports the requested device honestly.

Select the lowest current that passes every criterion on both backends. The
audit-only external-current curve predicts `100 pA`, but that value is not
privileged: the rule, rather than the prediction, chooses the point. If no
point passes both backends, Stage 0 is `CALIBRATION_NO_GO`. Do not interpolate,
add a point, relax a bound, or change heterogeneity.

## Replication, lesion, and inhibitory response

Run only the selected point on seeds `1019` and `1021`. It must pass the same
ten `100 ms` bins and asynchrony criteria on both backends.

For each seed and backend, construct a matched intrinsic-drive lesion with the
same cell parameters and topology but `intrinsic_current_pA = 0`. It must emit
exactly zero GPi/SNr spikes over `1000 ms`. The intact and lesion external
current vectors must both remain exactly zero.

Then construct matched source-on and source-off inhibitory-response brains:

- `inhibitory_source`: `20` deterministic
  `IZH2007_FS_CORTICAL_INTERNEURON` cells, no intrinsic current;
- `gpi_snr`: the locked `40`-cell population at the selected intrinsic current;
- one `inhibitory_source -> gpi_snr` GABA-A pathway, density `1.0`, weight `8.0`,
  jitter `0`, plasticity off; and
- all other settings unchanged.

Run `500 ms` baseline, `200 ms` inhibition, and `500 ms` release. GPi/SNr
external current stays zero throughout. In the source-on arm only, the host
drives the inhibitory source at `1000 pA` during the fixed inhibition phase.
The source-off twin follows the same schedule with source current zero.

The inhibitory response passes only if:

1. all five baseline `100 ms` GPi/SNr bins are `40-80 Hz`;
2. the source fires during inhibition and is silent outside it;
3. source-on GPi/SNr firing across the `200 ms` inhibition phase is at most
   `10%` of its preceding `200 ms` baseline rate and strictly below source-off;
4. source-on GPi/SNr firing returns to `40-80 Hz` by the second `100 ms` release
   bin and remains there;
5. no release bin exceeds `1.5x` the arm's mean baseline rate;
6. target GABA-A conductance is nonzero only after source activity; and
7. complete initial weights and the intrinsic vector remain byte-identical.

The source current is an experimental stimulus to an upstream inhibitory cell,
not a substitute for GPi/SNr pacemaking or behavioral control.

## Checkpoint continuation gate

At the selected current, run a fresh seed-`1019` intact brain for `300` steps,
save a checkpoint, and continue for `500` steps while hashing the complete spike
raster and final membrane, recovery, conductance, intrinsic-current, and weight
arrays. Load the checkpoint into a fresh bridge and run the same `500` steps.

For each backend, uninterrupted and resumed hashes must match exactly. The
loaded intrinsic vector must equal the configured vector, while an old
default-off checkpoint without that dataset must load with the attribute
`None`. A checkpoint mismatch is `CHECKPOINT_NO_GO`; do not exempt this feature
as static configuration.

## Default-off compatibility gate

Before implementation, source `8994b5102` produced the following Gate A v2
fingerprints at seed `271828`, `300` fixed steps, NMDA off, inherited GPi and
thalamus currents, and `250 pA` shared practice drive:

| backend | raster | final `v` | final `u` | final `g_e` | final `g_i` |
|---|---|---|---|---|---|
| NumPy | `4bfec7fa4c4865db6e31dced73d3d1385820682cf062439a547190853ef3c79d` | `2e848ed2673a192408f118d66fcded3cf9a21719ae909ef4190eab9fc76ff54b` | `cd1b254e7482a26b6a3054777edd411d7a7deff30d3bb1207ae2e474da7b7313` | `33e31475a067f8ac34cc85462b2db8386191d2101eda91421f06d61c80c29b3a` | `65b329f4d6992523da618e1ce43aca126ca0ceb1d339bfa8af6c07e21dc81890` |
| CuPy | `690867e2c44ac456ee1f3a0cb8db9addeef8448753170b587561767c6e51ec2b` | `d1706d17f1a1136a57672546fb643e10f991476c32098ae1906b3b3ec88683df` | `f319dbcfcb1d09f983ad86ddf912484820d3fce94a2b93e135d73b0219c96317` | `11fc8612831e72007d6540dc997d2159d833448d1a3d73b5a29b90f267ba29bc` | `90b3c1c3825eba353c19bbf18017254a498007bdb8cf2cbab4e59acacd61f305` |

Both backends share weight hash
`a9021bcda62b216e67ff1c14c46b011b8590056352514e8672234613b6704b82`
and external-current hash
`4b36942def742bbd214715a7d3e387fb111051ff213935eff1b346e346c2c551`.

After implementation, a default-off bridge must reproduce all corresponding
hashes exactly and report `cp_intrinsic_current_pA is None`. Existing focused
selector V10-V12 tests and checkpoint tests must also pass. A mismatch blocks
scientific execution until explained and separately preregistered.

## Performance gate

On the RTX 3090, benchmark three repetitions of `20000` read-only steps after
`500` unscored steps for:

1. source `8994b5102` default-off baseline;
2. new source, default off;
3. new source, active `100 pA` on `40` of `600` otherwise unchanged neurons;
4. new source with step megakernel v1 active; and
5. new source with step megakernel v2 active.

Record every wall time, median step time, exact array bytes, CuPy memory-pool
bytes, kernel mode, GPU name, and source. The new default-off median may be at
most `1.02x` the old baseline. The active normal path and each active megakernel
may be at most `1.10x` their matching new-source default-off path. Exact feature
storage may not exceed `4 * n_neurons` bytes, and no active path may perform a
per-step device-to-host synchronization.

Performance failure does not invalidate physiology, but it returns
`PHYSIOLOGY_GO_PERFORMANCE_NO_GO` and blocks selector integration until the
implementation is fused or otherwise corrected.

## Verdict and stop rules

Stage 0 is `TONIC_OUTPUT_GO` only if calibration, both fresh replications, the
intrinsic lesion, inhibitory response, checkpoint continuation, default-off
compatibility, topology/provenance audits, and performance gates all pass on
both backends.

Any failure archives complete telemetry and returns a named no-go. Do not alter
the ladder, reuse a failed verdict seed, hide the first bin, initialize neurons
after scoring begins, add OU noise, drive GPi/SNr through the external array,
or proceed to the continuous selector. A no-go returns to research on explicit
NaP/NALCN-like and SK-like intrinsic conductances or a corrected Izhikevich
pacemaker formulation.

Only `TONIC_OUTPUT_GO` opens the separate V13 center-surround selector
preregistration. It does not claim explicit ion-channel biology, action
selection, reward learning, or conversation.

## Evidence read before filing

- `2026-08-04-neural-vocal-credit-gateB-v13-continuous-bg-selector-RESEARCH-GATE.md`
- `2026-08-04-neural-vocal-credit-gateB-v12-disinhibitory-boundary-CONSTRUCTION-NO-GO.md`
- `2026-08-03-neural-vocal-selector-gateA-v2-4seed-GO.md`
- Local catalog A.03, A.04, A.10, A.13, and A.14.
- PBR-160 chapters 8 and 9 (Nambu; Deniau et al.).
- Nakanishi, Kita, and Kitai (1987), [PubMed](https://pubmed.ncbi.nlm.nih.gov/3427482/).
- Nambu, Tokuno, and Takada (2002), [PubMed](https://pubmed.ncbi.nlm.nih.gov/12067746/).
- Atherton and Bevan (2005), [Journal of Neuroscience/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6725542/).
- Lutas et al. (2016), [eLife/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4902561/).
