---
type: research-finding
status: complete
date: 2026-08-04
mechanism: v13-replacement-calibration
---

# V13 replacement calibration: biological and experimental basis

## Decision

The strict matched-state replay removes the known NumPy/CuPy execution
confound for the locked V13 trajectory. It does not validate the earlier
calibration, select a current, or justify changing the scientific contract.
Replacement calibration should therefore rerun the already frozen five-point
intrinsic-current ladder with canonical byte-identical populations and strict
arithmetic, in the controller-enforced order. It should select the lowest point
that satisfies the preregistered physiological and causal criteria on both
backends.

The current is a reduced-model control variable, not a measured ion-channel
current. No local primary source supplies a preparation-matched conversion from
NaP, NALCN-like, Ih, Cav2.2, and SK conductances to V13
`intrinsic_current_pA`. Calling any particular current "biological" would
overstate the evidence. The biologically defensible targets are the resulting
firing, regularity, inhibitory pause, phase reset, recovery, and lesion
responses.

## What prior work rules out

The following regimes must remain visible and must not be rediscovered by
another unconstrained tuning cycle:

| Regime | Observation | Interpretation for replacement calibration |
|---|---|---|
| Trial-scoped selector with host stop/reset and host GPi drive | Gate A could select one channel only because the host ended the action at the first crossing, applied a reset pulse and washout, and continuously drove GPi. | Rejected as a continuous biological selector. Calibration may not hide startup, stop on a winner, clear state, or inject runtime GPi current. |
| V11 recurrent action boundary | The boundary activated without motor output. | Rejected decomposition; do not calibrate a downstream boundary around a selector that is not continuous. |
| V12 guarded feed-forward boundary | Local inhibitory signs were causal, but both motor channels crossed during startup and both action windows on CuPy. | Rejected as a construction regime; do not retune these boundary weights as a V13 substitute. |
| Zero intrinsic drive | The named GPi/GPe/STN pacemaker populations were silent for an uninterrupted zero-input audit. | Rejected for autonomous output. The intact population must fire from the first scored window, while the matched intrinsic-drive lesion should be silent. |
| `75 pA` on the original ladder | Diagnostic population rates were about `25.9-26.5 Hz` on the two backends. Every cell fired and timing was dispersed, but every rate bin was below the locked range. | Too weak for the selected 40-80 Hz GPi/SNr phenotype. It remains a required lower bracket, not a candidate to promote from old evidence. |
| `125 pA` | Both diagnostic population means were `81.6 Hz`; several bins exceeded `80 Hz`, and one backend also exceeded the same-step fraction limit. | Marginally over-driven and too close to synchrony for the locked phenotype. It remains an upper bracket. |
| `150-175 pA` | Diagnostic means were approximately `94.9-106.6 Hz`, with increasing first-spike concentration and same-step firing failures. | Rejected over-drive regime for this reduced population. Do not relax the rate or synchrony gates to admit it. |
| Old `100 pA` observation | It was the only common passing ladder point, but the calibration order was violated. The dependent replication was also procedurally undefined. | Diagnostic only. It must not be carried forward as the selected value or used to narrow the replacement scan. |
| Backend-native heterogeneous populations | The same integer produced different `C`, `a`, `b`, `d`, initial states, and cell identities. Suppression followed population origin: about `1.8-2.0%` of baseline for one origin versus `17.9-18.2%` for the other. | Rejected paired-comparison method. Each replacement population must be generated once, sealed, and replayed byte-identically on both backends. |
| Retuning the fixed inhibitory pathway after the diagnostic miss | One old backend-native population missed the `<=10%` suppression criterion while the other passed. The mismatch was traced primarily to initialization, then strict replay closed the remaining arithmetic divergence. | No post-hoc weight, density, source-drive, duration, or threshold change is justified. First rerun the frozen contract with the confounds removed. |

The old compatibility failure and the first strict-replay failure were
engineering failures, not evidence against the tonic-output mechanism. The
deterministic compatibility correction, canonical state transplant, and strict
replay v2 now isolate the replacement experiment sufficiently to make a
cross-backend calibration interpretable.

## Candidate calibration ranges

### Replacement Stage 0: frozen reduced-model range

The only admissible range for the pending replacement run is the preregistered
ladder:

```text
75, 100, 125, 150, 175 pA
```

This is an engineering bracket around the existing Izhikevich GPi/SNr preset.
It is defensible for the replacement experiment because it spans a measured
under-driven regime, the target operating region, and clearly over-driven
regimes without adding points after observing results. The replacement must
retain the frozen heterogeneous preset distributions (`C` centered at `60`,
`a` at `0.05`, `b` at `2`, and `d` at `25`, with their existing spreads).
Those distributions are model priors, not measured biological distributions.

The locked observable range remains `40-80 Hz` in every consecutive `100 ms`
bin from time zero. This matches the local catalog's SNr/GPi autonomous-output
range and the Deniau et al. synthesis in *Progress in Brain Research* volume
160, chapter 9. The population must also remain asynchronous: every cell fires,
no more than one quarter fires on one step, and first spikes span at least eight
distinct steps and eight milliseconds. These timing limits are conservative
engineering guards against an artificial synchronized volley; the catalog does
not establish them as species-independent biological constants.

The existing inhibitory challenge should remain unchanged for this replacement
because the correction protocol freezes it. Suppression to at most `10%` of
the preceding baseline, recovery to `40-80 Hz` by the second `100 ms` release
bin, and rebound no greater than `1.5x` baseline are preregistered experiment
bounds. The local biology supports a strong GABA-A-mediated delay/pause,
synchronized GPe-driven phase reset, and ordinary recovery, but it does not
establish those three numerical cutoffs as universal GPi/SNr values. They test
whether the reduced substrate is controllable enough for the later selector;
they should not be presented as direct biological measurements.

### Later explicit-conductance replacement

If the constant-current phenotype passes but proves fragile during the
continuous selector, the next biologically stronger calibration should vary
mechanisms rather than search a wider constant-current ladder:

- NaP/slowly inactivating TTX-sensitive sodium drive: intact autonomous firing;
  a channel lesion should strongly reduce or abolish pacemaking.
- NALCN-like tonic cation drive: use a partial-loss target rather than an
  all-or-none target. Lutas et al. (2016) measured a reduction from `21.0` to
  `11.9 spikes/s`, or about `57%` of control, while firing persisted.
- Cav2.2-coupled SK recovery: preserve afterhyperpolarization and regularity;
  SK block should increase firing irregularity more than it destroys tonic
  firing. Atherton and Bevan (2005) supports the mechanism, but the local
  record does not provide a preparation-matched numerical CV threshold.
- Ih: treat as a secondary contributor and recovery observable, not the sole
  pacemaker, unless a preparation-matched primary result supports that role.

Those channel-level ranges require a separate preregistration and an SNr/GPi
conductance family. They must not be folded into the pending reduced-model
replacement calibration.

## Required observables

Replacement calibration should preserve every existing locked measure and
report enough detail to distinguish a rate match from a functioning output
population:

1. Per-`100 ms` population rate from the first step, full-run rate, per-cell
   rates, silent-cell count, and total spikes.
2. Per-cell interspike intervals, ISI coefficient of variation and local CV2,
   first-spike distribution, maximum same-step fraction, and pairwise spike
   coincidence. CV/CV2 are diagnostic until a numerical range is separately
   preregistered.
3. Exact external-current and intrinsic-vector invariance; finite `v`, `u`,
   and conductance state; immutable weights and topology.
4. Matched intrinsic-drive lesion response. Silence establishes causal
   dependence on the reduced pacemaker term, not ion-channel closure.
5. Source-on versus source-off inhibition: source raster, target GABA-A
   conductance, pause latency, rate ratio, fraction of silenced cells, last
   pre-pause and first post-pause spike, and phase dispersion before and after
   inhibition.
6. Release latency, each release-bin rate, overshoot, delayed rebound, and
   return of ISI/CV2 toward baseline without state clearing.
7. Byte hashes for the canonical population, initial state, topology,
   stimulus schedule, and restored device arrays before step zero.
8. Exact NumPy/CuPy `v`, `u`, and spike trajectories under the strict path,
   plus the existing checkpoint round-trip and default-off compatibility
   controls.

## Falsification criteria

The replacement hypothesis is falsified for Stage 0 if any of the following
occurs under the frozen controller-governed protocol:

- no ladder point satisfies every locked intact criterion on both backends;
- the selected point differs between backends for one canonical population;
- firing starts only after a hidden settling period, a subset remains silent,
  or the population forms synchronized volleys;
- the zero-drive lesion continues tonic firing, or intact firing requires
  runtime external GPi current;
- matched GABA-A input does not produce the locked pause, source-off changes
  similarly, recovery misses its window, or rebound exceeds its bound;
- any non-intervention state, topology, weight, or stimulus hash differs
  between matched arms;
- strict arithmetic loses trajectory identity, checkpoint continuation fails,
  or any registered stage violates the required order; or
- a result depends on changing the ladder, heterogeneity, inhibitory pathway,
  or thresholds after inspecting replacement data.

A Stage-0 pass would falsify only the claim that the reduced substrate cannot
provide autonomous, suppressible, recoverable tonic output. It would not prove
that V13 has a continuous selector, biological ion channels, local reward
credit, or conversational behavior.

## What remains unknown

- There is no measured mapping from V13 picoamps to NaP/NALCN/SK/Ih channel
  densities or to a specific GPi/SNr species, age, temperature, and recording
  preparation.
- The existing `C/a/b/d` spreads are not empirical GPi/SNr parameter
  distributions. It is unknown whether they reproduce biological rate, ISI,
  pause, and rebound covariation across cells.
- The local catalog supports strong GABA timing effects and phase reset, but
  not a universal `<=10%` rate ratio, `200 ms` challenge, second-bin recovery,
  or `1.5x` rebound bound.
- The dense all-to-all inhibitory challenge collapses striatal distal and GPe
  proximal synapses into one compartment. Real origin, location, chloride
  reversal, conductance waveform, and short-term plasticity are not calibrated.
- GPi and SNr are not interchangeable cell families. A shared reduced preset
  may pass this engineering gate while missing nucleus-specific physiology.
- It is unknown whether a Stage-0 operating point remains stable after adding
  hyperdirect STN excitation, direct striatal inhibition, GPe input, recurrent
  feedback, and learning. That requires a separate continuous-selector gate,
  not a broader Stage-0 scan.

## Evidence consulted

Local RAG was checked first and reported `RAG_WORKFLOW_READY`. Targeted searches
of the `finding`, `catalog`, and `paper` corpora located the V13 process and
backend findings, feature-catalog entry A.04, and the local text extraction of
Tepper, Abercrombie, and Bolam (eds.), *GABA and the Basal Ganglia*, *Progress
in Brain Research* 160 (2007), especially chapters 7-9. The source records
support autonomous `40-80 Hz` output, intrinsic sodium/cation drive,
Cav2.2-SK control of regularity, effective GABA-A spike delay and GPe phase
reset, and compartment-specific inhibitory inputs.

Project evidence consulted:

- `2026-08-04-neural-vocal-credit-gateB-v13-continuous-bg-selector-RESEARCH-GATE.md`
- `2026-08-04-neural-vocal-credit-gateB-v13-tonic-output-substrate-PREREGISTRATION.md`
- `2026-08-04-neural-vocal-credit-gateB-v13-calibration-order-UNDEFINED.md`
- `2026-08-04-neural-vocal-credit-gateB-v13-backend-state-transplant-DIAGNOSTIC-RESULT.md`
- `2026-08-04-v13-backend-neutral-heterogeneity-RESEARCH.md`
- strict-arithmetic replay v1 and v2 records and their raw comparison evidence.

Primary evidence already present in the local record:

- Atherton and Bevan (2005), autonomous SNr firing and Cav2.2-SK control,
  <https://doi.org/10.1523/JNEUROSCI.1475-05.2005>. <!--derived-->
- Lutas et al. (2016), NALCN contribution to persistent SNr activity,
  <https://doi.org/10.7554/eLife.15271>. <!--derived-->
- Connelly et al. (2010), distinct striatonigral and pallidonigral short-term
  plasticity, <https://doi.org/10.1523/JNEUROSCI.3895-10.2010>. <!--derived-->
- Simmons et al. (2020), pallidonigral conductance barrages and baseline-rate
  dependence, <https://doi.org/10.1152/jn.00678.2019>. <!--derived-->
- Nakanishi, Kita, and Kitai (1987), autonomous SNr membrane properties,
  <https://pubmed.ncbi.nlm.nih.gov/3427482/>.

No online source was needed: the local catalog answered the pending
replacement-calibration question. The missing quantitative mappings are
recorded above as unknown rather than filled with adjacent-preparation values.
