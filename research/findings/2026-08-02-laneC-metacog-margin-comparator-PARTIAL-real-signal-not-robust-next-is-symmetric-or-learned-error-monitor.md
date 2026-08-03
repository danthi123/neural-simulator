---
type: finding
status: contributing
date: 2026-08-02
mechanism: second-order-metacognition-dynamic-acc-apfc-conflict-monitor
artifacts:
  - research/findings/raw/lanes/metacog/metacog_legacy_refactor_smoke_s42.json
  - research/findings/raw/lanes/metacog/metacog_margin_smoke_s42.json
  - research/findings/raw/lanes/metacog/metacog_margin_sweep_e2_i3p5_s42.json
  - research/findings/raw/lanes/metacog/metacog_margin_6seed_e2_i3p5.json
  - research/findings/raw/lanes/metacog/metacog_margin_abs_smoke_e2_i3p5_s42.json
  - research/findings/raw/lanes/metacog/metacog_margin_abs_6seed_e2_i3p5.json
  - research/findings/raw/lanes/metacog/metacog_learned_acc_smoke_s42.json
  - research/findings/raw/lanes/metacog/metacog_learned_acc_6seed.json
  - research/findings/raw/lanes/metacog/metacog_learned_acc_balance_s102.json
  - research/findings/raw/lanes/metacog/metacog_learned_acc_symmetric_s102.json
  - research/findings/raw/lanes/metacog/metacog_learned_acc_response_homeostasis_s102.json
  - research/findings/raw/lanes/metacog/metacog_learned_acc_dynamic_s102.json
  - research/findings/raw/lanes/metacog/metacog_learned_acc_dynamic_6seed.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_smoke_s42.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_6seed_smoke.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_stress_s100_101_102.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_s102_report80.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_s101_noise60_report80.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_s101_sig80_320_report80.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_s101_report80_snapshotfix.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_s101_report80_resp1plus200.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_stress_s100_101_102_report80_resp1plus200.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_6seed_report80_resp1plus200_s42.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_6seed_report80_resp1plus200_s43.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_6seed_report80_resp1plus200_s44.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_6seed_report80_resp1plus200_s100.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_6seed_report80_resp1plus200_s101.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_6seed_report80_resp1plus200_s102.json
  - research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_6seed_report80_resp1plus200_fanout_aggregate.json
---

# lane C metacognition: dynamic ACC/aPFC conflict monitor clears the 6-seed type-2 gate

<!--derived-->
**One-line verdict.** Fixed margin readouts and the first static learned ACC/aPFC monitor were real but seed-fragile.
The opt-in dynamic ACC/aPFC conflict formulation now clears the six-seed type-2 gate: `--confidence-read learned_acc
--learned-feature-mode dynamic` passed **6/6 seeds**, with mean type2 AUC 0.831, mean meta-d 2.431, and all
meta-lesion, domain-dissociation, permuted-confidence, within-class, and type-1 operating-window controls passing.
This is a functional metacognition-correlate GO for the isolated Lane C monitor. A runner-level self-schema relay now
shows the signal can be read by a spiking `self_schema` confidence pool, and the post-handoff response-balanced
operating point (`--learned-report-steps 80 --response1-tonic-pa 200`) promotes that relay to **GO 6/6**. It is still
not production-wired and not a claim of subjective experience; the remaining blocker is production integration, not
relay robustness.

## What changed

<!--derived-->
The runner now has two confidence sources:

- `meta_rate` (default): legacy slow-NMDA magnitude read, preserved for reproduction.
- `margin`: opt-in comparator read. Class k excites meta subpool k and excites inhibitory relay k; relay k suppresses
  the opposite meta subpool. Confidence is the winning meta subpool's spiking rate.

The legacy smoke reproduced the negative (`type2_auc=0.502`, `meta_d=0.010` on seed 42). The comparator smoke at the
default weights lifted the signal but missed (`type2_auc=0.632`, `meta_d=0.821`, controls mostly clean). A small
gain sweep found the only useful scout point at `meta_exc_w=2.0`, `meta_inh_w=3.5`: seed 42 smoke **GO**
(`type2_auc=0.664`, `meta_d=1.041`, `m_ratio=0.62`, meta-lesion/permutation/within-class all pass). Higher gains
suppressed or inverted the read, so the operating point is narrow.

## Frozen 6-seed validation

<!--derived-->
Command:

```bash
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._second_order_metacog_monitor_derisk \
  --seeds 42 43 44 100 101 102 --n-trials 160 --backend numpy \
  --confidence-read margin --meta-exc-w 2.0 --meta-inh-w 3.5 \
  --json research/findings/raw/lanes/metacog/metacog_margin_6seed_e2_i3p5.json
```

Aggregate:

| metric | result |
|---|---:|
| verdict | PARTIAL |
| seed GO count | 2/6 |
| mean type1 accuracy | 0.685 |
| mean d1 | +1.071 |
| mean type2 AUC | 0.635 |
| mean meta-d | 0.895 |
| mean M-ratio | 0.826 |
| all type1 in window | true |
| all meta-lesion collapse | true |
| all domain dissociation | true |
| all permuted collapse | false |
| all within-class OK | false |

Per seed:

| seed | GO | type1 acc | type2 AUC | meta-d | M-ratio | within-class min | permuted AUC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | no | 0.688 | 0.627 | 0.769 | 0.795 | 0.567 | 0.539 |
| 43 | no | 0.713 | 0.630 | 0.818 | 0.726 | 0.525 | 0.513 |
| 44 | no | 0.744 | 0.610 | 0.670 | 0.519 | 0.569 | 0.572 |
| 100 | yes | 0.656 | 0.650 | 0.934 | 1.174 | 0.596 | 0.502 |
| 101 | yes | 0.631 | 0.751 | 1.950 | 1.500 | 0.700 | 0.529 |
| 102 | no | 0.681 | 0.539 | 0.228 | 0.242 | 0.603 | 0.538 |

## Interpretation

<!--derived-->
The margin comparator refutes the strongest form of the previous "monitor is dead" reading: metacognitive signal is
present, meta-lesion collapses it without changing first-order d1, and several seeds show high meta-d. But it does
not close the faculty. The fixed comparator is class-asymmetric and fragile: some seeds miss the global type2 AUC bar,
some miss within-class correctness, and seed 44 leaks under the permuted-confidence control. The result is a verdict
on this fixed, hand-balanced comparator, not on metacognition.

## Absolute-margin follow-up

<!--derived-->
I also tested the simplest symmetric readout transform: use the same opponent comparator circuit but score confidence
as the absolute difference between the two meta subpools (`--confidence-read margin_abs`). Seed 42 smoke passed
(`type2_auc=0.698`, `meta_d=1.257`, `m_ratio=0.75`, controls clean), but the frozen 6-seed promotion was
**NEGATIVE, 0/6**:

| metric | result |
|---|---:|
| verdict | NEGATIVE |
| seed GO count | 0/6 |
| mean type1 accuracy | 0.685 |
| mean d1 | +1.071 |
| mean type2 AUC | 0.598 |
| mean meta-d | 0.631 |
| mean M-ratio | 0.602 |
| all type1 in window | true |
| all meta-lesion collapse | true |
| all domain dissociation | true |
| all permuted collapse | true |
| all within-class OK | false |

| seed | GO | type1 acc | type2 AUC | meta-d | M-ratio | within-class min | permuted AUC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | no | 0.688 | 0.637 | 0.833 | 0.861 | 0.640 | 0.503 |
| 43 | no | 0.713 | 0.545 | 0.273 | 0.243 | 0.476 | 0.487 |
| 44 | no | 0.744 | 0.612 | 0.686 | 0.531 | 0.518 | 0.493 |
| 100 | no | 0.656 | 0.643 | 0.881 | 1.108 | 0.608 | 0.518 |
| 101 | no | 0.631 | 0.648 | 1.076 | 0.828 | 0.605 | 0.526 |
| 102 | no | 0.681 | 0.506 | 0.038 | 0.041 | 0.537 | 0.485 |

This is informative: taking `abs(meta_1 - meta_0)` removes the obvious class-direction permutation leak, but it also
throws away enough calibrated confidence information that every seed misses the GO bar. The next mechanism should not
be another static readout transform of the same fixed comparator.

## Next mechanism

<!--derived-->
Do **not** keep broad-cranking `meta_exc_w/meta_inh_w`: the useful region is narrow and the higher-gain sweep already
suppressed/inverted the signal; and do not simply take absolute differences, because that variant went 0/6. The next
biology-grounded step is a monitor that is either:

1. **homeostatically calibrated**: balanced opponent subpools plus divisive normalization/homeostatic scaling so each
   class has the same confidence transfer curve before type-2 scoring, with the calibration itself measured rather
   than assumed; or
2. **learned error/conflict monitor**: an ACC/aPFC-style read trained from outcome/error feedback to predict
   correct-vs-error from the first-order competition, preserving the meta-lesion and permuted-confidence controls.

The structural-map integration path should now advance in order: first the self-schema confidence relay, then
self-model-driven abstain/hedge behavior alongside the host moat. The WKV run4 RF spiking-forward path is a separate
GPU lane.

## Learned ACC/aPFC follow-up

<!--derived-->
The learned monitor branch was implemented as `--confidence-read learned_acc`: a bounded runner-local calibration block
learns ACC/aPFC-style error/conflict weights over the workspace's own spike-rate features, then reports confidence by
driving `meta_schema` as a spiking aPFC/meta rate. Seed 42 smoke passed (`type2_auc=0.796`, `meta_d=1.977`,
`m_ratio=0.939`; meta-lesion AUC 0.500, permuted AUC 0.512).

Frozen six-seed command:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._second_order_metacog_monitor_derisk \
  --seeds 42 43 44 100 101 102 --n-trials 160 \
  --confidence-read learned_acc --backend numpy \
  --json research/findings/raw/lanes/metacog/metacog_learned_acc_6seed.json
```

Aggregate:

| metric | result |
|---|---:|
| verdict | PARTIAL |
| seed GO count | 2/6 |
| mean type1 accuracy | 0.718 |
| mean d1 | +1.248 |
| mean type2 AUC | 0.683 |
| mean meta-d | 1.283 |
| mean M-ratio | 0.996 |
| all type1 in window | true |
| all meta-lesion collapse | true |
| all domain dissociation | true |
| all permuted collapse | false |
| all within-class OK | false |

| seed | GO | type1 acc | type2 AUC | meta-d | M-ratio | within-class min | permuted AUC | primary miss |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 42 | yes | 0.781 | 0.723 | 1.413 | 0.923 | 0.687 | 0.466 | none |
| 43 | no | 0.750 | 0.734 | 1.658 | 1.111 | 0.691 | 0.613 | permutation |
| 44 | no | 0.719 | 0.619 | 0.784 | 0.649 | 0.556 | 0.490 | type2 |
| 100 | yes | 0.700 | 0.780 | 1.881 | 1.786 | 0.568 | 0.500 | none |
| 101 | no | 0.681 | 0.770 | 1.962 | 1.507 | 0.277 | 0.501 | within-class |
| 102 | no | 0.675 | 0.469 | 0.000 | 0.000 | 0.288 | 0.519 | type2/meta/within-class |

Interpretation: learned ACC/aPFC is a stronger direction than fixed readout transforms because the mean meta-d and
M-ratio are high and every seed preserves meta-lesion/domain dissociation. It is still not robust: one seed leaks
under permutation, two seeds invert or lose within-class correctness, and seed 102 loses the main type-2 signal. Next
should target class-balanced calibration or a neural/homeostatic equalizer before any self-report integration.

## Calibration scouts

<!--derived-->
Two seed-102 scouts tested the cheapest calibration fixes and were not promoted. Class-balanced calibration
(`--learned-balance-classes`) remained **NEGATIVE**: type2_auc 0.467, meta_d 0.000, within-class min 0.264, although
lesion/domain/permutation controls still collapsed as expected. Response-symmetric feature masking
(`--learned-symmetric-features`, masking `signed_margin` and `response_sign`) was also **NEGATIVE**: type2_auc 0.415,
meta_d 0.000, within-class min 0.377, permuted AUC 0.504. These scouts rule out simple reweighting or removal of the
obvious signed shortcuts as the seed-102 rescue. The next Lane C mechanism should be a real neural/homeostatic
equalizer or a different monitor formulation, not a six-seed promotion of either scout.

## Response-homeostasis scout

<!--derived-->
The first post-static rescue added `--learned-response-homeostasis`, a response-channel feature centering/scaling
step before the ACC calibration. This is a homeostatic equalizer over the chosen-response subpools, intended to remove
fixed response-channel rate bias while leaving the learned ACC/aPFC rule otherwise unchanged.

Command:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._second_order_metacog_monitor_derisk \
  --seed 102 --n-trials 160 --confidence-read learned_acc --learned-response-homeostasis \
  --backend numpy \
  --json research/findings/raw/lanes/metacog/metacog_learned_acc_response_homeostasis_s102.json
```

It was **NEGATIVE** on the hardest seed: type2_auc 0.451, meta_d 0.000, within-class min 0.424. Lesion and
permutation controls still collapsed, but there was no intact metacognitive signal to attribute. Verdict: simple
response-channel gain equalization is not enough; the monitor needs a different state variable, not just a calibrated
static feature vector.

## Dynamic ACC/aPFC conflict monitor

<!--derived-->
The successful formulation adds `--learned-feature-mode dynamic` to the learned ACC/aPFC branch. The calibration still
uses delayed correctness/error feedback and the report is still a spiking `meta_schema`/aPFC rate, but the learned
features now include late-window workspace conflict and response-persistence terms from the same first-order trace:
late winner/runner rates, late margin/balance/conflict, chosen-vs-unchosen late rates, chosen-rate drop, margin drop,
response persistence, and late shared-inhibition rate. This targets the ACC-like temporal conflict signal that the
static averaged winner/runner read missed.

Seed-102 rescue command:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._second_order_metacog_monitor_derisk \
  --seed 102 --n-trials 160 --confidence-read learned_acc --learned-feature-mode dynamic \
  --backend numpy \
  --json research/findings/raw/lanes/metacog/metacog_learned_acc_dynamic_s102.json
```

Seed 102 flipped from static learned-ACC failure to **GO**: type1 accuracy 0.675, type2_auc 0.783, meta_d 1.939,
M-ratio 2.144, within-class min 0.697, meta-lesion collapsed, and permuted confidence collapsed.

Frozen six-seed command:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._second_order_metacog_monitor_derisk \
  --seeds 42 43 44 100 101 102 --n-trials 160 \
  --confidence-read learned_acc --learned-feature-mode dynamic --backend numpy \
  --json research/findings/raw/lanes/metacog/metacog_learned_acc_dynamic_6seed.json
```

Aggregate:

| metric | result |
|---|---:|
| verdict | GO |
| seed GO count | 6/6 |
| mean type1 accuracy | 0.718 |
| mean d1 | +1.248 |
| mean type2 AUC | 0.831 |
| mean meta-d | 2.431 |
| mean M-ratio | 1.987 |
| all type1 in window | true |
| all meta-lesion collapse | true |
| all domain dissociation | true |
| all permuted collapse | true |
| all within-class OK | true |

Per seed:

| seed | GO | type1 acc | type2 AUC | meta-d | M-ratio | within-class min | permuted AUC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | yes | 0.781 | 0.775 | 1.796 | 1.174 | 0.744 | 0.505 |
| 43 | yes | 0.750 | 0.883 | 3.102 | 2.079 | 0.880 | 0.537 |
| 44 | yes | 0.719 | 0.851 | 2.623 | 2.172 | 0.804 | 0.549 |
| 100 | yes | 0.700 | 0.833 | 2.301 | 2.184 | 0.782 | 0.522 |
| 101 | yes | 0.681 | 0.858 | 2.823 | 2.168 | 0.748 | 0.531 |
| 102 | yes | 0.675 | 0.783 | 1.939 | 2.144 | 0.697 | 0.408 |

Interpretation: the prior static learned monitor was reading a brittle end-state magnitude. Adding late conflict and
persistence exposes a robust second-order signal: the monitor predicts correctness within each stimulus class,
collapses under meta-lesion without changing first-order d1, and collapses when confidence is decorrelated from the
actual trial. The isolated monitor is no longer the Lane C blocker; the blocker moved to integration into
self-schema/production.

## Self-schema integration de-risk

<!--derived-->
I added a separate runner-level integration probe,
`research.runners._laneC_self_schema_metacog_integration_derisk`. It reuses the dynamic learned ACC/aPFC monitor,
builds a shared bridge with `workspace`, `meta_schema`, and `self_schema`, and installs a fixed on-substrate
`meta_schema -> self_schema` confidence projection. The confidence score is then read from the spiking
`self_schema` pool, not directly from the learned host-side logistic confidence. Controls lesion the meta assembly,
lesion the meta-to-self readout, and permute confidence against trial correctness.

Seed-42 smoke command:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneC_self_schema_metacog_integration_derisk \
  --smoke --seed 42 --backend numpy \
  --json research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_smoke_s42.json
```

The smoke passed: type1 accuracy 0.719, self-schema type2 AUC 0.837, self meta-d 2.401,
M-ratio 2.175, self-vs-meta Spearman +0.787, and all meta-lesion, self-read-lesion, and permutation controls
collapsed.

Six-seed smoke-scale command:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneC_self_schema_metacog_integration_derisk \
  --seeds 42 43 44 100 101 102 --n-trials 64 --learned-calib-trials 64 \
  --backend numpy \
  --json research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_6seed_smoke.json
```

This was **PARTIAL, 3/6**. Mean self-schema type2 AUC was 0.808 and mean self meta-d was 2.253, with meta-lesion,
self-read-lesion, and domain controls clean. Failures were specific: seed100 leaked under permutation, seed101 missed
the first-order operating window, and seed102 failed the provisional self-vs-meta tracking bar.

Targeted full-budget stress command:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneC_self_schema_metacog_integration_derisk \
  --seeds 100 101 102 --n-trials 160 --backend numpy \
  --json research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_stress_s100_101_102.json
```

Stress result: **PARTIAL, 1/3**. Seed100 became GO at full budget, which explains its smoke permutation failure as
small-sample noise. Seeds101 and 102 still failed:

| seed | GO | type1 acc | self AUC | self meta-d | M-ratio | self-vs-meta | primary miss |
|---:|---|---:|---:|---:|---:|---:|---|
| 100 | yes | 0.650 | 0.786 | 1.913 | 2.514 | +0.899 | none |
| 101 | no | 0.556 | 0.797 | 2.490 | 2.793 | +0.743 | type1 window + tracking |
| 102 | no | 0.756 | 0.783 | 1.910 | 1.385 | +0.691 | tracking |

Aggregate stress controls were clean: all meta-lesion collapses true, all self-read-lesion collapses true, all
permuted-confidence collapses true, and all domain-dissociation controls true. At this stage, the failure was not
"no self-schema signal"; it was robustness of the relay/operating window. The next step was to tune the response
operating point while preserving the lesion and permutation collapses; the later response-balanced promotion section
below supersedes this stage-local partial verdict.

## Relay scouts before handoff

<!--derived-->
Two narrow follow-ups tested whether the remaining integration failures were relay sampling or first-order operating
point artifacts.

First, seed102 was rerun with a longer report window (`--learned-report-steps 80`) and default task difficulty:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneC_self_schema_metacog_integration_derisk \
  --seed 102 --n-trials 160 --learned-report-steps 80 --backend numpy \
  --json research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_s102_report80.json
```

This flipped seed102 to **GO**: type1 accuracy 0.744, self AUC 0.815, self meta-d 2.241, M-ratio 1.684,
self-vs-meta Spearman +0.854, and all meta-lesion, self-read-lesion, permutation, and domain controls passed. This
supports the seed102 stress failure as relay spike-count / settling noise, not a missing metacognitive source.

Seed101 then received the same longer report window plus two small first-order operating-point scouts:

| artifact | change | verdict | type1 acc | self AUC | self meta-d | self-vs-meta | controls |
|---|---|---:|---:|---:|---:|---:|---|
| `metacog_self_schema_dynamic_integration_s101_noise60_report80.json` | `--stim-noise 60` | NEGATIVE | 0.556 | 0.872 | 3.326 | +0.968 | all collapsed |
| `metacog_self_schema_dynamic_integration_s101_sig80_320_report80.json` | `--sig-lo 80 --sig-hi 320` | NEGATIVE | 0.550 | 0.947 | 4.718 | +0.956 | all collapsed |

Both seed101 scouts failed only the first-order operating window (`type1_accuracy < 0.60`). The self-schema signal,
tracking, meta-lesion, self-read lesion, permutation, and domain controls were all strong. The next restart should not
chase the relay for seed101; it should diagnose the combined bridge's first-order response bias / operating point
(for example class-channel homeostasis or response-balanced drive) while preserving the successful `report_steps=80`
relay setting.

## Response-balanced relay promotion

<!--derived-->
After relaunch, I checked the bridge state snapshot path first. Preserving the full conductance/refractory/NMDA/GABA
state made the trial reset more faithful, but by itself it did **not** rescue seed101: type1 accuracy was 0.481 while
self-schema AUC 0.886, self meta-d 4.587, self-vs-meta Spearman +0.983, and all controls were clean. That confirmed
the remaining miss was a first-order response-channel operating point, not a dead self-schema confidence relay.

The targeted operating-point fix was a response-channel tonic offset,
`--response1-tonic-pa 200`, evaluated with the longer report window:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneC_self_schema_metacog_integration_derisk \
  --seed 101 --n-trials 160 --learned-report-steps 80 --response1-tonic-pa 200 \
  --backend numpy \
  --json research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_s101_report80_resp1plus200.json
```

Seed101 flipped to **GO**: type1 accuracy 0.600, self AUC 0.797, self meta-d 2.348, self-vs-meta Spearman +0.981,
with meta-lesion, self-read lesion, permutation, and domain controls all passing. The same operating point then
promoted the previously failing stress set:

| seed | GO | type1 acc | self AUC | self meta-d | self-vs-meta |
|---:|---|---:|---:|---:|---:|
| 100 | yes | 0.625 | 0.674 | 1.271 | +0.974 |
| 101 | yes | 0.600 | 0.797 | 2.348 | +0.981 |
| 102 | yes | 0.613 | 0.791 | 2.456 | +0.946 |

Aggregate stress controls were all clean: meta-lesion collapse, self-read-lesion collapse, permuted-confidence
collapse, and domain dissociation.

Finally, I fanned the same frozen operating point across the full six-seed set as independent one-seed CPU jobs and
aggregated the results in
`research/findings/raw/lanes/metacog/metacog_self_schema_dynamic_integration_6seed_report80_resp1plus200_fanout_aggregate.json`:

| metric | result |
|---|---:|
| verdict | GO |
| seed GO count | 6/6 |
| mean type1 accuracy | 0.640 |
| mean d1 | +1.048 |
| mean self-schema type2 AUC | 0.769 |
| mean self meta-d | 2.180 |
| mean self M-ratio | 2.177 |
| mean self-vs-meta Spearman | +0.950 |
| all meta-lesion collapse | true |
| all self-read-lesion collapse | true |
| all permuted-confidence collapse | true |
| all domain dissociation | true |

| seed | GO | type1 acc | self AUC | self meta-d | self-vs-meta |
|---:|---|---:|---:|---:|---:|
| 42 | yes | 0.688 | 0.755 | 1.849 | +0.945 |
| 43 | yes | 0.688 | 0.823 | 3.034 | +0.898 |
| 44 | yes | 0.625 | 0.771 | 2.120 | +0.952 |
| 100 | yes | 0.625 | 0.674 | 1.271 | +0.974 |
| 101 | yes | 0.600 | 0.797 | 2.348 | +0.981 |
| 102 | yes | 0.613 | 0.791 | 2.456 | +0.946 |

Interpretation: STEP 2 is now robust at the runner level. The brain has a spiking `self_schema` confidence pool that
tracks a dynamic ACC/aPFC reliability signal and collapses under the right lesions/controls. This still does **not**
claim subjective experience, and it does **not** mean production honesty is solved. The next build is production
integration: route the self-schema confidence pool into abstain/hedge behavior alongside the existing host moat, then
test a familiar-but-wrong battery where familiarity alone accepts but metacognitive confidence should hedge or abstain.
